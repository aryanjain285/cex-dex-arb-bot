"""Inference that survives the fact that market observations are not independent.

Everything here exists to stop one specific error, which is the easiest way for
this project to reach a confident wrong conclusion.

A price dislocation persists. Sample it every five seconds and consecutive
observations are nearly the same fact recorded twice. A hundred thousand rows may
contain a few thousand independent facts. Since the standard error of a mean is
sigma/sqrt(n), using the row count inflates every t-statistic by sqrt(n/n_eff) --
a factor of five or more for a strongly persistent series. That turns

    "mean dislocation -1.5 bps, cannot rule out zero"

into

    "mean dislocation -1.5 bps, p < 0.001".

The perverse part: sampling FASTER makes the inflation worse. So the error rewards
the instinct to collect more data, and a run that recorded ten times as often would
report ten times the confidence in the same underlying fact.

The correction is batch means. Split the series into contiguous batches long enough
that each batch mean is roughly independent of its neighbour, then do the inference
on the batch means. The effective sample size is the number of batches. It is
chosen from the measured integrated autocorrelation time rather than fixed, because
a fixed batch length is either too short for a persistent series (no correction) or
too long for an independent one (real results become unfindable).

Nothing here subtracts variance from an expected value. Variance is risk, not cost;
this codebase has already made that mistake once, in the rotation model.
"""
from __future__ import annotations

import math
import statistics as _stats
from decimal import Decimal
from typing import Dict, Iterable, List, Optional, Sequence, Union

__all__ = [
    "describe",
    "exceedance",
    "autocorrelation",
    "integrated_autocorrelation_time",
    "effective_sample_size",
    "batch_means_interval",
    "persistence_runs",
]

Number = Union[float, int, Decimal]

# 95% two-sided normal quantile. A t-quantile would be more correct for very few
# batches; `batch_means_interval` refuses below 8 batches instead, where the
# difference between the two starts to matter.
_Z_95 = 1.959963985


def _floats(values: Iterable[Number]) -> List[float]:
    """Decimal in, float out, at the analysis boundary and nowhere else.

    Prices and edges are carried as Decimal everywhere upstream, because a binary
    float shifts a price in exactly the digits this strategy trades on. Statistics
    are a different matter: a variance is not a price, nobody settles against it,
    and float is what the arithmetic here needs. The conversion is confined to this
    one function so it cannot creep back toward the money.
    """
    return [float(v) for v in values]


def describe(values: Iterable[Number]) -> Dict[str, Optional[float]]:
    """Location, spread and tails of a sample.

    Percentiles rather than just the mean because the distribution's shape is the
    result here: a mean of -1.5 bps with a p99 of +12 bps is a different market
    from a mean of -1.5 bps with a p99 of -1.4 bps, and only one of them has
    anything worth chasing.
    """
    data = sorted(_floats(values))
    n = len(data)
    if n == 0:
        return {
            "n": 0, "mean": None, "sd": None, "min": None, "max": None,
            "p1": None, "p10": None, "p50": None, "p90": None, "p99": None,
        }

    def pct(p: float) -> float:
        if n == 1:
            return data[0]
        # Linear interpolation between order statistics; no scipy dependency and
        # no silent off-by-one from integer indexing.
        position = (n - 1) * p
        low = int(math.floor(position))
        high = min(low + 1, n - 1)
        weight = position - low
        return data[low] * (1 - weight) + data[high] * weight

    return {
        "n": n,
        "mean": _stats.fmean(data),
        # None, not 0.0: one sample carries no information about spread, and a
        # reported zero reads as certainty.
        "sd": _stats.stdev(data) if n > 1 else None,
        "min": data[0],
        "max": data[-1],
        "p1": pct(0.01),
        "p10": pct(0.10),
        "p50": pct(0.50),
        "p90": pct(0.90),
        "p99": pct(0.99),
    }


def exceedance(
    values: Iterable[Number], thresholds: Sequence[Number]
) -> Dict[float, Optional[float]]:
    """Fraction of observations STRICTLY above each threshold.

    Strictly, because a threshold here is a floor -- the minimum edge worth
    trading -- and an observation exactly at the floor is not a trade.

    This is the strategy's central empirical question stated as a curve: not "is
    there edge on average" but "how often is there enough edge", which is a
    different quantity and the one that determines whether a bot has anything to do.
    """
    data = _floats(values)
    if not data:
        return {float(t): None for t in thresholds}
    n = len(data)
    return {
        float(t): sum(1 for v in data if v > float(t)) / n
        for t in thresholds
    }


def autocorrelation(values: Iterable[Number], lag: int) -> Optional[float]:
    """Sample autocorrelation at `lag`, or None if the series is too short.

    None rather than 0.0 for an impossible lag: zero correlation is a strong claim,
    and "no information" must not be mistaken for "no dependence" by a caller
    summing over lags.
    """
    data = _floats(values)
    n = len(data)
    if lag < 0:
        raise ValueError(f"lag must be non-negative, got {lag}")
    if n < lag + 2:
        return None
    mean = _stats.fmean(data)
    deviations = [v - mean for v in data]
    denominator = sum(d * d for d in deviations)
    if denominator == 0:
        # A constant series. Perfectly correlated with itself at every lag; saying
        # so is more useful than dividing by zero.
        return 1.0
    numerator = sum(
        deviations[i] * deviations[i + lag] for i in range(n - lag)
    )
    return numerator / denominator


def integrated_autocorrelation_time(
    values: Iterable[Number], max_lag: Optional[int] = None
) -> float:
    """tau = 1 + 2 * sum_k rho_k, truncated where rho first goes non-positive.

    tau is roughly "how many consecutive observations one independent fact is
    spread across", so n/tau is the information content of the series.

    The truncation is the standard initial-positive-sequence rule. It matters more
    than it looks: the tail of an estimated autocorrelation function is mostly
    noise, and summing all of it adds variance without adding signal -- which for
    long series makes tau wander, sometimes below 1, quietly restoring the
    uncorrected sample size.
    """
    data = _floats(values)
    n = len(data)
    if n < 4:
        return 1.0
    # n // 4 caps the sum where estimates are still based on a decent number of
    # pairs; beyond that each rho_k rests on too few terms to be worth summing.
    if max_lag is None:
        max_lag = max(1, n // 4)

    total = 0.0
    for lag in range(1, max_lag + 1):
        rho = autocorrelation(data, lag)
        if rho is None or rho <= 0:
            break
        total += rho
    tau = 1.0 + 2.0 * total
    # Never below 1: a series cannot carry more information than it has samples,
    # and a tau under 1 would inflate the effective size above n.
    return max(1.0, tau)


def effective_sample_size(values: Iterable[Number]) -> int:
    """n / tau, floored at 1 and capped at n.

    This is the number that goes into a standard error. Using len(values) instead
    is the error the whole module exists to prevent.
    """
    data = _floats(values)
    n = len(data)
    if n == 0:
        return 0
    tau = integrated_autocorrelation_time(data)
    return max(1, min(n, int(n / tau)))


def batch_means_interval(
    values: Iterable[Number],
    confidence_z: float = _Z_95,
    min_batches: int = 8,
) -> Dict[str, Optional[float]]:
    """A 95% interval for the mean that accounts for persistence.

    Batch length comes from the measured autocorrelation time, so an independent
    series is barely penalised and a persistent one is penalised in proportion to
    how much it repeats itself. A fixed batch length cannot do both: too short and
    it fails to correct, too long and every real result becomes unfindable.

    Returns `lower`/`upper` of None with a `reason` when the sample cannot support
    an interval. That is deliberate -- a wide interval computed from three batches
    looks like a result, while None looks like what it is.
    """
    data = _floats(values)
    n = len(data)
    empty = {
        "n": n, "mean": None, "sd": None, "lower": None, "upper": None,
        "effective_n": 0, "batches": 0, "batch_size": 0, "tau": None,
        "standard_error": None, "excludes_zero": None, "reason": None,
    }
    if n == 0:
        return {**empty, "reason": "no observations"}

    mean = _stats.fmean(data)
    sd = _stats.stdev(data) if n > 1 else None

    # Sample size first, before variance. Fewer observations than the minimum batch
    # count cannot yield independent batches under ANY variance -- including zero,
    # where the degenerate branch below would otherwise hand back a confident
    # [mean, mean] built from a handful of points. Five identical readings say
    # nothing about the population; two hundred at least say the feed was static.
    if n < min_batches:
        return {
            **empty,
            "mean": mean, "sd": sd,
            "reason": (
                f"only {n} observations; {min_batches} are needed before an "
                f"interval means anything, whatever the variance"
            ),
        }

    # Zero variance, handled before tau. A constant series correlates with itself
    # at 1.0 at every lag, so tau explodes and the batch test below would refuse
    # with "too few batches" -- which reads as "not enough data" when the truth is
    # the opposite: the sample mean has no uncertainty at all. A pool that did not
    # trade over the window produces exactly this, and it is a real result.
    if n > 1 and sd == 0.0:
        return {
            **empty,
            "n": n, "mean": mean, "sd": 0.0,
            "lower": mean, "upper": mean,
            "effective_n": 1, "batches": 1, "batch_size": n,
            "tau": float(n), "standard_error": 0.0,
            "excludes_zero": mean != 0.0,
            "reason": (
                "every observation is identical, so the interval is degenerate. "
                "Note this says nothing about the population: a stalled feed and a "
                "genuinely static market look the same here."
            ),
        }

    tau = integrated_autocorrelation_time(data)

    # One batch per autocorrelation time, and at least min_batches of them, so the
    # batch means are close to independent while still numerous enough for their
    # spread to mean something.
    batch_size = max(1, int(math.ceil(tau)))
    batches = n // batch_size
    if batches < min_batches:
        return {
            **empty,
            "mean": mean, "sd": sd, "tau": tau,
            "effective_n": effective_sample_size(data),
            "batches": batches, "batch_size": batch_size,
            "reason": (
                f"only {batches} independent batches of {batch_size} available "
                f"from {n} observations (autocorrelation time {tau:.1f}); "
                f"{min_batches} are needed before an interval means anything"
            ),
        }

    batch_means = [
        _stats.fmean(data[i * batch_size:(i + 1) * batch_size])
        for i in range(batches)
    ]
    batch_sd = _stats.stdev(batch_means) if len(batch_means) > 1 else 0.0
    standard_error = batch_sd / math.sqrt(batches)
    half_width = confidence_z * standard_error
    lower, upper = mean - half_width, mean + half_width

    return {
        "n": n,
        "mean": mean,
        "sd": sd,
        "lower": lower,
        "upper": upper,
        # The honest sample size. Reported so a reader can see how much of the raw
        # count was repetition.
        "effective_n": batches,
        "batches": batches,
        "batch_size": batch_size,
        "tau": tau,
        "standard_error": standard_error,
        # The strategy's actual question, at the confidence level requested.
        "excludes_zero": (lower > 0) or (upper < 0),
        "reason": None,
    }


def persistence_runs(flags: Sequence[bool]) -> List[int]:
    """Lengths of consecutive True runs, in observations.

    Multiply by the measured cadence to get opportunity lifetimes in seconds. That
    is the quantity that decides whether a 2.3-second detection loop is fast enough:
    an edge with a median lifetime of one observation cannot be captured by a system
    that needs two to notice it.

    A run still open at the end of the sample IS counted. Dropping it biases
    lifetimes downward, and the longest opportunity in a recording is very often
    the one that had not closed when recording stopped.
    """
    runs: List[int] = []
    current = 0
    for flag in flags:
        if flag:
            current += 1
        elif current:
            runs.append(current)
            current = 0
    if current:
        runs.append(current)
    return runs
