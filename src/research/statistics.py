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
    "decay_profile",
    "half_life_seconds",
    "autocorrelation_converged",
    "sign_test_p_value",
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


def decay_profile(
    values: Sequence[Number],
    cadence_seconds: float,
    lags_seconds: Sequence[float] = (2.0, 5.0, 12.0, 30.0, 60.0, 300.0),
) -> Dict[float, Optional[float]]:
    """Autocorrelation of a series at lags expressed in SECONDS.

    This is the number that decides whether a detection loop is fast enough. An edge
    whose autocorrelation has fallen to zero by 12 seconds cannot be captured by a
    system that settles on a 12-second block, however good its cost model. Stated in
    seconds rather than in observations because the cadence differs between runs and
    is itself a measured quantity -- reporting "correlation 0.4 at lag 1" invites the
    reader to supply their own cadence, which is how a 2.3s figure and a 36s figure
    get compared as if they were the same.

    A lag shorter than the cadence is UNMEASURABLE and returns None rather than the
    lag-1 value. Rounding a 2-second question up to a 36-second answer is how the
    previous placebo in this project came to compare each quote with a copy of itself.
    """
    if cadence_seconds <= 0:
        raise ValueError(f"cadence_seconds must be positive, got {cadence_seconds}")
    out: Dict[float, Optional[float]] = {}
    for lag_seconds in lags_seconds:
        if lag_seconds < cadence_seconds:
            out[float(lag_seconds)] = None
            continue
        lag = int(round(lag_seconds / cadence_seconds))
        out[float(lag_seconds)] = autocorrelation(values, max(1, lag))
    return out


def half_life_seconds(
    values: Sequence[Number], cadence_seconds: float, max_lag_seconds: float = 600.0
) -> Optional[float]:
    """How long until the autocorrelation first falls below 0.5, in seconds.

    A compact summary of the decay profile, and the direct answer to "how long does an
    opportunity stay an opportunity". Returns None when the series never decays that
    far within the window -- a persistent basis rather than a fleeting dislocation,
    which is a different phenomenon and must not be reported as a very long half-life.
    """
    if cadence_seconds <= 0:
        raise ValueError("cadence_seconds must be positive")
    max_lag = int(max_lag_seconds / cadence_seconds)
    previous_lag, previous_rho = 0.0, 1.0
    for lag in range(1, max(2, max_lag) + 1):
        rho = autocorrelation(values, lag)
        if rho is None:
            return None
        if rho < 0.5:
            # Linear interpolation between the bracketing lags, so the answer is not
            # quantised to the sampling interval.
            span = previous_rho - rho
            fraction = (previous_rho - 0.5) / span if span > 0 else 0.0
            return (previous_lag + fraction) * cadence_seconds
        previous_lag, previous_rho = float(lag), rho
    return None

def autocorrelation_converged(values: Iterable[Number]) -> bool:
    """Did the autocorrelation actually decay inside this sample?

    The integrated autocorrelation time is estimated by summing rho_k until rho first
    goes non-positive, capped at n/4 because beyond that each rho_k rests on too few
    pairs to be worth summing. That cap makes the estimator CENSORED: if the series has
    not decorrelated by lag n/4, the sum stops early and tau comes back far too small.

    Which defeats any guard built on tau for the case it most needs to catch. Measured on
    an AR(1) with phi=0.99, whose true tau is about 115:

        n=  30   tau estimated  2.92   ->  10.3 "correlation times"
        n=  60   tau estimated 17.84   ->   3.4
        n=4000   tau estimated 115.37  ->  34.7

    At n=30 the estimator reports a MORE independent sample than at n=60, because it
    cannot see past lag 7. A short, strongly persistent series therefore looks like a
    long, weakly persistent one, and that is exactly the sample whose persistence must
    not be trusted.

    So the honest test is whether the sum terminated on its own. If rho is still positive
    at the largest measurable lag, the correlation time is not merely uncertain -- it is
    below the resolution of the sample, and so is anything derived from it.
    """
    data = _floats(values)
    n = len(data)
    if n < 8:
        return False
    max_lag = max(1, n // 4)
    for lag in range(1, max_lag + 1):
        rho = autocorrelation(data, lag)
        if rho is None:
            return False
        if rho <= 0:
            # Terminated naturally: the series forgot itself within the sample.
            return True
    return False

def sign_test_p_value(values: Iterable[Number]) -> Optional[float]:
    """Two-sided p-value for "this series is symmetric about zero", on EFFECTIVE draws.

    The point of doing it this way rather than counting sign flips against a threshold:
    a binary "standing basis" label hides its own sample-size dependence, and this
    project got that wrong in the obvious direction. ETH/USDC 0.05% Arbitrum classified
    as a standing basis with 0.0% sign flips on 236 observations; an hour later, with
    1,713, it flipped sign in 48.7% of them.

    The 236-observation reading was not unreasonable and was not right. Its
    autocorrelation time was 20, so it held about 11 independent draws, and 11 draws all
    on one side has probability 2^-10 = 0.001 under a symmetric null -- suggestive, and
    nowhere near the certainty a bare label conveys. The honest output is the p-value,
    because it carries the sample size with it.

    Uses the effective sample size, not the row count. Using the row count would make a
    persistent series look overwhelmingly significant purely for having been sampled
    often, which is the same inflation `batch_means_interval` exists to prevent.
    """
    data = _floats(values)
    if len(data) < 8:
        return None
    effective = effective_sample_size(data)
    if effective < 2:
        return None

    positive = sum(1 for v in data if v > 0)
    negative = sum(1 for v in data if v < 0)
    total = positive + negative
    if total == 0:
        return None
    # Scale the observed split down to the effective sample, rounding the minority side
    # UP so the test is conservative: it never claims more asymmetry than it has.
    minority_fraction = min(positive, negative) / total
    minority = int(math.ceil(minority_fraction * effective))

    # Two-sided binomial tail at p=0.5.
    cumulative = sum(math.comb(effective, k) for k in range(0, minority + 1))
    tail = cumulative / (2 ** effective)
    return min(1.0, 2.0 * tail)
