"""Turn a store of observations into statements a capital decision can rest on.

Design rules, each one a failure mode this had to avoid.

ONE REPORT PER MARKET. A market is (pair, chain, fee tier), not a pair. WETH/USDC
0.05% on Arbitrum and on Ethereum share a CEX book and nothing else: different
liquidity, different competition, gas differing by two orders of magnitude.
Averaging them produces a number describing neither, and the output looks tidier
for it, which is why the error survives review.

EVERY MEAN CARRIES ITS UNCERTAINTY, COMPUTED FOR CORRELATED DATA. "Mean gross
-1.5 bps" invites a conclusion. "-1.5 bps, 95% CI [-1.9, -1.1], effective n 340 of
4,000" invites the right one. Observations seconds apart are nearly the same fact
recorded twice, so the raw count would inflate every t-statistic; see
`statistics.batch_means_interval`.

REFUSALS ARE COUNTED, NOT DROPPED. An observation the simulator declined to price
is not an observation of no edge. Dropped silently, a pool too thin to quote at any
size reports the same "no opportunities" as a deep pool at parity -- and those two
findings have opposite implications.

GROSS IS REPORTED EVEN WHEN NET IS HOPELESS. Gross dislocation is the research
signal: it survives every cost assumption, so it says whether the phenomenon exists
at all, separately from whether this configuration can capture it. A report that
only said "nothing clears the floor" would conflate "there is nothing there" with
"our costs are too high", and only one of those is fixable.

LATENCY IS A REPORTED VARIABLE, NOT AN ASSUMPTION. Decisions are made from one
observation and re-priced against a later one, with size and direction frozen. The
delay is swept, so its cost is measured rather than argued about.
"""
from __future__ import annotations

import statistics as _stats
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Dict, List, Optional, Sequence, Tuple

from loguru import logger

from .evaluate import (
    CostModel,
    ObservationResult,
    evaluate_observation,
    resolve_with_latency,
)
from .observations import Observation, ObservationStore
from .statistics import (
    batch_means_interval,
    describe,
    exceedance,
    persistence_runs,
)

__all__ = ["MarketReport", "analyse_store", "group_key", "format_report"]

# Thresholds the exceedance curve is reported at, in bps. Chosen to span the
# decision: 0 is "does the dislocation ever favour us at all", 5 is the configured
# floor, and 20-50 is where a trade would survive a bad fill and a gas spike.
DEFAULT_THRESHOLDS = (0.0, 5.0, 10.0, 20.0, 50.0)

# Delays swept by default, in seconds. 2.3 is the measured detector cadence of this
# system; 12 is one Ethereum block, which is the floor on settlement.
DEFAULT_LATENCY_DELAYS = (0.0, 2.3, 12.0)

MarketKey = Tuple[str, str, int]


def group_key(observation: Observation) -> MarketKey:
    """(pair, chain, fee) -- the identity of a market, not of a pair."""
    return (observation.cex_symbol, observation.chain, observation.pool_fee)


@dataclass(frozen=True)
class MarketReport:
    cex_symbol: str
    chain: str
    pool_fee: int
    costs: CostModel
    notionals: Tuple[Decimal, ...]

    # -- what was observed
    observations: int = 0
    # Rows that could not be costed at all (no gas price recorded, one-sided book).
    # Reported because they are not evidence of no edge.
    uncostable: int = 0
    # Rows costed but where no size on the grid could be priced -- the pool was too
    # thin, or the quote would have left the observed tick window.
    unpriceable: int = 0
    span_seconds: Optional[float] = None
    median_cadence_seconds: Optional[float] = None
    endpoints: Tuple[str, ...] = ()

    # -- the research signal
    gross_bps: Dict[str, Optional[float]] = field(default_factory=dict)
    gross_interval: Dict[str, Optional[float]] = field(default_factory=dict)
    net_bps: Dict[str, Optional[float]] = field(default_factory=dict)
    net_interval: Dict[str, Optional[float]] = field(default_factory=dict)
    exceedance_gross: Dict[float, Optional[float]] = field(default_factory=dict)
    exceedance_net: Dict[float, Optional[float]] = field(default_factory=dict)

    # -- tradeability
    tradeable_observations: int = 0
    optimal_notionals: Dict[str, Optional[float]] = field(default_factory=dict)
    direction_counts: Dict[str, int] = field(default_factory=dict)

    # -- time structure
    median_lifetime_seconds: Optional[float] = None
    p90_lifetime_seconds: Optional[float] = None
    opportunity_episodes: int = 0

    # -- the fixed-probe comparison
    probe_notional: Optional[Decimal] = None
    probe_understatement_bps: Optional[float] = None

    # -- latency, by requested delay in seconds
    latency: Dict[float, Dict[str, Optional[float]]] = field(default_factory=dict)

    def tradeable_fraction(self) -> Optional[float]:
        costed = self.observations - self.uncostable
        return (self.tradeable_observations / costed) if costed else None


def analyse_store(
    store: ObservationStore,
    costs: CostModel,
    notionals: Sequence[Decimal],
    *,
    base_is_token0: bool,
    probe_notional: Optional[Decimal] = None,
    thresholds: Sequence[float] = DEFAULT_THRESHOLDS,
    latency_delays: Sequence[float] = DEFAULT_LATENCY_DELAYS,
    since: Optional[float] = None,
    until: Optional[float] = None,
) -> List[MarketReport]:
    """One report per market found in the store.

    `base_is_token0` is a single flag here rather than per market, which is correct
    only when every recorded pool puts the base on the same side. The recorder's
    targets are checked against this by the caller; a mismatch inverts the DEX price
    and shows up as an implausible edge, which the survey's plausibility bound
    catches.
    """
    grouped: Dict[MarketKey, List[Observation]] = {}
    for observation in store.read_all(since=since, until=until):
        grouped.setdefault(group_key(observation), []).append(observation)

    reports = []
    for key in sorted(grouped):
        reports.append(_analyse_market(
            key, grouped[key], costs, notionals,
            base_is_token0=base_is_token0,
            probe_notional=probe_notional,
            thresholds=thresholds,
            latency_delays=latency_delays,
        ))
    return reports


def _analyse_market(
    key: MarketKey,
    observations: List[Observation],
    costs: CostModel,
    notionals: Sequence[Decimal],
    *,
    base_is_token0: bool,
    probe_notional: Optional[Decimal],
    thresholds: Sequence[float],
    latency_delays: Sequence[float],
) -> MarketReport:
    symbol, chain, fee = key
    observations = sorted(observations, key=lambda o: o.ts)

    results: List[ObservationResult] = []
    uncostable = 0
    for observation in observations:
        result = evaluate_observation(
            observation, costs, notionals,
            base_is_token0=base_is_token0, probe_notional=probe_notional,
        )
        if result.reason is not None:
            uncostable += 1
            continue
        results.append(result)

    gross_series = [
        float(r.best_gross_bps) for r in results if r.best_gross_bps is not None
    ]
    # Unpriceable: costable, but no size on the grid could be quoted. A distinct
    # count from `uncostable` because they have different fixes -- a wider tick
    # window versus a gas price the recorder failed to read.
    unpriceable = sum(1 for r in results if r.best_gross_bps is None)

    # Net for every costable observation, using the best net achievable at any size
    # -- or the least-bad priceable point when nothing clears the floor. Without the
    # latter the net distribution would contain only winners, which is exactly the
    # selection bias that makes a strategy look profitable.
    net_series: List[float] = []
    optimal_sizes: List[float] = []
    direction_counts: Dict[str, int] = {}
    tradeable_flags: List[bool] = []
    probe_gaps: List[float] = []

    for result in results:
        best_net = _best_net_any_size(result)
        if best_net is not None:
            net_series.append(float(best_net))
        is_tradeable = result.best is not None
        tradeable_flags.append(is_tradeable)
        if is_tradeable:
            optimal_sizes.append(float(result.best.notional_quote))
            direction_counts[result.best.direction] = (
                direction_counts.get(result.best.direction, 0) + 1
            )
            if result.probe_net_bps is not None:
                probe_gaps.append(float(result.best.net_bps - result.probe_net_bps))

    cadence = _median_cadence([o.ts for o in observations])
    lifetimes_rows = persistence_runs(tradeable_flags)
    lifetimes_seconds = (
        [r * cadence for r in lifetimes_rows] if cadence else None
    )

    latency_stats = {}
    for delay in latency_delays:
        latency_stats[float(delay)] = _latency_study(
            results, observations, delay, cadence, base_is_token0
        )

    return MarketReport(
        cex_symbol=symbol, chain=chain, pool_fee=fee,
        costs=costs, notionals=tuple(notionals),
        observations=len(observations),
        uncostable=uncostable,
        unpriceable=unpriceable,
        span_seconds=(
            observations[-1].ts - observations[0].ts if len(observations) > 1 else 0.0
        ),
        median_cadence_seconds=cadence,
        endpoints=tuple(sorted({
            o.rpc_endpoint for o in observations if o.rpc_endpoint
        })),
        gross_bps=describe(gross_series),
        gross_interval=batch_means_interval(gross_series),
        net_bps=describe(net_series),
        net_interval=batch_means_interval(net_series),
        exceedance_gross=exceedance(gross_series, thresholds),
        exceedance_net=exceedance(net_series, thresholds),
        tradeable_observations=sum(tradeable_flags),
        optimal_notionals=describe(optimal_sizes),
        direction_counts=direction_counts,
        median_lifetime_seconds=(
            _stats.median(lifetimes_seconds) if lifetimes_seconds else None
        ),
        p90_lifetime_seconds=(
            _percentile(lifetimes_seconds, 0.90) if lifetimes_seconds else None
        ),
        opportunity_episodes=len(lifetimes_rows),
        probe_notional=probe_notional,
        probe_understatement_bps=(
            _stats.fmean(probe_gaps) if probe_gaps else None
        ),
        latency=latency_stats,
    )


def _best_net_any_size(result: ObservationResult) -> Optional[Decimal]:
    """The best net bps at any priceable size, floor or no floor.

    Deliberately not `result.best.net_bps`, which is None whenever nothing clears
    the floor. Using that would make the net distribution contain only the winners
    -- the selection bias that makes any strategy look profitable.
    """
    best = None
    for curve in result.curves.values():
        for point in curve.curve:
            if point.net_bps is None:
                continue
            if best is None or point.net_bps > best:
                best = point.net_bps
    return best


def _median_cadence(timestamps: Sequence[float]) -> Optional[float]:
    if len(timestamps) < 2:
        return None
    gaps = [b - a for a, b in zip(timestamps, timestamps[1:]) if b > a]
    return _stats.median(gaps) if gaps else None


def _percentile(values: Sequence[float], p: float) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * p
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    weight = position - low
    return ordered[low] * (1 - weight) + ordered[high] * weight


def _latency_study(
    results: Sequence[ObservationResult],
    observations: Sequence[Observation],
    delay: float,
    cadence: Optional[float],
    base_is_token0: bool,
) -> Dict[str, Optional[float]]:
    """What a decision is worth `delay` seconds after it was made.

    The tolerance scales with cadence: at a 10-second cadence there is no
    observation within half a second of any target, so a fixed tolerance would
    report every trade unresolved and the study would silently produce nothing. One
    cadence either side is the tightest window that can contain a successor at all.
    """
    tolerance = max(0.5, (cadence or 1.0) * 1.0)
    decided = [r for r in results if r.best is not None]
    if not decided:
        return {
            "decisions": 0, "resolved": 0, "unresolved": 0,
            "mean_realised_net_bps": None, "median_realised_net_bps": None,
            "mean_decay_bps": None, "mean_realised_delay_seconds": None,
            "fraction_still_profitable": None, "tolerance_seconds": tolerance,
        }

    realised, decays, delays_seen = [], [], []
    unresolved = 0
    for result in decided:
        resolved = resolve_with_latency(
            result, observations, delay_seconds=delay,
            tolerance_seconds=tolerance, base_is_token0=base_is_token0,
        )
        if resolved.realised_net_bps is None:
            unresolved += 1
            continue
        realised.append(resolved.realised_net_bps)
        if resolved.decay_bps is not None:
            decays.append(resolved.decay_bps)
        if resolved.realised_delay_seconds is not None:
            delays_seen.append(resolved.realised_delay_seconds)

    return {
        "decisions": len(decided),
        "resolved": len(realised),
        "unresolved": unresolved,
        "mean_realised_net_bps": _stats.fmean(realised) if realised else None,
        "median_realised_net_bps": _stats.median(realised) if realised else None,
        "mean_decay_bps": _stats.fmean(decays) if decays else None,
        # The achieved delay, not the requested one: with irregular cadence they
        # differ, and the achieved one is what the numbers measure.
        "mean_realised_delay_seconds": (
            _stats.fmean(delays_seen) if delays_seen else None
        ),
        # The number that decides whether the strategy is viable at this latency.
        "fraction_still_profitable": (
            sum(1 for v in realised if v > 0) / len(realised) if realised else None
        ),
        "tolerance_seconds": tolerance,
    }


# --- presentation ---------------------------------------------------------


def _fmt(value, spec=".2f", absent="-"):
    return absent if value is None else format(value, spec)


def format_report(report: MarketReport) -> str:
    """A human-readable block. Uncertainty and refusals are never omitted."""
    lines = []
    lines.append(
        f"=== {report.cex_symbol}  {report.chain}  fee {report.pool_fee} ==="
    )
    span_hours = (report.span_seconds or 0) / 3600
    lines.append(
        f"  observations {report.observations:,} over {span_hours:.2f}h "
        f"at {_fmt(report.median_cadence_seconds)}s median cadence"
    )
    if report.uncostable or report.unpriceable:
        lines.append(
            f"  NOT EVIDENCE OF NO EDGE: {report.uncostable:,} uncostable "
            f"(no gas price / one-sided book), {report.unpriceable:,} unpriceable "
            f"(pool too thin, or outside the observed tick window)"
        )

    gross, gi = report.gross_bps, report.gross_interval
    lines.append(
        f"  gross bps   mean {_fmt(gross.get('mean'))}  "
        f"median {_fmt(gross.get('p50'))}  "
        f"p90 {_fmt(gross.get('p90'))}  max {_fmt(gross.get('max'))}"
    )
    if gi.get("lower") is not None:
        lines.append(
            f"              95% CI [{_fmt(gi['lower'])}, {_fmt(gi['upper'])}]  "
            f"effective n {gi['effective_n']:,} of {gi['n']:,} "
            f"(autocorrelation time {_fmt(gi.get('tau'), '.1f')})  "
            f"excludes zero: {gi.get('excludes_zero')}"
        )
    else:
        lines.append(f"              no interval: {gi.get('reason')}")

    net = report.net_bps
    lines.append(
        f"  net bps     mean {_fmt(net.get('mean'))}  "
        f"median {_fmt(net.get('p50'))}  p90 {_fmt(net.get('p90'))}  "
        f"max {_fmt(net.get('max'))}"
    )
    exceed = "  ".join(
        f">{int(t)}: {('-' if v is None else f'{v:.4%}')}"
        for t, v in sorted(report.exceedance_net.items())
    )
    lines.append(f"  net exceedance   {exceed}")

    fraction = report.tradeable_fraction()
    lines.append(
        f"  tradeable   {report.tradeable_observations:,} observations"
        + (f" ({fraction:.4%} of costable)" if fraction is not None else "")
        + f", {report.opportunity_episodes} episodes"
    )
    if report.median_lifetime_seconds is not None:
        lines.append(
            f"  lifetime    median {_fmt(report.median_lifetime_seconds, '.1f')}s  "
            f"p90 {_fmt(report.p90_lifetime_seconds, '.1f')}s"
        )
    if report.direction_counts:
        lines.append(f"  directions  {report.direction_counts}")
    if report.optimal_notionals.get("n"):
        opt = report.optimal_notionals
        lines.append(
            f"  best size   median {_fmt(opt.get('p50'), ',.0f')}  "
            f"p10 {_fmt(opt.get('p10'), ',.0f')}  p90 {_fmt(opt.get('p90'), ',.0f')}"
        )
    if report.probe_understatement_bps is not None:
        lines.append(
            f"  a fixed {report.probe_notional:,.0f} probe understates the best "
            f"achievable net edge by {report.probe_understatement_bps:.2f} bps "
            f"on average"
        )
    for delay in sorted(report.latency):
        stat = report.latency[delay]
        if not stat.get("decisions"):
            continue
        profitable = stat["fraction_still_profitable"]
        profitable_text = "-" if profitable is None else f"{profitable:.2%}"
        lines.append(
            f"  latency {delay:>5.1f}s  decided {stat['decisions']:,}  "
            f"resolved {stat['resolved']:,}  unresolved {stat['unresolved']:,}  "
            f"realised net {_fmt(stat['mean_realised_net_bps'])} bps  "
            f"decay {_fmt(stat['mean_decay_bps'])} bps  "
            f"still profitable {profitable_text}"
        )
    return "\n".join(lines)
