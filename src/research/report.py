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
from dataclasses import dataclass, field, replace
from decimal import Decimal
from typing import Any, Dict, List, Optional, Sequence, Tuple

from loguru import logger

from .evaluate import (
    CostModel,
    ObservationResult,
    best_priceable_decision,
    evaluate_observation,
    resolve_with_latency,
)
from .observations import Observation, ObservationStore, mid_dislocation_bps
from .statistics import (
    batch_means_interval,
    describe,
    exceedance,
    persistence_runs,
)

__all__ = [
    "MarketReport", "analyse_store", "group_key", "format_report",
    "scrambled_control", "format_summary_table", "classify_dislocation",
    "BASIS_FLIP_THRESHOLD",
]

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
    # The RAW dislocation: pool mid against CEX mid, before any fee, spread, impact
    # or direction choice. The only figure here with nothing subtracted from it, and
    # therefore the one that says whether the phenomenon exists at all. A negative
    # best-gross is ambiguous between "the venues are at parity and the fees are
    # unavoidable" and "the venues disagree, just not enough"; those have opposite
    # implications and only this separates them.
    dislocation_bps: Dict[str, Optional[float]] = field(default_factory=dict)
    abs_dislocation_bps: Dict[str, Optional[float]] = field(default_factory=dict)
    # Standing basis or fluctuating dislocation. The same +3 bps reads as a highly
    # reliable signal under one and as an unharvestable price under the other.
    basis_kind: Dict[str, Any] = field(default_factory=dict)
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

    dislocations = []
    for observation in observations:
        value = mid_dislocation_bps(observation, base_is_token0)
        if value is not None:
            dislocations.append(float(value))

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
        dislocation_bps=describe(dislocations),
        basis_kind=classify_dislocation(dislocations),
        # Magnitude, because either sign is tradeable in principle: the feasibility
        # question is whether |dislocation| ever exceeds the round-trip cost.
        abs_dislocation_bps=describe([abs(v) for v in dislocations]),
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
    basis = "tradeable"

    if not decided:
        # No observation cleared the floor -- which is every market in this dataset.
        # Fall back to the best PRICEABLE size: "had you traded the best available
        # size at t, what would it have been worth at t+delta". A counterfactual, so
        # the basis is reported alongside; without it the cost of latency goes
        # unmeasured for want of a qualifying trade, and that cost is one of the two
        # things the research exists to establish.
        basis = "counterfactual: best priceable size, floor ignored"
        decided = []
        for result in results:
            counterfactual = best_priceable_decision(result)
            if counterfactual is None:
                continue
            decided.append(replace(result, best=counterfactual))

    if not decided:
        return {
            "decisions": 0, "resolved": 0, "unresolved": 0,
            "mean_realised_net_bps": None, "median_realised_net_bps": None,
            "mean_decay_bps": None, "mean_realised_delay_seconds": None,
            "fraction_still_profitable": None, "tolerance_seconds": tolerance,
            "basis": "none: no size on the grid could be priced",
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
        # Which set of decisions this was computed on. A latency figure from
        # hypothetical trades is not the same statement as one from real ones, and
        # nothing downstream can tell them apart without this.
        "basis": basis,
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

    raw, absraw = report.dislocation_bps, report.abs_dislocation_bps
    if raw.get("n"):
        lines.append(
            f"  RAW dislocation (pool mid vs CEX mid, no fees at all)"
        )
        lines.append(
            f"              signed  mean {_fmt(raw.get('mean'))}  "
            f"p1 {_fmt(raw.get('p1'))}  p99 {_fmt(raw.get('p99'))}"
        )
        lines.append(
            f"              |size|  median {_fmt(absraw.get('p50'))}  "
            f"p90 {_fmt(absraw.get('p90'))}  p99 {_fmt(absraw.get('p99'))}  "
            f"max {_fmt(absraw.get('max'))}"
        )
        # The comparison that decides feasibility, stated rather than left implied.
        round_trip = float(report.pool_fee) / 100.0 + float(report.costs.taker_fee_bps)
        median_abs = absraw.get("p50") or 0.0
        shortfall = round_trip / median_abs if median_abs > 0 else None
        lines.append(
            f"              a one-way trade must clear {round_trip:.1f} bps "
            f"(pool fee {report.pool_fee / 100:.0f} + taker "
            f"{report.costs.taker_fee_bps})"
            + (
                f"; the median dislocation is {shortfall:.1f}x too small"
                if shortfall and shortfall > 1
                else "; the median dislocation clears it"
            )
        )
        kind = report.basis_kind.get("kind")
        if kind == "standing_basis":
            lines.append(
                f"              STANDING BASIS: the sign flips in only "
                f"{report.basis_kind['sign_flip_fraction']:.1%} of observations. "
                f"This is a price, not an error -- what the market charges for the "
                f"asset being on this chain rather than in that custodian. Capturing "
                f"it once is an inventory move; capturing it again needs the "
                f"inventory bridged back, and the bridge costs the basis. NOT a "
                f"repeatable per-trade edge."
            )
        elif kind == "fluctuating":
            lines.append(
                f"              fluctuating: the sign flips in "
                f"{report.basis_kind['sign_flip_fraction']:.1%} of observations, so "
                f"inventory returns on its own and the constraint is cost"
            )
        elif report.basis_kind.get("reason"):
            lines.append(f"              basis: {report.basis_kind['reason']}")

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
            + (f"  [{stat['basis']}]" if stat.get("basis") != "tradeable" else "")
        )
    return "\n".join(lines)


# --- negative control -----------------------------------------------------


def scrambled_control(
    observations: Sequence[Observation],
    costs: CostModel,
    notionals: Sequence[Decimal],
    *,
    base_is_token0: bool,
    offset_seconds: float,
) -> Dict[str, Any]:
    """Compare the real pairing against a deliberately mismatched one.

    Every other check in this stack is internal: it verifies the code does what it
    was designed to do. None can tell whether the design measures the market. A sign
    error, an inverted token order or a mis-specified cost all produce a distribution
    of "edge" that looks like data.

    So: evaluate each observation's CEX book against a pool snapshot from a distant
    time. Whatever structure survives cannot be a real dislocation, because the two
    sides never coexisted. The scrambled distribution's upper tail is then a bound on
    how much apparent edge pure mismatch can manufacture, and an observed edge below
    that bound proves nothing whatever its mean.

    The offset is validated against the observed cadence. An earlier placebo in this
    project used a one-second delay against twelve-second blocks, so 69% of its
    "placebo" quotes were identical to the real ones and it could not have detected
    anything.
    """
    observations = sorted(observations, key=lambda o: o.ts)
    empty = {"n": 0, "mean": None, "sd": None, "p50": None, "p90": None, "p99": None}

    cadence = _median_cadence([o.ts for o in observations])
    if cadence is not None and offset_seconds < cadence * 2:
        raise ValueError(
            f"offset_seconds {offset_seconds} is less than twice the observed "
            f"cadence ({cadence:.1f}s). A scramble shorter than the sampling "
            f"interval pairs many rows with near-copies of themselves, which is how "
            f"a placebo comes to measure nothing -- the previous one in this project "
            f"produced 69% identical pairs and could not have detected anything."
        )

    def _series(pairs):
        out = []
        for book_obs, pool_obs in pairs:
            merged = _swap_pool(book_obs, pool_obs)
            result = evaluate_observation(
                merged, costs, notionals, base_is_token0=base_is_token0
            )
            if result.best_gross_bps is not None:
                out.append(float(result.best_gross_bps))
        return out

    true_series = _series([(o, o) for o in observations])

    # Pair each book with the pool snapshot `offset_seconds` later, wrapping around.
    # Wrapping rather than truncating so the two samples have the same size and the
    # comparison is not also a comparison of sample lengths.
    scrambled_pairs = []
    identical = 0
    for observation in observations:
        target = observation.ts + offset_seconds
        span = observations[-1].ts - observations[0].ts
        if span > 0 and target > observations[-1].ts:
            target = observations[0].ts + ((target - observations[0].ts) % span)
        partner = min(observations, key=lambda o: abs(o.ts - target))
        if partner.ts == observation.ts:
            identical += 1
        scrambled_pairs.append((observation, partner))

    reason = None
    if len(observations) < 3:
        scrambled_series = []
        reason = (
            f"only {len(observations)} observations; a scramble needs enough of a "
            f"span that the mismatched pairs are genuinely from different times"
        )
    else:
        scrambled_series = _series(scrambled_pairs)
        if identical:
            reason = (
                f"{identical} of {len(observations)} scrambled pairs reproduced the "
                f"true pairing, so the control is diluted by that fraction"
            )

    true_stats = describe(true_series) if true_series else dict(empty)
    scrambled_stats = describe(scrambled_series) if scrambled_series else dict(empty)

    noise_bound = scrambled_stats.get("p99")
    exceeds = None
    if noise_bound is not None and true_stats.get("p99") is not None:
        # The question that matters: is the real tail heavier than what mismatch
        # alone produces? If not, observations above that level prove nothing.
        exceeds = true_stats["p99"] > noise_bound

    return {
        "true": true_stats,
        "scrambled": scrambled_stats,
        "offset_seconds": offset_seconds,
        "median_cadence_seconds": cadence,
        "identical_pairs": identical,
        # The level below which an apparent edge is indistinguishable from noise.
        "noise_bound_bps": noise_bound,
        "exceeds_noise": exceeds,
        "reason": reason,
    }


def _swap_pool(book_observation: Observation, pool_observation: Observation) -> Observation:
    """A book from one instant with a pool from another.

    Built by replacement rather than by constructing a new Observation field-by-field
    so that any field added later is carried across automatically -- a scramble that
    silently dropped a new field would differ from the real evaluation in more ways
    than the one under test.
    """
    if pool_observation is book_observation:
        return book_observation
    return replace(
        book_observation,
        pool=pool_observation.pool,
        gas_price_wei=pool_observation.gas_price_wei,
    )


def format_summary_table(reports: Sequence[MarketReport]) -> str:
    """One line per market, sorted by the best gross edge observed.

    Sorted by GROSS rather than net because gross is the research signal: it says
    whether the phenomenon exists, separately from whether this cost structure can
    capture it. Sorting by net would rank the markets by our own fee assumptions and
    bury a genuinely dislocated pool behind a cheap one that never moves.

    `obs` and `eff` both appear because the gap between them is the finding half the
    time. 800 observations carrying 40 independent facts and 800 carrying 780 support
    very different claims, and only one number distinguishes them.
    """
    lines = [
        f"{'market':<30} {'obs':>6} {'eff':>5} {'gross':>8} {'p99':>8} "
        f"{'net':>8} {'>5bps':>8} {'life':>7} {'unpr':>6}",
        "-" * 96,
    ]
    ordered = sorted(
        reports,
        key=lambda r: (r.gross_bps.get("mean") is None, -(r.gross_bps.get("mean") or 0)),
    )
    for report in ordered:
        label = f"{report.cex_symbol} {report.chain} {report.pool_fee}"
        gross = report.gross_bps
        exceed = report.exceedance_net.get(5.0)
        lines.append(
            f"{label:<30} {report.observations:>6,} "
            f"{report.gross_interval.get('effective_n') or 0:>5,} "
            f"{_fmt(gross.get('mean')):>8} {_fmt(gross.get('p99')):>8} "
            f"{_fmt(report.net_bps.get('mean')):>8} "
            f"{('-' if exceed is None else f'{exceed:.3%}'):>8} "
            f"{_fmt(report.median_lifetime_seconds, '.0f'):>7} "
            f"{report.unpriceable:>6,}"
        )
    lines.append("")
    lines.append(
        "gross/net/p99 in bps, best of both directions at the best size on the grid. "
        "life = median seconds an opportunity persists. unpr = observations no size "
        "on the grid could price, which is not evidence of no edge."
    )
    return "\n".join(lines)


# --- standing basis versus fluctuating dislocation ------------------------

# Below this fraction of sign changes, a dislocation is treated as a standing basis
# rather than something a taker can harvest repeatedly. 5% is deliberately
# conservative: it takes a genuinely one-sided series to earn the label, because the
# label is the stronger claim.
BASIS_FLIP_THRESHOLD = 0.05

# Fewer observations than this and the question is not asked. Three readings sharing a
# sign are not evidence of a standing basis, and calling them one would be the
# strongest available conclusion from the weakest available sample.
MIN_OBSERVATIONS_TO_CLASSIFY = 20


def classify_dislocation(values, flip_threshold: float = BASIS_FLIP_THRESHOLD):
    """Standing basis, or fluctuating dislocation? They mean opposite things.

    FLUCTUATING: the sign changes. The pool crosses the exchange price in both
    directions, so a taker buys on whichever venue is cheap and sells on the other,
    and inventory returns on its own. The binding constraint is cost.

    STANDING: the sign does not change. The pool is persistently richer or cheaper,
    which is a price rather than an error -- what the market charges for the asset
    being on that chain instead of in that custodian. Capturing it once is an
    inventory repositioning; capturing it twice needs the inventory moved back across
    the bridge, and the bridge costs the basis. That is why the basis exists.

    The distinction matters because it inverts how the same number reads. A report
    showing "+3 bps, 100% of observations, on both ETH pairs" looks like an unusually
    reliable signal. It is the opposite: a signal that cannot be harvested repeatedly.
    """
    data = [float(v) for v in values if v is not None]
    if len(data) < MIN_OBSERVATIONS_TO_CLASSIFY:
        return {
            "kind": "unknown",
            "sign_flip_fraction": None,
            "median_bps": (_stats.median(data) if data else None),
            "flip_threshold": flip_threshold,
            "n": len(data),
            "reason": (
                f"{len(data)} observations is too few to distinguish a standing "
                f"basis from a fluctuating one; {MIN_OBSERVATIONS_TO_CLASSIFY} needed"
            ),
        }

    positive = sum(1 for v in data if v > 0)
    negative = sum(1 for v in data if v < 0)
    # The fraction on the MINORITY side. A basis is one-sided, so this is near zero;
    # a fluctuating series has both sides well represented.
    minority = min(positive, negative) / len(data)

    return {
        "kind": "standing_basis" if minority <= flip_threshold else "fluctuating",
        "sign_flip_fraction": minority,
        "median_bps": _stats.median(data),
        "flip_threshold": flip_threshold,
        "n": len(data),
        "reason": None,
    }
