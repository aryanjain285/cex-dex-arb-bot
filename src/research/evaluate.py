"""Turn a recorded observation into a decision, and price what latency does to it.

The two things this module must not do.

IT MUST NOT PEEK. A decision at time t uses the observation at t and nothing else.
The temptation is structural rather than careless: the obvious way to write a
latency model is to look up the later observation and compute the best trade
available then. That silently re-optimises size and direction using information the
bot could not have had, which does not make the backtest slightly optimistic -- it
makes it profitable on pure noise, because the strategy only ever trades when it
already knows the answer. So `resolve_with_latency` takes a decision whose size and
direction are already fixed, and advances only the prices.

IT MUST NOT ASSUME AWAY LATENCY. The existing simulator fills at the recorded touch,
instantly and in full. For a strategy whose edge is single-digit basis points, whose
detection loop was measured at 2.32 seconds, and whose settlement layer produces a
block every 12, instantaneous execution does not simplify the problem -- it deletes
the dominant cost. A trade with no successor observation near the intended delay is
UNRESOLVED, not filled: silently dropping those rows would select the periods when
the recorder kept up, and those are the calm ones.

Everything is computed against an explicit `CostModel` that travels with the result,
because a basis-point figure without its cost assumptions cannot be compared to
another one.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from typing import Dict, List, Optional, Sequence

from .observations import Observation
from .optimiser import SizeCurve, SizePoint, optimise_size

__all__ = [
    "CostModel",
    "Decision",
    "ObservationResult",
    "ResolvedTrade",
    "evaluate_observation",
    "resolve_with_latency",
]

ZERO = Decimal("0")
TEN_THOUSAND = Decimal("10000")
DIRECTIONS = ("CEX_to_DEX", "DEX_to_CEX")


@dataclass(frozen=True)
class CostModel:
    """The assumptions every basis-point figure in a report depends on.

    Frozen and hashable so a result set can be labelled by it. Two runs with
    different cost models produce numbers that must not be pooled, and the only
    reliable way to prevent that is for the model to be part of the result.
    """

    taker_fee_bps: Decimal
    cex_legs: int
    gas_units: int
    rotation_cost_quote: Decimal
    floor_bps: Decimal

    def __post_init__(self):
        if self.taker_fee_bps < 0:
            raise ValueError(f"taker_fee_bps must be non-negative, got {self.taker_fee_bps}")
        if self.cex_legs < 1:
            raise ValueError(f"cex_legs must be at least 1, got {self.cex_legs}")
        if self.gas_units <= 0:
            raise ValueError(
                f"gas_units must be positive, got {self.gas_units}; a zero gas "
                f"limit is a zero gas cost by another name, and gas is the same "
                f"order of magnitude as the edge being measured"
            )
        if self.rotation_cost_quote < 0:
            raise ValueError("rotation_cost_quote must be non-negative")
        if self.floor_bps < 0:
            raise ValueError("floor_bps must be non-negative")


@dataclass(frozen=True)
class Decision:
    """What the strategy would have done, decided from one observation only."""

    direction: str
    size_base: Decimal
    notional_quote: Decimal
    net_bps: Decimal
    gross_bps: Decimal
    cex_price: Decimal
    dex_price: Decimal
    decided_at: float


@dataclass(frozen=True)
class ObservationResult:
    """Everything one observation supports, decided and undecided alike."""

    ts: float
    cex_symbol: str
    costs: CostModel
    curves: Dict[str, SizeCurve] = field(default_factory=dict)
    best: Optional[Decision] = None
    # The best GROSS dislocation at any priceable size, in either direction. The
    # research signal: it survives every cost assumption, so it is reported even
    # when -- especially when -- nothing is tradeable.
    best_gross_bps: Optional[Decimal] = None
    best_gross_direction: Optional[str] = None
    best_gross_notional: Optional[Decimal] = None
    # What a single fixed probe size would have concluded, for comparison.
    probe_net_bps: Optional[Decimal] = None
    probe_notional: Optional[Decimal] = None
    # Set when the observation could not be evaluated at all.
    reason: Optional[str] = None


@dataclass(frozen=True)
class ResolvedTrade:
    """A frozen decision, re-priced against a later observation."""

    direction: str
    size_base: Decimal
    decided_at: float
    decided_net_bps: Decimal
    resolved_at: Optional[float] = None
    realised_delay_seconds: Optional[float] = None
    realised_net_bps: Optional[float] = None
    realised_gross_bps: Optional[float] = None
    # Slippage attributable to the delay, in bps: decided minus realised.
    decay_bps: Optional[float] = None
    unresolved_reason: Optional[str] = None


# --- evaluating one observation -------------------------------------------


def evaluate_observation(
    observation: Observation,
    costs: CostModel,
    notionals: Sequence[Decimal],
    *,
    base_is_token0: bool,
    probe_notional: Optional[Decimal] = None,
) -> ObservationResult:
    """Both directions, over a grid of notionals, from this observation alone.

    `base_is_token0` is required rather than inferred: getting it wrong inverts the
    DEX price, a factor of price squared, and there is no value of the answer that
    looks obviously wrong in a table of basis points.
    """
    gas_quote = observation.gas_quote(costs.gas_units)
    if gas_quote is None:
        # Not zero. A missing gas price makes an observation uncostable, and
        # treating it as free is the single easiest way to make this strategy look
        # profitable -- its edge and its gas are the same size.
        return ObservationResult(
            ts=observation.ts,
            cex_symbol=observation.cex_symbol,
            costs=costs,
            reason=(
                "no gas price recorded for this observation, so it cannot be "
                "costed. Treating gas as zero would invert the result."
            ),
        )
    if not observation.cex_bids or not observation.cex_asks:
        return ObservationResult(
            ts=observation.ts, cex_symbol=observation.cex_symbol, costs=costs,
            reason="the CEX book is not two-sided in this observation",
        )

    curves: Dict[str, SizeCurve] = {}
    for direction in DIRECTIONS:
        curves[direction] = optimise_size(
            pool=observation.pool,
            direction=direction,
            cex_bids=observation.cex_bids,
            cex_asks=observation.cex_asks,
            notionals=notionals,
            taker_fee_bps=costs.taker_fee_bps,
            gas_quote=gas_quote,
            base_is_token0=base_is_token0,
            rotation_cost_quote=costs.rotation_cost_quote,
            cex_legs=costs.cex_legs,
            floor_bps=costs.floor_bps,
        )

    best_decision = _best_decision(curves, observation)
    gross_bps, gross_direction, gross_notional = _best_gross(curves)
    probe_bps = (
        _probe_net_bps(curves, probe_notional)
        if probe_notional is not None else None
    )

    return ObservationResult(
        ts=observation.ts,
        cex_symbol=observation.cex_symbol,
        costs=costs,
        curves=curves,
        best=best_decision,
        best_gross_bps=gross_bps,
        best_gross_direction=gross_direction,
        best_gross_notional=gross_notional,
        probe_net_bps=probe_bps,
        probe_notional=probe_notional,
    )


def _best_decision(
    curves: Dict[str, SizeCurve], observation: Observation
) -> Optional[Decision]:
    candidates = []
    for direction, curve in curves.items():
        if curve.best is None:
            continue
        candidates.append((direction, curve.best))
    if not candidates:
        return None
    direction, point = max(candidates, key=lambda pair: pair[1].net_bps)
    return Decision(
        direction=direction,
        size_base=point.size_base,
        notional_quote=point.notional_quote,
        net_bps=point.net_bps,
        gross_bps=point.gross_bps,
        cex_price=point.cex_price,
        dex_price=point.dex_price,
        decided_at=observation.ts,
    )


def _best_gross(curves: Dict[str, SizeCurve]):
    best = (None, None, None)
    for direction, curve in curves.items():
        if curve.best_gross_bps is None:
            continue
        if best[0] is None or curve.best_gross_bps > best[0]:
            best = (curve.best_gross_bps, direction, curve.best_gross_notional)
    return best


def _probe_net_bps(
    curves: Dict[str, SizeCurve], probe_notional: Decimal
) -> Optional[Decimal]:
    """Net bps at the notional nearest the probe, over both directions.

    "Nearest on the grid" rather than "exactly the probe" so the comparison works
    on a geometric grid that need not contain the probe exactly. The distance is
    small by construction and the alternative -- evaluating one extra size -- would
    give the probe an accuracy the real system does not have either.
    """
    best = None
    for curve in curves.values():
        priceable = curve.priceable()
        if not priceable:
            continue
        nearest = min(
            priceable, key=lambda p: abs(p.notional_requested - probe_notional)
        )
        if best is None or nearest.net_bps > best:
            best = nearest.net_bps
    return best


# --- latency ---------------------------------------------------------------


def resolve_with_latency(
    result: ObservationResult,
    future: Sequence[Observation],
    *,
    delay_seconds: float,
    tolerance_seconds: float,
    base_is_token0: bool,
) -> ResolvedTrade:
    """Re-price a decision against the observation nearest `decided_at + delay`.

    The decision's direction and size are FROZEN. Only prices advance. This is the
    look-ahead guard, and it is the reason this function takes a `Decision` rather
    than recomputing one: re-optimising at resolution time would let the trade flip
    direction on information the bot never had, and any noise series would then
    appear profitable.

    An observation outside `tolerance_seconds` of the target time leaves the trade
    unresolved. Widening the tolerance silently instead would mean the realised
    delay differs from the reported one, and the reported one is the whole variable
    under study.
    """
    decision = result.best
    if decision is None:
        return ResolvedTrade(
            direction="", size_base=ZERO, decided_at=result.ts,
            decided_net_bps=ZERO,
            unresolved_reason="there was no trade to resolve",
        )

    target = decision.decided_at + delay_seconds
    candidates = [
        o for o in future
        if abs(o.ts - target) <= tolerance_seconds and o.ts >= decision.decided_at
    ]
    if not candidates:
        return ResolvedTrade(
            direction=decision.direction, size_base=decision.size_base,
            decided_at=decision.decided_at, decided_net_bps=decision.net_bps,
            unresolved_reason=(
                f"no observation within {tolerance_seconds}s of "
                f"{target:.3f} (decided {decision.decided_at:.3f} + "
                f"{delay_seconds}s delay)"
            ),
        )
    # Nearest to the target, not the first past it: with irregular cadence "first
    # past" is systematically later than the delay requested, so the model would
    # measure a longer latency than it reports.
    later = min(candidates, key=lambda o: abs(o.ts - target))

    gas_quote = later.gas_quote(result.costs.gas_units)
    if gas_quote is None:
        return ResolvedTrade(
            direction=decision.direction, size_base=decision.size_base,
            decided_at=decision.decided_at, decided_net_bps=decision.net_bps,
            resolved_at=later.ts,
            realised_delay_seconds=later.ts - decision.decided_at,
            unresolved_reason="no gas price in the resolving observation",
        )

    # The frozen trade, at the later prices. One notional, one direction: this is a
    # re-pricing, not a search.
    curve = optimise_size(
        pool=later.pool,
        direction=decision.direction,
        cex_bids=later.cex_bids,
        cex_asks=later.cex_asks,
        notionals=[decision.notional_quote],
        taker_fee_bps=result.costs.taker_fee_bps,
        gas_quote=gas_quote,
        base_is_token0=base_is_token0,
        rotation_cost_quote=result.costs.rotation_cost_quote,
        cex_legs=result.costs.cex_legs,
        # floor_bps=0: the floor is a decision rule, and the decision has already
        # been made. Applying it again here would discard exactly the trades that
        # went wrong, which is the loss this function exists to measure.
        floor_bps=ZERO,
    )
    point = curve.curve[0]
    realised_delay = later.ts - decision.decided_at

    if point.net_bps is None:
        return ResolvedTrade(
            direction=decision.direction, size_base=decision.size_base,
            decided_at=decision.decided_at, decided_net_bps=decision.net_bps,
            resolved_at=later.ts, realised_delay_seconds=realised_delay,
            unresolved_reason=(
                f"the trade could not be priced at resolution time: {point.reason}"
            ),
        )

    return ResolvedTrade(
        direction=decision.direction,
        size_base=decision.size_base,
        decided_at=decision.decided_at,
        decided_net_bps=decision.net_bps,
        resolved_at=later.ts,
        realised_delay_seconds=realised_delay,
        realised_net_bps=float(point.net_bps),
        realised_gross_bps=float(point.gross_bps),
        decay_bps=float(decision.net_bps - point.net_bps),
    )
