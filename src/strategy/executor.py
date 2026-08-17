from dataclasses import dataclass
from decimal import Decimal
from typing import Dict, List, Optional, Tuple

from loguru import logger

from ..core import clock
from ..core.config import PairConfig
from ..core.types import Opportunity, ExecutionSummary, ExecutionLeg
from ..exchange.cex_base import CexClient
from ..exchange.dex_base import DexClient
from ..infra import metrics
from ..risk.limits import RiskManager


@dataclass(frozen=True)
class SanityThresholds:
    price_floor: Decimal
    price_ceiling: Decimal
    max_edge_bps: Decimal


DEFAULT_SANITY_THRESHOLDS = SanityThresholds(
    price_floor=Decimal("0"),
    price_ceiling=Decimal("1000000"),
    max_edge_bps=Decimal("50000"),
)
TEN_THOUSAND = Decimal("10000")
ZERO = Decimal("0")


def build_threshold_map(pair_configs: Optional[List[PairConfig]]) -> Dict[str, SanityThresholds]:
    if not pair_configs:
        return {}

    thresholds: Dict[str, SanityThresholds] = {}
    for cfg in pair_configs:
        price_floor = cfg.price_floor_quote if cfg.price_floor_quote is not None else DEFAULT_SANITY_THRESHOLDS.price_floor
        price_ceiling = cfg.price_ceiling_quote if cfg.price_ceiling_quote is not None else DEFAULT_SANITY_THRESHOLDS.price_ceiling
        max_edge = Decimal(cfg.max_edge_bps) if cfg.max_edge_bps is not None else DEFAULT_SANITY_THRESHOLDS.max_edge_bps
        thresholds[cfg.cex_symbol] = SanityThresholds(price_floor=price_floor, price_ceiling=price_ceiling, max_edge_bps=max_edge)
    return thresholds


def determine_leg_prices(opp: Opportunity) -> Tuple[Decimal, Decimal]:
    if opp.direction == "DEX_to_CEX":
        return opp.dex_price, opp.cex_price
    return opp.cex_price, opp.dex_price


def build_legs(opp: Opportunity) -> List[ExecutionLeg]:
    if opp.direction == "DEX_to_CEX":
        dex_leg = ExecutionLeg(
            venue="DEX",
            side="buy",
            price_quote=opp.dex_price,
            size=opp.size,
            fees_quote=ZERO,
        )
        cex_leg = ExecutionLeg(
            venue="CEX",
            side="sell",
            price_quote=opp.cex_price,
            size=opp.size,
            fees_quote=opp.cex_fee_quote,
        )
        return [dex_leg, cex_leg]

    cex_leg = ExecutionLeg(
        venue="CEX",
        side="buy",
        price_quote=opp.cex_price,
        size=opp.size,
        fees_quote=opp.cex_fee_quote,
    )
    dex_leg = ExecutionLeg(
        venue="DEX",
        side="sell",
        price_quote=opp.dex_price,
        size=opp.size,
        fees_quote=ZERO,
    )
    return [cex_leg, dex_leg]


def _count_rejection(opp: Opportunity, reason: str) -> None:
    """Record a refusal, in both executors, without ever raising.

    Telemetry must not be able to change what the executor returns -- a
    Counter.inc() that raised inside executor.run() previously discarded a
    trade's loss before the risk manager could see it.

    `status="invalid"` on trades_executed is kept alongside the new series so
    existing dashboards and alerts do not silently lose their signal.
    """
    try:
        metrics.opportunities_rejected_total.labels(
            pair=opp.pair.cex_symbol,
            direction=opp.direction,
            reason=reason,
        ).inc()
        metrics.trades_executed.labels(
            pair=opp.pair.cex_symbol,
            direction=opp.direction,
            status="invalid",
        ).inc()
    except Exception as exc:  # pragma: no cover - telemetry is never fatal
        logger.error(f"Failed to count an execution rejection: {exc}")


def evaluate_opportunity(
    opp: Opportunity,
    thresholds: SanityThresholds,
    now: Optional[float] = None,
) -> Tuple[List[ExecutionLeg], Optional[Decimal], Optional[Decimal], Optional[str]]:
    """Gate an opportunity and normalise it into legs.

    This deliberately does NOT recompute the trade economics. The detector
    owns them: it priced both venues with depth-weighted quotes and summed
    every cost exactly once in `costs.evaluate_trade`. An executor that
    re-derived PnL previously reapplied a slippage deduction for impact that
    the DEX quote already included, so the two components of the pipeline
    disagreed about the value of the same trade.

    What remains here is an independent sanity gate -- defence in depth
    against a bad price or a units error reaching execution.

    `now` is injectable so the deadline can be tested exactly rather than raced
    against the wall clock.
    """
    # Checked first, and before any price gate. An opportunity is a claim about
    # two prices at one instant; past its deadline it is not a worse opportunity
    # but a statement about a market that no longer exists. Reporting staleness
    # rather than the resulting price anomaly also points at the real cause,
    # which is latency.
    #
    # `valid_until` was previously written by the detector on every opportunity
    # and read nowhere, so the TTL was decorative.
    now = clock.now() if now is None else now
    if now >= opp.valid_until:
        return [], None, None, "opportunity_expired"

    buy_price, sell_price = determine_leg_prices(opp)

    if opp.cex_price <= ZERO:
        return [], None, None, "cex_price_non_positive"
    if opp.dex_price <= ZERO:
        return [], None, None, "dex_price_non_positive"

    if opp.cex_price < thresholds.price_floor or opp.cex_price > thresholds.price_ceiling:
        return [], None, None, "cex_price_out_of_range"
    if opp.dex_price < thresholds.price_floor or opp.dex_price > thresholds.price_ceiling:
        return [], None, None, "dex_price_out_of_range"

    if buy_price <= ZERO:
        return [], None, None, "buy_price_non_positive"
    if sell_price <= ZERO:
        return [], None, None, "sell_price_non_positive"

    if opp.size <= ZERO:
        return [], None, None, "size_non_positive"

    # The detector's net edge, re-checked against this pair's own ceiling.
    edge_bps = opp.edge_bps
    if abs(edge_bps) > thresholds.max_edge_bps:
        return [], None, edge_bps, "edge_beyond_threshold"

    legs = build_legs(opp)
    return legs, opp.expected_pnl_quote, edge_bps, None


class TransactionExecutor:
    """
    Converts an `Opportunity` into two normalised execution legs and computes
    the expected PnL and edge.

    - `DEX_to_CEX`: buy base on the DEX first, then sell base on the CEX.
    - `CEX_to_DEX`: buy base on the CEX first, then sell base on the DEX.

    `ExecutionLeg.price_quote` is always a quote-per-base price and
    `ExecutionLeg.size` is always a base quantity.

    PnL and edge are taken from the Opportunity, not recomputed here --
    `costs.evaluate_trade` is the single place costs are summed.

    Configurable price and edge sanity checks are applied so that anomalous
    market data cannot pollute the calculation.

    NOTE: this class does not currently place orders. It builds the plan and
    reports the expected outcome; wiring it to `CexClient.create_order` and
    `DexClient.execute_swap` is outstanding work.
    """

    def __init__(
        self,
        cex_client: Optional[CexClient],
        dex_client: Optional[DexClient],
        risk_manager: Optional[RiskManager],
        pair_configs: Optional[List[PairConfig]] = None,
    ):
        self.cex_client = cex_client
        self.dex_client = dex_client
        self.risk_manager = risk_manager
        self._thresholds = build_threshold_map(pair_configs)

    async def run(self, opp: Opportunity) -> ExecutionSummary:
        """Compute legs, PnL, and edge for the supplied `Opportunity`."""
        start_ts = clock.now()
        thresholds = self._thresholds.get(opp.pair.cex_symbol, DEFAULT_SANITY_THRESHOLDS)

        legs, pnl_quote, edge_bps, error = evaluate_opportunity(opp, thresholds)
        if error:
            logger.warning(
                "Opportunity failed sanity check: pair={} direction={} reason={}",
                opp.pair.cex_symbol,
                opp.direction,
                error,
            )
            _count_rejection(opp, error)
            return self._build_invalid_summary(opp, start_ts)

        summary = ExecutionSummary(
            pair=opp.pair,
            direction=opp.direction,
            size=opp.size,
            legs=legs,
            gas_quote=opp.gas_cost_quote,
            pnl_quote=pnl_quote,
            edge_bps=edge_bps,
            hedged=False,
            started_ts=start_ts,
            completed_ts=clock.now(),
        )

        # Metrics are emitted last and defensively: telemetry must never be
        # able to destroy a trade record. A Counter.inc() with a negative PnL
        # previously raised here, discarding the loss before the risk manager
        # could see it.
        try:
            metrics.trades_executed.labels(
                pair=opp.pair.cex_symbol,
                direction=opp.direction,
                status="success",
            ).inc()
            metrics.pnl_quote.labels(pair=opp.pair.cex_symbol).inc(float(pnl_quote))
        except Exception as exc:  # pragma: no cover - telemetry must not be fatal
            logger.error(f"Failed to emit execution metrics: {exc}")
        return summary

    def _build_invalid_summary(self, opp: Opportunity, start_ts: float) -> ExecutionSummary:
        return ExecutionSummary(
            pair=opp.pair,
            direction=opp.direction,
            size=opp.size,
            legs=[],
            gas_quote=ZERO,
            pnl_quote=ZERO,
            edge_bps=ZERO,
            hedged=False,
            started_ts=start_ts,
            completed_ts=clock.now(),
        )


class PaperExecutor:
    """
    An executor that only logs, and never places real trades.

    Produces the same legs and PnL as `TransactionExecutor` and applies the
    same sanity checks.
    """

    def __init__(self, pair_configs: Optional[List[PairConfig]] = None):
        logger.info("Paper executor initialised. Opportunities will be logged, not traded.")
        self._thresholds = build_threshold_map(pair_configs)

    async def run(self, opp: Opportunity) -> ExecutionSummary:
        start_ts = clock.now()
        thresholds = self._thresholds.get(opp.pair.cex_symbol, DEFAULT_SANITY_THRESHOLDS)

        legs, pnl_quote, edge_bps, error = evaluate_opportunity(opp, thresholds)
        if error:
            logger.warning(
                f"[PAPER MODE] Opportunity failed sanity check: pair={opp.pair.cex_symbol} direction={opp.direction} reason={error}"
            )
            _count_rejection(opp, error)
            return self._build_invalid_summary(opp, start_ts)

        logger.info(
            "[PAPER MODE] Opportunity detected: {} {} | CEX={:.4f} | DEX={:.4f} | PnL={:.4f}",
            opp.direction,
            opp.pair.cex_symbol,
            float(opp.cex_price),
            float(opp.dex_price),
            float(pnl_quote),
        )
        return ExecutionSummary(
            pair=opp.pair,
            direction=opp.direction,
            size=opp.size,
            legs=legs,
            gas_quote=opp.gas_cost_quote,
            pnl_quote=pnl_quote,
            edge_bps=edge_bps,
            hedged=False,
            started_ts=start_ts,
            completed_ts=clock.now(),
        )

    def _build_invalid_summary(self, opp: Opportunity, start_ts: float) -> ExecutionSummary:
        return ExecutionSummary(
            pair=opp.pair,
            direction=opp.direction,
            size=opp.size,
            legs=[],
            gas_quote=ZERO,
            pnl_quote=ZERO,
            edge_bps=ZERO,
            hedged=False,
            started_ts=start_ts,
            completed_ts=clock.now(),
        )
