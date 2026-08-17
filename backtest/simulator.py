"""Replay a recorded market against the live strategy components.

The point of this module is that NOTHING about the decision is re-implemented
here. The detector, the router, the risk manager and the executor are the
production objects; only the two venue clients are replaced, and they do nothing
but hand over what the dataset recorded. A backtest that re-derives economics
tests its own arithmetic rather than the strategy's.

The previous version failed that test in four independent ways -- it implemented
`get_quote` where the detector calls `get_book`, returned a `Quote` where a
`DexQuote` was required, and read three attributes that do not exist
(`is_complete_success`, `net_pnl_quote`, `MarketPair.quote`) -- so it could not
process a single row. It also applied its own `slippage = Decimal("0.001")` to the
DEX price, which invented a number AND double-counted price impact that the DEX
quote already contains.

Two limitations of a bid/ask/price CSV are stated here rather than assumed,
because either one silently flatters the strategy:

DEPTH. The dataset records prices, not ladders, so the book is synthesised with
one level per side at `depth_per_level_base`. Every fill is therefore an
assumption about liquidity that the data cannot support. The detector's real
depth check still applies to it, so a size beyond that assumption is refused
rather than filled at the top of book.

GAS. Every row must carry the swap's gas cost, either as a `gas_quote` column or
as `gas_price_gwei` together with `native_price_quote`. A missing gas cost raises.
Treating it as zero is the single easiest way to make a losing strategy look
profitable, and this strategy's whole question is whether a few basis points
survive its costs.

REPLAY TIME. The detector rejects a book whose FEED is stale, measured against
the process clock. Historical timestamps would make every book stale and the
backtest would find nothing, so a replayed book is stamped with the current time
-- in a replay the feed is live by definition. The historical instant is carried
separately and is what the report prints.
"""
from __future__ import annotations

from decimal import Decimal
from typing import List, Optional

import pandas as pd
from loguru import logger

from src.core import clock
from src.core.config import AppConfig
from src.core.types import (
    BookSnapshot, CexOrder, DexQuote, DexSwapParams, DexTxReceipt,
    ExecutionSummary, MarketPair, OrderUpdate,
)
from src.risk.limits import RiskManager
from src.strategy.detector import OpportunityDetector
from src.strategy.executor import TransactionExecutor
from src.strategy.router import SimpleRouter

ZERO = Decimal("0")

# Default synthesised depth per book level, in base units. Large enough not to
# be the binding constraint at the default 1000-notional target, so a run that
# finds nothing is telling you about prices rather than about this number. It is
# a constructor argument precisely so a run can test sensitivity to it.
DEFAULT_DEPTH_PER_LEVEL_BASE = Decimal("1000")


def _decimal(value, column: str) -> Decimal:
    """Convert a dataset cell to Decimal without going through binary float.

    `load_dataset` already maps the price columns through `str()` for this
    reason; this guards the columns it does not.
    """
    if value is None or (isinstance(value, float) and pd.isna(value)):
        raise ValueError(f"column {column!r} is empty")
    return value if isinstance(value, Decimal) else Decimal(str(value))


def gas_quote_for_row(tick: pd.Series, gas_units: int) -> Decimal:
    """The swap's gas cost in the quote currency, for one row.

    Raises rather than defaulting. A backtest with no gas is not a cheaper
    backtest, it is a different and easier strategy than the one being tested.
    """
    if "gas_quote" in tick.index:
        try:
            return _decimal(tick["gas_quote"], "gas_quote")
        except ValueError:
            pass  # fall through to the gwei form

    has_gwei = "gas_price_gwei" in tick.index
    has_native = "native_price_quote" in tick.index
    if has_gwei and has_native:
        try:
            gwei = _decimal(tick["gas_price_gwei"], "gas_price_gwei")
            native = _decimal(tick["native_price_quote"], "native_price_quote")
        except ValueError as exc:
            raise ValueError(
                f"gas cannot be priced for this row: {exc}. Provide gas_quote, "
                f"or gas_price_gwei together with native_price_quote."
            ) from exc
        # gwei -> native units -> quote currency
        return gwei * Decimal("1e-9") * Decimal(gas_units) * native

    raise ValueError(
        "the dataset carries no usable gas cost. Add a `gas_quote` column (the "
        "swap's cost in the quote currency), or `gas_price_gwei` together with "
        "`native_price_quote`. Gas is not treated as zero: a few basis points of "
        "edge is exactly the scale gas operates at, so a zero would invert the "
        "result."
    )


# --- venue doubles -------------------------------------------------------


class BacktestCexClient:
    """Serves the recorded bid/ask as a one-level-per-side book.

    Deliberately not a subclass of CexClient: the abstract base declares an
    order-placement surface this object has no business implementing, and
    inheriting it previously hid the fact that the method the detector actually
    calls -- get_book -- was missing.
    """

    def __init__(self, pair: MarketPair, depth_per_level_base: Decimal):
        self.pair = pair
        self.depth_per_level_base = depth_per_level_base
        self._tick: Optional[pd.Series] = None

    def set_tick(self, tick: pd.Series) -> None:
        self._tick = tick

    async def connect(self) -> None:
        pass

    async def close(self) -> None:
        pass

    async def get_book(self, pair: MarketPair) -> Optional[BookSnapshot]:
        if self._tick is None:
            return None
        bid = _decimal(self._tick["cex_bid_price"], "cex_bid_price")
        ask = _decimal(self._tick["cex_ask_price"], "cex_ask_price")
        now = clock.now()
        return BookSnapshot(
            pair=pair,
            bids=[(bid, self.depth_per_level_base)],
            asks=[(ask, self.depth_per_level_base)],
            # Both stamps are replay time. See REPLAY TIME in the module
            # docstring: the alternative is a feed that is stale by construction.
            timestamp=now,
            feed_timestamp=now,
        )

    async def create_order(self, order: CexOrder) -> OrderUpdate:
        """Fill at the recorded touch, immediately and completely.

        An optimistic assumption, and named as one: a real taker order can be
        partially filled or filled worse. Status is lowercase to match the live
        client, which is what every caller compares against.
        """
        price = (
            self._tick["cex_ask_price"] if order.side == "buy"
            else self._tick["cex_bid_price"]
        )
        return OrderUpdate(
            order_id=f"backtest-cex-{self._tick.name.timestamp()}",
            status="filled",
            avg_fill_price=_decimal(price, "fill price"),
            filled_size=order.size,
            ts=clock.now(),
        )


class BacktestDexClient:
    """Serves the recorded DEX price and the row's gas cost, unmodified.

    The recorded price is used exactly as recorded. A QuoterV2 price is already
    net of both the pool fee and price impact for the size quoted, so adjusting
    it here would double-count the impact -- which is what the previous
    hardcoded 0.1% did.
    """

    def __init__(self, pair: MarketPair, gas_units: int):
        self.pair = pair
        self.gas_units = gas_units
        self._tick: Optional[pd.Series] = None

    def set_tick(self, tick: pd.Series) -> None:
        self._tick = tick

    async def get_quote(
        self, pair: MarketPair, size: Decimal, side: str,
        estimate_gas: bool = False,
    ) -> Optional[DexQuote]:
        if self._tick is None:
            return None
        return DexQuote(
            price=_decimal(self._tick["dex_price"], "dex_price"),
            gas_cost_quote=gas_quote_for_row(self._tick, self.gas_units),
        )

    async def execute_swap(self, params: DexSwapParams) -> DexTxReceipt:
        price = _decimal(self._tick["dex_price"], "dex_price")
        return DexTxReceipt(
            tx_hash=f"0xbacktest{int(self._tick.name.timestamp())}",
            status=True,
            block_number=0,
            gas_used=self.gas_units,
            effective_gas_price=0,
            avg_fill_price=price,
            filled_size=params.amount_in,
        )


# --- simulator -----------------------------------------------------------


class Simulator:
    def __init__(
        self,
        config: AppConfig,
        data: pd.DataFrame,
        depth_per_level_base: Decimal = DEFAULT_DEPTH_PER_LEVEL_BASE,
    ):
        self.config = config
        self.data = data
        self.depth_per_level_base = depth_per_level_base
        self.results: List[ExecutionSummary] = []
        # Rows the detector looked at but declined, so a run that trades nothing
        # can be distinguished from a run that never evaluated anything.
        self.rows_evaluated = 0
        self.opportunities_found = 0
        self.trades_refused = 0

        if not self.config.pairs:
            raise ValueError("no pairs configured; nothing to backtest")

        # The backtest covers the first configured pair only, which is a real
        # limitation and not an oversight: the CSV schema carries one pair's
        # prices per row.
        self.pair_config = self.config.pairs[0]
        self.market_pair = MarketPair(
            base=self.pair_config.base,
            quote_cex=self.pair_config.quote,
            quote_dex=self.pair_config.quote,  # direct pairs only
            cex_symbol=self.pair_config.cex_symbol,
            dex_chain=self.pair_config.dex_chain,
            dex_pool_fee=self.pair_config.dex_pool_fee,
            base_precision=(
                self.pair_config.base_precision
                if self.pair_config.base_precision is not None else 8
            ),
            quote_precision=(
                self.pair_config.quote_precision
                if self.pair_config.quote_precision is not None else 8
            ),
            min_net_bps=self.pair_config.min_net_bps,
            max_slippage_bps=self.pair_config.max_slippage_bps,
            max_size_quote=self.pair_config.max_size_quote,
            price_floor_quote=self.pair_config.price_floor_quote,
            price_ceiling_quote=self.pair_config.price_ceiling_quote,
            max_edge_bps=self.pair_config.max_edge_bps,
        )

        self.cex_client = BacktestCexClient(self.market_pair, depth_per_level_base)
        self.dex_client = BacktestDexClient(
            self.market_pair, config.dex.swap_gas_estimate_units
        )

        # Production components, unmodified. This is the whole point.
        self.detector = OpportunityDetector(
            config.strategy, self.cex_client, self.dex_client, [self.market_pair]
        )
        self.router = SimpleRouter()
        # state_path=None: a replay must not read or write the live bot's daily
        # loss budget. Sharing it let a simulated losing day halt the live bot,
        # and a simulated winning day raise its loss allowance.
        self.risk_manager = RiskManager(config.risk, state_path=None)
        self.executor = TransactionExecutor(
            self.cex_client, self.dex_client, self.risk_manager, config.pairs
        )

    def validate_dataset(self) -> None:
        """Check the whole dataset before evaluating any of it.

        Gas in particular must be validated here rather than per row. The
        detector isolates per-pair failures on purpose -- a raising client
        becomes a recorded rejection, not a crash -- so a dataset with no gas
        column would produce a clean run reporting zero opportunities. A silent
        "no edge here" is the worst possible outcome for a missing cost.
        """
        required = ("cex_bid_price", "cex_ask_price", "dex_price")
        missing = [c for c in required if c not in self.data.columns]
        if missing:
            raise ValueError(f"dataset is missing required columns: {missing}")

        for timestamp, tick in self.data.iterrows():
            try:
                gas_quote_for_row(tick, self.config.dex.swap_gas_estimate_units)
            except ValueError as exc:
                raise ValueError(f"row {timestamp}: {exc}") from exc

    async def run(self) -> None:
        self.validate_dataset()
        logger.info(
            f"Backtest starting over {len(self.data)} rows "
            f"({self.market_pair.cex_symbol}), synthesised depth "
            f"{self.depth_per_level_base} base per level."
        )
        for timestamp, tick in self.data.iterrows():
            self.cex_client.set_tick(tick)
            self.dex_client.set_tick(tick)
            self.rows_evaluated += 1

            opportunities = await self.detector.detect()
            self.opportunities_found += len(opportunities)

            for opp in opportunities:
                logger.info(
                    f"[{timestamp}] {opp.direction} net {opp.edge_bps:.2f} bps "
                    f"on {opp.size} base"
                )
                plan = self.router.plan(opp)
                if not self.risk_manager.is_trade_allowed(plan):
                    self.trades_refused += 1
                    continue

                summary = await self.executor.run(plan)
                if summary.legs:
                    self.results.append(summary)
                    # Feeds the same position and daily-loss accounting the live
                    # system uses, so a backtest can actually hit its own limits
                    # rather than trading an unbounded book.
                    self.risk_manager.update_state(summary)
                else:
                    # The executor's own sanity gate declined it -- an expired
                    # deadline or a price outside the pair's bounds.
                    self.trades_refused += 1

        logger.success(
            f"Backtest complete: {self.rows_evaluated} rows, "
            f"{self.opportunities_found} opportunities, {len(self.results)} trades, "
            f"{self.trades_refused} refused."
        )

    def report(self) -> None:
        quote_asset = self.market_pair.quote_cex
        print("\n--- Backtest Report ---")
        print(f"Pair:   {self.market_pair.cex_symbol} ({self.market_pair.dex_chain})")
        if len(self.data):
            print(f"Period: {self.data.index.min()} to {self.data.index.max()}")
        print(f"Rows evaluated:      {self.rows_evaluated}")
        print(f"Opportunities found: {self.opportunities_found}")
        print(f"Trades refused:      {self.trades_refused}")
        print(f"Synthesised depth:   {self.depth_per_level_base} base per level")

        if not self.results:
            # Stated explicitly: no trades is a result, not a failure to run.
            print("Trades executed:     0")
            print("No trades. With every row evaluated, that is a statement "
                  "about the market rather than about the harness.")
            print("------------------")
            return

        total_pnl = sum((s.pnl_quote for s in self.results), ZERO)
        wins = [s for s in self.results if s.pnl_quote > ZERO]
        losses = [s for s in self.results if s.pnl_quote <= ZERO]

        print(f"Trades executed:     {len(self.results)}")
        print(f"Total PnL:           {total_pnl:.4f} {quote_asset}")
        print(f"Win rate:            {len(wins) / len(self.results):.2%}")
        if wins:
            print(f"Average win:         "
                  f"{sum((w.pnl_quote for w in wins), ZERO) / len(wins):.4f}")
        if losses:
            print(f"Average loss:        "
                  f"{sum((l.pnl_quote for l in losses), ZERO) / len(losses):.4f}")
        print("------------------")
