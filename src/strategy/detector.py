"""Opportunity detection, and the audit record of every evaluation.

Decides on *net* economics computed from *depth-weighted* prices. Costs are
summed in exactly one place -- `costs.evaluate_trade` -- so the same quantity
cannot be charged twice.

Every evaluation produces an `EvaluationRecord`, including rejections and the
reason for them. That is deliberate: a threshold crossing on its own says
almost nothing, whereas the distribution of near misses reveals whether an
edge exists, how large it typically is, and whether the threshold is anywhere
near the right place. Recording only the crossings would leave a multi-week
run unable to answer the question it exists to answer.

Two conventions are load-bearing and easy to get wrong:

- A DEX quote is already net of the pool fee and of the price impact for the
  size requested. Nothing further is subtracted for either.
- `DexClient.get_quote(side="sell")` takes a size in *base* units, while
  `side="buy"` takes an amount of *quote* currency to spend. Passing a base
  amount to the buy side understates slippage and inflates the DEX_to_CEX
  edge; the two call sites below are deliberately asymmetric because of it.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from decimal import Decimal
from typing import Dict, List, Optional, Protocol, Tuple

from loguru import logger

from src.core import clock
from src.core.config import StrategyConfig
from src.core.types import BookSnapshot, MarketPair, Opportunity
from src.exchange.cex_base import CexClient
from src.exchange.dex_base import DexClient
from src.infra.evaluation_store import EvaluationRecord
from src.infra import metrics
from src.infra.metrics import opportunities_found
from src.exchange.errors import RpcError
from src.exchange.pool_selector import PoolSelector
from src.strategy.placebo import DelayedQuoteBuffer
from src.strategy.costs import (
    required_net_bps,
    rotation_risk_bps,
    BookFill,
    TradeEconomics,
    amortised_rotation_cost,
    evaluate_trade,
    walk_book,
)

ZERO = Decimal("0")
TEN_THOUSAND = Decimal("10000")


class RejectionReason:
    """Why an evaluation did not become an opportunity.

    Distinct values matter: a dataset that cannot separate "no liquidity" from
    "no feed" from "edge too small" cannot be used to calibrate anything.
    """

    NO_BOOK = "no_book"
    ONE_SIDED_BOOK = "one_sided_book"
    STALE_FEED = "stale_feed"
    INSUFFICIENT_DEPTH = "insufficient_depth"
    NO_DEX_QUOTE = "no_dex_quote"
    NO_INTERMEDIATE_PRICE = "no_intermediate_price"
    BAD_INPUTS = "bad_inputs"
    BELOW_FLOOR = "below_floor"
    ABOVE_SANITY = "above_sanity"
    NOT_BEST_DIRECTION = "not_best_direction"
    # A token on this pair is not cleared for capital: fee-on-transfer,
    # rebasing, or not withdrawable from the CEX. Recorded rather than
    # silently skipped, so a denial is visible in the dataset instead of
    # looking like a pair that simply never had an opportunity.
    TOKEN_DENIED = "token_denied"
    # The chain node did not answer -- throttled, timed out, or down. A
    # DIFFERENT fact from no_dex_quote: that one means the pool is empty and
    # the pair should be dropped, this one means slow down or buy a better
    # node. Collapsing them made a throttled bot look like an empty market.
    RPC_ERROR = "rpc_error"
    ERROR = "error"


class RecordSink(Protocol):
    """Anything that can persist an EvaluationRecord."""

    def record(self, record: EvaluationRecord) -> int: ...


@dataclass
class _Evaluation:
    """One direction's evaluation, before the best-of decision is made."""

    direction: str
    econ: Optional[TradeEconomics] = None
    reason: Optional[str] = None
    cex_best_bid: Optional[Decimal] = None
    cex_best_ask: Optional[Decimal] = None
    depth_levels_used: Optional[int] = None
    book_age_s: Optional[float] = None
    placebo_net_bps: Optional[Decimal] = None
    # The fee tier actually quoted. Recorded rather than taken from the
    # pair, because with routing enabled they differ -- and a row that names
    # a pool the price did not come from cannot be re-derived.
    dex_pool_fee_used: Optional[int] = None


class OpportunityDetector:
    def __init__(
        self,
        strategy_config: StrategyConfig,
        cex_client: CexClient,
        dex_client: DexClient,
        pairs: List[MarketPair],
        store: Optional[RecordSink] = None,
    ):
        self.strategy_config = strategy_config
        self.cex_client = cex_client
        self.dex_client = dex_client
        self.pairs = pairs
        self.store = store
        self._intermediate_price_cache: Dict[str, Tuple[Tuple[Decimal, Decimal], float]] = {}
        self._rotation_cost_quote = self._compute_rotation_cost()
        # A threshold, not a cost. See _compute_rotation_risk_bps.
        self._rotation_risk_bps = self._compute_rotation_risk_bps()
        placebo_cfg = strategy_config.placebo
        self._placebo = (
            DelayedQuoteBuffer(placebo_cfg.delay_seconds)
            if placebo_cfg.enabled else None
        )
        # Built once. Constructing it per evaluation would put a validation
        # error in the hot loop, where the only safe response is to keep going.
        self._token_policy = strategy_config.token_policy.build()

        routing = strategy_config.dex_routing
        self._pool_selector = (
            PoolSelector(
                dex_client, routing.candidate_fee_tiers, routing.refresh_seconds
            )
            if routing.enabled else None
        )
        logger.info(
            f"Opportunity detector initialised, monitoring {len(pairs)} pairs "
            f"({'recording' if store else 'NOT recording'} evaluations)."
        )
        logger.info(self._token_policy.describe())
        if self._rotation_cost_quote > ZERO:
            logger.info(
                f"Inventory rotation priced at {self._rotation_cost_quote:.4f} "
                f"per trade."
            )
        else:
            logger.warning(
                "Inventory rotation is priced at ZERO. This asserts that moving "
                "inventory between venues is free, which it is not."
            )

    def _compute_rotation_cost(self) -> Decimal:
        """Per-trade rotation cost, fixed for the life of the detector.

        Computed once rather than per evaluation: the inputs are configuration,
        not market data, so recomputing could only introduce divergence between
        the number used and the number recorded.
        """
        rotation = self.strategy_config.rotation
        if not rotation.enabled:
            return ZERO
        return amortised_rotation_cost(
            withdrawal_fee_quote=Decimal(str(rotation.withdrawal_fee_quote)),
            bridge_gas_quote=Decimal(str(rotation.bridge_gas_quote)),
            float_quote=Decimal(str(rotation.float_quote)),
            notional_quote=Decimal(str(self.strategy_config.target_notional_usd)),
        )

    def _compute_rotation_risk_bps(self) -> Decimal:
        """Risk charge for unhedged inventory in transit, as a floor adjustment.

        Deliberately NOT subtracted from the measured edge. Inventory exposure while
        in transit is variance, not a negative mean -- E[dP] = 0 under zero drift --
        and the previous model subtracted it as an expense, which made every
        recorded net_bps not the expected value of anything.
        """
        rotation = self.strategy_config.rotation
        if not rotation.enabled:
            return ZERO
        return rotation_risk_bps(
            float_quote=Decimal(str(rotation.float_quote)),
            notional_quote=Decimal(str(self.strategy_config.target_notional_usd)),
            transfer_risk_bps=Decimal(str(rotation.transfer_risk_bps)),
        )

    # ------------------------------------------------------------------
    # entry point
    # ------------------------------------------------------------------

    async def detect(self) -> List[Opportunity]:
        """Evaluate every pair concurrently.

        Failures are isolated per pair: without `return_exceptions=True` a
        single raising pair aborts the whole gather and the caller detects
        nothing at all for that cycle. An exception is itself recorded, so a
        pair that always fails is visible in the data rather than silently
        absent from it.
        """
        results = await asyncio.gather(
            *(self._evaluate_pair(p) for p in self.pairs), return_exceptions=True
        )

        opportunities: List[Opportunity] = []
        for pair, result in zip(self.pairs, results):
            if isinstance(result, BaseException):
                logger.warning(
                    f"Pair {pair.cex_symbol} failed evaluation: "
                    f"{type(result).__name__}: {result}"
                )
                self._emit(pair, _Evaluation(direction="", reason=RejectionReason.ERROR))
                continue
            if result is not None:
                opportunities.append(result)
        return opportunities

    async def check_pair(self, pair: MarketPair) -> Optional[Opportunity]:
        """Kept for callers that only want the opportunity."""
        return await self._evaluate_pair(pair)

    def _pair_symbols(self, pair: MarketPair) -> List[str]:
        """Every token this pair would move, including the ones not in its name.

        A synthetic pair trades against a third asset on-chain (`quote_dex`) and
        converts through a fourth CEX symbol (`intermediate_symbol`). Those are
        the easiest to overlook and just as capable of taking a transfer fee, so
        they are checked explicitly rather than assumed benign.
        """
        symbols = [pair.base, pair.quote_cex, pair.quote_dex]
        if pair.intermediate_symbol:
            # e.g. "ETH/USDT" -> both sides
            symbols.extend(
                part for part in pair.intermediate_symbol.split("/") if part
            )
        # De-duplicated but order-stable, so the rejection message names the
        # first offending token deterministically.
        seen, unique = set(), []
        for symbol in symbols:
            if symbol and symbol not in seen:
                seen.add(symbol)
                unique.append(symbol)
        return unique

    async def _evaluate_pair(self, pair: MarketPair) -> Optional[Opportunity]:
        # Before any network call. Quoting a token that can never be traded
        # spends rate limit and adds latency to every other pair in the cycle.
        verdict = self._token_policy.check(*self._pair_symbols(pair))
        if verdict.allowed:
            # And can the inventory actually reach this chain? A token existing on
            # a chain is not the same as the exchange settling it there, and a
            # price advantage on a chain we cannot move to is the price of a
            # bridge rather than an edge. Checked per symbol, because the base and
            # the quote can have different network support.
            for symbol in self._pair_symbols(pair):
                chain_verdict = self._token_policy.check_chain(symbol, pair.dex_chain)
                if not chain_verdict.allowed:
                    verdict = chain_verdict
                    break
        if not verdict.allowed:
            logger.warning(f"{pair.cex_symbol} blocked by token policy: {verdict.reason}")
            self._emit(pair, _Evaluation(
                direction="", reason=RejectionReason.TOKEN_DENIED))
            return None

        evaluations = (
            await self._evaluate_synthetic(pair)
            if pair.is_synthetic
            else await self._evaluate_direct(pair)
        )
        return self._decide(pair, evaluations)

    # ------------------------------------------------------------------
    # book handling
    # ------------------------------------------------------------------

    async def _usable_book(
        self, pair: MarketPair
    ) -> Tuple[Optional[BookSnapshot], Optional[str]]:
        """Fetch a book, returning it or the reason it is unusable.

        Detection deliberately has no REST fallback. A single-quote REST call
        would be depth-blind, and at scale it breaches the exchange weight
        budget.
        """
        book = await self.cex_client.get_book(pair)
        if book is None:
            logger.debug(f"No order book available for {pair.cex_symbol}; skipping.")
            return None, RejectionReason.NO_BOOK
        if not book.bids or not book.asks:
            logger.debug(f"Order book for {pair.cex_symbol} is one-sided; skipping.")
            return None, RejectionReason.ONE_SIDED_BOOK

        # Reject on FEED age, not per-symbol age. Binance suppresses unchanged
        # books, so a quiet illiquid pair legitimately goes seconds between
        # frames while its quoted price stays correct -- and illiquid pairs are
        # precisely this strategy's universe. A dead connection, by contrast,
        # invalidates every book at once.
        feed_age = book.feed_age_seconds(clock.now())
        try:
            metrics.feed_age_seconds.set(feed_age)
        except Exception:  # pragma: no cover
            pass
        if feed_age > self.strategy_config.max_book_age_seconds:
            logger.warning(
                f"Market data feed is stale ({feed_age:.2f}s since the last frame, "
                f"limit {self.strategy_config.max_book_age_seconds}s); "
                f"skipping {pair.cex_symbol}."
            )
            return None, RejectionReason.STALE_FEED
        return book, None

    @staticmethod
    def _levels_consumed(levels: List[tuple], fill: BookFill) -> int:
        """How many book levels the fill actually touched.

        Recorded so the realised depth requirement is measurable directly,
        rather than inferred from the VWAP after the fact.
        """
        remaining, used = fill.filled_base, 0
        for _, available in levels:
            if remaining <= ZERO:
                break
            used += 1
            remaining -= available
        return used

    # ------------------------------------------------------------------
    # direct pairs
    # ------------------------------------------------------------------

    async def _quoting_pair(
        self, pair: MarketPair, side: str, size: Decimal
    ) -> MarketPair:
        """The pair as it should be quoted, with the best fee tier substituted.

        Returns the pair unchanged when routing is disabled or the selector has
        nothing better to offer, so the configured tier remains the fallback in
        every failure path.
        """
        if self._pool_selector is None:
            return pair
        fee = await self._pool_selector.best_fee(pair, side=side, size=size)
        if fee == pair.dex_pool_fee:
            return pair
        return pair.model_copy(update={"dex_pool_fee": fee})

    async def _both_directions(self, pair: MarketPair, *coros) -> List[_Evaluation]:
        """Evaluate both directions concurrently, isolating each one's failures.

        Two defects in one line. The old form was:

            return [await self._eval_cex_to_dex(...), await self._eval_dex_to_cex(...)]

        Those awaits are SEQUENTIAL, so with a measured 0.31-0.83s median RPC
        latency plus a gas call each, the two directions sampled the pool up to a
        second apart -- and `_decide` then picked "the better direction" from two
        observations that were never simultaneous. On a moving pool that comparison
        is partly a coin flip on which direction was measured first. The CEX side
        was already consistent: one book snapshot is taken and shared.

        And a raising direction took the whole pair down: the exception propagated
        to `detect`, which recorded a single pair-level ERROR and discarded the
        other direction's perfectly good evaluation.

        `return_exceptions=True` fixes the second while gather fixes the first. A
        failed direction becomes its own ERROR record, so the dataset distinguishes
        "this direction failed" from "this pair failed".
        """
        results = await asyncio.gather(*coros, return_exceptions=True)
        evaluations: List[_Evaluation] = []
        for direction, result in zip(("CEX_to_DEX", "DEX_to_CEX"), results):
            if isinstance(result, BaseException):
                logger.warning(
                    f"{pair.cex_symbol} {direction} failed: "
                    f"{type(result).__name__}: {result}"
                )
                evaluations.append(
                    _Evaluation(direction=direction, reason=RejectionReason.ERROR)
                )
            else:
                evaluations.append(result)
        return evaluations

    async def _evaluate_direct(self, pair: MarketPair) -> List[_Evaluation]:
        book, reason = await self._usable_book(pair)
        if book is None:
            return [_Evaluation(direction="", reason=reason)]

        notional = Decimal(str(self.strategy_config.target_notional_usd))
        return await self._both_directions(
            pair,
            self._eval_cex_to_dex(pair, book, notional, cex_legs=1),
            self._eval_dex_to_cex(pair, book, notional, cex_legs=1),
        )

    async def _eval_cex_to_dex(
        self,
        pair: MarketPair,
        book: BookSnapshot,
        notional: Decimal,
        cex_legs: int,
        dex_price_scale: Decimal = Decimal("1"),
    ) -> _Evaluation:
        """Buy base on the CEX, sell it on the DEX."""
        ev = _Evaluation(
            direction="CEX_to_DEX",
            cex_best_bid=book.best_bid,
            cex_best_ask=book.best_ask,
            book_age_s=book.age_seconds(clock.now()),
        )

        # The DEX leg sells base, so ask the selector which pool sells best at
        # this size. Falls back to the configured tier on any failure.
        quoting_pair = await self._quoting_pair(
            pair, side="sell", size=notional / book.best_ask
        )
        ev.dex_pool_fee_used = quoting_pair.dex_pool_fee

        # Size from top of book, then price that size by walking the ladder.
        # The resulting VWAP is never better than best_ask.
        size_base = notional / book.best_ask
        fill = walk_book(book.asks, size_base, ascending=True)
        ev.depth_levels_used = self._levels_consumed(book.asks, fill)
        if not fill.complete or fill.vwap is None:
            logger.debug(
                f"{pair.cex_symbol}: insufficient CEX ask depth for {size_base} base "
                f"(filled {fill.filled_base})."
            )
            ev.reason = RejectionReason.INSUFFICIENT_DEPTH
            return ev

        try:
            quote = await self.dex_client.get_quote(
                quoting_pair, size=fill.filled_base, side="sell", estimate_gas=True
            )
        except RpcError as exc:
            logger.warning(f"{pair.cex_symbol} CEX_to_DEX: {exc}")
            ev.reason = RejectionReason.RPC_ERROR
            return ev
        if quote is None or quote.price <= ZERO:
            ev.reason = RejectionReason.NO_DEX_QUOTE
            return ev

        ev.econ = self._economics(
            pair, direction="CEX_to_DEX", size_base=fill.filled_base,
            cex_price=fill.vwap, dex_price=quote.price * dex_price_scale,
            gas_quote=quote.gas_cost_quote, cex_legs=cex_legs,
        )
        if ev.econ is None:
            ev.reason = RejectionReason.BAD_INPUTS

        ev.placebo_net_bps = self._placebo_net_bps(
            pair, "sell", quote.price, dex_price_scale,
            direction="CEX_to_DEX", size_base=fill.filled_base,
            cex_price=fill.vwap, gas_quote=quote.gas_cost_quote,
            cex_legs=cex_legs,
        )
        return ev

    async def _eval_dex_to_cex(
        self,
        pair: MarketPair,
        book: BookSnapshot,
        notional: Decimal,
        cex_legs: int,
        dex_price_scale: Decimal = Decimal("1"),
        dex_spend: Optional[Decimal] = None,
    ) -> _Evaluation:
        """Buy base on the DEX, sell it on the CEX."""
        ev = _Evaluation(
            direction="DEX_to_CEX",
            cex_best_bid=book.best_bid,
            cex_best_ask=book.best_ask,
            book_age_s=book.age_seconds(clock.now()),
        )

        # The DEX leg buys base with quote units, so the better pool is the one
        # that charges less per base.
        quoting_pair = await self._quoting_pair(
            pair, side="buy", size=notional if dex_spend is None else dex_spend
        )
        ev.dex_pool_fee_used = quoting_pair.dex_pool_fee

        # `side="buy"` consumes an amount of the DEX quote token, not a base
        # amount. For a direct pair that is the target notional; for a
        # synthetic pair the caller converts it into the intermediate asset.
        spend = notional if dex_spend is None else dex_spend
        try:
            quote = await self.dex_client.get_quote(
                quoting_pair, size=spend, side="buy", estimate_gas=True
            )
        except RpcError as exc:
            logger.warning(f"{pair.cex_symbol} DEX_to_CEX: {exc}")
            ev.reason = RejectionReason.RPC_ERROR
            return ev
        if quote is None or quote.price <= ZERO:
            ev.reason = RejectionReason.NO_DEX_QUOTE
            return ev

        dex_price = quote.price * dex_price_scale
        base_out = notional / dex_price
        fill = walk_book(book.bids, base_out, ascending=False)
        ev.depth_levels_used = self._levels_consumed(book.bids, fill)
        if not fill.complete or fill.vwap is None:
            logger.debug(
                f"{pair.cex_symbol}: insufficient CEX bid depth for {base_out} base "
                f"(filled {fill.filled_base})."
            )
            ev.reason = RejectionReason.INSUFFICIENT_DEPTH
            return ev

        ev.econ = self._economics(
            pair, direction="DEX_to_CEX", size_base=base_out,
            cex_price=fill.vwap, dex_price=dex_price,
            gas_quote=quote.gas_cost_quote, cex_legs=cex_legs,
        )
        if ev.econ is None:
            ev.reason = RejectionReason.BAD_INPUTS

        ev.placebo_net_bps = self._placebo_net_bps(
            pair, "buy", quote.price, dex_price_scale,
            direction="DEX_to_CEX", size_base=base_out,
            cex_price=fill.vwap, gas_quote=quote.gas_cost_quote,
            cex_legs=cex_legs,
        )
        return ev

    # ------------------------------------------------------------------
    # synthetic pairs
    # ------------------------------------------------------------------

    async def _intermediate_prices(
        self, pair: MarketPair, intermediate_symbol: str
    ) -> Optional[Tuple[Decimal, Decimal]]:
        """Best (ask, bid) for the intermediate asset against the CEX quote.

        The intermediate pair need not be a configured trading pair, but its
        book must be subscribed on the CEX client for this to resolve.
        """
        symbol = f"{intermediate_symbol}/{pair.quote_cex}"
        now = clock.now()
        cached = self._intermediate_price_cache.get(symbol)
        if cached is not None:
            prices, stamped = cached
            if now - stamped < self.strategy_config.intermediate_price_cache_seconds:
                return prices

        lookup = MarketPair(
            base=intermediate_symbol, quote_cex=pair.quote_cex,
            quote_dex=pair.quote_cex, cex_symbol=symbol,
            dex_chain=pair.dex_chain, dex_pool_fee=pair.dex_pool_fee,
        )
        book, _ = await self._usable_book(lookup)
        if book is None:
            logger.debug(
                f"Cannot price synthetic {pair.cex_symbol}: no book for {symbol}."
            )
            return None

        prices = (book.best_ask, book.best_bid)
        self._intermediate_price_cache[symbol] = (prices, now)
        return prices

    async def _evaluate_synthetic(self, pair: MarketPair) -> List[_Evaluation]:
        """Price a pair whose DEX pool quotes in an intermediate asset.

        The round trip is CEX leg, one on-chain swap, CEX leg -- two CEX orders
        and one gas payment, hence cex_legs=2 while gas is charged once inside
        evaluate_trade.
        """
        intermediate_symbol = pair.intermediate_symbol
        if not intermediate_symbol:
            logger.warning(
                f"{pair.cex_symbol} is marked synthetic but has no "
                f"intermediate_symbol; skipping."
            )
            return [_Evaluation(direction="", reason=RejectionReason.BAD_INPUTS)]

        book, reason = await self._usable_book(pair)
        if book is None:
            return [_Evaluation(direction="", reason=reason)]

        prices = await self._intermediate_prices(pair, intermediate_symbol)
        if prices is None or prices[0] <= ZERO or prices[1] <= ZERO:
            return [
                _Evaluation(direction="", reason=RejectionReason.NO_INTERMEDIATE_PRICE)
            ]
        intermediate_ask, intermediate_bid = prices

        notional = Decimal(str(self.strategy_config.target_notional_usd))
        return await self._both_directions(
            pair,
            # Selling base on the DEX yields the intermediate asset, which is
            # then sold on the CEX -- convert at the intermediate *bid*.
            self._eval_cex_to_dex(
                pair, book, notional, cex_legs=2, dex_price_scale=intermediate_bid
            ),
            # Buying base on the DEX spends the intermediate asset, which must
            # first be bought on the CEX -- convert at the intermediate *ask*.
            self._eval_dex_to_cex(
                pair, book, notional, cex_legs=2,
                dex_price_scale=intermediate_ask,
                dex_spend=notional / intermediate_ask,
            ),
        )

    # ------------------------------------------------------------------
    # decision + recording
    # ------------------------------------------------------------------

    def _economics(
        self, pair: MarketPair, *, direction: str, size_base: Decimal,
        cex_price: Decimal, dex_price: Decimal, gas_quote: Decimal, cex_legs: int,
    ) -> Optional[TradeEconomics]:
        try:
            return evaluate_trade(
                direction=direction, size_base=size_base, cex_price=cex_price,
                dex_price=dex_price,
                taker_fee_bps=self.strategy_config.taker_fee_bps,
                gas_quote=gas_quote, cex_legs=cex_legs,
                rotation_cost_quote=self._rotation_cost_quote,
            )
        except ValueError as exc:
            logger.debug(f"{pair.cex_symbol} {direction}: rejected inputs: {exc}")
            return None


    def _placebo_net_bps(
        self, pair: MarketPair, side: str, live_dex_price: Decimal,
        dex_price_scale: Decimal, *, direction: str, size_base: Decimal,
        cex_price: Decimal, gas_quote: Decimal, cex_legs: int,
    ) -> Optional[Decimal]:
        """Net edge this same book would show against a stale DEX quote.

        The live CEX side is held fixed and only the DEX quote is aged, which
        isolates the question: would a deliberately worse view of the DEX have
        produced the same edge? If so, the edge is a property of the delay
        rather than of the market.

        Returns None until the buffer is warm. Substituting the live quote in
        the meantime would make the control silently agree with the live arm.

        Two deliberate simplifications, stated so the numbers are not read as
        more than they are:

        * Size is held at the live size rather than re-derived from the stale
          price. Re-deriving would change the CEX depth consumed too, mixing a
          size effect into what is meant to isolate a price effect.
        * The delay is wall-clock and must exceed the slowest quoted chain's
          block time, which the config validator enforces. A DEX quote changes
          only when a block lands, so a sub-block delay compares a quote to
          itself: measured live, a one-second delay against Ethereum produced
          live and placebo values identical in 69% of paired observations.
        """
        if self._placebo is None:
            return None

        # Push before reading. The buffer's contract is "the quote from
        # delay_cycles pushes ago", so on cycle N the live quote must already be
        # in the series for the read to land on cycle N - delay_cycles. Reading
        # first would silently return the quote from delay_cycles + 1 ago.
        self._placebo.push(pair.cex_symbol, side, live_dex_price)
        stale = self._placebo.delayed(pair.cex_symbol, side)
        if stale is None:
            return None

        econ = self._economics(
            pair, direction=direction, size_base=size_base,
            cex_price=cex_price, dex_price=stale * dex_price_scale,
            gas_quote=gas_quote, cex_legs=cex_legs,
        )
        return econ.net_bps if econ is not None else None

    def _floor_for(self, pair: MarketPair) -> Decimal:
        """The edge this pair must clear: its base floor plus the risk charge.

        The risk charge is here rather than inside the economics on purpose. Unhedged
        inventory in transit is variance, not a negative mean, so subtracting it from
        the measured edge -- which is what the old rotation model did -- made every
        recorded net_bps not the expected value of anything, and silently rewrote
        history's edges whenever the risk assumption changed.

        Both numbers are recorded: `min_net_bps` on each row is this combined floor,
        and `net_bps` is the honest expected edge. A later reader can re-decide the
        same rows under a different risk policy without re-measuring anything.
        """
        base = (
            Decimal(pair.min_net_bps)
            if pair.min_net_bps is not None
            else self.strategy_config.min_net_bps
        )
        return required_net_bps(
            base_floor_bps=base, risk_charge_bps=self._rotation_risk_bps
        )

    def _decide(
        self, pair: MarketPair, evaluations: List[_Evaluation]
    ) -> Optional[Opportunity]:
        """Classify every evaluation, record all of them, return the best."""
        floor = self._floor_for(pair)
        ceiling = self.strategy_config.max_net_bps_sanity

        viable: List[_Evaluation] = []
        for ev in evaluations:
            if ev.econ is None:
                continue
            if ev.econ.net_bps > ceiling:
                logger.warning(
                    f"{pair.cex_symbol} {ev.direction}: net {ev.econ.net_bps:.1f} bps "
                    f"exceeds the sanity ceiling of {ceiling} bps. This usually means "
                    f"a decimals or units error, not an opportunity. Skipping."
                )
                ev.reason = RejectionReason.ABOVE_SANITY
            elif ev.econ.net_bps < floor:
                ev.reason = RejectionReason.BELOW_FLOOR
            else:
                viable.append(ev)

        best = max(viable, key=lambda e: e.econ.net_quote) if viable else None
        for ev in viable:
            if ev is not best:
                ev.reason = RejectionReason.NOT_BEST_DIRECTION

        for ev in evaluations:
            self._emit(pair, ev, floor=floor, taken=(ev is best))

        if best is None:
            return None

        opportunities_found.labels(pair=pair.cex_symbol, direction=best.direction).inc()
        return self._to_opportunity(pair, best.econ, best.dex_pool_fee_used)

    def _emit(
        self, pair: MarketPair, ev: _Evaluation,
        floor: Optional[Decimal] = None, taken: bool = False,
    ) -> None:
        """Persist one evaluation. Never allowed to disrupt trading.

        An audit trail that can halt the strategy is worse than one that
        occasionally loses a row, so every failure here is logged and
        swallowed.
        """
        econ = ev.econ

        # Recomputed here rather than threaded down from _evaluate_pair. It is a
        # handful of dict lookups, and deriving it at the point of recording
        # makes it impossible for the stored label to drift from the pair the row
        # describes.
        try:
            policy_verdict = self._token_policy.classify(*self._pair_symbols(pair))
        except Exception as exc:  # pragma: no cover - never fatal
            logger.debug(f"Could not classify {pair.cex_symbol}: {exc}")
            policy_verdict = None

        # Counted regardless of whether a store is configured. This is the
        # series that distinguishes a quiet market from a wedged loop: the
        # most common decision -- rejected below the floor -- previously
        # produced no metric and no log at the configured level, so a cycle
        # that evaluated everything and rejected everything was
        # indistinguishable from a cycle that did nothing.
        try:
            metrics.evaluations_total.labels(
                pair=pair.cex_symbol,
                direction=ev.direction or "none",
                outcome="taken" if taken else "rejected",
                reason="none" if taken else (ev.reason or "unknown"),
            ).inc()
            if ev.book_age_s is not None:
                metrics.book_age_seconds.labels(pair=pair.cex_symbol).set(ev.book_age_s)
        except Exception as exc:  # pragma: no cover - telemetry is never fatal
            logger.debug(f"Failed to emit evaluation metrics: {exc}")

        if self.store is None:
            return
        try:
            self.store.record(EvaluationRecord(
                ts=clock.now(),
                cex_symbol=pair.cex_symbol,
                base=pair.base,
                quote_cex=pair.quote_cex,
                dex_chain=pair.dex_chain,
                # The tier actually quoted, which differs from the configured
                # one whenever routing found a better pool. A row naming a
                # pool the price did not come from cannot be re-derived.
                dex_pool_fee=(
                    ev.dex_pool_fee_used
                    if ev.dex_pool_fee_used is not None else pair.dex_pool_fee
                ),
                is_synthetic=pair.is_synthetic,
                outcome="taken" if taken else "rejected",
                direction=ev.direction or None,
                reason=None if taken else ev.reason,
                size_base=econ.size_base if econ else None,
                notional_quote=econ.notional_quote if econ else None,
                cex_price=econ.cex_price if econ else None,
                cex_best_bid=ev.cex_best_bid,
                cex_best_ask=ev.cex_best_ask,
                dex_price=econ.dex_price if econ else None,
                gross_quote=econ.gross_quote if econ else None,
                cex_fee_quote=econ.cex_fee_quote if econ else None,
                gas_quote=econ.gas_quote if econ else None,
                rotation_cost_quote=econ.rotation_cost_quote if econ else None,
                net_quote=econ.net_quote if econ else None,
                net_bps=econ.net_bps if econ else None,
                placebo_net_bps=ev.placebo_net_bps,
                policy_verdict=policy_verdict,
                cex_legs=econ.cex_legs if econ else None,
                book_age_s=ev.book_age_s,
                depth_levels_used=ev.depth_levels_used,
                min_net_bps=floor if floor is not None else self._floor_for(pair),
                taker_fee_bps=self.strategy_config.taker_fee_bps,
            ))
        except Exception as exc:
            logger.error(f"Failed to record evaluation for {pair.cex_symbol}: {exc}")

    def _to_opportunity(
        self,
        pair: MarketPair,
        econ: TradeEconomics,
        dex_pool_fee_used: Optional[int] = None,
    ) -> Opportunity:
        # The pair carried on the opportunity must name the pool the price came
        # from. The executor builds the swap from this object, so a mismatch would
        # send the trade to a different pool than the one that was quoted -- and
        # with routing enabled the configured tier is frequently not the one used.
        if dex_pool_fee_used is not None and dex_pool_fee_used != pair.dex_pool_fee:
            pair = pair.model_copy(update={"dex_pool_fee": dex_pool_fee_used})

        return Opportunity(
            pair=pair,
            direction=econ.direction,
            size=econ.size_base,
            cex_price=econ.cex_price,
            dex_price=econ.dex_price,
            dex_chain=pair.dex_chain,
            dex_pool_fee=pair.dex_pool_fee,
            edge_bps=econ.net_bps,
            slippage_bps=Decimal(pair.max_slippage_bps or 0),
            gas_cost_quote=econ.gas_quote,
            cex_fee_quote=econ.cex_fee_quote,
            expected_pnl_quote=econ.net_quote,
            valid_until=clock.now() + self.strategy_config.opportunity_ttl_seconds,
        )
