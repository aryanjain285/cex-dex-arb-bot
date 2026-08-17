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
from src.strategy.costs import (
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
        logger.info(
            f"Opportunity detector initialised, monitoring {len(pairs)} pairs "
            f"({'recording' if store else 'NOT recording'} evaluations)."
        )
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

    async def _evaluate_pair(self, pair: MarketPair) -> Optional[Opportunity]:
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

    async def _evaluate_direct(self, pair: MarketPair) -> List[_Evaluation]:
        book, reason = await self._usable_book(pair)
        if book is None:
            return [_Evaluation(direction="", reason=reason)]

        notional = Decimal(str(self.strategy_config.target_notional_usd))
        return [
            await self._eval_cex_to_dex(pair, book, notional, cex_legs=1),
            await self._eval_dex_to_cex(pair, book, notional, cex_legs=1),
        ]

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

        quote = await self.dex_client.get_quote(
            pair, size=fill.filled_base, side="sell", estimate_gas=True
        )
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

        # `side="buy"` consumes an amount of the DEX quote token, not a base
        # amount. For a direct pair that is the target notional; for a
        # synthetic pair the caller converts it into the intermediate asset.
        spend = notional if dex_spend is None else dex_spend
        quote = await self.dex_client.get_quote(
            pair, size=spend, side="buy", estimate_gas=True
        )
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
        return [
            # Selling base on the DEX yields the intermediate asset, which is
            # then sold on the CEX -- convert at the intermediate *bid*.
            await self._eval_cex_to_dex(
                pair, book, notional, cex_legs=2, dex_price_scale=intermediate_bid
            ),
            # Buying base on the DEX spends the intermediate asset, which must
            # first be bought on the CEX -- convert at the intermediate *ask*.
            await self._eval_dex_to_cex(
                pair, book, notional, cex_legs=2,
                dex_price_scale=intermediate_ask,
                dex_spend=notional / intermediate_ask,
            ),
        ]

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

    def _floor_for(self, pair: MarketPair) -> Decimal:
        return (
            Decimal(pair.min_net_bps)
            if pair.min_net_bps is not None
            else self.strategy_config.min_net_bps
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
        return self._to_opportunity(pair, best.econ)

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
                dex_pool_fee=pair.dex_pool_fee,
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
                cex_legs=econ.cex_legs if econ else None,
                book_age_s=ev.book_age_s,
                depth_levels_used=ev.depth_levels_used,
                min_net_bps=floor if floor is not None else self._floor_for(pair),
                taker_fee_bps=self.strategy_config.taker_fee_bps,
            ))
        except Exception as exc:
            logger.error(f"Failed to record evaluation for {pair.cex_symbol}: {exc}")

    def _to_opportunity(self, pair: MarketPair, econ: TradeEconomics) -> Opportunity:
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
