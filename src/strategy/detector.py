"""Opportunity detection.

Decides on *net* economics computed from *depth-weighted* prices. Costs are
summed in exactly one place -- `costs.evaluate_trade` -- so the same quantity
cannot be charged twice.

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
from decimal import Decimal
from typing import Dict, List, Optional, Tuple

from loguru import logger

from src.core import clock
from src.core.config import StrategyConfig
from src.core.types import BookSnapshot, MarketPair, Opportunity
from src.exchange.cex_base import CexClient
from src.exchange.dex_base import DexClient
from src.infra.metrics import opportunities_found
from src.strategy.costs import BookFill, TradeEconomics, evaluate_trade, walk_book

ZERO = Decimal("0")
TEN_THOUSAND = Decimal("10000")


class OpportunityDetector:
    def __init__(
        self,
        strategy_config: StrategyConfig,
        cex_client: CexClient,
        dex_client: DexClient,
        pairs: List[MarketPair],
    ):
        self.strategy_config = strategy_config
        self.cex_client = cex_client
        self.dex_client = dex_client
        self.pairs = pairs
        self._intermediate_price_cache: Dict[str, Tuple[Tuple[Decimal, Decimal], float]] = {}
        logger.info(f"Opportunity detector initialised, monitoring {len(pairs)} pairs.")

    # ------------------------------------------------------------------
    # entry point
    # ------------------------------------------------------------------

    async def detect(self) -> List[Opportunity]:
        """Evaluate every pair concurrently.

        Failures are isolated per pair: `return_exceptions=True` is essential,
        because without it a single raising pair aborts the whole gather and
        the caller detects nothing at all for that cycle.
        """
        results = await asyncio.gather(
            *(self.check_pair(p) for p in self.pairs), return_exceptions=True
        )

        opportunities: List[Opportunity] = []
        for pair, result in zip(self.pairs, results):
            if isinstance(result, BaseException):
                logger.warning(
                    f"Pair {pair.cex_symbol} failed evaluation: "
                    f"{type(result).__name__}: {result}"
                )
                continue
            if result is not None:
                opportunities.append(result)
        return opportunities

    async def check_pair(self, pair: MarketPair) -> Optional[Opportunity]:
        if pair.is_synthetic:
            return await self._check_synthetic_pair(pair)
        return await self._check_direct_pair(pair)

    # ------------------------------------------------------------------
    # book handling
    # ------------------------------------------------------------------

    async def _usable_book(self, pair: MarketPair) -> Optional[BookSnapshot]:
        """Fetch a book and reject it unless it is present, two-sided, fresh.

        Detection deliberately has no REST fallback. A single-quote REST call
        would be depth-blind, and at scale it breaches the exchange weight
        budget: 50 pairs on a 200ms loop costs roughly 5x the 6000/min limit.
        """
        book = await self.cex_client.get_book(pair)
        if book is None:
            logger.debug(f"No order book available for {pair.cex_symbol}; skipping.")
            return None
        if not book.bids or not book.asks:
            logger.debug(f"Order book for {pair.cex_symbol} is one-sided; skipping.")
            return None

        # Reject on FEED age, not per-symbol age. Binance suppresses
        # unchanged books, so a quiet illiquid pair legitimately goes seconds
        # between frames while its quoted price stays correct -- and illiquid
        # pairs are precisely this strategy's universe. A dead connection, by
        # contrast, invalidates every book at once.
        now = clock.now()
        feed_age = book.feed_age_seconds(now)
        if feed_age > self.strategy_config.max_book_age_seconds:
            logger.warning(
                f"Market data feed is stale ({feed_age:.2f}s since the last frame, "
                f"limit {self.strategy_config.max_book_age_seconds}s); "
                f"skipping {pair.cex_symbol}."
            )
            return None
        return book

    def _fill_or_none(
        self, levels: List[tuple], size_base: Decimal, ascending: bool, pair: MarketPair
    ) -> Optional[BookFill]:
        """Walk a book side, requiring a complete fill.

        An incomplete fill means the venue cannot supply the trade. Using the
        partial VWAP as though the whole size were achievable is the specific
        error this rejection exists to prevent.
        """
        fill = walk_book(levels, size_base, ascending=ascending)
        if not fill.complete or fill.vwap is None:
            logger.debug(
                f"{pair.cex_symbol}: insufficient CEX depth for {size_base} base "
                f"(filled {fill.filled_base}); skipping direction."
            )
            return None
        return fill

    # ------------------------------------------------------------------
    # direct pairs
    # ------------------------------------------------------------------

    async def _check_direct_pair(self, pair: MarketPair) -> Optional[Opportunity]:
        book = await self._usable_book(pair)
        if book is None:
            return None

        notional = Decimal(str(self.strategy_config.target_notional_usd))

        candidates: List[TradeEconomics] = []

        cex_to_dex = await self._evaluate_cex_to_dex(pair, book, notional, cex_legs=1)
        if cex_to_dex is not None:
            candidates.append(cex_to_dex)

        dex_to_cex = await self._evaluate_dex_to_cex(pair, book, notional, cex_legs=1)
        if dex_to_cex is not None:
            candidates.append(dex_to_cex)

        return self._best_opportunity(pair, candidates)

    async def _evaluate_cex_to_dex(
        self,
        pair: MarketPair,
        book: BookSnapshot,
        notional: Decimal,
        cex_legs: int,
        dex_price_scale: Decimal = Decimal("1"),
    ) -> Optional[TradeEconomics]:
        """Buy base on the CEX, sell it on the DEX."""
        # Size from top-of-book, then price that size by walking the ladder.
        # The resulting VWAP is never better than best_ask, so the realised
        # notional is at or slightly above target -- conservative on price.
        size_base = notional / book.best_ask
        fill = self._fill_or_none(book.asks, size_base, ascending=True, pair=pair)
        if fill is None:
            return None

        dex_quote = await self.dex_client.get_quote(
            pair, size=fill.filled_base, side="sell", estimate_gas=True
        )
        if dex_quote is None or dex_quote.price <= ZERO:
            return None

        return self._economics(
            pair,
            direction="CEX_to_DEX",
            size_base=fill.filled_base,
            cex_price=fill.vwap,
            dex_price=dex_quote.price * dex_price_scale,
            gas_quote=dex_quote.gas_cost_quote,
            cex_legs=cex_legs,
        )

    async def _evaluate_dex_to_cex(
        self,
        pair: MarketPair,
        book: BookSnapshot,
        notional: Decimal,
        cex_legs: int,
        dex_price_scale: Decimal = Decimal("1"),
        dex_spend: Optional[Decimal] = None,
    ) -> Optional[TradeEconomics]:
        """Buy base on the DEX, sell it on the CEX."""
        # `side="buy"` consumes an amount of the DEX quote token, not a base
        # amount. For a direct pair that is the target notional; for a
        # synthetic pair the caller converts the notional into the
        # intermediate asset first and passes it as dex_spend.
        spend = notional if dex_spend is None else dex_spend
        dex_quote = await self.dex_client.get_quote(
            pair, size=spend, side="buy", estimate_gas=True
        )
        if dex_quote is None or dex_quote.price <= ZERO:
            return None

        dex_price = dex_quote.price * dex_price_scale
        base_out = notional / dex_price
        fill = self._fill_or_none(book.bids, base_out, ascending=False, pair=pair)
        if fill is None:
            return None

        return self._economics(
            pair,
            direction="DEX_to_CEX",
            size_base=base_out,
            cex_price=fill.vwap,
            dex_price=dex_price,
            gas_quote=dex_quote.gas_cost_quote,
            cex_legs=cex_legs,
        )

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
            base=intermediate_symbol,
            quote_cex=pair.quote_cex,
            quote_dex=pair.quote_cex,
            cex_symbol=symbol,
            dex_chain=pair.dex_chain,
            dex_pool_fee=pair.dex_pool_fee,
        )
        book = await self._usable_book(lookup)
        if book is None:
            logger.debug(
                f"Cannot price synthetic {pair.cex_symbol}: no book for {symbol}."
            )
            return None

        prices = (book.best_ask, book.best_bid)
        self._intermediate_price_cache[symbol] = (prices, now)
        return prices

    async def _check_synthetic_pair(self, pair: MarketPair) -> Optional[Opportunity]:
        """Price a pair whose DEX pool quotes in an intermediate asset.

        The round trip is: CEX leg, one on-chain swap, CEX leg. That is two
        CEX orders and one gas payment -- hence cex_legs=2 while gas is
        charged once, inside evaluate_trade.
        """
        intermediate_symbol = pair.intermediate_symbol
        if not intermediate_symbol:
            logger.warning(
                f"{pair.cex_symbol} is marked synthetic but has no "
                f"intermediate_symbol; skipping."
            )
            return None

        book = await self._usable_book(pair)
        if book is None:
            return None

        prices = await self._intermediate_prices(pair, intermediate_symbol)
        if prices is None:
            return None
        intermediate_ask, intermediate_bid = prices
        if intermediate_ask <= ZERO or intermediate_bid <= ZERO:
            return None

        notional = Decimal(str(self.strategy_config.target_notional_usd))
        candidates: List[TradeEconomics] = []

        # Selling base on the DEX yields the intermediate asset, which is then
        # sold on the CEX -- so convert at the intermediate *bid*.
        cex_to_dex = await self._evaluate_cex_to_dex(
            pair, book, notional, cex_legs=2, dex_price_scale=intermediate_bid
        )
        if cex_to_dex is not None:
            candidates.append(cex_to_dex)

        # Buying base on the DEX spends the intermediate asset, which must
        # first be bought on the CEX -- so convert at the intermediate *ask*.
        dex_to_cex = await self._evaluate_dex_to_cex(
            pair,
            book,
            notional,
            cex_legs=2,
            dex_price_scale=intermediate_ask,
            dex_spend=notional / intermediate_ask,
        )
        if dex_to_cex is not None:
            candidates.append(dex_to_cex)

        return self._best_opportunity(pair, candidates)

    # ------------------------------------------------------------------
    # decision
    # ------------------------------------------------------------------

    def _economics(
        self,
        pair: MarketPair,
        *,
        direction: str,
        size_base: Decimal,
        cex_price: Decimal,
        dex_price: Decimal,
        gas_quote: Decimal,
        cex_legs: int,
    ) -> Optional[TradeEconomics]:
        try:
            return evaluate_trade(
                direction=direction,
                size_base=size_base,
                cex_price=cex_price,
                dex_price=dex_price,
                taker_fee_bps=self.strategy_config.taker_fee_bps,
                gas_quote=gas_quote,
                cex_legs=cex_legs,
            )
        except ValueError as exc:
            logger.debug(f"{pair.cex_symbol} {direction}: rejected inputs: {exc}")
            return None

    def _best_opportunity(
        self, pair: MarketPair, candidates: List[TradeEconomics]
    ) -> Optional[Opportunity]:
        """Pick the most profitable viable direction, if any."""
        floor = (
            Decimal(pair.min_net_bps)
            if pair.min_net_bps is not None
            else self.strategy_config.min_net_bps
        )
        ceiling = self.strategy_config.max_net_bps_sanity

        viable: List[TradeEconomics] = []
        for econ in candidates:
            if econ.net_bps > ceiling:
                logger.warning(
                    f"{pair.cex_symbol} {econ.direction}: net {econ.net_bps:.1f} bps "
                    f"exceeds the sanity ceiling of {ceiling} bps. This usually means "
                    f"a decimals or units error, not an opportunity. Skipping."
                )
                continue
            if econ.net_bps >= floor:
                viable.append(econ)

        if not viable:
            return None

        best = max(viable, key=lambda e: e.net_quote)
        opportunities_found.labels(pair=pair.cex_symbol, direction=best.direction).inc()
        return self._to_opportunity(pair, best)

    def _to_opportunity(self, pair: MarketPair, econ: TradeEconomics) -> Opportunity:
        cex_price = econ.buy_price if econ.direction == "CEX_to_DEX" else econ.sell_price
        dex_price = econ.sell_price if econ.direction == "CEX_to_DEX" else econ.buy_price
        return Opportunity(
            pair=pair,
            direction=econ.direction,
            size=econ.size_base,
            cex_price=cex_price,
            dex_price=dex_price,
            dex_chain=pair.dex_chain,
            dex_pool_fee=pair.dex_pool_fee,
            edge_bps=econ.net_bps,
            slippage_bps=Decimal(pair.max_slippage_bps or 0),
            gas_cost_quote=econ.gas_quote,
            cex_fee_quote=econ.cex_fee_quote,
            expected_pnl_quote=econ.net_quote,
            valid_until=clock.now() + self.strategy_config.opportunity_ttl_seconds,
        )
