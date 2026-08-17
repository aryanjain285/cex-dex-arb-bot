"""Choose which Uniswap v3 fee tier to quote, per pair and per side.

`pairs.yaml` names one `dex_pool_fee` per pair and the detector quoted only that
pool. Uniswap v3 lists the same asset pair at up to four fee tiers with
independent liquidity, so a static choice is a standing bet that one tier is
always best. Measured live at a 1000 notional on 2026-08-17:

    ETH/USDT   tier 500 (configured) 1892.49    tier 100  1893.49    5.3 bps
    ETH/USDC   tier 500 (configured) 1891.05    tier 100  1891.74    3.7 bps

Against a 5 bps net floor the tier choice alone is larger than the edge being
chased, so this is not a refinement.

Two things make it cheap and correct:

QuoterV2's price is already net of the pool fee AND of price impact for the size
quoted. Comparing tiers at the intended size therefore compares executable
prices, and no fee arithmetic belongs here -- the 100 tier is not automatically
better, it is only better when its liquidity is deep enough at this size, which
is exactly what the quote reports.

Selection is refreshed on a TTL, not per cycle. Four tiers on both sides of three
pairs at a 0.2s loop would be 120 RPC calls a second; on a TTL it is 24 calls per
refresh interval. The hot loop then quotes one tier.
"""
from __future__ import annotations

from decimal import Decimal
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from loguru import logger

from ..core import clock
from ..core.types import MarketPair

__all__ = ["PoolSelector", "DEFAULT_FEE_TIERS"]

# The four tiers Uniswap v3 deploys by default. 100 (0.01%) exists mainly for
# stable and correlated pairs, and is where the measured ETH improvement was.
DEFAULT_FEE_TIERS = (100, 500, 3000, 10000)


class PoolSelector:
    def __init__(
        self,
        dex_client,
        candidate_fee_tiers: Sequence[int],
        refresh_seconds: float,
        now_fn: Optional[Callable[[], float]] = None,
    ):
        tiers = [int(f) for f in candidate_fee_tiers]
        if not tiers:
            raise ValueError(
                "candidate_fee_tiers is empty; there is nothing to choose between"
            )
        if refresh_seconds <= 0:
            raise ValueError(
                f"refresh_seconds must be positive, got {refresh_seconds}. Zero "
                f"would re-quote every tier on every detection cycle."
            )
        self.dex_client = dex_client
        self.candidate_fee_tiers = tiers
        self.refresh_seconds = float(refresh_seconds)
        self._now_fn = now_fn
        # (pair, side) -> (chosen fee, when chosen)
        self._chosen: Dict[Tuple[str, str], Tuple[int, float]] = {}

    def _now(self) -> float:
        return clock.now() if self._now_fn is None else self._now_fn()

    # ------------------------------------------------------------------

    async def best_fee(self, pair: MarketPair, side: str, size: Decimal) -> int:
        """The fee tier to quote for this pair and side, from cache or by measuring.

        Falls back to the pair's configured tier on any failure. Selection is an
        optimisation, and an optimisation that can stop the bot is not one: the
        configured tier is the operator's stated intent, and the detector's own
        no-quote handling covers the rest.
        """
        key = (pair.cex_symbol, side)
        cached = self._chosen.get(key)
        if cached is not None and self._now() - cached[1] < self.refresh_seconds:
            return cached[0]

        if len(self.candidate_fee_tiers) == 1:
            # Nothing to compare. Spending a quote to discover that would be pure
            # cost.
            chosen = self.candidate_fee_tiers[0]
            self._chosen[key] = (chosen, self._now())
            return chosen

        try:
            chosen = await self._measure(pair, side, size)
        except Exception as exc:
            logger.warning(
                f"Fee-tier selection failed for {pair.cex_symbol} {side} "
                f"({type(exc).__name__}: {exc}); using the configured tier "
                f"{pair.dex_pool_fee}."
            )
            return pair.dex_pool_fee

        if chosen is None:
            logger.debug(
                f"No tier could be quoted for {pair.cex_symbol} {side}; using the "
                f"configured tier {pair.dex_pool_fee}."
            )
            return pair.dex_pool_fee

        previous = cached[0] if cached else None
        if previous is not None and previous != chosen:
            logger.info(
                f"Fee tier for {pair.cex_symbol} {side} moved from {previous} to "
                f"{chosen}: liquidity has migrated between pools."
            )
        elif previous is None and chosen != pair.dex_pool_fee:
            logger.info(
                f"Fee tier for {pair.cex_symbol} {side}: quoting {chosen} rather "
                f"than the configured {pair.dex_pool_fee}, which prices worse at "
                f"this size."
            )
        self._chosen[key] = (chosen, self._now())
        return chosen

    async def _measure(
        self, pair: MarketPair, side: str, size: Decimal
    ) -> Optional[int]:
        """Quote every candidate tier at this size and return the best.

        Direction matters and is easy to get wrong: selling base for quote wants
        the HIGHEST price, buying base with quote wants the LOWEST. A selector
        that maximised unconditionally would pick the worst pool on the buy side,
        and the mistake would be invisible -- it would look like a thin market.
        """
        best_fee: Optional[int] = None
        best_price: Optional[Decimal] = None

        for fee in self.candidate_fee_tiers:
            pool = await self.dex_client.get_pool_address(
                pair.base, pair.quote_dex, pair.dex_chain, fee
            )
            if not pool:
                continue

            candidate_pair = pair.model_copy(update={"dex_pool_fee": fee})
            quote = await self.dex_client.get_quote(
                candidate_pair, size=size, side=side
            )
            if quote is None or quote.price <= 0:
                continue

            if best_price is None:
                best_fee, best_price = fee, quote.price
            elif side == "sell" and quote.price > best_price:
                best_fee, best_price = fee, quote.price
            elif side == "buy" and quote.price < best_price:
                best_fee, best_price = fee, quote.price

        return best_fee

    # ------------------------------------------------------------------

    def describe(self) -> str:
        """The current selection, for the startup log and for diagnosis.

        A choice the operator cannot see is a choice they cannot check.
        """
        if not self._chosen:
            return "fee-tier selection: nothing measured yet"
        parts = [
            f"{symbol} {side}={fee}"
            for (symbol, side), (fee, _) in sorted(self._chosen.items())
        ]
        return "fee-tier selection: " + ", ".join(parts)

    def selection(self) -> Dict[Tuple[str, str], int]:
        return {key: fee for key, (fee, _) in self._chosen.items()}
