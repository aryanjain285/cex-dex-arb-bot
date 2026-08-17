"""Native-token USD price, with an explicit staleness contract.

Gas is paid in a chain's native token but enters the trade economics in the
quote currency, so a conversion rate is required on every gas-inclusive quote.

This module fails closed. If the rate cannot be established it returns None
and the caller must decline to quote, rather than substituting a guess. An
earlier implementation returned a hardcoded price on failure, which silently
scaled every gas cost and therefore silently shifted every profitability
decision -- with no error raised and nothing in the logs.

Three time windows govern behaviour:

- while the last successful fetch is *strictly* younger than `ttl_seconds`, the
  cached price is served without a network call;
- at or beyond the TTL, a refresh is attempted;
- if that refresh fails, the previous price is served only while it is strictly
  inside `stale_grace_seconds`, and refused at or beyond it.

Both boundaries are exclusive on purpose. With inclusive comparisons a TTL of
zero still served from cache, and a grace window of zero still served a stale
price, whenever two calls happened to land in the same clock tick -- so the two
settings that mean "never cache" and "never serve stale" did neither reliably.
That failure was invisible on a slow clock and appeared on a fast one, which is
the worst way for a staleness contract to break.
"""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Awaitable, Callable, Dict, Optional

import aiohttp
from loguru import logger

from ..core import clock

__all__ = ["NativePriceOracle", "coingecko_fetcher"]

ZERO = Decimal("0")

# Chain -> CoinGecko asset id for that chain's native token.
COINGECKO_IDS: Dict[str, str] = {
    "ethereum": "ethereum",
    "arbitrum": "ethereum",
    "base": "ethereum",
    "bsc": "binancecoin",
}

Fetcher = Callable[[str], Awaitable[Optional[Decimal]]]


@dataclass
class _Entry:
    price: Decimal
    fetched_at: float


async def coingecko_fetcher(chain: str) -> Optional[Decimal]:
    """Fetch a native-token USD price from CoinGecko.

    Returns None rather than raising for an unmapped chain, so an unsupported
    chain degrades to "cannot price gas" instead of an unhandled exception.
    """
    asset_id = COINGECKO_IDS.get(chain)
    if asset_id is None:
        logger.warning(f"No native-token price mapping for chain {chain!r}.")
        return None

    url = (
        "https://api.coingecko.com/api/v3/simple/price"
        f"?ids={asset_id}&vs_currencies=usd"
    )
    timeout = aiohttp.ClientTimeout(total=8)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.get(url) as response:
            response.raise_for_status()
            payload = await response.json()
    raw = payload.get(asset_id, {}).get("usd")
    return None if raw is None else Decimal(str(raw))


class NativePriceOracle:
    def __init__(
        self,
        ttl_seconds: float,
        fetcher: Fetcher = coingecko_fetcher,
        stale_grace_seconds: float = 0.0,
        now_fn: Callable[[], float] = clock.now,
    ):
        if ttl_seconds < 0:
            raise ValueError("ttl_seconds must be non-negative")
        if stale_grace_seconds < 0:
            raise ValueError("stale_grace_seconds must be non-negative")
        self.ttl_seconds = ttl_seconds
        self.stale_grace_seconds = stale_grace_seconds
        self._fetcher = fetcher
        # Injectable so the window boundaries can be tested exactly rather than
        # inferred from however fast the wall clock happens to tick. Defaults to
        # the single process clock; nothing in production passes anything else.
        self._now = now_fn
        self._cache: Dict[str, _Entry] = {}

    async def get_usd_price(self, chain: str) -> Optional[Decimal]:
        """Return the native-token USD price, or None if it cannot be trusted."""
        now = self._now()
        cached = self._cache.get(chain)

        if cached is not None and now - cached.fetched_at < self.ttl_seconds:
            return cached.price

        try:
            price = await self._fetcher(chain)
        except Exception as exc:
            logger.warning(f"Native price fetch failed for {chain}: {exc}")
            price = None

        if price is not None and price > ZERO:
            self._cache[chain] = _Entry(price=price, fetched_at=now)
            return price

        # Refresh failed. Serve the previous value only inside the grace window.
        if cached is not None:
            age = now - cached.fetched_at
            if age < self.stale_grace_seconds:
                logger.warning(
                    f"Serving a stale native price for {chain} "
                    f"({age:.1f}s old, grace {self.stale_grace_seconds}s)."
                )
                return cached.price
            logger.error(
                f"Native price for {chain} is {age:.1f}s old, beyond the "
                f"{self.stale_grace_seconds}s grace window. Refusing to price gas."
            )
        else:
            logger.error(
                f"No native price available for {chain} and nothing cached. "
                f"Refusing to price gas."
            )
        return None
