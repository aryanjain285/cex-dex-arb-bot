"""The native-token price oracle must never invent a number.

Gas costs are denominated in the chain's native token and must be converted to
the quote currency to enter the economics. The previous implementation fetched
this per quote with no cache and, on any failure, silently returned a
hardcoded Decimal("3000") for ETH.

That is the most dangerous line in a trading system: a wrong native price
silently scales every gas cost, which silently shifts every profitability
decision, with no error and no log. For real capital the only acceptable
behaviour is to fail closed -- report that the price is unavailable and let
the caller decline to trade.
"""
from decimal import Decimal

import pytest

from src.exchange.price_oracle import NativePriceOracle


def D(x) -> Decimal:
    return Decimal(str(x))


class Fetcher:
    """Records calls and returns queued results (value or Exception)."""

    def __init__(self, *results):
        self.results = list(results)
        self.calls = 0

    async def __call__(self, chain: str):
        self.calls += 1
        result = self.results.pop(0) if self.results else None
        if isinstance(result, Exception):
            raise result
        return result


async def test_returns_the_fetched_price():
    oracle = NativePriceOracle(ttl_seconds=60, fetcher=Fetcher(D(2500)))
    assert await oracle.get_usd_price("ethereum") == D(2500)


async def test_caches_within_ttl_instead_of_refetching_per_quote():
    fetcher = Fetcher(D(2500), D(9999))
    oracle = NativePriceOracle(ttl_seconds=60, fetcher=fetcher)

    first = await oracle.get_usd_price("ethereum")
    second = await oracle.get_usd_price("ethereum")

    assert first == second == D(2500)
    assert fetcher.calls == 1, "a cached price must not trigger a second fetch"


async def test_refetches_after_ttl_expires():
    fetcher = Fetcher(D(2500), D(2600))
    oracle = NativePriceOracle(ttl_seconds=0, fetcher=fetcher)

    assert await oracle.get_usd_price("ethereum") == D(2500)
    assert await oracle.get_usd_price("ethereum") == D(2600)
    assert fetcher.calls == 2


async def test_returns_none_when_the_fetch_fails_and_nothing_is_cached():
    """Fail closed. No invented number, ever."""
    oracle = NativePriceOracle(
        ttl_seconds=60, fetcher=Fetcher(RuntimeError("coingecko down"))
    )
    assert await oracle.get_usd_price("ethereum") is None


async def test_returns_none_when_the_fetch_returns_a_non_positive_price():
    oracle = NativePriceOracle(ttl_seconds=60, fetcher=Fetcher(D(0)))
    assert await oracle.get_usd_price("ethereum") is None


async def test_serves_a_stale_price_only_within_the_grace_window():
    """A brief outage may reuse the last good price, but only briefly, and the
    window is explicit rather than unbounded."""
    fetcher = Fetcher(D(2500), RuntimeError("blip"), RuntimeError("blip"))
    oracle = NativePriceOracle(ttl_seconds=0, fetcher=fetcher, stale_grace_seconds=60)

    assert await oracle.get_usd_price("ethereum") == D(2500)
    # fetch fails, but the previous value is inside the grace window
    assert await oracle.get_usd_price("ethereum") == D(2500)

    strict = NativePriceOracle(
        ttl_seconds=0, fetcher=Fetcher(D(2500), RuntimeError("blip")),
        stale_grace_seconds=0,
    )
    assert await strict.get_usd_price("ethereum") == D(2500)
    assert await strict.get_usd_price("ethereum") is None, (
        "with no grace window a failed refresh must not serve a stale price"
    )


async def test_prices_are_tracked_per_chain():
    class PerChain:
        calls = 0
        async def __call__(self, chain):
            PerChain.calls += 1
            return {"ethereum": D(2500), "bsc": D(600)}[chain]

    oracle = NativePriceOracle(ttl_seconds=60, fetcher=PerChain())
    assert await oracle.get_usd_price("ethereum") == D(2500)
    assert await oracle.get_usd_price("bsc") == D(600)


async def test_no_hardcoded_fallback_constant_exists_in_the_module():
    """Guards against the fallback being reintroduced.

    The original defect was a literal Decimal("3000") returned on failure.
    """
    import inspect

    from src.exchange import price_oracle

    source = inspect.getsource(price_oracle)
    assert "3000" not in source, "a hardcoded price fallback must not exist"
