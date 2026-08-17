"""Gas pricing must fail closed.

A quote that requests gas costs but cannot price the native token has no
trustworthy economics. Returning the quote anyway -- with a guessed or zero
gas cost -- lets an unprofitable trade look profitable. The only safe answer
is to decline the quote.
"""
from decimal import Decimal

import pytest

from src.exchange.price_oracle import NativePriceOracle


def D(x) -> Decimal:
    return Decimal(str(x))


async def failing_fetcher(chain):
    raise RuntimeError("price source unavailable")


async def test_gas_quote_is_none_when_the_native_price_is_unavailable():
    """The unit under test is the conversion helper, isolated from web3."""
    from src.exchange.univ3 import UniV3DexClient

    oracle = NativePriceOracle(ttl_seconds=60, fetcher=failing_fetcher)
    result = await UniV3DexClient._gas_cost_in_quote(
        oracle=oracle,
        chain="ethereum",
        gas_units=200_000,
        gas_price_wei=10 ** 9,
    )
    assert result is None, "unpriceable gas must yield None, not a guess"


async def test_gas_quote_is_computed_from_the_oracle_price():
    from src.exchange.univ3 import UniV3DexClient

    async def fetcher(chain):
        return D(2000)

    oracle = NativePriceOracle(ttl_seconds=60, fetcher=fetcher)
    # 200,000 gas at 10 gwei = 0.002 native; at $2000 => $4.00
    result = await UniV3DexClient._gas_cost_in_quote(
        oracle=oracle,
        chain="ethereum",
        gas_units=200_000,
        gas_price_wei=10 * 10 ** 9,
    )
    assert result == D("4.00")


async def test_no_hardcoded_price_fallback_remains_in_univ3():
    """The original defect lived here: Decimal("3000") on any failure."""
    import inspect

    from src.exchange import univ3

    source = inspect.getsource(univ3)
    assert 'Decimal("3000")' not in source
    assert "3000" not in source.replace("dex_pool_fee", "").replace("3000)", "")


def test_gas_units_and_deadline_are_configurable_not_literal():
    """Both were inline literals: a flat 200_000 gas estimate applied to every
    chain, and a 600-second swap deadline -- ten minutes, during which an
    arbitrage transaction landing late is a guaranteed loss."""
    from src.core.config import DexConfig, DexContracts

    cfg = DexConfig(
        uniswap_v3={
            "ethereum": DexContracts(
                router="0x" + "11" * 20,
                quoter_v2="0x" + "22" * 20,
                weth="0x" + "33" * 20,
            )
        }
    )
    assert cfg.swap_gas_estimate_units > 0
    assert cfg.swap_deadline_seconds > 0
    assert cfg.swap_deadline_seconds <= 120, (
        "an arbitrage swap deadline must be short; a late fill is a loss"
    )
    assert cfg.native_price_ttl_seconds > 0
