"""The raw dislocation: pool mid against CEX mid, before any cost.

This is the number that decides whether the strategy is possible at all, and it is the
only one in the report with nothing subtracted from it. Every other figure carries
something:

    best gross   the pool fee, half the CEX spread, price impact -- and it is a max
                 over two directions, so measurement noise is rectified into it
    net          the taker fee and gas on top

A negative best-gross is therefore ambiguous: it can mean "the two venues are at
parity and the fees are simply unavoidable", or "the venues genuinely disagree, just
not by enough". Those have opposite implications -- the first cannot be fixed by any
execution improvement whatever, the second is a fee and universe problem -- and only
the raw dislocation tells them apart.
"""
from decimal import Decimal, getcontext

import pytest

from src.exchange.pool_state import PoolSnapshot
from src.exchange.univ3_math import TickInfo
from src.research.observations import Observation, mid_dislocation_bps


def _pool(price, decimals0=18, decimals1=6):
    getcontext().prec = 60
    raw = Decimal(price) * (Decimal(10) ** decimals1) / (Decimal(10) ** decimals0)
    liquidity = 10 ** 24
    return PoolSnapshot(
        sqrt_price_x96=int(Decimal(2 ** 96) * raw.sqrt()),
        liquidity=liquidity, tick=0, fee=500, tick_spacing=10,
        ticks=[TickInfo(tick=-500000, liquidity_net=liquidity),
               TickInfo(tick=500000, liquidity_net=-liquidity)],
        decimals0=decimals0, decimals1=decimals1,
        token0="0x" + "11" * 20, token1="0x" + "22" * 20, chain="ethereum",
        known_lower_tick=-500000, known_upper_tick=500000,
    )


def _obs(cex_mid, pool_price, decimals0=18, decimals1=6):
    return Observation(
        ts=0.0, cex_symbol="ETHUSDT", base="ETH", quote="USDT", chain="ethereum",
        pool_fee=500, pool_address="0x" + "ab" * 20,
        cex_bids=[(Decimal(cex_mid) * Decimal("0.99999"), Decimal("100"))],
        cex_asks=[(Decimal(cex_mid) * Decimal("1.00001"), Decimal("100"))],
        pool=_pool(pool_price, decimals0, decimals1),
    )


def test_parity_is_zero():
    assert abs(mid_dislocation_bps(_obs("1900", "1900"), base_is_token0=True)) < Decimal("0.1")


def test_a_pool_above_the_exchange_is_positive():
    # +50 bps
    value = mid_dislocation_bps(_obs("1900", "1909.5"), base_is_token0=True)
    assert value == pytest.approx(Decimal("50"), abs=Decimal("0.5"))


def test_a_pool_below_the_exchange_is_negative():
    value = mid_dislocation_bps(_obs("1900", "1890.5"), base_is_token0=True)
    assert value == pytest.approx(Decimal("-50"), abs=Decimal("0.5"))


def test_no_fee_is_subtracted():
    """The pool fee is 5 bps here and must not appear. Subtracting it would make the
    raw dislocation just another cost-laden figure, and its whole purpose is to be the
    one number that is not."""
    value = mid_dislocation_bps(_obs("1900", "1900"), base_is_token0=True)
    assert abs(value) < Decimal("0.1")


def test_token_order_is_respected():
    """Base as token1: the pool's token0-in-token1 price is the reciprocal. Getting
    this wrong is a factor of price squared -- the same orientation error that once
    produced a 36-billion-bps reading."""
    inverted = _obs("1900", Decimal(1) / Decimal("1909.5"), decimals0=6, decimals1=18)
    value = mid_dislocation_bps(inverted, base_is_token0=False)
    assert value == pytest.approx(Decimal("50"), abs=Decimal("1"))


def test_a_one_sided_book_has_no_mid():
    observation = Observation(
        ts=0.0, cex_symbol="X", base="A", quote="B", chain="ethereum",
        pool_fee=500, pool_address="0x" + "ab" * 20,
        cex_bids=[(Decimal("100"), Decimal("1"))], cex_asks=[],
        pool=_pool("100"),
    )
    assert mid_dislocation_bps(observation, base_is_token0=True) is None
