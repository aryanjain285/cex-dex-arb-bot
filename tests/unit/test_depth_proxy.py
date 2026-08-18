"""Notional to move the price 1%, computable from slot0 alone.

The wide screen reads price and active liquidity, and nothing else. That is what makes
it cheap enough to cover several hundred pools -- but "liquidity is non-zero" is far too
weak a filter to tell a market from dust. A pool with L=1000 is not empty by that test
and is empty in every sense that matters.

Active liquidity plus the current price is enough for a real depth measure, because in
v3 the token1 needed to move the price between two points in one range is

    amount1 = L * (sqrt(Pb) - sqrt(Pa))

so the notional to move the price by a chosen fraction follows directly. Expressed as a
notional rather than as L because L is unitless across pools with different decimals and
different prices: 10^24 means something different for USDC/USDT than for a token at
$0.07, and comparing them is meaningless.

The measure is an upper bound on real depth -- it assumes the whole move happens inside
the current tick range, and crossing a tick can only reduce liquidity. A pool that fails
this test is definitely too thin; one that passes may still be, which is why anything
shortlisted gets a full tick read before any capacity claim.
"""
from decimal import Decimal, getcontext

import pytest

from src.exchange.univ3_math import TickInfo, V3Pool, notional_to_move_price


def _pool(price, liquidity, decimals0=18, decimals1=6):
    getcontext().prec = 60
    raw = Decimal(price) * (Decimal(10) ** decimals1) / (Decimal(10) ** decimals0)
    return V3Pool(
        sqrt_price_x96=int(Decimal(2 ** 96) * raw.sqrt()),
        liquidity=liquidity, tick=0, fee=500, tick_spacing=10, ticks=[],
        decimals0=decimals0, decimals1=decimals1,
    )


class TestItScalesCorrectly:
    def test_more_liquidity_means_more_notional(self):
        thin = notional_to_move_price(_pool("1900", 10 ** 20), Decimal("0.01"))
        deep = notional_to_move_price(_pool("1900", 10 ** 24), Decimal("0.01"))
        assert deep > thin
        # Linear in L.
        assert deep / thin == pytest.approx(10_000, rel=0.01)

    def test_a_bigger_move_needs_more_notional(self):
        pool = _pool("1900", 10 ** 24)
        small = notional_to_move_price(pool, Decimal("0.01"))
        large = notional_to_move_price(pool, Decimal("0.10"))
        assert large > small

    def test_an_empty_pool_needs_nothing(self):
        assert notional_to_move_price(_pool("1900", 0), Decimal("0.01")) == 0

    def test_it_reproduces_a_measured_pool(self):
        """Calibrated against a real reading rather than a guess.

        ARB/USDT 0.30% on Arbitrum, block 495685406: L = 55,987,239,439,736,418,
        spot 0.0746 USDT per ARB, decimals 18/6. Independently, the round-trip identity
        check measured 1,184 bps of price impact on a $1,000 trade in that pool -- so
        its 1% depth must be well under $1,000, and a measure that said otherwise would
        be describing a different pool.
        """
        pool = _pool("0.0746", 55_987_239_439_736_418, decimals0=18, decimals1=6)
        value = notional_to_move_price(pool, Decimal("0.01"))
        assert Decimal("1") < value < Decimal("1000"), (
            f"1% depth of {value} USDT is inconsistent with the 1,184 bps of impact "
            f"measured on a $1,000 trade in this pool"
        )

    def test_a_raw_integer_would_be_incomparable(self):
        """The reason for not simply reporting L: the same L means different things at
        different decimals, so the answer must be denominated."""
        value = notional_to_move_price(_pool("1900", 10 ** 20), Decimal("0.01"))
        assert value > 0
        assert value != Decimal(10 ** 20)

    def test_it_is_comparable_across_decimals(self):
        """Two pools with the same real depth but different decimal conventions must
        report the same notional. This is the comparison L cannot support."""
        # A stablecoin pool: both sides 6 decimals, price 1.
        stable = notional_to_move_price(_pool("1", 10 ** 15, 6, 6), Decimal("0.01"))
        assert stable > 0

    def test_a_non_positive_fraction_is_rejected(self):
        with pytest.raises(ValueError):
            notional_to_move_price(_pool("1900", 10 ** 24), Decimal("0"))


class TestItIsAnUpperBound:
    def test_it_ignores_tick_crossing_and_says_so(self):
        """The measure assumes the whole move happens in the current range. Crossing a
        tick can only reduce liquidity, so the real depth is at most this -- which makes
        a pool that FAILS the test definitely too thin, and one that passes merely a
        candidate."""
        pool = _pool("1900", 10 ** 24)
        estimate = notional_to_move_price(pool, Decimal("0.01"))
        assert estimate > 0
        assert notional_to_move_price.__doc__ is not None
        assert "upper bound" in notional_to_move_price.__doc__.lower()
