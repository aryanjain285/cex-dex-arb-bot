"""The simulator must not invent liquidity past the edge of what it observed.

Found by differential test against QuoterV2 on a THIN pool. Inside the recorded
tick range the local math agrees exactly with the deployed contract, including
six-tick crossings. Past the last recorded tick it did this instead:

    ARB/USDT 0.05% on Arbitrum, block 495685462, 6 recorded ticks
       10,000 ARB in ->  local 628,013,590   chain 358,742,981   (+75%)
      100,000 ARB in ->  local 2,538,350,490 chain 364,408,363   (+597%)

and reported `range_exhausted=False` on every one of them.

The mechanism: with no further RECORDED tick, the loop targeted the extreme of
the tick space and filled the whole remaining input against the current
liquidity -- as if the last observed range extended to infinity. It then took the
`remaining <= 0` exit, which does not set the flag.

Why this is the worst possible failure direction: the invented liquidity always
makes the price BETTER than reality, and the flag that exists to say "do not
trust this" stays clear. A size curve built on it slopes up where reality slopes
down, so `argmax` picks the largest size available and reports profit that cannot
be filled. Every guard downstream -- `price_for_amount_in` returning None, the
optimiser marking a point unpriceable -- was already correct and was simply never
reached.

The distinction the code was missing is epistemic, not arithmetic. "No more
initialised ticks in the pool" and "no more initialised ticks inside the window I
scanned" produce identical tick lists and require opposite answers. So the window
is now recorded explicitly, and outside it the answer is "unknown", never a number.
"""
from decimal import Decimal

import pytest

from src.exchange.univ3_math import (
    TickInfo,
    V3Pool,
    sqrt_price_x96_from_tick,
)


def _pool(*, ticks, liquidity, tick=0, fee=500, spacing=10,
          known_lower=None, known_upper=None):
    return V3Pool(
        sqrt_price_x96=sqrt_price_x96_from_tick(tick),
        liquidity=liquidity,
        tick=tick,
        fee=fee,
        tick_spacing=spacing,
        ticks=ticks,
        decimals0=18,
        decimals1=6,
        known_lower_tick=known_lower,
        known_upper_tick=known_upper,
    )


# Liquidity that a modest swap can walk straight through: the recorded band is
# narrow, so a large swap must run out of observed data.
NARROW = [
    TickInfo(tick=-200, liquidity_net=0),
    TickInfo(tick=-100, liquidity_net=0),
    TickInfo(tick=100, liquidity_net=0),
    TickInfo(tick=200, liquidity_net=0),
]


class TestTheFlagIsHonest:
    def test_walking_past_the_last_recorded_tick_sets_range_exhausted(self):
        pool = _pool(ticks=NARROW, liquidity=10 ** 15)
        # Large enough to push the price below tick -200, the last thing observed.
        result = pool.swap_exact_in_detailed(10 ** 24, zero_for_one=True)
        assert result.range_exhausted is True, (
            "the swap left the observed range; reporting otherwise is the defect "
            "that made a thin pool look 75% deeper than it is"
        )

    def test_a_swap_inside_the_recorded_range_is_not_flagged(self):
        """The flag must discriminate. Always-True is as useless as always-False."""
        pool = _pool(ticks=NARROW, liquidity=10 ** 15)
        result = pool.swap_exact_in_detailed(10 ** 9, zero_for_one=True)
        assert result.range_exhausted is False
        assert result.amount_out > 0

    def test_no_output_is_invented_beyond_the_observed_edge(self):
        """The real test of the fix: not just the flag, but the number.

        A swap 1,000x larger than one that reaches the observed edge must not
        produce 1,000x the output, because no liquidity was observed out there.
        """
        pool = _pool(ticks=NARROW, liquidity=10 ** 15)
        at_edge = pool.swap_exact_in_detailed(10 ** 21, zero_for_one=True)
        far_past = pool.swap_exact_in_detailed(10 ** 24, zero_for_one=True)
        assert far_past.amount_out <= at_edge.amount_out * 2, (
            f"output grew from {at_edge.amount_out} to {far_past.amount_out} on "
            f"input the pool was never observed to be able to absorb"
        )

    def test_the_unfilled_input_is_reported_as_unconsumed(self):
        """`amount_in_consumed` is what the observed liquidity could actually take.
        Claiming the whole input was consumed hides the shortfall."""
        pool = _pool(ticks=NARROW, liquidity=10 ** 15)
        result = pool.swap_exact_in_detailed(10 ** 24, zero_for_one=True)
        assert result.amount_in_consumed < 10 ** 24


class TestTheScannedWindowIsUsedWhenKnown:
    """Liquidity is genuinely flat where no tick is initialised -- so a scan that
    found no ticks in its window still prices exactly, inside that window. Losing
    that would make the fix needlessly blind on pools with sparse ticks."""

    def test_a_swap_inside_the_scanned_window_prices_without_flagging(self):
        pool = _pool(ticks=[], liquidity=10 ** 15, spacing=1,
                     known_lower=-60, known_upper=60)
        result = pool.swap_exact_in_detailed(10 ** 6, zero_for_one=True)
        assert result.range_exhausted is False
        assert result.amount_out > 0

    def test_a_swap_leaving_the_scanned_window_is_flagged(self):
        pool = _pool(ticks=[], liquidity=10 ** 15, spacing=1,
                     known_lower=-60, known_upper=60)
        result = pool.swap_exact_in_detailed(10 ** 24, zero_for_one=True)
        assert result.range_exhausted is True

    def test_no_ticks_and_no_window_refuses_rather_than_guessing(self):
        """Observed nothing, so every size is unknown. This is the ARB/USDT 0.01%
        pool: zero initialised ticks in range, and the old code happily quoted
        1,000,000 ARB against it."""
        pool = _pool(ticks=[], liquidity=10 ** 15, spacing=1)
        result = pool.swap_exact_in_detailed(10 ** 24, zero_for_one=True)
        assert result.range_exhausted is True


class TestBothDirections:
    @pytest.mark.parametrize("zero_for_one", [True, False])
    def test_the_bound_applies_in_both_directions(self, zero_for_one):
        """Travelling up bounds on the upper tick, down on the lower. Applying one
        bound to both directions leaves half the defect in place."""
        pool = _pool(ticks=NARROW, liquidity=10 ** 15)
        result = pool.swap_exact_in_detailed(10 ** 24, zero_for_one=zero_for_one)
        assert result.range_exhausted is True


class TestThePriceGuardNowFires:
    """`price_for_amount_in` already refused on an exhausted range. It was simply
    never told. This is the end-to-end path the optimiser depends on."""

    def test_an_oversized_quote_returns_no_price(self):
        pool = _pool(ticks=NARROW, liquidity=10 ** 15)
        assert pool.price_for_amount_in(Decimal(10 ** 6), zero_for_one=True) is None

    def test_a_sensible_quote_still_returns_a_price(self):
        pool = _pool(ticks=NARROW, liquidity=10 ** 15)
        price = pool.price_for_amount_in(Decimal("0.000001"), zero_for_one=True)
        assert price is not None and price > 0

    def test_the_curve_marks_the_point_where_knowledge_ends(self):
        """The size curve must show None past the observed edge, not a rising
        line -- that line is what would make argmax pick an unfillable size."""
        pool = _pool(ticks=NARROW, liquidity=10 ** 15)
        sizes = [Decimal("0.000001"), Decimal("1"), Decimal(10 ** 6)]
        curve = pool.price_curve(sizes, zero_for_one=True)
        prices = [p for _, p in curve]
        assert prices[0] is not None
        assert prices[-1] is None
