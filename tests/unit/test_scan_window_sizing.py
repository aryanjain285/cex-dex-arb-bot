"""The observed window must mean the same thing on every fee tier.

`tick_range` counts TICK SPACINGS, and spacing is a property of the fee tier:

    fee     spacing    +/-60 spacings in ticks    in price
    0.01%         1                    +/-60        +/-0.6%
    0.05%        10                   +/-600        +/-6.2%
    0.30%        60                 +/-3,600       +/-43.3%
    1.00%       200                +/-12,000    +/-232%/-70%

So one configured number produced a window that varied by 200x across the pools
being compared. Measured consequence on ARB/USDT at 10:40 today: the 0.01% pool
could not price a ONE-token swap -- the window was narrower than the price impact
of the smallest size on the grid -- while the 1.00% pool had a window wider than
the entire plausible price range and never needed a second read.

That is not a cost/accuracy trade-off, it is an incomparability: any statistic
computed across tiers was silently weighting them by their spacing. Maximum
priceable size is what the window buys, and it has to be set in units that mean
the same thing everywhere -- a fraction of price.

The cap on tick count is a real budget constraint (one RPC call per initialised
tick), so it stays. What must not happen is a cap that truncates the tick list
while leaving the window claiming its full span: the swap math trusts that window
literally, so an over-claimed window is the same defect as extrapolating past it.
"""
from decimal import Decimal

import pytest

from src.exchange.pool_state import (
    DEFAULT_PRICE_WINDOW,
    spacings_for_price_window,
)


class TestTheWindowIsPriceRelative:
    @pytest.mark.parametrize("spacing", [1, 10, 60, 200])
    def test_every_spacing_covers_the_same_price_range(self, spacing):
        """+/-10% of price is +/-10% of price whether the tier quantises to 1 tick
        or 200. Each tier rounds UP to the next whole spacing, so coverage is never
        less than asked for."""
        spacings = spacings_for_price_window(Decimal("0.10"), spacing)
        covered_ticks = spacings * spacing
        # 1.0001^953 = 1.10, so +/-10% needs 953 ticks.
        assert covered_ticks >= 953, (
            f"spacing {spacing}: {covered_ticks} ticks does not reach +/-10%"
        )
        # And not wastefully more than one spacing beyond it.
        assert covered_ticks - spacing < 953 + spacing

    def test_a_wider_window_asks_for_more(self):
        assert (spacings_for_price_window(Decimal("0.25"), 10)
                > spacings_for_price_window(Decimal("0.10"), 10))

    def test_at_least_one_spacing_is_always_scanned(self):
        """A window so tight it rounds to zero spacings would record no ticks and no
        window, making the pool unquotable at any size."""
        assert spacings_for_price_window(Decimal("0.0000001"), 200) >= 1

    def test_the_default_is_stated_as_a_fraction_of_price(self):
        assert isinstance(DEFAULT_PRICE_WINDOW, Decimal)
        assert Decimal("0.01") <= DEFAULT_PRICE_WINDOW <= Decimal("1")

    def test_a_non_positive_window_is_rejected(self):
        with pytest.raises(ValueError):
            spacings_for_price_window(Decimal("0"), 10)
        with pytest.raises(ValueError):
            spacings_for_price_window(Decimal("-0.1"), 10)


class TestTheClaimedWindowMatchesTheScan:
    """The property that keeps the swap math honest, checked on the reader's own
    arithmetic rather than through a live call."""

    def test_truncating_the_tick_list_shrinks_the_claimed_window(self):
        from src.exchange.pool_state import clamp_ticks_to_budget

        # Ticks either side of the current price, more than the budget allows.
        ticks = [-3000, -2000, -1000, -500, 500, 1000, 2000, 3000]
        kept, lower, upper = clamp_ticks_to_budget(
            ticks, current_tick=0, lower_bound=-4000, upper_bound=4000, max_ticks=4
        )
        assert len(kept) <= 4
        # Whatever was dropped, the window must not claim to cover it.
        assert lower >= min(kept) or lower > -4000
        assert upper <= max(kept) or upper < 4000
        for dropped in set(ticks) - set(kept):
            assert not (lower <= dropped <= upper), (
                f"tick {dropped} was dropped but the window still claims to cover "
                f"[{lower}, {upper}] -- the swap math would extrapolate through it"
            )

    def test_an_untruncated_list_keeps_the_full_window(self):
        from src.exchange.pool_state import clamp_ticks_to_budget

        ticks = [-1000, 1000]
        kept, lower, upper = clamp_ticks_to_budget(
            ticks, current_tick=0, lower_bound=-4000, upper_bound=4000, max_ticks=50
        )
        assert kept == ticks
        assert (lower, upper) == (-4000, 4000)

    def test_the_kept_ticks_are_the_ones_nearest_the_price(self):
        """Dropping near ticks and keeping far ones would shrink the window to
        nothing while spending the whole budget."""
        from src.exchange.pool_state import clamp_ticks_to_budget

        ticks = [-3000, -50, 50, 3000]
        kept, _, _ = clamp_ticks_to_budget(
            ticks, current_tick=0, lower_bound=-4000, upper_bound=4000, max_ticks=2
        )
        assert set(kept) == {-50, 50}

    def test_ticks_on_one_side_only_still_bound_both_sides(self):
        """A pool whose initialised ticks all sit above the price: the lower window
        edge is then the scan bound, and must not be silently widened."""
        from src.exchange.pool_state import clamp_ticks_to_budget

        ticks = [100, 200, 300, 400]
        kept, lower, upper = clamp_ticks_to_budget(
            ticks, current_tick=0, lower_bound=-4000, upper_bound=4000, max_ticks=2
        )
        assert set(kept) == {100, 200}
        assert lower == -4000, "nothing below the price was dropped, so the lower edge stands"
        assert upper < 300, "300 was dropped, so the window must stop short of it"
