"""A quote that runs out of recorded ticks must say so, not under-report.

`swap_exact_in` walks initialised ticks and stops when it runs out. A snapshot only
records ticks within the range it scanned, so a large enough swap reaches the edge
of what was recorded -- and then the loop stops with input left over and returns
the partial output.

Under-reporting is the safe DIRECTION (it understates the edge, so it cannot invent
a trade), but silence is not safe. Two states become indistinguishable:

    "this pool really is that thin"        -> the pair is unviable, drop it
    "we only scanned 50 tick spacings"     -> re-read with a wider range

Those call for opposite actions, and the second masquerading as the first would
quietly shrink the tradeable universe. So the swap reports whether it hit the
boundary, and the human-units helper refuses rather than returning a price it knows
to be understated.

This matters most for the size curve, which deliberately probes large sizes to find
`argmax_q Pi(q)`: it is the function most likely to walk off the end of the data.
"""
from decimal import Decimal

import pytest

from src.exchange.univ3_math import (
    SwapResult, TickInfo, V3Pool, sqrt_price_x96_from_tick,
)


def D(x) -> Decimal:
    return Decimal(str(x))


def _narrow_pool(liquidity=10 ** 18) -> V3Pool:
    """Liquidity only within +/-10 ticks, so a modest swap exits the recorded range."""
    return V3Pool(
        sqrt_price_x96=sqrt_price_x96_from_tick(0),
        liquidity=liquidity, tick=0, fee=500, tick_spacing=10,
        ticks=[TickInfo(tick=-10, liquidity_net=liquidity),
               TickInfo(tick=10, liquidity_net=-liquidity)],
        decimals0=18, decimals1=18,
    )


def _wide_pool(liquidity=10 ** 21) -> V3Pool:
    return V3Pool(
        sqrt_price_x96=sqrt_price_x96_from_tick(0),
        liquidity=liquidity, tick=0, fee=500, tick_spacing=10,
        ticks=[TickInfo(tick=-60000, liquidity_net=liquidity),
               TickInfo(tick=60000, liquidity_net=-liquidity)],
        decimals0=18, decimals1=18,
    )


def test_a_swap_within_the_range_reports_no_exhaustion():
    result = _wide_pool().swap_exact_in_detailed(10 ** 16, zero_for_one=True)

    assert isinstance(result, SwapResult)
    assert result.amount_out > 0
    assert result.amount_in_consumed == 10 ** 16
    assert result.range_exhausted is False


def test_a_swap_that_walks_off_the_data_reports_exhaustion():
    result = _narrow_pool().swap_exact_in_detailed(10 ** 30, zero_for_one=True)

    assert result.range_exhausted is True
    assert result.amount_out > 0, "it should still report what it could fill"
    assert result.amount_in_consumed < 10 ** 30


def test_the_consumed_input_is_reported_so_the_shortfall_is_visible():
    """How much of the requested size the recorded data could actually absorb --
    which is the number that says whether to re-read with a wider range."""
    result = _narrow_pool().swap_exact_in_detailed(10 ** 30, zero_for_one=True)

    assert 0 < result.amount_in_consumed < 10 ** 30


def test_the_human_units_helper_refuses_an_exhausted_quote():
    """A price computed from a partial fill is not the price of the requested size.

    Returning it would understate the edge -- safe -- but a caller cannot tell it
    from a genuine thin-pool price, and those need opposite responses.
    """
    price = _narrow_pool().price_for_amount_in(D(1000), zero_for_one=True)

    assert price is None


def test_the_helper_still_prices_a_size_the_data_covers():
    price = _wide_pool().price_for_amount_in(D("0.001"), zero_for_one=True)

    assert price is not None
    assert price > 0


def test_the_size_curve_marks_exhausted_sizes_rather_than_dropping_them():
    """The curve must show WHERE the data ran out. Dropping those points silently
    would make a truncated curve look like a complete one, and `argmax` over it
    would return the largest size the scan happened to cover."""
    pool = _narrow_pool()
    sizes = [D("0.000001"), D("0.0001"), D(1), D(1000)]

    curve = pool.price_curve(sizes, zero_for_one=True)

    assert len(curve) == len(sizes)
    priced = [(s, p) for s, p in curve if p is not None]
    unpriced = [(s, p) for s, p in curve if p is None]
    assert priced, "the small sizes must price"
    assert unpriced, "the large sizes must be marked, not silently omitted"
    # And the boundary is monotonic: everything above the first failure fails too.
    first_unpriced = min(s for s, _ in unpriced)
    assert all(p is None for s, p in curve if s > first_unpriced)


def test_the_plain_swap_helper_still_returns_an_integer():
    """`swap_exact_in` stays the simple interface for callers that only want the
    amount -- the differential test against QuoterV2 among them."""
    out = _wide_pool().swap_exact_in(10 ** 16, zero_for_one=True)

    assert isinstance(out, int)
    assert out > 0


def test_an_exhausted_swap_still_returns_its_partial_amount_from_the_plain_helper():
    """Conservative by construction: the plain helper reports what could be filled,
    because a caller comparing against QuoterV2 wants the amount, and QuoterV2
    itself would revert rather than partially fill -- so a difference here is
    informative rather than a bug."""
    out = _narrow_pool().swap_exact_in(10 ** 30, zero_for_one=True)

    assert out > 0
