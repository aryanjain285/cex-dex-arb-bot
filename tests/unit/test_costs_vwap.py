"""Depth-aware pricing.

The detector previously priced every trade at best-bid/best-ask regardless of
size, while the order book depth it had already fetched went unused. On the
thin altcoin pools this strategy targets, top-of-book can be a small fraction
of the intended trade size, so the price used was one the trade could never
actually achieve -- and the error always favoured trading.
"""
from decimal import Decimal

import pytest

from src.strategy.costs import walk_book


def D(x) -> Decimal:
    return Decimal(str(x))


def test_single_level_with_ample_depth_returns_that_price():
    levels = [(D(100), D(10))]
    fill = walk_book(levels, D(3))
    assert fill.vwap == D(100)
    assert fill.filled_base == D(3)
    assert fill.complete is True


def test_multiple_levels_are_volume_weighted():
    # 2 @ 100 then 3 @ 110 -> (2*100 + 3*110) / 5 = 106
    levels = [(D(100), D(2)), (D(110), D(3))]
    fill = walk_book(levels, D(5))
    assert fill.vwap == D(106)
    assert fill.filled_base == D(5)
    assert fill.complete is True


def test_partial_consumption_of_final_level():
    # 2 @ 100 then 1 of the 5 @ 110 -> (2*100 + 1*110) / 3 = 103.333...
    levels = [(D(100), D(2)), (D(110), D(5))]
    fill = walk_book(levels, D(3))
    assert fill.filled_base == D(3)
    assert fill.vwap == (D(200) + D(110)) / D(3)
    assert fill.complete is True


def test_insufficient_depth_is_reported_not_hidden():
    """The critical case: the book cannot fill the request.

    walk_book must report the shortfall rather than returning the VWAP of
    what it could fill as though the whole size were achievable.
    """
    levels = [(D(100), D(1)), (D(110), D(1))]
    fill = walk_book(levels, D(10))
    assert fill.complete is False
    assert fill.filled_base == D(2)
    assert fill.requested_base == D(10)
    assert fill.vwap == D(105)


def test_empty_book_returns_no_fill():
    fill = walk_book([], D(5))
    assert fill.complete is False
    assert fill.filled_base == D(0)
    assert fill.vwap is None


def test_vwap_is_worse_than_top_of_book_when_size_crosses_levels():
    """Regression guard for the actual defect.

    Pricing at top-of-book would return 100; walking the book returns a
    worse price. The gap is the cost the detector was ignoring.
    """
    levels = [(D(100), D(1)), (D(120), D(9))]
    top_of_book = levels[0][0]
    fill = walk_book(levels, D(10))
    assert fill.vwap > top_of_book
    assert fill.vwap == D(118)


def test_zero_size_requests_nothing():
    fill = walk_book([(D(100), D(5))], D(0))
    assert fill.filled_base == D(0)
    assert fill.complete is True
    assert fill.vwap is None


def test_negative_size_is_rejected():
    with pytest.raises(ValueError):
        walk_book([(D(100), D(5))], D(-1))


def test_levels_must_be_monotonic_in_the_direction_given():
    """Guards against passing an unsorted or wrong-side book.

    walk_book consumes levels in the order supplied, so a caller that hands
    it an unsorted book would silently get a wrong VWAP. Detect it instead.
    """
    unsorted_asks = [(D(110), D(1)), (D(100), D(1))]
    with pytest.raises(ValueError):
        walk_book(unsorted_asks, D(2), ascending=True)
