"""Optimal size, not a fixed $1,000 probe.

The system has only ever asked "is $1,000 profitable?". That is the wrong question,
and it was forced by the RPC path: one size cost one eth_call, so twenty sizes cost
twenty. With local pool math twenty sizes are free, and the right question becomes
answerable:

    Pi(q) = P_sell(q)*q - P_buy(q)*q - C(q)        q* = argmax_q Pi(q)

The shape of Pi is not monotonic and that is the entire point. Gas is fixed per
trade, so small sizes cannot cover it; price impact grows with size, so large sizes
give it all back. The optimum is interior, and a fixed probe finds it only by luck:

    $100     +12 bps gross, gas swamps it        -> net negative
    $250      +9 bps                             -> net POSITIVE
    $1,000    +1 bps                             -> net negative   <- what we probed
    $2,500    -8 bps                             -> net negative

A $1,000 probe reports "no trade" while a $250 trade existed. That is a false
negative produced by the measurement apparatus, and it is exactly the failure mode
this whole exercise is meant to eliminate.

Both venues are walked properly: the CEX side through its book ladder, the DEX side
through the pool's tick liquidity. Neither is a top-of-book approximation, because a
size curve built on top of book would be a straight line and would find no optimum
at all.
"""
from decimal import Decimal

import pytest

from src.research.optimiser import (
    SizePoint, geometric_size_grid, optimise_size,
)
from src.exchange.univ3_math import TickInfo, V3Pool, sqrt_price_x96_from_tick


def D(x) -> Decimal:
    return Decimal(str(x))


def _pool(liquidity=10 ** 24, fee=500, tick=0) -> V3Pool:
    """A deep pool at price 1.0, so impact is gentle and the optimum is interior."""
    return V3Pool(
        sqrt_price_x96=sqrt_price_x96_from_tick(tick),
        liquidity=liquidity, tick=tick, fee=fee, tick_spacing=10,
        ticks=[TickInfo(tick=-60000, liquidity_net=liquidity),
               TickInfo(tick=60000, liquidity_net=-liquidity)],
        decimals0=18, decimals1=18,
    )


def _asks(price, size=D(1000), levels=5):
    """An ask ladder: ASCENDING, each level a little worse for a buyer."""
    return [(D(price) * (1 + D("0.0001") * i), size) for i in range(levels)]


def _bids(price, size=D(1000), levels=5):
    """A bid ladder: DESCENDING, each level a little worse for a seller.

    Separate from `_asks` because `walk_book` enforces the ordering -- and it was
    right to: my first fixture handed it an ascending bid ladder, which would have
    made every seller's VWAP better than the touch.
    """
    return [(D(price) * (1 - D("0.0001") * i), size) for i in range(levels)]


# --- the grid -----------------------------------------------------------


def test_the_grid_spans_orders_of_magnitude():
    """Geometric, not linear: the interesting region spans $50 to $50,000 and a
    linear grid either misses the small end or wastes most of its points."""
    grid = geometric_size_grid(minimum=D(50), maximum=D(50_000), points=10)

    assert len(grid) == 10
    assert grid[0] == D(50)
    assert grid[-1] == D(50_000)
    assert grid == sorted(grid)
    # Each step is a constant ratio, within rounding.
    ratios = [grid[i + 1] / grid[i] for i in range(len(grid) - 1)]
    assert max(ratios) / min(ratios) < D("1.01")


def test_a_single_point_grid_is_the_minimum():
    assert geometric_size_grid(minimum=D(100), maximum=D(100), points=1) == [D(100)]


def test_an_inverted_grid_is_rejected():
    with pytest.raises(ValueError):
        geometric_size_grid(minimum=D(1000), maximum=D(100), points=5)


def test_a_non_positive_minimum_is_rejected():
    """Geometric spacing has no meaning from zero, and silently clamping would
    produce a grid that does not span what the caller asked for."""
    with pytest.raises(ValueError):
        geometric_size_grid(minimum=D(0), maximum=D(1000), points=5)


# --- the curve ----------------------------------------------------------


def test_gas_makes_small_sizes_unprofitable():
    """The left-hand side of the curve. A fixed cost per trade cannot be covered by
    an arbitrarily small notional, however good the price is."""
    result = optimise_size(
        pool=_pool(), direction="CEX_to_DEX",
        cex_bids=_bids(D("0.98")), cex_asks=_asks(D("0.98")),
        notionals=[D("0.001"), D(1), D(100)],
        taker_fee_bps=D("7.5"), gas_quote=D(5), rotation_cost_quote=D(0),
        base_is_token0=True,
    )

    by_size = {p.notional_requested: p for p in result.curve}
    assert by_size[D("0.001")].net_bps < by_size[D(100)].net_bps, (
        "a fixed $5 of gas must hurt a tiny trade far more than a large one"
    )


def test_impact_makes_large_sizes_unprofitable():
    """The right-hand side. A thin pool gives back the whole edge at size."""
    # 10^22 raw at 18 decimals is 10,000 units of virtual liquidity, so a 5,000
    # unit swap is heavy impact while still being priceable -- the comparison needs
    # both points to have a price, not one of them to be refused.
    result = optimise_size(
        pool=_pool(liquidity=10 ** 22), direction="CEX_to_DEX",
        cex_bids=_bids(D("0.98"), size=D(10 ** 9)),
        cex_asks=_asks(D("0.98"), size=D(10 ** 9)),
        notionals=[D(1), D(100), D(5000)],
        taker_fee_bps=D("7.5"), gas_quote=D("0.01"), rotation_cost_quote=D(0),
        base_is_token0=True,
    )

    by_size = {p.notional_requested: p for p in result.curve}
    assert by_size[D(5000)].net_bps is not None, "both points must be priceable"
    assert by_size[D(5000)].net_bps < by_size[D(1)].net_bps


def test_the_optimum_is_interior_and_is_found():
    """The whole point, stated as a test: with gas on the left and impact on the
    right, the best size is neither the smallest nor the largest probed."""
    # Calibrated so both ends genuinely lose and the middle genuinely wins:
    #   gross at tiny size    ~95 bps (CEX 0.99 vs pool 1.0, less the 5 bps fee)
    #   gas $2                covered above roughly $2,000 of notional
    #   impact                bites above roughly $20,000 against 2.5e6 of
    #                         virtual liquidity
    # so the optimum sits between, which is what makes this test meaningful rather
    # than an accident of the numbers.
    result = optimise_size(
        pool=_pool(liquidity=25 * 10 ** 23), direction="CEX_to_DEX",
        cex_bids=_bids(D("0.99"), size=D(10 ** 9)),
        cex_asks=_asks(D("0.99"), size=D(10 ** 9)),
        notionals=geometric_size_grid(minimum=D("0.01"), maximum=D(500_000), points=14),
        taker_fee_bps=D("7.5"), gas_quote=D(2), rotation_cost_quote=D(0),
        base_is_token0=True,
    )

    assert result.best is not None
    sizes = [p.notional_requested for p in result.curve]
    assert result.best.notional_requested != min(sizes), "optimum at the smallest probe"
    assert result.best.notional_requested != max(sizes), "optimum at the largest probe"


def test_a_fixed_probe_can_miss_a_real_opportunity():
    """The false negative the fixed $1,000 probe produces, demonstrated.

    Constructed so a small size is profitable and a large one is not. A single
    large probe reports no trade; the curve finds it.
    """
    # The largest probe is deliberately far into the impact-dominated region, so a
    # single-probe measurement at that size reports no trade while a smaller one is
    # profitable.
    sizes = geometric_size_grid(minimum=D(10), maximum=D(500_000), points=12)
    args = dict(
        pool=_pool(liquidity=25 * 10 ** 23), direction="CEX_to_DEX",
        cex_bids=_bids(D("0.99"), size=D(10 ** 9)),
        cex_asks=_asks(D("0.99"), size=D(10 ** 9)),
        taker_fee_bps=D("7.5"), gas_quote=D(2), rotation_cost_quote=D(0),
        base_is_token0=True,
    )

    full = optimise_size(notionals=sizes, **args)
    just_the_big_one = optimise_size(notionals=[max(sizes)], **args)

    assert full.best is not None
    assert full.best.net_bps > just_the_big_one.curve[0].net_bps, (
        "the curve must find a better size than the largest probe"
    )


def test_the_curve_reports_every_probed_size():
    """Including the losing ones. A curve that only kept the winners could not show
    where the boundary is, which is the interesting part for capacity."""
    sizes = [D("0.001"), D(1), D(1000)]
    result = optimise_size(
        pool=_pool(), direction="CEX_to_DEX",
        cex_bids=_bids(D(1)), cex_asks=_asks(D(1)), notionals=sizes,
        taker_fee_bps=D("7.5"), gas_quote=D(1), rotation_cost_quote=D(0),
        base_is_token0=True,
    )

    assert [p.notional_requested for p in result.curve] == sizes


def test_a_size_the_book_cannot_fill_is_marked_not_dropped():
    """CEX depth is finite. A size beyond it has no price, and that is information
    about capacity -- silently dropping it would make the curve look complete."""
    result = optimise_size(
        pool=_pool(), direction="CEX_to_DEX",
        cex_bids=_bids(D(1), size=D(1), levels=2),
        cex_asks=_asks(D(1), size=D(1), levels=2),
        notionals=[D(1), D(1_000_000)],
        taker_fee_bps=D("7.5"), gas_quote=D(0), rotation_cost_quote=D(0),
        base_is_token0=True,
    )

    big = [p for p in result.curve if p.notional_requested == D(1_000_000)][0]
    assert big.net_bps is None
    assert big.reason and "depth" in big.reason.lower()


def test_a_size_beyond_the_pools_recorded_ticks_is_marked():
    """Same discipline on the DEX side: the local quote refuses when it would leave
    the recorded tick range, and that refusal must reach the curve rather than
    appearing as a bad price."""
    thin = V3Pool(
        sqrt_price_x96=sqrt_price_x96_from_tick(0),
        liquidity=10 ** 18, tick=0, fee=500, tick_spacing=10,
        ticks=[TickInfo(tick=-10, liquidity_net=10 ** 18),
               TickInfo(tick=10, liquidity_net=-10 ** 18)],
        decimals0=18, decimals1=18,
    )

    result = optimise_size(
        pool=thin, direction="CEX_to_DEX",
        cex_bids=_bids(D(1), size=D(10 ** 9)), cex_asks=_asks(D(1), size=D(10 ** 9)),
        notionals=[D(1_000_000)],
        taker_fee_bps=D("7.5"), gas_quote=D(0), rotation_cost_quote=D(0),
        base_is_token0=True,
    )

    point = result.curve[0]
    assert point.net_bps is None
    assert point.reason


def test_no_profitable_size_reports_no_best_rather_than_the_least_bad():
    """When nothing clears, `best` must be None. Returning the least-negative point
    would read as an opportunity in any downstream summary."""
    result = optimise_size(
        pool=_pool(), direction="CEX_to_DEX",
        cex_bids=_bids(D(1)), cex_asks=_asks(D(1)),
        notionals=[D(1), D(10), D(100)],
        taker_fee_bps=D("7.5"), gas_quote=D(1), rotation_cost_quote=D(2),
        base_is_token0=True,
        floor_bps=D(5),
    )

    assert result.best is None
    assert result.best_gross_bps is not None, (
        "the gross figure is still reported -- it is the research signal even when "
        "nothing is tradeable"
    )


def test_the_gross_edge_is_reported_separately_from_the_net():
    """Costs are a policy; the gross dislocation is the market. Research needs the
    second even when the first makes everything unprofitable -- which is the
    situation this system is actually in."""
    result = optimise_size(
        pool=_pool(), direction="CEX_to_DEX",
        cex_bids=_bids(D("0.99")), cex_asks=_asks(D("0.99")),
        notionals=[D(1), D(10)],
        taker_fee_bps=D("7.5"), gas_quote=D(1), rotation_cost_quote=D(0),
        base_is_token0=True,
    )

    for point in result.curve:
        if point.net_bps is not None:
            assert point.gross_bps > point.net_bps


def test_both_directions_are_supported():
    for direction in ("CEX_to_DEX", "DEX_to_CEX"):
        result = optimise_size(
            pool=_pool(), direction=direction,
            cex_bids=_bids(D("0.99")), cex_asks=_asks(D("0.99")),
            notionals=[D(1), D(10)],
            taker_fee_bps=D("7.5"), gas_quote=D("0.01"),
            rotation_cost_quote=D(0),
            base_is_token0=True,
        )
        assert result.curve, direction


def test_an_empty_size_grid_is_rejected():
    with pytest.raises(ValueError):
        optimise_size(
            pool=_pool(), direction="CEX_to_DEX",
            cex_bids=_bids(D(1)), cex_asks=_asks(D(1)), notionals=[],
            taker_fee_bps=D("7.5"), gas_quote=D(0), rotation_cost_quote=D(0),
            base_is_token0=True,
        )
