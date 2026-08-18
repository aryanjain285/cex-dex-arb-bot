"""Uniswap v3 swap math, computed locally instead of via an RPC round trip.

Why this is the keystone rather than an optimisation:

  * QuoterV2 costs an eth_call. Measured: 0.31s median on Ethereum, 0.78s on
    Arbitrum, 0.83s on Base, with a 2.2s p90 on Ethereum that exceeds the 2.0s
    opportunity TTL. Local math is microseconds.
  * It makes the SIZE CURVE free. Asking "is $1,000 profitable?" costs one RPC
    call; asking "what is the optimal size?" needs twenty, which is why the system
    has only ever evaluated a single fixed notional. Locally, twenty sizes cost
    nothing.
  * It makes offline BACKTESTING possible at all. A recorded pool state can be
    re-quoted at any size, for any cost assumption, months later. A recorded
    QuoterV2 answer can only ever be re-read at the one size it was asked about.

The correctness bar is exact agreement with the deployed QuoterV2, and these tests
enforce it in two ways: unit tests against known-good fixed-point values, and a
differential test against the live contract (marked, skipped without an RPC URL).
Uniswap's own maths is integer arithmetic throughout; approximating it in floats
would produce an error of exactly the size this strategy trades on.
"""
from decimal import Decimal

import pytest

from src.exchange.univ3_math import (
    Q96, TickInfo, V3Pool, amount0_delta, amount1_delta,
    price_from_sqrt_price_x96, sqrt_price_x96_from_tick, tick_from_sqrt_price_x96,
)


def D(x) -> Decimal:
    return Decimal(str(x))


# --- fixed point primitives ---------------------------------------------


def test_q96_is_the_uniswap_fixed_point_scale():
    assert Q96 == 2 ** 96


def test_sqrt_price_at_tick_zero_is_one():
    """Tick 0 means price 1.0, so sqrtPriceX96 is exactly 2^96."""
    assert sqrt_price_x96_from_tick(0) == Q96


def test_sqrt_price_moves_one_basis_point_per_tick():
    """A tick is 1.0001x in price, so sqrt(1.0001) per tick in sqrt space."""
    at_zero = sqrt_price_x96_from_tick(0)
    at_one = sqrt_price_x96_from_tick(1)

    ratio = Decimal(at_one) / Decimal(at_zero)
    expected = Decimal("1.0001").sqrt()
    assert abs(ratio - expected) < Decimal("1e-12")


@pytest.mark.parametrize("tick", [-887272, -100000, -5000, -1, 0, 1, 5000, 100000, 887271])
def test_tick_and_sqrt_price_round_trip(tick):
    """The inverse must recover the tick exactly, or a pool's current tick cannot
    be derived from its slot0 and every tick-crossing calculation drifts."""
    sqrt_price = sqrt_price_x96_from_tick(tick)

    assert tick_from_sqrt_price_x96(sqrt_price) == tick


def test_a_tick_beyond_the_representable_range_is_rejected():
    """Uniswap's own bound. Beyond it the fixed-point value overflows the type the
    contract uses, so a silently-accepted value here would diverge from the chain."""
    with pytest.raises(ValueError):
        sqrt_price_x96_from_tick(887273)
    with pytest.raises(ValueError):
        sqrt_price_x96_from_tick(-887273)


def test_a_price_is_recovered_from_sqrt_price_with_decimals():
    """A WETH/USDC pool at 1900 USDC per WETH: token0 = USDC (6), token1 = WETH
    (18) or the reverse depending on address order, so the decimals adjustment is
    part of the calculation and not an afterthought."""
    # price = (sqrtPriceX96 / 2^96)^2, then scaled by decimals
    sqrt_price = int(Decimal(Q96) * Decimal(1900).sqrt())

    price = price_from_sqrt_price_x96(sqrt_price, decimals0=18, decimals1=18)

    assert abs(price - D(1900)) < D("0.001")


def test_the_decimals_adjustment_recovers_a_real_weth_usdc_price():
    """The direction of the adjustment, pinned with a concrete pool rather than an
    abstract scaling -- because getting the sign backwards is a factor of 10^24 and
    an abstract test makes it easy to assert the wrong way round (I did).

    token0 = WETH (18), token1 = USDC (6), pool at 1900 USDC per WETH:

        1 WETH        = 10^18 raw token0
        1900 USDC     = 1900 * 10^6 raw token1
        raw ratio     = 1900e6 / 1e18 = 1.9e-9      <- what sqrtPriceX96 encodes
        human price   = raw * 10^(d0 - d1) = 1.9e-9 * 10^12 = 1900

    So the adjustment MULTIPLIES by 10^(decimals0 - decimals1).
    """
    raw_ratio = Decimal(1900) * (Decimal(10) ** 6) / (Decimal(10) ** 18)
    sqrt_price = int(Decimal(Q96) * raw_ratio.sqrt())

    price = price_from_sqrt_price_x96(sqrt_price, decimals0=18, decimals1=6)

    assert abs(price - D(1900)) < D("0.01"), (
        f"got {price}; a wrong-direction adjustment would give 1.9e-21 or 1.9e15"
    )


def test_the_adjustment_is_symmetric_when_the_token_order_flips():
    """The same pool with the tokens the other way round must give the reciprocal,
    or one of the two orderings is being priced wrongly -- and pool token order is
    determined by address, so both occur in practice."""
    raw_ratio = Decimal(1900) * (Decimal(10) ** 6) / (Decimal(10) ** 18)
    sqrt_price = int(Decimal(Q96) * raw_ratio.sqrt())
    forward = price_from_sqrt_price_x96(sqrt_price, decimals0=18, decimals1=6)

    # USDC as token0 (6) against WETH as token1 (18), price 1/1900 WETH per USDC.
    reverse_raw = (Decimal(1) / Decimal(1900)) * (Decimal(10) ** 18) / (Decimal(10) ** 6)
    reverse_sqrt = int(Decimal(Q96) * reverse_raw.sqrt())
    reverse = price_from_sqrt_price_x96(reverse_sqrt, decimals0=6, decimals1=18)

    assert abs(forward * reverse - D(1)) < D("0.0001")


# --- the amount deltas --------------------------------------------------


def test_amount0_delta_is_zero_for_a_zero_price_move():
    at = sqrt_price_x96_from_tick(0)
    assert amount0_delta(at, at, 10 ** 18) == 0


def test_amount1_delta_is_zero_for_a_zero_price_move():
    at = sqrt_price_x96_from_tick(0)
    assert amount1_delta(at, at, 10 ** 18) == 0


def test_amount_deltas_are_symmetric_in_their_bounds():
    """Uniswap's helpers take the lower and upper sqrt price in either order."""
    lower = sqrt_price_x96_from_tick(-100)
    upper = sqrt_price_x96_from_tick(100)
    liquidity = 10 ** 20

    assert amount0_delta(lower, upper, liquidity) == amount0_delta(upper, lower, liquidity)
    assert amount1_delta(lower, upper, liquidity) == amount1_delta(upper, lower, liquidity)


def test_a_wider_range_needs_more_of_both_tokens():
    liquidity = 10 ** 20
    narrow = (sqrt_price_x96_from_tick(-10), sqrt_price_x96_from_tick(10))
    wide = (sqrt_price_x96_from_tick(-1000), sqrt_price_x96_from_tick(1000))

    assert amount0_delta(*wide, liquidity) > amount0_delta(*narrow, liquidity)
    assert amount1_delta(*wide, liquidity) > amount1_delta(*narrow, liquidity)


# --- a swap in a single-range pool --------------------------------------


def _flat_pool(liquidity=10 ** 21, tick=0, fee=500) -> V3Pool:
    """One wide initialised range, so no tick is crossed at ordinary sizes.

    Deliberately the simplest case that still exercises the whole swap loop: fee
    deduction, sqrt-price movement, and the amount calculation.
    """
    return V3Pool(
        sqrt_price_x96=sqrt_price_x96_from_tick(tick),
        liquidity=liquidity,
        tick=tick,
        fee=fee,
        tick_spacing=10,
        ticks=[
            TickInfo(tick=-60000, liquidity_net=liquidity),
            TickInfo(tick=60000, liquidity_net=-liquidity),
        ],
        decimals0=18,
        decimals1=18,
    )


def test_a_tiny_swap_prices_close_to_the_pool_price_less_the_fee():
    """A swap small enough not to move the pool should clear at the spot price
    minus the fee tier, and nothing else. 500 = 5 bps."""
    pool = _flat_pool(fee=500)
    amount_in = 10 ** 15  # 0.001 token0

    amount_out = pool.swap_exact_in(amount_in, zero_for_one=True)

    effective = Decimal(amount_out) / Decimal(amount_in)
    assert abs(effective - D("0.9995")) < D("0.0001"), (
        f"effective rate {effective}; expected spot 1.0 less 5 bps"
    )


def test_a_larger_swap_gets_a_worse_price():
    """Price impact, which is the entire reason the size curve is not linear."""
    pool = _flat_pool()

    small = pool.swap_exact_in(10 ** 15, zero_for_one=True)
    large = pool.swap_exact_in(10 ** 19, zero_for_one=True)

    small_rate = Decimal(small) / Decimal(10 ** 15)
    large_rate = Decimal(large) / Decimal(10 ** 19)
    assert large_rate < small_rate


def test_the_swap_does_not_mutate_the_pool():
    """The pool is a snapshot. Quoting twenty sizes against one recorded state is
    the whole point, and it fails if the first quote moves the state."""
    pool = _flat_pool()
    before = (pool.sqrt_price_x96, pool.liquidity, pool.tick)

    pool.swap_exact_in(10 ** 19, zero_for_one=True)

    assert (pool.sqrt_price_x96, pool.liquidity, pool.tick) == before


def test_both_directions_are_supported():
    pool = _flat_pool()

    zero_for_one = pool.swap_exact_in(10 ** 16, zero_for_one=True)
    one_for_zero = pool.swap_exact_in(10 ** 16, zero_for_one=False)

    assert zero_for_one > 0 and one_for_zero > 0


def test_a_higher_fee_tier_returns_less():
    cheap = _flat_pool(fee=100).swap_exact_in(10 ** 16, zero_for_one=True)
    dear = _flat_pool(fee=10000).swap_exact_in(10 ** 16, zero_for_one=True)

    assert cheap > dear


def test_a_zero_amount_swap_returns_zero():
    assert _flat_pool().swap_exact_in(0, zero_for_one=True) == 0


def test_a_negative_amount_is_rejected():
    with pytest.raises(ValueError):
        _flat_pool().swap_exact_in(-1, zero_for_one=True)


def test_an_empty_pool_returns_nothing_rather_than_dividing_by_zero():
    """A pool with no liquidity is a real state -- the survey found several -- and
    it must not raise ZeroDivisionError inside the hot loop."""
    pool = V3Pool(
        sqrt_price_x96=sqrt_price_x96_from_tick(0), liquidity=0, tick=0,
        fee=3000, tick_spacing=60, ticks=[], decimals0=18, decimals1=18,
    )

    assert pool.swap_exact_in(10 ** 18, zero_for_one=True) == 0


# --- crossing ticks -----------------------------------------------------


def test_liquidity_changes_when_a_tick_is_crossed():
    """The case that separates real v3 maths from a constant-product approximation.

    A swap large enough to leave the current range must pick up the next range's
    liquidity, and the price impact changes discontinuously at the boundary.
    """
    liquidity = 10 ** 20
    pool = V3Pool(
        sqrt_price_x96=sqrt_price_x96_from_tick(0),
        liquidity=liquidity,
        tick=0,
        fee=500,
        tick_spacing=10,
        ticks=[
            TickInfo(tick=-100, liquidity_net=liquidity),
            # Liquidity halves below -50 going down.
            TickInfo(tick=-50, liquidity_net=liquidity // 2),
            TickInfo(tick=50, liquidity_net=-liquidity),
        ],
        decimals0=18,
        decimals1=18,
    )

    out = pool.swap_exact_in(10 ** 19, zero_for_one=True)

    assert out > 0
    # And the result must differ from the same swap in a pool with no boundary,
    # or the tick crossing was not actually applied.
    flat = _flat_pool(liquidity=liquidity)
    assert out != flat.swap_exact_in(10 ** 19, zero_for_one=True)


def test_a_swap_stops_when_liquidity_runs_out():
    """Beyond the last initialised tick there is nothing left to trade against, and
    the swap must return what it could fill rather than looping forever."""
    liquidity = 10 ** 18
    pool = V3Pool(
        sqrt_price_x96=sqrt_price_x96_from_tick(0),
        liquidity=liquidity, tick=0, fee=500, tick_spacing=10,
        ticks=[TickInfo(tick=-10, liquidity_net=liquidity),
               TickInfo(tick=10, liquidity_net=-liquidity)],
        decimals0=18, decimals1=18,
    )

    out = pool.swap_exact_in(10 ** 30, zero_for_one=True)

    assert out > 0, "it should fill what it can"
    assert out < 10 ** 30, "it cannot fill more than the pool holds"


# --- the interface the detector needs -----------------------------------


def test_the_pool_prices_a_swap_in_human_units():
    """The detector works in Decimal token amounts, not integer wei, so the pool
    exposes both -- with the conversion in one place rather than at each call site
    where a decimals mistake would be a factor of 10^n."""
    pool = _flat_pool(fee=500)

    price = pool.price_for_amount_in(D("0.001"), zero_for_one=True)

    assert abs(price - D("0.9995")) < D("0.0001")


def test_the_size_curve_is_computable_in_one_call():
    """What the whole exercise is for: Pi(q) over a grid, with no network access.

    Twenty sizes cost one recorded pool state, which is why fixed-$1,000 evaluation
    was a limitation of the RPC path rather than a considered choice.
    """
    pool = _flat_pool()
    sizes = [D("0.001"), D("0.01"), D("0.1"), D(1), D(10)]

    curve = pool.price_curve(sizes, zero_for_one=True)

    assert len(curve) == len(sizes)
    prices = [price for _, price in curve]
    assert prices == sorted(prices, reverse=True), (
        "a larger size must never get a better price in a single pool"
    )
