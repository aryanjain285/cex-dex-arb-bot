"""The DEX buy leg spends QUOTE units, not base units.

Caught by running the optimiser against a live ETH/USDT pool: the DEX_to_CEX curve
returned 36,117,456,820 bps of gross edge and a notional of zero. A ten-order-of
-magnitude number is never a market; it is a units error.

The cause is the same one the detector had and the README lists as fixed there:

    CEX_to_DEX   sell base on the DEX   -> the pool input is BASE
    DEX_to_CEX   buy  base on the DEX   -> the pool input is QUOTE

`price_for_amount_in` takes whatever the input token is for the chosen direction,
so passing a base amount for a buy makes the pool price a swap of the wrong token.
For ETH/USDT that is 0.5 USDT instead of 0.5 WETH: a factor of 1,900, and it lands
in the numerator of the edge.

So the curve is parameterised by NOTIONAL, which is well defined for both
directions, and each direction converts it to the input its own DEX leg actually
consumes. The tests below pin that in a way a units error cannot satisfy.
"""
from decimal import Decimal

import pytest

from src.exchange.univ3_math import TickInfo, V3Pool, sqrt_price_x96_from_tick
from src.research.optimiser import optimise_size


def D(x) -> Decimal:
    return Decimal(str(x))


def _pool_at(price, liquidity=10 ** 26, decimals0=18, decimals1=18) -> V3Pool:
    """A pool whose token0-in-token1 price is approximately `price`.

    Deep, so impact does not confound the units question being tested.
    """
    from decimal import getcontext
    getcontext().prec = 60
    raw = D(price) * (Decimal(10) ** decimals1) / (Decimal(10) ** decimals0)
    sqrt_price = int(Decimal(2 ** 96) * raw.sqrt())
    tick_liquidity = liquidity
    return V3Pool(
        sqrt_price_x96=sqrt_price,
        liquidity=liquidity,
        tick=0,  # not used for pricing; the sqrt price is authoritative
        fee=500, tick_spacing=10,
        ticks=[TickInfo(tick=-800000, liquidity_net=tick_liquidity),
               TickInfo(tick=800000, liquidity_net=-tick_liquidity)],
        decimals0=decimals0, decimals1=decimals1,
    )


def _bids(price, size=D(10 ** 6), levels=5):
    return [(D(price) * (1 - D("0.00001") * i), size) for i in range(levels)]


def _asks(price, size=D(10 ** 6), levels=5):
    return [(D(price) * (1 + D("0.00001") * i), size) for i in range(levels)]


# The regression: a realistic ETH-like price, where a units error is a factor of
# 1,900 and therefore impossible to mistake for a market move.
PRICE = D(1900)


def test_the_dex_to_cex_gross_edge_is_a_plausible_number():
    """The bug produced 36,117,456,820 bps. Any real market is under a few hundred."""
    result = optimise_size(
        pool=_pool_at(PRICE), direction="DEX_to_CEX",
        cex_bids=_bids(PRICE * D("1.002")), cex_asks=_asks(PRICE * D("1.002")),
        notionals=[D(1000), D(10_000)],
        taker_fee_bps=D("7.5"), gas_quote=D("0.02"), rotation_cost_quote=D(0),
        base_is_token0=True,
    )

    for point in result.curve:
        assert point.gross_bps is not None
        assert abs(point.gross_bps) < D(1000), (
            f"gross {point.gross_bps} bps is not a market; it is a units error"
        )


def test_the_notional_is_what_was_asked_for():
    """The bug reported a notional of zero, which is the same error seen from the
    other side: base amounts multiplied by a price computed from the wrong token."""
    result = optimise_size(
        pool=_pool_at(PRICE), direction="DEX_to_CEX",
        cex_bids=_bids(PRICE), cex_asks=_asks(PRICE),
        notionals=[D(1000)],
        taker_fee_bps=D("7.5"), gas_quote=D("0.02"), rotation_cost_quote=D(0),
        base_is_token0=True,
    )

    point = result.curve[0]
    assert point.notional_quote is not None
    assert abs(point.notional_quote - D(1000)) / D(1000) < D("0.02"), (
        f"asked for a 1000 notional, priced {point.notional_quote}"
    )


def test_the_base_size_is_the_notional_divided_by_the_price():
    """A $1,000 notional at 1,900 is about 0.526 base, not 1,000 base. The bug had
    these confused, which is why it survived: both are 'the size'."""
    result = optimise_size(
        pool=_pool_at(PRICE), direction="DEX_to_CEX",
        cex_bids=_bids(PRICE), cex_asks=_asks(PRICE),
        notionals=[D(1000)],
        taker_fee_bps=D("7.5"), gas_quote=D("0.02"), rotation_cost_quote=D(0),
        base_is_token0=True,
    )

    point = result.curve[0]
    assert abs(point.size_base - D(1000) / PRICE) / (D(1000) / PRICE) < D("0.02")


def test_both_directions_agree_on_a_fair_market():
    """With the two venues at the same price, both directions must lose by roughly
    the same amount -- the fee and the pool fee. A units error in one direction
    breaks the symmetry immediately."""
    args = dict(
        pool=_pool_at(PRICE),
        cex_bids=_bids(PRICE), cex_asks=_asks(PRICE),
        notionals=[D(10_000)],
        taker_fee_bps=D("7.5"), gas_quote=D("0.02"), rotation_cost_quote=D(0),
        base_is_token0=True,
    )
    forward = optimise_size(direction="CEX_to_DEX", **args).curve[0]
    reverse = optimise_size(direction="DEX_to_CEX", **args).curve[0]

    assert forward.gross_bps is not None and reverse.gross_bps is not None
    assert abs(forward.gross_bps - reverse.gross_bps) < D(2), (
        f"a fair market should cost both directions the same: "
        f"{forward.gross_bps} vs {reverse.gross_bps}"
    )


def test_token_order_is_respected():
    """Pool token order comes from the address, so base is token0 in some pools and
    token1 in others. Getting it wrong inverts the price -- a factor of 1,900^2 on
    an ETH pool, which no test should be able to miss.
    """
    # base as token1: the pool's token0-in-token1 price is then 1/1900.
    inverted = _pool_at(D(1) / PRICE)

    result = optimise_size(
        pool=inverted, direction="CEX_to_DEX",
        cex_bids=_bids(PRICE), cex_asks=_asks(PRICE),
        notionals=[D(10_000)],
        taker_fee_bps=D("7.5"), gas_quote=D("0.02"), rotation_cost_quote=D(0),
        base_is_token0=False,
    )

    point = result.curve[0]
    assert point.gross_bps is not None
    assert abs(point.gross_bps) < D(1000), (
        f"gross {point.gross_bps} with base as token1: the token order is being "
        f"ignored"
    )


def test_a_genuinely_profitable_dex_to_cex_is_found():
    """The positive control: with the CEX bid above the pool price, buying on the
    DEX and selling on the CEX must show a positive gross."""
    result = optimise_size(
        pool=_pool_at(PRICE), direction="DEX_to_CEX",
        cex_bids=_bids(PRICE * D("1.01")), cex_asks=_asks(PRICE * D("1.01")),
        notionals=[D(10_000)],
        taker_fee_bps=D("7.5"), gas_quote=D("0.02"), rotation_cost_quote=D(0),
        base_is_token0=True,
    )

    point = result.curve[0]
    assert point.gross_bps > D(50), f"expected roughly +95 bps, got {point.gross_bps}"


def test_a_genuinely_profitable_cex_to_dex_is_found():
    result = optimise_size(
        pool=_pool_at(PRICE), direction="CEX_to_DEX",
        cex_bids=_bids(PRICE * D("0.99")), cex_asks=_asks(PRICE * D("0.99")),
        notionals=[D(10_000)],
        taker_fee_bps=D("7.5"), gas_quote=D("0.02"), rotation_cost_quote=D(0),
        base_is_token0=True,
    )

    point = result.curve[0]
    assert point.gross_bps > D(50), f"expected roughly +95 bps, got {point.gross_bps}"
