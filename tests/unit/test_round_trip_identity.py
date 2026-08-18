"""An arithmetic identity the whole pricing chain must satisfy.

Buy base on the CEX and sell it on the DEX; then do the reverse, same instant, same
size. Together the two round trips pay the pool fee twice, both half-spreads and both
price impacts, and they collect the dislocation once in each direction, where it
cancels. So:

    gross(CEX_to_DEX) + gross(DEX_to_CEX)  =  -(2*pool_fee + spread + impact)

This is a property of arithmetic, not of the market, so it holds on every observation
whatever prices did. That makes it the most useful check available here, because it
exercises the entire chain at once -- tick math, token ordering, decimals, book
walking, and both directions' unit conversions -- and it is precisely what a
plausible-looking table of basis points cannot reveal.

It would have caught, on its own, every pricing defect this project has had:

  * the buy-leg units error (a base amount passed where the pool wanted quote):
    residual off by a factor of the price;
  * the inverted DEX price (out-per-in used where quote-per-base was needed):
    same;
  * a wrong base_is_token0: residual off by price squared;
  * the simulator extrapolating past its observed tick range: residual ABOVE the
    fee floor.

One caveat, established by the tests below rather than assumed. A basis-point figure
is normalised by its own cost, and the two directions have different costs when the
venues disagree, so the sum carries a second-order term of about
dislocation x edge. At the 1-20 bps dislocations this strategy exists to find that
term is under 0.05 bps; at a 2.7% gap it is 7 bps and lifts the residual above the
fee floor legitimately. So the identity is a first-order check, tight exactly in the
regime of interest, and the one-sided bound is a diagnostic rather than an invariant.

Verified against recorded live observations alongside these tests: ETH/USDT 0.30% on
Ethereum came in at -60.05 bps against an expected -60.05, USDC/USDT 0.01% at -2.10
against -2.10, and ETH/USDC 0.05% at -10.10 against -10.05.
"""
from decimal import Decimal, getcontext

import pytest

from src.exchange.univ3_math import TickInfo, V3Pool
from src.research.optimiser import optimise_size


def _pool(price, fee, decimals0=18, decimals1=6, liquidity=10 ** 27):
    """A pool deep enough that impact is negligible at the tested size.

    Deep on purpose: the identity holds at any depth, but only a deep pool lets the
    residual be compared to the fee floor to a fraction of a basis point. On a thin
    pool the impact term dominates and the test would pass for the wrong reason.
    """
    getcontext().prec = 60
    raw = Decimal(price) * (Decimal(10) ** decimals1) / (Decimal(10) ** decimals0)
    return V3Pool(
        sqrt_price_x96=int(Decimal(2 ** 96) * raw.sqrt()),
        liquidity=liquidity,
        tick=0,
        fee=fee,
        tick_spacing=10,
        ticks=[TickInfo(tick=-600000, liquidity_net=liquidity),
               TickInfo(tick=600000, liquidity_net=-liquidity)],
        decimals0=decimals0,
        decimals1=decimals1,
        known_lower_tick=-600000,
        known_upper_tick=600000,
    )


def _ladder(mid, spread_bps, levels=8, size=Decimal("10000000")):
    half = Decimal(spread_bps) / Decimal(20000)
    bid, ask = mid * (1 - half), mid * (1 + half)
    return (
        [(bid * (1 - Decimal("0.000001") * i), size) for i in range(levels)],
        [(ask * (1 + Decimal("0.000001") * i), size) for i in range(levels)],
    )


def _round_trip(pool, bids, asks, notional, base_is_token0):
    total = Decimal(0)
    for direction in ("CEX_to_DEX", "DEX_to_CEX"):
        curve = optimise_size(
            pool=pool, direction=direction, cex_bids=bids, cex_asks=asks,
            notionals=[notional], taker_fee_bps=Decimal("7.5"),
            gas_quote=Decimal("0.01"), base_is_token0=base_is_token0,
            floor_bps=Decimal(0),
        )
        point = curve.curve[0]
        assert point.gross_bps is not None, point.reason
        total += point.gross_bps
    return total


@pytest.mark.parametrize("fee,fee_bps", [(100, 1), (500, 5), (3000, 30), (10000, 100)])
def test_the_round_trip_costs_two_pool_fees_at_every_tier(fee, fee_bps):
    pool = _pool("1900", fee)
    bids, asks = _ladder(Decimal("1900"), spread_bps=1)
    residual = _round_trip(pool, bids, asks, Decimal("1000"), base_is_token0=True)
    expected = -(Decimal(2 * fee_bps) + Decimal(1))
    assert abs(residual - expected) < Decimal("0.5"), (
        f"fee tier {fee}: round trip cost {residual} bps, expected about {expected}"
    )


def test_the_identity_is_tight_for_realistic_dislocations():
    """The dislocation cancels in the sum to FIRST order, which is what makes this a
    check on the code rather than on the market -- but only to first order.

    A basis-point figure is normalised by its own cost, and the two directions have
    different costs when the venues disagree: one buys at the CEX price, the other at
    the pool price. So the sum carries a second-order term of roughly
    dislocation x edge. At the 1-20 bps dislocations this strategy exists to find,
    that term is under 0.05 bps and the identity is effectively exact -- which is why
    the live check came in at 0.00-0.04 bps on the deep pools. See the test below for
    what happens when the dislocation is large.
    """
    pool = _pool("1900", 500)
    residuals = []
    # +/-20 bps: an order of magnitude beyond anything observed on these pairs.
    for cex_mid in ("1896.2", "1900", "1903.8"):
        bids, asks = _ladder(Decimal(cex_mid), spread_bps=1)
        residuals.append(
            _round_trip(pool, bids, asks, Decimal("1000"), base_is_token0=True)
        )
    assert max(residuals) - min(residuals) < Decimal("0.5"), (
        f"the residual moved with the dislocation: {residuals}. At these magnitudes "
        f"the second-order term is negligible, so movement here is a bug."
    )


def test_the_second_order_term_is_understood_not_ignored():
    """A large dislocation legitimately loosens the identity, and by a predictable
    amount. Documented here because the alternative is discovering it later and
    mistaking it for a defect -- or worse, taking the one-sided bound below as
    absolute when it is only first-order.

    Measured: a 2.7% gap between venues gives a residual of -3.89 bps against a
    -11 bps fee floor. The gap is 7.1 bps, and dislocation x edge is
    0.027 x 270 = 7.3 bps.
    """
    pool = _pool("1900", 500)
    bids, asks = _ladder(Decimal("1850"), spread_bps=1)
    residual = _round_trip(pool, bids, asks, Decimal("1000"), base_is_token0=True)
    edge_bps = Decimal("270")
    predicted_slack = Decimal("0.027") * edge_bps
    slack = residual - Decimal("-11")
    assert abs(slack - predicted_slack) < Decimal("2"), (
        f"slack {slack} bps against a predicted second-order term of "
        f"{predicted_slack} bps: the deviation is not the normalisation term, so "
        f"something else is wrong"
    )


def test_the_identity_holds_with_the_base_as_token1():
    """Token order comes from the address, so half of all pools have it the other way.
    Getting it wrong is a factor of price squared, and the residual is where that
    shows."""
    pool = _pool(Decimal(1) / Decimal(1900), 500, decimals0=6, decimals1=18)
    bids, asks = _ladder(Decimal("1900"), spread_bps=1)
    residual = _round_trip(pool, bids, asks, Decimal("1000"), base_is_token0=False)
    assert abs(residual - Decimal("-11")) < Decimal("1"), (
        f"residual {residual} with base as token1"
    )


def test_a_wrong_token_order_breaks_the_identity_loudly():
    """The negative control for the check itself. If a mis-specified token order still
    satisfied the identity, the identity would be proving nothing."""
    pool = _pool("1900", 500)
    bids, asks = _ladder(Decimal("1900"), spread_bps=1)
    try:
        residual = _round_trip(pool, bids, asks, Decimal("1000"), base_is_token0=False)
    except AssertionError:
        return  # Refusing to price at all is an equally clear signal.
    assert abs(residual - Decimal("-11")) > Decimal("100"), (
        f"an inverted token order produced a residual of {residual}, close enough to "
        f"correct that the identity would not have caught it"
    )


def test_the_residual_never_exceeds_the_fee_floor():
    """One-sided, and the important direction: impact can only subtract, so a residual
    ABOVE the floor means one leg is pricing against liquidity the other cannot see --
    exactly what the simulator did before it stopped extrapolating past its observed
    tick range.

    First-order only. A large dislocation can lift the residual above the floor
    legitimately, through the normalisation term measured in the test above, so this
    bound is a diagnostic for the small-dislocation regime rather than an invariant.
    Here the venues agree, so it applies cleanly.
    """
    pool = _pool("1900", 500)
    bids, asks = _ladder(Decimal("1900"), spread_bps=1)
    for notional in (Decimal("100"), Decimal("1000"), Decimal("100000")):
        residual = _round_trip(pool, bids, asks, notional, base_is_token0=True)
        assert residual <= Decimal("-10"), (
            f"at {notional} notional the round trip cost only {residual} bps, less "
            f"than the two pool fees it must pay"
        )


def test_a_thin_pool_still_satisfies_the_one_sided_bound():
    """Where impact dominates, the identity stops being tight and stays valid. Recorded
    live: ARB/USDT 0.30% showed -1,257 bps against a -73 bps fee floor at $1,000
    notional -- 1,184 bps of pure price impact, which is the pool being nearly empty
    rather than the arithmetic being wrong."""
    pool = _pool("1900", 3000, liquidity=10 ** 18)
    bids, asks = _ladder(Decimal("1900"), spread_bps=1)
    residual = _round_trip(pool, bids, asks, Decimal("1000"), base_is_token0=True)
    assert residual <= Decimal("-61")
