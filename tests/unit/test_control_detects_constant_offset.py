"""A fixed offset is not a signal, and the control cannot always tell you so.

Working out what the scrambled control actually computes is what separates two things I
first conflated. The scrambled dislocation is

    pool(t') - cex(t)  =  [pool(t') - cex(t')]  +  [cex(t') - cex(t)]
                       =  the true gap at t'    +  the venue's own move over the offset

so scrambling always ADDS the variance of the underlying price's movement. It never
removes anything. Which gives two regimes:

    the level moves a lot relative to the gap   scrambled >> true, the control
                                                discriminates. ETH/USDC: true p99 0.07 bps
                                                against a scrambled 9.70.

    the level barely moves relative to the gap  scrambled ~= true, and the control CANNOT
                                                discriminate -- there was nothing for
                                                scrambling to disturb. USDC/USDT on Base:
                                                true p99 8.77 against a scrambled 8.77.

I first read that second case as proof of a constant offset. It is not proof of anything
about the gap; it is a statement about the control's POWER on that pair. So the two are
reported separately, and constancy is judged from the true distribution's own spread --
a direct measurement rather than an inference from a test with no power.

The case that forced this, and it would have been the headline:

    USDC/USDT 0.01% on Base, 331 observations over 2 hours
      raw dislocation   mean +9.47 bps, p1 +9.23, p99 +9.83
      net edge          +0.85 bps, the ONLY positive in the entire dataset
      control           true p99 8.77 vs scrambled p99 8.77

A 0.6 bps range around a 9.4 bps median, across two hours. That is a fixed offset between
the two things being compared -- almost certainly bridged USDT on Base against native USDT
on Binance, which are not the same asset and hold a persistent peg difference -- and not a
market gap. The standing-basis test flagged it too, by a different route, which is what a
second opinion is for.
"""
import random

import pytest

from decimal import Decimal

from src.exchange.pool_state import PoolSnapshot
from src.exchange.univ3_math import TickInfo
from src.research.evaluate import CostModel
from src.research.observations import Observation
from src.research.report import scrambled_control


def _pool(price):
    from decimal import getcontext
    getcontext().prec = 60
    raw = Decimal(str(price)) * (Decimal(10) ** 6) / (Decimal(10) ** 18)
    liquidity = 10 ** 24
    return PoolSnapshot(
        sqrt_price_x96=int(Decimal(2 ** 96) * raw.sqrt()),
        liquidity=liquidity, tick=0, fee=500, tick_spacing=10,
        ticks=[TickInfo(tick=-500000, liquidity_net=liquidity),
               TickInfo(tick=500000, liquidity_net=-liquidity)],
        decimals0=18, decimals1=6, block_number=1,
        address="0x" + "ab" * 20, token0="0x" + "11" * 20,
        token1="0x" + "22" * 20, chain="base",
        known_lower_tick=-500000, known_upper_tick=500000,
    )


def _obs(ts, cex, dex):
    spread = Decimal("0.00002")
    return Observation(
        ts=ts, cex_symbol="X/Y", base="X", quote="Y", chain="base",
        pool_fee=500, pool_address="0x" + "ab" * 20,
        cex_bids=[(Decimal(str(cex)) * (1 - spread), Decimal("100000"))],
        cex_asks=[(Decimal(str(cex)) * (1 + spread), Decimal("100000"))],
        cex_feed_ts=ts, pool=_pool(dex),
        gas_price_wei=10 ** 9, native_price_quote=Decimal("1900"),
    )


COSTS = CostModel(
    taker_fee_bps=Decimal("7.5"), cex_legs=1, gas_units=200_000,
    rotation_cost_quote=Decimal("0"), floor_bps=Decimal("5"),
)
NOTIONALS = [Decimal("1000")]


def _constant_offset(n=300, offset_bps=9.4):
    """Both venues moving together with a FIXED gap -- the measured Base case.

    The level noise is deliberately tiny: this reproduces a STABLECOIN pair, whose price
    barely moves, and that is what leaves the control unable to discriminate. A fixture
    with an ETH-sized level would give scrambling something to add and would test the
    opposite regime.
    """
    observations = []
    rng = random.Random(4)
    for i in range(n):
        level = 1.0 + rng.gauss(0, 0.000002)
        observations.append(
            _obs(float(i * 20), cex=level, dex=level * (1 + offset_bps / 10_000))
        )
    return observations


def _time_varying(n=300):
    """A gap that genuinely moves, so its own spread is wide relative to its centre."""
    observations = []
    rng = random.Random(9)
    for i in range(n):
        level = 1.0 + rng.gauss(0, 0.0002)
        wobble = 1 + (rng.gauss(0, 30) / 10_000)
        observations.append(_obs(float(i * 20), cex=level, dex=level * wobble))
    return observations


class TestConstancyIsMeasuredDirectly:
    def test_a_fixed_gap_is_recognised_from_its_own_spread(self):
        result = scrambled_control(
            _constant_offset(), COSTS, NOTIONALS,
            base_is_token0=True, offset_seconds=1800.0,
        )
        assert result["constant_offset"] is True, (
            "a gap whose own p1-to-p99 range is a small fraction of its median is a "
            "fixed offset, not a market signal"
        )

    def test_the_verdict_explains_itself_with_the_numbers(self):
        result = scrambled_control(
            _constant_offset(), COSTS, NOTIONALS,
            base_is_token0=True, offset_seconds=1800.0,
        )
        assert result["reason"] is not None
        assert "fixed offset" in result["reason"].lower()

    def test_a_time_varying_gap_is_not_called_fixed(self):
        result = scrambled_control(
            _time_varying(), COSTS, NOTIONALS,
            base_is_token0=True, offset_seconds=1800.0,
        )
        assert result["constant_offset"] is False

    def test_one_observation_establishes_nothing(self):
        """A single reading has p1 == p99 == median by definition. Without a floor, every
        one-row sample would be declared a fixed offset -- the strongest conclusion from
        the weakest evidence, which the persistence classifier already got wrong once."""
        result = scrambled_control(
            [_obs(0.0, 1.0, 1.0)], COSTS, NOTIONALS,
            base_is_token0=True, offset_seconds=1800.0,
        )
        assert result["constant_offset"] is None


class TestTheControlReportsItsOwnPower:
    def test_a_static_level_leaves_the_control_powerless(self):
        """Scrambling can only add the venue's own movement. On a pair whose level barely
        moves there is nothing to add, so an unchanged distribution is a fact about the
        test rather than about the market."""
        result = scrambled_control(
            _constant_offset(), COSTS, NOTIONALS,
            base_is_token0=True, offset_seconds=1800.0,
        )
        assert result["control_has_power"] is False

    def test_the_noise_bound_is_still_reported(self):
        result = scrambled_control(
            _constant_offset(), COSTS, NOTIONALS,
            base_is_token0=True, offset_seconds=1800.0,
        )
        assert result["noise_bound_bps"] is not None
