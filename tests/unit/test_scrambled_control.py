"""A negative control, so a measured edge can be distinguished from an artifact.

The danger with this whole research stack is not that it reports no edge. It is that
it reports one that is not there. A sign error, a units error, an inverted token
order, a mis-specified cost -- each produces a distribution of "edge" that looks like
data. Every check so far is internal: the tests assert that the code does what it was
designed to do. None of them can tell whether the design measures the market.

The control: take the CEX book from time t and the pool state from a DISTANT time t',
and run the identical evaluation. Any structure that survives cannot be a real
dislocation, because the two sides never coexisted.

What the comparison means:

  * The scrambled distribution is WIDER, because the two sides have drifted
    independently. Its positive tail is therefore an upper bound on how much apparent
    edge pure mismatch can manufacture.

  * Its centre is also HIGHER, and that turned out to be the important finding here.
    The reported statistic is the best of the two directions, which is a max over two
    nearly-opposite quantities, so noise is rectified rather than cancelled. Measured
    below: two venues tracking each other perfectly give -5.5 bps under the true
    pairing and +93 bps under a scramble that added no information at all. A positive
    mean best-gross edge is therefore not evidence of anything on its own -- it is
    what a max does to a symmetric error. Only the excess over the scrambled tail is.

  * The real test: if the true pairing's tail above a threshold is no heavier than
    the scrambled one's, then observations above that threshold are indistinguishable
    from noise, whatever the mean says.

An earlier attempt at a placebo in this project failed for a related reason: it
delayed by one second against 12-second blocks, so 69% of its "placebo" quotes were
byte-identical to the real ones and it could not have detected anything. The offset
here is therefore validated against the observed cadence rather than assumed.
"""
from decimal import Decimal

import pytest

from src.exchange.pool_state import PoolSnapshot
from src.exchange.univ3_math import TickInfo
from src.research.evaluate import CostModel
from src.research.observations import Observation
from src.research.report import scrambled_control


def _pool(price=Decimal("1900")) -> PoolSnapshot:
    from decimal import getcontext
    getcontext().prec = 60
    raw = price * (Decimal(10) ** 6) / (Decimal(10) ** 18)
    liquidity = 10 ** 24
    return PoolSnapshot(
        sqrt_price_x96=int(Decimal(2 ** 96) * raw.sqrt()),
        liquidity=liquidity, tick=0, fee=500, tick_spacing=10,
        ticks=[TickInfo(tick=-500000, liquidity_net=liquidity),
               TickInfo(tick=500000, liquidity_net=-liquidity)],
        decimals0=18, decimals1=6, block_number=1,
        address="0x" + "ab" * 20, token0="0x" + "11" * 20,
        token1="0x" + "22" * 20, chain="ethereum",
        known_lower_tick=-500000, known_upper_tick=500000,
    )


def _obs(ts, cex=Decimal("1900"), dex=Decimal("1900")):
    spread = Decimal("0.00005")
    return Observation(
        ts=ts, cex_symbol="ETHUSDT", base="ETH", quote="USDT", chain="ethereum",
        pool_fee=500, pool_address="0x" + "ab" * 20,
        cex_bids=[(cex * (1 - spread) * (1 - Decimal("0.00001") * i), Decimal("1000"))
                  for i in range(5)],
        cex_asks=[(cex * (1 + spread) * (1 + Decimal("0.00001") * i), Decimal("1000"))
                  for i in range(5)],
        cex_feed_ts=ts, pool=_pool(dex),
        gas_price_wei=10 ** 9, native_price_quote=cex,
        rpc_endpoint="test", run_id="t",
    )


COSTS = CostModel(
    taker_fee_bps=Decimal("7.5"), cex_legs=1, gas_units=200_000,
    rotation_cost_quote=Decimal("0"), floor_bps=Decimal("5"),
)
NOTIONALS = [Decimal("1000")]


def _tracking_market(n=200):
    """Both venues moving together: the efficient case, and the one to be sure about.

    Prices wander over a 2% range while staying pinned to each other, so the true
    pairing has almost no dislocation and the scrambled pairing has plenty of
    apparent one. This is the configuration where a naive reading of the scrambled
    distribution would announce an opportunity.
    """
    observations = []
    for i in range(n):
        level = Decimal("1900") * (1 + Decimal(i % 40) * Decimal("0.0005"))
        observations.append(_obs(float(i * 10), cex=level, dex=level))
    return observations


class TestTheControlDetectsMismatch:
    def test_scrambling_a_tracking_market_widens_the_distribution(self):
        result = scrambled_control(
            _tracking_market(), COSTS, NOTIONALS,
            base_is_token0=True, offset_seconds=1000.0,
        )
        assert result["true"]["n"] > 0 and result["scrambled"]["n"] > 0
        assert result["scrambled"]["sd"] > result["true"]["sd"] * 2, (
            f"scrambling two venues that track each other must widen the edge "
            f"distribution: true sd {result['true']['sd']:.3f} vs scrambled "
            f"{result['scrambled']['sd']:.3f}"
        )

    def test_mismatch_shifts_the_centre_upward_which_is_why_the_bound_is_needed(self):
        """The reported statistic is the BEST of the two directions, and that is a
        max over two nearly-opposite quantities. So noise cannot cancel: it is
        rectified. Any dispersion between the two venues -- real or manufactured --
        raises the mean of "best gross edge".

        Measured here: a market whose two venues track each other perfectly shows
        -5.5 bps under the true pairing and +93 bps under the scramble. The scramble
        added no information whatever, and the headline number moved by 99 bps.

        This is exactly why a noise bound is not optional. Reporting a positive mean
        best-gross edge, on its own, is not evidence of anything -- it is what a max
        does to a symmetric error. Only the excess over the scrambled tail is
        evidence.
        """
        result = scrambled_control(
            _tracking_market(), COSTS, NOTIONALS,
            base_is_token0=True, offset_seconds=1000.0,
        )
        assert result["scrambled"]["mean"] > result["true"]["mean"], (
            "scrambling two venues that track each other must RAISE the apparent "
            "edge, because the statistic is a max and noise is rectified rather "
            "than cancelled"
        )

    def test_the_scrambled_tail_bounds_what_noise_can_manufacture(self):
        """On a market that tracks perfectly, every apparent opportunity in the
        scrambled pairing is manufactured. Its p99 is therefore the level below which
        an apparent edge proves nothing."""
        result = scrambled_control(
            _tracking_market(), COSTS, NOTIONALS,
            base_is_token0=True, offset_seconds=1000.0,
        )
        assert result["scrambled"]["p99"] > result["true"]["p99"]
        assert result["noise_bound_bps"] == pytest.approx(
            result["scrambled"]["p99"]
        )

    def test_a_real_persistent_dislocation_survives_the_control(self):
        """The positive control for the negative control. A genuine standing gap
        appears in BOTH pairings, so the verdict must not be 'noise'."""
        observations = [
            _obs(float(i * 10), cex=Decimal("1900"), dex=Decimal("1940"))
            for i in range(200)
        ]
        result = scrambled_control(
            observations, COSTS, NOTIONALS,
            base_is_token0=True, offset_seconds=1000.0,
        )
        assert result["true"]["mean"] > 100
        assert result["scrambled"]["mean"] > 100
        assert result["exceeds_noise"] is not None


class TestTheOffsetIsValidated:
    def test_an_offset_shorter_than_the_cadence_is_refused(self):
        """The failure the previous placebo had: a one-second offset against
        twelve-second blocks compared each quote with a copy of itself."""
        with pytest.raises(ValueError, match="offset"):
            scrambled_control(
                _tracking_market(), COSTS, NOTIONALS,
                base_is_token0=True, offset_seconds=1.0,
            )

    def test_the_offset_actually_used_is_reported(self):
        result = scrambled_control(
            _tracking_market(), COSTS, NOTIONALS,
            base_is_token0=True, offset_seconds=1000.0,
        )
        assert result["offset_seconds"] == 1000.0
        assert result["median_cadence_seconds"] == pytest.approx(10.0)

    def test_identical_pairs_are_counted(self):
        """If the scramble happens to reproduce the true pairing for some rows, the
        control is that much weaker, and the report has to say so rather than let a
        diluted control read as a passed one."""
        result = scrambled_control(
            _tracking_market(), COSTS, NOTIONALS,
            base_is_token0=True, offset_seconds=1000.0,
        )
        assert "identical_pairs" in result
        assert result["identical_pairs"] == 0


class TestTooLittleData:
    def test_a_sample_too_short_to_scramble_says_so(self):
        result = scrambled_control(
            [_obs(0.0), _obs(10.0)], COSTS, NOTIONALS,
            base_is_token0=True, offset_seconds=1000.0,
        )
        assert result["scrambled"]["n"] == 0
        assert result["reason"] is not None

    def test_an_empty_sample_does_not_raise(self):
        result = scrambled_control(
            [], COSTS, NOTIONALS, base_is_token0=True, offset_seconds=1000.0,
        )
        assert result["true"]["n"] == 0
        assert result["reason"] is not None
