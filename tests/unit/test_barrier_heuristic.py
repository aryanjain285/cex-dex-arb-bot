"""A large gap that never closes is evidence of a barrier, not of an opportunity.

This inverts how such a reading should be interpreted, and the inversion is the point.

BNB/WETH on Ethereum measured +455 bps against Binance, standing -- the sign never
changed across every observation. The pool had real active liquidity, and the price ratio
was 1.046, comfortably inside the plausible-same-asset band. So neither the liquidity
guard nor the collision guard fires, and the market appeared as the single best
opportunity in the dataset: the only one of twelve deep pools whose dislocation cleared
its cost floor.

It is not an opportunity. `0xB8c77482e45F1F44dE1745F52C74426C631bDD52` is the LEGACY BNB
ERC-20 on Ethereum; BNB has been native to BSC since 2019, and Binance does not support
withdrawing BNB to Ethereum as that token. There is no settlement path, so the gap is the
price of a stranded asset -- something you can observe and cannot capture.

The general principle is stronger than the specific case, and does not require knowing
about BNB: IF A 455 BPS GAP ON A LIQUID ASSET WERE ARBITRAGEABLE, IT WOULD NOT PERSIST.
Someone faster and cheaper would have taken it within blocks. Persistence at that
magnitude is therefore information about a barrier.

So the classifier flags it, and flags in the right direction. Confirming which barrier
needs Binance's signed withdrawal-network endpoint, which a read-only research process
deliberately cannot reach -- so this raises the question rather than answering it, which
is still the opposite of what "YES, clears the floor" did.
"""
import pytest

from src.research.report import BARRIER_SUSPECTED_BPS, classify_dislocation


class TestTheMeasuredCase:
    def test_the_bnb_reading_is_flagged_as_a_barrier(self):
        """+455 bps, sign never changes."""
        result = classify_dislocation([455.0 + (i % 7) * 0.8 for i in range(60)])
        assert result["kind"] == "standing_basis"
        assert result["barrier_suspected"] is True

    def test_it_would_otherwise_have_been_the_best_market_in_the_dataset(self):
        """The counterfactual that makes this worth encoding: on magnitude alone it beats
        every genuine market by an order of magnitude."""
        barrier = classify_dislocation([455.0] * 60)
        genuine = classify_dislocation([26.1] * 60)
        assert abs(barrier["median_bps"]) > abs(genuine["median_bps"]) * 10
        assert barrier["barrier_suspected"] is True


class TestItDoesNotFireOnRealMarkets:
    def test_a_small_standing_basis_is_not_a_barrier(self):
        """+2.6 bps on ETH/USDC Arbitrum is a settlement basis and an ordinary one. It is
        unharvestable because it is small and one-sided, which the standing-basis label
        already says -- adding a barrier claim would overstate it."""
        result = classify_dislocation([2.6 + (i % 5) * 0.1 for i in range(60)])
        assert result["kind"] == "standing_basis"
        assert result["barrier_suspected"] is False

    def test_the_link_reading_is_not_flagged(self):
        """LINK/WETH at -26 bps standing: below its floor, and well below the barrier
        threshold. A real basis, no barrier claim."""
        result = classify_dislocation([-26.1 - (i % 4) * 0.5 for i in range(60)])
        assert result["kind"] == "standing_basis"
        assert result["barrier_suspected"] is False

    def test_a_large_but_FLUCTUATING_gap_is_not_a_barrier(self):
        """The distinction that carries the whole argument. A large gap whose sign
        changes IS being closed, repeatedly -- that is what a fluctuating sign means. It
        is the persistence, not the size, that implies a barrier."""
        result = classify_dislocation([300.0, -300.0] * 40)
        assert result["kind"] == "fluctuating"
        assert result["barrier_suspected"] is False


class TestTheThresholdIsExplicit:
    def test_the_default_is_well_beyond_competitive_arbitrage(self):
        assert BARRIER_SUSPECTED_BPS >= 50

    def test_it_is_reported_so_the_verdict_can_be_checked(self):
        result = classify_dislocation([455.0] * 60)
        assert result["barrier_threshold_bps"] == BARRIER_SUSPECTED_BPS

    def test_it_is_configurable(self):
        values = [150.0] * 60
        assert classify_dislocation(values, barrier_bps=100.0)["barrier_suspected"] is True
        assert classify_dislocation(values, barrier_bps=200.0)["barrier_suspected"] is False

    def test_the_sign_of_the_gap_does_not_matter(self):
        """A persistent discount is as unsettleable as a persistent premium."""
        assert classify_dislocation([-455.0] * 60)["barrier_suspected"] is True


class TestTooLittleData:
    def test_an_unclassifiable_sample_makes_no_barrier_claim(self):
        result = classify_dislocation([455.0, 456.0, 454.0])
        assert result["kind"] == "unknown"
        assert result["barrier_suspected"] is None, (
            "None, not False: three readings cannot rule a barrier out any more than "
            "they can establish one"
        )
