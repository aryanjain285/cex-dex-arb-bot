"""A standing basis is not a small opportunity. It is a different thing entirely.

The recorded data forced this distinction. ETH/USDC 0.05% on Arbitrum shows a mean
dislocation of +2.79 bps against Binance -- and a 1st percentile of +0.29 bps. The
pool is above the exchange essentially always, not on average.

Those two readings support opposite conclusions:

  FLUCTUATING. The sign changes. The pool crosses the exchange price in both
  directions, so a taker can buy on whichever venue is cheap and sell on the other,
  and inventory returns on its own over time. Size is bounded by depth and the
  constraint is cost.

  STANDING. The sign does not change. The pool is persistently richer, which is a
  price, not an error: it is what the market charges for the asset being on that chain
  rather than in that custodian. Trading it once captures the gap; trading it again
  requires moving inventory back across the bridge, and the bridge costs the basis --
  that is why the basis exists. So the "opportunity" is a one-off inventory
  repositioning, and reported as a per-trade edge it would be counted many times over.

The failure mode this guards is specific and expensive: a report showing "+3 bps of
dislocation, 100% of the time, on both ETH pairs" reads as a highly reliable signal.
It is the opposite -- it is a signal that cannot be harvested repeatedly. The fraction
of observations where the sign flips is what separates them, so the report states it.

The samples here are deliberately large. Persistence is established by a sign test on the
EFFECTIVE sample, so a short series cannot support the claim whatever its values -- see
test_classify_needs_correlation_times, where that is the subject. These tests supply
enough sample to isolate the behaviour they are about.
"""
import pytest

from src.research.report import classify_dislocation


class TestStandingBasis:
    def test_a_persistently_positive_dislocation_is_a_basis(self):
        result = classify_dislocation([2.5, 2.8, 3.1, 2.9, 3.4, 2.7] * 200)
        assert result["kind"] == "standing_basis"
        assert result["sign_flip_fraction"] == 0.0

    def test_the_measured_arbitrum_case_is_a_basis(self):
        """Reproduced from the recorded distribution: mean +2.79, p1 +0.29."""
        values = [0.29 + (i % 40) * 0.05 for i in range(800)]
        result = classify_dislocation(values)
        assert result["kind"] == "standing_basis"

    def test_a_persistently_negative_dislocation_is_also_a_basis(self):
        """Sign does not matter -- persistence does. USDC/USDT on Arbitrum sits at
        -0.71 bps with a p99 of -0.71, which is equally unharvestable."""
        result = classify_dislocation([-0.71, -0.75, -0.68, -0.81] * 300)
        assert result["kind"] == "standing_basis"

    def test_a_basis_reports_its_magnitude_as_the_bridge_price(self):
        result = classify_dislocation([3.0] * 400 + [3.0] * 400)
        assert result["median_bps"] == pytest.approx(3.0)


class TestFluctuating:
    def test_a_sign_changing_dislocation_is_fluctuating(self):
        result = classify_dislocation([2.0, -2.0] * 400)
        assert result["kind"] == "fluctuating"
        assert result["sign_flip_fraction"] > 0.4

    def test_a_mostly_positive_series_that_does_cross_is_fluctuating(self):
        """The interesting middle case: a real basis WITH real fluctuation around it.
        Harvestable, because the sign does cross -- just less often."""
        values = ([3.0] * 8 + [-1.0] * 2) * 80
        result = classify_dislocation(values)
        assert result["kind"] == "fluctuating"
        assert 0.1 < result["sign_flip_fraction"] < 0.3


class TestHonestEdges:
    def test_an_empty_series_is_unclassified(self):
        result = classify_dislocation([])
        assert result["kind"] == "unknown"
        assert result["sign_flip_fraction"] is None

    def test_a_series_too_short_to_judge_says_so(self):
        """Three observations that happen to share a sign are not evidence of a
        standing basis, and calling them one would be the strongest possible
        conclusion from the weakest possible sample."""
        result = classify_dislocation([1.0, 1.1, 1.2])
        assert result["kind"] == "unknown"

    def test_the_threshold_is_reported_so_the_verdict_can_be_checked(self):
        result = classify_dislocation([2.0, -2.0] * 400)
        assert result["flip_threshold"] is not None

    def test_a_series_straddling_zero_by_a_hair_is_not_a_basis(self):
        """Values on both sides of zero, all tiny. Not a basis, and not an
        opportunity either -- the magnitude decides that separately."""
        result = classify_dislocation([0.05, -0.04, 0.03, -0.06] * 300)
        assert result["kind"] == "fluctuating"
        assert abs(result["median_bps"]) < 0.1
