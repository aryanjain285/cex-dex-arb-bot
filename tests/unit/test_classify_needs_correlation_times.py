"""Persistence is a claim about a population, so it needs a test, not a threshold.

The classifier called ETH/USDC 0.05% Arbitrum a STANDING BASIS at +2.6 bps with 0.0% sign
flips, on 236 observations. An hour later, with 1,713, the same market flipped sign in
48.7% of them. Both readings were accurate about their own samples, and the first was
still wrong.

It was wrong because a binary label hides its own sample-size dependence. That sample's
autocorrelation time was about 20 observations, so it held roughly eleven independent
draws -- and eleven draws all on one side has probability 0.001 under a symmetric null.
Suggestive, and nowhere near the certainty the label conveyed.

Two things were needed to fix it, and the first attempt at each was wrong in an
instructive way:

  A SIGN TEST ON THE EFFECTIVE SAMPLE, not a flip-fraction threshold. The p-value carries
  the sample size with it. Using the raw row count instead would make any persistent
  series overwhelmingly significant purely for having been sampled often -- the same
  inflation batch means exists to prevent.

  A FLOOR HIGH ENOUGH THAT THE AUTOCORRELATION ESTIMATE MEANS ANYTHING. tau is summed
  only to lag n/4, so on a short series it is censored downward, which inflates the
  effective sample size and therefore the significance. Measured on an AR(1) with
  phi=0.99, true tau about 115:

      n= 30   tau  2.92   p=0.002   spuriously significant
      n= 60   tau 17.84   refused: the autocorrelation has not decayed
      n=200   tau 34.19   p=0.063   correctly NOT significant
      n=450   tau 61.93   p=0.016

  At n=30 the sample reports MORE independent draws than at n=60, because it cannot see
  past lag 7. A guard built on correlation times alone is therefore self-defeating for
  precisely the samples it exists to catch.
"""
import random

import pytest

from src.research.report import (
    MIN_CORRELATION_TIMES_TO_CLASSIFY,
    MIN_OBSERVATIONS_TO_CLASSIFY,
    classify_dislocation,
)
from src.research.statistics import autocorrelation_converged, sign_test_p_value


def _persistent(n, level=3.0, phi=0.99, seed=5):
    """A slowly-varying series that stays one side of zero. Large tau by construction."""
    rng = random.Random(seed)
    values, x = [], 0.0
    for _ in range(n):
        x = phi * x + rng.gauss(0.0, 0.1)
        values.append(level + x)
    return values


def _alternating(n):
    return [3.0 if i % 2 == 0 else -3.0 for i in range(n)]


class TestTheRegressionItself:
    def test_a_sample_holding_about_eleven_draws_is_not_significant(self):
        """The exact failure. 200 observations of a strongly persistent series: every
        value one side of zero, and p=0.063 -- so NOT a standing basis."""
        result = classify_dislocation(_persistent(200))
        assert result["kind"] == "fluctuating", (
            f"called {result['kind']} at p={result['sign_test_p']}; a one-sided sample "
            f"holding ~6 independent draws does not establish persistence"
        )
        assert result["sign_test_p"] > 0.05

    def test_a_long_enough_sample_does_establish_it(self):
        result = classify_dislocation(_persistent(4000))
        assert result["kind"] == "standing_basis"
        assert result["sign_test_p"] < 0.001


class TestShortSamplesAreRefused:
    @pytest.mark.parametrize("n", [10, 30, 60, 99])
    def test_below_the_floor_nothing_is_classified(self, n):
        result = classify_dislocation(_persistent(n))
        assert result["kind"] == "unknown"
        assert result["reason"] is not None

    def test_the_censored_estimator_case_is_refused_not_believed(self):
        """n=30 previously produced p=0.002 -- more significant than n=60 -- because tau
        was censored to 2.92. It must not be classified at all."""
        result = classify_dislocation(_persistent(30))
        assert result["kind"] == "unknown"
        assert result["sign_test_p"] is None

    def test_a_series_that_has_not_decorrelated_is_refused(self):
        """A separate guard from the observation floor, and it has to be: 150 observations
        clears the count and still has a positive autocorrelation at every measurable lag,
        so the correlation time is below the sample's resolution and nothing derived from
        it can be trusted."""
        values = _persistent(150, phi=0.999)
        assert autocorrelation_converged(values) is False
        result = classify_dislocation(values)
        assert result["kind"] == "unknown"
        assert "decay" in result["reason"].lower(), result["reason"]


class TestFluctuatingIsRecognised:
    def test_an_alternating_series_is_fluctuating_with_no_significance(self):
        result = classify_dislocation(_alternating(400))
        assert result["kind"] == "fluctuating"
        assert result["sign_test_p"] == pytest.approx(1.0)

    def test_an_independent_one_sided_series_is_classified_from_a_short_sample(self):
        """A near-uncorrelated series has tau about 1, so 200 observations really are 200
        independent draws. Penalising it would make every real result unfindable, which is
        the failure mode opposite to the one being fixed."""
        rng = random.Random(11)
        values = [3.0 + rng.gauss(0, 0.1) for _ in range(200)]
        result = classify_dislocation(values)
        assert result["kind"] == "standing_basis"
        assert result["sign_test_p"] < 0.001


class TestTheSignTestItself:
    def test_it_uses_the_effective_sample_not_the_row_count(self):
        """Two series with the same one-sidedness and different autocorrelation must not
        get the same p-value. If they do, the test is counting rows."""
        rng = random.Random(3)
        independent = [3.0 + rng.gauss(0, 0.1) for _ in range(1000)]
        correlated = _persistent(1000, phi=0.995)
        assert sign_test_p_value(independent) < sign_test_p_value(correlated), (
            "a correlated series carries less evidence for the same apparent "
            "one-sidedness"
        )

    def test_a_balanced_series_is_not_significant(self):
        assert sign_test_p_value(_alternating(400)) == pytest.approx(1.0)

    def test_too_short_gives_no_p_value(self):
        assert sign_test_p_value([1.0, 2.0, 3.0]) is None


class TestTheBarrierClaimDependsOnPersistence:
    def test_no_barrier_is_claimed_when_persistence_is_unknown(self):
        result = classify_dislocation(_persistent(30, level=455.0))
        assert result["kind"] == "unknown"
        assert result["barrier_suspected"] is None

    def test_no_barrier_is_claimed_when_persistence_is_not_significant(self):
        """A large gap on a sample too weak to establish persistence is not a barrier
        claim, because the barrier argument rests entirely on the gap surviving."""
        result = classify_dislocation(_persistent(200, level=455.0))
        assert result["kind"] == "fluctuating"
        assert result["barrier_suspected"] is False

    def test_a_barrier_is_claimed_when_both_hold(self):
        result = classify_dislocation(_persistent(4000, level=455.0))
        assert result["kind"] == "standing_basis"
        assert result["barrier_suspected"] is True


class TestTheSettingsAreStated:
    def test_the_floor_is_high_enough_for_an_autocorrelation_estimate(self):
        assert MIN_OBSERVATIONS_TO_CLASSIFY >= 100

    def test_the_correlation_time_requirement_is_stated(self):
        assert MIN_CORRELATION_TIMES_TO_CLASSIFY >= 3
