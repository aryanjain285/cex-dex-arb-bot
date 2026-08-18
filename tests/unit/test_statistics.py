"""Statistics that cannot manufacture significance from correlated samples.

This is the module most able to produce a confident wrong answer, so its tests are
written against the specific ways that happens.

THE CENTRAL PROBLEM. Observations five seconds apart are not independent draws.
A price dislocation persists for tens of seconds, so a hundred thousand
observations may contain a few thousand independent facts. The standard error of a
mean is sigma/sqrt(n), so using the raw count inflates every t-statistic by
sqrt(n/n_eff) -- which for a strongly autocorrelated series is a factor of five or
more. That is not a subtle bias: it turns "the mean dislocation is -1.5 bps, and we
cannot rule out zero" into "the mean dislocation is -1.5 bps, p < 0.001". Sampling
FASTER would make the effect stronger, so the error rewards exactly the wrong
instinct.

The fix is batch means: split the series into contiguous batches long enough that
each batch mean is roughly independent of the next, and do the inference on those.
The effective sample size is then the number of batches, not the number of
observations, and it falls as autocorrelation rises.

The tests below pin that behaviour from both ends -- an independent series must not
be penalised, and a correlated one must be -- because a conservative-in-all-cases
implementation would pass a one-sided test while making every real result
unfindable.
"""
import math
import random

import pytest

from src.research.statistics import (
    autocorrelation,
    batch_means_interval,
    describe,
    effective_sample_size,
    exceedance,
    integrated_autocorrelation_time,
    persistence_runs,
)


def _iid(n=4000, mean=0.0, sd=1.0, seed=7):
    rng = random.Random(seed)
    return [rng.gauss(mean, sd) for _ in range(n)]


def _ar1(n=4000, phi=0.95, mean=0.0, sd=1.0, seed=11):
    """A first-order autoregressive series: this is what a persistent price
    dislocation sampled on a fixed clock actually looks like."""
    rng = random.Random(seed)
    values, x = [], 0.0
    for _ in range(n):
        x = phi * x + rng.gauss(0.0, sd)
        values.append(mean + x)
    return values


class TestDescribe:
    def test_it_reports_the_shape_not_just_the_mean(self):
        stats = describe(_iid(2000, mean=5.0, sd=2.0))
        assert stats["n"] == 2000
        assert abs(stats["mean"] - 5.0) < 0.2
        assert abs(stats["sd"] - 2.0) < 0.2
        assert stats["p50"] is not None
        assert stats["p90"] > stats["p50"] > stats["p10"]
        assert stats["max"] >= stats["p99"]

    def test_an_empty_sample_reports_n_zero_rather_than_raising(self):
        """A pair with no priceable observations is a real outcome, and the report
        must be able to print it next to the pairs that had some."""
        stats = describe([])
        assert stats["n"] == 0
        assert stats["mean"] is None

    def test_a_single_observation_has_no_standard_deviation(self):
        stats = describe([1.0])
        assert stats["n"] == 1
        assert stats["sd"] is None, (
            "reporting 0.0 would imply certainty from one sample"
        )


class TestExceedance:
    def test_it_counts_the_fraction_above_each_threshold(self):
        values = [1.0, 5.0, 10.0, 20.0, 50.0]
        result = exceedance(values, [0.0, 10.0, 100.0])
        assert result[0.0] == 1.0
        assert result[10.0] == pytest.approx(0.4)  # 20 and 50, strictly above
        assert result[100.0] == 0.0

    def test_the_comparison_is_strict(self):
        """`>=` would count an observation exactly at the floor as tradeable. The
        floor is the minimum acceptable edge, so equality is not a trade."""
        assert exceedance([5.0], [5.0])[5.0] == 0.0

    def test_an_empty_sample_gives_no_fractions(self):
        assert exceedance([], [5.0])[5.0] is None


class TestAutocorrelation:
    def test_an_independent_series_has_near_zero_lag_one_correlation(self):
        assert abs(autocorrelation(_iid(), lag=1)) < 0.05

    def test_a_persistent_series_has_high_lag_one_correlation(self):
        assert autocorrelation(_ar1(phi=0.95), lag=1) > 0.85

    def test_correlation_decays_with_lag(self):
        series = _ar1(phi=0.9)
        assert (autocorrelation(series, lag=1)
                > autocorrelation(series, lag=5)
                > autocorrelation(series, lag=20))

    def test_a_lag_longer_than_the_series_is_undefined_not_zero(self):
        assert autocorrelation([1.0, 2.0, 3.0], lag=10) is None


class TestEffectiveSampleSize:
    def test_an_independent_series_keeps_almost_all_of_its_samples(self):
        n = 4000
        ess = effective_sample_size(_iid(n))
        assert ess > n * 0.5, (
            f"an iid series was penalised down to {ess} of {n}; a method that "
            f"discounts everything makes real results unfindable"
        )

    def test_a_persistent_series_loses_most_of_its_samples(self):
        n = 4000
        ess = effective_sample_size(_ar1(n, phi=0.95))
        assert ess < n * 0.2, (
            f"a strongly autocorrelated series kept {ess} of {n} effective "
            f"samples; this is the error that turns 'cannot rule out zero' into "
            f"'p < 0.001'"
        )

    def test_more_persistence_means_fewer_effective_samples(self):
        assert (effective_sample_size(_ar1(phi=0.98))
                < effective_sample_size(_ar1(phi=0.80)))

    def test_the_effective_size_never_exceeds_the_real_one(self):
        for series in (_iid(500), _ar1(500), [1.0] * 10):
            assert effective_sample_size(series) <= len(series)

    def test_it_is_at_least_one_for_a_non_empty_series(self):
        assert effective_sample_size([1.0, 1.0, 1.0]) >= 1

    def test_the_autocorrelation_time_rises_with_persistence(self):
        assert (integrated_autocorrelation_time(_ar1(phi=0.95))
                > integrated_autocorrelation_time(_ar1(phi=0.50)))


class TestBatchMeansInterval:
    def test_an_independent_series_gets_a_narrow_interval(self):
        result = batch_means_interval(_iid(4000, mean=1.0, sd=1.0))
        assert result["mean"] == pytest.approx(1.0, abs=0.1)
        width = result["upper"] - result["lower"]
        # sigma/sqrt(n) = 1/63 = 0.016; a 95% interval is about 4x that.
        assert 0.02 < width < 0.25, f"interval width {width} is implausible"

    def test_a_correlated_series_gets_a_wider_interval_than_the_naive_one(self):
        """The whole point. Same n, same marginal variance, far less information."""
        series = _ar1(4000, phi=0.95, sd=1.0)
        result = batch_means_interval(series)
        naive_half_width = 1.96 * (result["sd"] / math.sqrt(len(series)))
        actual_half_width = (result["upper"] - result["lower"]) / 2
        assert actual_half_width > naive_half_width * 2, (
            f"batch means gave a half-width of {actual_half_width:.4f} against a "
            f"naive {naive_half_width:.4f}; it is not correcting for persistence"
        )

    def test_the_interval_contains_the_true_mean_for_a_correlated_series(self):
        """A correction that merely widens without staying centred would be safe and
        useless."""
        result = batch_means_interval(_ar1(6000, phi=0.9, mean=3.0))
        assert result["lower"] < 3.0 < result["upper"]

    def test_it_reports_the_effective_sample_size_it_used(self):
        result = batch_means_interval(_ar1(4000, phi=0.95))
        assert result["effective_n"] < 4000
        assert result["batches"] >= 2

    def test_too_few_samples_gives_no_interval_rather_than_a_wrong_one(self):
        result = batch_means_interval([1.0, 2.0])
        assert result["lower"] is None and result["upper"] is None
        assert result["reason"] is not None

    def test_a_zero_variance_series_has_a_degenerate_but_valid_interval(self):
        """Every observation identical is a real case -- a pool that did not trade.
        It must not divide by zero."""
        result = batch_means_interval([2.5] * 200)
        assert result["mean"] == 2.5
        assert result["lower"] == result["upper"] == 2.5

    def test_significance_is_reported_against_zero(self):
        """The question the strategy actually asks: is the mean edge above zero,
        allowing for persistence?"""
        positive = batch_means_interval(_iid(4000, mean=1.0, sd=0.5))
        assert positive["excludes_zero"] is True
        centred = batch_means_interval(_iid(4000, mean=0.0, sd=1.0))
        assert centred["excludes_zero"] is False


class TestPersistenceRuns:
    def test_it_measures_how_long_a_condition_holds(self):
        # Above 0 at indices 1-3 and 6-7: runs of 3 and 2.
        flags = [False, True, True, True, False, False, True, True, False]
        runs = persistence_runs(flags)
        assert runs == [3, 2]

    def test_a_run_still_open_at_the_end_is_counted(self):
        """Dropping it would bias lifetimes downward, and the longest opportunity in
        a sample is often the one still open when recording stopped."""
        assert persistence_runs([False, True, True]) == [2]

    def test_no_occurrences_gives_no_runs(self):
        assert persistence_runs([False, False]) == []

    def test_every_sample_true_is_one_long_run(self):
        assert persistence_runs([True] * 5) == [5]
