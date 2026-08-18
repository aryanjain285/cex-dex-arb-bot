"""How fast an edge decays, in seconds, because that is what sets the latency budget.

An edge whose autocorrelation is gone by twelve seconds cannot be captured by a system
that settles on a twelve-second block, however good its cost model. So the decay has to
be measured, and measured in SECONDS rather than in observations -- the cadence differs
between runs and is itself measured, so "correlation 0.4 at lag 1" invites the reader to
supply their own cadence, and a 2s run and a 36s run then get compared as if they were
the same thing.

The trap this guards against is the one the project's earlier placebo fell into: asking
a 2-second question of data sampled every 36 seconds, and silently getting a 36-second
answer. A lag shorter than the cadence is unmeasurable and must say so.
"""
import random

import pytest

from src.research.statistics import decay_profile, half_life_seconds


def _ar1(n=4000, phi=0.9, seed=3):
    rng = random.Random(seed)
    values, x = [], 0.0
    for _ in range(n):
        x = phi * x + rng.gauss(0.0, 1.0)
        values.append(x)
    return values


class TestTheProfileIsInSeconds:
    def test_correlation_falls_with_lag(self):
        profile = decay_profile(_ar1(phi=0.9), cadence_seconds=1.0)
        values = [profile[lag] for lag in (2.0, 5.0, 12.0, 30.0) if profile[lag] is not None]
        assert values == sorted(values, reverse=True)

    def test_the_same_series_at_a_slower_cadence_decays_over_more_seconds(self):
        """Identical numbers, different sampling rate: the profile in seconds must
        stretch. If it did not, the function would be reporting lags in observations
        under a label that says seconds."""
        series = _ar1(phi=0.9)
        fast = decay_profile(series, cadence_seconds=1.0)
        slow = decay_profile(series, cadence_seconds=10.0)
        assert fast[30.0] < slow[30.0], (
            "at a 10s cadence, a 30s lag is 3 observations rather than 30, so the "
            "correlation must be HIGHER"
        )

    def test_a_lag_below_the_cadence_is_unmeasurable(self):
        profile = decay_profile(_ar1(), cadence_seconds=36.0)
        assert profile[2.0] is None
        assert profile[12.0] is None
        assert profile[60.0] is not None

    def test_a_non_positive_cadence_is_rejected(self):
        with pytest.raises(ValueError):
            decay_profile(_ar1(), cadence_seconds=0.0)


class TestHalfLife:
    def test_a_persistent_series_has_a_longer_half_life(self):
        slow = half_life_seconds(_ar1(phi=0.95), cadence_seconds=1.0)
        fast = half_life_seconds(_ar1(phi=0.5), cadence_seconds=1.0)
        assert slow is not None and fast is not None
        assert slow > fast

    def test_the_half_life_scales_with_the_cadence(self):
        series = _ar1(phi=0.9)
        assert (half_life_seconds(series, cadence_seconds=10.0)
                == pytest.approx(half_life_seconds(series, cadence_seconds=1.0) * 10))

    def test_an_undecaying_series_reports_none_not_a_huge_number(self):
        """A standing basis is a different phenomenon from a slow dislocation, and
        reporting it as a ten-hour half-life would invite exactly the wrong trade."""
        assert half_life_seconds([1.0] * 500, cadence_seconds=1.0) is None

    def test_it_is_not_quantised_to_the_sampling_interval(self):
        """Interpolated between bracketing lags: a half-life reported only in whole
        observations cannot distinguish 1.1s from 1.9s, which at a 2s cadence is the
        difference between catchable and not."""
        series = _ar1(phi=0.6)
        value = half_life_seconds(series, cadence_seconds=1.0)
        assert value is not None
        assert value != round(value), f"half life {value} looks quantised"
