"""The configured loop interval is not the cadence, and the gap was invisible.

`loop_interval_seconds: 0.2` reads as "we poll at 5 Hz". Measured from the audit
trail of a 517-second live run:

    median cycle time    2.32 s
    p90                 10.7 – 12.5 s
    max                 31.7 – 39.7 s

So the real cadence is 0.43 Hz -- twelve times slower than the configured
interval, and highly variable. The cause is RPC latency: a single
quoteExactInputSingle takes 0.31 s (Ethereum median), 0.78 s (Arbitrum) or 0.83 s
(Base), and the detector issues two per pair.

Nothing in the system said so. `loop_interval_seconds` is the idle SLEEP between
cycles, not the period, and every intuition of the form "5 cycles is about a
second" is wrong by that same factor of twelve. The first version of the placebo
arm was built on exactly that intuition and measured nothing as a result.

So the loop now compares intended cadence against observed and says when they
diverge. A number that reads like a fact and is off by 12x is worse than an absent
one.
"""
import pytest

from src.infra.cadence import CadenceWatch


class FakeClock:
    def __init__(self, t=0.0):
        self.t = t

    def __call__(self):
        return self.t

    def advance(self, seconds):
        self.t += seconds


def test_a_cycle_matching_the_interval_says_nothing():
    """No warning when the configured cadence is real. A watch that always
    complained would be ignored within a day."""
    watch = CadenceWatch(interval_seconds=0.2, tolerance_factor=3.0)

    for _ in range(50):
        assert watch.observe(0.25) is None


def test_a_slow_cycle_is_reported():
    watch = CadenceWatch(interval_seconds=0.2, tolerance_factor=3.0)

    message = None
    for _ in range(watch.sample_size):
        message = watch.observe(2.32)

    assert message is not None
    assert "2.3" in message
    assert "0.2" in message


def test_the_message_states_the_factor_and_the_real_rate():
    """An operator needs both: the factor says how wrong the config is, the rate
    says what they actually have."""
    watch = CadenceWatch(interval_seconds=0.2, tolerance_factor=3.0)

    message = None
    for _ in range(watch.sample_size):
        message = watch.observe(2.32)

    assert "12" in message  # 2.32 / 0.2
    assert "Hz" in message


def test_it_waits_for_a_sample_before_judging():
    """One slow cycle is noise -- a cold cache, a reconnect. The complaint has to
    be about the distribution, or it fires on startup every time."""
    watch = CadenceWatch(interval_seconds=0.2, tolerance_factor=3.0)

    assert watch.observe(5.0) is None, "a single slow cycle must not warn"


def test_it_uses_the_median_not_the_mean():
    """One 40-second stall in a run of fast cycles is a tail event, not a cadence
    problem, and a mean would report it as one."""
    watch = CadenceWatch(interval_seconds=0.2, tolerance_factor=3.0)

    message = None
    for _ in range(watch.sample_size - 1):
        message = watch.observe(0.25)
    message = watch.observe(40.0)

    assert message is None, "a single outlier moved the verdict"


def test_it_does_not_repeat_itself_every_cycle():
    """A warning printed 400 times an hour is noise that trains an operator to
    ignore the one that matters."""
    watch = CadenceWatch(interval_seconds=0.2, tolerance_factor=3.0,
                         repeat_seconds=60.0, now_fn=(clock := FakeClock()))

    messages = []
    # The clock advances only a little per cycle, so all 80 observations fall
    # inside one 60-second repeat window. Advancing by the cycle time instead
    # would span three windows and three warnings would be correct.
    for _ in range(watch.sample_size * 4):
        result = watch.observe(2.32)
        if result:
            messages.append(result)
        clock.advance(0.5)

    assert len(messages) == 1, f"warned {len(messages)} times inside one window"


def test_it_warns_again_after_the_repeat_window():
    clock = FakeClock()
    watch = CadenceWatch(interval_seconds=0.2, tolerance_factor=3.0,
                         repeat_seconds=60.0, now_fn=clock)

    messages = []
    for _ in range(watch.sample_size * 2):
        result = watch.observe(2.32)
        if result:
            messages.append(result)
        clock.advance(40.0)

    assert len(messages) >= 2


def test_the_window_is_bounded():
    """This runs for weeks."""
    watch = CadenceWatch(interval_seconds=0.2, tolerance_factor=3.0)

    for _ in range(10_000):
        watch.observe(0.25)

    assert watch.sample_count() <= watch.sample_size


def test_a_non_positive_interval_is_rejected():
    with pytest.raises(ValueError):
        CadenceWatch(interval_seconds=0)


def test_a_tolerance_below_one_is_rejected():
    """A factor under 1 would demand the cycle be FASTER than the sleep between
    cycles, which is impossible, so it would warn forever."""
    with pytest.raises(ValueError):
        CadenceWatch(interval_seconds=0.2, tolerance_factor=0.5)
