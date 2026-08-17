"""Saying so when the configured cadence is not the real one.

`strategy.loop_interval_seconds: 0.2` reads as "we poll at 5 Hz". Measured from
the audit trail of a 517-second live run:

    median cycle time    2.32 s
    p90                 10.7 - 12.5 s
    max                 31.7 - 39.7 s

The real cadence is 0.43 Hz: twelve times slower than the configured interval, and
highly variable. The cause is RPC latency -- a single quoteExactInputSingle takes
0.31 s on Ethereum, 0.78 s on Arbitrum, 0.83 s on Base at the median, and the
detector issues two per pair.

Nothing said so. `loop_interval_seconds` is the idle SLEEP between cycles, not the
period, and it is easy to read as the period. Every intuition of the form "five
cycles is about a second" is then wrong by that same factor of twelve -- and the
first version of the placebo arm was built on exactly that intuition, which is why
it measured nothing.

A number that reads like a fact and is wrong by 12x is worse than an absent one, so
the loop now compares intended against observed and reports the gap.

Three properties keep it useful rather than noisy:

  * it judges the MEDIAN of a sample, not a single cycle -- one 40-second stall is
    a tail event, and a mean would report it as a cadence problem;
  * it waits for a full sample before judging, so it does not fire on a cold start;
  * it repeats at most once per window, because a warning printed 400 times an
    hour trains an operator to ignore the one that matters.
"""
from __future__ import annotations

import statistics
from collections import deque
from typing import Callable, Deque, Optional

from ..core import clock

__all__ = ["CadenceWatch"]


class CadenceWatch:
    def __init__(
        self,
        interval_seconds: float,
        tolerance_factor: float = 3.0,
        sample_size: int = 20,
        repeat_seconds: float = 300.0,
        now_fn: Optional[Callable[[], float]] = None,
    ):
        if interval_seconds <= 0:
            raise ValueError(
                f"interval_seconds must be positive, got {interval_seconds}"
            )
        if tolerance_factor < 1:
            raise ValueError(
                f"tolerance_factor must be at least 1, got {tolerance_factor}. "
                f"Below 1 it would demand the cycle be faster than the sleep "
                f"between cycles, which is impossible, so it would warn forever."
            )
        if sample_size < 2:
            raise ValueError("sample_size must be at least 2")

        self.interval_seconds = float(interval_seconds)
        self.tolerance_factor = float(tolerance_factor)
        self.sample_size = int(sample_size)
        self.repeat_seconds = float(repeat_seconds)
        self._now_fn = now_fn
        self._samples: Deque[float] = deque(maxlen=self.sample_size)
        self._last_warned: Optional[float] = None

    def _now(self) -> float:
        return clock.monotonic() if self._now_fn is None else self._now_fn()

    def sample_count(self) -> int:
        return len(self._samples)

    def observe(self, cycle_seconds: float) -> Optional[str]:
        """Record one cycle. Returns a message when the gap is worth reporting."""
        self._samples.append(float(cycle_seconds))
        if len(self._samples) < self.sample_size:
            # Not enough to judge a distribution. Warning on a cold start would
            # fire on every restart and mean nothing.
            return None

        median = statistics.median(sorted(self._samples))
        if median <= self.interval_seconds * self.tolerance_factor:
            return None

        now = self._now()
        if self._last_warned is not None and now - self._last_warned < self.repeat_seconds:
            return None
        self._last_warned = now

        factor = median / self.interval_seconds
        return (
            f"Real cadence is {median:.2f}s per cycle -- {factor:.0f}x the "
            f"configured loop_interval_seconds of {self.interval_seconds:g}s. "
            f"The detector is polling at {1 / median:.2f} Hz, not "
            f"{1 / self.interval_seconds:.0f} Hz. loop_interval_seconds is the "
            f"idle sleep between cycles, not the period: the period is set by RPC "
            f"latency, and any reasoning in units of cycles is wrong by this "
            f"factor. A dedicated node is the fix; the configuration is not."
        )
