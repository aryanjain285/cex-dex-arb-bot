"""A placebo arm for the edge measurement.

The methodological objection this answers: markout computed from later rows
re-samples *both* venues, so it measures whether the detector would still fire,
not whether the trade was worth anything. If the entire apparent edge were a
stale CEX book, the decay curve would look exactly like a real, decaying
arbitrage. Nothing in the data would distinguish them.

The control is cheap and direct. Alongside each live evaluation, evaluate the
same CEX book against a DEX quote from N seconds ago. That is deliberately a
*worse* input, so under the null hypothesis -- that the measured edge is an
artefact of data staleness rather than genuine mispricing -- the placebo
distribution matches the live one. If the two diverge, the live edge contains
something a deliberate delay does not explain.

No extra network calls: the delayed quote is one already fetched and kept.

WHY THE DELAY IS IN SECONDS

The first version delayed by a count of detection cycles -- 5 cycles at a 0.2s
loop, about one second. Run live for 103 seconds against Ethereum, it produced 94
paired observations whose live and placebo values were IDENTICAL in 69% of cases,
with a median difference of 0.00 bps. That looks like decisive support for the
null and is in fact evidence of nothing.

A Uniswap v3 quote changes only when a block lands. On Ethereum that is about 12
seconds, so every quote taken inside the same block is the same number by
construction, and a one-second delay compares a quote to itself. The control was
measuring the block interval, not the market.

So: the delay is expressed in seconds, and `min_delay_seconds_for` derives the
floor from the block time of the slowest chain being quoted. The config validator
enforces it. A placebo shorter than a block cannot answer the question it exists
to answer, and it fails in the direction that produces a comforting result.
"""
from __future__ import annotations

from collections import deque
from decimal import Decimal
from typing import Callable, Deque, Dict, Iterable, Optional, Tuple

from ..core import clock

__all__ = [
    "DelayedQuoteBuffer",
    "CHAIN_BLOCK_SECONDS",
    "DEFAULT_BLOCK_SECONDS",
    "min_delay_seconds_for",
]

# Nominal block intervals, in seconds. Approximate on purpose: they set a
# lower bound on a measurement delay, so being a little high is harmless and
# being low defeats the control.
CHAIN_BLOCK_SECONDS: Dict[str, float] = {
    "ethereum": 12.0,   # fixed 12s slots since the merge
    "arbitrum": 0.25,   # ~250ms, and faster under load
    "base": 2.0,        # 2s, moving toward 1s
    "optimism": 2.0,
    "polygon": 2.0,
    "bsc": 3.0,
}

# For a chain not in the table. Pessimistic: an unknown chain must not be able to
# lower the floor, because the consequence of too short a delay is a control that
# silently confirms whatever it is pointed at.
DEFAULT_BLOCK_SECONDS = 12.0

# How many block intervals the delay must span. Two rather than one: samples
# exactly one interval apart can still land in the same block, and the point of
# the control is that the two observations are genuinely different views.
BLOCK_MULTIPLE = 2.0


def min_delay_seconds_for(chains: Iterable[str]) -> float:
    """The smallest placebo delay that can distinguish two DEX observations.

    Driven by the SLOWEST chain being quoted: a delay that works for Arbitrum's
    250ms blocks tells you nothing about an Ethereum pool in the same run.
    """
    intervals = [
        CHAIN_BLOCK_SECONDS.get(str(chain).lower(), DEFAULT_BLOCK_SECONDS)
        for chain in chains
    ]
    # No chains configured yet is not a reason to relax the floor.
    slowest = max(intervals) if intervals else DEFAULT_BLOCK_SECONDS
    return slowest * BLOCK_MULTIPLE


class DelayedQuoteBuffer:
    """Per (pair, side) history returning the newest quote at least N seconds old.

    Bounded by time: entries older than the delay plus one serving window are
    discarded. This runs in a hot loop for weeks, and at a 0.2s interval a 30s
    window is 150 entries per series, so the bound is enforced rather than
    assumed.
    """

    def __init__(
        self,
        delay_seconds: float,
        now_fn: Optional[Callable[[], float]] = None,
    ):
        # Late-bound on purpose. `now_fn=clock.now` as a default argument would
        # capture the function at import time, so the process's single time source
        # could not be substituted afterwards -- and the detector constructs this
        # object itself, leaving a test no other way in.
        if delay_seconds <= 0:
            raise ValueError(
                f"delay_seconds must be positive, got {delay_seconds}. A zero "
                f"delay is not a placebo -- it is the live arm."
            )
        self.delay_seconds = float(delay_seconds)
        self._now_fn = now_fn
        self._series: Dict[Tuple[str, str], Deque[Tuple[float, Decimal]]] = {}

    def _now(self) -> float:
        return clock.now() if self._now_fn is None else self._now_fn()

    def push(self, pair_symbol: str, side: str, price: Decimal) -> None:
        """Record the live quote, and drop history no longer needed."""
        key = (pair_symbol, side)
        series = self._series.get(key)
        if series is None:
            series = deque()
            self._series[key] = series
        now = self._now()
        series.append((now, price))

        # Keep one entry older than the delay -- that is the one being served --
        # and discard anything older than that.
        cutoff = now - self.delay_seconds
        while len(series) >= 2 and series[1][0] <= cutoff:
            series.popleft()

    def delayed(self, pair_symbol: str, side: str) -> Optional[Decimal]:
        """The newest quote at least `delay_seconds` old, or None.

        Returning None rather than the most recent value matters: substituting
        the live quote would make the placebo silently identical to the live arm,
        and the control would read as confirming the edge.

        The NEWEST eligible entry, not the oldest: with a 30s delay and quotes at
        0, 10 and 50 seconds, the answer at t=50 is the quote from t=10. Serving
        the t=0 quote would be a 50-second delay wearing a 30-second label.
        """
        series = self._series.get((pair_symbol, side))
        if not series:
            return None
        cutoff = self._now() - self.delay_seconds
        chosen: Optional[Decimal] = None
        for timestamp, price in series:
            if timestamp <= cutoff:
                chosen = price
            else:
                break
        return chosen

    def size(self, pair_symbol: str, side: str) -> int:
        series = self._series.get((pair_symbol, side))
        return 0 if series is None else len(series)
