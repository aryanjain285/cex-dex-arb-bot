"""A placebo arm for the edge measurement.

The methodological objection this answers: markout computed from later rows
re-samples *both* venues, so it measures whether the detector would still fire,
not whether the trade was worth anything. If the entire apparent edge were a
stale CEX book, the decay curve would look exactly like a real, decaying
arbitrage. Nothing in the data would distinguish them.

The control is cheap and direct. Alongside each live evaluation, evaluate the
same CEX book against a DEX quote from N cycles ago. That is deliberately a
*worse* input, so under the null hypothesis -- that the measured edge is an
artefact of data staleness rather than genuine mispricing -- the placebo
distribution matches the live one. If the two diverge, the live edge contains
something a deliberate delay does not explain.

No extra network calls: the delayed quote is one already fetched and kept.
"""
from __future__ import annotations

from collections import deque
from decimal import Decimal
from typing import Deque, Dict, Optional, Tuple

__all__ = ["DelayedQuoteBuffer"]

_Key = Tuple[str, str]


class DelayedQuoteBuffer:
    """Per (pair, side) ring buffer returning the quote from N cycles ago.

    Bounded by construction. This runs in a hot loop for weeks, so keeping full
    history would be a leak; only `delay_cycles + 1` entries are retained,
    which is the minimum needed to serve the delayed value.
    """

    def __init__(self, delay_cycles: int):
        if delay_cycles < 1:
            raise ValueError(
                f"delay_cycles must be at least 1, got {delay_cycles}. A zero "
                f"delay is not a placebo -- it is the live arm."
            )
        self.delay_cycles = delay_cycles
        self._series: Dict[_Key, Deque[Decimal]] = {}

    def push(self, pair_symbol: str, side: str, price: Decimal) -> None:
        """Record the live quote for this cycle."""
        key = (pair_symbol, side)
        series = self._series.get(key)
        if series is None:
            # +1 so that after `delay_cycles` pushes the oldest is still
            # present and becomes available on the next one.
            series = deque(maxlen=self.delay_cycles + 1)
            self._series[key] = series
        series.append(price)

    def delayed(self, pair_symbol: str, side: str) -> Optional[Decimal]:
        """The quote from `delay_cycles` pushes ago, or None if not yet known.

        Returning None rather than the most recent value matters: substituting
        the live quote would make the placebo silently identical to the live
        arm and the control would read as confirming the edge.
        """
        series = self._series.get((pair_symbol, side))
        if series is None or len(series) <= self.delay_cycles:
            return None
        return series[0]

    def size(self, pair_symbol: str, side: str) -> int:
        series = self._series.get((pair_symbol, side))
        return 0 if series is None else len(series)
