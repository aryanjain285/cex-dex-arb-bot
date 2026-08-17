"""Single source of time for the whole system.

Two clocks exist and they must never be mixed:

- `now()` is unix epoch seconds. Use it for every timestamp that is compared
  against another timestamp, persisted, or measured against an exchange event
  time. All `Quote`, `Opportunity`, and `ExecutionSummary` timestamps use this.

- `monotonic()` is for measuring elapsed durations within a single process
  only. It has an arbitrary epoch, so its values are meaningless outside the
  process and must never be persisted or compared against `now()`.

Mixing the two produced a real defect: quote timestamps were built from
`asyncio.get_running_loop().time()` (monotonic) while execution timestamps
used `time.time()` (epoch), leaving them ~1.7e9 apart and making both
staleness checks and markout measurement impossible to express.
"""
from __future__ import annotations

import time

__all__ = ["now", "now_ms", "monotonic"]


def now() -> float:
    """Current time as unix epoch seconds."""
    return time.time()


def now_ms() -> int:
    """Current time as unix epoch milliseconds, for exchange APIs."""
    return int(time.time() * 1000)


def monotonic() -> float:
    """Monotonic reading for duration measurement only.

    Never persist this value and never compare it against `now()`.
    """
    return time.monotonic()
