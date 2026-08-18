"""Refreshing pool state cheaply, so high-frequency recording is affordable.

A full pool read costs roughly 8 calls plus one per initialised tick in the scanned
range: 50-200 RPC calls on a busy pool. At the 8 req/s a public endpoint sustains,
one refresh takes 10-25 seconds, which puts continuous recording of twenty pools --
and therefore statistical significance -- out of reach.

Almost none of that data changes between swaps. `liquidityNet` at each initialised
tick only moves when someone mints or burns a position. What changes on every swap
is `slot0` and the active `liquidity`: two calls.

So: read the tick set once, then refresh two fields. Roughly a fiftyfold reduction,
and the difference between one pool a minute and twenty pools at five-second
resolution.

THE RISK, AND WHAT GUARDS IT

A stale tick set silently prices a pool that no longer exists, and "silently" is the
operative word -- the quote looks perfectly normal. Two guards, because the two
failure modes are different:

  * PRICE MOVED OUT OF RANGE. The cached ticks cover a bounded price range. If the
    price leaves it, a swap would cross ticks we do not hold, so the ticks are
    re-read. This is checked on every refresh and is exact.
  * LIQUIDITY CHANGED IN PLACE. A mint or burn alters liquidityNet without moving
    the price at all, so the range check cannot see it. A TTL is the backstop.

Both are conservative in the same direction: they cause an unnecessary re-read, never
a stale quote.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Callable, Dict, Optional, Tuple

from loguru import logger

from ..core import clock
from .pool_state import PoolSnapshot

__all__ = ["PoolStateCache", "DEFAULT_FULL_REREAD_SECONDS"]

# How long a tick set may be reused when the price has not left its range. Mints and
# burns are the thing this bounds, and they are not frequent enough on a real pool to
# need a shorter window -- but a stale tick set is a wrong quote, so it is bounded.
DEFAULT_FULL_REREAD_SECONDS = 120.0


@dataclass
class _Entry:
    snapshot: PoolSnapshot
    ticks_read_at: float
    # Inclusive tick bounds the cached tick set is valid for.
    lower_tick: int
    upper_tick: int


class PoolStateCache:
    """Cheap incremental refresh over an expensive full read.

    `reader` needs two methods, which keeps this testable without a chain:

        read_full(chain, address, **kwargs) -> PoolSnapshot
        read_slot0(chain, address) -> (sqrt_price_x96, tick, liquidity, block)
    """

    def __init__(
        self,
        reader,
        full_reread_seconds: float = DEFAULT_FULL_REREAD_SECONDS,
        now_fn: Optional[Callable[[], float]] = None,
    ):
        if full_reread_seconds <= 0:
            raise ValueError("full_reread_seconds must be positive")
        self.reader = reader
        self.full_reread_seconds = float(full_reread_seconds)
        self._now_fn = now_fn
        self._entries: Dict[Tuple[str, str], _Entry] = {}
        self._full_reads = 0
        self._cheap_refreshes = 0

    def _now(self) -> float:
        return clock.now() if self._now_fn is None else self._now_fn()

    @staticmethod
    def _key(chain: str, address: str) -> Tuple[str, str]:
        # Chain included: pool addresses are not unique across chains, and a
        # collision would quote one chain's pool with another's state.
        return (str(chain).lower(), str(address).lower())

    # ------------------------------------------------------------------

    async def get(self, chain: str, address: str, **kwargs) -> PoolSnapshot:
        """The cached snapshot, reading it fully if not held."""
        key = self._key(chain, address)
        entry = self._entries.get(key)
        if entry is None:
            return await self._full_read(chain, address, **kwargs)
        return entry.snapshot

    async def refresh(self, chain: str, address: str, **kwargs) -> PoolSnapshot:
        """Update price and liquidity, re-reading ticks only when necessary."""
        key = self._key(chain, address)
        entry = self._entries.get(key)
        if entry is None:
            return await self._full_read(chain, address, **kwargs)

        if self._now() - entry.ticks_read_at >= self.full_reread_seconds:
            # A mint or burn changes liquidityNet without moving the price, so the
            # range check below cannot see it. This is the backstop for that.
            logger.debug(
                f"{chain} {address}: tick set is "
                f"{self._now() - entry.ticks_read_at:.0f}s old; re-reading"
            )
            return await self._full_read(chain, address, **kwargs)

        sqrt_price_x96, tick, liquidity, block = await self.reader.read_slot0(
            chain, address
        )
        self._cheap_refreshes += 1

        # STRICTLY inside. At the exact boundary there is zero headroom in one
        # direction: a swap moving that way immediately needs ticks beyond the
        # cached set, so the price being "in range" would be true and useless.
        # `SwapResult.range_exhausted` still catches the near-boundary case; this
        # avoids the degenerate one.
        if not entry.lower_tick < tick < entry.upper_tick:
            logger.debug(
                f"{chain} {address}: tick {tick} left the cached range "
                f"[{entry.lower_tick}, {entry.upper_tick}]; re-reading ticks"
            )
            return await self._full_read(chain, address, **kwargs)

        snapshot = replace(
            entry.snapshot,
            sqrt_price_x96=int(sqrt_price_x96),
            tick=int(tick),
            liquidity=int(liquidity),
            block_number=int(block) if block is not None else entry.snapshot.block_number,
            observed_at=self._now(),
        )
        self._entries[key] = _Entry(
            snapshot=snapshot,
            ticks_read_at=entry.ticks_read_at,
            lower_tick=entry.lower_tick,
            upper_tick=entry.upper_tick,
        )
        return snapshot

    async def _full_read(self, chain: str, address: str, **kwargs) -> PoolSnapshot:
        snapshot = await self.reader.read_full(chain, address, **kwargs)
        self._full_reads += 1
        # Prefer the window the reader actually recorded. Re-deriving it here from
        # tick_range_scanned would be a second source of truth for the same fact,
        # and the two can disagree: the reader clamps its scan to whole bitmap
        # words, so its real window is not always exactly tick +/- span. The swap
        # math bounds itself by the snapshot's window, so the cache's staleness
        # check must use the same one or the two guards disagree about where
        # knowledge ends.
        if snapshot.known_lower_tick is not None and snapshot.known_upper_tick is not None:
            lower, upper = snapshot.known_lower_tick, snapshot.known_upper_tick
        else:
            span = snapshot.tick_range_scanned * snapshot.tick_spacing
            lower, upper = snapshot.tick - span, snapshot.tick + span
        self._entries[self._key(chain, address)] = _Entry(
            snapshot=snapshot,
            ticks_read_at=self._now(),
            lower_tick=lower,
            upper_tick=upper,
        )
        return snapshot

    # ------------------------------------------------------------------

    def ticks_age_seconds(self, chain: str, address: str) -> Optional[float]:
        """How old the cached TICK data is, separately from the price.

        A recorded observation needs both: a replay cannot otherwise tell a fresh
        quote from one built on minute-old liquidity, and the second is a weaker
        observation that should be weighted or excluded rather than trusted equally.
        """
        entry = self._entries.get(self._key(chain, address))
        if entry is None:
            return None
        return self._now() - entry.ticks_read_at

    def stats(self) -> dict:
        """The saving, measured rather than assumed.

        This is the justification for the class existing, so it should be checkable
        in a live run and not merely argued for in a docstring.
        """
        total = self._full_reads + self._cheap_refreshes
        return {
            "full_reads": self._full_reads,
            "cheap_refreshes": self._cheap_refreshes,
            "refresh_ratio": (self._cheap_refreshes / total) if total else 0.0,
            "pools_cached": len(self._entries),
        }
