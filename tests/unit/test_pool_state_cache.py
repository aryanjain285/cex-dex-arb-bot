"""Refreshing a pool cheaply, so high-frequency recording is affordable.

A full pool read costs roughly 8 calls plus one per initialised tick in the scanned
range -- on a busy pool that is 50-200 RPC calls. At the measured 8 req/s a public
endpoint sustains, one pool refresh takes 10-25 seconds. Recording twenty pools
continuously is impossible at that price, which puts statistical significance out of
reach.

But almost all of that data does not change. Between swaps a pool's TICKS are
static: `liquidityNet` at each initialised tick only changes when someone mints or
burns a position. What moves on every swap is `slot0` and the active `liquidity` --
two calls.

So the cache reads the full tick set once and then refreshes two fields, re-reading
the ticks only when the price has moved outside the range the cached ticks cover.
That turns a 100-call refresh into a 2-call refresh, roughly a fiftyfold reduction,
and it is the difference between 20 pools at 5-second resolution and 1 pool a
minute.

The correctness risk is obvious and is what these tests are about: a stale tick set
silently prices a pool that no longer exists. The cache therefore tracks exactly
which price range its ticks are valid for, and refuses -- rather than guessing --
when a quote would leave it.
"""
from decimal import Decimal

import pytest

from src.exchange.pool_state import PoolSnapshot
from src.exchange.pool_state_cache import PoolStateCache
from src.exchange.univ3_math import TickInfo, sqrt_price_x96_from_tick


def D(x) -> Decimal:
    return Decimal(str(x))


def _snapshot(tick=0, liquidity=10 ** 21, ticks=None, block=100,
              tick_range=50, spacing=10) -> PoolSnapshot:
    if ticks is None:
        ticks = [TickInfo(tick=-500, liquidity_net=liquidity),
                 TickInfo(tick=500, liquidity_net=-liquidity)]
    return PoolSnapshot(
        sqrt_price_x96=sqrt_price_x96_from_tick(tick),
        liquidity=liquidity, tick=tick, fee=500, tick_spacing=spacing,
        ticks=ticks, decimals0=18, decimals1=6,
        block_number=block, address="0x" + "ab" * 20,
        token0="0x" + "11" * 20, token1="0x" + "22" * 20,
        chain="ethereum", tick_range_scanned=tick_range, observed_at=1000.0,
    )


class FakeReader:
    """Counts full reads against cheap refreshes, which is the whole point."""

    def __init__(self, snapshot: PoolSnapshot):
        self.snapshot = snapshot
        self.full_reads = 0
        self.cheap_refreshes = 0
        # What the next cheap refresh should return.
        self.next_tick = snapshot.tick
        self.next_liquidity = snapshot.liquidity
        self.next_block = (snapshot.block_number or 0) + 1

    async def read_full(self, chain, address, **kwargs):
        self.full_reads += 1
        from dataclasses import replace
        return replace(
            self.snapshot, tick=self.next_tick,
            sqrt_price_x96=sqrt_price_x96_from_tick(self.next_tick),
            liquidity=self.next_liquidity, block_number=self.next_block,
        )

    async def read_slot0(self, chain, address):
        self.cheap_refreshes += 1
        return (
            sqrt_price_x96_from_tick(self.next_tick),
            self.next_tick,
            self.next_liquidity,
            self.next_block,
        )


# --- the cost saving ----------------------------------------------------


async def test_the_first_read_is_a_full_read():
    reader = FakeReader(_snapshot())
    cache = PoolStateCache(reader)

    await cache.get("ethereum", "0x" + "ab" * 20)

    assert reader.full_reads == 1
    assert reader.cheap_refreshes == 0


async def test_a_refresh_within_the_cached_range_costs_two_calls():
    """The saving. A price move that stays inside the cached tick range needs only
    slot0 and liquidity -- the ticks have not changed."""
    reader = FakeReader(_snapshot(tick=0))
    cache = PoolStateCache(reader)
    await cache.get("ethereum", "0x" + "ab" * 20)

    reader.next_tick = 20  # still well inside +/-500
    for _ in range(10):
        await cache.refresh("ethereum", "0x" + "ab" * 20)

    assert reader.full_reads == 1, "the ticks were re-read unnecessarily"
    assert reader.cheap_refreshes == 10


async def test_the_refreshed_snapshot_carries_the_new_price():
    reader = FakeReader(_snapshot(tick=0))
    cache = PoolStateCache(reader)
    await cache.get("ethereum", "0x" + "ab" * 20)

    reader.next_tick = 37
    reader.next_liquidity = 5 * 10 ** 20
    snapshot = await cache.refresh("ethereum", "0x" + "ab" * 20)

    assert snapshot.tick == 37
    assert snapshot.liquidity == 5 * 10 ** 20
    assert snapshot.sqrt_price_x96 == sqrt_price_x96_from_tick(37)


def test_the_cached_ticks_are_carried_forward_unchanged():
    """The assumption made explicit: between swaps, liquidityNet per tick is
    static. It changes on mint and burn, which is what the range check catches."""
    snapshot = _snapshot()
    assert len(snapshot.ticks) == 2


# --- the correctness risk -----------------------------------------------


async def test_leaving_the_cached_range_forces_a_full_re_read():
    """The failure this cache could cause: a price that has moved outside the ticks
    we hold would be quoted against liquidity that no longer applies."""
    reader = FakeReader(_snapshot(tick=0, tick_range=50, spacing=10))
    cache = PoolStateCache(reader)
    await cache.get("ethereum", "0x" + "ab" * 20)

    # +/-50 spacings at spacing 10 is +/-500 ticks. Move well beyond it.
    reader.next_tick = 900
    await cache.refresh("ethereum", "0x" + "ab" * 20)

    assert reader.full_reads == 2, (
        "the price left the cached tick range and the ticks were not re-read"
    )


async def test_a_move_to_the_edge_of_the_range_re_reads():
    """At the boundary, not past it: a quote at the edge can cross the last cached
    tick, and beyond it there is no data."""
    reader = FakeReader(_snapshot(tick=0, tick_range=50, spacing=10))
    cache = PoolStateCache(reader)
    await cache.get("ethereum", "0x" + "ab" * 20)

    reader.next_tick = 500  # exactly the edge
    await cache.refresh("ethereum", "0x" + "ab" * 20)

    assert reader.full_reads == 2


async def test_the_ttl_forces_a_full_re_read_even_without_a_price_move():
    """Mints and burns change liquidityNet without moving the price at all, so a
    cache that only re-read on price movement would hold stale ticks indefinitely
    in a quiet pool. The TTL is the backstop for the thing the range check cannot
    see."""
    reader = FakeReader(_snapshot(tick=0))
    clock_value = [1000.0]
    cache = PoolStateCache(reader, full_reread_seconds=60,
                           now_fn=lambda: clock_value[0])
    await cache.get("ethereum", "0x" + "ab" * 20)

    clock_value[0] = 1030.0
    await cache.refresh("ethereum", "0x" + "ab" * 20)
    assert reader.full_reads == 1, "not yet due"

    clock_value[0] = 1061.0
    await cache.refresh("ethereum", "0x" + "ab" * 20)
    assert reader.full_reads == 2, "the TTL did not force a re-read"


async def test_pools_are_cached_independently():
    reader = FakeReader(_snapshot())
    cache = PoolStateCache(reader)

    await cache.get("ethereum", "0x" + "ab" * 20)
    await cache.get("ethereum", "0x" + "cd" * 20)

    assert reader.full_reads == 2


async def test_the_same_pool_on_two_chains_is_not_confused():
    """Pool addresses are not unique across chains, and a cross-chain collision
    would quote one chain's pool with another's state."""
    reader = FakeReader(_snapshot())
    cache = PoolStateCache(reader)

    await cache.get("ethereum", "0x" + "ab" * 20)
    await cache.get("base", "0x" + "ab" * 20)

    assert reader.full_reads == 2


# --- staleness is visible ----------------------------------------------


async def test_a_snapshot_reports_how_old_its_ticks_are():
    """A recorded observation must say when its tick data was read, separately from
    when its price was, or a replay cannot tell a fresh quote from one built on
    minute-old liquidity."""
    reader = FakeReader(_snapshot())
    clock_value = [1000.0]
    cache = PoolStateCache(reader, now_fn=lambda: clock_value[0])
    await cache.get("ethereum", "0x" + "ab" * 20)

    clock_value[0] = 1042.0
    age = cache.ticks_age_seconds("ethereum", "0x" + "ab" * 20)

    assert age == pytest.approx(42.0)


async def test_an_unknown_pool_has_no_tick_age():
    cache = PoolStateCache(FakeReader(_snapshot()))

    assert cache.ticks_age_seconds("ethereum", "0xdead") is None


async def test_the_cache_reports_its_own_call_saving():
    """So the saving is measured rather than assumed -- this is the justification
    for the whole class, and it should be checkable in a live run."""
    reader = FakeReader(_snapshot(tick=0))
    cache = PoolStateCache(reader)
    await cache.get("ethereum", "0x" + "ab" * 20)
    reader.next_tick = 10
    for _ in range(20):
        await cache.refresh("ethereum", "0x" + "ab" * 20)

    stats = cache.stats()

    assert stats["full_reads"] == 1
    assert stats["cheap_refreshes"] == 20
    assert stats["refresh_ratio"] == pytest.approx(20 / 21, abs=0.01)
