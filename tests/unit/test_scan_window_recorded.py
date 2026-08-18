"""The scan window must survive from the reader, through storage, into the math.

`V3Pool` now refuses to quote past the window whose liquidity it observed. That is
only worth anything if the window actually arrives. Three places can drop it, and
each failure is silent:

  * the reader knows the window (it computed the bounds to filter tick candidates)
    but might not pass it on -- then every snapshot falls back to its outermost
    recorded tick, and a sparse pool becomes unquotable at any size;
  * `to_row`/`from_row` might not persist it -- then a replayed observation prices
    differently from the live one that recorded it, which would quietly invalidate
    every backtest against stored state;
  * the cache's cheap refresh rebuilds a snapshot from two fields plus the cached
    ticks, and must carry the window across unchanged.

None of those would raise. All three would change reported prices.
"""
from decimal import Decimal

from src.exchange.pool_state import DEFAULT_TICK_RANGE, PoolSnapshot
from src.exchange.univ3_math import TickInfo, sqrt_price_x96_from_tick


def _snapshot(**overrides):
    fields = dict(
        sqrt_price_x96=sqrt_price_x96_from_tick(0),
        liquidity=10 ** 15,
        tick=0,
        fee=500,
        tick_spacing=10,
        ticks=[TickInfo(tick=-100, liquidity_net=5), TickInfo(tick=100, liquidity_net=-5)],
        decimals0=18,
        decimals1=6,
        block_number=1234,
        address="0x" + "ab" * 20,
        token0="0x" + "11" * 20,
        token1="0x" + "22" * 20,
        chain="ethereum",
        tick_range_scanned=60,
        observed_at=1_700_000_000.0,
        known_lower_tick=-600,
        known_upper_tick=600,
    )
    fields.update(overrides)
    return PoolSnapshot(**fields)


def test_the_window_round_trips_through_storage():
    original = _snapshot()
    restored = PoolSnapshot.from_row(original.to_row())
    assert restored.known_lower_tick == original.known_lower_tick == -600
    assert restored.known_upper_tick == original.known_upper_tick == 600


def test_a_restored_snapshot_prices_identically_to_the_original():
    """The property that matters, rather than field equality: a replayed snapshot
    must produce the same number as the live one, or a backtest measures its own
    serialisation."""
    original = _snapshot()
    restored = PoolSnapshot.from_row(original.to_row())
    for size in (Decimal("0.001"), Decimal("1"), Decimal("1000")):
        assert (original.price_for_amount_in(size, zero_for_one=True)
                == restored.price_for_amount_in(size, zero_for_one=True))


def test_a_row_written_before_the_window_existed_still_loads():
    """Additive migration: rows recorded by the previous build have no window
    columns. They must load with the conservative fallback rather than raise --
    the stored observations are the only history there is."""
    row = _snapshot().to_row()
    row.pop("known_lower_tick", None)
    row.pop("known_upper_tick", None)
    restored = PoolSnapshot.from_row(row)
    assert restored.known_lower_tick is None
    assert restored.known_upper_tick is None
    # And it must still be quotable, bounded by its outermost recorded tick.
    assert restored.price_for_amount_in(Decimal("0.000000001"), zero_for_one=True) is not None


def test_the_reader_records_the_window_it_scanned():
    """The reader computes these bounds already, to filter tick candidates. Reading
    them off the source is the check: a snapshot whose window does not match the
    range it says it scanned is describing a different observation."""
    import inspect

    from src.exchange import pool_state

    source = inspect.getsource(pool_state.fetch_pool_state)
    assert "known_lower_tick=lower_bound" in source, (
        "fetch_pool_state must pass the scanned window to the snapshot; without it "
        "every pool falls back to its outermost recorded tick"
    )
    assert "known_upper_tick=upper_bound" in source


def test_the_default_range_is_what_the_snapshot_reports():
    snap = _snapshot(tick_range_scanned=DEFAULT_TICK_RANGE)
    assert snap.tick_range_scanned == DEFAULT_TICK_RANGE
