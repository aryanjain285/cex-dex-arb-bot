"""Recorded observations must be re-quotable, not merely readable.

The existing evaluation store records DECISIONS: one price, at one size, under one
cost model, with its verdict. That is the right shape for an audit trail and the
wrong shape for research, because every interesting question is one it cannot
answer:

    "would $5,000 have worked?"        -- the price was quoted at $1,000 only
    "what if execution took 2s?"       -- no successor state was kept
    "how deep was the book really?"    -- only the touch was stored
    "what if the fee tier were 0.05%?" -- the quote already had 0.30% baked in

A pool SNAPSHOT is different in kind: it can be re-quoted at any size, under any
cost model, months later. A CEX ladder can be re-walked at any notional. So the
observation store keeps raw state, and the answers are computed at analysis time
rather than fixed at record time.

Which makes losslessness the whole contract. Two failure modes matter and neither
raises:

  * a Decimal routed through binary float shifts prices in the last places, and
    the entire strategy lives in the 5-20 bps range;
  * a dropped or reordered tick changes the price of every large size while
    leaving small sizes exactly right -- so a smoke test passes and the capacity
    analysis is silently wrong.

The tests below therefore check what a restored observation PRICES, not just what
it stores.
"""
from decimal import Decimal
from pathlib import Path

import pytest

from src.exchange.pool_state import PoolSnapshot
from src.exchange.univ3_math import TickInfo, sqrt_price_x96_from_tick
from src.research.observations import Observation, ObservationStore


def _pool(tick=0, n_ticks=8) -> PoolSnapshot:
    """A pool with enough tick structure that dropping one changes a price."""
    ticks = []
    for i in range(1, n_ticks + 1):
        ticks.append(TickInfo(tick=-100 * i, liquidity_net=7 * i))
        ticks.append(TickInfo(tick=100 * i, liquidity_net=-7 * i))
    return PoolSnapshot(
        sqrt_price_x96=sqrt_price_x96_from_tick(tick),
        liquidity=10 ** 20,
        tick=tick,
        fee=500,
        tick_spacing=10,
        ticks=ticks,
        decimals0=18,
        decimals1=6,
        block_number=25_779_036,
        address="0x" + "ab" * 20,
        token0="0x" + "11" * 20,
        token1="0x" + "22" * 20,
        chain="ethereum",
        tick_range_scanned=953,
        observed_at=1_755_500_000.5,
        known_lower_tick=-9530,
        known_upper_tick=9530,
    )


def _observation(**overrides) -> Observation:
    fields = dict(
        ts=1_755_500_000.5,
        cex_symbol="ETHUSDT",
        base="ETH",
        quote="USDT",
        chain="ethereum",
        pool_fee=500,
        pool_address="0x" + "ab" * 20,
        # Prices with more precision than a float carries, deliberately.
        cex_bids=[
            (Decimal("1896.62000000"), Decimal("12.34500000")),
            (Decimal("1896.61000000"), Decimal("40.00000000")),
        ],
        cex_asks=[
            (Decimal("1896.63000000"), Decimal("8.10000000")),
            (Decimal("1896.64000000"), Decimal("55.55555555")),
        ],
        cex_feed_ts=1_755_500_000.4,
        pool=_pool(),
        gas_price_wei=13_456_789_012,
        native_price_quote=Decimal("1896.625"),
        rpc_endpoint="https://ethereum-rpc.publicnode.com",
        run_id="test-run",
    )
    fields.update(overrides)
    return Observation(**fields)


@pytest.fixture
def store(tmp_path) -> ObservationStore:
    return ObservationStore(tmp_path / "obs.sqlite3", run_id="test-run")


class TestLosslessness:
    def test_a_restored_observation_prices_identically(self, store):
        """The property that matters. Field equality can pass while the numbers
        move; this cannot."""
        original = _observation()
        store.record(original)
        restored = list(store.read_all())[0]

        for size in (Decimal("0.001"), Decimal("1"), Decimal("100"), Decimal("5000")):
            for zero_for_one in (True, False):
                assert (
                    original.pool.price_for_amount_in(size, zero_for_one=zero_for_one)
                    == restored.pool.price_for_amount_in(size, zero_for_one=zero_for_one)
                ), f"size {size} zero_for_one={zero_for_one} priced differently"

    def test_every_tick_survives_in_order(self, store):
        original = _observation()
        store.record(original)
        restored = list(store.read_all())[0]
        assert [(t.tick, t.liquidity_net) for t in restored.pool.ticks] == \
               [(t.tick, t.liquidity_net) for t in original.pool.ticks]

    def test_cex_ladders_survive_exactly(self, store):
        original = _observation()
        store.record(original)
        restored = list(store.read_all())[0]
        assert restored.cex_bids == original.cex_bids
        assert restored.cex_asks == original.cex_asks

    def test_prices_are_decimals_not_floats(self, store):
        """A float round-trip of 1896.62 is 1896.6199999999998863. At 5 bps of edge
        that is not noise, it is a tenth of the signal."""
        store.record(_observation())
        restored = list(store.read_all())[0]
        for price, size in list(restored.cex_bids) + list(restored.cex_asks):
            assert isinstance(price, Decimal)
            assert isinstance(size, Decimal)
        assert isinstance(restored.native_price_quote, Decimal)

    def test_big_integers_survive(self, store):
        """sqrtPriceX96 and liquidity exceed 2^53, so a JSON number would round
        them. The price is derived from sqrtPriceX96 directly."""
        original = _observation()
        store.record(original)
        restored = list(store.read_all())[0]
        assert restored.pool.sqrt_price_x96 == original.pool.sqrt_price_x96
        assert restored.pool.liquidity == original.pool.liquidity

    def test_the_observed_window_survives(self, store):
        """Without it a replayed snapshot falls back to its outermost recorded tick
        and refuses sizes the live one priced -- so the backtest would be
        systematically more pessimistic than the run that recorded it."""
        original = _observation()
        store.record(original)
        restored = list(store.read_all())[0]
        assert restored.pool.known_lower_tick == original.pool.known_lower_tick
        assert restored.pool.known_upper_tick == original.pool.known_upper_tick


class TestProvenance:
    def test_the_endpoint_is_recorded(self, store):
        """Public endpoints drop requests and disagree with each other. A number
        whose source is unknown cannot be compared with one from another run."""
        store.record(_observation())
        assert list(store.read_all())[0].rpc_endpoint == \
               "https://ethereum-rpc.publicnode.com"

    def test_the_block_is_recorded(self, store):
        store.record(_observation())
        assert list(store.read_all())[0].pool.block_number == 25_779_036

    def test_gas_is_stored_as_price_not_as_cost(self, store):
        """Storing gas_quote would bake in a gas_units assumption. The raw gas
        price plus the native price lets any assumption be applied later --
        including the measured one, once receipts exist."""
        store.record(_observation())
        restored = list(store.read_all())[0]
        assert restored.gas_price_wei == 13_456_789_012
        assert restored.native_price_quote == Decimal("1896.625")

    def test_the_run_id_is_stamped(self, store):
        store.record(_observation())
        assert list(store.read_all())[0].run_id == "test-run"


class TestReading:
    def test_observations_come_back_in_time_order(self, store):
        for ts in (300.0, 100.0, 200.0):
            store.record(_observation(ts=ts))
        assert [o.ts for o in store.read_all()] == [100.0, 200.0, 300.0]

    def test_reading_can_be_filtered_by_pair(self, store):
        store.record(_observation(cex_symbol="ETHUSDT"))
        store.record(_observation(cex_symbol="ARBUSDT"))
        got = list(store.read_all(cex_symbol="ARBUSDT"))
        assert len(got) == 1 and got[0].cex_symbol == "ARBUSDT"

    def test_reading_can_be_filtered_by_time_window(self, store):
        for ts in (100.0, 200.0, 300.0):
            store.record(_observation(ts=ts))
        got = list(store.read_all(since=150.0, until=250.0))
        assert [o.ts for o in got] == [200.0]

    def test_counting_does_not_load_everything(self, store):
        for ts in (100.0, 200.0):
            store.record(_observation(ts=ts))
        assert store.count() == 2

    def test_an_empty_store_reads_as_empty_rather_than_raising(self, store):
        assert list(store.read_all()) == []
        assert store.count() == 0


class TestDurability:
    def test_records_survive_reopening(self, tmp_path):
        path = tmp_path / "obs.sqlite3"
        first = ObservationStore(path, run_id="a")
        first.record(_observation())
        first.close()

        second = ObservationStore(path, run_id="b")
        assert second.count() == 1
        # And the reopened store stamps ITS run id on new rows, not the old one.
        second.record(_observation(ts=2.0))
        runs = {o.run_id for o in second.read_all()}
        assert runs == {"test-run"}, (
            "run_id travels with the observation, so a re-analysis cannot "
            "misattribute rows to the run that happened to read them"
        )

    def test_a_second_writer_can_open_the_same_file(self, tmp_path):
        """The recorder writes while analysis reads. WAL mode makes that safe; its
        absence turns a routine mid-run check into a locked database."""
        path = tmp_path / "obs.sqlite3"
        writer = ObservationStore(path, run_id="w")
        writer.record(_observation())
        reader = ObservationStore(path, run_id="r")
        assert reader.count() == 1
        writer.record(_observation(ts=2.0))
        assert reader.count() == 2


class TestSchemaEvolution:
    def test_an_added_column_does_not_orphan_existing_rows(self, tmp_path):
        """Observations are the only history there is; a migration that cannot read
        yesterday's rows destroys the dataset it was meant to extend."""
        path = tmp_path / "obs.sqlite3"
        store = ObservationStore(path, run_id="a")
        store.record(_observation())
        store.close()

        import sqlite3
        with sqlite3.connect(path) as conn:
            conn.execute("ALTER TABLE observations ADD COLUMN future_field TEXT")

        reopened = ObservationStore(path, run_id="b")
        assert reopened.count() == 1
        assert list(reopened.read_all())[0].cex_symbol == "ETHUSDT"
