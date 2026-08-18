"""The recorder must be honest about what it failed to see.

A recording run is the only source of evidence the research has. Its failure modes
are therefore worse than a crash, because a crash is visible:

  * one pair's RPC failing must not stop the others -- otherwise a week of
    recording silently becomes a week of recording ONE pair, on whichever chain
    happened to be reliable, and the cross-pair comparison is then a comparison of
    endpoint quality;
  * a failed cycle must be COUNTED. A gap in a time series looks exactly like a
    period of no activity, so an unrecorded failure turns "the endpoint was down"
    into "the market was quiet";
  * a half-observation must never be stored. An observation whose CEX side is
    present and whose pool side is missing would be silently treated as a valid
    instant by anything computing a spread;
  * the achieved cadence must be measured, not assumed. This system has already
    been caught configuring 0.2s and achieving 2.32s, which is a 12x error in
    every per-second statistic derived from it.
"""
import asyncio
from decimal import Decimal

import pytest

from src.exchange.pool_state import PoolSnapshot
from src.exchange.univ3_math import TickInfo, sqrt_price_x96_from_tick
from src.research.observations import ObservationStore
from src.research.recorder import Recorder, RecorderTarget


def _pool(tick=0) -> PoolSnapshot:
    return PoolSnapshot(
        sqrt_price_x96=sqrt_price_x96_from_tick(tick),
        liquidity=10 ** 20,
        tick=tick,
        fee=500,
        tick_spacing=10,
        ticks=[TickInfo(tick=-1000, liquidity_net=5),
               TickInfo(tick=1000, liquidity_net=-5)],
        decimals0=18,
        decimals1=6,
        block_number=100,
        address="0x" + "ab" * 20,
        token0="0x" + "11" * 20,
        token1="0x" + "22" * 20,
        chain="ethereum",
        known_lower_tick=-9530,
        known_upper_tick=9530,
    )


class FakeBook:
    def __init__(self, bid=Decimal("1900"), ask=Decimal("1901")):
        self.bids = [(bid, Decimal("10"))]
        self.asks = [(ask, Decimal("10"))]
        self.feed_timestamp = 1.0


class FakeCex:
    def __init__(self, books=None, fail_for=()):
        self._books = books or {}
        self._fail_for = set(fail_for)
        self.calls = 0

    async def get_book(self, pair):
        self.calls += 1
        if pair.cex_symbol in self._fail_for:
            raise RuntimeError(f"book feed down for {pair.cex_symbol}")
        return self._books.get(pair.cex_symbol, FakeBook())


class FakePools:
    """Stands in for the pool state cache."""

    def __init__(self, fail_for=(), snapshots=None):
        self._fail_for = set(fail_for)
        self._snapshots = snapshots or {}
        self.calls = 0

    async def get(self, chain, address, **kwargs):
        self.calls += 1
        if address in self._fail_for:
            from src.exchange.errors import RpcError
            raise RpcError(f"rpc down for {address}")
        return self._snapshots.get(address, _pool())


class FakeGas:
    def __init__(self, wei=10 ** 10, native=Decimal("1900"), fail=False):
        self.wei, self.native, self.fail = wei, native, fail

    async def read(self, chain):
        if self.fail:
            raise RuntimeError("gas oracle down")
        return self.wei, self.native


def _target(symbol="ETHUSDT", address="0x" + "ab" * 20):
    """The REAL MarketPair, not a stand-in.

    An earlier version of this fixture was an ad-hoc class with the attributes the
    recorder happened to read, including a `quote` field MarketPair does not have --
    it distinguishes `quote_cex` from `quote_dex`. Every test passed and the live
    recorder failed 100% of its observations with AttributeError, which the failure
    counting caught and the DEBUG-level log hid. A fake that defines its own contract
    tests the fake.
    """
    from src.core.types import MarketPair

    return RecorderTarget(
        pair=MarketPair(
            base="ETH", quote_cex="USDT", quote_dex="USDT",
            cex_symbol=symbol, dex_chain="ethereum", dex_pool_fee=500,
        ),
        pool_address=address,
    )


@pytest.fixture
def store(tmp_path):
    return ObservationStore(tmp_path / "obs.sqlite3", run_id="rec-test")


class TestOneCycle:
    @pytest.mark.asyncio
    async def test_a_cycle_records_one_observation_per_target(self, store):
        recorder = Recorder(
            store=store, cex=FakeCex(), pools=FakePools(), gas=FakeGas(),
            targets=[_target("ETHUSDT", "0x" + "ab" * 20),
                     _target("ARBUSDT", "0x" + "cd" * 20)],
        )
        await recorder.cycle()
        assert store.count() == 2
        assert set(store.pairs()) == {"ETHUSDT", "ARBUSDT"}

    @pytest.mark.asyncio
    async def test_the_recorded_observation_carries_both_venues(self, store):
        recorder = Recorder(store=store, cex=FakeCex(), pools=FakePools(),
                            gas=FakeGas(), targets=[_target()])
        await recorder.cycle()
        obs = list(store.read_all())[0]
        assert obs.best_bid == Decimal("1900")
        assert obs.pool.sqrt_price_x96 > 0
        assert obs.gas_price_wei == 10 ** 10
        assert obs.native_price_quote == Decimal("1900")


class TestFailureIsolation:
    @pytest.mark.asyncio
    async def test_one_failing_pool_does_not_stop_the_others(self, store):
        broken = "0x" + "cd" * 20
        recorder = Recorder(
            store=store, cex=FakeCex(), pools=FakePools(fail_for=[broken]),
            gas=FakeGas(),
            targets=[_target("ETHUSDT", "0x" + "ab" * 20),
                     _target("ARBUSDT", broken)],
        )
        await recorder.cycle()
        assert store.count() == 1
        assert store.pairs() == ["ETHUSDT"]

    @pytest.mark.asyncio
    async def test_one_failing_book_does_not_stop_the_others(self, store):
        recorder = Recorder(
            store=store, cex=FakeCex(fail_for=["ARBUSDT"]), pools=FakePools(),
            gas=FakeGas(),
            targets=[_target("ETHUSDT", "0x" + "ab" * 20),
                     _target("ARBUSDT", "0x" + "cd" * 20)],
        )
        await recorder.cycle()
        assert store.count() == 1

    @pytest.mark.asyncio
    async def test_failures_are_counted_per_target(self, store):
        broken = "0x" + "cd" * 20
        recorder = Recorder(
            store=store, cex=FakeCex(), pools=FakePools(fail_for=[broken]),
            gas=FakeGas(),
            targets=[_target("ETHUSDT", "0x" + "ab" * 20),
                     _target("ARBUSDT", broken)],
        )
        await recorder.cycle()
        stats = recorder.stats()
        assert stats["recorded"] == 1
        assert stats["failed"] == 1
        assert stats["failures_by_pair"]["ARBUSDT"] == 1

    @pytest.mark.asyncio
    async def test_a_missing_book_records_nothing_rather_than_half(self, store):
        class NoBook(FakeCex):
            async def get_book(self, pair):
                return None

        recorder = Recorder(store=store, cex=NoBook(), pools=FakePools(),
                            gas=FakeGas(), targets=[_target()])
        await recorder.cycle()
        assert store.count() == 0, (
            "an observation with one venue missing would be read as a valid "
            "instant by anything computing a spread"
        )
        assert recorder.stats()["failed"] == 1

    @pytest.mark.asyncio
    async def test_an_empty_book_side_records_nothing(self, store):
        class OneSided(FakeCex):
            async def get_book(self, pair):
                book = FakeBook()
                book.asks = []
                return book

        recorder = Recorder(store=store, cex=OneSided(), pools=FakePools(),
                            gas=FakeGas(), targets=[_target()])
        await recorder.cycle()
        assert store.count() == 0

    @pytest.mark.asyncio
    async def test_a_gas_failure_still_records_the_market(self, store):
        """Gas is a cost input, not part of the market state. Losing it must not
        cost the observation -- but the row must say the gas is absent rather than
        imply it was zero."""
        recorder = Recorder(store=store, cex=FakeCex(), pools=FakePools(),
                            gas=FakeGas(fail=True), targets=[_target()])
        await recorder.cycle()
        assert store.count() == 1
        obs = list(store.read_all())[0]
        assert obs.gas_price_wei is None
        assert obs.gas_quote(200_000) is None, (
            "an absent gas price must not evaluate to a zero cost"
        )


class TestCadenceIsMeasured:
    @pytest.mark.asyncio
    async def test_the_achieved_cadence_is_reported_not_the_configured_one(self, store):
        recorder = Recorder(store=store, cex=FakeCex(), pools=FakePools(),
                            gas=FakeGas(), targets=[_target()],
                            interval_seconds=0.01)
        await recorder.run(max_cycles=3)
        stats = recorder.stats()
        assert stats["cycles"] == 3
        assert stats["measured_cadence_seconds"] is not None
        assert stats["measured_cadence_seconds"] > 0

    @pytest.mark.asyncio
    async def test_running_for_a_fixed_number_of_cycles_stops(self, store):
        recorder = Recorder(store=store, cex=FakeCex(), pools=FakePools(),
                            gas=FakeGas(), targets=[_target()],
                            interval_seconds=0.0)
        await asyncio.wait_for(recorder.run(max_cycles=5), timeout=5)
        assert store.count() == 5

    @pytest.mark.asyncio
    async def test_stop_ends_the_loop(self, store):
        recorder = Recorder(store=store, cex=FakeCex(), pools=FakePools(),
                            gas=FakeGas(), targets=[_target()],
                            interval_seconds=0.01)
        task = asyncio.create_task(recorder.run())
        await asyncio.sleep(0.05)
        recorder.stop()
        await asyncio.wait_for(task, timeout=5)
        assert store.count() >= 1


class TestNoTargets:
    @pytest.mark.asyncio
    async def test_recording_with_no_targets_is_an_error(self, store):
        """A recorder with nothing to record would run for a week and produce an
        empty file, which reads as 'no opportunities' rather than 'misconfigured'."""
        with pytest.raises(ValueError, match="no targets"):
            Recorder(store=store, cex=FakeCex(), pools=FakePools(),
                     gas=FakeGas(), targets=[])


class TestSlowTargetsDoNotBlockFastOnes:
    """A cycle must not be as slow as its slowest pool.

    Measured cost of a full pool read, 2026-08-18: 14-16s on Arbitrum, 34-131s on
    Ethereum and Base -- per-call latency on public endpoints, not rate limiting.
    Tick data is re-read on a timer, so every few minutes some pool pays that cost.
    If the cycle collected all results before writing any, then Arbitrum's twelve
    fast observations would be discarded down to one, because they waited for a Base
    pool that took two minutes.

    The consequence is not just fewer rows. It is a sampling rate that varies with
    the slowest endpoint in the set, so the cadence -- and therefore every
    per-second and lifetime statistic -- would be set by whichever chain is worst.
    """

    @pytest.mark.asyncio
    async def test_a_fast_target_is_written_before_a_slow_one_finishes(self, store):
        released = asyncio.Event()

        class MixedSpeed(FakePools):
            async def get(self, chain, address, **kwargs):
                self.calls += 1
                if address.endswith("cd" * 2):
                    await released.wait()
                return _pool()

        recorder = Recorder(
            store=store, cex=FakeCex(), pools=MixedSpeed(), gas=FakeGas(),
            targets=[_target("ETHUSDT", "0x" + "ab" * 20),
                     _target("ARBUSDT", "0x" + "cd" * 20)],
        )
        task = asyncio.create_task(recorder.cycle())

        # Give the fast target a chance to complete and be written.
        for _ in range(50):
            await asyncio.sleep(0)
            if store.count() >= 1:
                break

        assert store.count() == 1, (
            "the fast pool's observation was not written while the slow pool was "
            "still reading; a slow endpoint would set the cadence for every chain"
        )
        released.set()
        await asyncio.wait_for(task, timeout=5)
        assert store.count() == 2

    @pytest.mark.asyncio
    async def test_every_observation_still_arrives_exactly_once(self, store):
        """Writing incrementally must not duplicate or drop rows."""
        recorder = Recorder(
            store=store, cex=FakeCex(), pools=FakePools(), gas=FakeGas(),
            targets=[_target(f"SYM{i}", "0x" + f"{i:02d}" * 20) for i in range(6)],
        )
        await recorder.cycle()
        assert store.count() == 6
        assert len(set(store.pairs())) == 6


class TestTheRecorderRefreshesRatherThanReadsTheCache:
    """A recorder that re-serves cached state records the same instant repeatedly.

    `PoolStateCache.get` returns the held snapshot untouched; `refresh` re-reads
    slot0 -- the price and active liquidity, two calls. A first run of this recorder
    reported 22 full reads, 0 cheap refreshes, and 44 observations: the second cycle
    recorded byte-identical pool state.

    Nothing about that looks wrong in the output. The row count grows, the timestamps
    advance, and every downstream statistic is silently poisoned: the autocorrelation
    goes to 1, so the effective sample size collapses toward one, and opportunity
    lifetimes become however long the recorder ran. A dataset that says the price
    never moved is worse than a smaller one.
    """

    @pytest.mark.asyncio
    async def test_each_cycle_re_reads_the_pool(self, store):
        class CountingPools(FakePools):
            def __init__(self):
                super().__init__()
                self.gets = 0
                self.refreshes = 0

            async def get(self, chain, address, **kwargs):
                self.gets += 1
                return _pool()

            async def refresh(self, chain, address, **kwargs):
                self.refreshes += 1
                return _pool(tick=self.refreshes)

        pools = CountingPools()
        recorder = Recorder(store=store, cex=FakeCex(), pools=pools,
                            gas=FakeGas(), targets=[_target()])
        await recorder.cycle()
        await recorder.cycle()

        assert pools.refreshes >= 1, (
            "the recorder must call refresh(), not get(); get() re-serves the held "
            "snapshot and records the same instant twice"
        )

    @pytest.mark.asyncio
    async def test_consecutive_observations_differ(self, store):
        """The property that matters, stated on the data rather than the call count."""
        class MovingPools(FakePools):
            def __init__(self):
                super().__init__()
                self.n = 0

            async def refresh(self, chain, address, **kwargs):
                self.n += 1
                return _pool(tick=self.n * 10)

            async def get(self, chain, address, **kwargs):
                return await self.refresh(chain, address, **kwargs)

        recorder = Recorder(store=store, cex=FakeCex(), pools=MovingPools(),
                            gas=FakeGas(), targets=[_target()])
        await recorder.cycle()
        await recorder.cycle()

        prices = [o.pool.sqrt_price_x96 for o in store.read_all()]
        assert len(prices) == 2
        assert prices[0] != prices[1], (
            "two cycles recorded the same pool price; the store is accumulating "
            "copies rather than observations"
        )

    @pytest.mark.asyncio
    async def test_a_reader_without_refresh_still_works(self, store):
        """Not every pool source has a cheap path. Falling back to get() is correct;
        silently doing nothing is not."""
        recorder = Recorder(store=store, cex=FakeCex(), pools=FakePools(),
                            gas=FakeGas(), targets=[_target()])
        await recorder.cycle()
        assert store.count() == 1
