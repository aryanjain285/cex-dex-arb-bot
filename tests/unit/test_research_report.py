"""The report must state what it cannot conclude as clearly as what it can.

This is the module a decision about real capital would be read from, so its tests
are about honesty rather than formatting.

Three specific failures it must not have:

  * POOLING INCOMPARABLE THINGS. The same pair on two chains, or two fee tiers, is
    two markets. Averaging them produces a number describing neither, and it is the
    kind of error that survives review because the output looks tidier.

  * REPORTING A MEAN WITHOUT ITS UNCERTAINTY. "Mean gross -1.5 bps" invites a
    conclusion; "-1.5 bps, 95% CI [-1.9, -1.1], effective n 340 of 4,000" invites
    the right one. Since observations seconds apart are barely independent, the
    effective count is the honest one and must appear beside the raw one.

  * COUNTING REFUSALS AS ABSENCE. An observation the simulator declined to price is
    not an observation of no edge. If unpriceable rows were dropped silently, a pool
    too thin to quote at any size would report the same "no opportunities" as a deep
    pool genuinely at parity, and those two findings mean opposite things.
"""
from decimal import Decimal

import pytest

from src.exchange.pool_state import PoolSnapshot
from src.exchange.univ3_math import TickInfo
from src.research.evaluate import CostModel
from src.research.observations import Observation, ObservationStore
from src.research.report import analyse_store, group_key


def _pool(price=Decimal("1900")) -> PoolSnapshot:
    from decimal import getcontext
    getcontext().prec = 60
    raw = price * (Decimal(10) ** 6) / (Decimal(10) ** 18)
    liquidity = 10 ** 24
    return PoolSnapshot(
        sqrt_price_x96=int(Decimal(2 ** 96) * raw.sqrt()),
        liquidity=liquidity, tick=0, fee=500, tick_spacing=10,
        ticks=[TickInfo(tick=-500000, liquidity_net=liquidity),
               TickInfo(tick=500000, liquidity_net=-liquidity)],
        decimals0=18, decimals1=6, block_number=1,
        address="0x" + "ab" * 20, token0="0x" + "11" * 20,
        token1="0x" + "22" * 20, chain="ethereum",
        known_lower_tick=-500000, known_upper_tick=500000,
    )


def _obs(ts, cex=Decimal("1900"), dex=Decimal("1900"), chain="ethereum",
         fee=500, symbol="ETHUSDT", gas_wei=10 ** 9):
    spread = Decimal("0.00005")
    return Observation(
        ts=ts, cex_symbol=symbol, base="ETH", quote="USDT", chain=chain,
        pool_fee=fee, pool_address="0x" + "ab" * 20,
        cex_bids=[(cex * (1 - spread) * (1 - Decimal("0.00001") * i), Decimal("1000"))
                  for i in range(5)],
        cex_asks=[(cex * (1 + spread) * (1 + Decimal("0.00001") * i), Decimal("1000"))
                  for i in range(5)],
        cex_feed_ts=ts, pool=_pool(dex),
        gas_price_wei=gas_wei, native_price_quote=cex,
        rpc_endpoint="test", run_id="t",
    )


COSTS = CostModel(
    taker_fee_bps=Decimal("7.5"), cex_legs=1, gas_units=200_000,
    rotation_cost_quote=Decimal("0"), floor_bps=Decimal("5"),
)
NOTIONALS = [Decimal("1000"), Decimal("10000")]


@pytest.fixture
def store(tmp_path):
    return ObservationStore(tmp_path / "obs.sqlite3", run_id="report-test")


class TestGrouping:
    def test_chain_and_fee_are_part_of_the_identity(self):
        """Same CEX symbol, different pool: two markets, not one sample."""
        a = _obs(0.0, chain="ethereum", fee=500)
        b = _obs(0.0, chain="arbitrum", fee=500)
        c = _obs(0.0, chain="ethereum", fee=3000)
        assert group_key(a) != group_key(b)
        assert group_key(a) != group_key(c)

    def test_the_same_market_groups_together(self):
        assert group_key(_obs(0.0)) == group_key(_obs(5.0))

    def test_each_market_gets_its_own_report(self, store):
        for ts in range(40):
            store.record(_obs(float(ts), chain="ethereum"))
            store.record(_obs(float(ts), chain="arbitrum"))
        reports = analyse_store(store, COSTS, NOTIONALS, base_is_token0=True)
        assert {r.chain for r in reports} == {"ethereum", "arbitrum"}


class TestUncertaintyIsReported:
    def test_the_mean_comes_with_an_interval_and_an_effective_n(self, store):
        for ts in range(400):
            dex = Decimal("1900") + Decimal(ts % 20) / 10
            store.record(_obs(float(ts), dex=dex))
        report = analyse_store(store, COSTS, NOTIONALS, base_is_token0=True)[0]
        assert report.gross_bps["n"] == 400
        assert report.gross_interval["mean"] is not None
        assert report.gross_interval["effective_n"] <= 400

    def test_a_short_sample_refuses_an_interval_rather_than_inventing_one(self, store):
        for ts in range(5):
            store.record(_obs(float(ts)))
        report = analyse_store(store, COSTS, NOTIONALS, base_is_token0=True)[0]
        assert report.gross_interval["lower"] is None
        assert report.gross_interval["reason"] is not None


class TestRefusalsAreCounted:
    def test_unpriceable_observations_are_reported_not_dropped(self, store):
        for ts in range(20):
            store.record(_obs(float(ts), gas_wei=10 ** 9 if ts % 2 else None))
        report = analyse_store(store, COSTS, NOTIONALS, base_is_token0=True)[0]
        assert report.observations == 20
        assert report.uncostable == 10
        assert report.gross_bps["n"] == 10

    def test_a_pool_that_cannot_be_priced_at_all_says_so(self, store):
        for ts in range(20):
            store.record(_obs(float(ts), gas_wei=None))
        report = analyse_store(store, COSTS, NOTIONALS, base_is_token0=True)[0]
        assert report.uncostable == 20
        assert report.gross_bps["n"] == 0
        assert report.gross_interval["lower"] is None


class TestOpportunityStatistics:
    def test_a_dislocated_market_shows_exceedance_above_the_floor(self, store):
        for ts in range(60):
            store.record(_obs(float(ts), cex=Decimal("1900"), dex=Decimal("1930")))
        report = analyse_store(store, COSTS, NOTIONALS, base_is_token0=True)[0]
        assert report.exceedance_net[5.0] > 0.9
        assert report.tradeable_observations > 50

    def test_a_fair_market_shows_no_exceedance(self, store):
        for ts in range(60):
            store.record(_obs(float(ts)))
        report = analyse_store(store, COSTS, NOTIONALS, base_is_token0=True)[0]
        assert report.exceedance_net[5.0] == 0.0
        assert report.tradeable_observations == 0

    def test_opportunity_lifetime_is_measured_in_seconds_not_rows(self, store):
        for ts in range(30):
            dex = Decimal("1930") if 10 <= ts < 20 else Decimal("1900")
            store.record(_obs(float(ts), dex=dex))
        report = analyse_store(store, COSTS, NOTIONALS, base_is_token0=True)[0]
        assert report.median_lifetime_seconds is not None
        assert 8 <= report.median_lifetime_seconds <= 12

    def test_the_measured_cadence_is_reported(self, store):
        for ts in range(0, 60, 3):
            store.record(_obs(float(ts)))
        report = analyse_store(store, COSTS, NOTIONALS, base_is_token0=True)[0]
        assert report.median_cadence_seconds == pytest.approx(3.0, abs=0.01)


class TestProbeComparison:
    def test_the_fixed_probe_is_compared_against_the_best_size(self, store):
        for ts in range(40):
            store.record(_obs(float(ts), cex=Decimal("1900"), dex=Decimal("1930")))
        report = analyse_store(
            store, COSTS, NOTIONALS, base_is_token0=True,
            probe_notional=Decimal("1000"),
        )[0]
        assert report.probe_understatement_bps is not None
        assert report.probe_understatement_bps >= 0


class TestLatency:
    def test_latency_degrades_a_decaying_edge(self, store):
        for ts in range(0, 120, 2):
            dex = Decimal("1930") if (ts // 2) % 2 == 0 else Decimal("1900")
            store.record(_obs(float(ts), dex=dex))
        report = analyse_store(
            store, COSTS, NOTIONALS, base_is_token0=True,
            latency_delays=(0.0, 2.0),
        )[0]
        instant = report.latency[0.0]["mean_realised_net_bps"]
        delayed = report.latency[2.0]["mean_realised_net_bps"]
        assert instant is not None and delayed is not None
        assert delayed < instant

    def test_unresolved_trades_are_counted_separately(self, store):
        for ts in (0.0, 1.0, 2.0):
            store.record(_obs(ts, dex=Decimal("1930")))
        report = analyse_store(
            store, COSTS, NOTIONALS, base_is_token0=True, latency_delays=(30.0,),
        )[0]
        assert report.latency[30.0]["unresolved"] > 0

    def test_absence_of_the_study_is_not_a_zero_latency_result(self, store):
        for ts in range(20):
            store.record(_obs(float(ts)))
        report = analyse_store(
            store, COSTS, NOTIONALS, base_is_token0=True, latency_delays=(),
        )[0]
        assert report.latency == {}


class TestProvenance:
    def test_the_cost_model_and_span_travel_with_the_report(self, store):
        for ts in range(20):
            store.record(_obs(float(ts)))
        report = analyse_store(store, COSTS, NOTIONALS, base_is_token0=True)[0]
        assert report.costs == COSTS
        assert report.span_seconds == pytest.approx(19.0)
        assert report.notionals == tuple(NOTIONALS)

    def test_an_empty_store_produces_no_reports(self, store):
        assert analyse_store(store, COSTS, NOTIONALS, base_is_token0=True) == []
