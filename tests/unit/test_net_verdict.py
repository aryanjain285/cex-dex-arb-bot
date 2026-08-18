"""The report states its statistical verdict rather than leaving it to be eyeballed.

A negative mean is not a result. A negative mean whose confidence interval excludes zero
is. And the two failure modes are opposite in what they license:

    interval entirely below zero   a positive net edge is excluded -- act on that
    interval spans zero            nothing is established -- collect more, and note
                                   that EFFECTIVE n governs how much more, not the
                                   raw count
    interval entirely above zero   check the negative control and the barrier flag
                                   before believing it

Printing only a mean invites the first reading regardless of which case holds, which is
how a run with 200 observations and an effective sample size of 9 comes to be reported as
a conclusion.
"""
from decimal import Decimal

import pytest

from src.exchange.pool_state import PoolSnapshot
from src.exchange.univ3_math import TickInfo
from src.research.evaluate import CostModel
from src.research.observations import Observation, ObservationStore
from src.research.report import analyse_store, format_report


def _pool(price=Decimal("1900")):
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


def _obs(ts, cex=Decimal("1900"), dex=Decimal("1900")):
    spread = Decimal("0.00005")
    return Observation(
        ts=ts, cex_symbol="ETHUSDT", base="ETH", quote="USDT", chain="ethereum",
        pool_fee=500, pool_address="0x" + "ab" * 20,
        cex_bids=[(cex * (1 - spread), Decimal("1000"))],
        cex_asks=[(cex * (1 + spread), Decimal("1000"))],
        cex_feed_ts=ts, pool=_pool(dex),
        gas_price_wei=10 ** 9, native_price_quote=cex,
    )


COSTS = CostModel(
    taker_fee_bps=Decimal("7.5"), cex_legs=1, gas_units=200_000,
    rotation_cost_quote=Decimal("0"), floor_bps=Decimal("5"),
)
NOTIONALS = [Decimal("1000")]


@pytest.fixture
def store(tmp_path):
    return ObservationStore(tmp_path / "obs.sqlite3", run_id="verdict")


def _report(store):
    return analyse_store(store, COSTS, NOTIONALS, base_is_token0=True,
                         latency_delays=())[0]


def test_a_clearly_negative_edge_is_stated_as_excluded(store):
    """Venues at parity: net is about -10 bps and never near zero."""
    for i in range(200):
        store.record(_obs(float(i * 30), dex=Decimal("1900") + Decimal(i % 3)))
    text = format_report(_report(store))
    assert "95% CI" in text
    assert "excluded at 95% confidence" in text, text


def test_a_clearly_positive_edge_demands_the_control_be_checked(store):
    """A large dislocation. The verdict must not simply endorse it -- every positive
    found in this project so far has been a trap."""
    for i in range(200):
        store.record(_obs(float(i * 30), cex=Decimal("1900"),
                          dex=Decimal("1960") + Decimal(i % 3)))
    text = format_report(_report(store))
    assert "ABOVE zero" in text
    assert "negative control" in text, text


def test_a_short_sample_says_no_interval_rather_than_a_verdict(store):
    for i in range(4):
        store.record(_obs(float(i * 30)))
    text = format_report(_report(store))
    assert "no net interval" in text
    assert "VERDICT" not in text


def test_the_effective_sample_size_appears_beside_the_raw_one(store):
    """The number that governs how much more data is needed. Printing only the raw count
    is how 200 observations with an effective size of 9 becomes a conclusion."""
    for i in range(200):
        store.record(_obs(float(i * 30), dex=Decimal("1900") + Decimal(i % 3)))
    text = format_report(_report(store))
    assert "effective n" in text
    report = _report(store)
    assert report.net_interval["effective_n"] <= report.net_interval["n"]
