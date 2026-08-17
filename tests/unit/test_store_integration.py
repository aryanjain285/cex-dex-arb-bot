"""End-to-end: a detection cycle must land rows in a real SQLite store.

The unit tests use a collecting fake, which proves the detector emits records
but not that they survive the store's type enforcement and uniqueness
constraints. This closes that gap with the production store.
"""
from decimal import Decimal

import pytest

from src.core.config import StrategyConfig
from src.infra.evaluation_store import EvaluationStore
from src.strategy.detector import OpportunityDetector, RejectionReason
from tests.fakes import D, FakeCex, FakeDex, flat_book, make_pair


@pytest.fixture
def store(tmp_path):
    s = EvaluationStore(tmp_path / "audit.sqlite3", run_id="itest")
    yield s
    s.close()


def strategy(**kw) -> StrategyConfig:
    defaults = dict(target_notional_usd=1000, taker_fee_bps=D("7.5"), min_net_bps=D(5))
    defaults.update(kw)
    return StrategyConfig(**defaults)


async def test_a_cycle_persists_two_rows_per_pair(store):
    pairs = [make_pair("ETH/USDT"), make_pair("ARB/USDT", base="ARB")]
    cex = FakeCex({"ETH/USDT": flat_book(1000, 1000), "ARB/USDT": flat_book(1, 1)})
    dex = FakeDex(sell_price=1050, buy_price=1050)

    await OpportunityDetector(strategy(), cex, dex, pairs, store=store).detect()

    assert store.count() == 4, "two directions x two pairs"
    assert store.count(outcome="taken") >= 1


async def test_persisted_rows_survive_type_enforcement_and_are_exact(store):
    """The detector's Decimals must pass the store's float rejection, and
    round-trip without loss."""
    pair = make_pair()
    cex = FakeCex({"ETH/USDT": flat_book(1000, 1000)})
    dex = FakeDex(sell_price=1050, buy_price=1050, gas=D("0.017180312464164"))

    await OpportunityDetector(strategy(), cex, dex, [pair], store=store).detect()

    taken = [r for r in store.all_rows() if r["outcome"] == "taken"]
    assert taken, "expected a taken row"
    row = taken[0]
    assert Decimal(row["gas_quote"]) == D("0.017180312464164")
    recomputed = (Decimal(row["gross_quote"]) - Decimal(row["cex_fee_quote"])
                  - Decimal(row["gas_quote"]))
    assert recomputed == Decimal(row["net_quote"])


async def test_repeated_cycles_do_not_violate_the_uniqueness_constraint(store):
    """Rows differ by timestamp, so successive cycles must all persist."""
    import asyncio

    pair = make_pair()
    cex = FakeCex({"ETH/USDT": flat_book(1000, 1000)})
    dex = FakeDex(sell_price=1050, buy_price=1050)
    det = OpportunityDetector(strategy(), cex, dex, [pair], store=store)

    for _ in range(3):
        await det.detect()
        await asyncio.sleep(0.01)

    assert store.count() == 6, f"3 cycles x 2 directions, got {store.count()}"


async def test_markout_ladder_is_computable_from_a_real_run(store):
    """The payoff: markout falls out of the persisted series as a query."""
    import asyncio

    pair = make_pair()
    cex = FakeCex({"ETH/USDT": flat_book(1000, 1000)})
    dex = FakeDex(sell_price=1050, buy_price=1050)
    det = OpportunityDetector(strategy(), cex, dex, [pair], store=store)

    for _ in range(4):
        await det.detect()
        await asyncio.sleep(0.12)

    taken = [r for r in store.all_rows()
             if r["outcome"] == "taken" and r["direction"] == "CEX_to_DEX"]
    assert len(taken) >= 2
    ladder = store.markout(taken[0]["id"], offsets_seconds=[0.12], tolerance=0.1)
    assert ladder[0.12] is not None, "a later observation of the same direction exists"


async def test_rejection_reasons_are_queryable(store):
    """The near-miss distribution must be filterable, which is the entire
    point of persisting rejections."""
    pair = make_pair()
    cex = FakeCex({"ETH/USDT": flat_book(1000, 1000)})
    dex = FakeDex(sell_price=1000.4, buy_price=1000.4)

    await OpportunityDetector(
        strategy(min_net_bps=D(50)), cex, dex, [pair], store=store
    ).detect()

    rows = store.all_rows()
    reasons = {r["reason"] for r in rows}
    assert RejectionReason.BELOW_FLOOR in reasons
    below = [r for r in rows if r["reason"] == RejectionReason.BELOW_FLOOR]
    assert Decimal(below[0]["min_net_bps"]) == D(50)
    assert below[0]["net_bps"] is not None
