"""Durable, auditable record of every evaluation the bot performs.

Paper trading previously produced one stdout line per crossing and nothing
else, so a multi-week run yielded a scrollback buffer rather than a dataset.

Two properties matter most here:

- EVERY evaluation is persisted, not only the ones that cross the threshold.
  The near-misses are where you learn whether an edge is real, how long it
  survives, and whether the threshold is set anywhere near correctly.

- Money values round-trip EXACTLY. They are stored as text, never as floats.
  A binary float cannot represent 0.1, and an audit trail whose numbers drift
  from the numbers the decision used is not an audit trail.
"""
from decimal import Decimal
from pathlib import Path

import pytest

from src.infra.evaluation_store import (
    SCHEMA_VERSION,
    EvaluationRecord,
    EvaluationStore,
)


def D(x) -> Decimal:
    return Decimal(str(x))


@pytest.fixture
def store(tmp_path: Path):
    s = EvaluationStore(tmp_path / "audit.sqlite3", run_id="run-1")
    yield s
    s.close()


def taken_record(**overrides) -> EvaluationRecord:
    # Derived, not hardcoded, so the fixture cannot drift out of self-consistency.
    _gross, _fee, _gas = D("9.3100000000000001"), D("0.75"), D("0.017180312464164")
    _net = _gross - _fee - _gas
    _notional = D("1000")
    base = dict(
        ts=1_700_000_000.0,
        cex_symbol="ETH/USDT",
        base="WETH",
        quote_cex="USDT",
        dex_chain="ethereum",
        dex_pool_fee=500,
        is_synthetic=False,
        direction="CEX_to_DEX",
        outcome="taken",
        reason=None,
        size_base=D("0.5256628608675539"),
        notional_quote=D("1000"),
        cex_price=D("1902.36"),
        cex_best_bid=D("1902.35"),
        cex_best_ask=D("1902.36"),
        dex_price=D("1920.071292139320002"),
        gross_quote=_gross,
        cex_fee_quote=_fee,
        gas_quote=_gas,
        net_quote=_net,
        net_bps=_net / _notional * D(10000),
        cex_legs=1,
        book_age_s=0.021,
        depth_levels_used=1,
        min_net_bps=D("5"),
        taker_fee_bps=D("7.5"),
    )
    base.update(overrides)
    return EvaluationRecord(**base)


# --------------------------------------------------------------------------

def test_records_a_taken_evaluation_and_reads_it_back(store: EvaluationStore):
    row_id = store.record(taken_record())
    assert row_id > 0

    rows = store.all_rows()
    assert len(rows) == 1
    assert rows[0]["cex_symbol"] == "ETH/USDT"
    assert rows[0]["outcome"] == "taken"
    assert rows[0]["run_id"] == "run-1"
    assert rows[0]["schema_version"] == SCHEMA_VERSION


def test_rejected_evaluations_are_persisted_too(store: EvaluationStore):
    """The requirement: every evaluation, not just crossings."""
    store.record(taken_record())
    store.record(taken_record(outcome="rejected", reason="below_floor",
                              net_bps=D("2.1")))
    store.record(taken_record(outcome="rejected", reason="insufficient_depth",
                              direction=None, net_bps=None, net_quote=None))

    rows = store.all_rows()
    assert len(rows) == 3
    outcomes = sorted(r["outcome"] for r in rows)
    assert outcomes == ["rejected", "rejected", "taken"]
    reasons = {r["reason"] for r in rows}
    assert reasons == {None, "below_floor", "insufficient_depth"}


def test_money_values_round_trip_exactly(store: EvaluationStore):
    """The audit property. Stored as text, recovered as the same Decimal.

    A float column would silently alter these digits, so the recorded trail
    would disagree with the arithmetic the decision was made on.
    """
    awkward = D("0.1") + D("0.2")          # 0.3 exactly in Decimal
    precise = D("1902.071292139320002083528893409411130282")
    store.record(taken_record(net_quote=awkward, dex_price=precise))

    row = store.all_rows()[0]
    assert Decimal(row["net_quote"]) == awkward
    assert Decimal(row["net_quote"]) == D("0.3")
    assert Decimal(row["dex_price"]) == precise
    # and prove the float path would have lost it
    assert float(0.1) + float(0.2) != 0.3


def test_nulls_are_preserved_rather_than_coerced(store: EvaluationStore):
    store.record(taken_record(outcome="rejected", reason="no_book",
                              direction=None, size_base=None, cex_price=None,
                              dex_price=None, net_quote=None, net_bps=None))
    row = store.all_rows()[0]
    assert row["direction"] is None
    assert row["net_bps"] is None
    assert row["size_base"] is None


def test_inputs_are_stored_so_any_row_can_be_reverified(store: EvaluationStore):
    """Auditability: a row must be interpretable without knowing which config
    was live, and its arithmetic must be independently reproducible."""
    store.record(taken_record())
    row = store.all_rows()[0]

    # the fee and floor in force at decision time
    assert Decimal(row["taker_fee_bps"]) == D("7.5")
    assert Decimal(row["min_net_bps"]) == D("5")
    # re-derive net from the stored components
    recomputed = (
        Decimal(row["gross_quote"])
        - Decimal(row["cex_fee_quote"])
        - Decimal(row["gas_quote"])
    )
    assert recomputed == Decimal(row["net_quote"])


def test_top_of_book_is_stored_so_the_depth_effect_is_measurable(store):
    """Storing both the VWAP used and the best bid/ask makes it possible to
    quantify, after the fact, how much depth-blindness would have cost."""
    store.record(taken_record(cex_price=D("1910"), cex_best_ask=D("1902.36")))
    row = store.all_rows()[0]
    vwap = Decimal(row["cex_price"])
    top = Decimal(row["cex_best_ask"])
    assert (vwap / top - 1) * 10000 > 0


def test_run_id_separates_runs(tmp_path: Path):
    path = tmp_path / "audit.sqlite3"
    first = EvaluationStore(path, run_id="run-A")
    first.record(taken_record())
    first.close()

    second = EvaluationStore(path, run_id="run-B")
    second.record(taken_record())
    rows = second.all_rows()
    second.close()

    assert {r["run_id"] for r in rows} == {"run-A", "run-B"}


def test_survives_reopen(tmp_path: Path):
    path = tmp_path / "audit.sqlite3"
    s = EvaluationStore(path, run_id="r")
    s.record(taken_record())
    s.close()

    s2 = EvaluationStore(path, run_id="r")
    assert len(s2.all_rows()) == 1
    s2.close()


def test_markout_reads_later_observations_of_the_same_pair(store: EvaluationStore):
    """Markout comes free from the evaluation series.

    Because every evaluation is persisted, the edge at t+N is just a later row
    for the same pair -- no extra RPC calls, no extra exchange weight, and no
    separate subsystem. The detector's loop interval sets the resolution.
    """
    t0 = 1_700_000_000.0
    anchor = store.record(taken_record(ts=t0, net_bps=D("80")))
    store.record(taken_record(ts=t0 + 0.5, net_bps=D("60")))
    store.record(taken_record(ts=t0 + 1.0, net_bps=D("30")))
    store.record(taken_record(ts=t0 + 5.0, net_bps=D("-4")))
    # a different pair must not leak into the ladder
    store.record(taken_record(ts=t0 + 1.0, cex_symbol="OTHER/USDT", net_bps=D("999")))

    ladder = store.markout(anchor, offsets_seconds=[0.5, 1.0, 5.0], tolerance=0.25)

    assert ladder[0.5] == D("60")
    assert ladder[1.0] == D("30")
    assert ladder[5.0] == D("-4")


def test_markout_reports_none_when_no_observation_is_near_the_offset(store):
    t0 = 1_700_000_000.0
    anchor = store.record(taken_record(ts=t0, net_bps=D("80")))
    store.record(taken_record(ts=t0 + 30.0, net_bps=D("10")))

    ladder = store.markout(anchor, offsets_seconds=[1.0], tolerance=0.25)
    assert ladder[1.0] is None
