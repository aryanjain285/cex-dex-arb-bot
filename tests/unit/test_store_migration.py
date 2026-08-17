"""Schema evolution must migrate, not crash.

The controls audit flagged that `CREATE TABLE IF NOT EXISTS` plus a bumped
SCHEMA_VERSION constant produces an OperationalError against an existing
database rather than a migration -- and that `schema_version` was written per
row but never read. Proven by adding a column: every store test failed with
"table evaluations has no column named rotation_cost_quote".
"""
import sqlite3
from decimal import Decimal
from pathlib import Path

import pytest

from src.infra.evaluation_store import SCHEMA_VERSION, EvaluationRecord, EvaluationStore


def _record(**kw):
    base = dict(ts=1.0, cex_symbol="A/B", base="A", quote_cex="B",
                dex_chain="ethereum", dex_pool_fee=500, is_synthetic=False,
                outcome="taken", direction="CEX_to_DEX")
    base.update(kw)
    return EvaluationRecord(**base)


def test_a_new_column_is_added_to_an_existing_database(tmp_path: Path):
    """Simulate an older schema by dropping a column, then reopening."""
    path = tmp_path / "audit.sqlite3"
    s = EvaluationStore(path, run_id="r1")
    s.record(_record())
    s.close()

    # emulate a v1 database that predates rotation_cost_quote
    conn = sqlite3.connect(str(path))
    conn.execute("ALTER TABLE evaluations DROP COLUMN rotation_cost_quote")
    conn.commit(); conn.close()

    # reopening must migrate rather than raise
    s2 = EvaluationStore(path, run_id="r2")
    row_id = s2.record(_record(ts=2.0, rotation_cost_quote=Decimal("2")))
    assert row_id > 0
    rows = s2.all_rows()
    assert any(r["rotation_cost_quote"] == "2" for r in rows)
    s2.close()


def test_migration_preserves_existing_rows(tmp_path: Path):
    path = tmp_path / "audit.sqlite3"
    s = EvaluationStore(path, run_id="r1")
    s.record(_record(net_bps=Decimal("42")))
    s.close()

    conn = sqlite3.connect(str(path))
    conn.execute("ALTER TABLE evaluations DROP COLUMN rotation_cost_quote")
    conn.commit(); conn.close()

    s2 = EvaluationStore(path, run_id="r2")
    rows = s2.all_rows()
    assert len(rows) == 1, "the pre-migration row must survive"
    assert rows[0]["net_bps"] == "42"
    assert rows[0]["rotation_cost_quote"] is None, "backfilled as NULL, not invented"
    s2.close()


def test_schema_version_is_readable_not_merely_written(tmp_path: Path):
    """A version stamp nobody reads protects nothing."""
    s = EvaluationStore(tmp_path / "a.sqlite3", run_id="r")
    assert s.schema_version() == SCHEMA_VERSION
    s.close()


def test_every_record_field_has_a_column(tmp_path: Path):
    """Structural guard: the dataclass and the table cannot drift apart.

    This is the defect that just occurred -- a patch updated the field list and
    the decimal-column tuple but missed CREATE TABLE, so every insert failed.
    """
    from dataclasses import fields

    s = EvaluationStore(tmp_path / "a.sqlite3", run_id="r")
    cols = {r[1] for r in s._conn.execute("PRAGMA table_info(evaluations)")}
    missing = {f.name for f in fields(EvaluationRecord)} - cols
    assert not missing, f"EvaluationRecord fields with no column: {missing}"
    s.close()
