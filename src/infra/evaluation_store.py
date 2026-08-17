"""Durable audit trail of every evaluation the detector performs.

Design commitments, each of which exists for a specific reason:

- EVERY evaluation is recorded, not only the profitable ones. A threshold
  crossing tells you almost nothing on its own; the distribution of near
  misses is what reveals whether an edge exists, how large it typically is,
  and whether the threshold is set anywhere near the right place.

- Money is stored as TEXT and reconstructed with Decimal. SQLite's REAL is an
  IEEE-754 double, which cannot represent 0.1. Storing decision arithmetic in
  a lossy type produces an audit trail that disagrees with the decision, which
  is worse than having none.

- The INPUTS are stored alongside the outputs -- the taker fee and the net
  floor in force, plus best bid/ask next to the VWAP actually used. That makes
  each row independently re-derivable long after the config has changed, and
  makes the cost of depth-blindness measurable after the fact.

- Markout needs no extra machinery. Because every evaluation is persisted, the
  edge at t+N is simply a later row for the same pair, so the ladder is a
  query rather than a subsystem: no additional RPC calls and no additional
  exchange rate-limit weight. The detector's loop interval sets the resolution.
"""
from __future__ import annotations

import sqlite3
from dataclasses import asdict, dataclass, fields
from decimal import Decimal
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

from loguru import logger

__all__ = ["SCHEMA_VERSION", "EvaluationRecord", "EvaluationStore"]

# Bump when the column set changes so old rows stay interpretable.
SCHEMA_VERSION = 4

# Columns holding exact decimal quantities. Stored as TEXT, never REAL.
_DECIMAL_COLUMNS = (
    "size_base",
    "notional_quote",
    "cex_price",
    "cex_best_bid",
    "cex_best_ask",
    "dex_price",
    "gross_quote",
    "cex_fee_quote",
    "gas_quote",
    "rotation_cost_quote",
    "net_quote",
    "net_bps",
    "placebo_net_bps",
    "min_net_bps",
    "taker_fee_bps",
)


@dataclass
class EvaluationRecord:
    """One evaluation of one direction of one pair, at one instant.

    `outcome` is "taken" or "rejected". `reason` carries the rejection cause
    and is None for a taken evaluation. Fields are Optional because a rejection
    can occur before the economics were computable -- for example when no
    order book was available at all.
    """

    ts: float
    cex_symbol: str
    base: str
    quote_cex: str
    dex_chain: str
    dex_pool_fee: int
    is_synthetic: bool
    outcome: str
    direction: Optional[str] = None
    reason: Optional[str] = None
    size_base: Optional[Decimal] = None
    notional_quote: Optional[Decimal] = None
    cex_price: Optional[Decimal] = None
    cex_best_bid: Optional[Decimal] = None
    cex_best_ask: Optional[Decimal] = None
    dex_price: Optional[Decimal] = None
    gross_quote: Optional[Decimal] = None
    cex_fee_quote: Optional[Decimal] = None
    gas_quote: Optional[Decimal] = None
    rotation_cost_quote: Optional[Decimal] = None
    net_quote: Optional[Decimal] = None
    net_bps: Optional[Decimal] = None
    # Net edge the same book would have shown against a DEX quote from
    # `placebo.delay_cycles` ago. Under the null that the edge is a
    # staleness artefact, this matches net_bps.
    placebo_net_bps: Optional[Decimal] = None
    cex_legs: Optional[int] = None
    book_age_s: Optional[float] = None
    depth_levels_used: Optional[int] = None
    min_net_bps: Optional[Decimal] = None
    taker_fee_bps: Optional[Decimal] = None
    # Mode-independent token-policy label: "allowed", "not_allowlisted" or
    # "denied". Recorded on every row so a broad measurement run still
    # yields an honest tradeable subset, instead of mixing real edges with a
    # fee-on-transfer token's transfer tax appearing as one.
    policy_verdict: Optional[str] = None


_RECORD_COLUMNS = [f.name for f in fields(EvaluationRecord)]

# SQL types for the non-decimal columns, used when migrating an older table.
_COLUMN_TYPES = {
    "ts": "REAL",
    "dex_pool_fee": "INTEGER",
    "is_synthetic": "INTEGER",
    "cex_legs": "INTEGER",
    "book_age_s": "REAL",
    "depth_levels_used": "INTEGER",
}

_CREATE_TABLE = f"""
CREATE TABLE IF NOT EXISTS evaluations (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id            TEXT    NOT NULL,
    schema_version    INTEGER NOT NULL,
    ts                REAL    NOT NULL,
    cex_symbol        TEXT    NOT NULL,
    base              TEXT    NOT NULL,
    quote_cex         TEXT    NOT NULL,
    dex_chain         TEXT    NOT NULL,
    dex_pool_fee      INTEGER NOT NULL,
    is_synthetic      INTEGER NOT NULL,
    outcome           TEXT    NOT NULL,
    direction         TEXT,
    reason            TEXT,
    size_base         TEXT,
    notional_quote    TEXT,
    cex_price         TEXT,
    cex_best_bid      TEXT,
    cex_best_ask      TEXT,
    dex_price         TEXT,
    gross_quote       TEXT,
    cex_fee_quote     TEXT,
    gas_quote         TEXT,
    rotation_cost_quote TEXT,
    net_quote         TEXT,
    net_bps           TEXT,
    cex_legs          INTEGER,
    book_age_s        REAL,
    depth_levels_used INTEGER,
    min_net_bps       TEXT,
    taker_fee_bps     TEXT,
    policy_verdict    TEXT
);
-- A replayed or double-recorded evaluation would bias every count-based
-- statistic drawn from the dataset, so identical rows are refused.
CREATE UNIQUE INDEX IF NOT EXISTS idx_eval_unique
    ON evaluations (run_id, ts, cex_symbol, direction, outcome);
CREATE INDEX IF NOT EXISTS idx_eval_symbol_ts ON evaluations (cex_symbol, ts);
CREATE INDEX IF NOT EXISTS idx_eval_run       ON evaluations (run_id);
CREATE INDEX IF NOT EXISTS idx_eval_outcome   ON evaluations (outcome, ts);
"""


class EvaluationStore:
    def __init__(self, path: Union[str, Path], run_id: str):
        self.path = Path(path)
        self.run_id = run_id
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.path), isolation_level=None)
        self._conn.row_factory = sqlite3.Row
        # WAL so readers never block the writer, and the dataset can be
        # queried while a run is in progress.
        self._conn.execute("PRAGMA journal_mode=WAL")
        # FULL, not NORMAL: under WAL, NORMAL does not fsync commits, so an
        # OS crash or power loss can lose recent rows. An audit trail must
        # survive more than process death.
        self._conn.execute("PRAGMA synchronous=FULL")
        self._conn.executescript(_CREATE_TABLE)
        self._migrate()
        logger.info(f"Evaluation store open at {self.path} (run_id={run_id}).")


    # ------------------------------------------------------------------

    def _migrate(self) -> None:
        """Bring an existing table up to the current column set.

        `CREATE TABLE IF NOT EXISTS` does nothing to a table that already
        exists, so adding a field to `EvaluationRecord` previously produced
        `OperationalError: table evaluations has no column named ...` on the
        next insert -- a crash rather than a migration. That happened for real
        when `rotation_cost_quote` was added.

        Only additive migrations are handled, which is the only kind this
        schema needs: existing rows are backfilled as NULL rather than having a
        value invented for them, because a fabricated value in an audit trail
        is worse than a missing one.
        """
        existing = {row[1] for row in self._conn.execute("PRAGMA table_info(evaluations)")}
        for column in _RECORD_COLUMNS:
            if column in existing:
                continue
            sql_type = "TEXT" if column in _DECIMAL_COLUMNS else _COLUMN_TYPES.get(column, "TEXT")
            self._conn.execute(f"ALTER TABLE evaluations ADD COLUMN {column} {sql_type}")
            logger.warning(
                f"Migrated evaluation store: added column {column} ({sql_type}). "
                f"Pre-existing rows carry NULL for it."
            )
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS schema_meta ("
            "  key TEXT PRIMARY KEY, value TEXT NOT NULL)"
        )
        self._conn.execute(
            "INSERT INTO schema_meta (key, value) VALUES ('schema_version', ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (str(SCHEMA_VERSION),),
        )

    def schema_version(self) -> int:
        """The stored schema version.

        Written per row AND recorded once here, so it can actually be checked
        by a reader -- a version stamp nobody reads protects nothing.
        """
        row = self._conn.execute(
            "SELECT value FROM schema_meta WHERE key = 'schema_version'"
        ).fetchone()
        return int(row[0]) if row else 0

    # ------------------------------------------------------------------

    def record(self, record: EvaluationRecord) -> int:
        """Persist one evaluation. Returns its row id.

        Raises TypeError if any money field was passed as a float. The
        Decimal-as-TEXT discipline is the store's central guarantee, and it was
        previously documented but unenforced -- a single careless caller could
        persist an IEEE-754 artifact (0.1 + 0.2 became '0.30000000000000004'),
        silently reintroducing into the audit trail exactly the imprecision the
        design exists to exclude. Enforced here rather than trusted.
        """
        payload = asdict(record)
        for column in _DECIMAL_COLUMNS:
            value = payload.get(column)
            if isinstance(value, float):
                raise TypeError(
                    f"{column} must be a Decimal or int, not float "
                    f"(got {value!r}). Floats cannot represent decimal money "
                    f"exactly; wrap it as Decimal(str(value)) at the source."
                )
            # str() on a Decimal or int is exact and losslessly reversible.
            payload[column] = None if value is None else str(value)
        payload["is_synthetic"] = 1 if record.is_synthetic else 0
        payload["run_id"] = self.run_id
        payload["schema_version"] = SCHEMA_VERSION

        columns = ["run_id", "schema_version"] + _RECORD_COLUMNS
        placeholders = ", ".join(f":{c}" for c in columns)
        cursor = self._conn.execute(
            f"INSERT INTO evaluations ({', '.join(columns)}) VALUES ({placeholders})",
            payload,
        )
        return int(cursor.lastrowid)

    def all_rows(self) -> List[Dict]:
        rows = self._conn.execute(
            "SELECT * FROM evaluations ORDER BY id"
        ).fetchall()
        return [dict(r) for r in rows]

    def count(self, outcome: Optional[str] = None) -> int:
        if outcome is None:
            sql, args = "SELECT COUNT(*) FROM evaluations", ()
        else:
            sql, args = "SELECT COUNT(*) FROM evaluations WHERE outcome = ?", (outcome,)
        return int(self._conn.execute(sql, args).fetchone()[0])

    # ------------------------------------------------------------------

    def markout(
        self,
        evaluation_id: int,
        offsets_seconds: Sequence[float],
        tolerance: float = 0.25,
    ) -> Dict[float, Optional[Decimal]]:
        """Net edge for the same pair at each offset after the anchor row.

        Matched on the same pair AND the same direction AND the same run.
        Filtering on direction is essential: both directions of a pair are
        evaluated in the same cycle and land milliseconds apart, so a
        symbol-only match would happily return the opposite direction's row --
        whose net_bps is roughly the negative of the anchor's, turning the
        decay curve into a coin flip on intra-cycle write order.

        For each offset, the observation closest to `anchor.ts + offset` is
        used, provided it falls within `tolerance` seconds. Ties break on `id`
        so the result is deterministic rather than dependent on SQLite's
        row ordering. Offsets with no
        nearby observation return None rather than the nearest available row,
        because silently substituting a distant sample would misrepresent how
        fast the edge decayed -- which is the entire question markout answers.
        """
        anchor = self._conn.execute(
            "SELECT ts, cex_symbol, direction, run_id FROM evaluations WHERE id = ?",
            (evaluation_id,),
        ).fetchone()
        if anchor is None:
            raise KeyError(f"no evaluation with id {evaluation_id}")

        ladder: Dict[float, Optional[Decimal]] = {}
        for offset in offsets_seconds:
            target = anchor["ts"] + offset
            row = self._conn.execute(
                """
                SELECT net_bps, ts FROM evaluations
                WHERE cex_symbol = ?
                  AND direction IS ?
                  AND run_id = ?
                  AND id != ?
                  AND ts BETWEEN ? AND ?
                  AND net_bps IS NOT NULL
                ORDER BY ABS(ts - ?) ASC, id ASC
                LIMIT 1
                """,
                (
                    anchor["cex_symbol"],
                    anchor["direction"],
                    anchor["run_id"],
                    evaluation_id,
                    target - tolerance,
                    target + tolerance,
                    target,
                ),
            ).fetchone()
            ladder[offset] = None if row is None else Decimal(row["net_bps"])
        return ladder

    # ------------------------------------------------------------------

    def close(self) -> None:
        try:
            self._conn.close()
        except Exception as exc:  # pragma: no cover
            logger.warning(f"Failed to close the evaluation store cleanly: {exc}")
