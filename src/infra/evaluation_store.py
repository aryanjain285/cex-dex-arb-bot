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
SCHEMA_VERSION = 1

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
    "net_quote",
    "net_bps",
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
    net_quote: Optional[Decimal] = None
    net_bps: Optional[Decimal] = None
    cex_legs: Optional[int] = None
    book_age_s: Optional[float] = None
    depth_levels_used: Optional[int] = None
    min_net_bps: Optional[Decimal] = None
    taker_fee_bps: Optional[Decimal] = None


_RECORD_COLUMNS = [f.name for f in fields(EvaluationRecord)]

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
    net_quote         TEXT,
    net_bps           TEXT,
    cex_legs          INTEGER,
    book_age_s        REAL,
    depth_levels_used INTEGER,
    min_net_bps       TEXT,
    taker_fee_bps     TEXT
);
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
        # WAL: durable across process death, and readers never block the writer,
        # so the dataset can be queried while a run is in progress.
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.executescript(_CREATE_TABLE)
        logger.info(f"Evaluation store open at {self.path} (run_id={run_id}).")

    # ------------------------------------------------------------------

    def record(self, record: EvaluationRecord) -> int:
        """Persist one evaluation. Returns its row id."""
        payload = asdict(record)
        for column in _DECIMAL_COLUMNS:
            value = payload.get(column)
            # str() on a Decimal is exact and losslessly reversible.
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

        For each offset, the observation closest to `anchor.ts + offset` is
        used, provided it falls within `tolerance` seconds. Offsets with no
        nearby observation return None rather than the nearest available row,
        because silently substituting a distant sample would misrepresent how
        fast the edge decayed -- which is the entire question markout answers.
        """
        anchor = self._conn.execute(
            "SELECT ts, cex_symbol, direction FROM evaluations WHERE id = ?",
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
                  AND id != ?
                  AND ts BETWEEN ? AND ?
                  AND net_bps IS NOT NULL
                ORDER BY ABS(ts - ?) ASC
                LIMIT 1
                """,
                (
                    anchor["cex_symbol"],
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
