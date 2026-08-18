"""Raw market state, recorded so it can be RE-QUOTED later.

The distinction from the evaluation store is the whole design. That one records
decisions -- one price, at one size, under one cost model, with its verdict -- which
is right for an audit trail and useless for research, because every interesting
question is one it cannot answer:

    "would $5,000 have worked?"         the price was quoted at $1,000 only
    "what if execution took 2s?"        no successor state was kept
    "how deep was the book really?"     only the touch was stored
    "what if the fee tier were 0.05%?"  the quote had 0.30% baked in

A pool snapshot is different in kind. It holds sqrtPriceX96, the active liquidity
and the initialised ticks around the price, so the local swap math can quote it at
ANY size, months later, under any cost assumption. A full CEX ladder can be
re-walked at any notional. Recording those two things, rather than their
conclusions, is what turns a log into a dataset.

Consequences worth stating, because they set what the analysis may claim:

GAS IS STORED AS A PRICE, NOT A COST. `gas_quote` would bake in an assumed gas
limit. The raw gas price plus the native token's quote price lets any limit be
applied later -- including the measured one, once real receipts exist.

PROVENANCE TRAVELS WITH THE ROW. Which endpoint answered, which block, which run.
Public endpoints drop requests and disagree with each other, so a number whose
source is unknown cannot honestly be compared against one from another run.

LOSSLESSNESS IS THE CONTRACT. Decimals are stored as text and large integers as
text, because both failure modes here are silent: a float round trip moves prices
in the last places, and this strategy's entire signal is 5-20 bps; a dropped tick
changes every large size while leaving small ones exact, so a smoke test passes
while the capacity analysis is wrong.
"""
from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path
from typing import Iterable, Iterator, List, Optional, Sequence, Tuple, Union

from loguru import logger

from ..exchange.pool_state import PoolSnapshot

__all__ = [
    "Observation", "ObservationStore", "SCHEMA_VERSION",
    "mid_dislocation_bps",
]

# Bumped when the meaning of a column changes. Additive columns do not need it;
# a reinterpretation of existing data does, because otherwise two incompatible
# generations of rows sit in one table looking identical.
SCHEMA_VERSION = 1

Level = Tuple[Decimal, Decimal]


@dataclass(frozen=True)
class Observation:
    """One instant of both venues, recorded in full.

    Frozen: an observation is a fact about a past moment. Analysis that mutated one
    would be rewriting the record it is drawing conclusions from.
    """

    ts: float
    cex_symbol: str
    base: str
    quote: str
    chain: str
    pool_fee: int
    pool_address: str
    # Full ladders, not the touch. The touch cannot answer a capacity question,
    # and a synthesised ladder answers it with an assumption.
    cex_bids: Sequence[Level]
    cex_asks: Sequence[Level]
    pool: PoolSnapshot
    cex_feed_ts: Optional[float] = None
    # Raw gas price. See the module docstring: not a cost.
    gas_price_wei: Optional[int] = None
    native_price_quote: Optional[Decimal] = None
    rpc_endpoint: Optional[str] = None
    run_id: Optional[str] = None

    def __post_init__(self):
        # Tuples so a caller cannot append to a recorded fact through the list it
        # happened to pass in.
        object.__setattr__(self, "cex_bids", tuple(
            (Decimal(str(p)), Decimal(str(s))) for p, s in self.cex_bids
        ))
        object.__setattr__(self, "cex_asks", tuple(
            (Decimal(str(p)), Decimal(str(s))) for p, s in self.cex_asks
        ))

    # -- convenience, so callers do not each re-derive these ---------------

    @property
    def best_bid(self) -> Optional[Decimal]:
        return self.cex_bids[0][0] if self.cex_bids else None

    @property
    def best_ask(self) -> Optional[Decimal]:
        return self.cex_asks[0][0] if self.cex_asks else None

    @property
    def cex_mid(self) -> Optional[Decimal]:
        if not self.cex_bids or not self.cex_asks:
            return None
        return (self.cex_bids[0][0] + self.cex_asks[0][0]) / 2

    def gas_quote(self, gas_units: int) -> Optional[Decimal]:
        """Gas cost in the quote currency under an explicit gas-limit assumption.

        Returns None rather than zero when the inputs are missing. A zero gas cost
        is the single easiest way to make this strategy look profitable, since its
        edge and its gas are the same order of magnitude.
        """
        if self.gas_price_wei is None or self.native_price_quote is None:
            return None
        native_units = Decimal(self.gas_price_wei) * Decimal(gas_units) / Decimal(10 ** 18)
        return native_units * self.native_price_quote


# --- storage --------------------------------------------------------------

_CREATE = """
CREATE TABLE IF NOT EXISTS observations (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    schema_version  INTEGER NOT NULL,
    ts              REAL    NOT NULL,
    cex_symbol      TEXT    NOT NULL,
    base            TEXT    NOT NULL,
    quote           TEXT    NOT NULL,
    chain           TEXT    NOT NULL,
    pool_fee        INTEGER NOT NULL,
    pool_address    TEXT    NOT NULL,
    cex_bids        TEXT    NOT NULL,
    cex_asks        TEXT    NOT NULL,
    cex_feed_ts     REAL,
    pool_state      TEXT    NOT NULL,
    gas_price_wei   TEXT,
    native_price_quote TEXT,
    rpc_endpoint    TEXT,
    run_id          TEXT
)
"""

# ts first: every read is time-ordered or time-windowed, and every analysis groups
# by pair. Without this, a multi-day store makes each read a full scan.
_INDICES = (
    "CREATE INDEX IF NOT EXISTS idx_obs_ts ON observations(ts)",
    "CREATE INDEX IF NOT EXISTS idx_obs_pair_ts ON observations(cex_symbol, ts)",
)

_INSERT_COLUMNS = (
    "schema_version", "ts", "cex_symbol", "base", "quote", "chain", "pool_fee",
    "pool_address", "cex_bids", "cex_asks", "cex_feed_ts", "pool_state",
    "gas_price_wei", "native_price_quote", "rpc_endpoint", "run_id",
)


def _levels_to_json(levels: Sequence[Level]) -> str:
    # Strings, not JSON numbers: a JSON number is a float on the way back, and
    # 1896.62 does not survive that intact.
    return json.dumps([[str(p), str(s)] for p, s in levels])


def _levels_from_json(raw: str) -> Tuple[Level, ...]:
    return tuple((Decimal(p), Decimal(s)) for p, s in json.loads(raw))


def _pool_to_json(pool: PoolSnapshot) -> str:
    row = pool.to_row()
    # to_row already renders the big integers as strings; json.dumps would turn
    # them back into numbers if they were ints, so assert the contract holds
    # rather than trusting it silently.
    for key in ("sqrt_price_x96", "liquidity"):
        if key in row and not isinstance(row[key], str):
            row[key] = str(row[key])
    return json.dumps(row)


def _pool_from_json(raw: str) -> PoolSnapshot:
    return PoolSnapshot.from_row(json.loads(raw))


class ObservationStore:
    """Append-only store of raw market observations.

    Append-only by design: there is no update method. An observation is a fact
    about a past instant, and a dataset whose history can be edited cannot support
    a claim about what the market did.
    """

    def __init__(self, path: Union[str, Path], run_id: str):
        self.path = Path(path)
        self.run_id = run_id
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # check_same_thread=False: the recorder writes from an asyncio task while a
        # separate read may come from the reporting path. Writes are serialised by
        # the connection's own lock.
        self._conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        # WAL so analysis can read the file WHILE the recorder is still writing to
        # it. Without it a routine mid-run check locks the database, and the
        # instinctive response -- kill the reader, or worse the recorder -- costs
        # the run.
        self._conn.execute("PRAGMA journal_mode=WAL")
        # NORMAL rather than FULL: a crash may lose the last few observations,
        # which is acceptable for a sampled time series and buys an order of
        # magnitude of write throughput. Losing the last second of a recording is
        # not a correctness problem; recording at a tenth of the rate is.
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.execute(_CREATE)
        for index in _INDICES:
            self._conn.execute(index)
        self._conn.commit()
        self._known_columns = self._columns()

    # -- schema ------------------------------------------------------------

    def _columns(self) -> List[str]:
        return [r["name"] for r in
                self._conn.execute("PRAGMA table_info(observations)")]

    # -- writing -----------------------------------------------------------

    def record(self, observation: Observation) -> int:
        """Append one observation. Returns its row id."""
        values = {
            "schema_version": SCHEMA_VERSION,
            "ts": float(observation.ts),
            "cex_symbol": observation.cex_symbol,
            "base": observation.base,
            "quote": observation.quote,
            "chain": observation.chain,
            "pool_fee": int(observation.pool_fee),
            "pool_address": observation.pool_address,
            "cex_bids": _levels_to_json(observation.cex_bids),
            "cex_asks": _levels_to_json(observation.cex_asks),
            "cex_feed_ts": (
                float(observation.cex_feed_ts)
                if observation.cex_feed_ts is not None else None
            ),
            "pool_state": _pool_to_json(observation.pool),
            # TEXT: a gas price in wei is comfortably inside 2^63 today, but
            # storing it as text costs nothing and removes the question.
            "gas_price_wei": (
                str(observation.gas_price_wei)
                if observation.gas_price_wei is not None else None
            ),
            "native_price_quote": (
                str(observation.native_price_quote)
                if observation.native_price_quote is not None else None
            ),
            "rpc_endpoint": observation.rpc_endpoint,
            # The observation's own run id when it has one, so a re-analysis
            # cannot misattribute rows to whichever run happened to read them.
            "run_id": observation.run_id or self.run_id,
        }
        placeholders = ", ".join("?" for _ in _INSERT_COLUMNS)
        cursor = self._conn.execute(
            f"INSERT INTO observations ({', '.join(_INSERT_COLUMNS)}) "
            f"VALUES ({placeholders})",
            [values[c] for c in _INSERT_COLUMNS],
        )
        self._conn.commit()
        return int(cursor.lastrowid)

    def record_many(self, observations: Iterable[Observation]) -> int:
        """One transaction for a batch. The recorder's per-cycle path.

        A commit per row costs a disk sync per row; at twenty pools per cycle that
        is the difference between recording at cadence and falling behind it.
        """
        count = 0
        rows = []
        for observation in observations:
            rows.append(observation)
            count += 1
        if not rows:
            return 0
        with self._conn:  # one transaction
            for observation in rows:
                self.record(observation)
        return count

    # -- reading -----------------------------------------------------------

    def read_all(
        self,
        cex_symbol: Optional[str] = None,
        since: Optional[float] = None,
        until: Optional[float] = None,
        limit: Optional[int] = None,
    ) -> Iterator[Observation]:
        """Observations in time order, optionally filtered.

        A generator: a week of twenty pools at five-second cadence is millions of
        rows, and materialising them to filter in Python would defeat the indices.
        """
        clauses, params = [], []
        if cex_symbol is not None:
            clauses.append("cex_symbol = ?")
            params.append(cex_symbol)
        if since is not None:
            clauses.append("ts >= ?")
            params.append(float(since))
        if until is not None:
            clauses.append("ts <= ?")
            params.append(float(until))
        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        sql = f"SELECT * FROM observations{where} ORDER BY ts, id"
        if limit is not None:
            sql += f" LIMIT {int(limit)}"
        for row in self._conn.execute(sql, params):
            yield self._from_row(row)

    def count(
        self,
        cex_symbol: Optional[str] = None,
        since: Optional[float] = None,
        until: Optional[float] = None,
    ) -> int:
        clauses, params = [], []
        if cex_symbol is not None:
            clauses.append("cex_symbol = ?")
            params.append(cex_symbol)
        if since is not None:
            clauses.append("ts >= ?")
            params.append(float(since))
        if until is not None:
            clauses.append("ts <= ?")
            params.append(float(until))
        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        row = self._conn.execute(
            f"SELECT COUNT(*) AS n FROM observations{where}", params
        ).fetchone()
        return int(row["n"])

    def pairs(self) -> List[str]:
        return [r["cex_symbol"] for r in self._conn.execute(
            "SELECT DISTINCT cex_symbol FROM observations ORDER BY cex_symbol"
        )]

    def time_span(self) -> Optional[Tuple[float, float]]:
        row = self._conn.execute(
            "SELECT MIN(ts) AS lo, MAX(ts) AS hi FROM observations"
        ).fetchone()
        if row["lo"] is None:
            return None
        return float(row["lo"]), float(row["hi"])

    @staticmethod
    def _from_row(row: sqlite3.Row) -> Observation:
        return Observation(
            ts=float(row["ts"]),
            cex_symbol=row["cex_symbol"],
            base=row["base"],
            quote=row["quote"],
            chain=row["chain"],
            pool_fee=int(row["pool_fee"]),
            pool_address=row["pool_address"],
            cex_bids=_levels_from_json(row["cex_bids"]),
            cex_asks=_levels_from_json(row["cex_asks"]),
            cex_feed_ts=(
                float(row["cex_feed_ts"]) if row["cex_feed_ts"] is not None else None
            ),
            pool=_pool_from_json(row["pool_state"]),
            gas_price_wei=(
                int(row["gas_price_wei"])
                if row["gas_price_wei"] is not None else None
            ),
            native_price_quote=(
                Decimal(row["native_price_quote"])
                if row["native_price_quote"] is not None else None
            ),
            rpc_endpoint=row["rpc_endpoint"],
            run_id=row["run_id"],
        )

    # -- lifecycle ---------------------------------------------------------

    def close(self) -> None:
        try:
            self._conn.commit()
        finally:
            self._conn.close()

    def __enter__(self) -> "ObservationStore":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def mid_dislocation_bps(
    observation: "Observation", base_is_token0: bool
) -> Optional[Decimal]:
    """Pool mid against CEX mid, in bps. No fee, no spread, no impact, no direction.

    The cleanest possible statement of the phenomenon, and the one that decides
    whether ANY cost structure could work. Every other figure in the report has
    something subtracted from it:

        best gross   includes the pool fee, half the CEX spread, and price impact,
                     and is a max over two directions so noise is rectified into it
        net          adds the taker fee and gas

    So a negative best-gross can mean "the venues are at parity and the fees are
    unavoidable" or "the venues genuinely disagree but not enough". Those have opposite
    implications -- the first is unfixable by any execution improvement, the second is
    a fee and universe problem -- and only the raw dislocation separates them.

    Signed: positive means the pool is above the CEX. The magnitude is what matters
    for feasibility, since either sign is tradeable in principle.
    """
    mid = observation.cex_mid
    if mid is None or mid <= 0:
        return None
    # An empty pool still reports a price, and it is meaningless. Uniswap v3 sets the
    # price at initialisation and leaves it there until someone trades, so a pool
    # created and never used keeps whatever its creator chose, forever, with nothing
    # behind it -- and the factory returns its address regardless, so existence is not
    # evidence of a market.
    #
    # Measured on the wide screen over 568 pools: AUCTION/USDC showed 36,586,588 bps,
    # CHR/USDT 728,469,999, AAVE/USDC on Base 3.9e52, and dozens sat at exactly
    # -10,000 bps, which is a price of zero. Read as dislocations these are the largest
    # opportunities in the dataset by many orders of magnitude, and every one is an
    # empty pool. Any ranking by edge surfaces them first.
    if observation.pool.liquidity <= 0:
        return None
    spot = observation.pool.spot_price()  # token0 in token1
    if spot is None or spot <= 0:
        return None
    # spot_price is token0-in-token1. Quote-per-base is that when base is token0, and
    # its reciprocal otherwise -- the same orientation question that produced a
    # 36-billion-bps reading in the optimiser before it was made explicit.
    pool_price = spot if base_is_token0 else (Decimal(1) / spot)
    return (pool_price - mid) / mid * Decimal(10000)
