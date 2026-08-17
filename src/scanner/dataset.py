"""Loading and validating the DEX pool dataset.

This module exists because of one line that appeared in four places:

    base_details_raw.get('decimals', 18)

A silent default of 18 on a field that scales price by ten orders of magnitude.
Measured against the shipped dataset: all 1,062 pools lack a `decimals` field
entirely, and 291 of them (27%) contain at least one non-18-decimal token. On a
USDC pool the resulting error is exactly 10^12 -- a true price of 0.022145
computed as 2.2145e-14.

That error currently fails safe, but only through three independent accidents:
the net-edge sanity ceiling rejects the overstated direction, the CEX depth
check rejects the understated one, and the quoter returns a partial fill rather
than reverting. None of those was written to catch a decimals bug. Relying on
them is relying on luck, so a missing decimal now raises.

The dataset itself is gated too. The shipped file is 327 days old and records
nothing about the query, the filters, or the block heights that produced it --
so there is no way to know what it represents. A dataset that influences
trading decisions needs provenance and an expiry, not a timestamp nobody reads.
"""
from __future__ import annotations

import datetime
import json
from pathlib import Path
from typing import Any, Dict, List, Union

from loguru import logger

from ..core import clock

__all__ = ["DatasetError", "require_decimals", "load_pool_dataset"]

# ERC-20 `decimals` is a uint8, but values above ~36 do not occur in practice
# and indicate corruption. Real observed values in this dataset span 0 (SLP)
# to 27 (WTON), so the range must not be narrowed to "18 or 6".
MAX_PLAUSIBLE_DECIMALS = 36


class DatasetError(RuntimeError):
    """Raised when pool data cannot be trusted.

    Deliberately fatal. The alternative -- defaulting, warning, and continuing
    -- is what produced the 10^12 error, and a warning in a log nobody reads is
    indistinguishable from silence.
    """


def require_decimals(token: Dict[str, Any], context: str) -> int:
    """Return a token's decimals, or raise.

    `context` should identify the pool or token so the error is actionable
    without a debugger -- a bare "missing decimals" tells an operator nothing
    about which of a thousand pools is at fault.
    """
    if "decimals" not in token:
        raise DatasetError(
            f"{context}: token {token.get('symbol', '?')} "
            f"({token.get('address', '?')}) has no `decimals` field. Refusing to "
            f"default to 18 -- on a 6-decimal token that is a 10^12 pricing "
            f"error. Regenerate the dataset with the scanner."
        )

    value = token["decimals"]
    if isinstance(value, bool) or not isinstance(value, int):
        raise DatasetError(
            f"{context}: token {token.get('symbol', '?')} has non-integer "
            f"decimals {value!r}."
        )
    if not 0 <= value <= MAX_PLAUSIBLE_DECIMALS:
        raise DatasetError(
            f"{context}: token {token.get('symbol', '?')} has implausible "
            f"decimals {value}. Expected 0..{MAX_PLAUSIBLE_DECIMALS}."
        )
    return value


def _parse_timestamp(payload: Dict[str, Any], path: Path) -> datetime.datetime:
    raw = payload.get("last_updated_utc")
    if not raw:
        raise DatasetError(
            f"{path}: no `last_updated_utc`. A dataset with no provenance cannot "
            f"be checked for staleness and must not drive trading decisions."
        )
    try:
        stamp = datetime.datetime.fromisoformat(str(raw))
    except ValueError as exc:
        raise DatasetError(f"{path}: unparseable last_updated_utc {raw!r}") from exc
    if stamp.tzinfo is None:
        stamp = stamp.replace(tzinfo=datetime.timezone.utc)
    return stamp


def load_pool_dataset(
    path: Union[str, Path], max_age_hours: float
) -> List[Dict[str, Any]]:
    """Load and validate the pool dataset, or raise `DatasetError`.

    Four gates, each closing a way a bad dataset silently reached the strategy:
    the file must exist, it must declare when it was built, it must not be
    older than `max_age_hours`, and its declared `pool_count` must match what
    it actually contains -- a truncated write must not be read as a smaller
    universe. Then every token must carry a valid `decimals`.
    """
    path = Path(path)
    if not path.exists():
        raise DatasetError(
            f"Pool dataset not found: {path}. Generate it with "
            f"`python src/scanner/dex_pool_scanner.py`."
        )

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DatasetError(f"Cannot read {path}: {exc}") from exc

    stamp = _parse_timestamp(payload, path)
    age_hours = (
        datetime.datetime.fromtimestamp(clock.now(), tz=datetime.timezone.utc) - stamp
    ).total_seconds() / 3600.0
    if age_hours > max_age_hours:
        raise DatasetError(
            f"{path} is stale: {age_hours:.1f}h old, limit {max_age_hours}h "
            f"(built {stamp.isoformat()}). Pool composition, liquidity and fee "
            f"tiers all drift; trading on a stale universe prices pools that may "
            f"no longer exist. Re-run the scanner."
        )

    pools = payload.get("pools")
    if not isinstance(pools, list):
        raise DatasetError(f"{path}: `pools` is missing or not a list.")

    declared = payload.get("pool_count")
    if declared is not None and int(declared) != len(pools):
        raise DatasetError(
            f"{path}: pool_count says {declared} but the file contains "
            f"{len(pools)} pools. The file is truncated or was written "
            f"concurrently; refusing to treat it as a smaller universe."
        )

    for pool in pools:
        context = f"{path.name} pool {pool.get('poolAddress', '?')}"
        for side in ("token0", "token1"):
            token = pool.get(side)
            if not isinstance(token, dict):
                raise DatasetError(f"{context}: {side} is missing or malformed.")
            require_decimals(token, context)

    logger.info(
        f"Loaded {len(pools)} pools from {path} "
        f"({age_hours:.1f}h old, all decimals validated)."
    )
    return pools
