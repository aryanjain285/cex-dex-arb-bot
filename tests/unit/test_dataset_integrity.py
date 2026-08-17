"""A missing token decimal must be a hard error, never a default of 18.

The on-chain audit measured that all 1,062 pools in the shipped dataset lack a
`decimals` field, and that 291 of them (27%) contain at least one non-18-decimal
token. Both consumers read `.get('decimals', 18)`. On a USDC pool that is a
10^12 pricing error -- verified live: 0.022145 became 2.2145e-14.

It currently fails safe, but by three independent accidents (a sanity ceiling,
a CEX depth check, and the quoter's partial-fill behaviour), none of which was
designed for the purpose. `.get('decimals', 18)` is a silent default on a field
that scales price by twelve orders of magnitude, so it is replaced by a raise.

The dataset itself is also gated: 327 days stale, with no record of the query
or filters that produced it. A dataset that influences trading needs provenance
and an expiry, not just a timestamp nobody reads.
"""
import json
from decimal import Decimal

import pytest

from src.scanner.dataset import (
    DatasetError,
    load_pool_dataset,
    require_decimals,
)


def _pool(t0_dec=18, t1_dec=6, **kw):
    t0 = {"symbol": "ALT", "address": "0x" + "11" * 20}
    t1 = {"symbol": "USDC", "address": "0x" + "22" * 20}
    if t0_dec is not None:
        t0["decimals"] = t0_dec
    if t1_dec is not None:
        t1["decimals"] = t1_dec
    pool = {
        "protocol": "uniswap", "chain": "ethereum",
        "poolAddress": "0x" + "33" * 20, "poolType": "ALT_STABLE",
        "feeTier": 3000, "tvlUSD": 350000.0, "volume24hUSD": 1000000.0,
        "token0": t0, "token1": t1,
    }
    pool.update(kw)
    return pool


def _dataset(pools, **meta):
    payload = {
        "last_updated_utc": "2026-08-17T00:00:00+00:00",
        "pool_count": len(pools),
        "pools": pools,
    }
    payload.update(meta)
    return payload


# --------------------------------------------------------------------------
# require_decimals
# --------------------------------------------------------------------------

def test_present_decimals_are_returned():
    assert require_decimals({"symbol": "USDC", "decimals": 6}, "ctx") == 6


def test_a_missing_decimal_raises_rather_than_defaulting_to_18():
    """The single most dangerous default in the codebase."""
    with pytest.raises(DatasetError, match="decimals"):
        require_decimals({"symbol": "USDC", "address": "0x1"}, "pool 0xabc")


def test_the_error_names_the_context_so_it_is_actionable():
    with pytest.raises(DatasetError, match="pool 0xabc"):
        require_decimals({"symbol": "USDC"}, "pool 0xabc")


@pytest.mark.parametrize("bad", [None, "", "six", -1, 0.5, 79])
def test_an_implausible_decimal_is_rejected(bad):
    """0 is legitimate (SLP) and 27 exists (WTON), but a non-integer, a
    negative, or something above the ERC-20 practical ceiling is corrupt."""
    with pytest.raises(DatasetError):
        require_decimals({"symbol": "X", "decimals": bad}, "ctx")


@pytest.mark.parametrize("ok", [0, 6, 8, 9, 18, 24, 27])
def test_real_world_decimal_values_are_accepted(ok):
    """Values actually observed on-chain in this dataset: SLP=0, USDC=6,
    WBTC=8, HDRN=9, WETH=18, NEAR=24, WTON=27."""
    assert require_decimals({"symbol": "X", "decimals": ok}, "ctx") == ok


# --------------------------------------------------------------------------
# dataset loading and provenance
# --------------------------------------------------------------------------

def test_a_dataset_whose_pools_lack_decimals_is_refused(tmp_path):
    """The shipped September dataset must be refused, loudly."""
    path = tmp_path / "pools.json"
    path.write_text(json.dumps(_dataset([_pool(t0_dec=None, t1_dec=None)])), encoding="utf-8")

    with pytest.raises(DatasetError, match="decimals"):
        load_pool_dataset(path, max_age_hours=24 * 365)


def test_a_stale_dataset_is_refused(tmp_path):
    path = tmp_path / "pools.json"
    path.write_text(json.dumps(_dataset(
        [_pool()], last_updated_utc="2025-09-23T20:13:44+00:00")), encoding="utf-8")

    with pytest.raises(DatasetError, match="stale|older"):
        load_pool_dataset(path, max_age_hours=24)


def test_a_fresh_valid_dataset_loads(tmp_path):
    from src.core import clock
    import datetime

    fresh = datetime.datetime.fromtimestamp(
        clock.now(), tz=datetime.timezone.utc).isoformat()
    path = tmp_path / "pools.json"
    path.write_text(json.dumps(_dataset([_pool()], last_updated_utc=fresh)),
                    encoding="utf-8")

    pools = load_pool_dataset(path, max_age_hours=24)
    assert len(pools) == 1
    assert pools[0]["token1"]["decimals"] == 6


def test_a_dataset_with_no_timestamp_is_refused(tmp_path):
    """Provenance is mandatory: a dataset that cannot say when it was built
    cannot be checked for staleness."""
    path = tmp_path / "pools.json"
    payload = {"pool_count": 1, "pools": [_pool()]}
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(DatasetError, match="last_updated_utc|provenance"):
        load_pool_dataset(path, max_age_hours=24)


def test_a_missing_file_raises_a_clear_error(tmp_path):
    with pytest.raises(DatasetError, match="not found"):
        load_pool_dataset(tmp_path / "nope.json", max_age_hours=24)


def test_the_declared_pool_count_must_match_the_actual_count(tmp_path):
    """A truncated write must not be read as a smaller universe."""
    from src.core import clock
    import datetime

    fresh = datetime.datetime.fromtimestamp(
        clock.now(), tz=datetime.timezone.utc).isoformat()
    path = tmp_path / "pools.json"
    payload = _dataset([_pool()], last_updated_utc=fresh)
    payload["pool_count"] = 99          # claims 99, contains 1
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(DatasetError, match="pool_count"):
        load_pool_dataset(path, max_age_hours=24)


def test_the_shipped_dataset_is_actually_refused():
    """Not hypothetical: the file in the repo must fail this gate."""
    from pathlib import Path

    shipped = Path("data/target_pools_Dex.json")
    if not shipped.exists():
        pytest.skip("shipped dataset not present")

    with pytest.raises(DatasetError):
        load_pool_dataset(shipped, max_age_hours=24)
