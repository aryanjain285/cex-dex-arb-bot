"""A token symbol is not an identity. The shipped dataset proves it.

`data/target_pools_Dex.json` contains this pool on Base:

    poolAddress 0x48413707b70355597404018e7c603b261fcadf3f
    token0  WETH  0x4200000000000000000000000000000000000006   (canonical)
    token1  WETH  0x71b35ecb35104773537f849fbc353f81303a5860   (impostor)
    tvlUSD          270,439
    volume24hUSD 62,013,886

$62m of reported 24h volume against $270k of TVL -- a 229x turnover ratio -- is
the wash-trading signature of a token that exists to be found by a scanner.
It is the only pool in 1,062 whose two sides share a symbol, and the impostor
address appears in exactly one pool side while canonical Base WETH appears in
323.

The symbol-based token policy cannot catch this: the impostor's symbol IS on the
allowlist. Only the address can distinguish them, and `config/tokens.yaml` is the
trusted registry of which address a symbol means on which chain.

There is a second defect in the same code path, independent of impostors. The
base/quote assignment was:

    base, quote = (token0, token1) if token0_symbol == opp["base"] else (token1, token0)

If NEITHER token matches the expected base, the else branch silently declares
token1 to be the base. The pair is then built around the wrong token's address
and decimals, and every quote for it prices something else entirely.
"""
import json

import pytest

from src.app import ArbiBotApp
from src.core.config import load_config

CANONICAL_BASE_WETH = "0x4200000000000000000000000000000000000006"
IMPOSTOR_BASE_WETH = "0x71b35ecb35104773537f849fbc353f81303a5860"
BASE_USDC = "0x833589fcd6edb6e08f4c7c32d4f71b54bda02913"


class _StubLogger:
    def __init__(self):
        self.messages = []

    def _record(self, level, message):
        self.messages.append(f"{level}: {message}")

    def info(self, m, *a, **k):
        self._record("INFO", m)

    def warning(self, m, *a, **k):
        self._record("WARNING", m)

    def error(self, m, *a, **k):
        self._record("ERROR", m)

    def debug(self, m, *a, **k):
        self._record("DEBUG", m)


def _opportunity(symbol, base, quote, token0, token1, chain="base", fee=500):
    return {
        "symbol": symbol,
        "base": base,
        "quote": quote,
        "dex_candidates": [{
            "chain": chain,
            "fee": fee,
            "raw_pool_data": {
                "token0": token0,
                "token1": token1,
                "is_synthetic": False,
                "intermediate_symbol": None,
            },
        }],
    }


def _token(symbol, address, decimals=18):
    return {"symbol": symbol, "address": address, "decimals": decimals}


def _load(tmp_path, opportunities):
    path = tmp_path / "auto_discovery.json"
    path.write_text(json.dumps({"opportunities": opportunities}), encoding="utf-8")

    config = load_config()
    config.scanner.auto_discovery.persist_path = str(path)

    stub = _StubLogger()
    app_self = type("_Stub", (), {"logger": stub})()
    return ArbiBotApp._load_discovered_pairs(app_self, config), stub.messages


def test_the_canonical_pool_still_loads(tmp_path):
    """Positive control, using the real Base addresses from tokens.yaml."""
    pairs, messages = _load(tmp_path, [_opportunity(
        "ETH/USDC", "WETH", "USDC",
        _token("WETH", CANONICAL_BASE_WETH),
        _token("USDC", BASE_USDC, decimals=6),
    )])

    assert [p.cex_symbol for p in pairs] == ["ETH/USDC"], messages
    assert pairs[0].base_address.lower() == CANONICAL_BASE_WETH


def test_the_pipeline_spelling_of_wrapped_native_also_resolves(tmp_path):
    """autodiscovery normalises WETH to ETH before writing the file, while
    tokens.yaml keys the wrapped contract as WETH. Both spellings must reach the
    same registered address, or the guard would reject every ETH pool as an
    impostor -- a fail-closed bug that would look like a working safety check.
    """
    pairs, messages = _load(tmp_path, [_opportunity(
        "ETH/USDC", "ETH", "USDC",
        _token("WETH", CANONICAL_BASE_WETH),
        _token("USDC", BASE_USDC, decimals=6),
    )])

    assert [p.cex_symbol for p in pairs] == ["ETH/USDC"], messages
    assert pairs[0].base_address.lower() == CANONICAL_BASE_WETH


def test_the_impostor_weth_pool_from_the_shipped_dataset_is_rejected(tmp_path):
    """The exact pool, with the exact addresses, as it appears on disk."""
    pairs, messages = _load(tmp_path, [_opportunity(
        "ETH/USDC", "WETH", "USDC",
        _token("WETH", IMPOSTOR_BASE_WETH),
        _token("USDC", BASE_USDC, decimals=6),
    )])

    assert pairs == [], "a counterfeit WETH was accepted as WETH"
    assert any(IMPOSTOR_BASE_WETH[:10] in m or "address" in m.lower()
               for m in messages), f"the rejection was not explained: {messages}"


def test_a_mismatched_quote_address_is_rejected_too(tmp_path):
    """Either side can be the impostor."""
    pairs, _ = _load(tmp_path, [_opportunity(
        "ETH/USDC", "WETH", "USDC",
        _token("WETH", CANONICAL_BASE_WETH),
        _token("USDC", "0x" + "cc" * 20, decimals=6),
    )])

    assert pairs == []


def test_a_pool_where_neither_token_is_the_expected_base_is_rejected(tmp_path):
    """The silent-misassignment defect.

    Neither token is WETH, so the old code would have declared token1 (USDC) to
    be the base and built a pair whose base address and decimals belong to a
    different token than the CEX symbol it is quoted against.
    """
    pairs, messages = _load(tmp_path, [_opportunity(
        "ETH/USDC", "WETH", "USDC",
        _token("USDT", "0xfde4c96c8593536e31f229ea8f37b2ada2699bb2", decimals=6),
        _token("USDC", BASE_USDC, decimals=6),
    )])

    assert pairs == [], (
        "a pool containing neither side of the pair was accepted"
    )
    assert any("base" in m.lower() for m in messages), messages


def test_an_unregistered_symbol_is_left_to_the_token_policy(tmp_path):
    """Address verification only applies where tokens.yaml knows the symbol.

    For anything else the token policy is the gate -- and under default-deny it
    already refuses. This test pins the division of labour so a future change
    cannot leave a gap between the two checks.
    """
    pairs, _ = _load(tmp_path, [_opportunity(
        "NEWCOIN/USDC", "NEWCOIN", "USDC",
        _token("NEWCOIN", "0x" + "dd" * 20),
        _token("USDC", BASE_USDC, decimals=6),
    )])

    assert pairs == []


def test_the_shipped_dataset_still_contains_exactly_one_same_symbol_pool():
    """A canary on the evidence this whole module rests on.

    If a regenerated dataset no longer contains it, that is worth knowing --
    either the pool is gone or the scanner changed what it collects.
    """
    from pathlib import Path

    payload = json.loads(
        Path("data/target_pools_Dex.json").read_text(encoding="utf-8")
    )
    same = [
        p for p in payload["pools"]
        if p["token0"]["symbol"].strip().upper()
        == p["token1"]["symbol"].strip().upper()
    ]

    assert len(same) == 1, f"expected the one known impostor pool, found {len(same)}"
    addresses = {same[0]["token0"]["address"].lower(),
                 same[0]["token1"]["address"].lower()}
    assert IMPOSTOR_BASE_WETH in addresses
