"""The second auto-promotion path: `data/auto_discovery.json`.

`load_config` gates `data/discovered_pairs.yaml`, but `app.py` reads a *different*
file -- `auto_discovery.json` -- and builds MarketPairs from it directly, then
hands them to the detector and the executor alongside the configured pairs. A
gate on one file and not the other is not a gate.

The detector also refuses a denied pair per evaluation, which is the guarantee
that matters for capital. This test covers the layer above it, for two reasons:
a pair filtered here never appears in the "monitoring N pairs" count that an
operator reads as the trading universe, and it never reaches the executor at all
rather than relying on a downstream check to keep declining it.
"""
import json

from src.app import ArbiBotApp
from src.core.config import load_config


class _StubLogger:
    """Collects messages instead of writing, so the test can assert a dropped
    pair was announced. A silent drop is indistinguishable from the scanner
    having found nothing."""

    def __init__(self):
        self.messages = []

    def _record(self, level, message, *args, **kwargs):
        self.messages.append(f"{level}: {message}")

    def info(self, message, *a, **k):
        self._record("INFO", message)

    def warning(self, message, *a, **k):
        self._record("WARNING", message)

    def error(self, message, *a, **k):
        self._record("ERROR", message)

    def debug(self, message, *a, **k):
        self._record("DEBUG", message)


# Real registered addresses from config/tokens.yaml. The impostor guard checks a
# pool's address against that registry, so a made-up address for a registered
# symbol is now rejected -- correctly, but it would make these fixtures test the
# address guard instead of what they are about.
REGISTERED = {
    ("ARB", "arbitrum"): "0x912CE59144191C1204E64559FE8253a0e49E6548",
    ("USDT", "arbitrum"): "0xFd086bC7CD5C481DCC9C85ebE478A1C0b69FCbb9",
    ("USDT", "base"): "0xfde4C96c8593536E31F229EA8f37b2ADa2699bb2",
}


def _address_for(symbol, chain, fallback):
    return REGISTERED.get((symbol, chain), fallback)


def _opportunity(symbol, base, quote, token0_symbol, chain="base", fee=3000,
                 is_synthetic=False, intermediate=None):
    return {
        "symbol": symbol,
        "base": base,
        "quote": quote,
        "dex_candidates": [
            {
                "chain": chain,
                "fee": fee,
                "raw_pool_data": {
                    "token0": {
                        "symbol": token0_symbol,
                        "address": _address_for(token0_symbol, chain,
                                                "0x" + "aa" * 20),
                        # Integers, matching what dex_pool_scanner.py writes:
                        # it coerces the subgraph's string BigInt with int().
                        # A string here would be rejected by require_decimals
                        # and every assertion below would pass for the wrong
                        # reason.
                        "decimals": 18,
                    },
                    "token1": {
                        "symbol": "USDT",
                        "address": _address_for("USDT", chain, "0x" + "bb" * 20),
                        "decimals": 6,
                    },
                    "is_synthetic": is_synthetic,
                    "intermediate_symbol": intermediate,
                },
            }
        ],
    }


def _load_pairs(tmp_path, opportunities):
    """Call the loader with a stub self, so no sockets or metrics server open."""
    path = tmp_path / "auto_discovery.json"
    path.write_text(json.dumps({"opportunities": opportunities}), encoding="utf-8")

    config = load_config()
    config.scanner.auto_discovery.persist_path = str(path)

    stub = _StubLogger()
    app_self = type("_Stub", (), {"logger": stub})()
    pairs = ArbiBotApp._load_discovered_pairs(app_self, config)
    return pairs, stub.messages


def test_a_clean_discovered_pair_is_loaded(tmp_path):
    """Positive control: a gate that dropped everything would pass the rest."""
    pairs, _ = _load_pairs(tmp_path, [
        _opportunity("ARB/USDT", "ARB", "USDT", "ARB", chain="arbitrum", fee=500)
    ])

    assert [p.cex_symbol for p in pairs] == ["ARB/USDT"]


def test_a_denied_token_never_reaches_the_pair_list(tmp_path):
    pairs, messages = _load_pairs(tmp_path, [
        _opportunity("LINGO/USDT", "LINGO", "USDT", "LINGO")
    ])

    assert pairs == [], "a fee-on-transfer token was handed to the executor"
    assert any("LINGO" in m for m in messages), (
        f"the drop was not announced: {messages}"
    )


def test_a_token_outside_the_allowlist_never_reaches_the_pair_list(tmp_path):
    pairs, _ = _load_pairs(tmp_path, [
        _opportunity("NEWCOIN/USDT", "NEWCOIN", "USDT", "NEWCOIN")
    ])

    assert pairs == []


def test_an_unknown_chain_never_reaches_the_pair_list(tmp_path):
    """A pair on a chain with no DEX contracts can never be quoted, so it would
    log a warning every cycle forever while counting toward the pair total."""
    pairs, _ = _load_pairs(tmp_path, [
        _opportunity("ARB/USDT", "ARB", "USDT", "ARB", chain="solana")
    ])

    assert pairs == []


def test_the_synthetic_dex_side_asset_is_checked(tmp_path):
    """A synthetic pair's on-chain leg touches an asset that appears nowhere in
    the pair's name, so base and quote alone cannot clear it."""
    pairs, _ = _load_pairs(tmp_path, [
        _opportunity("ARB/USDT", "ARB", "USDT", "ARB", chain="arbitrum", fee=500,
                     is_synthetic=True, intermediate="LINGO")
    ])

    assert pairs == [], "the intermediate asset was not checked"


def test_one_denied_pair_does_not_drop_the_others(tmp_path):
    pairs, _ = _load_pairs(tmp_path, [
        _opportunity("LINGO/USDT", "LINGO", "USDT", "LINGO"),
        _opportunity("ARB/USDT", "ARB", "USDT", "ARB", chain="arbitrum", fee=500),
    ])

    assert [p.cex_symbol for p in pairs] == ["ARB/USDT"]
