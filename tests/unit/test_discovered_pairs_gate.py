"""Auto-discovered pairs go straight into live trading, unreviewed.

`load_config` appends the contents of `data/discovered_pairs.yaml` to
`config.pairs` AFTER pydantic has finished validating, and `app.py` additionally
merges `data/auto_discovery.json` into the pair list it hands to both the
detector and the executor. So the volume scanner is an auto-promotion path: it
can put a token into the trading set with no human ever seeing it.

Because the append happens after validation, every cross-check in
`AppConfig.validate_coherence` was bypassed for exactly the pairs that had the
least human scrutiny -- the token policy, the known-chain check, all of it.

The asymmetry in the fix is deliberate. A pair in `config/pairs.yaml` is a
statement of intent by a person, so a hazard there is a configuration error and
the process refuses to start. A pair in a discovery file is machine output, so a
hazard there is expected occasionally: it is dropped with a warning and the rest
of the run proceeds. Refusing to start would let one bad discovery take the
whole system offline.
"""
import textwrap

import pytest

from src.core.config import load_config


def _write(tmp_path, body: str):
    path = tmp_path / "discovered_pairs.yaml"
    path.write_text(textwrap.dedent(body), encoding="utf-8")
    return str(path)


def _load(tmp_path, body: str):
    return load_config(discovered_pairs_path=_write(tmp_path, body))


def _symbols(config):
    return {p.cex_symbol for p in config.pairs}


def test_a_clean_discovered_pair_is_added():
    """The positive control. Without it, a gate that rejected everything would
    pass every test below."""
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmp:
        config = _load(Path(tmp), """
            pairs:
              - config:
                  base: "ARB"
                  quote: "USDT"
                  cex_symbol: "ARB/USDC"
                  max_slippage_bps: 30
                  max_size_quote: 1000
                  dex_chain: "arbitrum"
                  dex_pool_fee: 500
        """)

    assert "ARB/USDC" in _symbols(config)


def test_a_discovered_pair_on_a_denied_token_is_dropped_not_traded(tmp_path):
    """LINGO is fee-on-transfer and sits in the highest-volume Base pool in the
    scanned dataset, so it is the single most likely token for the scanner to
    surface unattended."""
    config = _load(tmp_path, """
        pairs:
          - config:
              base: "LINGO"
              quote: "WETH"
              cex_symbol: "LINGO/ETH"
              max_slippage_bps: 30
              max_size_quote: 1000
              dex_chain: "base"
              dex_pool_fee: 3000
    """)

    assert "LINGO/ETH" not in _symbols(config), (
        "a fee-on-transfer token reached the live pair list"
    )


def test_a_discovered_pair_outside_the_allowlist_is_dropped(tmp_path):
    config = _load(tmp_path, """
        pairs:
          - config:
              base: "NEWCOIN"
              quote: "USDT"
              cex_symbol: "NEWCOIN/USDT"
              max_slippage_bps: 30
              max_size_quote: 1000
              dex_chain: "ethereum"
              dex_pool_fee: 3000
    """)

    assert "NEWCOIN/USDT" not in _symbols(config)


def test_a_discovered_pair_on_an_unknown_chain_is_dropped(tmp_path):
    """The same check a hand-written pair gets. A pair on a chain with no DEX
    contracts can never be quoted and would log a warning every cycle forever."""
    config = _load(tmp_path, """
        pairs:
          - config:
              base: "ARB"
              quote: "USDT"
              cex_symbol: "ARB/BUSD"
              max_slippage_bps: 30
              max_size_quote: 1000
              dex_chain: "solana"
              dex_pool_fee: 3000
    """)

    assert "ARB/BUSD" not in _symbols(config)


def test_the_static_pairs_survive_a_bad_discovery_file(tmp_path):
    """One bad discovery must not take the system offline."""
    config = _load(tmp_path, """
        pairs:
          - config:
              base: "LINGO"
              quote: "WETH"
              cex_symbol: "LINGO/ETH"
              max_slippage_bps: 30
              max_size_quote: 1000
              dex_chain: "base"
              dex_pool_fee: 3000
    """)

    assert "ETH/USDT" in _symbols(config), "the configured pairs must still load"


def test_a_dropped_pair_is_logged_loudly(tmp_path, caplog):
    """Silent filtering of a discovery is indistinguishable from the scanner
    having found nothing, which would hide a systematic policy mismatch."""
    import logging

    from loguru import logger as loguru_logger

    messages = []
    sink_id = loguru_logger.add(lambda m: messages.append(m), level="WARNING")
    try:
        _load(tmp_path, """
            pairs:
              - config:
                  base: "LINGO"
                  quote: "WETH"
                  cex_symbol: "LINGO/ETH"
                  max_slippage_bps: 30
                  max_size_quote: 1000
                  dex_chain: "base"
                  dex_pool_fee: 3000
        """)
    finally:
        loguru_logger.remove(sink_id)

    joined = "".join(str(m) for m in messages)
    assert "LINGO" in joined, f"no warning named the dropped pair: {joined!r}"
