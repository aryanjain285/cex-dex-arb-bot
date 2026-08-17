import asyncio
from datetime import datetime, timezone
from pathlib import Path

import pytest

from src.scanner.spike import (
    VolumeSpike,
    VolumeSpikeStore,
    SymbolSnapshot,
    ArbitrageSignal,
)


def test_symbol_snapshot_precision():
    assert SymbolSnapshot.precision_from_step("0.010000") == 2
    assert SymbolSnapshot.precision_from_step("1") == 0
    assert SymbolSnapshot.precision_from_step("0.000001") == 6


def test_volume_spike_roundtrip(tmp_path: Path):
    spike = VolumeSpike(
        symbol="ABCUSDT",
        base="ABC",
        quote="USDT",
        current_volume=2000.0,
        previous_volume=1000.0,
        ratio=2.0,
        closed_at=datetime.now(timezone.utc),
        base_precision=4,
        quote_precision=2,
    )
    store = VolumeSpikeStore(tmp_path / "spikes.json")
    store.save([spike])
    loaded = store.load()
    assert len(loaded) == 1
    loaded_spike = loaded[0]
    assert loaded_spike.symbol == "ABCUSDT"
    assert loaded_spike.base_precision == 4
    assert loaded_spike.ratio == pytest.approx(2.0)


def test_arbitrage_signal_dict_conversion():
    """`edge_bps`/`effective_edge_bps` became `gross_bps`/`net_bps`.

    The rename is not cosmetic: `effective_edge_bps` was the raw spread minus a
    flat `cost_buffer_bps` fudge, while `net_bps` is what remains after the actual
    taker fee and gas, computed by the same function the detector uses. The
    signal also records the inputs (probe size, gas, fee) and that it is
    depth-blind, so a screen hit cannot be mistaken for a tradeable edge.
    """
    signal = ArbitrageSignal(
        symbol="ABCUSDT",
        direction="DEX_to_CEX",
        gross_bps=120.567,
        net_bps=100.123,
        cex_price=1.23,
        dex_price=1.1,
        dex_chain="ethereum",
        dex_fee_tier=3000,
        probe_size_base=1.0,
        gas_quote=1.5,
        taker_fee_bps=7.5,
    )
    data = signal.as_dict()
    assert data["direction"] == "DEX_to_CEX"
    assert data["dex_chain"] == "ethereum"
    assert data["gross_bps"] == pytest.approx(120.57, abs=0.01)
    assert data["net_bps"] == pytest.approx(100.12, abs=0.01)
    assert data["depth_aware"] is False
    assert data["taker_fee_bps"] == pytest.approx(7.5)
