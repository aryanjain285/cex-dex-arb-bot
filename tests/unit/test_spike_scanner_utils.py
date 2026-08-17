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
    signal = ArbitrageSignal(
        symbol="ABCUSDT",
        direction="DEX_to_CEX",
        edge_bps=120.567,
        effective_edge_bps=100.123,
        cex_price=1.23,
        dex_price=1.1,
        dex_chain="ethereum",
        dex_fee_tier=3000,
    )
    data = signal.as_dict()
    assert data["direction"] == "DEX_to_CEX"
    assert data["dex_chain"] == "ethereum"
    assert data["edge_bps"] == pytest.approx(120.57, abs=0.01)
