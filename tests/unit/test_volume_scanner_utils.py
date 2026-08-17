import yaml

from src.scanner.volume import DiscoveredPairStore, SymbolInfo, VolumeScannerService


def test_precision_from_step_handles_decimals():
    assert SymbolInfo._precision_from_step("0.01000000") == 2
    assert SymbolInfo._precision_from_step("1.00000000") == 0
    assert SymbolInfo._precision_from_step("0.00010000") == 4
    assert SymbolInfo._precision_from_step("1E-6") == 6


def test_discovered_pair_store_roundtrip(tmp_path):
    path = tmp_path / "dynamic_pairs.yaml"
    store = DiscoveredPairStore(path)

    data = store.load()
    assert data == {"pairs": []}

    payload = {"pairs": [{"config": {"cex_symbol": "ETH/USDT"}, "metadata": {"source": "unit"}}]}
    store.save(payload)

    assert path.exists()
    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    assert raw["pairs"][0]["config"]["cex_symbol"] == "ETH/USDT"

    loaded = store.load()
    assert loaded["pairs"][0]["config"]["cex_symbol"] == "ETH/USDT"


def test_extract_symbol_handles_nested_configs():
    entry = {"config": {"cex_symbol": "BTC/USDT"}}
    assert VolumeScannerService._extract_symbol(entry) == "BTC/USDT"
    assert VolumeScannerService._extract_symbol({"cex_symbol": "ARB/USDT"}) == "ARB/USDT"
    assert VolumeScannerService._extract_symbol({"config": {}}) is None
    assert VolumeScannerService._extract_symbol({}) is None


def test_token_available_on_chain_helper():
    class DummyDexClient:
        tokens_config = {
            "AAA": {"ethereum": object()},
            "BBB": {"arbitrum": object()},
        }

    dummy = DummyDexClient()
    assert VolumeScannerService._token_available_on_chain(dummy, "AAA", "ethereum")
    assert not VolumeScannerService._token_available_on_chain(dummy, "AAA", "arbitrum")
    assert not VolumeScannerService._token_available_on_chain(dummy, "CCC", "ethereum")
