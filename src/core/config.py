import os
import yaml
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Set

from pydantic import BaseModel, Field, SecretStr, ValidationError, model_validator
from dotenv import load_dotenv
from loguru import logger

# --- Helper function to load YAML with environment variable substitution ---
def load_yaml_with_env(path: str) -> dict:
    with open(path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Substitute ${VAR_NAME} with environment variables
    expanded_content = os.path.expandvars(content)
    return yaml.safe_load(expanded_content)

# --- Pydantic Models for Configuration ---
# All component models must be defined before the main AppConfig model

class TokenDetails(BaseModel):
    address: str
    decimals: int

class NetworkConfig(BaseModel):
    default_chain: str
    rpc_urls: Dict[str, str] = {}
    native_token: Dict[str, str] = {}
    max_pending_seconds: int
    gas_estimation_chain: str
    priority_fee_gwei: float
    max_fee_gwei: float

    @model_validator(mode='after')
    def load_rpc_urls_from_env(self) -> 'NetworkConfig':
        raw_urls = {
            "ethereum": os.getenv("ETH_RPC_URL", ""),
            "arbitrum": os.getenv("ARBITRUM_RPC_URL", ""),
            "bsc": os.getenv("BSC_RPC_URL", ""),
            "base": os.getenv("BASE_RPC_URL", ""),
        }
        # drop entries with empty URLs
        self.rpc_urls = {chain: url for chain, url in raw_urls.items() if url}
        return self

class DexContracts(BaseModel):
    router: str
    quoter_v2: str
    weth: str
    factory: Optional[str] = None

class DexConfig(BaseModel):
    uniswap_v3: Dict[str, DexContracts]

class CexConfig(BaseModel):
    name: str
    base_url: str
    ws_url: str
    api_key_env: str
    api_secret_env: str
    recv_window_ms: int

class RiskConfig(BaseModel):
    max_notional_per_leg_quote: float
    max_position_per_asset: float
    circuit_breaker_bps: int
    cancel_all_on_start: bool
    cancel_all_on_shutdown: bool

class RebalanceConfig(BaseModel):
    enable: bool
    target_ratio: float
    trigger_bps: int
    method: Literal["on_cex", "on_dex", "cross"]

class InventoryConfig(BaseModel):
    rebalance: RebalanceConfig

class ObservabilityConfig(BaseModel):
    metrics_port: int
    log_level: str
    redact_keys: List[str]

class DashboardConfig(BaseModel):
    enabled: bool = False
    redis_url: str = "redis://localhost:6379/0"
    channel: str = "bot_dashboard_channel"


class VolumeScannerConfig(BaseModel):
    enabled: bool = True
    quote_assets: List[str] = Field(default_factory=lambda: ["USDT", "USDC"])
    lookback_hours: int = 24
    anomaly_multiplier: float = 3.0
    min_hourly_quote_volume: float = 500000.0
    max_candidates: int = 20
    preferred_chains: List[str] = Field(default_factory=lambda: ["ethereum", "arbitrum", "base"])
    preferred_fee_tiers: List[int] = Field(default_factory=lambda: [500, 3000, 10000])
    persist_path: str = "data/discovered_pairs.yaml"
    cooldown_hours: int = 6
    default_pair_params: Dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode='after')
    def validate_parameters(self) -> 'VolumeScannerConfig':
        if self.lookback_hours < 1:
            raise ValueError("lookback_hours must be >= 1")
        if self.anomaly_multiplier < 1:
            raise ValueError("anomaly_multiplier must be >= 1")
        if self.max_candidates < 1:
            raise ValueError("max_candidates must be >= 1")
        if self.min_hourly_quote_volume < 0:
            raise ValueError("min_hourly_quote_volume must not be negative")
        return self


class SpikeArbitrageConfig(BaseModel):
    edge_threshold_bps: float = 30.0
    cost_buffer_bps: float = 15.0
    probe_size_base: float = 1.0
    dex_chains: List[str] = Field(default_factory=lambda: ["ethereum", "arbitrum", "base"])
    dex_fee_tiers: List[int] = Field(default_factory=lambda: [500, 3000, 10000])

    @model_validator(mode='after')
    def validate_params(self) -> 'SpikeArbitrageConfig':
        if self.edge_threshold_bps < 0:
            raise ValueError("edge_threshold_bps must not be negative")
        if self.cost_buffer_bps < 0:
            raise ValueError("cost_buffer_bps must not be negative")
        if self.probe_size_base <= 0:
            raise ValueError("probe_size_base must be > 0")
        if not self.dex_chains:
            raise ValueError("dex_chains requires at least one chain")
        if not self.dex_fee_tiers:
            raise ValueError("dex_fee_tiers requires at least one fee tier")
        return self


class AutoDiscoveryConfig(BaseModel):
    enabled: bool = True
    quote_asset: str = "USDT"
    lookback_hours: int = 2
    ratio_threshold: float = 1.5
    scan_interval_minutes: int = 30
    max_candidates: int = 10
    concurrency: int = 20
    persist_path: str = "data/auto_discovery.json"
    missing_tokens_path: str = "data/missing_tokens.json"
    dex_pool_dataset: str = "data/target_pools_Dex.json"
    target_quote_currencies: List[str] = Field(default_factory=lambda: ["USDT", "USDC"])
    arbitrage: SpikeArbitrageConfig = Field(default_factory=SpikeArbitrageConfig)


class VolumeSpikeConfig(BaseModel):
    enabled: bool = True
    quote_asset: str = "USDT"
    lookback_hours: int = 2
    ratio_threshold: float = 1.5
    scan_interval_minutes: int = 30
    persist_path: str = "data/volume_spikes.json"
    max_candidates: int = 10
    concurrency: int = 20
    arbitrage: SpikeArbitrageConfig = Field(default_factory=SpikeArbitrageConfig)

    @model_validator(mode='after')
    def validate_spike_config(self) -> 'VolumeSpikeConfig':
        if self.lookback_hours < 2:
            raise ValueError("lookback_hours must cover the current and previous hour; >= 2 recommended")
        if self.ratio_threshold <= 1:
            raise ValueError("ratio_threshold must be > 1")
        if self.scan_interval_minutes <= 0:
            raise ValueError("scan_interval_minutes must be > 0")
        if self.max_candidates < 1:
            raise ValueError("max_candidates must be > 0")
        if self.concurrency < 1:
            raise ValueError("concurrency must be > 0")
        return self


class ScannerConfig(BaseModel):
    volume: VolumeScannerConfig = Field(default_factory=VolumeScannerConfig)
    spike: VolumeSpikeConfig = Field(default_factory=VolumeSpikeConfig)
    auto_discovery: Optional[AutoDiscoveryConfig] = None


class StrategyConfig(BaseModel):
    target_notional_usd: int = 10
    min_edge_bps: int
    max_slippage_bps: int

class PairConfig(BaseModel):
    base: str
    quote: str
    cex_symbol: str
    min_edge_bps: int
    max_slippage_bps: int
    max_size_quote: int
    dex_chain: str
    dex_pool_fee: int
    price_floor_quote: Optional[Decimal] = None
    price_ceiling_quote: Optional[Decimal] = None
    max_edge_bps: Optional[int] = None
    edge_safety_multiplier: Optional[Decimal] = None
    base_precision: Optional[int] = None
    quote_precision: Optional[int] = None

class SecretsConfig(BaseModel):
    binance_api_key: SecretStr
    binance_api_secret: SecretStr
    dex_wallet_private_key: SecretStr

    @classmethod
    def load(cls):
        api_key = os.getenv("BINANCE_API_KEY", "")
        api_secret = os.getenv("BINANCE_API_SECRET", "")
        private_key = os.getenv("DEX_WALLET_PRIVATE_KEY", "")

        if not private_key:
            raise ValueError("Critical: DEX_WALLET_PRIVATE_KEY is missing or empty in your .env file.")
        if not api_key:
            logger.warning("BINANCE_API_KEY  is not set; CEX functionality will be limited.")
        if not api_secret:
            logger.warning("BINANCE_API_SECRET  is not set; CEX functionality will be limited.")

        return cls(
            binance_api_key=api_key,
            binance_api_secret=api_secret,
            dex_wallet_private_key=private_key,
        )

# Main AppConfig model
class AppConfig(BaseModel):
    env: str
    network: NetworkConfig
    dex: DexConfig
    cex: CexConfig
    risk: RiskConfig
    inventory: InventoryConfig
    observability: ObservabilityConfig
    dashboard: DashboardConfig
    strategy: StrategyConfig
    pairs: List[PairConfig]
    tokens: Dict[str, Dict[str, TokenDetails]]
    secrets: SecretsConfig = Field(default_factory=SecretsConfig.load)
    scanner: ScannerConfig = Field(default_factory=ScannerConfig)


def load_config(
    default_path: str = "config/default.yaml",
    pairs_path: str = "config/pairs.yaml",
    tokens_path: str = "config/tokens.yaml",
    scanner_path: str = "config/scanner.yaml",
    discovered_pairs_path: str = "data/discovered_pairs.yaml"
) -> AppConfig:
    """Loads configuration from YAML files and environment variables."""
    load_dotenv()
    # Load YAML files
    default_cfg_data = load_yaml_with_env(default_path)
    pairs_cfg_data = load_yaml_with_env(pairs_path)
    tokens_cfg_data = load_yaml_with_env(tokens_path)
    scanner_cfg_data = load_yaml_with_env(scanner_path) if Path(scanner_path).exists() else {}

    # Combine data
    full_config_data = {**default_cfg_data, **pairs_cfg_data, **tokens_cfg_data}
    if scanner_cfg_data:
        # The content of scanner.yaml is expected to be nested under a 'scanner' key.
        # We merge this with any existing scanner config from the default yaml.
        default_scanner_conf = full_config_data.get('scanner', {})
        scanner_yaml_conf = scanner_cfg_data.get('scanner', {})
        default_scanner_conf.update(scanner_yaml_conf)
        full_config_data['scanner'] = default_scanner_conf

    # Parse with Pydantic
    config = AppConfig(**full_config_data)

    # Load dynamically discovered pairs, if any
    discovered_pairs_file = Path(discovered_pairs_path)
    if discovered_pairs_file.exists():
        try:
            discovered_data = load_yaml_with_env(str(discovered_pairs_file))
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning(f"Failed to load the dynamic pairs file: {exc}")
            return config

        if not discovered_data:
            return config

        raw_pairs = discovered_data.get('pairs', [])
        if not isinstance(raw_pairs, list):
            logger.warning("Dynamic pairs file has the wrong shape; expected a list.")
            return config

        existing_symbols: Set[str] = {p.cex_symbol for p in config.pairs}
        for entry in raw_pairs:
            if not isinstance(entry, dict):
                logger.warning(f"Ignoring a malformed pair record: {entry}")
                continue
            pair_data = entry.get('config') if isinstance(entry, dict) else None
            if pair_data is None:
                pair_data = entry
            if not isinstance(pair_data, dict):
                logger.warning(f"Ignoring an unparseable pair configuration: {entry}")
                continue

            try:
                pair_cfg = PairConfig(**pair_data)
            except ValidationError as exc:
                logger.warning(f"Dynamic pair configuration failed validation: {exc}")
                continue

            if pair_cfg.cex_symbol in existing_symbols:
                continue
            config.pairs.append(pair_cfg)
            existing_symbols.add(pair_cfg.cex_symbol)

    return config

# Example usage:
if __name__ == '__main__':
    # This requires .env file to be present in the root directory
    from dotenv import load_dotenv
    load_dotenv()
    
    config = load_config()
    print("Config loaded successfully!")
    print(f"Environment: {config.env}")
    print(f"Default Chain: {config.network.default_chain}")
    print(f"CEX: {config.cex.name}")
    print(f"Monitored Pairs: {[p.cex_symbol for p in config.pairs]}")
    # Accessing secrets
    print(f"Binance Key (first 5 chars): {config.secrets.binance_api_key.get_secret_value()[:5]}...")
