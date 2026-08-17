"""Config values that are individually valid but jointly incoherent.

Every audit found at least one of these. The pattern is a setting that looks
like a limit, validates fine on its own, and cannot possibly fire given
another setting -- which is worse than an absent limit, because an operator
reads it and believes they are protected.

The worked example: `max_notional_per_leg_quote` was 10000 while trade size
always derives from `target_notional_usd` (1000). A 10x gap made the only live
risk gate unreachable by construction.
"""
import pytest

from src.core.config import (
    CexConfig, DashboardConfig, DexConfig, DexContracts, InventoryConfig,
    ObservabilityConfig, PairConfig, RebalanceConfig, RiskConfig,
    RotationConfig, SecretsConfig, StrategyConfig,
)
from src.core.config import AppConfig


def _app(**overrides):
    base = dict(
        env="dev",
        network=dict(default_chain="ethereum", max_pending_seconds=30,
                     gas_estimation_chain="ethereum", priority_fee_gwei=2.0,
                     max_fee_gwei=60.0, native_token={"ethereum": "ETH"}),
        dex=DexConfig(uniswap_v3={"ethereum": DexContracts(
            router="0x" + "11" * 20, quoter_v2="0x" + "22" * 20,
            weth="0x" + "33" * 20)}),
        cex=CexConfig(name="binance", base_url="https://x", ws_url="wss://x/ws",
                      api_key_env="A", api_secret_env="B", recv_window_ms=5000),
        risk=RiskConfig(max_notional_per_leg_quote=1200, max_position_per_asset=2.0,
                        circuit_breaker_bps=250, cancel_all_on_start=False,
                        cancel_all_on_shutdown=False, max_daily_loss_quote=250.0),
        inventory=InventoryConfig(rebalance=RebalanceConfig(
            enable=False, target_ratio=0.5, trigger_bps=500, method="on_cex")),
        observability=ObservabilityConfig(metrics_port=9000, log_level="INFO",
                                         redact_keys=[]),
        dashboard=DashboardConfig(enabled=False),
        strategy=StrategyConfig(target_notional_usd=1000),
        pairs=[PairConfig(base="WETH", quote="USDT", cex_symbol="ETH/USDT",
                          max_slippage_bps=30, max_size_quote=5000,
                          dex_chain="ethereum", dex_pool_fee=500)],
        tokens={},
        secrets=SecretsConfig(binance_api_key="k", binance_api_secret="s",
                              dex_wallet_private_key="0x" + "11" * 32),
    )
    base.update(overrides)
    return AppConfig(**base)


def test_a_coherent_configuration_loads():
    cfg = _app()
    assert cfg.risk.max_notional_per_leg_quote == 1200


def test_an_unreachable_notional_cap_is_rejected():
    """The exact defect: a cap 10x the largest producible size."""
    with pytest.raises(Exception, match="unreachable|max_notional_per_leg_quote"):
        _app(risk=RiskConfig(
            max_notional_per_leg_quote=10000, max_position_per_asset=2.0,
            circuit_breaker_bps=250, cancel_all_on_start=False,
            cancel_all_on_shutdown=False, max_daily_loss_quote=250.0))


def test_a_cap_below_the_target_notional_is_rejected():
    """The opposite incoherence: a cap that blocks every single trade, so the
    bot would run forever and never trade while looking healthy."""
    with pytest.raises(Exception, match="max_notional_per_leg_quote"):
        _app(risk=RiskConfig(
            max_notional_per_leg_quote=500, max_position_per_asset=2.0,
            circuit_breaker_bps=250, cancel_all_on_start=False,
            cancel_all_on_shutdown=False, max_daily_loss_quote=250.0))


def test_a_daily_loss_limit_smaller_than_one_trade_is_rejected():
    """A limit that a single losing trade would breach is not a risk control,
    it is a one-trade kill switch that will look like a malfunction."""
    with pytest.raises(Exception, match="max_daily_loss_quote"):
        _app(risk=RiskConfig(
            max_notional_per_leg_quote=1200, max_position_per_asset=2.0,
            circuit_breaker_bps=250, cancel_all_on_start=False,
            cancel_all_on_shutdown=False, max_daily_loss_quote=1.0))


def test_live_mode_requires_a_daily_loss_limit():
    """Paper mode may run without one; prod may not."""
    with pytest.raises(Exception, match="max_daily_loss_quote|prod"):
        _app(env="prod", risk=RiskConfig(
            max_notional_per_leg_quote=1200, max_position_per_asset=2.0,
            circuit_breaker_bps=250, cancel_all_on_start=False,
            cancel_all_on_shutdown=False, max_daily_loss_quote=None))


def test_live_mode_requires_rotation_to_be_priced():
    """Disabling rotation asserts that moving inventory is free. Acceptable
    while measuring in paper mode; not acceptable with real capital."""
    with pytest.raises(Exception, match="rotation|prod"):
        _app(env="prod", strategy=StrategyConfig(
            target_notional_usd=1000, rotation=RotationConfig(enabled=False)))


def test_live_mode_requires_the_audit_trail():
    """A prod run with no dataset cannot be reconstructed afterwards."""
    with pytest.raises(Exception, match="evaluation_store|prod"):
        _app(env="prod", observability=ObservabilityConfig(
            metrics_port=9000, log_level="INFO", redact_keys=[],
            evaluation_store_enabled=False))


def test_a_pair_configured_for_an_unknown_chain_is_rejected():
    """A pair on a chain with no DEX contracts can never be quoted, and would
    silently log a warning on every cycle forever."""
    with pytest.raises(Exception, match="chain|solana"):
        _app(pairs=[PairConfig(
            base="SOL", quote="USDT", cex_symbol="SOL/USDT",
            max_slippage_bps=30, max_size_quote=5000,
            dex_chain="solana", dex_pool_fee=500)])
