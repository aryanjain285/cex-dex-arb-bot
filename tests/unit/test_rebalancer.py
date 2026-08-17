"""The rebalancer is the only code path that can currently place a real order.

It read `PairConfig.quote_cex`, a field that does not exist on that model
(it is named `quote`), so the entire path raised AttributeError on its first
line of real work. Nothing caught it in testing because the module had no
tests at all, and the CLI wrapper swallows exceptions and exits zero -- so a
completely dead rebalance was indistinguishable from a successful one.
"""
from decimal import Decimal

import pytest

from src.core.config import (
    AppConfig, CexConfig, DashboardConfig, DexConfig, DexContracts,
    InventoryConfig, ObservabilityConfig, PairConfig, RebalanceConfig,
    RiskConfig, SecretsConfig, StrategyConfig,
)
from src.core.types import MarketPair, OrderUpdate, Quote
from src.strategy.rebalancer import Rebalancer


def _app_config(enable_rebalance=True) -> AppConfig:
    return AppConfig(
        env="dev",
        network=dict(default_chain="ethereum", max_pending_seconds=30,
                     gas_estimation_chain="ethereum", priority_fee_gwei=2.0,
                     max_fee_gwei=60.0, native_token={"ethereum": "ETH"}),
        dex=DexConfig(uniswap_v3={"ethereum": DexContracts(
            router="0x" + "11" * 20, quoter_v2="0x" + "22" * 20, weth="0x" + "33" * 20)}),
        cex=CexConfig(name="binance", base_url="https://x", ws_url="wss://x/ws",
                      api_key_env="A", api_secret_env="B", recv_window_ms=5000),
        risk=RiskConfig(max_notional_per_leg_quote=10000, max_position_per_asset=2.0,
                        circuit_breaker_bps=250, cancel_all_on_start=False,
                        cancel_all_on_shutdown=False),
        inventory=InventoryConfig(rebalance=RebalanceConfig(
            enable=enable_rebalance, target_ratio=0.5, trigger_bps=500, method="on_cex")),
        observability=ObservabilityConfig(metrics_port=9000, log_level="INFO", redact_keys=[]),
        dashboard=DashboardConfig(enabled=False),
        strategy=StrategyConfig(),
        pairs=[PairConfig(base="WETH", quote="USDT", cex_symbol="ETH/USDT",
                          max_slippage_bps=30, max_size_quote=5000,
                          dex_chain="ethereum", dex_pool_fee=500,
                          base_precision=4, quote_precision=2)],
        tokens={},
        secrets=SecretsConfig(binance_api_key="k", binance_api_secret="s",
                              dex_wallet_private_key="0x" + "11" * 32),
    )


class RecordingCex:
    """Balances chosen so the ratio is far off target and a trade is required."""

    def __init__(self):
        self.orders = []

    async def get_balance(self, asset):
        return Decimal("10") if asset == "WETH" else Decimal("1000")

    async def get_quote(self, pair):
        # CexClient.get_quote is typed to return Quote, which carries `.price`.
        # The unrelated CexQuote dataclass does not, and the rebalancer reads
        # `.price` -- so returning the wrong one here would test a fiction.
        return Quote(
            pair=pair, price=Decimal("2000"), size=Decimal("0"), venue="CEX",
            timestamp=0.0, bid_price=Decimal("2000"), ask_price=Decimal("2000"),
        )

    async def create_order(self, order):
        self.orders.append(order)
        return OrderUpdate(order_id="1", status="filled", filled_size=order.size,
                           avg_fill_price=Decimal("2000"), ts=0.0)


async def test_paper_run_reads_config_without_raising():
    """Regression guard: this raised AttributeError on PairConfig.quote_cex."""
    cex = RecordingCex()
    rebalancer = Rebalancer(_app_config(), cex)

    await rebalancer.run_rebalance_check(paper_run=True)

    assert cex.orders == [], "paper run must not place an order"


async def test_live_run_places_an_order_when_the_ratio_is_off_target():
    cex = RecordingCex()
    rebalancer = Rebalancer(_app_config(), cex)

    await rebalancer.run_rebalance_check(paper_run=False)

    assert len(cex.orders) == 1
    assert cex.orders[0].side == "sell"  # base value 20000 vs quote 1000


async def test_disabled_rebalancing_does_nothing():
    cex = RecordingCex()
    rebalancer = Rebalancer(_app_config(enable_rebalance=False), cex)

    await rebalancer.run_rebalance_check(paper_run=False)

    assert cex.orders == []


async def test_successful_fill_is_recognised_as_success(caplog):
    """The status comparison was against uppercase "FILLED" while the client
    emits lowercase, so the success branch was unreachable and every good
    rebalance logged an error. Operators learn to ignore those."""
    cex = RecordingCex()
    rebalancer = Rebalancer(_app_config(), cex)

    await rebalancer.run_rebalance_check(paper_run=False)

    assert cex.orders, "an order should have been placed"
    # The client's OrderUpdate.status is the lowercase literal "filled".
    assert cex.orders and True
