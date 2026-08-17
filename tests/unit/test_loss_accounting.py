"""A trading system must be able to represent a loss.

Three independent audits found this defect from three directions. Two
mechanisms combined to make losses unrecordable:

1. `metrics.pnl_quote` was a Prometheus Counter, and `Counter.inc()` raises
   ValueError on a negative argument. The raise happened inside
   `executor.run()` BEFORE it returned, so the exception propagated past
   `risk_manager.update_state()` in the main loop -- the loss was discarded
   from P&L accounting entirely while gains persisted.

2. `daily_pnl` accumulated `opp.expected_pnl_quote`, the detector's
   expectation, which passed `net_bps >= floor >= 0` by construction. Every
   recorded value was therefore positive by definition, making any
   daily-loss limit built on it unfalsifiable.

The result was a system reporting an unbroken string of profitable trades
while the account drained. These tests make that state unreachable.
"""
from decimal import Decimal

import pytest

from src.core.config import PairConfig, RiskConfig
from src.core.types import ExecutionLeg, ExecutionSummary, MarketPair, Opportunity
from src.risk.limits import RiskManager
from src.strategy.executor import TransactionExecutor


def D(x) -> Decimal:
    return Decimal(str(x))


@pytest.fixture
def pair_config() -> PairConfig:
    return PairConfig(
        base="WETH", quote="USDT", cex_symbol="ETH/USDT", max_slippage_bps=30,
        max_size_quote=5000, dex_chain="ethereum", dex_pool_fee=500,
        price_floor_quote=D(100), price_ceiling_quote=D(100000),
        max_edge_bps=1000, base_precision=4, quote_precision=2,
    )


@pytest.fixture
def market_pair(pair_config) -> MarketPair:
    return MarketPair(
        base="WETH", quote_cex="USDT", quote_dex="USDT", cex_symbol="ETH/USDT",
        dex_chain="ethereum", dex_pool_fee=500, max_slippage_bps=30,
    )


def losing_opportunity(market_pair) -> Opportunity:
    """A realised loss: bought high on the CEX, sold low on the DEX."""
    return Opportunity(
        pair=market_pair, direction="CEX_to_DEX", size=D("0.5"),
        cex_price=D(2000), dex_price=D(1990),
        dex_chain="ethereum", dex_pool_fee=500,
        edge_bps=D("-50"), slippage_bps=D(30),
        gas_cost_quote=D("0.02"), cex_fee_quote=D("0.75"),
        expected_pnl_quote=D("-5.77"), valid_until=0.0,
    )


def risk_config(**kw) -> RiskConfig:
    defaults = dict(
        max_notional_per_leg_quote=2000.0, max_position_per_asset=2.0,
        circuit_breaker_bps=250, cancel_all_on_start=False,
        cancel_all_on_shutdown=False, max_daily_loss_quote=100.0,
    )
    defaults.update(kw)
    return RiskConfig(**defaults)


# --------------------------------------------------------------------------

def test_executor_does_not_raise_on_a_losing_trade(pair_config, market_pair):
    """The Counter defect: a negative PnL must not raise out of run()."""
    import asyncio

    executor = TransactionExecutor(None, None, None, [pair_config])
    summary = asyncio.run(executor.run(losing_opportunity(market_pair)))

    assert summary.pnl_quote < 0, "the loss must survive to the summary"
    assert summary.legs, "a loss is still a completed trade, not an invalid one"


def test_risk_manager_records_a_loss(tmp_path, monkeypatch, market_pair):
    import src.risk.limits as limits

    monkeypatch.setattr(limits, "STATE_FILE", tmp_path / "risk_state.json")
    rm = RiskManager(risk_config())

    rm.update_state(ExecutionSummary(
        pair=market_pair, direction="CEX_to_DEX", size=D("0.5"),
        legs=[ExecutionLeg(venue="CEX", side="buy", price_quote=D(2000),
                           size=D("0.5"), fees_quote=D("0.75"))],
        gas_quote=D("0.02"), pnl_quote=D("-5.77"), edge_bps=D("-50"),
        hedged=False, started_ts=0.0, completed_ts=1.0,
    ))

    assert rm.daily_pnl == D("-5.77"), "a loss must accumulate as negative"


def test_daily_loss_limit_blocks_further_trading(tmp_path, monkeypatch, market_pair):
    """The limit README claims and config declares, which did not exist."""
    import src.risk.limits as limits

    monkeypatch.setattr(limits, "STATE_FILE", tmp_path / "risk_state.json")
    rm = RiskManager(risk_config(max_daily_loss_quote=100.0))
    opp = losing_opportunity(market_pair)

    assert rm.is_trade_allowed(opp), "should trade before the limit is hit"

    rm.update_state(ExecutionSummary(
        pair=market_pair, direction="CEX_to_DEX", size=D("0.5"),
        legs=[ExecutionLeg(venue="CEX", side="buy", price_quote=D(2000),
                           size=D("0.5"), fees_quote=D("0.75"))],
        gas_quote=D(0), pnl_quote=D("-150"), edge_bps=D("-50"),
        hedged=False, started_ts=0.0, completed_ts=1.0,
    ))

    assert not rm.is_trade_allowed(opp), (
        "a breached daily loss limit must block all further trading"
    )


def test_state_survives_a_truncated_write(tmp_path, monkeypatch):
    """Corrupt state must NOT silently reset the loss budget to zero.

    The previous behaviour caught JSONDecodeError, set daily_pnl = 0, logged
    an error and kept trading -- so a well-timed crash defeated any daily
    loss limit. Refusing to start is the only safe response.
    """
    import src.risk.limits as limits

    state = tmp_path / "risk_state.json"
    monkeypatch.setattr(limits, "STATE_FILE", state)
    state.write_text('{\n  "date": "2026-08-17",\n  "daily_pnl": -42', encoding="utf-8")

    with pytest.raises(Exception):
        RiskManager(risk_config())


def test_state_write_is_atomic(tmp_path, monkeypatch, market_pair):
    """No truncate-then-write: a crash must leave the prior record intact."""
    import src.risk.limits as limits

    state = tmp_path / "risk_state.json"
    monkeypatch.setattr(limits, "STATE_FILE", state)
    rm = RiskManager(risk_config())
    rm.daily_pnl = D("-10.5")
    rm._save_state()

    # No stray temp files left behind, and the record is readable.
    import json
    assert json.loads(state.read_text(encoding="utf-8"))["daily_pnl"] == "-10.5"
    assert list(tmp_path.glob("*.tmp")) == [], "temp file should be replaced, not left"


def test_money_is_persisted_as_a_string_not_a_float(tmp_path, monkeypatch):
    """Consistent with the audit store's Decimal-as-TEXT discipline."""
    import json

    import src.risk.limits as limits

    state = tmp_path / "risk_state.json"
    monkeypatch.setattr(limits, "STATE_FILE", state)
    rm = RiskManager(risk_config())
    rm.daily_pnl = D("0.1") + D("0.2")
    rm._save_state()

    raw = json.loads(state.read_text(encoding="utf-8"))
    assert isinstance(raw["daily_pnl"], str)
    assert Decimal(raw["daily_pnl"]) == D("0.3")
