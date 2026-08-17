"""Risk state must not be shared between a backtest and the live bot.

`STATE_FILE = Path("data/risk_state.json")` was a module constant, so every
RiskManager in the process -- live, paper, backtest, or test -- read and wrote the
same file. Two consequences, both discovered by running the repaired backtest and
watching it load the live state:

    Risk manager initialised. PnL today: 194.0800

* A backtest CONSUMES the live daily loss budget. Replay a losing day and the
  live bot inherits the loss, and can halt on it. The bot then refuses to trade
  for reasons that exist only in a simulation.
* A backtest can also CLEAR a halt or inflate the day's PnL, which is worse: the
  live loss limit is the last line of defence before capital is gone, and a
  simulation must not be able to move it.

The fix is a per-instance state path, with None meaning "in memory, persist
nothing" -- which is what a simulation should always use.
"""
import json
from decimal import Decimal
from pathlib import Path

import pytest

from src.core.config import RiskConfig
from src.core.types import ExecutionLeg, ExecutionSummary, MarketPair
from src.risk.limits import STATE_FILE, RiskManager
from tests.fakes import make_pair


def _risk_config(**kw) -> RiskConfig:
    defaults = dict(
        max_notional_per_leg_quote=1200,
        max_position_per_asset=10.0,
        circuit_breaker_bps=250,
        cancel_all_on_start=False,
        cancel_all_on_shutdown=False,
        max_daily_loss_quote=250.0,
    )
    defaults.update(kw)
    return RiskConfig(**defaults)


def _summary(pnl: Decimal) -> ExecutionSummary:
    # A leg is required: update_state ignores a summary with none, because
    # nothing was executed and there is nothing to record.
    leg = ExecutionLeg(venue="CEX", side="buy", price_quote=Decimal("1000"),
                       size=Decimal("0.1"), fees_quote=Decimal("0.075"))
    return ExecutionSummary(
        pair=make_pair(),
        direction="CEX_to_DEX",
        size=Decimal("0.1"),
        legs=[leg],
        gas_quote=Decimal("0"),
        pnl_quote=pnl,
        edge_bps=Decimal("10"),
        hedged=False,
        started_ts=0.0,
        completed_ts=0.0,
    )


def test_an_in_memory_manager_writes_nothing(tmp_path, monkeypatch):
    """The simulation mode. Nothing on disk changes, wherever it points."""
    sentinel = tmp_path / "risk_state.json"
    sentinel.write_text(json.dumps({"date": "1970-01-01", "daily_pnl": "0"}),
                        encoding="utf-8")
    monkeypatch.setattr("src.risk.limits.STATE_FILE", sentinel)
    before = sentinel.read_text(encoding="utf-8")

    manager = RiskManager(_risk_config(), state_path=None)
    manager.update_state(_summary(Decimal("-100")))

    assert sentinel.read_text(encoding="utf-8") == before, (
        "an in-memory risk manager persisted state anyway"
    )
    assert manager.daily_pnl == Decimal("-100"), (
        "it must still track PnL in memory, or limits could not fire at all"
    )


def test_an_in_memory_manager_starts_clean_even_when_a_state_file_exists(tmp_path, monkeypatch):
    """A backtest must not inherit the live bot's loss for the day."""
    state = tmp_path / "risk_state.json"
    from src.risk.limits import get_current_date_str

    state.write_text(
        json.dumps({"date": get_current_date_str(), "daily_pnl": "-240",
                    "halted": False, "halt_reason": "", "positions": {}}),
        encoding="utf-8",
    )
    monkeypatch.setattr("src.risk.limits.STATE_FILE", state)

    manager = RiskManager(_risk_config(), state_path=None)

    assert manager.daily_pnl == Decimal("0")
    assert not manager.halted


def test_an_in_memory_manager_does_not_inherit_a_halt(tmp_path, monkeypatch):
    """And the reverse of the previous test's danger: a simulation must not be
    able to observe -- or later clear -- a live halt."""
    from src.risk.limits import get_current_date_str

    state = tmp_path / "risk_state.json"
    state.write_text(
        json.dumps({"date": get_current_date_str(), "daily_pnl": "0",
                    "halted": True, "halt_reason": "manual", "positions": {}}),
        encoding="utf-8",
    )
    monkeypatch.setattr("src.risk.limits.STATE_FILE", state)

    simulated = RiskManager(_risk_config(), state_path=None)
    simulated.halt("simulated blowup")

    persisted = json.loads(state.read_text(encoding="utf-8"))
    assert persisted["halted"] is True, "the live halt must be untouched"
    assert persisted["halt_reason"] == "manual", (
        "a simulation overwrote the live halt reason"
    )


def test_an_explicit_path_is_honoured(tmp_path):
    """Two live managers can be isolated from each other too -- which is what a
    second bot instance on the same machine needs."""
    path = tmp_path / "own_state.json"

    manager = RiskManager(_risk_config(), state_path=path)
    manager.update_state(_summary(Decimal("-12.5")))

    assert path.exists()
    assert json.loads(path.read_text(encoding="utf-8"))["daily_pnl"] == "-12.5"

    reloaded = RiskManager(_risk_config(), state_path=path)
    assert reloaded.daily_pnl == Decimal("-12.5")


def test_the_default_is_still_the_shared_live_path(tmp_path, monkeypatch):
    """The live bot's behaviour must not change: omitting the argument keeps
    using the one persisted file, which is what makes a halt survive a restart."""
    path = tmp_path / "risk_state.json"
    monkeypatch.setattr("src.risk.limits.STATE_FILE", path)

    manager = RiskManager(_risk_config())
    manager.update_state(_summary(Decimal("-1")))

    assert path.exists(), "the default manager must still persist"


async def test_the_backtest_does_not_touch_the_live_risk_state(tmp_path, monkeypatch):
    """The end-to-end guard, through the Simulator itself."""
    import textwrap

    from backtest.datasets import load_dataset
    from backtest.simulator import Simulator
    from src.core.config import load_config
    from src.risk.limits import get_current_date_str

    live_state = tmp_path / "risk_state.json"
    live_state.write_text(
        json.dumps({"date": get_current_date_str(), "daily_pnl": "-100",
                    "halted": False, "halt_reason": "", "positions": {}}),
        encoding="utf-8",
    )
    monkeypatch.setattr("src.risk.limits.STATE_FILE", live_state)
    before = live_state.read_text(encoding="utf-8")

    csv = tmp_path / "d.csv"
    csv.write_text(textwrap.dedent("""
        timestamp,cex_bid_price,cex_ask_price,dex_price,gas_quote
        2026-01-01T00:00:00Z,1000,1000.1,1010.0,0.10
    """).strip() + "\n", encoding="utf-8")

    sim = Simulator(load_config(), load_dataset(str(csv)))
    await sim.run()

    assert sim.results, "the fixture must actually trade, or this proves nothing"
    assert live_state.read_text(encoding="utf-8") == before, (
        "the backtest wrote to the live risk state"
    )
    assert sim.risk_manager.daily_pnl != Decimal("-100"), (
        "the backtest inherited the live daily loss"
    )
