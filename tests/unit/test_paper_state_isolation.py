"""Paper PnL must not be written into the live risk state.

`app.py` built `RiskManager(config.risk)` regardless of mode, so a paper run
accumulated its fictional PnL into `data/risk_state.json` -- the same file the
live bot uses to enforce its daily loss limit.

The dangerous direction is the profitable one. A paper run that "makes" 300 quote
units raises the live bot's loss allowance by 300 before it has traded at all,
because the limit is checked against the day's accumulated PnL. The loss limit is
the last control between a malfunction and the end of the capital, and a
simulation must not be able to move it.

The reverse also matters operationally: a paper run that loses can halt the live
bot, which then refuses to trade for a reason that never happened.
"""
from pathlib import Path

import pytest

from src.app import risk_state_path_for_mode
from src.risk.limits import STATE_FILE


def test_live_mode_uses_the_canonical_state_file():
    """The halt-survives-restart guarantee depends on this being the one file."""
    assert risk_state_path_for_mode("live") == STATE_FILE


def test_paper_mode_uses_a_separate_file():
    paper = risk_state_path_for_mode("paper")

    assert paper != STATE_FILE
    assert paper.parent == STATE_FILE.parent, (
        "both live beside each other, so an operator finds them together"
    )
    assert "paper" in paper.name


def test_paper_state_still_persists():
    """Paper state is not thrown away: a paper run's own PnL and halts must
    survive a restart, or a multi-day measurement run would silently reset its
    accounting every time the process bounced."""
    assert risk_state_path_for_mode("paper") is not None


def test_an_unknown_mode_does_not_silently_become_live():
    """A typo in the mode must not point at the live file. Anything unrecognised
    gets its own namespace, so the worst case is separate bookkeeping rather than
    contaminated live state."""
    other = risk_state_path_for_mode("dry-run")

    assert other != STATE_FILE
    assert "dry-run" in other.name


def test_the_paths_are_distinct_across_modes():
    paths = {risk_state_path_for_mode(m) for m in ("live", "paper", "dry-run")}
    assert len(paths) == 3
