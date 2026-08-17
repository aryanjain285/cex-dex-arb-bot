"""Every recorded evaluation says whether its tokens are cleared for capital.

There is a real tension in the measurement phase. The allowlist is short -- five
tokens -- so an allowlisted paper run observes three pairs, which is too thin to
measure an edge distribution. Denylist mode observes the whole market, but then
the distribution silently mixes two different things: edges on tokens we could
trade, and edges on tokens we never would (a fee-on-transfer token shows a
120 bps "edge" that is really its transfer tax).

Reporting one number over that mixture would overstate the strategy, which is
the exact failure mode paper trading exists to prevent.

The fix is to record the verdict rather than choose between breadth and honesty:
every row carries the verdict of the STRICT (default-deny) policy, whatever mode
the run is in. Breadth for the measurement, and an honest tradeable subset from
the same dataset -- and the allowlist can then be grown from evidence instead of
from a guess.
"""
from decimal import Decimal

import pytest

from src.core.config import (
    RotationConfig, StrategyConfig, TokenPolicyConfig,
)
from src.strategy.detector import OpportunityDetector
from tests.fakes import FakeCex, FakeDex, flat_book, make_pair


class Recorder:
    def __init__(self):
        self.rows = []

    def record(self, r):
        self.rows.append(r)
        return len(self.rows)


def _strategy(mode: str) -> StrategyConfig:
    return StrategyConfig(
        target_notional_usd=1000, taker_fee_bps=Decimal("7.5"),
        min_net_bps=Decimal(5), rotation=RotationConfig(enabled=False),
        token_policy=TokenPolicyConfig(mode=mode, allowed=["WETH", "USDT"]),
    )


async def _run(mode: str, pair):
    rec = Recorder()
    det = OpportunityDetector(
        _strategy(mode),
        FakeCex({pair.cex_symbol: flat_book(bid=1000, ask=1000)}),
        FakeDex(sell_price=1050, buy_price=1050),
        [pair], store=rec,
    )
    found = await det.detect()
    return found, rec.rows


async def test_an_allowlisted_pair_is_recorded_as_allowed():
    _, rows = await _run("denylist", make_pair())  # WETH/USDT

    assert rows
    assert all(r.policy_verdict == "allowed" for r in rows), (
        f"got {[r.policy_verdict for r in rows]}"
    )


async def test_a_non_allowlisted_pair_is_evaluated_but_flagged_in_denylist_mode():
    """Breadth AND honesty: the row exists, and it says it is not tradeable."""
    pair = make_pair("NEW/USDT", base="NEWCOIN")

    found, rows = await _run("denylist", pair)

    assert rows, "the evaluation must still be recorded for the measurement"
    assert all(r.policy_verdict == "not_allowlisted" for r in rows)
    # It was genuinely evaluated, not short-circuited: the economics are there.
    assert any(r.net_bps is not None for r in rows)
    assert found, "denylist mode must still surface the opportunity"


async def test_a_hazardous_token_is_flagged_as_denied_not_merely_unlisted():
    """"Denied" and "nobody has looked at it" are different claims, and the
    analysis will want to treat them differently."""
    pair = make_pair("LINGO/USDT", base="LINGO")

    _, rows = await _run("denylist", pair)

    assert rows
    assert all(r.policy_verdict == "denied" for r in rows), (
        f"got {[r.policy_verdict for r in rows]}"
    )


async def test_allowlist_mode_records_the_denial_instead_of_an_evaluation():
    """In allowlist mode the pair never reaches the economics, so the row is a
    rejection -- but it still carries the verdict, so the two modes produce
    comparable datasets."""
    from src.strategy.detector import RejectionReason

    pair = make_pair("LINGO/USDT", base="LINGO")

    found, rows = await _run("allowlist", pair)

    assert found == []
    assert rows
    assert all(r.reason == RejectionReason.TOKEN_DENIED for r in rows)
    assert all(r.policy_verdict == "denied" for r in rows)


async def test_the_verdict_survives_a_round_trip_through_the_store(tmp_path):
    """A column that is written but unreadable is not an audit trail."""
    from src.infra.evaluation_store import EvaluationStore

    store = EvaluationStore(tmp_path / "e.sqlite3", run_id="verdict-test")
    try:
        pair = make_pair("LINGO/USDT", base="LINGO")
        det = OpportunityDetector(
            _strategy("denylist"),
            FakeCex({"LINGO/USDT": flat_book(bid=1000, ask=1000)}),
            FakeDex(sell_price=1050, buy_price=1050),
            [pair], store=store,
        )
        await det.detect()

        rows = store.all_rows()
        assert rows
        assert all(r["policy_verdict"] == "denied" for r in rows)
    finally:
        store.close()
