"""An expired opportunity must not execute.

`Opportunity.valid_until` was computed by the detector on every opportunity and
read by nobody -- so the TTL was decorative. `grep -rn valid_until src/` found
exactly one site: the assignment.

The field exists because the arbitrage it describes is a claim about two prices
at one instant. Between detection and execution the book can move, and once
execution actually places orders -- network round trips, signatures, block
inclusion -- that gap stops being negligible. Acting on a stale claim is how a
measured edge turns into a realised loss, and it is the failure mode that a
latency-sensitive strategy hits first.

Enforced now, before execution is wired, because the alternative is wiring
execution against an unenforced deadline and then discovering the gap in
production.
"""
from decimal import Decimal

import pytest

from src.core import clock
from src.core.types import Opportunity
from src.strategy.executor import (
    DEFAULT_SANITY_THRESHOLDS, PaperExecutor, TransactionExecutor,
    evaluate_opportunity,
)
from tests.fakes import make_pair


def D(x) -> Decimal:
    return Decimal(str(x))


def _opportunity(valid_until: float) -> Opportunity:
    return Opportunity(
        pair=make_pair(),
        direction="CEX_to_DEX",
        size=D("0.1"),
        cex_price=D(1000),
        dex_price=D(1010),
        dex_chain="ethereum",
        dex_pool_fee=500,
        edge_bps=D(50),
        slippage_bps=D(0),
        gas_cost_quote=D(1),
        cex_fee_quote=D("0.75"),
        expected_pnl_quote=D("0.25"),
        valid_until=valid_until,
    )


def test_a_live_opportunity_passes():
    """Positive control: the gate must not reject everything."""
    now = clock.now()
    legs, pnl, edge, error = evaluate_opportunity(
        _opportunity(valid_until=now + 5.0), DEFAULT_SANITY_THRESHOLDS, now=now
    )

    assert error is None
    assert legs, "a live opportunity must produce legs"


def test_an_expired_opportunity_is_refused_with_a_specific_reason():
    now = clock.now()
    legs, pnl, edge, error = evaluate_opportunity(
        _opportunity(valid_until=now - 0.001), DEFAULT_SANITY_THRESHOLDS, now=now
    )

    assert error == "opportunity_expired", (
        "an expired opportunity must be refused, and the reason must say so -- "
        "a generic rejection cannot be distinguished from a bad price in the "
        "metrics"
    )
    assert legs == [], "no legs may be built for an expired opportunity"


def test_the_deadline_is_exclusive():
    """At exactly valid_until the opportunity is over.

    An inclusive deadline makes `valid_until` mean "and also this instant",
    which is the same class of off-by-one that made a zero TTL a no-op in the
    price oracle.
    """
    now = clock.now()
    _, _, _, error = evaluate_opportunity(
        _opportunity(valid_until=now), DEFAULT_SANITY_THRESHOLDS, now=now
    )

    assert error == "opportunity_expired"


def test_expiry_is_checked_before_the_price_gates():
    """Order matters for diagnosis, not for safety.

    A stale opportunity whose price also drifted out of range should report
    staleness: that is the actionable fact, and it points at latency rather than
    at market data.
    """
    now = clock.now()
    opp = _opportunity(valid_until=now - 1.0)
    broken = opp.model_copy(update={"cex_price": D(0)})

    _, _, _, error = evaluate_opportunity(
        broken, DEFAULT_SANITY_THRESHOLDS, now=now
    )

    assert error == "opportunity_expired"


async def test_the_live_executor_refuses_an_expired_opportunity():
    """The gate has to be reached through the real entry point, not only by
    calling the helper directly."""
    executor = TransactionExecutor(None, None, None, None)

    summary = await executor.run(_opportunity(valid_until=clock.now() - 1.0))

    assert summary.legs == []
    assert summary.pnl_quote == Decimal("0")


async def test_the_paper_executor_refuses_an_expired_opportunity():
    """Paper mode must apply the same deadline as live mode.

    A paper run that executes trades live mode would have skipped reports a
    higher fill rate and a better edge than the strategy can achieve -- which
    defeats the purpose of measuring in paper first.
    """
    executor = PaperExecutor(None)

    summary = await executor.run(_opportunity(valid_until=clock.now() - 1.0))

    assert summary.legs == []


async def test_the_paper_executor_still_fills_a_live_opportunity():
    executor = PaperExecutor(None)

    summary = await executor.run(_opportunity(valid_until=clock.now() + 5.0))

    assert summary.legs, "a live opportunity must still fill in paper mode"
