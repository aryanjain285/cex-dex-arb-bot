"""A refused opportunity must be countable, with its reason.

Two gaps this closes:

* `trades_executed{status="invalid"}` lumped every refusal together, so "the
  price was out of range" and "we were too slow" were the same number. For a
  latency-sensitive strategy the expiry rate is the single most operationally
  important number there is -- it says whether the edge is being lost to the
  market or to the plumbing.
* The paper executor emitted NO metric at all on refusal. Paper mode is the
  measurement phase, so the run whose numbers matter most was the one producing
  no telemetry about what it was throwing away.
"""
from decimal import Decimal

import pytest

from src.core import clock
from src.core.types import Opportunity
from src.infra import metrics
from src.strategy.executor import PaperExecutor, TransactionExecutor
from tests.fakes import make_pair


def D(x) -> Decimal:
    return Decimal(str(x))


def _opportunity(valid_until: float, **overrides) -> Opportunity:
    fields = dict(
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
    fields.update(overrides)
    return Opportunity(**fields)


def _count(pair: str, direction: str, reason: str) -> float:
    """Read the counter directly. A metric asserted through a mock proves the
    mock was called, not that the series exists with the right labels."""
    value = metrics.opportunities_rejected_total.labels(
        pair=pair, direction=direction, reason=reason
    )._value.get()
    return float(value)


async def test_an_expired_opportunity_is_counted_with_its_reason():
    before = _count("ETH/USDT", "CEX_to_DEX", "opportunity_expired")

    await TransactionExecutor(None, None, None, None).run(
        _opportunity(valid_until=clock.now() - 1.0)
    )

    after = _count("ETH/USDT", "CEX_to_DEX", "opportunity_expired")
    assert after == before + 1


async def test_the_paper_executor_counts_refusals_too():
    before = _count("ETH/USDT", "DEX_to_CEX", "opportunity_expired")

    await PaperExecutor(None).run(
        _opportunity(valid_until=clock.now() - 1.0, direction="DEX_to_CEX")
    )

    after = _count("ETH/USDT", "DEX_to_CEX", "opportunity_expired")
    assert after == before + 1, "paper mode must report what it discards"


async def test_reasons_are_distinguished_from_one_another():
    """The point of the label: a price problem and a latency problem must not
    land in the same bucket."""
    expired_before = _count("ETH/USDT", "CEX_to_DEX", "opportunity_expired")
    price_before = _count("ETH/USDT", "CEX_to_DEX", "cex_price_non_positive")

    await PaperExecutor(None).run(
        _opportunity(valid_until=clock.now() + 5.0, cex_price=D(0))
    )

    assert _count("ETH/USDT", "CEX_to_DEX", "cex_price_non_positive") == price_before + 1
    assert _count("ETH/USDT", "CEX_to_DEX", "opportunity_expired") == expired_before, (
        "a price rejection must not be counted as an expiry"
    )


async def test_a_successful_opportunity_increments_nothing():
    before = _count("ETH/USDT", "CEX_to_DEX", "opportunity_expired")

    summary = await PaperExecutor(None).run(_opportunity(valid_until=clock.now() + 5.0))

    assert summary.legs
    assert _count("ETH/USDT", "CEX_to_DEX", "opportunity_expired") == before
