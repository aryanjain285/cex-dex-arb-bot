"""The detector must quote the selected tier, and record which one it used.

A selector nothing consults is a report. Two properties matter:

* the DEX is quoted at the selected tier, not the configured one;
* the evaluation row records the tier ACTUALLY used. `dex_pool_fee` was already
  persisted, so if the detector quoted tier 100 while the row said 500, every
  stored row would be unreproducible -- and the audit trail's whole purpose is
  that a decision can be re-derived from it later.
"""
from decimal import Decimal

import pytest

from src.core.config import (
    DexRoutingConfig, RotationConfig, StrategyConfig, TokenPolicyConfig,
)
from src.core.types import DexQuote
from src.strategy.detector import OpportunityDetector
from tests.fakes import FakeCex, flat_book, make_pair


def D(x) -> Decimal:
    return Decimal(str(x))


class TieredDex:
    """Different prices per fee tier, and a record of which were quoted."""

    def __init__(self, prices_by_fee, gas=D(0)):
        self.prices_by_fee = {int(k): D(v) for k, v in prices_by_fee.items()}
        self.gas = gas
        self.quoted_fees = []

    async def get_pool_address(self, base, quote, chain, fee):
        return ("0x" + f"{fee:040x}") if fee in self.prices_by_fee else None

    async def get_quote(self, pair, size, side, estimate_gas=False):
        self.quoted_fees.append(pair.dex_pool_fee)
        price = self.prices_by_fee.get(pair.dex_pool_fee)
        return None if price is None else DexQuote(price=price, gas_cost_quote=self.gas)


class Recorder:
    def __init__(self):
        self.rows = []

    def record(self, r):
        self.rows.append(r)
        return len(self.rows)


def _strategy(**overrides) -> StrategyConfig:
    fields = dict(
        target_notional_usd=1000, taker_fee_bps=D("7.5"), min_net_bps=D(5),
        rotation=RotationConfig(enabled=False),
        token_policy=TokenPolicyConfig(mode="denylist"),
    )
    fields.update(overrides)
    return StrategyConfig(**fields)


async def test_each_direction_quotes_its_own_best_tier():
    """The two sides genuinely differ, and that is the point.

    With tier 100 at 1100 and tier 3000 at 1000: selling base wants the HIGHER
    price, so CEX_to_DEX picks 100; buying base wants the LOWER price per base, so
    DEX_to_CEX picks 3000. A selector that maximised on both sides would send the
    buy leg to the worst pool available, and the mistake would be invisible.
    """
    dex = TieredDex({100: 1100, 3000: 1000})
    det = OpportunityDetector(
        _strategy(dex_routing=DexRoutingConfig(
            enabled=True, candidate_fee_tiers=[100, 3000], refresh_seconds=300)),
        FakeCex({"ETH/USDT": flat_book(bid=1000, ask=1000)}),
        dex, [make_pair(dex_pool_fee=3000)],
    )

    await det.detect()

    assert 100 in dex.quoted_fees, "the better selling tier was never quoted"
    assert 3000 in dex.quoted_fees, "the better buying tier was never quoted"


async def test_the_recorded_fee_matches_the_direction_that_produced_it():
    """Otherwise a stored row names a pool its price did not come from, and the
    row cannot be re-derived -- which is the audit trail's whole purpose."""
    dex = TieredDex({100: 1100, 3000: 1000})
    rec = Recorder()
    det = OpportunityDetector(
        _strategy(dex_routing=DexRoutingConfig(
            enabled=True, candidate_fee_tiers=[100, 3000], refresh_seconds=300)),
        FakeCex({"ETH/USDT": flat_book(bid=1000, ask=1000)}),
        dex, [make_pair(dex_pool_fee=3000)], store=rec,
    )

    await det.detect()

    priced = {r.direction: r.dex_pool_fee for r in rec.rows if r.dex_price is not None}
    assert priced.get("CEX_to_DEX") == 100, priced
    assert priced.get("DEX_to_CEX") == 3000, priced


async def test_routing_disabled_keeps_the_configured_tier():
    """The escape hatch. An operator who wants one specific pool must be able to
    pin it -- for a pool they have vetted, or to reproduce an earlier run."""
    dex = TieredDex({100: 1100, 3000: 1000})
    det = OpportunityDetector(
        _strategy(dex_routing=DexRoutingConfig(enabled=False)),
        FakeCex({"ETH/USDT": flat_book(bid=1000, ask=1000)}),
        dex, [make_pair(dex_pool_fee=3000)],
    )

    await det.detect()

    assert set(dex.quoted_fees) == {3000}, (
        "the selector ran even though routing is disabled"
    )


async def test_the_opportunity_carries_the_tier_it_was_priced_at():
    """The executor builds the swap from the opportunity, so a mismatch here would
    send the trade to a different pool than the one that was priced."""
    dex = TieredDex({100: 1100, 3000: 1000})
    det = OpportunityDetector(
        _strategy(dex_routing=DexRoutingConfig(
            enabled=True, candidate_fee_tiers=[100, 3000], refresh_seconds=300)),
        FakeCex({"ETH/USDT": flat_book(bid=1000, ask=1000)}),
        dex, [make_pair(dex_pool_fee=3000)],
    )

    found = await det.detect()

    assert found, "a 10% dislocation must produce an opportunity"
    assert found[0].dex_pool_fee == 100
    assert found[0].pair.dex_pool_fee == 100, (
        "the pair on the opportunity still points at the configured pool"
    )


async def test_a_selector_failure_does_not_stop_detection():
    """Routing is an optimisation; it must not be able to halt trading."""
    class Broken(TieredDex):
        async def get_pool_address(self, *a, **k):
            raise RuntimeError("rpc down")

    dex = Broken({3000: 1100})
    det = OpportunityDetector(
        _strategy(dex_routing=DexRoutingConfig(
            enabled=True, candidate_fee_tiers=[100, 3000], refresh_seconds=300)),
        FakeCex({"ETH/USDT": flat_book(bid=1000, ask=1000)}),
        dex, [make_pair(dex_pool_fee=3000)],
    )

    found = await det.detect()

    assert found, "detection stopped because tier selection failed"
    assert found[0].dex_pool_fee == 3000
