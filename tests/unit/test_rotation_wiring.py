"""Rotation cost must reach the live decision, not just exist as a function."""
from decimal import Decimal
import pytest
from src.core.config import RotationConfig, StrategyConfig
from src.strategy.detector import OpportunityDetector, RejectionReason
from tests.fakes import D, FakeCex, FakeDex, flat_book, make_pair


def strategy(**kw):
    defaults = dict(target_notional_usd=1000, taker_fee_bps=D("7.5"), min_net_bps=D(1))
    defaults.update(kw)
    return StrategyConfig(**defaults)


async def test_rotation_cost_is_applied_to_detected_opportunities():
    """A trade profitable without rotation must be rejected with it."""
    pair = make_pair()
    cex = FakeCex({"ETH/USDT": flat_book(bid=1000, ask=1000)})
    dex = FakeDex(sell_price=1001, buy_price=1001)   # 10 bps gross, ~2.5 net

    free = OpportunityDetector(
        strategy(rotation=RotationConfig(enabled=False)), cex, dex, [pair])
    priced = OpportunityDetector(
        strategy(rotation=RotationConfig(
            enabled=True, withdrawal_fee_quote=4.0, bridge_gas_quote=0.0,
            float_quote=5000.0, transfer_risk_bps=0.0)), cex, dex, [pair])

    assert await free.detect(), "profitable before rotation is priced"
    assert not await priced.detect(), "an 0.80/trade rotation must invert it"


async def test_rotation_cost_is_recorded_in_the_audit_trail():
    pair = make_pair()
    cex = FakeCex({"ETH/USDT": flat_book(bid=1000, ask=1000)})
    dex = FakeDex(sell_price=1050, buy_price=1050)

    class Rec:
        def __init__(self): self.rows = []
        def record(self, r): self.rows.append(r); return len(self.rows)

    rec = Rec()
    await OpportunityDetector(
        strategy(rotation=RotationConfig(
            enabled=True, withdrawal_fee_quote=4.0, bridge_gas_quote=1.0,
            float_quote=5000.0, transfer_risk_bps=10.0)),
        cex, dex, [pair], store=rec).detect()

    priced = [r for r in rec.rows if r.rotation_cost_quote is not None]
    assert priced, "rotation cost must be persisted"
    assert priced[0].rotation_cost_quote > 0
    # (4 + 1 + 5000*10/1e4) / 5 = (5 + 5)/5 = 2
    # $4 withdrawal + $1 bridge gas over 5 trades = $1.00. It was $2.00 while the
    # model also subtracted 10 bps of transfer risk on the float as though variance
    # were a negative mean -- that half is now a floor adjustment, not a cost.
    assert priced[0].rotation_cost_quote == D(1)
    # And the exposure it used to double as is visible on the floor instead.
    assert priced[0].min_net_bps == D(11), (
        f"expected the 1 bps base plus a 10 bps risk charge, got "
        f"{priced[0].min_net_bps}"
    )


def test_a_float_too_small_for_the_notional_fails_at_config_load():
    """Better to refuse to start than to silently mis-price every trade."""
    with pytest.raises(Exception):
        StrategyConfig(target_notional_usd=1000, rotation=RotationConfig(
            enabled=True, withdrawal_fee_quote=4.0, bridge_gas_quote=0.0,
            float_quote=500.0, transfer_risk_bps=0.0))


def test_disabled_rotation_is_explicit_not_silent():
    """Zero rotation cost asserts that moving inventory is free. That must be
    a deliberate choice, visible in config, not a default nobody noticed."""
    cfg = StrategyConfig(rotation=RotationConfig(enabled=False))
    assert cfg.rotation.enabled is False
