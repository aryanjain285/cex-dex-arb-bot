"""Inventory rotation is a real cost and it was entirely unmodelled.

Two audits reached this independently -- the risk officer from the code, the
quant from the arithmetic. A CEX<->DEX arb is NOT a round trip: it is an
inventory rotation. `CEX_to_DEX` buys base on the exchange and sells base from
the on-chain wallet. Those are different assets in different custody, so to
trade again you must physically move inventory, which costs a withdrawal fee,
on-chain gas, and minutes of unhedged price exposure.

`net = gross - cex_fee - gas` captured none of it. At $1000 notional and a
5 bps floor the expected profit is $0.50/trade, against a ~$4 withdrawal fee.
Amortised over the trades a float supports, the strategy was negative-EV at
its own threshold while reporting every trade as profitable.
"""
from decimal import Decimal

import pytest

from src.strategy.costs import amortised_rotation_cost, evaluate_trade

FEE_BPS = Decimal("7.5")


def D(x) -> Decimal:
    return Decimal(str(x))


# --------------------------------------------------------------------------
# the amortisation itself
# --------------------------------------------------------------------------

def test_rotation_cost_is_the_fee_divided_by_trades_it_supports():
    """A $4 withdrawal that funds 5 trades costs $0.80 per trade."""
    cost = amortised_rotation_cost(
        withdrawal_fee_quote=D(4), bridge_gas_quote=D(0),
        float_quote=D(5000), notional_quote=D(1000), transfer_risk_bps=D(0),
    )
    assert cost == D("0.8")


def test_bridge_gas_is_included_in_the_rotation():
    cost = amortised_rotation_cost(
        withdrawal_fee_quote=D(4), bridge_gas_quote=D(1),
        float_quote=D(5000), notional_quote=D(1000), transfer_risk_bps=D(0),
    )
    assert cost == D(1), "(4 + 1) / 5 trades"


def test_transfer_price_risk_is_charged_on_the_moved_notional():
    """Inventory in transit is unhedged. A 50 bps expected adverse move on the
    rotated float, amortised the same way, is a real cost of doing business."""
    cost = amortised_rotation_cost(
        withdrawal_fee_quote=D(0), bridge_gas_quote=D(0),
        float_quote=D(5000), notional_quote=D(1000), transfer_risk_bps=D(50),
    )
    # 5000 * 50/10000 = 25 of risk, over 5 trades = 5 per trade
    assert cost == D(5)


def test_a_larger_float_amortises_the_fee_further():
    small = amortised_rotation_cost(
        withdrawal_fee_quote=D(4), bridge_gas_quote=D(0),
        float_quote=D(2000), notional_quote=D(1000), transfer_risk_bps=D(0))
    large = amortised_rotation_cost(
        withdrawal_fee_quote=D(4), bridge_gas_quote=D(0),
        float_quote=D(20000), notional_quote=D(1000), transfer_risk_bps=D(0))
    assert small > large
    assert small == D(2) and large == D("0.2")


def test_a_float_smaller_than_one_trade_is_rejected():
    """You cannot run a $1000 notional off a $500 float; that is a
    misconfiguration, not a very high cost."""
    with pytest.raises(ValueError):
        amortised_rotation_cost(
            withdrawal_fee_quote=D(4), bridge_gas_quote=D(0),
            float_quote=D(500), notional_quote=D(1000), transfer_risk_bps=D(0))


@pytest.mark.parametrize("bad", [
    {"withdrawal_fee_quote": Decimal("-1")},
    {"bridge_gas_quote": Decimal("-1")},
    {"transfer_risk_bps": Decimal("-1")},
    {"notional_quote": Decimal("0")},
])
def test_invalid_rotation_inputs_are_rejected(bad):
    args = dict(withdrawal_fee_quote=D(4), bridge_gas_quote=D(0),
                float_quote=D(5000), notional_quote=D(1000),
                transfer_risk_bps=D(0))
    args.update(bad)
    with pytest.raises(ValueError):
        amortised_rotation_cost(**args)


# --------------------------------------------------------------------------
# it must reach the decision
# --------------------------------------------------------------------------

def test_rotation_cost_reduces_net_pnl():
    without = evaluate_trade(
        direction="CEX_to_DEX", size_base=D(1), cex_price=D(1000),
        dex_price=D(1010), taker_fee_bps=FEE_BPS, gas_quote=D(0))
    with_rot = evaluate_trade(
        direction="CEX_to_DEX", size_base=D(1), cex_price=D(1000),
        dex_price=D(1010), taker_fee_bps=FEE_BPS, gas_quote=D(0),
        rotation_cost_quote=D("0.8"))

    assert with_rot.net_quote == without.net_quote - D("0.8")
    assert with_rot.rotation_cost_quote == D("0.8")


@pytest.mark.parametrize("direction", ["CEX_to_DEX", "DEX_to_CEX"])
def test_the_cost_identity_now_includes_rotation(direction):
    """The identity that keeps double-counting unrepresentable, extended.

    Every cost appears exactly once, and net is exactly their sum subtracted
    from gross -- nothing more.
    """
    econ = evaluate_trade(
        direction=direction, size_base=D("1.5"),
        cex_price=D(2000) if direction == "DEX_to_CEX" else D(1980),
        dex_price=D(1980) if direction == "DEX_to_CEX" else D(2000),
        taker_fee_bps=FEE_BPS, gas_quote=D("0.42"), rotation_cost_quote=D("1.1"))

    assert econ.net_quote == (
        econ.gross_quote - econ.cex_fee_quote - econ.gas_quote
        - econ.rotation_cost_quote
    )


def test_rotation_cost_makes_a_marginal_trade_unprofitable():
    """The headline finding, as a test. A trade that clears the floor on the
    old model must fail once rotation is priced."""
    # 10 bps gross, 7.5 bps fee -> +2.5 bps net before rotation
    args = dict(direction="CEX_to_DEX", size_base=D(1), cex_price=D(1000),
                dex_price=D(1001), taker_fee_bps=FEE_BPS, gas_quote=D(0))
    assert evaluate_trade(**args).net_quote > 0
    assert evaluate_trade(**args, rotation_cost_quote=D("0.8")).net_quote < 0


def test_negative_rotation_cost_is_rejected():
    with pytest.raises(ValueError):
        evaluate_trade(
            direction="CEX_to_DEX", size_base=D(1), cex_price=D(1000),
            dex_price=D(1010), taker_fee_bps=FEE_BPS, gas_quote=D(0),
            rotation_cost_quote=D("-1"))
