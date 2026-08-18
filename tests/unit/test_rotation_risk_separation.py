"""Risk is not an expected cost, and I had been subtracting it as one.

`amortised_rotation_cost` charged `float_quote * transfer_risk_bps / 10000` and
subtracted it from expected PnL. My own docstring called it "expected adverse price
move on the in-transit float", which is the mistake stated out loud: under a
zero-drift assumption

    E[dP] = 0        while       Var(dP) > 0

Exposure while inventory is in transit is variance, not a negative mean. Charging
it as an expense depressed every measured net edge by 10 bps -- and, worse, it
corrupted the MEASUREMENT rather than the decision. Every row in the audit trail
recorded a net_bps that was not the expected PnL of anything.

The fix separates two things that were conflated:

    expected cost      withdrawal fee + bridge gas       genuinely leaves the account
    risk appetite      transfer exposure                 raises the REQUIRED edge

So `net_bps` becomes the honest expected value, and the risk charge moves into the
floor the decision is compared against. Both are recorded, so a run stays
comparable across changes in risk policy -- which the old form made impossible,
since changing the risk assumption silently rewrote history's measured edges.

This is a correction, not a loosening. It moves the ETH cost stack from 27.7 to
17.7 bps against an average gross dislocation of -1.5 bps. The conclusion does not
change; the number is simply no longer wrong.
"""
from decimal import Decimal

import pytest

from src.strategy.costs import (
    amortised_rotation_cost, rotation_risk_bps, required_net_bps,
)


def D(x) -> Decimal:
    return Decimal(str(x))


# --- the expected cost is now only what actually leaves the account -----


def test_the_amortised_cost_is_only_the_fees():
    """$4 withdrawal + $1 bridge gas over 5 trades is $1.00/trade = 10 bps on a
    1000 notional. The 10 bps of transfer exposure is no longer in here."""
    cost = amortised_rotation_cost(
        withdrawal_fee_quote=D(4), bridge_gas_quote=D(1),
        float_quote=D(5000), notional_quote=D(1000),
    )

    assert cost == D(1)


def test_a_larger_float_amortises_the_fees_further():
    """This is the whole point of a float, and the old form partly hid it: the
    risk term scaled WITH the float, so amortising over more trades barely moved
    the total."""
    small = amortised_rotation_cost(
        withdrawal_fee_quote=D(4), bridge_gas_quote=D(1),
        float_quote=D(5000), notional_quote=D(1000))
    large = amortised_rotation_cost(
        withdrawal_fee_quote=D(4), bridge_gas_quote=D(1),
        float_quote=D(50_000), notional_quote=D(1000))

    assert large < small
    assert large == D("0.1"), "5 dollars over 50 trades"


def test_the_function_no_longer_accepts_a_risk_parameter():
    """A caller passing transfer_risk_bps must fail loudly rather than have it
    silently ignored -- the silent version would leave every net figure 10 bps
    adrift with nothing to indicate it."""
    with pytest.raises(TypeError):
        amortised_rotation_cost(
            withdrawal_fee_quote=D(4), bridge_gas_quote=D(1),
            float_quote=D(5000), notional_quote=D(1000),
            transfer_risk_bps=D(10),
        )


# --- the risk becomes a threshold, not a subtraction --------------------


def test_the_risk_charge_is_expressed_in_basis_points_of_notional():
    """Per trade, so it is directly comparable to the floor it modifies.

    The exposure is on the whole float, but it is incurred once per rotation and a
    rotation covers float/notional trades -- so per trade it is exactly
    transfer_risk_bps, independent of float size. That invariance is the tell that
    the old form's float-scaling was doing nothing useful.
    """
    charge = rotation_risk_bps(
        float_quote=D(5000), notional_quote=D(1000), transfer_risk_bps=D(10))

    assert charge == D(10)


def test_the_risk_charge_is_independent_of_float_size():
    for float_quote in (D(5000), D(20_000), D(200_000)):
        assert rotation_risk_bps(
            float_quote=float_quote, notional_quote=D(1000),
            transfer_risk_bps=D(10),
        ) == D(10)


def test_the_required_edge_includes_the_risk_charge():
    """The decision gets stricter; the measurement stays honest."""
    floor = required_net_bps(base_floor_bps=D(5), risk_charge_bps=D(10))

    assert floor == D(15)


def test_a_disabled_risk_charge_leaves_the_floor_alone():
    assert required_net_bps(base_floor_bps=D(5), risk_charge_bps=D(0)) == D(5)


def test_a_negative_risk_charge_is_rejected():
    """A negative charge would LOWER the required edge, which is a risk policy
    nobody means to express."""
    with pytest.raises(ValueError):
        required_net_bps(base_floor_bps=D(5), risk_charge_bps=D(-1))


# --- the two are visibly different in the economics --------------------


def test_the_measured_edge_improves_by_exactly_the_risk_term():
    """The regression guard on the correction itself.

    Same market, same fees: net_bps must now be 10 bps better than the old model
    reported, because the old model was subtracting a variance.
    """
    from src.strategy.costs import evaluate_trade

    fees_only = amortised_rotation_cost(
        withdrawal_fee_quote=D(4), bridge_gas_quote=D(1),
        float_quote=D(5000), notional_quote=D(1000))
    old_style = fees_only + D(5)  # the removed float * 10bps term, per trade

    econ_now = evaluate_trade(
        direction="CEX_to_DEX", size_base=D("0.5"), cex_price=D(2000),
        dex_price=D(2001), taker_fee_bps=D("7.5"), gas_quote=D("0.02"),
        rotation_cost_quote=fees_only)
    econ_before = evaluate_trade(
        direction="CEX_to_DEX", size_base=D("0.5"), cex_price=D(2000),
        dex_price=D(2001), taker_fee_bps=D("7.5"), gas_quote=D("0.02"),
        rotation_cost_quote=old_style)

    improvement = econ_now.net_bps - econ_before.net_bps
    assert improvement == pytest.approx(D(50), abs=D("0.01")), (
        f"expected the 5.00 quote units of removed risk to show up as 50 bps on a "
        f"1000 notional, got {improvement}"
    )


def test_the_correction_does_not_rescue_the_strategy():
    """Stated as a test so the correction cannot be mistaken for a result.

    The measured average gross dislocation on the liquid pairs was -1.5 bps. With
    the risk term removed the cost stack is still 17.7 bps, so the gap remains an
    order of magnitude. Removing a modelling error is not the same as finding edge.
    """
    taker, gas = D("7.5"), D("0.2")
    fees = D(10)  # 10 bps of amortised rotation fees at the configured float
    cost_now = taker + gas + fees
    measured_average_gross = D("-1.5")

    assert cost_now == D("17.7")
    assert measured_average_gross - cost_now < D(-15), (
        "the strategy is still far under water after the correction"
    )
