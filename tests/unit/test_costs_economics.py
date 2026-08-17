"""Trade economics: exactly one place where costs are summed.

The previous model compared a pre-fee edge against a threshold assembled from
cost estimates, which let the same economic quantity be counted twice. Price
impact was deducted from PnL as `slippage_cost` AND added to the required
edge AND already present inside the quoted DEX price -- measured against a
live pool, the quoter returns a price net of both the pool fee and the impact
for the requested size.

These tests pin the corrected model: net PnL is computed once, from
fee-inclusive quotes, and the only deductions are the CEX taker fee and gas.
"""
from decimal import Decimal

import pytest

from src.strategy.costs import evaluate_trade

FEE_BPS = Decimal("7.5")  # Binance spot taker with BNB discount


def D(x) -> Decimal:
    return Decimal(str(x))


# --------------------------------------------------------------------------
# direction: CEX_to_DEX  (buy base on the CEX, sell it on the DEX)
# --------------------------------------------------------------------------

def test_cex_to_dex_net_is_gross_minus_taker_fee_and_gas():
    econ = evaluate_trade(
        direction="CEX_to_DEX",
        size_base=D(1),
        cex_price=D(1000),
        dex_price=D(1010),
        taker_fee_bps=FEE_BPS,
        gas_quote=D(1),
    )
    assert econ.gross_quote == D(10)
    assert econ.cex_fee_quote == D("0.75")      # 1000 * 1 * 0.00075
    assert econ.gas_quote == D(1)
    assert econ.net_quote == D("8.25")


def test_cex_to_dex_notional_is_the_capital_committed_on_the_buy():
    econ = evaluate_trade(
        direction="CEX_to_DEX",
        size_base=D(2),
        cex_price=D(1000),
        dex_price=D(1010),
        taker_fee_bps=FEE_BPS,
        gas_quote=D(0),
    )
    assert econ.notional_quote == D(2000)       # buy side = CEX


# --------------------------------------------------------------------------
# direction: DEX_to_CEX  (buy base on the DEX, sell it on the CEX)
# --------------------------------------------------------------------------

def test_dex_to_cex_net_is_gross_minus_taker_fee_and_gas():
    econ = evaluate_trade(
        direction="DEX_to_CEX",
        size_base=D(1),
        cex_price=D(1010),
        dex_price=D(1000),
        taker_fee_bps=FEE_BPS,
        gas_quote=D(1),
    )
    assert econ.gross_quote == D(10)
    assert econ.cex_fee_quote == D("0.7575")    # fee is on the CEX sale at 1010
    assert econ.net_quote == D("8.2425")


def test_dex_to_cex_notional_uses_the_dex_buy_price_not_the_cex_price():
    """Regression guard for the risk-sizing defect.

    The risk gate computed notional as cex_price * size unconditionally, but
    for DEX_to_CEX the capital is committed on the DEX leg. The two diverge
    most when the spread is widest -- exactly when the limit matters.
    """
    econ = evaluate_trade(
        direction="DEX_to_CEX",
        size_base=D(1),
        cex_price=D(1500),
        dex_price=D(1000),
        taker_fee_bps=FEE_BPS,
        gas_quote=D(0),
    )
    assert econ.notional_quote == D(1000)


# --------------------------------------------------------------------------
# the core regression: no slippage term anywhere
# --------------------------------------------------------------------------

@pytest.mark.parametrize("direction", ["CEX_to_DEX", "DEX_to_CEX"])
def test_net_is_exactly_gross_minus_fee_minus_gas_and_nothing_else(direction):
    """The identity that makes double-counting unrepresentable.

    If any additional cost term is ever reintroduced -- a slippage deduction,
    a pool fee, a second impact charge -- this identity breaks. That is the
    point: the DEX quote already carries the pool fee and the price impact
    for the requested size, so there is nothing further to subtract.
    """
    econ = evaluate_trade(
        direction=direction,
        size_base=D("1.5"),
        cex_price=D(2000) if direction == "DEX_to_CEX" else D(1980),
        dex_price=D(1980) if direction == "DEX_to_CEX" else D(2000),
        taker_fee_bps=FEE_BPS,
        gas_quote=D("0.42"),
    )
    assert econ.net_quote == econ.gross_quote - econ.cex_fee_quote - econ.gas_quote


def test_net_bps_is_relative_to_notional():
    econ = evaluate_trade(
        direction="CEX_to_DEX",
        size_base=D(1),
        cex_price=D(1000),
        dex_price=D(1010),
        taker_fee_bps=FEE_BPS,
        gas_quote=D(1),
    )
    # net 8.25 on notional 1000 -> 82.5 bps
    assert econ.net_bps == D("82.5")


# --------------------------------------------------------------------------
# synthetic pairs: two CEX legs, one on-chain swap
# --------------------------------------------------------------------------

def test_synthetic_pair_charges_two_taker_fees():
    """A synthetic trade is: buy ALT on the CEX, swap ALT->WETH on-chain,
    sell WETH on the CEX. Two CEX legs, so two taker fees.

    The previous model charged one, understating cost by ~7.5 bps on the
    dominant code path (837 of 1,062 pools in the shipped dataset).
    """
    direct = evaluate_trade(
        direction="CEX_to_DEX", size_base=D(1), cex_price=D(1000),
        dex_price=D(1010), taker_fee_bps=FEE_BPS, gas_quote=D(1), cex_legs=1,
    )
    synthetic = evaluate_trade(
        direction="CEX_to_DEX", size_base=D(1), cex_price=D(1000),
        dex_price=D(1010), taker_fee_bps=FEE_BPS, gas_quote=D(1), cex_legs=2,
    )
    assert synthetic.cex_fee_quote == direct.cex_fee_quote * 2
    assert synthetic.net_quote == direct.net_quote - direct.cex_fee_quote


def test_synthetic_pair_charges_gas_once_only():
    """Only one on-chain swap happens, regardless of the CEX leg count.

    The previous model doubled the gas for synthetic pairs, reasoning 'two
    legs' -- but the second leg is a CEX order and burns no gas. That error
    partially masked the missing taker fee, which is why neither was caught.
    """
    econ = evaluate_trade(
        direction="CEX_to_DEX", size_base=D(1), cex_price=D(1000),
        dex_price=D(1010), taker_fee_bps=FEE_BPS, gas_quote=D("2.5"), cex_legs=2,
    )
    assert econ.gas_quote == D("2.5")


# --------------------------------------------------------------------------
# guards
# --------------------------------------------------------------------------

def test_unprofitable_trade_reports_negative_net_rather_than_raising():
    econ = evaluate_trade(
        direction="CEX_to_DEX", size_base=D(1), cex_price=D(1000),
        dex_price=D(1000), taker_fee_bps=FEE_BPS, gas_quote=D(5),
    )
    assert econ.net_quote < 0
    assert econ.net_bps < 0


@pytest.mark.parametrize(
    "kwargs",
    [
        {"size_base": Decimal("0")},
        {"size_base": Decimal("-1")},
        {"cex_price": Decimal("0")},
        {"dex_price": Decimal("0")},
        {"cex_price": Decimal("-1")},
        {"taker_fee_bps": Decimal("-1")},
        {"cex_legs": 0},
    ],
)
def test_invalid_inputs_are_rejected(kwargs):
    base = dict(
        direction="CEX_to_DEX", size_base=D(1), cex_price=D(1000),
        dex_price=D(1010), taker_fee_bps=FEE_BPS, gas_quote=D(1),
    )
    base.update(kwargs)
    with pytest.raises(ValueError):
        evaluate_trade(**base)


def test_unknown_direction_is_rejected():
    with pytest.raises(ValueError):
        evaluate_trade(
            direction="SIDEWAYS", size_base=D(1), cex_price=D(1000),
            dex_price=D(1010), taker_fee_bps=FEE_BPS, gas_quote=D(1),
        )
