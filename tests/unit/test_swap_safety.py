"""The swap path must be unable to send an unprotected transaction.

`execute_swap` had `amount_out_minimum = 0` with a comment saying it MUST be
derived before production use. A comment is not a control. With a zero floor the
router accepts ANY output, so a sandwich attack can take the entire trade value:
the attacker moves the pool, our swap executes at whatever price results, and the
attacker closes. There is no upper bound on that loss beyond the pool's own
liquidity, which is a different and much larger number than the trade size.

It also approved `2**256 - 1` to the router -- an unlimited allowance. If the
router is ever compromised, or the address is wrong, the blast radius is the whole
balance of that token rather than the trade.

And it priced gas with the legacy `gasPrice` field while the config already
carried `priority_fee_gwei` and `max_fee_gwei`, which nothing read. On a
post-London chain a legacy transaction is accepted but the fee is not adjustable
after the fact, so a transaction submitted into a rising base fee simply sits.

The fixes are structural rather than advisory:

* `DexSwapParams.min_amount_out` is REQUIRED and must be positive, so pydantic
  rejects an unprotected swap at construction -- at every call site, including
  ones not written yet. Nothing currently constructs this type, which is exactly
  when to make it strict.
* the approval is bounded to the amount being spent;
* fees are EIP-1559, from the values already in config.
"""
from decimal import Decimal

import pytest

from src.core.types import DexSwapParams
from src.exchange.univ3 import min_amount_out_wei


def D(x) -> Decimal:
    return Decimal(str(x))


# --- the slippage floor --------------------------------------------------


def test_the_floor_is_the_expected_output_less_the_tolerance():
    # 1000 USDC expected, 30 bps tolerance -> 997 USDC, in 6-decimal units
    assert min_amount_out_wei(D(1000), 30, 6) == 997_000_000


def test_a_zero_tolerance_requires_the_full_expected_output():
    assert min_amount_out_wei(D(1000), 0, 6) == 1_000_000_000


def test_the_floor_rounds_down():
    """Rounding up would set a floor above what the quote promised, so a swap
    that filled exactly as quoted would revert."""
    result = min_amount_out_wei(D("1.0000005"), 0, 6)
    assert result == 1_000_000


def test_a_negative_tolerance_is_rejected():
    """A negative tolerance would demand MORE than the quote, which reverts every
    swap -- and would look like a market problem rather than a config error."""
    with pytest.raises(ValueError):
        min_amount_out_wei(D(1000), -1, 6)


def test_a_tolerance_of_a_whole_turn_is_rejected():
    """10000 bps is 100%: the floor becomes zero, which is the unprotected case
    this function exists to prevent."""
    with pytest.raises(ValueError):
        min_amount_out_wei(D(1000), 10_000, 6)


def test_a_non_positive_expected_output_is_rejected():
    with pytest.raises(ValueError):
        min_amount_out_wei(D(0), 30, 6)


def test_the_floor_is_never_zero_for_a_real_trade():
    """The property that matters, over a range of sizes and decimals: a positive
    expected output with a sane tolerance never produces an unprotected floor."""
    for expected in ("0.01", "1", "1000", "123456.789"):
        for slippage in (1, 30, 100, 500):
            for decimals in (6, 8, 18):
                floor = min_amount_out_wei(D(expected), slippage, decimals)
                assert floor > 0, (expected, slippage, decimals)


def test_a_floor_that_rounds_to_zero_is_refused_rather_than_clamped():
    """A dust-size hole this property test found.

    1 raw unit of expected output (0.000001 at 6 decimals) with a 1 bps tolerance
    gives a floor of 0.9999 raw units, which truncates to zero -- exactly the
    unprotected case. Clamping to 1 would turn "no protection" into a fiction, so
    the swap is refused. Gas alone exceeds such a trade's entire output by orders
    of magnitude; upstream sizing should never produce it.
    """
    with pytest.raises(ValueError, match="rounds to zero|cannot be protected"):
        min_amount_out_wei(D("0.000001"), 1, 6)


def test_the_smallest_representable_floor_is_still_allowed():
    """The boundary: a 1-unit expectation with no tolerance needs the full unit,
    which is representable, so it is permitted."""
    assert min_amount_out_wei(D("0.000001"), 0, 6) == 1


# --- the params object refuses an unprotected swap ----------------------


def _params(**overrides):
    fields = dict(
        chain="ethereum",
        token_in_address="0x" + "11" * 20,
        token_in_decimals=18,
        token_out_address="0x" + "22" * 20,
        token_out_decimals=6,
        fee=500,
        amount_in=D("0.1"),
        min_amount_out=D("300"),
        slippage_bps=30,
    )
    fields.update(overrides)
    return DexSwapParams(**fields)


def test_a_well_formed_swap_is_accepted():
    params = _params()
    assert params.min_amount_out == D("300")


def test_a_swap_with_no_output_floor_cannot_be_constructed():
    """The structural fix. A missing field is a validation error at every call
    site, including the ones that do not exist yet."""
    with pytest.raises(Exception):
        DexSwapParams(
            chain="ethereum",
            token_in_address="0x" + "11" * 20,
            token_in_decimals=18,
            token_out_address="0x" + "22" * 20,
            token_out_decimals=6,
            fee=500,
            amount_in=D("0.1"),
            slippage_bps=30,
        )


def test_a_zero_output_floor_is_rejected():
    """The exact value the code used to hardcode."""
    with pytest.raises(Exception, match="min_amount_out"):
        _params(min_amount_out=D(0))


def test_a_negative_output_floor_is_rejected():
    with pytest.raises(Exception, match="min_amount_out"):
        _params(min_amount_out=D(-1))


def test_a_non_positive_amount_in_is_rejected():
    with pytest.raises(Exception, match="amount_in"):
        _params(amount_in=D(0))


def test_implausible_decimals_are_rejected():
    """A wrong decimals value is a 10^n pricing error, and n is large."""
    with pytest.raises(Exception):
        _params(token_out_decimals=99)


# --- the module no longer contains the unprotected literal --------------


def test_the_zero_output_minimum_is_gone_from_the_module():
    """Checked over the AST, so the module can still describe the old defect in
    prose while the guard is about executable code."""
    import ast
    import inspect

    from src.exchange import univ3

    tree = ast.parse(inspect.getsource(univ3))
    offenders = []
    for node in ast.walk(tree):
        # amount_out_minimum = 0, or a dict entry 'amountOutMinimum': 0
        if isinstance(node, ast.Assign):
            for target in node.targets:
                name = getattr(target, "id", "")
                if "amount_out_minimum" in str(name).lower():
                    if isinstance(node.value, ast.Constant) and node.value.value == 0:
                        offenders.append(f"line {node.lineno}: {name} = 0")
        if isinstance(node, ast.Dict):
            for key, value in zip(node.keys, node.values):
                if (isinstance(key, ast.Constant)
                        and str(key.value) == "amountOutMinimum"
                        and isinstance(value, ast.Constant)
                        and value.value == 0):
                    offenders.append(f"line {node.lineno}: 'amountOutMinimum': 0")

    assert not offenders, f"unprotected swap output floor: {offenders}"


def test_the_unlimited_approval_is_gone_from_the_module():
    """`2**256 - 1` to the router makes the blast radius the whole token balance
    rather than the trade."""
    import ast
    import inspect

    from src.exchange import univ3

    tree = ast.parse(inspect.getsource(univ3))
    for node in ast.walk(tree):
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Pow):
            base = getattr(node.left, "value", None)
            exponent = getattr(node.right, "value", None)
            assert not (base == 2 and exponent == 256), (
                f"line {node.lineno}: an unlimited token approval is back"
            )


def test_legacy_gas_pricing_is_gone_from_the_module():
    """EIP-1559 fields come from config values that previously nothing read."""
    import inspect

    from src.exchange import univ3

    source = inspect.getsource(univ3)
    assert "'gasPrice'" not in source and '"gasPrice"' not in source, (
        "a legacy gasPrice field remains; a legacy transaction submitted into a "
        "rising base fee sits until it is dropped, and cannot be repriced"
    )
    assert "maxFeePerGas" in source and "maxPriorityFeePerGas" in source
