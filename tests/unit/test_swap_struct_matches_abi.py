"""The swap struct must contain exactly the fields the ABI declares.

Known issue #1 in the README said: "the swap call passes a `deadline` field, but
the configured routers are SwapRouter02, which removed `deadline` from that
struct." Half right, and the wrong half was the diagnosis.

Verified against the deployed contracts:

    ABI/router.json    exactInputSingle((address,address,uint24,address,
                                        uint256,uint256,uint160))
    struct fields      tokenIn, tokenOut, fee, recipient, amountIn,
                       amountOutMinimum, sqrtPriceLimitX96      -- seven, no deadline
    selector           0x04e45aaf
    ethereum router    0x68b3465833fb72A70ecDF485E0e4C7bD8665Fc45  selector present
    arbitrum router    0x68b3465833fb72A70ecDF485E0e4C7bD8665Fc45  selector present
    base router        0x2626664c2603336E57B271c5C0b26F421741e481  selector present

So the ABI and the deployed contracts AGREE -- all three are SwapRouter02, and the
ABI is the SwapRouter02 ABI. What disagreed was `execute_swap`, which built an
eight-key dict including a `deadline` the struct does not have. The failure is at
encoding time, in our own process, on the first real swap.

That has a consequence beyond the crash: SwapRouter02's `exactInputSingle` cannot
take a deadline at all. Deadline protection there is via
`multicall(uint256 deadline, bytes[] data)`, which this ABI does not include. So
`dex.swap_deadline_seconds` is unenforceable on this path, and an arbitrage swap
that lands late is a guaranteed loss rather than a late win. Saying so is better
than passing a field that silently does nothing -- which is what the old code
would have done had web3 accepted it.
"""
import json
from pathlib import Path

import pytest


def _router_abi():
    return json.loads(Path("ABI/router.json").read_text(encoding="utf-8"))


def _struct_fields():
    for entry in _router_abi():
        if entry.get("name") != "exactInputSingle":
            continue
        for component in entry.get("inputs", []):
            if component["type"].startswith("tuple"):
                return [c["name"] for c in component["components"]]
    raise AssertionError("exactInputSingle not found in ABI/router.json")


def test_the_abi_is_the_swaprouter02_shape():
    """Seven fields and no deadline. If this ever gains a deadline, the configured
    router addresses need rechecking -- they are SwapRouter02 today."""
    fields = _struct_fields()

    assert fields == [
        "tokenIn", "tokenOut", "fee", "recipient",
        "amountIn", "amountOutMinimum", "sqrtPriceLimitX96",
    ]
    assert "deadline" not in fields


def test_the_code_builds_exactly_the_declared_fields():
    """The actual defect. An extra key fails at encoding time, in our own process,
    on the first real swap -- not on-chain, and not in any test that does not
    execute.

    Read from the source rather than by calling execute_swap, because calling it
    needs a funded wallet and a live chain.
    """
    import ast
    import inspect

    from src.exchange import univ3

    tree = ast.parse(inspect.getsource(univ3))
    built = None
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if getattr(target, "id", "") == "swap_params_struct":
                assert isinstance(node.value, ast.Dict), (
                    "swap_params_struct is no longer a literal dict; this guard "
                    "needs updating to follow it"
                )
                built = [
                    k.value for k in node.value.keys
                    if isinstance(k, ast.Constant)
                ]
    assert built is not None, "swap_params_struct not found in univ3.py"

    declared = _struct_fields()
    extra = [f for f in built if f not in declared]
    missing = [f for f in declared if f not in built]

    assert not extra, (
        f"the swap struct passes fields the ABI does not declare: {extra}. "
        f"web3 cannot encode them, so the first real swap fails in our own process."
    )
    assert not missing, (
        f"the swap struct omits declared fields: {missing}"
    )


def test_the_deadline_config_is_documented_as_unenforceable_on_this_path():
    """`dex.swap_deadline_seconds` is validated and cannot reach the router through
    exactInputSingle. A configured value that does nothing is the pattern this
    codebase has been removing all day, so at minimum the code must say so where
    someone will read it."""
    import inspect

    from src.exchange import univ3

    source = inspect.getsource(univ3)
    assert "multicall" in source, (
        "the deadline limitation is not explained anywhere in the swap path"
    )


def test_the_configured_routers_are_all_swaprouter02():
    """Pinned so a config change to a v1 router does not silently reintroduce the
    mismatch from the other direction."""
    from src.core.config import load_config

    known_swaprouter02 = {
        # Selector 0x04e45aaf confirmed present in the deployed bytecode on
        # 2026-08-17 for the first two.
        "0x68b3465833fb72a70ecdf485e0e4c7bd8665fc45",  # ethereum, arbitrum, others
        "0x2626664c2603336e57b271c5c0b26f421741e481",  # base
        # Uniswap's documented SwapRouter02 on BNB Chain. NOT verified against
        # deployed bytecode here, because no BSC RPC is configured -- so a BSC pair
        # needs that check run before it trades.
        "0xb971ef87ede563556b2ed4b1c0b0019111dd85d2",  # bsc
    }

    config = load_config()
    for chain, contracts in config.dex.uniswap_v3.items():
        assert contracts.router.lower() in known_swaprouter02, (
            f"{chain} router {contracts.router} is not a known SwapRouter02 "
            f"address. ABI/router.json is the SwapRouter02 ABI, so a different "
            f"router needs a different ABI -- verify before trading."
        )
