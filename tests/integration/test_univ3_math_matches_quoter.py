"""The local swap math must agree with the deployed QuoterV2, on real pools.

This is the test that decides whether the local simulator can be trusted. Unit
tests pin the fixed-point primitives, but only the chain can say whether the swap
loop reproduces what the contract would actually do -- including tick crossings,
rounding direction, and the exact fee arithmetic.

The relationship is deliberately one of oracle and implementation: QuoterV2 stays
in the codebase as the thing the local math is CHECKED against, not as the thing
the hot path calls. That inversion is the point of the whole exercise.

Skipped without an RPC URL, because it is a differential test against live state
and there is nothing to differ from otherwise. Run it before trusting a local
quote for anything that moves money:

    ETH_RPC_URL=... python -m pytest tests/integration -q

Tolerance is ZERO on the raw integer amount. A tolerance would hide precisely the
class of error this exists to catch: a rounding direction that makes local quotes
fractionally better than reality manufactures edge exactly where the strategy
looks for it.
"""
import os
from decimal import Decimal

import pytest

pytestmark = pytest.mark.skipif(
    not os.getenv("ETH_RPC_URL"),
    reason="differential test against live chain state; set ETH_RPC_URL to run",
)

# Real Ethereum pools spanning the fee tiers and both token orderings.
POOLS = [
    pytest.param("0x88e6A0c2dDD26FEEb64F039a2c41296FcB3f5640", id="USDC/WETH-500"),
    pytest.param("0x8ad599c3A0ff1De082011EFDDc58f1908eb6e6D8", id="USDC/WETH-3000"),
    pytest.param("0x11b815efB8f581194ae79006d24E0d814B7697F6", id="WETH/USDT-500"),
    pytest.param("0xC5aF84701f98Fa483eCe78aF83F11b6C38ACA71D", id="WETH/USDT-10000"),
]


def _client():
    from src.core.config import load_config
    from src.exchange.univ3 import UniV3DexClient

    config = load_config()
    return UniV3DexClient(config.dex, config.network, config.secrets, config.tokens)


@pytest.mark.parametrize("pool_address", POOLS)
@pytest.mark.parametrize("size_fraction", ["0.001", "0.1", "1", "10"])
async def test_local_math_reproduces_quoter_v2_exactly(pool_address, size_fraction):
    """Same pool, same block, same size: the integer output must be identical."""
    from src.exchange.pool_state import fetch_pool_state

    client = _client()
    pool = await fetch_pool_state(client, "ethereum", pool_address)

    amount_in = int(Decimal(size_fraction) * (Decimal(10) ** pool.decimals0))
    if amount_in <= 0:
        pytest.skip("size rounds to zero at this token's decimals")

    local = pool.swap_exact_in(amount_in, zero_for_one=True)
    onchain = await client.quote_exact_input_single_raw(
        chain="ethereum",
        token_in=pool.token0,
        token_out=pool.token1,
        fee=pool.fee,
        amount_in=amount_in,
        block_number=pool.block_number,
    )

    assert local == onchain, (
        f"local {local} vs QuoterV2 {onchain} on {pool_address} at block "
        f"{pool.block_number} for {amount_in} raw in. A difference here means the "
        f"local simulator cannot be trusted to price a trade."
    )


@pytest.mark.parametrize("pool_address", POOLS[:2])
async def test_the_other_direction_matches_too(pool_address):
    """one-for-zero exercises the mirror-image branch of the swap loop, which has
    its own price-update formula and its own rounding."""
    from src.exchange.pool_state import fetch_pool_state

    client = _client()
    pool = await fetch_pool_state(client, "ethereum", pool_address)

    amount_in = int(Decimal("1") * (Decimal(10) ** pool.decimals1))
    local = pool.swap_exact_in(amount_in, zero_for_one=False)
    onchain = await client.quote_exact_input_single_raw(
        chain="ethereum", token_in=pool.token1, token_out=pool.token0,
        fee=pool.fee, amount_in=amount_in, block_number=pool.block_number,
    )

    assert local == onchain


async def test_a_swap_large_enough_to_cross_ticks_still_matches():
    """The case that separates real v3 math from a constant-product approximation.

    Sized to move the pool by more than one tick spacing, so the swap loop must
    cross at least one initialised tick and pick up the next range's liquidity.
    """
    from src.exchange.pool_state import fetch_pool_state

    client = _client()
    pool = await fetch_pool_state(
        client, "ethereum", "0x88e6A0c2dDD26FEEb64F039a2c41296FcB3f5640"
    )

    # 500 WETH-equivalent: large enough to leave the current range on any pool.
    amount_in = int(Decimal(500) * (Decimal(10) ** pool.decimals0))
    local = pool.swap_exact_in(amount_in, zero_for_one=True)
    onchain = await client.quote_exact_input_single_raw(
        chain="ethereum", token_in=pool.token0, token_out=pool.token1,
        fee=pool.fee, amount_in=amount_in, block_number=pool.block_number,
    )

    assert local == onchain, (
        f"local {local} vs QuoterV2 {onchain} on a tick-crossing swap; the "
        f"crossing logic or the liquidity_net sign is wrong"
    )


async def test_the_local_price_curve_is_monotonic_on_a_real_pool():
    """A sanity property that holds for any single pool: more size never gets a
    better price. Cheap to check and it catches a sign error in the price update
    that the exact-match tests could in principle miss at one size.
    """
    from src.exchange.pool_state import fetch_pool_state

    client = _client()
    pool = await fetch_pool_state(
        client, "ethereum", "0x88e6A0c2dDD26FEEb64F039a2c41296FcB3f5640"
    )

    sizes = [Decimal("0.01"), Decimal("0.1"), Decimal(1), Decimal(10), Decimal(100)]
    curve = [p for _, p in pool.price_curve(sizes, zero_for_one=True) if p is not None]

    assert len(curve) >= 3
    assert curve == sorted(curve, reverse=True)
