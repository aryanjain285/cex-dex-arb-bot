"""Reading tick data, batched where possible and one-by-one where not.

Split out of `pool_state` so the two paths sit next to each other and can be read
as the same operation twice. They must return identical results -- batching is a
performance change and nothing else -- and a differential test pins that.

The encode/decode calls go through the contract and codec objects rather than
`eth_abi` directly, so a test can substitute them and exercise the batching logic
without reimplementing ABI encoding.
"""
from __future__ import annotations

import asyncio
from typing import Dict, List, Optional, Sequence, Tuple

from loguru import logger

from .multicall import Multicall

__all__ = ["read_bitmaps", "read_liquidity_nets"]

# ticks() return signature. Only liquidityNet (index 1) is used, but the whole tuple
# has to be decoded to reach it.
_TICKS_OUTPUTS = [
    "uint128", "int128", "uint256", "uint256", "int56", "uint160", "uint32", "bool",
]


def _multicall_for(client) -> Multicall:
    """One Multicall per client, so the per-chain deployment check is cached.

    Attached to the client rather than created per read: the check is a constant, and
    repeating it once per pool read would spend exactly the budget batching saves.
    """
    existing = getattr(client, "_multicall", None)
    if existing is None:
        existing = Multicall(client)
        try:
            setattr(client, "_multicall", existing)
        except AttributeError:  # pragma: no cover - defensive for frozen clients
            pass
    return existing


def _encode(contract, fn_name: str, args: Sequence):
    """Calldata for a contract call, across web3 API generations."""
    encoder = getattr(contract, "encode_abi", None)
    if encoder is not None:
        try:
            return encoder(abi_element_identifier=fn_name, args=list(args))
        except TypeError:
            return encoder(fn_name, list(args))
    return contract.encodeABI(fn_name=fn_name, args=list(args))


async def read_bitmaps(
    client,
    chain: str,
    pool,
    w3,
    words: Sequence[int],
    block_number: Optional[int],
) -> Dict[int, int]:
    """{word_position: bitmap}. Batched when Multicall3 is deployed."""
    words = list(words)
    if not words:
        return {}

    multicall = _multicall_for(client)
    if await multicall.available(chain):
        calls = [(pool.address, _encode(pool, "tickBitmap", [word])) for word in words]
        raw = await multicall.aggregate(chain, calls, block_number=block_number)
        out = {}
        for word, data in zip(words, raw):
            if data is None:
                # A reverting tickBitmap should not happen; treating it as zero
                # would silently narrow the tick list, so it is logged and skipped
                # rather than assumed empty.
                logger.debug(f"{chain} {pool.address}: tickBitmap({word}) reverted")
                continue
            out[word] = int(w3.codec.decode(["uint256"], data)[0])
        return out

    results = await asyncio.gather(*(
        client._rpc(chain, _pinned(pool.functions.tickBitmap(word), block_number))
        for word in words
    ))
    return {word: int(value) for word, value in zip(words, results)}


async def read_liquidity_nets(
    client,
    chain: str,
    pool,
    w3,
    ticks: Sequence[int],
    block_number: Optional[int],
) -> List[Tuple[int, int]]:
    """[(tick, liquidity_net)] in the order given, excluding zero-net ticks.

    Order is preserved because the caller pairs these with the tick list it derived
    the window from. A reordering would pair each tick with another tick's liquidity,
    which prices small swaps correctly and every large swap wrongly.
    """
    ticks = list(ticks)
    if not ticks:
        return []

    multicall = _multicall_for(client)
    if await multicall.available(chain):
        calls = [(pool.address, _encode(pool, "ticks", [tick])) for tick in ticks]
        raw = await multicall.aggregate(chain, calls, block_number=block_number)
        out = []
        for tick, data in zip(ticks, raw):
            if data is None:
                continue
            decoded = w3.codec.decode(_TICKS_OUTPUTS, data)
            net = int(decoded[1])
            if net != 0:
                out.append((tick, net))
        return out

    results = await asyncio.gather(*(
        client._rpc(chain, _pinned(pool.functions.ticks(tick), block_number))
        for tick in ticks
    ))
    return [
        (tick, int(data[1]))
        for tick, data in zip(ticks, results)
        if int(data[1]) != 0
    ]


def _pinned(fn, block_number: Optional[int]):
    if block_number is None:
        return fn.call
    return lambda: fn.call(block_identifier=block_number)
