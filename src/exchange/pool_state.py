"""Reading a Uniswap v3 pool's state, so it can be quoted locally and recorded.

A pool snapshot is the unit of everything downstream:

  * the detector can quote it at any size without an RPC round trip;
  * the size curve `Pi(q)` becomes free, so `argmax_q Pi(q)` is answerable;
  * and it can be WRITTEN DOWN, which is what makes offline backtesting possible.
    A recorded QuoterV2 answer can only be re-read at the size it was asked about;
    a recorded pool state can be re-quoted at any size under any cost assumption
    months later.

What has to be read, and why each part matters:

    slot0        sqrtPriceX96 and the current tick -- where the price is now
    liquidity    the active liquidity in the current range
    fee          the tier, needed for the fee deduction inside the swap
    tickSpacing  which ticks can be initialised
    ticks        liquidityNet at each initialised tick, so a swap that leaves the
                 current range picks up the next range's liquidity

The tick data is the expensive part and the part naive implementations skip. Skip
it and a large swap silently prices as though the current range extended forever,
which OVERSTATES the output -- the direction that invents edge. The tick bitmap is
read around the current tick out to a configurable range, and the snapshot records
how far it looked so a quote beyond that range can be refused rather than guessed.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from decimal import Decimal
from typing import List, Optional, Sequence, Tuple

from loguru import logger

from .univ3_math import MAX_TICK, MIN_TICK, TickInfo, V3Pool

__all__ = ["PoolSnapshot", "fetch_pool_state", "POOL_ABI", "DEFAULT_TICK_RANGE"]

# How many tick-spacings either side of the current tick to read. 500 spacings on a
# 0.05% pool (spacing 10) is 5000 ticks, about a 65% price move -- far beyond any
# size this strategy trades, and the snapshot records the bound so a quote that
# would leave it can be refused instead of silently extrapolated.
DEFAULT_TICK_RANGE = 500

# Minimal ABI: only what a snapshot needs. Kept here rather than in ABI/ because it
# is a read-only subset and coupling it to the file that also describes the router
# would invite the two to drift.
POOL_ABI = [
    {"inputs": [], "name": "slot0", "outputs": [
        {"name": "sqrtPriceX96", "type": "uint160"},
        {"name": "tick", "type": "int24"},
        {"name": "observationIndex", "type": "uint16"},
        {"name": "observationCardinality", "type": "uint16"},
        {"name": "observationCardinalityNext", "type": "uint16"},
        {"name": "feeProtocol", "type": "uint8"},
        {"name": "unlocked", "type": "bool"},
    ], "stateMutability": "view", "type": "function"},
    {"inputs": [], "name": "liquidity",
     "outputs": [{"name": "", "type": "uint128"}],
     "stateMutability": "view", "type": "function"},
    {"inputs": [], "name": "fee",
     "outputs": [{"name": "", "type": "uint24"}],
     "stateMutability": "view", "type": "function"},
    {"inputs": [], "name": "tickSpacing",
     "outputs": [{"name": "", "type": "int24"}],
     "stateMutability": "view", "type": "function"},
    {"inputs": [], "name": "token0",
     "outputs": [{"name": "", "type": "address"}],
     "stateMutability": "view", "type": "function"},
    {"inputs": [], "name": "token1",
     "outputs": [{"name": "", "type": "address"}],
     "stateMutability": "view", "type": "function"},
    {"inputs": [{"name": "", "type": "int24"}], "name": "ticks", "outputs": [
        {"name": "liquidityGross", "type": "uint128"},
        {"name": "liquidityNet", "type": "int128"},
        {"name": "feeGrowthOutside0X128", "type": "uint256"},
        {"name": "feeGrowthOutside1X128", "type": "uint256"},
        {"name": "tickCumulativeOutside", "type": "int56"},
        {"name": "secondsPerLiquidityOutsideX128", "type": "uint160"},
        {"name": "secondsOutside", "type": "uint32"},
        {"name": "initialized", "type": "bool"},
    ], "stateMutability": "view", "type": "function"},
    {"inputs": [{"name": "wordPosition", "type": "int16"}], "name": "tickBitmap",
     "outputs": [{"name": "", "type": "uint256"}],
     "stateMutability": "view", "type": "function"},
]


@dataclass(frozen=True)
class PoolSnapshot(V3Pool):
    """A pool state at one block, quotable offline and serialisable.

    Extends V3Pool with the provenance a recorded observation needs: which pool,
    which block, and how far the tick scan reached. Without the last of those, a
    replayed quote at an unusual size cannot tell "the pool really is that thin"
    from "we only read 500 spacings".
    """

    token0: str = ""
    token1: str = ""
    chain: str = ""
    tick_range_scanned: int = DEFAULT_TICK_RANGE
    observed_at: float = 0.0

    def to_row(self) -> dict:
        """Flat, JSON-safe, and lossless -- integers stay integers.

        Deliberately not floats: sqrtPriceX96 is a 160-bit integer and a float
        would silently truncate it, which is a price error rather than a
        formatting one.
        """
        return {
            "chain": self.chain,
            "address": self.address,
            "block_number": self.block_number,
            "observed_at": self.observed_at,
            "token0": self.token0,
            "token1": self.token1,
            "decimals0": self.decimals0,
            "decimals1": self.decimals1,
            "fee": self.fee,
            "tick_spacing": self.tick_spacing,
            "sqrt_price_x96": str(self.sqrt_price_x96),
            "liquidity": str(self.liquidity),
            "tick": self.tick,
            "tick_range_scanned": self.tick_range_scanned,
            "ticks": [[t.tick, str(t.liquidity_net)] for t in self.ticks],
        }

    @classmethod
    def from_row(cls, row: dict) -> "PoolSnapshot":
        return cls(
            sqrt_price_x96=int(row["sqrt_price_x96"]),
            liquidity=int(row["liquidity"]),
            tick=int(row["tick"]),
            fee=int(row["fee"]),
            tick_spacing=int(row["tick_spacing"]),
            ticks=[TickInfo(tick=int(t), liquidity_net=int(n))
                   for t, n in row.get("ticks", [])],
            decimals0=int(row["decimals0"]),
            decimals1=int(row["decimals1"]),
            block_number=row.get("block_number"),
            address=row.get("address"),
            token0=row.get("token0", ""),
            token1=row.get("token1", ""),
            chain=row.get("chain", ""),
            tick_range_scanned=int(row.get("tick_range_scanned", DEFAULT_TICK_RANGE)),
            observed_at=float(row.get("observed_at", 0.0)),
        )


def _initialised_ticks_from_bitmap(
    word_position: int, word: int, tick_spacing: int
) -> List[int]:
    """Which ticks a bitmap word says are initialised.

    Each word covers 256 compressed ticks; bit i set means the tick at
    (word_position * 256 + i) * tick_spacing is initialised.
    """
    if word == 0:
        return []
    return [
        (word_position * 256 + bit) * tick_spacing
        for bit in range(256)
        if word & (1 << bit)
    ]


async def fetch_pool_state(
    client,
    chain: str,
    pool_address: str,
    decimals0: Optional[int] = None,
    decimals1: Optional[int] = None,
    tick_range: int = DEFAULT_TICK_RANGE,
    block_number: Optional[int] = None,
) -> PoolSnapshot:
    """Read one pool into a snapshot that can be quoted locally.

    Every call is paced through the client's RPC limiter. Reading a pool costs
    roughly 8 calls plus one per bitmap word, which is why this belongs on a state
    -update cadence -- once per block, or once per poll interval -- rather than in
    the quote path it replaces.

    `decimals` are read from the tokens when not supplied. They are never assumed:
    a wrong decimals value is a factor of 10^n on every price, and that specific
    error has already cost this codebase a full rewrite of the dataset loader.
    """
    from web3 import Web3

    w3 = client._get_w3(chain)
    address = Web3.to_checksum_address(pool_address)
    pool = w3.eth.contract(address=address, abi=POOL_ABI)

    # Pin the block, so every field describes the same state. Without this a
    # snapshot can straddle a block boundary and mix a new price with old ticks --
    # which would quote a pool that never existed.
    if block_number is None:
        block_number = await client._rpc(chain, lambda: w3.eth.block_number)

    def _call(fn):
        return lambda: fn.call(block_identifier=block_number)

    slot0, liquidity, fee, tick_spacing, token0, token1 = await asyncio.gather(
        client._rpc(chain, _call(pool.functions.slot0())),
        client._rpc(chain, _call(pool.functions.liquidity())),
        client._rpc(chain, _call(pool.functions.fee())),
        client._rpc(chain, _call(pool.functions.tickSpacing())),
        client._rpc(chain, _call(pool.functions.token0())),
        client._rpc(chain, _call(pool.functions.token1())),
    )
    sqrt_price_x96, current_tick = int(slot0[0]), int(slot0[1])

    if decimals0 is None or decimals1 is None:
        erc20 = client.erc20_abi
        d0, d1 = await asyncio.gather(
            client._rpc(chain, w3.eth.contract(
                address=Web3.to_checksum_address(token0), abi=erc20
            ).functions.decimals().call),
            client._rpc(chain, w3.eth.contract(
                address=Web3.to_checksum_address(token1), abi=erc20
            ).functions.decimals().call),
        )
        decimals0 = int(d0) if decimals0 is None else decimals0
        decimals1 = int(d1) if decimals1 is None else decimals1

    # Which bitmap words cover the scanned range.
    compressed = current_tick // tick_spacing
    lowest = max(MIN_TICK // tick_spacing, compressed - tick_range)
    highest = min(MAX_TICK // tick_spacing, compressed + tick_range)
    words = range(lowest >> 8, (highest >> 8) + 1)

    bitmaps = await asyncio.gather(*(
        client._rpc(chain, _call(pool.functions.tickBitmap(word)))
        for word in words
    ))

    candidate_ticks: List[int] = []
    for word_position, word in zip(words, bitmaps):
        candidate_ticks.extend(
            _initialised_ticks_from_bitmap(word_position, int(word), tick_spacing)
        )
    # Keep only what the scan actually covers, so the snapshot's stated range is true.
    lower_bound, upper_bound = lowest * tick_spacing, highest * tick_spacing
    candidate_ticks = sorted(
        t for t in candidate_ticks if lower_bound <= t <= upper_bound
    )

    liquidity_nets = await asyncio.gather(*(
        client._rpc(chain, _call(pool.functions.ticks(tick)))
        for tick in candidate_ticks
    ))
    ticks = [
        TickInfo(tick=tick, liquidity_net=int(data[1]))
        for tick, data in zip(candidate_ticks, liquidity_nets)
        if int(data[1]) != 0
    ]

    from ..core import clock

    logger.debug(
        f"{chain} pool {address} at block {block_number}: {len(ticks)} initialised "
        f"ticks across +/-{tick_range} spacings"
    )
    return PoolSnapshot(
        sqrt_price_x96=sqrt_price_x96,
        liquidity=int(liquidity),
        tick=current_tick,
        fee=int(fee),
        tick_spacing=int(tick_spacing),
        ticks=ticks,
        decimals0=decimals0,
        decimals1=decimals1,
        block_number=int(block_number),
        address=address,
        token0=Web3.to_checksum_address(token0),
        token1=Web3.to_checksum_address(token1),
        chain=chain,
        tick_range_scanned=tick_range,
        observed_at=clock.now(),
    )
