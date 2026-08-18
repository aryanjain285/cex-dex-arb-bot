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

from ._tick_reader import read_bitmaps, read_liquidity_nets
from .univ3_math import MAX_TICK, MIN_TICK, TickInfo, V3Pool

__all__ = [
    "PoolSnapshot", "fetch_pool_state", "ChainPoolReader", "POOL_ABI",
    "DEFAULT_TICK_RANGE", "DEFAULT_PRICE_WINDOW", "DEFAULT_MAX_TICKS",
    "spacings_for_price_window", "clamp_ticks_to_budget",
]

# How many tick-spacings either side of the current tick to read.
#
# Every spacing costs an RPC call when initialised, so this is the main cost knob:
# 500 spacings made a single pool read take 10-25 seconds on a public endpoint and
# drew 429s. 60 spacings on a 0.05% pool (spacing 10) is 600 ticks, about a 6%
# price move -- comfortably beyond any size this strategy trades, while costing an
# order of magnitude less.
#
# The snapshot records the bound it actually scanned, and `SwapResult` reports when
# a quote would leave it, so too small a range produces a REFUSAL rather than a
# silently extrapolated price.
DEFAULT_TICK_RANGE = 60

# The above is in SPACINGS, and spacing is a property of the fee tier -- so one
# number produced windows that differed by 200x across the pools being compared:
#
#     fee     spacing   +/-60 spacings   in price
#     0.01%         1          +/-60       +/-0.6%
#     0.05%        10         +/-600       +/-6.2%
#     0.30%        60       +/-3,600      +/-43.3%
#     1.00%       200      +/-12,000    +/-232/-70%
#
# Measured on ARB/USDT: the 0.01% pool could not price a ONE-token swap, while the
# 1.00% pool's window exceeded any price the token will ever have. Since the window
# determines the maximum priceable size, any statistic computed across tiers was
# weighting them by their spacing. So the window is stated as a fraction of price,
# which means the same thing everywhere, and spacings are derived from it.
#
# Set from measurement, not taste. Full reads of three pools at four widths,
# 2026-08-18, counting RPC calls and the largest quotable size:
#
#   pool               window  ticks  calls   max priceable base
#   USDC/WETH 0.05%      any     160    ~170   10,000,000  (budget binds first)
#   ARB/USDT  0.30%     0.10      13      21        1,000
#   ARB/USDT  0.30%     0.25      25      33       10,000
#   ARB/USDT  0.30%     0.90      53      62       10,000  (pool runs dry, not window)
#   ARB/USDT  1.00%     0.25       2      10          100
#   ARB/USDT  1.00%     0.90       5      14        1,000
#
# The useful asymmetry: a wide window is CHEAP exactly where it helps. Thin pools
# have few initialised ticks, so widening costs a handful of calls and unlocks an
# order of magnitude of quotable size. Deep pools hit the tick budget regardless,
# and do not need width because their price impact is small. So take the wide end.
DEFAULT_PRICE_WINDOW = Decimal("0.90")

# One RPC call per initialised tick, so the tick count is the real budget. On a
# spacing-1 pool a +/-25% window can contain thousands. When the budget truncates,
# the CLAIMED window shrinks with it -- see `clamp_ticks_to_budget`. An over-claimed
# window is exactly as wrong as extrapolating past the observed range, because the
# swap math trusts the claim literally.
DEFAULT_MAX_TICKS = 160

# ln(1.0001). Ticks are 1.0001^tick, so a price fraction converts to ticks by
# dividing its log by this. Computed once rather than per call.
_LOG_TICK_BASE = Decimal("1.0001").ln()


def spacings_for_price_window(price_fraction: Decimal, tick_spacing: int) -> int:
    """How many tick SPACINGS cover +/-`price_fraction` of the current price.

    Rounds up, so the window is never narrower than requested, and returns at
    least 1: a window that rounded to zero spacings would record no ticks and no
    bounds, leaving the pool unquotable at every size.
    """
    if price_fraction <= 0:
        raise ValueError(
            f"price_fraction must be positive, got {price_fraction}; a zero or "
            f"negative window would record no liquidity and make the pool "
            f"unquotable rather than cheap to read"
        )
    if tick_spacing < 1:
        raise ValueError(f"tick_spacing must be at least 1, got {tick_spacing}")
    ticks_needed = (Decimal(1) + price_fraction).ln() / _LOG_TICK_BASE
    # Explicit ceil. Decimal's // truncates toward zero rather than flooring, so
    # the usual -(-a // b) idiom silently yields a FLOOR here -- which rounds the
    # window down and reintroduces exactly the under-coverage this function exists
    # to prevent.
    spacings = int(ticks_needed // tick_spacing)
    if spacings * tick_spacing < ticks_needed:
        spacings += 1
    return max(1, spacings)


def clamp_ticks_to_budget(
    ticks: Sequence[int],
    *,
    current_tick: int,
    lower_bound: int,
    upper_bound: int,
    max_ticks: int,
) -> Tuple[List[int], int, int]:
    """Keep the `max_ticks` ticks nearest the price, and shrink the claimed window.

    Returns (kept_ticks, claimed_lower, claimed_upper).

    The nearest ticks are kept because they are the ones a swap crosses first:
    spending the budget on far ticks would leave a hole next to the price, and a
    hole is worse than a narrow window -- the swap math would cross straight
    through it as though the liquidity were flat.

    The claimed window then stops short of the nearest DROPPED tick on each side.
    This is the invariant the swap math depends on: inside the claimed window,
    every initialised tick is known. A window that still spanned a dropped tick
    would make the simulator extrapolate through a real liquidity change, which is
    the same silent-optimism defect as running past the scan edge.
    """
    if max_ticks < 0:
        raise ValueError(f"max_ticks must be non-negative, got {max_ticks}")
    ordered = sorted(ticks)
    if len(ordered) <= max_ticks:
        return list(ordered), lower_bound, upper_bound

    kept = set(sorted(ordered, key=lambda t: abs(t - current_tick))[:max_ticks])
    dropped = [t for t in ordered if t not in kept]

    # Stop short of the nearest dropped tick on each side. "Short of" and not "at":
    # the tick itself changes liquidity, so the last knowable price is just inside.
    below = [t for t in dropped if t <= current_tick]
    above = [t for t in dropped if t > current_tick]
    claimed_lower = max(lower_bound, max(below) + 1) if below else lower_bound
    claimed_upper = min(upper_bound, min(above) - 1) if above else upper_bound
    return sorted(kept), claimed_lower, claimed_upper

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
            # Persisted so a replayed observation prices identically to the live
            # one that recorded it. Without these a backtest would measure its own
            # serialisation rather than the market.
            "known_lower_tick": self.known_lower_tick,
            "known_upper_tick": self.known_upper_tick,
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
            # Additive: rows recorded before the window existed load as None and
            # fall back to the outermost recorded tick, which under-claims
            # knowledge rather than over-claiming it.
            known_lower_tick=(
                int(row["known_lower_tick"])
                if row.get("known_lower_tick") is not None else None
            ),
            known_upper_tick=(
                int(row["known_upper_tick"])
                if row.get("known_upper_tick") is not None else None
            ),
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
    tick_range: Optional[int] = None,
    block_number: Optional[int] = None,
    price_window: Optional[Decimal] = None,
    max_ticks: int = DEFAULT_MAX_TICKS,
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

    # How wide a window to scan. Stated as a fraction of price so it means the same
    # thing on every fee tier; `tick_range` remains available as an explicit
    # override in spacings, for callers pinning a cost rather than a coverage.
    if tick_range is None:
        window = DEFAULT_PRICE_WINDOW if price_window is None else price_window
        tick_range = spacings_for_price_window(window, int(tick_spacing))
    elif price_window is not None:
        raise ValueError(
            "pass tick_range or price_window, not both: they are two ways of saying "
            "the same thing and disagreeing about it silently changes what the "
            "snapshot can price"
        )

    # Which bitmap words cover the scanned range.
    compressed = current_tick // tick_spacing
    lowest = max(MIN_TICK // tick_spacing, compressed - tick_range)
    highest = min(MAX_TICK // tick_spacing, compressed + tick_range)
    words = range(lowest >> 8, (highest >> 8) + 1)

    # Batched through Multicall3 where it is deployed, one call per word where it is
    # not. A full read is otherwise ~200 round trips, measured at 60-160s against
    # public endpoints, which makes recording more than a handful of pools impossible.
    bitmaps = await read_bitmaps(client, chain, pool, w3, words, block_number)

    candidate_ticks: List[int] = []
    for word_position, word in bitmaps.items():
        candidate_ticks.extend(
            _initialised_ticks_from_bitmap(word_position, int(word), tick_spacing)
        )
    # Keep only what the scan actually covers, so the snapshot's stated range is true.
    lower_bound, upper_bound = lowest * tick_spacing, highest * tick_spacing
    candidate_ticks = sorted(
        t for t in candidate_ticks if lower_bound <= t <= upper_bound
    )

    # One RPC call per initialised tick, so a dense spacing-1 pool can blow the
    # budget. Keep the ticks nearest the price and SHRINK the claimed window to
    # match -- a window that still spanned a dropped tick would make the swap math
    # extrapolate straight through a real liquidity change.
    candidate_ticks, lower_bound, upper_bound = clamp_ticks_to_budget(
        candidate_ticks,
        current_tick=current_tick,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        max_ticks=max_ticks,
    )

    ticks = [
        TickInfo(tick=tick, liquidity_net=net)
        for tick, net in await read_liquidity_nets(
            client, chain, pool, w3, candidate_ticks, block_number
        )
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
        # The window whose liquidity was actually observed. Already computed above
        # to filter tick candidates; passing it on is what lets the swap math say
        # "unknown" instead of extrapolating the last observed range to infinity.
        known_lower_tick=lower_bound,
        known_upper_tick=upper_bound,
    )


class ChainPoolReader:
    """The `PoolStateCache` reader interface, backed by a real chain.

    Two methods, deliberately asymmetric in cost:

        read_full   ~8 calls plus one per initialised tick   (50-200 calls)
        read_slot0  2 calls                                  (the hot path)

    Everything goes through the client's RPC limiter, so the cheap path is what
    makes recording many pools at a useful cadence affordable at all.
    """

    def __init__(
        self,
        client,
        tick_range: Optional[int] = None,
        price_window: Optional[Decimal] = None,
        max_ticks: int = DEFAULT_MAX_TICKS,
    ):
        """`price_window` is the coverage to aim for, as a fraction of price.

        Both default to None, which lets `fetch_pool_state` apply
        DEFAULT_PRICE_WINDOW. Passing `tick_range` pins the cost in spacings
        instead, which is the older behaviour and means different coverage on each
        fee tier -- correct only when comparing a pool against itself.
        """
        if tick_range is not None and price_window is not None:
            raise ValueError(
                "pass tick_range or price_window, not both: coverage and cost are "
                "two views of one number and a disagreement between them silently "
                "changes what every snapshot can price"
            )
        self.client = client
        self.tick_range = tick_range
        self.price_window = price_window
        self.max_ticks = max_ticks

    async def read_full(self, chain: str, address: str, **kwargs) -> PoolSnapshot:
        if self.tick_range is not None:
            kwargs.setdefault("tick_range", self.tick_range)
        if self.price_window is not None:
            kwargs.setdefault("price_window", self.price_window)
        kwargs.setdefault("max_ticks", self.max_ticks)
        return await fetch_pool_state(self.client, chain, address, **kwargs)

    async def read_slot0(self, chain: str, address: str):
        """(sqrtPriceX96, tick, liquidity, block) -- the fields a swap moves."""
        from web3 import Web3

        w3 = self.client._get_w3(chain)
        pool = w3.eth.contract(
            address=Web3.to_checksum_address(address), abi=POOL_ABI
        )
        # One pinned block for both reads: an unpinned pair can straddle a block
        # boundary and mix a new price with old liquidity, which is a state the
        # pool never had.
        block = await self.client._rpc(chain, lambda: w3.eth.block_number)

        slot0, liquidity = await asyncio.gather(
            self.client._rpc(
                chain, lambda: pool.functions.slot0().call(block_identifier=block)
            ),
            self.client._rpc(
                chain,
                lambda: pool.functions.liquidity().call(block_identifier=block),
            ),
        )
        return int(slot0[0]), int(slot0[1]), int(liquidity), int(block)
