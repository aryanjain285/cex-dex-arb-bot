"""Does the batched read agree with the chain, and how much does it save?

Batching is only worth its risk if the saving is real, and only safe if the result
is identical. Both are measured here against live pools, and the identity check runs
at the same PINNED BLOCK so a price move between the two reads cannot be mistaken
for a batching bug.

The comparison that matters is the tick list. A positional error leaves small swaps
exact -- they never leave the current range -- and makes every large swap wrong, so
comparing a single quote would not catch it.
"""
import asyncio
import time
from decimal import Decimal

from research_config import research_config

from src.exchange.multicall import Multicall
from src.exchange.pool_state import fetch_pool_state
from src.exchange.univ3 import UniV3DexClient

config = research_config("WARNING")

POOLS = [
    ("ethereum", "0x88e6A0c2dDD26FEEb64F039a2c41296FcB3f5640", "ETH/USDC 0.05%", 6, 18),
    ("ethereum", "0x3416cF6C708Da44DB2624D63ea0AAef7113527C6", "USDC/USDT 0.01%", 6, 6),
    ("arbitrum", "0xC6962004f452bE9203591991D15f6b388e09E8D0", "ETH/USDC 0.05% arb", 18, 6),
    ("base", "0xd0b53D9277642d899DF5C87A3966A349A798F224", "ETH/USDC 0.05% base", 18, 6),
]


class Counter:
    def __init__(self, inner):
        self._inner = inner
        self.calls = 0

    async def _rpc(self, chain, fn):
        self.calls += 1
        return await self._inner._rpc(chain, fn)

    def __getattr__(self, name):
        return getattr(self._inner, name)


async def main():
    inner = UniV3DexClient(config.dex, config.network, config.secrets, config.tokens)

    probe = Multicall(inner)
    for chain in ("ethereum", "arbitrum", "base"):
        print(f"multicall3 on {chain}: {await probe.available(chain)}")
    print()

    for chain, address, label, d0, d1 in POOLS:
        w3 = inner._get_w3(chain)
        block = await inner._rpc(chain, lambda: w3.eth.block_number)

        batched_client = Counter(inner)
        started = time.monotonic()
        try:
            batched = await fetch_pool_state(
                batched_client, chain, address, decimals0=d0, decimals1=d1,
                block_number=block,
            )
        except Exception as exc:
            print(f"{label}: batched read failed: {type(exc).__name__}: {exc}")
            continue
        batched_secs = time.monotonic() - started

        # Force the unbatched path by giving this client a Multicall that reports
        # unavailable -- the same fallback a chain without the contract would take.
        single_client = Counter(inner)
        unavailable = Multicall(single_client)
        unavailable._available = {chain: False}
        single_client._multicall = unavailable
        started = time.monotonic()
        try:
            single = await fetch_pool_state(
                single_client, chain, address, decimals0=d0, decimals1=d1,
                block_number=block,
            )
        except Exception as exc:
            print(f"{label}: unbatched read failed: {type(exc).__name__}: {exc}")
            continue
        single_secs = time.monotonic() - started

        a = [(t.tick, t.liquidity_net) for t in batched.ticks]
        b = [(t.tick, t.liquidity_net) for t in single.ticks]
        identical = (
            a == b
            and batched.liquidity == single.liquidity
            and batched.sqrt_price_x96 == single.sqrt_price_x96
            and (batched.known_lower_tick, batched.known_upper_tick)
            == (single.known_lower_tick, single.known_upper_tick)
        )

        print(f"{label}  block {block}")
        print(f"  batched   {batched_client.calls:>4} calls  {batched_secs:>6.1f}s  "
              f"{len(batched.ticks)} ticks")
        print(f"  unbatched {single_client.calls:>4} calls  {single_secs:>6.1f}s  "
              f"{len(single.ticks)} ticks")
        if single_client.calls and batched_client.calls:
            print(f"  saving    {single_client.calls / batched_client.calls:.0f}x "
                  f"fewer calls, {single_secs / max(batched_secs, 0.001):.1f}x faster")
        print(f"  IDENTICAL: {identical}")
        if not identical:
            print(f"    tick lists differ: batched {len(a)} vs unbatched {len(b)}")
            only_a = set(a) - set(b)
            only_b = set(b) - set(a)
            print(f"    only batched: {sorted(only_a)[:5]}")
            print(f"    only unbatched: {sorted(only_b)[:5]}")
        print()


asyncio.run(main())
