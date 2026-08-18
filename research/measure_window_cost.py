"""How wide should the observed window be? Measure, do not guess.

Making the window price-relative made coverage COMPARABLE across fee tiers, which
was the point, but it also revealed that the level was set by accident. The old
default gave +/-232% on a spacing-200 pool and +/-0.6% on a spacing-1 one; picking
+/-25% for both improved the latter and degraded the former.

The metric that decides it is not ticks or seconds -- it is MAX PRICEABLE NOTIONAL:
the largest trade the snapshot can quote without refusing. That is the ceiling on
what any size curve can answer, so it is the ceiling on what the research can
conclude.

Cost matters too, but asymmetrically: a full read happens once per pool, then cheap
refreshes carry the recording. So the question is what a wider window costs at
STARTUP, against how much of the size grid it unlocks.
"""
import asyncio
import time
from decimal import Decimal

from research_config import research_config

from src.exchange.pool_state import fetch_pool_state
from src.exchange.univ3 import UniV3DexClient

config = research_config("WARNING")

POOLS = [
    ("ethereum", "0x88e6A0c2dDD26FEEb64F039a2c41296FcB3f5640", "USDC/WETH 0.05%", 6, 18),
    ("arbitrum", "0x97bca422Ec0Ee4851F2110eA743C1cd0a14835a1", "ARB/USDT 0.30%", 18, 6),
    ("arbitrum", "0x80151aAe63b24A7e1837Fe578FB6bE026ae8AbBA", "ARB/USDT 1.00%", 18, 6),
]
WINDOWS = [Decimal("0.10"), Decimal("0.25"), Decimal("0.50"), Decimal("0.90")]


def max_priceable_base(snapshot, zero_for_one):
    """Largest power-of-ten base amount the snapshot will quote without refusing."""
    best = None
    amount = Decimal("0.000001")
    for _ in range(20):
        if snapshot.price_for_amount_in(amount, zero_for_one=zero_for_one) is not None:
            best = amount
        amount *= 10
    return best


class CountingClient:
    """Wraps the real client to count RPC calls, since that is the actual cost."""

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

    print(f"{'pool':<20} {'window':>7} {'ticks':>6} {'calls':>6} {'secs':>6} "
          f"{'max base priceable':>20} {'spot':>14}")
    for chain, address, label, d0, d1 in POOLS:
        for window in WINDOWS:
            client = CountingClient(inner)
            started = time.monotonic()
            try:
                snap = await fetch_pool_state(
                    client, chain, address, decimals0=d0, decimals1=d1,
                    price_window=window,
                )
            except Exception as exc:
                print(f"{label:<20} {float(window):>7.2f}  {type(exc).__name__}: {str(exc)[:40]}")
                continue
            elapsed = time.monotonic() - started
            largest = max_priceable_base(snap, zero_for_one=True)
            print(f"{label:<20} {float(window):>7.2f} {len(snap.ticks):>6} "
                  f"{client.calls:>6} {elapsed:>6.1f} "
                  f"{(str(largest) if largest else 'none'):>20} "
                  f"{float(snap.spot_price()):>14.6f}")
        print()


asyncio.run(main())
