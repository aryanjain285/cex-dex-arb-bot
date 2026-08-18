"""Swap-print dislocation with EXACT block timestamps. No interpolation anywhere.

Two previous attempts at this were contaminated by timing, each time in a way that
produced a plausible finding:

  Interpolating between five anchors across fourteen months was wrong by a median of
  51 minutes, which manufactured a +0.703 correlation between volatility and dislocation
  and made 25 of 29 windows appear to clear the cost floor. All of it was the error.

  Interpolating between two exact endpoints three hours apart was wrong by a median of 16
  seconds -- much better, and still enough. The residual was SIGNED: interpolated times ran
  late by 3 to 32 seconds throughout, so every swap was compared against a kline from after
  it happened. With ETH drifting up over the window that biases the dislocation negative,
  and both direction buckets duly came back negative (-3.14 and -13.53 bps), which is what
  a systematic offset looks like rather than a mechanism.

So this version fetches the real timestamp of every block that contains a swap. That costs
one call per block, which is why the window is short -- a bounded, honest measurement of a
small period beats an unbounded biased one.

The question it answers is the one that matters for architecture. Clock-sampled recording
of the same pool reports a 2.3 bps median dislocation; swap prints reported 8.67. If the
difference survives exact timing, then the dislocation lives in the interval between a
trade and whoever closes it -- which a 2-second poller mostly misses and a backrunner
competes for -- and the two figures answer different questions. If it does not survive,
the swap-print figure was timing all along.
"""
import argparse
import asyncio
import statistics
from collections import defaultdict
from datetime import datetime, timezone
from decimal import Decimal

import aiohttp
from research_config import research_config
from web3 import Web3

config = research_config("WARNING")

SWAP_ABI = [{
    "anonymous": False,
    "inputs": [
        {"indexed": True, "name": "sender", "type": "address"},
        {"indexed": True, "name": "recipient", "type": "address"},
        {"indexed": False, "name": "amount0", "type": "int256"},
        {"indexed": False, "name": "amount1", "type": "int256"},
        {"indexed": False, "name": "sqrtPriceX96", "type": "uint160"},
        {"indexed": False, "name": "liquidity", "type": "uint128"},
        {"indexed": False, "name": "tick", "type": "int24"},
    ],
    "name": "Swap",
    "type": "event",
}]

KLINES = "https://api.binance.com/api/v3/klines"


async def fetch_klines(session, symbol, start_ms, end_ms):
    out, cursor = [], start_ms
    while cursor < end_ms:
        params = {"symbol": symbol, "interval": "1s", "startTime": cursor,
                  "endTime": end_ms, "limit": 1000}
        async with session.get(KLINES, params=params) as response:
            if response.status == 429:
                await asyncio.sleep(5)
                continue
            if response.status == 400:
                # 1s klines are not offered for every symbol; fall back to 1m.
                params["interval"] = "1m"
                async with session.get(KLINES, params=params) as retry:
                    retry.raise_for_status()
                    batch = await retry.json()
            else:
                response.raise_for_status()
                batch = await response.json()
        if not batch:
            break
        out.extend(batch)
        if batch[-1][0] <= cursor:
            break
        cursor = batch[-1][0] + 1
        await asyncio.sleep(0.1)
    return out


def price_from_sqrt(sqrt_price_x96, decimals0, decimals1):
    raw = (Decimal(sqrt_price_x96) / Decimal(2 ** 96)) ** 2
    return raw * (Decimal(10) ** decimals0) / (Decimal(10) ** decimals1)


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool", default="0xC6962004f452bE9203591991D15f6b388e09E8D0")
    parser.add_argument("--chain", default="arbitrum")
    parser.add_argument("--symbol", default="ETHUSDT")
    parser.add_argument("--decimals0", type=int, default=18)
    parser.add_argument("--decimals1", type=int, default=6)
    parser.add_argument("--minutes", type=float, default=25.0)
    parser.add_argument("--fee", type=int, default=500)
    parser.add_argument("--taker-bps", type=float, default=7.5)
    parser.add_argument("--max-blocks", type=int, default=900,
                        help="cap on block timestamp lookups")
    args = parser.parse_args()

    from src.exchange.univ3 import UniV3DexClient
    client = UniV3DexClient(config.dex, config.network, config.secrets, config.tokens)
    w3 = client._get_w3(args.chain)
    head = await client._rpc(args.chain, lambda: w3.eth.block_number)
    head_ts = int((await client._rpc(
        args.chain, lambda: w3.eth.get_block(int(head))
    ))["timestamp"])

    # Blocks are about 0.25s on Arbitrum; the exact span is read back below, so this only
    # has to be roughly right.
    start_block = max(1, head - int(args.minutes * 60 / 0.25))
    start_ts = int((await client._rpc(
        args.chain, lambda: w3.eth.get_block(int(start_block))
    ))["timestamp"])
    print(f"blocks {start_block:,}-{head:,} covering "
          f"{datetime.fromtimestamp(start_ts, timezone.utc):%H:%M:%S} to "
          f"{datetime.fromtimestamp(head_ts, timezone.utc):%H:%M:%S} UTC "
          f"({(head_ts - start_ts) / 60:.1f} min)")

    pool = w3.eth.contract(
        address=Web3.to_checksum_address(args.pool), abi=SWAP_ABI
    )
    events = []
    chunk = max(1, (head - start_block) // 4)
    for begin in range(start_block, head, chunk):
        stop = min(head, begin + chunk)
        try:
            events.extend(await client._rpc(
                args.chain,
                lambda a=begin, b=stop: pool.events.Swap().get_logs(
                    from_block=a, to_block=b
                ),
            ))
        except Exception as exc:
            print(f"  {begin:,}-{stop:,}: {type(exc).__name__}: {str(exc)[:60]}")
    print(f"{len(events):,} swaps")
    if not events:
        return

    blocks = sorted({int(e["blockNumber"]) for e in events})
    if len(blocks) > args.max_blocks:
        # Keep the most recent, so the sample stays contiguous rather than scattered --
        # a scattered sample would need the klines for its whole span anyway.
        blocks = blocks[-args.max_blocks:]
        keep = set(blocks)
        events = [e for e in events if int(e["blockNumber"]) in keep]
        print(f"  capped to {len(blocks):,} blocks / {len(events):,} swaps")

    print(f"fetching exact timestamps for {len(blocks):,} blocks...")
    timestamps = {}
    for i, block in enumerate(blocks, 1):
        timestamps[block] = int((await client._rpc(
            args.chain, lambda b=block: w3.eth.get_block(int(b))
        ))["timestamp"])
        if i % 200 == 0:
            print(f"  {i}/{len(blocks)}")

    span_from = min(timestamps.values())
    span_to = max(timestamps.values())
    print(f"exact span {datetime.fromtimestamp(span_from, timezone.utc):%H:%M:%S} to "
          f"{datetime.fromtimestamp(span_to, timezone.utc):%H:%M:%S} UTC")

    async with aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(total=90)
    ) as session:
        klines = await fetch_klines(
            session, args.symbol, (span_from - 120) * 1000, (span_to + 120) * 1000
        )
    if not klines:
        print("no klines")
        return
    # Interval detected from the data rather than assumed, since 1s is not offered for
    # every symbol and silently falling back to 1m would change what "exact" means.
    interval_s = 1 if len(klines) > 1 and (klines[1][0] - klines[0][0]) == 1000 else 60
    closes = {}
    for kline in klines:
        closes[kline[0] // (interval_s * 1000)] = float(kline[4])
    print(f"{len(klines):,} klines at {interval_s}s resolution")

    rows = []
    for event in events:
        block = int(event["blockNumber"])
        if block not in timestamps:
            continue
        exact_ts = timestamps[block]
        cex = closes.get(exact_ts // interval_s)
        if cex is None:
            continue
        pool_price = price_from_sqrt(
            int(event["args"]["sqrtPriceX96"]), args.decimals0, args.decimals1
        )
        if pool_price <= 0 or cex <= 0:
            continue
        dislocation = float((pool_price / Decimal(str(cex)) - 1) * 10_000)
        amount0 = int(event["args"]["amount0"])
        notional = abs(int(event["args"]["amount1"])) / (10 ** args.decimals1)
        rows.append((notional, dislocation, amount0 > 0))

    if len(rows) < 30:
        print(f"only {len(rows)} usable swaps")
        return

    magnitudes = sorted(abs(r[1]) for r in rows)
    signed = [r[1] for r in rows]
    floor = args.fee / 100.0 + args.taker_bps

    print()
    print("=" * 78)
    print(f"EXACT-TIMESTAMP SWAP DISLOCATION  ({len(rows):,} swaps, "
          f"{interval_s}s CEX resolution)")
    print("=" * 78)
    print(f"  signed    mean {statistics.fmean(signed):+.2f}  "
          f"median {statistics.median(signed):+.2f} bps")
    print(f"  |size|    median {statistics.median(magnitudes):.2f}  "
          f"p90 {magnitudes[int(len(magnitudes) * 0.9)]:.2f}  "
          f"p99 {magnitudes[int(len(magnitudes) * 0.99)]:.2f}  "
          f"max {magnitudes[-1]:.2f} bps")
    print(f"  floor     {floor:.1f} bps")
    above = sum(1 for m in magnitudes if m > floor) / len(magnitudes)
    print(f"  fraction of swap prints above the floor: {above:.2%}")

    down = [r[1] for r in rows if r[2]]
    up = [r[1] for r in rows if not r[2]]
    print()
    if down and up:
        md, mu = statistics.median(down), statistics.median(up)
        print(f"  direction: pushed DOWN ({len(down):,}) median {md:+.2f} bps, "
              f"pushed UP ({len(up):,}) median {mu:+.2f} bps")
        print(f"             separation {mu - md:+.2f} bps")
        if abs(mu - md) > 2:
            print("  The print leans the way the last trade pushed, so this dislocation")
            print("  exists between a swap and whoever closes it. A clock-sampled reader")
            print("  mostly sees the closed state; a backrunner competes for the open one.")
        else:
            print("  The print does not lean with the trade direction, so it is not the")
            print("  swap's own displacement.")

    print()
    print("Compare against the clock-sampled recorder on this same pool. If the two")
    print("differ materially with timing now exact, they are measuring different things:")
    print("the pool at arbitrary instants, versus the pool immediately after trades.")
    print("Only the first is available to a polling loop.")


asyncio.run(main())
