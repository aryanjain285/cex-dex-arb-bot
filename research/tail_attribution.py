"""Is the tail a market dislocation, or the swap's own price impact?

The corrected historical reconstruction found the MEDIAN dislocation stable at 4-12 bps
across every volatility regime -- never clearing a 12.5 bps floor -- while the TAIL scales
hard with volatility: p90 rises from 11 bps in a quiet hour to 105 bps at 38.92 bps/min,
and single swaps reach 232 bps.

Read naively, that says the opportunity is in the tail and only opens in violent markets.
Before believing it, there is an alternative explanation that would produce exactly the
same numbers and mean the opposite.

A Swap event records `sqrtPriceX96` AFTER the swap. So a large market order that pushes the
pool away from the exchange leaves a print showing a large "dislocation" -- which is that
order's OWN price impact, not a gap anyone else could have traded. And large orders cluster
in volatile hours, so this alternative also predicts a tail that scales with volatility.

The two are distinguishable. If the tail is the swap's own impact, |dislocation| correlates
with that swap's SIZE. If it is a market-wide gap, it does not.

It matters because the two require completely different systems. A market-wide gap can be
taken by anyone who sees it within its lifetime. A displacement created by the swap in front
of you can only be taken by the next transaction in the block -- which means competing with
backrunners on priority fees, not polling every 2.3 seconds.
"""
import argparse
import asyncio
import json
import statistics
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path

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
        params = {"symbol": symbol, "interval": "1m", "startTime": cursor,
                  "endTime": end_ms, "limit": 1000}
        async with session.get(KLINES, params=params) as response:
            if response.status == 429:
                await asyncio.sleep(10)
                continue
            response.raise_for_status()
            batch = await response.json()
        if not batch:
            break
        out.extend(batch)
        if batch[-1][0] <= cursor:
            break
        cursor = batch[-1][0] + 1
        await asyncio.sleep(0.12)
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
    parser.add_argument("--hours-back", type=float, default=3.0)
    parser.add_argument("--fee", type=int, default=500)
    parser.add_argument("--taker-bps", type=float, default=7.5)
    args = parser.parse_args()

    from src.exchange.univ3 import UniV3DexClient
    client = UniV3DexClient(config.dex, config.network, config.secrets, config.tokens)
    w3 = client._get_w3(args.chain)
    head = await client._rpc(args.chain, lambda: w3.eth.block_number)
    head_block = await client._rpc(args.chain, lambda: w3.eth.get_block(int(head)))
    head_ts = int(head_block["timestamp"])

    # Recent window only, so block timing needs no search: read both endpoints directly.
    from_ts = head_ts - int(args.hours_back * 3600)
    # Arbitrum blocks are about 0.25s, refined against the endpoint actually fetched.
    guess = max(1, head - int(args.hours_back * 3600 / 0.25))
    guess_block = await client._rpc(args.chain, lambda: w3.eth.get_block(int(guess)))
    guess_ts = int(guess_block["timestamp"])
    print(f"head {head:,} at {datetime.fromtimestamp(head_ts, timezone.utc):%H:%M:%S}, "
          f"start block {guess:,} at "
          f"{datetime.fromtimestamp(guess_ts, timezone.utc):%H:%M:%S} "
          f"({(head_ts - guess_ts) / 3600:.2f}h of history)")

    async with aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(total=90)
    ) as session:
        klines = await fetch_klines(
            session, args.symbol, (guess_ts - 120) * 1000, (head_ts + 60) * 1000
        )
    minute_close = {}
    for kline in klines:
        minute_close[kline[0] // 60_000] = float(kline[4])
    print(f"{len(klines)} klines")

    pool = w3.eth.contract(
        address=Web3.to_checksum_address(args.pool), abi=SWAP_ABI
    )
    # Chunked, because a query returning over 10,000 logs is rejected outright.
    span = head - guess
    chunk = max(1, span // 8)
    events = []
    for start in range(guess, head, chunk):
        stop = min(head, start + chunk)
        try:
            events.extend(await client._rpc(
                args.chain,
                lambda a=start, b=stop: pool.events.Swap().get_logs(
                    from_block=a, to_block=b
                ),
            ))
        except Exception as exc:
            print(f"  chunk {start:,}-{stop:,}: {type(exc).__name__}: {str(exc)[:60]}")
    print(f"{len(events):,} swaps")
    if not events:
        return

    rows = []
    for event in events:
        sqrt_price = int(event["args"]["sqrtPriceX96"])
        amount0 = int(event["args"]["amount0"])
        amount1 = int(event["args"]["amount1"])
        pool_price = price_from_sqrt(sqrt_price, args.decimals0, args.decimals1)
        fraction = (event["blockNumber"] - guess) / max(1, span)
        swap_ts = guess_ts + fraction * (head_ts - guess_ts)
        cex = minute_close.get(int(swap_ts) // 60)
        if not cex or cex <= 0 or pool_price <= 0:
            continue
        dislocation = float((pool_price / Decimal(str(cex)) - 1) * 10_000)
        # Notional in the quote token, from whichever side is the quote.
        notional = abs(amount1) / (10 ** args.decimals1)
        # Which way the swap pushed the pool. amount0 > 0 means token0 went IN, so the
        # pool price of token0 in token1 FELL. Sign convention checked against the
        # resulting dislocation below rather than asserted.
        pushed_down = amount0 > 0
        rows.append((notional, abs(dislocation), dislocation, pushed_down))

    if len(rows) < 30:
        print("too few usable swaps")
        return

    notionals = [r[0] for r in rows]
    magnitudes = [r[1] for r in rows]
    print()
    print(f"swap notional: median {statistics.median(notionals):,.0f}, "
          f"p90 {sorted(notionals)[int(len(notionals) * 0.9)]:,.0f}, "
          f"max {max(notionals):,.0f}")
    print(f"|dislocation|: median {statistics.median(magnitudes):.2f} bps, "
          f"p90 {sorted(magnitudes)[int(len(magnitudes) * 0.9)]:.2f}, "
          f"max {max(magnitudes):.2f}")

    try:
        r = statistics.correlation(notionals, magnitudes)
    except Exception:
        r = None
    print()
    print(f"correlation between swap SIZE and |dislocation|: "
          f"{('-' if r is None else f'{r:+.3f}')}")

    # The same question by size bucket, which is more robust than a linear correlation on
    # a heavy-tailed size distribution.
    ordered = sorted(rows, key=lambda row: row[0])
    buckets = 5
    per = max(1, len(ordered) // buckets)
    print()
    print(f"{'size quintile':<16} {'median notional':>16} {'median |d|':>12} "
          f"{'p90 |d|':>10} {'max |d|':>10}")
    for i in range(buckets):
        chunk_rows = ordered[i * per:(i + 1) * per]
        if not chunk_rows:
            continue
        mags = sorted(r[1] for r in chunk_rows)
        print(f"{'Q' + str(i + 1):<16} "
              f"{statistics.median([r[0] for r in chunk_rows]):>16,.0f} "
              f"{statistics.median(mags):>12.2f} "
              f"{mags[int(len(mags) * 0.9)]:>10.2f} {max(mags):>10.2f}")

    floor = args.fee / 100.0 + args.taker_bps
    big = [r for r in ordered[-per:]]
    small = [r for r in ordered[:per]]
    big_med = statistics.median([r[1] for r in big])
    small_med = statistics.median([r[1] for r in small])
    print()
    print(f"largest quintile median |d| {big_med:.2f} bps vs smallest "
          f"{small_med:.2f} bps")
    if big_med > small_med * 1.5:
        print()
        print("THE TAIL IS THE SWAPS' OWN IMPACT. |dislocation| rises with the size of")
        print("the swap that produced the print, so the large readings are large orders")
        print("displacing the pool -- not a gap available to anyone else. Capturing that")
        print("means being the NEXT transaction in the block, which is a priority-fee")
        print("auction against backrunners, not a polling loop.")
    else:
        print()
        print("The tail is NOT explained by swap size: large prints are not attached to")
        print("large orders. That is consistent with a market-wide gap, which is takeable")
        print("by anyone who sees it inside its lifetime.")
    # Does the dislocation point the way the swap pushed?
    down = [r[2] for r in rows if r[3]]
    up = [r[2] for r in rows if not r[3]]
    print()
    print("DIRECTION: does the print sit on the side the swap pushed the pool?")
    if down and up:
        median_down = statistics.median(down)
        median_up = statistics.median(up)
        print(f"  swaps pushing the pool DOWN ({len(down):,}): median signed "
              f"dislocation {median_down:+.2f} bps")
        print(f"  swaps pushing the pool UP   ({len(up):,}): median signed "
              f"dislocation {median_up:+.2f} bps")
        separation = median_up - median_down
        print(f"  separation {separation:+.2f} bps")
        if abs(separation) > 2:
            print()
            print("  THE PRINT IS THE DISPLACEMENT THE SWAP JUST CREATED. The pool sits")
            print("  on whichever side the last trade pushed it, which means this")
            print("  dislocation exists in the interval between a swap and whoever")
            print("  closes it -- not as a standing gap. A clock-sampled reader mostly")
            print("  sees the closed state, which is why the live recorder reports")
            print("  2.3 bps for the same pool while swap prints report 8.67. Both are")
            print("  correct about different things, and only one of them is available")
            print("  to a polling loop.")
            print()
            print("  Capturing it means being the next transaction in the block: a")
            print("  priority-fee auction against backrunners, not a faster poller.")
        else:
            print()
            print("  The print does NOT lean the way the swap pushed, so it is not the")
            print("  swap's own displacement. Consistent with a standing gap that a")
            print("  clock-sampled reader should also have seen -- and it did not, so")
            print("  the difference between 8.67 and 2.3 bps needs another explanation.")
    else:
        print("  one-sided sample, cannot compare")

    print()
    print(f"for reference, the cost floor here is {floor:.1f} bps")
    print()
    print("Sampling note: swap prints CONDITION ON A TRADE HAVING HAPPENED, so they are")
    print("not a clock-sampled view of the pool. Whatever the mechanism, a distribution")
    print("built from them describes the moments just after trades, and the live")
    print("recorder describes the pool at arbitrary instants. Comparing the two")
    print("directly is the error; each answers its own question.")


asyncio.run(main())
