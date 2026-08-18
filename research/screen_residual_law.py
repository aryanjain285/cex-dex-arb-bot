"""Does any pool break the residual-equals-pool-fee law, with depth to trade it?

The central finding is that the post-swap residual dislocation equals the pool fee, because
arbitrage closes each gap until the remainder no longer covers the irreducible cost of using
that pool. Measured on four pool/tier combinations:

    ETH/USDC 0.05%   5 bps fee  ->   4.69 bps residual
    ETH/USDT 0.05%   5 bps fee  ->   4.70 bps
    ETH/USDC 0.30%  30 bps fee  ->  22.38 bps
    ETH/USDT 0.30%  30 bps fee  ->  30.30 bps

If that holds everywhere, the strategy has no room anywhere, and four pools is not
everywhere. The law should hold exactly where arbitrage is COMPETITIVE, so the interesting
case is a pool nobody is watching -- and the depth probe already showed why those are
usually useless: a pool thin enough to be ignored is thin enough that no size fits.

So the search is for a pool that breaks the law AND has depth. Both conditions, because
either alone is worthless:

    residual >> fee, no depth      a big number nobody can fill
    residual ~= fee, deep          arbitraged, which is every pool measured so far

Method: for each candidate, take the most recent swaps, get the EXACT timestamp of every
block containing one (no interpolation -- interpolation has already manufactured two
retracted findings today), compare against 1-second Binance klines, and report the residual
against the fee alongside the 1% depth from the same swap's liquidity field.

Ranked by (residual - fee) x depth, which is roughly the size of the opportunity if one
exists.
"""
import argparse
import asyncio
import json
import statistics
from collections import defaultdict
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

import aiohttp
from research_config import research_config
from web3 import Web3

from src.exchange.univ3_math import V3Pool, notional_to_move_price

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


async def klines_1s(session, symbol, start_ms, end_ms):
    out, cursor = [], start_ms
    while cursor < end_ms:
        params = {"symbol": symbol, "interval": "1s", "startTime": cursor,
                  "endTime": end_ms, "limit": 1000}
        try:
            async with session.get(KLINES, params=params) as response:
                if response.status == 429:
                    await asyncio.sleep(5)
                    continue
                if response.status != 200:
                    return out, None
                batch = await response.json()
        except Exception:
            return out, None
        if not batch:
            break
        out.extend(batch)
        if batch[-1][0] <= cursor:
            break
        cursor = batch[-1][0] + 1
        await asyncio.sleep(0.08)
    return out, 1


def price_from_sqrt(sqrt_price_x96, decimals0, decimals1):
    raw = (Decimal(sqrt_price_x96) / Decimal(2 ** 96)) ** 2
    return raw * (Decimal(10) ** decimals0) / (Decimal(10) ** decimals1)


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--chain", default="arbitrum")
    parser.add_argument("--targets", default="research/targets_wide.json")
    parser.add_argument("--minutes", type=float, default=45.0)
    parser.add_argument("--pools", type=int, default=22)
    parser.add_argument("--swaps-per-pool", type=int, default=12)
    parser.add_argument("--min-depth", type=float, default=2000.0)
    args = parser.parse_args()

    from src.exchange.univ3 import UniV3DexClient
    client = UniV3DexClient(config.dex, config.network, config.secrets, config.tokens)
    w3 = client._get_w3(args.chain)
    head = await client._rpc(args.chain, lambda: w3.eth.block_number)
    head_ts = int((await client._rpc(
        args.chain, lambda: w3.eth.get_block(int(head))
    ))["timestamp"])
    start_block = max(1, head - int(args.minutes * 60 / 0.25))

    targets = [
        t for t in json.loads(Path(args.targets).read_text(encoding="utf-8"))
        if t["chain"] == args.chain and t["quote"] in ("USDT", "USDC")
    ]
    # Ordered by MEASURED DEPTH, not alphabetically. A first version took the first N
    # names and drew eighteen pools with zero swaps in forty-five minutes -- which is a
    # real finding about how dead the mid-cap Uniswap universe is, and useless for
    # measuring a residual. The depth probe already ranked every pool, so use it.
    depth_by_key = {}
    probe_path = Path("research/depth_probe.json")
    if probe_path.exists():
        for row in json.loads(probe_path.read_text(encoding="utf-8")):
            if row["chain"] != args.chain:
                continue
            key = (row["asset"], row["fee"])
            depth_by_key[key] = max(depth_by_key.get(key, 0.0), row["depth_usd"])

    chosen = {}
    for t in sorted(targets, key=lambda t: (t["base"], t["fee"], t["quote"] != "USDT")):
        chosen.setdefault((t["base"], t["fee"]), t)
    candidates = sorted(
        chosen.values(),
        key=lambda t: -depth_by_key.get((t["base"], t["fee"]), 0.0),
    )[: args.pools]
    known = sum(1 for t in candidates if depth_by_key.get((t["base"], t["fee"]), 0.0) > 0)
    print(f"selected by measured 1% depth ({known} of {len(candidates)} have a "
          f"depth reading)")
    print(f"{len(candidates)} pools on {args.chain}, last {args.minutes:.0f} min "
          f"(blocks {start_block:,}-{head:,})")

    rows = []
    activity = []
    for i, target in enumerate(candidates, 1):
        pool = w3.eth.contract(
            address=Web3.to_checksum_address(target["pool_address"]), abi=SWAP_ABI
        )
        try:
            events = await client._rpc(
                args.chain,
                lambda: pool.events.Swap().get_logs(
                    from_block=start_block, to_block=head
                ),
            )
        except Exception as exc:
            print(f"  {target['cex_symbol']:<12} {target['fee']:>5}  "
                  f"{type(exc).__name__}")
            continue
        activity.append((target["cex_symbol"], target["fee"], len(events)))
        if len(events) < 4:
            print(f"  {target['cex_symbol']:<12} {target['fee']:>5}  "
                  f"only {len(events)} swaps, skipping")
            continue
        events = events[-args.swaps_per_pool:]

        blocks = sorted({int(e["blockNumber"]) for e in events})
        timestamps = {}
        for block in blocks:
            timestamps[block] = int((await client._rpc(
                args.chain, lambda b=block: w3.eth.get_block(int(b))
            ))["timestamp"])

        symbol = target["cex_symbol"].replace("/", "")
        span_from, span_to = min(timestamps.values()), max(timestamps.values())
        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=60)
        ) as session:
            klines, resolution = await klines_1s(
                session, symbol, (span_from - 30) * 1000, (span_to + 30) * 1000
            )
        if not klines or resolution is None:
            print(f"  {target['cex_symbol']:<12} {target['fee']:>5}  no 1s klines")
            continue
        closes = {k[0] // 1000: float(k[4]) for k in klines}

        base_is_token0 = (
            str(target["base_address"]).lower() < str(target["quote_address"]).lower()
        )
        magnitudes, depths = [], []
        for event in events:
            block = int(event["blockNumber"])
            cex = closes.get(timestamps[block])
            if cex is None or cex <= 0:
                continue
            sqrt_price = int(event["args"]["sqrtPriceX96"])
            liquidity = int(event["args"]["liquidity"])
            d0 = target["base_decimals"] if base_is_token0 else target["quote_decimals"]
            d1 = target["quote_decimals"] if base_is_token0 else target["base_decimals"]
            pool_price = price_from_sqrt(sqrt_price, d0, d1)
            if not base_is_token0 and pool_price > 0:
                pool_price = Decimal(1) / pool_price
            if pool_price <= 0:
                continue
            ratio = pool_price / Decimal(str(cex))
            if not (Decimal("0.5") <= ratio <= Decimal("2")):
                continue
            magnitudes.append(abs(float((ratio - 1) * 10_000)))
            snapshot = V3Pool(
                sqrt_price_x96=sqrt_price, liquidity=liquidity, tick=0,
                fee=target["fee"], tick_spacing=1, ticks=[],
                decimals0=d0, decimals1=d1,
            )
            depth = float(notional_to_move_price(snapshot, Decimal("0.01")))
            if not base_is_token0:
                # Denominated in the base when the base is token1, so not comparable.
                depth = 0.0
            depths.append(depth)

        if len(magnitudes) < 3:
            print(f"  {target['cex_symbol']:<12} {target['fee']:>5}  "
                  f"only {len(magnitudes)} matched")
            continue
        fee_bps = target["fee"] / 100.0
        residual = statistics.median(magnitudes)
        depth = statistics.median(depths) if depths else 0.0
        rows.append({
            "symbol": target["cex_symbol"], "fee": target["fee"],
            "fee_bps": fee_bps, "residual": residual,
            "ratio": residual / fee_bps if fee_bps > 0 else None,
            "excess": residual - fee_bps, "depth": depth, "n": len(magnitudes),
        })
        print(f"  {target['cex_symbol']:<12} {target['fee']:>5}  "
              f"residual {residual:>7.2f} vs fee {fee_bps:>6.2f}  "
              f"ratio {residual / fee_bps if fee_bps else 0:>5.2f}x  "
              f"depth {depth:>12,.0f}  n={len(magnitudes)}")

    dead = [a for a in activity if a[2] == 0]
    if dead:
        print()
        print(f"{len(dead)} of {len(activity)} pools had ZERO swaps in "
              f"{args.minutes:.0f} minutes: "
              + ", ".join(f"{s} {f}" for s, f, _ in dead[:10]))
        print("A pool nobody trades cannot be arbitraged, which is why its price can")
        print("drift -- and cannot be traded either, which is why the drift is worthless.")

    if not rows:
        print()
        print("nothing measured: every selected pool was inactive or unmatched")
        return

    print()
    print("=" * 96)
    print("DOES ANY POOL BREAK THE LAW, WITH DEPTH TO TRADE IT?")
    print("=" * 96)
    print(f"{'market':<20} {'fee bps':>8} {'residual':>10} {'ratio':>7} "
          f"{'excess':>8} {'1% depth':>14} {'n':>4}")
    for row in sorted(rows, key=lambda r: -(r["excess"] * max(r["depth"], 1))):
        print(f"{row['symbol'] + ' ' + str(row['fee']):<20} {row['fee_bps']:>8.2f} "
              f"{row['residual']:>10.2f} "
              f"{(row['ratio'] or 0):>6.2f}x {row['excess']:>8.2f} "
              f"{row['depth']:>14,.0f} {row['n']:>4}")

    ratios = [r["ratio"] for r in rows if r["ratio"]]
    print()
    print(f"residual / fee across {len(ratios)} pools: "
          f"median {statistics.median(ratios):.2f}x, "
          f"min {min(ratios):.2f}x, max {max(ratios):.2f}x")
    breakers = [
        r for r in rows
        if r["excess"] > 2.0 and r["depth"] >= args.min_depth
    ]
    print()
    if breakers:
        print(f"{len(breakers)} pool(s) show a residual more than 2 bps above their fee "
              f"WITH at least {args.min_depth:,.0f} of 1% depth:")
        for row in breakers:
            print(f"  {row['symbol']} {row['fee']}: residual {row['residual']:.2f} vs "
                  f"fee {row['fee_bps']:.2f}, excess {row['excess']:.2f} bps, "
                  f"depth {row['depth']:,.0f}")
        print()
        print("These are the only candidates the measurements support. Each needs a full")
        print("tick read and a longer window before anything is concluded -- a dozen")
        print("swaps in one window is a hint, not a result.")
    else:
        print("NONE. Every pool with tradeable depth sits at or below its own fee, which")
        print("is what an arbitraged market looks like from the inside. The pools that")
        print("exceed their fee cannot absorb size, which is presumably why they exceed it.")


asyncio.run(main())
