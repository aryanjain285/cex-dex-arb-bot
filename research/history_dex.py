"""Reconstruct historical dislocations, and answer the question live data cannot.

The live recording says the dislocation is about 2.4 bps and does not track volatility --
but it was taken on a 12th-percentile day, over a volatility range of 0.45 to 4.23 bps/min
against a 180-day p99 of 20.43. So it cannot distinguish two very different worlds:

    the dislocation is a roughly constant 2 bps feature, and no regime helps
    the dislocation is threshold-like, invisible until moves exceed the fee

Waiting for a violent day would settle it. History settles it now.

WHY THIS IS POSSIBLE WITHOUT AN ARCHIVE NODE. Public endpoints prune state, so `eth_call`
at an old block fails -- Ethereum's returns 403 beyond a ten-block span. But LOGS are
stored separately, and Arbitrum's public endpoint serves them back to June 2025, fourteen
months. A Uniswap v3 Swap event carries `sqrtPriceX96` AFTER the swap, which is the pool
price exactly, and `liquidity`, which gives the depth proxy. So the DEX side of history is
recoverable for free; the cap is 10,000 logs per query rather than any block range.

The CEX side is Binance one-minute klines, also public and also free.

SAMPLING IS STRATIFIED BY VOLATILITY, ON PURPOSE. Fourteen months of swaps is millions of
events, and fetching them uniformly would spend the whole budget on ordinary days -- which
the live recording already covers. The klines say which hours were violent, so the sample
targets the volatility distribution directly: the most extreme hours, plus a spread across
deciles for comparison. That is the experiment the live data cannot run.

WHAT IS STILL MISSING. Tick liquidity beyond the active range, so price IMPACT at size
cannot be reconstructed -- only the raw dislocation and the 1% depth proxy. And a swap
event is a print, not a quote: the price after a swap is where that trade left the pool,
which is the right number for a dislocation and is not the same as a resting quote.
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


async def fetch_klines(session, symbol, start_ms, end_ms):
    out = []
    cursor = start_ms
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


class BlockClock:
    """Timestamp -> block by BINARY SEARCH, and back exactly.

    The first version interpolated between five anchors spanning fourteen months, on the
    reasoning that Arbitrum block numbers are near-linear in time. Measured, that
    interpolation was wrong by a median of 51 minutes and a worst case of 121:

        target 2026-03-21 05:39 UTC -> the assumed block was 2026-03-21 03:38, -121 min
        target 2026-06-04 05:39 UTC -> the assumed block was 2026-06-04 06:51,  +72 min

    That error alone produced the entire historical result. A stale comparison price
    manufactures apparent dislocation of roughly volatility times sqrt(minutes of error),
    so 51 minutes manufactures 39 bps at median volatility and 416 bps at the most violent
    hour sampled -- against reported medians of 16-52 and 560. The same order at every
    level. Worse, the manufactured amount SCALES WITH VOLATILITY, which is exactly the
    +0.703 correlation the run found and exactly the finding it appeared to support.

    So blocks now define the time rather than the reverse. Binary search costs about 27
    calls to locate any block in a 150-million-block range, affordable for a few dozen
    windows, and both endpoints of every window are read from the chain. Within one hour,
    interpolating between two exact endpoints is accurate to seconds.
    """

    def __init__(self, client, chain, w3):
        self.client = client
        self.chain = chain
        self.w3 = w3
        self._cache = {}
        self.searches = 0

    async def timestamp_of(self, block):
        block = int(block)
        if block not in self._cache:
            data = await self.client._rpc(
                self.chain, lambda b=block: self.w3.eth.get_block(b)
            )
            self._cache[block] = int(data["timestamp"])
        return self._cache[block]

    async def block_at(self, target_ts, low, high, tolerance_seconds=30):
        """Lowest block whose timestamp is at least `target_ts`, within tolerance.

        None when the target predates `low`, which is the honest answer for a time outside
        the node's log retention -- clamping to the earliest available block would
        silently mislabel the window.
        """
        self.searches += 1
        low_ts = await self.timestamp_of(low)
        high_ts = await self.timestamp_of(high)
        if target_ts < low_ts or target_ts > high_ts:
            return None
        lo, hi = int(low), int(high)
        while hi - lo > 1:
            mid = (lo + hi) // 2
            mid_ts = await self.timestamp_of(mid)
            if abs(mid_ts - target_ts) <= tolerance_seconds:
                return mid
            if mid_ts < target_ts:
                lo = mid
            else:
                hi = mid
        return hi


def price_from_sqrt(sqrt_price_x96, decimals0, decimals1):
    raw = (Decimal(sqrt_price_x96) / Decimal(2 ** 96)) ** 2
    return raw * (Decimal(10) ** decimals0) / (Decimal(10) ** decimals1)


def realised_vol_bps(closes):
    if len(closes) < 3:
        return None
    returns = [
        float((b / a - 1) * 10_000)
        for a, b in zip(closes, closes[1:]) if a > 0 and b > 0
    ]
    return statistics.stdev(returns) if len(returns) > 1 else None


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool", default="0xC6962004f452bE9203591991D15f6b388e09E8D0",
                        help="WETH/USDC 0.05% on Arbitrum")
    parser.add_argument("--chain", default="arbitrum")
    parser.add_argument("--symbol", default="ETHUSDT")
    parser.add_argument("--fee", type=int, default=500)
    parser.add_argument("--decimals0", type=int, default=18, help="WETH")
    parser.add_argument("--decimals1", type=int, default=6, help="USDC")
    parser.add_argument("--base-is-token0", action="store_true", default=True)
    parser.add_argument("--days", type=int, default=180)
    parser.add_argument("--windows", type=int, default=60,
                        help="hour-long windows to sample")
    parser.add_argument("--taker-bps", type=float, default=7.5)
    parser.add_argument("--out", default="research/history_dislocation.json")
    args = parser.parse_args()

    from src.exchange.univ3 import UniV3DexClient
    client = UniV3DexClient(config.dex, config.network, config.secrets, config.tokens)
    w3 = client._get_w3(args.chain)
    head = await client._rpc(args.chain, lambda: w3.eth.block_number)

    now = datetime.now(timezone.utc)
    start = now - timedelta(days=args.days)
    start_ms, end_ms = int(start.timestamp() * 1000), int(now.timestamp() * 1000)

    print(f"pool {args.pool} on {args.chain}, head {head:,}")
    print(f"window {start:%Y-%m-%d} to {now:%Y-%m-%d}")

    async with aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(total=180)
    ) as session:
        print(f"fetching {args.days} days of {args.symbol} klines...")
        klines = await fetch_klines(session, args.symbol, start_ms, end_ms)
    if not klines:
        print("no klines")
        return
    print(f"  {len(klines):,} minutes")

    # Minute close, and hourly realised volatility.
    minute_close = {}
    hourly_closes = defaultdict(list)
    for kline in klines:
        open_ms, _o, _h, _l, close, *_ = kline
        minute_close[open_ms // 60_000] = float(close)
        hourly_closes[open_ms // 3_600_000].append(float(close))
    hourly_vol = {
        bucket: realised_vol_bps(closes)
        for bucket, closes in hourly_closes.items()
    }
    usable = {b: v for b, v in hourly_vol.items() if v is not None}
    if not usable:
        print("no usable volatility buckets")
        return

    ordered = sorted(usable.items(), key=lambda kv: kv[1])
    # Stratified: the most violent hours, plus a spread across the rest. The violent tail
    # is the half the live recording cannot reach, so it gets half the budget.
    tail = [b for b, _ in ordered[-(args.windows // 2):]]
    step = max(1, len(ordered) // (args.windows - len(tail)))
    spread = [b for b, _ in ordered[::step]][: args.windows - len(tail)]
    selected = sorted(set(tail) | set(spread))
    print(f"  sampling {len(selected)} hourly windows, "
          f"{len(tail)} of them from the volatility tail")

    clock = BlockClock(client, args.chain, w3)
    earliest_block = max(1, head - 150_000_000)
    earliest_ts = await clock.timestamp_of(earliest_block)
    print(f"  logs available from block {earliest_block:,} at "
          f"{datetime.fromtimestamp(earliest_ts, timezone.utc):%Y-%m-%d}")
    print(f"  locating each window by binary search, about 27 calls each")

    pool_contract = w3.eth.contract(
        address=Web3.to_checksum_address(args.pool), abi=SWAP_ABI
    )

    results = []
    skipped_no_blocks = skipped_failed = 0
    for i, bucket in enumerate(selected, 1):
        hour_start_ts = bucket * 3600
        if hour_start_ts < earliest_ts:
            skipped_no_blocks += 1
            continue
        from_block = await clock.block_at(hour_start_ts, earliest_block, head)
        to_block = await clock.block_at(hour_start_ts + 3600, earliest_block, head)
        if from_block is None or to_block is None or to_block <= from_block:
            skipped_no_blocks += 1
            continue
        # The ACTUAL span these blocks cover, read from the chain. Every swap is timed by
        # interpolating between these two, so the residual error is within-hour block-time
        # variation rather than fourteen months of accumulated drift.
        actual_from_ts = await clock.timestamp_of(from_block)
        actual_to_ts = await clock.timestamp_of(to_block)
        placement_error_s = abs(actual_from_ts - hour_start_ts)
        if placement_error_s > 120:
            # Refused rather than recorded with a caveat. At high volatility two minutes
            # of misplacement is tens of basis points of manufactured dislocation, the
            # same order as the signal being measured.
            skipped_no_blocks += 1
            continue
        try:
            logs = await client._rpc(
                args.chain,
                lambda f=from_block, t=to_block: pool_contract.events.Swap().get_logs(
                    from_block=f, to_block=t
                ),
            )
        except Exception as exc:
            skipped_failed += 1
            if skipped_failed <= 3:
                print(f"  window {i}: {type(exc).__name__}: {str(exc)[:70]}")
            continue

        dislocations, depths = [], []
        for event in logs:
            sqrt_price = int(event["args"]["sqrtPriceX96"])
            liquidity = int(event["args"]["liquidity"])
            pool_price = price_from_sqrt(sqrt_price, args.decimals0, args.decimals1)
            if not args.base_is_token0 and pool_price > 0:
                pool_price = Decimal(1) / pool_price
            # The kline covering this swap. Block time is interpolated, so the minute is
            # approximate -- and one minute of ETH movement at median volatility is about
            # 5 bps, which bounds the noise this adds.
            # Interpolated between two timestamps READ FROM THE CHAIN, not between
            # anchors months apart.
            fraction = (event["blockNumber"] - from_block) / max(
                1, to_block - from_block
            )
            swap_ts = actual_from_ts + fraction * (actual_to_ts - actual_from_ts)
            minute = int(swap_ts) // 60
            cex = minute_close.get(minute)
            if not cex or cex <= 0 or pool_price <= 0:
                continue
            ratio = pool_price / Decimal(str(cex))
            # The same plausibility guard the live path uses: a ratio far from 1 is not a
            # dislocation, it is a mismatch.
            if not (Decimal("0.5") <= ratio <= Decimal("2")):
                continue
            dislocations.append(float((ratio - 1) * 10_000))
            snapshot = V3Pool(
                sqrt_price_x96=sqrt_price, liquidity=liquidity, tick=0,
                fee=args.fee, tick_spacing=1, ticks=[],
                decimals0=args.decimals0, decimals1=args.decimals1,
            )
            depths.append(float(notional_to_move_price(snapshot, Decimal("0.01"))))

        if not dislocations:
            continue
        results.append({
            "hour": bucket,
            "when": datetime.fromtimestamp(hour_start_ts, timezone.utc).isoformat(),
            "volatility_bps_min": usable[bucket],
            "swaps": len(dislocations),
            "median_abs_dislocation": statistics.median(
                [abs(d) for d in dislocations]
            ),
            "p90_abs_dislocation": sorted([abs(d) for d in dislocations])[
                int(len(dislocations) * 0.9)
            ],
            "max_abs_dislocation": max(abs(d) for d in dislocations),
            "mean_signed": statistics.fmean(dislocations),
            "sign_flips": sum(1 for d in dislocations if d > 0) / len(dislocations),
            "median_depth_1pct": statistics.median(depths) if depths else None,
            "placement_error_s": placement_error_s,
            "window_span_s": actual_to_ts - actual_from_ts,
        })
        if i % 10 == 0:
            print(f"  {i}/{len(selected)} windows, {len(results)} usable")

    if not results:
        print("no usable windows")
        return

    Path(args.out).write_text(json.dumps(results, indent=2), encoding="utf-8")
    floor = args.fee / 100.0 + args.taker_bps

    print()
    print("=" * 100)
    print(f"HISTORICAL DISLOCATION vs VOLATILITY REGIME  ({len(results)} hourly windows)")
    print("=" * 100)
    print(f"{'when':<20} {'vol bps/min':>12} {'swaps':>7} {'med|d|':>8} "
          f"{'p90|d|':>8} {'max|d|':>8} {'>floor?':>8}")
    for row in sorted(results, key=lambda r: r["volatility_bps_min"]):
        clears = "YES" if row["median_abs_dislocation"] > floor else "no"
        print(f"{row['when'][:19]:<20} {row['volatility_bps_min']:>12.2f} "
              f"{row['swaps']:>7} {row['median_abs_dislocation']:>8.2f} "
              f"{row['p90_abs_dislocation']:>8.2f} {row['max_abs_dislocation']:>8.2f} "
              f"{clears:>8}")

    vols = [r["volatility_bps_min"] for r in results]
    meds = [r["median_abs_dislocation"] for r in results]
    print()
    print(f"floor for this pool: {floor:.1f} bps "
          f"(pool fee {args.fee / 100:.2f} + taker {args.taker_bps})")
    try:
        r = statistics.correlation(vols, meds)
        print(f"correlation between hourly volatility and median |dislocation|: {r:+.3f}")
    except Exception:
        pass
    cleared = [r for r in results if r["median_abs_dislocation"] > floor]
    print(f"windows whose MEDIAN dislocation clears the floor: "
          f"{len(cleared)} of {len(results)}")
    if cleared:
        print("  " + ", ".join(
            f"{r['when'][:16]} at {r['volatility_bps_min']:.1f} bps/min "
            f"-> {r['median_abs_dislocation']:.1f} bps"
            for r in sorted(cleared, key=lambda r: -r["median_abs_dislocation"])[:8]
        ))
    ever = [r for r in results if r["max_abs_dislocation"] > floor]
    print(f"windows where the MAXIMUM single swap cleared it: "
          f"{len(ever)} of {len(results)}")
    if skipped_no_blocks or skipped_failed:
        print(f"skipped: {skipped_no_blocks} outside the log retention window, "
              f"{skipped_failed} failed")
    print()
    placements = [r["placement_error_s"] for r in results]
    spans = [r["window_span_s"] for r in results]
    print(f"window placement error: median {statistics.median(placements):.0f}s, "
          f"worst {max(placements):.0f}s")
    print(f"window spans: median {statistics.median(spans):.0f}s against a nominal 3600")
    print("A previous version of this script got that placement wrong by a median of 51")
    print("minutes, which manufactured its entire result. It is reported here so these")
    print("numbers can be checked against what the timing alone could produce.")
    print()
    print("A swap event is a PRINT, not a quote: the price after a swap is where that")
    print("trade left the pool. That is the right number for a dislocation and is not a")
    print("resting quote. Block times are interpolated between anchors, so each swap is")
    print("matched to its minute approximately -- one minute of median ETH movement is")
    print("about 5 bps, which bounds the noise that adds. Tick liquidity beyond the")
    print("active range is not in the logs, so impact at size is not reconstructable.")


asyncio.run(main())
