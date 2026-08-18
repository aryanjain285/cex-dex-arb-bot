"""The other side of the same trade: is the flow into these pools informed or not?

A direct corollary of the central finding. If the residual dislocation equals the pool fee,
then the arbitrageurs closing gaps are being paid the fee — and the party paying it is the
liquidity provider. So the LP side of the identical trade is worth one measurement before
concluding there is nothing here.

    LP revenue  =  fee x total volume
    LP cost     =  adverse selection, paid only to informed flow
    LP edge     =  fee x volume  -  (gap closed) x informed volume

Which is measurable from swap logs, because a Swap event carries the price AFTER the swap.
Two consecutive swaps therefore bracket one trade: the earlier event's price is the pool
before, the later event's is the pool after. Comparing each against the CEX at that instant
says which way the trade moved the pool:

    moved TOWARD the CEX price   an arbitrageur closing a gap. The LP was picked off, and
                                 the amount is the gap that closed.
    moved AWAY                   someone trading for their own reasons. The LP earned the
                                 fee against no adverse selection.

WHAT THIS IS NOT. It is not an LP backtest. It ignores impermanent loss from the price
level moving, ignores where the LP's range sits, ignores fee compounding, and treats the
gap closed as the whole adverse cost. It answers one narrow question — what fraction of
flow is toxic, and how does the fee compare to what that flow extracts — which is the
question that decides whether a full LP model is worth building.

Exact block timestamps throughout, and 1-second klines. Interpolated timing has already
manufactured three retracted findings in this project, and adverse selection is a
difference between two nearby prices, which is exactly the quantity a timing error destroys.
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
    parser.add_argument("--fee", type=int, default=500)
    parser.add_argument("--minutes", type=float, default=45.0)
    parser.add_argument("--max-blocks", type=int, default=500)
    args = parser.parse_args()

    from src.exchange.univ3 import UniV3DexClient
    client = UniV3DexClient(config.dex, config.network, config.secrets, config.tokens)
    w3 = client._get_w3(args.chain)
    head = await client._rpc(args.chain, lambda: w3.eth.block_number)
    head_ts = int((await client._rpc(
        args.chain, lambda: w3.eth.get_block(int(head))
    ))["timestamp"])
    start_block = max(1, head - int(args.minutes * 60 / 0.25))

    pool = w3.eth.contract(
        address=Web3.to_checksum_address(args.pool), abi=SWAP_ABI
    )
    events = []
    chunk = max(1, (head - start_block) // 6)
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
            print(f"  {begin:,}-{stop:,}: {type(exc).__name__}")
    events.sort(key=lambda e: (int(e["blockNumber"]), int(e.get("logIndex", 0))))
    print(f"{len(events):,} swaps over ~{args.minutes:.0f} min")
    if len(events) < 20:
        print("too few swaps")
        return

    blocks = sorted({int(e["blockNumber"]) for e in events})
    if len(blocks) > args.max_blocks:
        step = len(blocks) / args.max_blocks
        keep = {blocks[int(i * step)] for i in range(args.max_blocks)}
        events = [e for e in events if int(e["blockNumber"]) in keep]
        blocks = sorted(keep)
        print(f"  capped to {len(blocks):,} blocks / {len(events):,} swaps")

    print(f"fetching exact timestamps for {len(blocks):,} blocks...")
    timestamps = {}
    for i, block in enumerate(blocks, 1):
        timestamps[block] = int((await client._rpc(
            args.chain, lambda b=block: w3.eth.get_block(int(b))
        ))["timestamp"])
        if i % 150 == 0:
            print(f"  {i}/{len(blocks)}")

    span_from, span_to = min(timestamps.values()), max(timestamps.values())
    async with aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(total=120)
    ) as session:
        klines = await fetch_klines(
            session, args.symbol, (span_from - 60) * 1000, (span_to + 60) * 1000
        )
    closes = {k[0] // 1000: float(k[4]) for k in klines}
    print(f"{len(klines):,} klines at 1s resolution")

    # MARKOUT, from the amounts in each event. The first version bracketed consecutive
    # prints and measured how much of the gap each trade closed -- which conflates the
    # pool's move with the exchange's move over the interval between swaps, since
    # consecutive swaps can be minutes apart. It reported a median gap closed of 0.16 bps
    # against a gap LEVEL of 4.69, which is the signature of measuring the wrong thing.
    #
    # A Swap event carries amount0 and amount1: the pool's actual deltas, fee included,
    # because the fee stays in the pool. So the liquidity provider's position change is
    # exactly (+amount_in, -amount_out), and valuing both legs at the exchange price gives
    # the provider's P&L on that trade directly. No bracketing, no interval, no
    # conflation -- and the fee is counted automatically rather than added on.
    informed, uninformed = [], []
    fee_bps = args.fee / 100.0
    markouts = []
    for event in events:
        block = int(event["blockNumber"])
        if block not in timestamps:
            continue
        cex = closes.get(timestamps[block])
        if not cex or cex <= 0:
            continue
        amount0 = int(event["args"]["amount0"]) / (10 ** args.decimals0)
        amount1 = int(event["args"]["amount1"]) / (10 ** args.decimals1)
        if amount0 == 0 or amount1 == 0:
            continue
        # Pool deltas: positive means the pool RECEIVED that token. Valued in token1 at
        # the exchange price, the provider's P&L is amount0 * cex + amount1.
        pnl = amount0 * cex + amount1
        notional = abs(amount1)
        if notional <= 0:
            continue
        markout_bps = pnl / notional * 10_000
        markouts.append((notional, markout_bps))
        if markout_bps < 0:
            # The provider lost on this trade: the taker executed better than the
            # exchange price, which is what an arbitrageur does.
            informed.append((notional, -markout_bps))
        else:
            uninformed.append((notional, markout_bps))

    total = len(informed) + len(uninformed)
    if total < 20:
        print("too few bracketed trades")
        return

    informed_count = len(informed) / total
    informed_volume = sum(n for n, _ in informed)
    uninformed_volume = sum(n for n, _ in uninformed)
    volume = informed_volume + uninformed_volume
    informed_share = informed_volume / volume if volume else 0.0

    # Revenue and cost, both in quote units over the window.
    revenue = volume * fee_bps / 10_000
    cost = sum(n * c / 10_000 for n, c in informed)

    print()
    print("=" * 80)
    print(f"FLOW COMPOSITION  ({total:,} bracketed trades, pool fee {fee_bps:.2f} bps)")
    print("=" * 80)
    print(f"  moved TOWARD the exchange (informed)  {len(informed):>6,} trades "
          f"({informed_count:.1%}), {informed_volume:>14,.0f} quote "
          f"({informed_share:.1%})")
    print(f"  moved AWAY (uninformed)               {len(uninformed):>6,} trades "
          f"({1 - informed_count:.1%}), {uninformed_volume:>14,.0f} quote "
          f"({1 - informed_share:.1%})")
    print()
    print(f"  median loss to informed flow          "
          f"{statistics.median([c for _, c in informed]):>8.2f} bps")
    print(f"  median gain from uninformed flow      "
          f"{statistics.median([c for _, c in uninformed]):>8.2f} bps")
    print()
    print("=" * 80)
    print("LP MARKOUT OVER THIS WINDOW, valued at the exchange price")
    print("=" * 80)
    print(f"  for reference, fee income at {fee_bps:.2f} bps on "
          f"{volume:,.0f} would be {revenue:>10,.2f} quote")
    print(f"  gross loss to informed flow  {cost:>14,.2f} quote")
    net = sum(n * m / 10_000 for n, m in markouts)
    net_bps = net / volume * 10_000 if volume else 0.0
    print(f"  MARKOUT, fee included        {net:>14,.2f} quote  "
          f"= {net_bps:>+7.2f} bps of volume")

    # Whether that number is distinguishable from zero. A markout is a mean of
    # per-trade P&L, so it carries the usual standard error -- and reporting -0.51 bps
    # from 263 trades without one would imply a precision the sample does not have.
    per_trade = [m for _, m in markouts]
    if len(per_trade) > 2:
        spread = statistics.stdev(per_trade)
        standard_error = spread / (len(per_trade) ** 0.5)
        mean_bps = statistics.fmean(per_trade)
        print(f"  per-trade markout            mean {mean_bps:>+7.2f} bps, "
              f"sd {spread:.2f}, se {standard_error:.2f} over {len(per_trade):,} trades")
        if abs(mean_bps) < 1.96 * standard_error:
            print(f"  NOT DISTINGUISHABLE FROM ZERO at 95%: the interval is "
                  f"[{mean_bps - 1.96 * standard_error:+.2f}, "
                  f"{mean_bps + 1.96 * standard_error:+.2f}] bps. Which is itself the")
            print(f"  expected result in a competitive market -- the fee is exactly what")
            print(f"  arbitrage competes away, so the provider ends at break-even before")
            print(f"  impermanent loss, and impermanent loss only subtracts.")
    print()
    if net > 0:
        print("  The provider ended this window ahead, fee included. That is the ONE")
        print("  direction the measurements point at, and it is a direction rather than")
        print("  a result: this is TRADING P&L only. It ignores impermanent loss from the")
        print("  level moving against the inventory, where the range sits, and every cost")
        print("  of holding the position. It says a full model is worth building.")
    else:
        print("  Informed flow extracted more than the fee earned over this window, so")
        print("  the LP side is not obviously better than the taker side and the")
        print("  symmetry of the central finding holds: the fee is what arbitrage")
        print("  competes away, and it is paid by whoever provides the liquidity.")
    print()
    print("Not an LP backtest. This is the markout on trades only: impermanent loss,")
    print("range placement, fee compounding and inventory cost are all absent. One narrow")
    print("question only -- is the flow toxic enough to eat the fee.")


asyncio.run(main())
