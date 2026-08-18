"""Is today representative? Months of exchange history, against one day of recording.

Every measurement so far comes from a few hours of one day. The conclusion drawn from
them -- a raw dislocation of about 2.3 bps against a 12.5 bps requirement -- is only as
general as that day, and the obvious threat to it is that the day was calm. A dislocation
between two venues is created by price movement one venue has not yet reflected, so it
should scale with volatility, and a 10th-percentile day would understate a normal one.

Binance klines are public, need no key, and go back years. They give the CEX side at
one-minute resolution, which is enough to answer the question that matters:

    WHERE DOES TODAY SIT IN THE VOLATILITY DISTRIBUTION OF THE LAST SIX MONTHS?

If today is typical, the live measurement generalises. If it is unusually quiet, every
number in the report needs scaling up before it can be argued about, and the honest report
says by how much.

WHAT THIS DELIBERATELY DOES NOT CLAIM. It does not reconstruct historical dislocations.
That needs the DEX side at the same resolution, which means either an archive node or a
subgraph key -- a public endpoint prunes state, so a pool's price six months ago is simply
not available here. So this bounds the REGIME rather than the edge, and the two must not
be conflated: a volatility percentile tells you whether to trust a measurement, not what
the measurement would have been.

Klines also carry no order book, so nothing here can speak to depth or size.
"""
import argparse
import asyncio
import statistics
from datetime import datetime, timedelta, timezone

import aiohttp
from research_config import research_config

config = research_config("WARNING")

BASE = "https://api.binance.com/api/v3/klines"
MAX_LIMIT = 1000


async def fetch_klines(session, symbol, interval, start_ms, end_ms):
    """All klines in a window, paginated. Binance caps a response at 1,000."""
    out = []
    cursor = start_ms
    while cursor < end_ms:
        params = {
            "symbol": symbol, "interval": interval,
            "startTime": cursor, "endTime": end_ms, "limit": MAX_LIMIT,
        }
        async with session.get(BASE, params=params) as response:
            if response.status == 429:
                # Public endpoint, weight-metered. Backing off is the only correct
                # response; hammering it earns an IP ban rather than data.
                await asyncio.sleep(10)
                continue
            response.raise_for_status()
            batch = await response.json()
        if not batch:
            break
        out.extend(batch)
        last_open = batch[-1][0]
        if last_open <= cursor:
            break
        cursor = last_open + 1
        await asyncio.sleep(0.15)
    return out


def realised_volatility_bps(closes):
    """Annualisation-free: the standard deviation of one-minute log returns, in bps.

    Deliberately not annualised. The quantity being compared is a dislocation measured
    over seconds, so the natural unit is per-minute movement -- and annualising invites
    comparison against equity vol numbers that mean something different.
    """
    if len(closes) < 3:
        return None
    returns = []
    for a, b in zip(closes, closes[1:]):
        if a > 0 and b > 0:
            returns.append((b / a - 1) * 10_000)
    if len(returns) < 2:
        return None
    return statistics.stdev(returns)


def percentile_of(value, population):
    if not population or value is None:
        return None
    below = sum(1 for v in population if v is not None and v < value)
    return below / len([v for v in population if v is not None])


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbols", default="ETHUSDT,BTCUSDT,ARBUSDT,LINKUSDT")
    parser.add_argument("--days", type=int, default=180)
    parser.add_argument("--interval", default="1m")
    args = parser.parse_args()

    now = datetime.now(timezone.utc)
    start = now - timedelta(days=args.days)
    start_ms = int(start.timestamp() * 1000)
    end_ms = int(now.timestamp() * 1000)

    # Today's window, for the comparison. Deliberately the same hours the recorders ran,
    # so the comparison is like for like rather than today-so-far against whole days.
    today_start = now - timedelta(hours=6)
    today_start_ms = int(today_start.timestamp() * 1000)

    print(f"Binance klines, {args.interval}, {args.days} days to {now:%Y-%m-%d %H:%M} UTC")
    print()

    async with aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(total=120)
    ) as session:
        for symbol in args.symbols.split(","):
            symbol = symbol.strip()
            if not symbol:
                continue
            try:
                klines = await fetch_klines(
                    session, symbol, args.interval, start_ms, end_ms
                )
            except Exception as exc:
                print(f"{symbol}: fetch failed ({type(exc).__name__}: {exc})")
                continue
            if not klines:
                print(f"{symbol}: no data")
                continue

            # Group into hourly buckets and compute realised vol per hour.
            hourly = {}
            for kline in klines:
                open_ms, _o, _h, _l, close, *_rest = kline
                bucket = open_ms // 3_600_000
                hourly.setdefault(bucket, []).append(float(close))
            vols = {
                bucket: realised_volatility_bps(closes)
                for bucket, closes in sorted(hourly.items())
            }
            population = [v for v in vols.values() if v is not None]

            today_buckets = [
                b for b in vols if b * 3_600_000 >= today_start_ms
            ]
            today_vols = [vols[b] for b in today_buckets if vols[b] is not None]
            today_median = statistics.median(today_vols) if today_vols else None

            ordered = sorted(population)

            def pct(p):
                if not ordered:
                    return None
                index = min(len(ordered) - 1, int(len(ordered) * p))
                return ordered[index]

            print(f"=== {symbol} ===")
            print(f"  {len(klines):,} klines, {len(population):,} hourly buckets")
            print(f"  per-minute realised volatility, bps:")
            print(f"    p10 {pct(0.10):>8.2f}   p25 {pct(0.25):>8.2f}   "
                  f"p50 {pct(0.50):>8.2f}   p75 {pct(0.75):>8.2f}   "
                  f"p90 {pct(0.90):>8.2f}   p99 {pct(0.99):>8.2f}")
            if today_median is not None:
                where = percentile_of(today_median, population)
                print(f"  the last 6 hours: median {today_median:.2f} bps/min, "
                      f"which is the {where:.0%} percentile of the last {args.days} days")
                ratio = (pct(0.50) / today_median) if today_median > 0 else None
                if where is not None and where < 0.25:
                    print(f"    TODAY IS QUIET. A median day is {ratio:.1f}x more "
                          f"volatile, so a dislocation measured today understates a "
                          f"typical one by roughly that factor if the two scale "
                          f"together.")
                elif where is not None and where > 0.75:
                    print(f"    TODAY IS BUSY. A median day is {ratio:.1f}x as "
                          f"volatile, so today's dislocation OVERSTATES a typical one.")
                else:
                    print(f"    Today is an ordinary day for this symbol, so a "
                          f"measurement taken today generalises without rescaling.")
            print()

    print("This bounds the REGIME, not the edge. Reconstructing historical dislocations")
    print("needs the DEX side at the same resolution, which means an archive node or a")
    print("subgraph key -- a public endpoint prunes state, so a pool's price six months")
    print("ago is not available here. A volatility percentile tells you whether to trust")
    print("a measurement, not what the measurement would have been.")


asyncio.run(main())
