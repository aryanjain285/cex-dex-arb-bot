"""Do the deep WETH-quoted pools show a dislocation that clears their higher floor?

The last live candidate class. The depth probe found that the only pools in a 227-asset
universe with real size are quoted in WETH, and `record_synthetic.py` records them with
the exchange's TOKEN/USDT ladder restated in WETH via the ETH/USDT mid.

The floor here is HIGHER than for a direct pair, and that is the point of the exercise:
a synthetic route reaches a deeper pool by paying for an extra leg.

    direct    pool fee + 1 taker fee
    synthetic pool fee + 2 taker fees   (the TOKEN leg and the ETH leg)

At 7.5 bps a side that is 45 bps on a 0.30% pool against 12.5 bps on a 0.05% direct
pair. So a synthetic dislocation has to be three to four times larger to be worth the
same amount, and the report states the floor next to the measurement rather than leaving
the comparison to the reader.
"""
import argparse
from collections import defaultdict
from decimal import Decimal

from research_config import research_config

from src.exchange.univ3_math import notional_to_move_price
from src.research.observations import ObservationStore, mid_dislocation_bps
from src.research.report import classify_dislocation
from src.research.statistics import describe, half_life_seconds

config = research_config("WARNING")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default="data/observations_synthetic.sqlite3")
    parser.add_argument("--taker-bps", type=float, default=7.5)
    parser.add_argument("--legs", type=int, default=2)
    args = parser.parse_args()

    store = ObservationStore(args.db, run_id="analyse-synthetic")
    total = store.count()
    span = store.time_span()
    print(f"{args.db}: {total:,} observations", end="")
    if span:
        print(f" over {(span[1] - span[0]) / 3600:.2f}h")
    else:
        print()
    if not total:
        print("nothing recorded yet")
        return

    weth = {
        chain: str(config.tokens["WETH"][chain].address).lower()
        for chain in config.tokens.get("WETH", {})
    }

    series = defaultdict(list)
    depths = defaultdict(list)
    times = defaultdict(list)
    for observation in store.read_all():
        chain_weth = weth.get(observation.chain)
        if chain_weth is None:
            continue
        # Quote is WETH by construction, so the base is whichever side is not WETH.
        base_is_token0 = str(observation.pool.token0).lower() != chain_weth
        value = mid_dislocation_bps(observation, base_is_token0)
        if value is None:
            continue
        key = (observation.cex_symbol, observation.chain, observation.pool_fee)
        series[key].append(value)
        times[key].append(observation.ts)
        depths[key].append(notional_to_move_price(observation.pool, Decimal("0.01")))

    print()
    print("=" * 108)
    print("SYNTHETIC ROUTE: deep WETH-quoted pools against the exchange via ETH")
    print("=" * 108)
    print(f"{'market':<34} {'n':>5} {'mean d':>9} {'med|d|':>8} {'p99|d|':>9} "
          f"{'floor':>7} {'clears':>7} {'kind':>15} {'half-life':>10}")
    any_clears = False
    for key in sorted(series):
        values = series[key]
        signed = describe(values)
        magnitude = describe([abs(v) for v in values])
        kind = classify_dislocation([float(v) for v in values])
        floor = float(key[2]) / 100.0 + args.taker_bps * args.legs
        median_abs = magnitude.get("p50") or 0.0
        clears = median_abs > floor
        any_clears = any_clears or clears
        cadence = None
        if len(times[key]) > 2:
            gaps = sorted(
                b - a for a, b in zip(times[key], times[key][1:]) if b > a
            )
            cadence = gaps[len(gaps) // 2] if gaps else None
        half = (
            half_life_seconds([float(v) for v in values], cadence)
            if cadence else None
        )
        label = f"{key[0]} {key[1]} {key[2]}"
        print(f"{label:<34} {len(values):>5} "
              f"{(signed.get('mean') or 0):>9.1f} {median_abs:>8.1f} "
              f"{(magnitude.get('p99') or 0):>9.1f} {floor:>7.1f} "
              f"{('YES' if clears else 'no'):>7} {kind['kind']:>15} "
              f"{('-' if half is None else f'{half:.0f}s'):>10}")

    print()
    print(f"floor = pool fee + {args.legs} x {args.taker_bps} bps taker. The synthetic")
    print("route pays for the ETH leg as well as the token leg, so it needs a larger")
    print("dislocation than a direct pair to be worth the same.")
    if not any_clears:
        print()
        print("NO market clears its floor on the median. The deepest pools available")
        print("in this universe do not show a dislocation large enough to pay for the")
        print("extra leg that reaching them requires.")

    print()
    print("1% depth actually observed (upper bound, assumes no tick crossed):")
    for key in sorted(depths, key=lambda k: -(describe(depths[k]).get("p50") or 0)):
        stats = describe(depths[key])
        print(f"  {key[0]:<26} {key[1]:<9} {key[2]:>5}  "
              f"median {float(stats.get('p50') or 0):>14,.2f} WETH")


main()
