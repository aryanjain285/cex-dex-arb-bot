"""Rank the wide universe by raw dislocation, with the filters that make it mean something.

The question: across 175 Binance-listed assets on three chains, does ANY market show a
dislocation that both exceeds its cost floor and changes sign? The narrow universe
(ETH and stablecoins) shows a +2.6 bps standing basis against a 12.5 bps floor -- the
phenomenon exists, is 5x too small, and does not fluctuate. That is a statement about
the most heavily arbitraged pairs in the market and says nothing about a mid-cap token
on a 0.30% pool.

Three filters, each of which removes a class of false positive that would otherwise rank
at the top:

  LIQUIDITY. An empty pool reports whatever price its creator set at initialisation and
  keeps it forever. Untreated, these produced dislocations up to 3.9e52 bps and occupied
  every top slot. A pool with no active liquidity yields no dislocation at all.

  DEPTH. Non-zero liquidity is far too weak. The notional needed to move the price 1% is
  computable from slot0, so pools that cannot absorb a stated minimum are reported
  separately rather than ranked against real markets.

  SIGN PERSISTENCE. A dislocation that never changes sign is a standing basis -- a price
  for the asset being on that chain rather than in that custodian -- and capturing it
  twice requires bridging inventory back at a cost equal to the basis. Ranking it as a
  repeatable edge would count it many times over.

What survives all three is the shortlist worth a full tick read.
"""
import argparse
import json
from collections import defaultdict
from decimal import Decimal
from pathlib import Path

from research_config import research_config

from src.exchange.univ3_math import notional_to_move_price
from src.research.observations import ObservationStore, mid_dislocation_bps
from src.research.report import classify_dislocation, group_key
from src.research.statistics import describe

config = research_config("WARNING")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default="data/screen.sqlite3")
    parser.add_argument("--targets", default="research/targets_wide.json")
    parser.add_argument("--taker-bps", type=float, default=7.5)
    parser.add_argument("--min-depth", type=float, default=5000.0,
                        help="minimum quote notional to move the price 1%%")
    parser.add_argument("--top", type=int, default=30)
    args = parser.parse_args()

    store = ObservationStore(args.db, run_id="analyse-screen")
    total = store.count()
    span = store.time_span()
    print(f"{args.db}: {total:,} observations", end="")
    if span:
        print(f" over {(span[1] - span[0]) / 3600:.2f}h")
    else:
        print()
    if not total:
        print("nothing to analyse yet")
        return

    targets = {
        (t["cex_symbol"], t["chain"], t["fee"]): t
        for t in json.loads(Path(args.targets).read_text(encoding="utf-8"))
    }

    dislocations = defaultdict(list)
    depths = defaultdict(list)
    no_liquidity = defaultdict(int)
    for observation in store.read_all():
        key = group_key(observation)
        target = targets.get(key)
        if target is None:
            continue
        base_is_token0 = (
            str(target["base_address"]).lower() < str(target["quote_address"]).lower()
        )
        value = mid_dislocation_bps(observation, base_is_token0)
        if value is None:
            no_liquidity[key] += 1
            continue
        dislocations[key].append(value)
        depths[key].append(notional_to_move_price(observation.pool, Decimal("0.01")))

    dead = [k for k in targets if k in no_liquidity and k not in dislocations]
    print(f"\n{len(dislocations)} markets with a tradeable price, "
          f"{len(dead)} pools with no active liquidity at all "
          f"(existence is not a market)")

    rows = []
    for key, values in dislocations.items():
        stats = describe([abs(v) for v in values])
        signed = describe(values)
        kind = classify_dislocation([float(v) for v in values])
        depth = describe(depths[key]) if depths[key] else {"p50": 0.0}
        floor = float(key[2]) / 100.0 + args.taker_bps
        rows.append({
            "key": key,
            "n": len(values),
            "median_abs": stats.get("p50") or 0.0,
            "p99_abs": stats.get("p99") or 0.0,
            "max_abs": stats.get("max") or 0.0,
            "mean_signed": signed.get("mean") or 0.0,
            "kind": kind["kind"],
            "flip": kind.get("sign_flip_fraction"),
            "depth_1pct": float(depth.get("p50") or 0.0),
            "floor": floor,
        })

    tradeable_depth = [r for r in rows if r["depth_1pct"] >= args.min_depth]
    too_thin = [r for r in rows if r["depth_1pct"] < args.min_depth]
    print(f"{len(tradeable_depth)} markets can absorb {args.min_depth:,.0f} of quote "
          f"within a 1% price move; {len(too_thin)} cannot")

    # The actual question.
    clears = [r for r in tradeable_depth if r["median_abs"] > r["floor"]]
    fluctuating = [r for r in clears if r["kind"] == "fluctuating"]

    print()
    print("=" * 104)
    print("THE QUESTION: a dislocation that BOTH exceeds its cost floor AND changes sign")
    print("=" * 104)
    print(f"  markets with real depth:                        {len(tradeable_depth)}")
    print(f"  ... whose median |dislocation| clears the floor: {len(clears)}")
    print(f"  ... and which is fluctuating, not a standing basis: {len(fluctuating)}")
    if fluctuating:
        print("\n  CANDIDATES -- these deserve a full tick read:")
        for r in sorted(fluctuating, key=lambda r: -r["median_abs"]):
            k = r["key"]
            print(f"    {k[0]:<14} {k[1]:<9} {k[2]:>5}  median |d| "
                  f"{r['median_abs']:>8.1f} bps vs floor {r['floor']:>5.1f}  "
                  f"flip {r['flip']:.1%}  depth {r['depth_1pct']:>12,.0f}  n={r['n']}")
    else:
        print("\n  NONE. Every market either cannot absorb size, or shows a dislocation")
        print("  below its own cost floor, or shows one that never changes sign.")

    print()
    print("=" * 104)
    print(f"TOP {args.top} BY MEDIAN |DISLOCATION|, among markets with real depth")
    print("=" * 104)
    print(f"{'market':<32} {'n':>5} {'med|d|':>8} {'p99|d|':>9} {'floor':>7} "
          f"{'clears':>7} {'kind':>15} {'depth 1%':>14}")
    for r in sorted(tradeable_depth, key=lambda r: -r["median_abs"])[:args.top]:
        k = r["key"]
        label = f"{k[0]} {k[1]} {k[2]}"
        print(f"{label:<32} {r['n']:>5} {r['median_abs']:>8.1f} {r['p99_abs']:>9.1f} "
              f"{r['floor']:>7.1f} {('YES' if r['median_abs'] > r['floor'] else 'no'):>7} "
              f"{r['kind']:>15} {r['depth_1pct']:>14,.0f}")

    print()
    print("Every figure is the RAW dislocation: pool mid against CEX mid, before pool")
    print("fee, taker fee, spread, impact or gas. It is an upper bound on any edge.")
    print("depth 1% is the quote notional that would move the pool price 1%, itself an")
    print("upper bound since it assumes no tick is crossed.")

    if too_thin:
        print(f"\n{len(too_thin)} markets excluded for depth below "
              f"{args.min_depth:,.0f}. Largest |dislocation| among them:")
        for r in sorted(too_thin, key=lambda r: -r["median_abs"])[:8]:
            k = r["key"]
            print(f"  {k[0]:<14} {k[1]:<9} {k[2]:>5}  median |d| "
                  f"{r['median_abs']:>10.1f} bps  depth {r['depth_1pct']:>12,.2f}  "
                  f"-- a big number on a pool that cannot fill it")


main()
