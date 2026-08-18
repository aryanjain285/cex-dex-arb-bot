"""What CEX fee would make this work? The one lever the measurements point at.

The structural finding: the post-swap residual dislocation equals the POOL FEE. Measured
across four pool/tier combinations on Arbitrum, with exact block timestamps and 1-second
klines:

    ETH/USDC 0.05%   pool fee  5 bps  ->  median |dislocation|  4.69 bps
    ETH/USDT 0.05%   pool fee  5 bps  ->  median |dislocation|  4.70 bps
    ETH/USDC 0.30%   pool fee 30 bps  ->  median |dislocation| 22.38 bps
    ETH/USDT 0.30%   pool fee 30 bps  ->  median |dislocation| 30.30 bps

That follows from what an arbitrageur does: close a gap until the remainder no longer
covers the cost, and the irreducible cost of trading through a pool is that pool's fee. So
the market is arbitraged down to the pool fee, leaving the competition's cost basis on the
table.

Which makes the deficit structural. Available is about the pool fee; required is the pool
fee plus the CEX fee plus gas; so the shortfall is about the CEX fee, at every tier. Moving
to a 0.30% pool raises both sides by the same 25 bps, which is why fee-tier selection --
the obvious lever -- does nothing.

The lever that does something is the CEX fee. This computes the number precisely: the
highest CEX fee at which each market clears, at each percentile of its own dislocation
distribution. Gas is included from the recorded gas prices rather than assumed.

TWO DISTRIBUTIONS, and the difference matters more than either number. Clock-sampled
observations are what a polling loop sees. Swap prints are what exists immediately after a
trade, before whoever closes it arrives. The second is larger and is contested at
block-level priority, so a strategy that can only poll must be judged against the first.
"""
import argparse
import json
import statistics
from collections import defaultdict
from decimal import Decimal
from pathlib import Path

from research_config import research_config

from src.research.observations import ObservationStore, mid_dislocation_bps
from src.research.report import group_key

config = research_config("WARNING")

# Binance spot schedule, in bps, so the answer can be read as a VIP tier rather than an
# abstract number. Regular tiers; BNB discount applies on top.
BINANCE_TIERS = [
    ("VIP0", 10.0, 10.0),
    ("VIP1", 9.0, 10.0),
    ("VIP2", 8.0, 10.0),
    ("VIP3", 4.2, 6.0),
    ("VIP4", 4.2, 5.4),
    ("VIP5", 3.6, 4.8),
    ("VIP6", 2.0, 4.0),
    ("VIP7", 1.6, 3.0),
    ("VIP8", 1.2, 2.4),
    ("VIP9", 0.6, 2.0),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default="data/observations_fast.sqlite3")
    parser.add_argument("--targets", default="research/targets_fast.json")
    parser.add_argument("--gas-units", type=int, default=200_000)
    parser.add_argument("--legs", type=int, default=1)
    args = parser.parse_args()

    store = ObservationStore(args.db, run_id="feesens")
    if not store.count():
        print("nothing recorded")
        return
    targets = {
        (t["cex_symbol"], t["chain"], t["fee"]): t
        for t in json.loads(Path(args.targets).read_text(encoding="utf-8"))
    }

    series = defaultdict(list)
    gas_bps = defaultdict(list)
    priceable = defaultdict(list)
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
            continue
        series[key].append(abs(float(value)))
        # Can the pool actually price the target notional? A raw mid-to-mid gap is not
        # tradeable if no size fits, and thin pools produce LARGE gaps precisely because
        # nobody arbitrages a pool they cannot trade in. Without this, ARB/USDC 0.05%
        # reports 11.36 bps of headroom on a pool measured at -1,184 bps of impact at
        # $1,000.
        priceable[key].append(
            observation.pool.price_for_amount_in(
                Decimal("1000") / (observation.cex_mid or Decimal(1)),
                zero_for_one=base_is_token0,
            ) is not None
        )
        gas = observation.gas_quote(args.gas_units)
        if gas is not None:
            # As a fraction of a $1,000 notional, which is the configured target size.
            gas_bps[key].append(float(gas / Decimal("1000") * 10_000))

    print(f"CLOCK-SAMPLED dislocation from {args.db}")
    print("This is what a polling loop can see. Swap prints show more and are contested")
    print("at block-level priority; see exact_swap_dislocation.py.")
    print()
    for key in sorted(series):
        values = sorted(series[key])
        if len(values) < 50:
            continue
        pool_fee_bps = key[2] / 100.0
        gas = statistics.median(gas_bps[key]) if gas_bps[key] else 0.0
        label = f"{key[0]} {key[1]} {key[2]}"
        print(f"=== {label} ===")
        fillable = (
            sum(priceable[key]) / len(priceable[key]) if priceable[key] else 0.0
        )
        print(f"  {len(values):,} observations, pool fee {pool_fee_bps:.2f} bps, "
              f"gas {gas:.2f} bps at $1,000, "
              f"{fillable:.0%} of observations can price $1,000")
        if fillable < 0.5:
            print(f"  NOT TRADEABLE: this pool cannot price the target notional in "
                  f"{1 - fillable:.0%} of observations, so its gap is a mid-to-mid")
            print(f"  number with no size behind it. A thin pool shows a LARGE gap "
                  f"precisely because nobody arbitrages a pool they cannot trade in.")
        print(f"  {'percentile':<12} {'|dislocation|':>14} {'max CEX fee that clears':>26}")
        for name, p in (("median", 0.50), ("p75", 0.75), ("p90", 0.90),
                        ("p99", 0.99), ("max", 1.0)):
            index = min(len(values) - 1, int(len(values) * p))
            dislocation = values[index]
            # available = pool fee + CEX fee*legs + gas  ->  solve for the CEX fee
            headroom = dislocation - pool_fee_bps - gas
            per_leg = headroom / args.legs
            if per_leg <= 0:
                verdict = "none: below the pool fee alone"
            else:
                affordable = [
                    tier for tier, maker, taker in BINANCE_TIERS if maker <= per_leg
                ]
                cheapest = affordable[0] if affordable else None
                verdict = f"{per_leg:.2f} bps"
                if cheapest:
                    verdict += f"  (>= {cheapest} maker)"
                else:
                    verdict += "  (below every Binance tier)"
            print(f"  {name:<12} {dislocation:>14.2f} {verdict:>26}")
        print()

    print("=" * 84)
    print("READ IT THIS WAY")
    print("=" * 84)
    print("The right-hand column is the largest per-leg CEX fee at which that percentile")
    print("of the dislocation would have been profitable. The configured taker fee is")
    print(f"{config.strategy.taker_fee_bps} bps.")
    print()
    print("Binance spot maker/taker, bps, before the BNB discount:")
    for tier, maker, taker in BINANCE_TIERS:
        print(f"  {tier:<6} maker {maker:>5.1f}   taker {taker:>5.1f}")
    print()
    print("A taker strategy needs the dislocation to exceed the pool fee plus the taker")
    print("fee, and the dislocation IS the pool fee. So the requirement is a CEX fee near")
    print("zero -- which on this schedule means maker orders at a high VIP tier, and maker")
    print("orders on the CEX leg change the strategy: a resting order is not a hedge until")
    print("it fills, so the DEX leg would carry unhedged inventory for as long as it takes.")
    print("That risk is not priced anywhere in this codebase and would have to be before")
    print("the maker route could be evaluated, not after.")


main()
