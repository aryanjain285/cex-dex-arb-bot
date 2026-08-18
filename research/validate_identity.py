"""An arithmetic identity that the whole pricing chain has to satisfy.

Buy base on the CEX and sell it on the DEX, then do the reverse, both at the same
instant and the same size. The two round trips together pay the pool fee twice, both
half-spreads, and both price impacts, and collect the dislocation once in each
direction -- where it cancels. So:

    gross(CEX_to_DEX) + gross(DEX_to_CEX)  ~  -(2 * pool_fee + cex_spread + impact)

This is not a property of the market. It is a property of arithmetic, so it must hold
on every observation regardless of what prices did. It exercises the entire chain at
once -- tick math, token ordering, decimals, book walking, both directions' unit
conversions -- and it is exactly the check that would have caught the buy-leg units
defect, the inverted DEX price, and a wrong base_is_token0, none of which a
plausible-looking table of basis points reveals.

The residual is reported per market rather than pooled: a single market breaking the
identity is a token-ordering bug in that pool, while all of them breaking it is a
bug in the shared math.
"""
import json
import statistics
from decimal import Decimal
from pathlib import Path

from research_config import research_config

from src.research.observations import ObservationStore
from src.research.optimiser import optimise_size
from src.research.report import group_key

config = research_config("WARNING")

STORE = "data/observations.sqlite3"
TARGETS = Path("research/targets.json")
NOTIONAL = Decimal("1000")
TAKER_BPS = Decimal("7.5")


def main():
    store = ObservationStore(STORE, run_id="validate")
    targets = {
        (t["cex_symbol"], t["chain"], t["fee"]): t
        for t in json.loads(TARGETS.read_text(encoding="utf-8"))
    }

    by_market = {}
    for observation in store.read_all():
        by_market.setdefault(group_key(observation), []).append(observation)

    print(f"{'market':<30} {'n':>4} {'median residual':>16} {'expected':>10} "
          f"{'error bps':>10}")
    worst = []
    for key in sorted(by_market):
        target = targets.get(key)
        if target is None:
            continue
        observations = by_market[key]
        base_is_token0 = (
            observations[0].pool.token0.lower()
            == str(target["base_address"]).lower()
        )
        pool_fee_bps = Decimal(key[2]) / Decimal(100)

        residuals, spreads = [], []
        for observation in observations:
            gas = observation.gas_quote(200_000)
            if gas is None:
                continue
            legs = {}
            for direction in ("CEX_to_DEX", "DEX_to_CEX"):
                curve = optimise_size(
                    pool=observation.pool, direction=direction,
                    cex_bids=observation.cex_bids, cex_asks=observation.cex_asks,
                    notionals=[NOTIONAL], taker_fee_bps=TAKER_BPS,
                    gas_quote=gas, base_is_token0=base_is_token0,
                    floor_bps=Decimal(0),
                )
                point = curve.curve[0]
                if point.gross_bps is None:
                    legs = {}
                    break
                legs[direction] = point.gross_bps
            if len(legs) != 2:
                continue
            residuals.append(float(legs["CEX_to_DEX"] + legs["DEX_to_CEX"]))
            bid, ask = observation.cex_bids[0][0], observation.cex_asks[0][0]
            spreads.append(float((ask - bid) / ((ask + bid) / 2) * 10000))

        if not residuals:
            print(f"  {str(key):<28} {'-':>4}  no costable observations")
            continue

        median_residual = statistics.median(residuals)
        median_spread = statistics.median(spreads)
        # Expected: two pool fees plus one CEX spread. Impact is extra and always
        # makes the residual MORE negative, so the identity is a one-sided bound:
        # the residual must not be ABOVE the expectation.
        expected = -(2 * float(pool_fee_bps) + median_spread)
        error = median_residual - expected
        label = f"{key[0]} {key[1]} {key[2]}"
        print(f"  {label:<28} {len(residuals):>4} {median_residual:>16.2f} "
              f"{expected:>10.2f} {error:>10.2f}")
        worst.append((abs(error), label, median_residual, expected))

    print()
    if not worst:
        print("nothing to validate yet")
        return
    worst.sort(reverse=True)
    print("The residual must never exceed the expectation: impact only subtracts.")
    violations = [w for w in worst if w[2] > w[3] + 0.5]
    if violations:
        print(f"VIOLATIONS ({len(violations)}): a residual above the fee floor means "
              f"the two directions are not pricing the same market.")
        for _, label, residual, expected in violations:
            print(f"  {label}: residual {residual:.2f} > expected {expected:.2f}")
    else:
        print(f"No violations across {len(worst)} markets. The largest gap between "
              f"residual and fee floor is {worst[0][0]:.2f} bps ({worst[0][1]}), "
              f"which is price impact at {NOTIONAL} notional.")


main()
