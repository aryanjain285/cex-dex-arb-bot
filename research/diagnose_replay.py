"""Why does the replay find nothing on some markets even with 200 bps injected?

At 200 bps of injected dislocation, 9 of 22 markets returned zero opportunities. Cost
arithmetic cannot explain that: 200 less a 30 bps pool fee, 7.5 bps taker and ~19 bps of
Ethereum gas at $1,000 leaves 143 bps against a 5 bps floor. So the replay has a defect,
and a positive control that itself has a defect is worse than no control -- it converts
"the harness works" into a false claim.

Prints, per market, every intermediate the detector depends on: the derived token order,
the raw spot, the CEX mid, and a quote in each direction at the target size. Whichever
one is wrong will be obvious against the others.
"""
import argparse
import asyncio
from collections import defaultdict
from decimal import Decimal

from research_config import research_config

from backtest.observation_replay import (
    ObservationReplayCex,
    ObservationReplayDex,
    _base_is_token0,
    build_market_pair,
)
from src.research.observations import ObservationStore
from src.research.report import group_key

config = research_config("WARNING")


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default="data/observations.sqlite3")
    parser.add_argument("--notional", type=float, default=1000.0)
    parser.add_argument("--inject-bps", type=float, default=200.0)
    args = parser.parse_args()

    store = ObservationStore(args.db, run_id="diagnose")
    latest = {}
    for observation in store.read_all():
        latest[group_key(observation)] = observation

    scale = Decimal(1) + Decimal(str(args.inject_bps)) / Decimal(10000)
    notional = Decimal(str(args.notional))
    gas_units = config.dex.swap_gas_estimate_units

    print(f"{'market':<30} {'tok0?':>6} {'spot':>16} {'cex mid':>12} "
          f"{'size base':>12} {'dex sell':>14} {'dex buy':>14} {'gas bps':>8}")
    for key in sorted(latest):
        observation = latest[key]
        from dataclasses import replace as dc_replace
        shifted = dc_replace(
            observation,
            cex_bids=[(p * scale, s) for p, s in observation.cex_bids],
            cex_asks=[(p * scale, s) for p, s in observation.cex_asks],
        )
        pair = build_market_pair(shifted)
        dex = ObservationReplayDex(pair, gas_units=gas_units)
        dex.set_observation(shifted)

        base_is_token0 = _base_is_token0(shifted)
        spot = observation.pool.spot_price()
        mid = shifted.cex_mid
        size = notional / mid if mid else Decimal(0)

        sell = await dex.get_quote(pair, size, "sell")
        buy = await dex.get_quote(pair, size, "buy")
        gas = observation.gas_quote(gas_units)
        gas_bps = (gas / notional * 10000) if gas else None

        label = f"{key[0]} {key[1]} {key[2]}"
        print(f"{label:<30} {str(base_is_token0):>6} {float(spot):>16.8f} "
              f"{float(mid):>12.4f} {float(size):>12.6f} "
              f"{('-' if sell is None else f'{float(sell.price):.4f}'):>14} "
              f"{('-' if buy is None else f'{float(buy.price):.4f}'):>14} "
              f"{('-' if gas_bps is None else f'{float(gas_bps):.1f}'):>8}")

    print()
    print("Read it like this: `spot` is token0-in-token1 straight from the pool, so for")
    print("a pool where the base is token1 it is the RECIPROCAL of the price. `dex sell`")
    print("and `dex buy` are quote-per-base and must both sit near the CEX mid. A dash")
    print("means the pool refused the size. A quote far from the mid means the token")
    print("order was derived wrongly, and the detector will then reject the resulting")
    print("edge as insane -- which looks exactly like finding no opportunity.")


asyncio.run(main())
