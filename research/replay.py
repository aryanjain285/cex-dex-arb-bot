"""Replay recorded observations through the production detector.

Complements `analyse.py` rather than duplicating it. `analyse.py` asks whether an
opportunity existed, using the research optimiser. This asks whether the SHIPPED code
path would have found it, using the actual detector, and the two can disagree -- which is
the only reason to run both. A market conclusion says nothing about whether the code
agrees, and the code is what would trade.

Reports counts, not PnL. A count of opportunities is a joint statement about the code and
the market. A PnL would additionally require a fill assumption, and a dataset of quotes
contains no evidence for one.
"""
import argparse
import asyncio
from collections import defaultdict
from decimal import Decimal

from research_config import research_config

from backtest.observation_replay import replay_store
from src.research.observations import ObservationStore
from src.research.report import group_key

config = research_config("WARNING")


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default="data/observations.sqlite3")
    parser.add_argument("--notional", type=float, default=1000.0)
    parser.add_argument("--min-net-bps", type=float, default=None)
    parser.add_argument("--taker-bps", type=float, default=None)
    parser.add_argument("--inject-bps", type=float, default=None,
                        help="POSITIVE CONTROL: shift the recorded CEX book by this "
                             "many bps and check the detector finds the edge")
    args = parser.parse_args()

    taker = Decimal(str(
        args.taker_bps if args.taker_bps is not None else config.strategy.taker_fee_bps
    ))
    floor = Decimal(str(
        args.min_net_bps if args.min_net_bps is not None
        else config.strategy.min_net_bps
    ))
    gas_units = config.dex.swap_gas_estimate_units

    store = ObservationStore(args.db, run_id="replay")
    total = store.count()
    print(f"{args.db}: {total:,} observations")
    if not total:
        return
    print(f"detector settings: target notional {args.notional:,.0f}, "
          f"floor {floor} bps, taker {taker} bps, gas {gas_units} units\n")

    if args.inject_bps is not None:
        await positive_control(
            store, gas_units, taker, Decimal(str(args.notional)), floor,
            Decimal(str(args.inject_bps)),
        )
        return

    markets = sorted({group_key(o) for o in store.read_all()})
    print(f"{'market':<34} {'obs':>7} {'evaluated':>10} {'uncostable':>11} "
          f"{'opportunities':>14} {'best net bps':>13}")
    grand = defaultdict(int)
    for symbol, chain, fee in markets:
        result = await replay_store(
            store, gas_units=gas_units, taker_fee_bps=taker,
            target_notional=Decimal(str(args.notional)),
            min_net_bps=floor, cex_symbol=symbol, chain=chain, pool_fee=fee,
        )
        best = result["best_net_bps"]
        label = f"{symbol} {chain} {fee}"
        print(f"{label:<34} {result['observations']:>7,} "
              f"{result['evaluated']:>10,} {result['uncostable']:>11,} "
              f"{result['opportunities']:>14,} "
              f"{('-' if best is None else f'{float(best):.2f}'):>13}")
        for key in ("observations", "evaluated", "uncostable", "opportunities"):
            grand[key] += result[key]
        for direction, count in result["by_direction"].items():
            grand[f"dir:{direction}"] += count

    print()
    print(f"total: {grand['observations']:,} observations, "
          f"{grand['evaluated']:,} evaluated, "
          f"{grand['uncostable']:,} uncostable, "
          f"{grand['opportunities']:,} opportunities")
    directions = {k[4:]: v for k, v in grand.items() if k.startswith("dir:")}
    if directions:
        print(f"by direction: {directions}")
    if grand["evaluated"] and not grand["opportunities"]:
        print()
        print("The production detector found nothing across every evaluated")
        print("observation. With the research stack independently reporting a raw")
        print("dislocation several times below the cost floor, the two agree -- which")
        print("is the point of running both. A disagreement would mean the shipped")
        print("code and the market model differ, and that would need resolving before")
        print("either could be trusted.")
    print()
    print("Counts, not PnL. A count is a joint statement about the code and the")
    print("market; a PnL needs a fill assumption, and a dataset of quotes has no")
    print("evidence for one.")


async def positive_control(store, gas_units, taker, notional, floor, inject_bps):
    """Shift the recorded CEX book and check the detector finds the resulting edge.

    "Found nothing" and "cannot find anything" produce identical output, and only one of
    them is a statement about the market. A synthetic fixture cannot settle it either:
    the pipeline that runs on real data includes the recorded ladders, the recorded pool
    ticks, the recorded gas prices and every decimals convention in the set, and a
    fixture exercises none of those.

    So: take real observations, move the CEX side by a known amount, and require the
    detector to find an edge of about that size less the fees. If it does, a zero on
    unshifted data means the market, not the harness.
    """
    from dataclasses import replace as dc_replace

    from backtest.observation_replay import (
        ObservationReplayCex, ObservationReplayDex, build_market_pair,
    )
    from src.core.config import StrategyConfig
    from src.strategy.detector import OpportunityDetector

    print(f"POSITIVE CONTROL: shifting every recorded CEX book by {inject_bps} bps")
    print()
    scale = Decimal(1) + inject_bps / Decimal(10000)

    print(f"{'market':<34} {'evaluated':>10} {'found':>8} {'median net bps':>15}")
    total_found = total_evaluated = 0
    from src.research.report import group_key as _gk

    markets = sorted({_gk(o) for o in store.read_all()})
    for symbol, chain, fee in markets:
        observations = [
            o for o in store.read_all(cex_symbol=symbol)
            if o.chain == chain and int(o.pool_fee) == int(fee)
        ]
        if not observations:
            continue
        pair = build_market_pair(observations[0])
        cex = ObservationReplayCex(pair)
        dex = ObservationReplayDex(pair, gas_units=gas_units)
        strategy = StrategyConfig(
            target_notional_usd=int(notional), min_net_bps=floor,
            max_net_bps_sanity=Decimal("10000"), taker_fee_bps=taker,
            opportunity_ttl_seconds=30, loop_interval_seconds=1.0,
            intermediate_price_cache_seconds=5.0, max_book_age_seconds=30.0,
            error_backoff_seconds=1.0, shutdown_drain_seconds=1.0,
            max_consecutive_errors=10,
        )
        detector = OpportunityDetector(strategy, cex, dex, [pair])

        edges, evaluated, found = [], 0, 0
        errors = defaultdict(int)
        # Capture the detector's own rejection reasons. Without these, "found nothing"
        # gives no way to tell a market fact from a gate nobody remembered was there --
        # and the detector records the distinction precisely so it need not be guessed.
        reasons = defaultdict(int)
        original_emit = detector._emit

        def capture(pair_, ev, floor=None, taken=False):
            if getattr(ev, "reason", None):
                reasons[ev.reason] += 1
            return original_emit(pair_, ev, floor=floor, taken=taken)

        detector._emit = capture
        for observation in observations:
            if observation.gas_quote(gas_units) is None:
                continue
            # Shift both sides of the book by the same factor, so the spread is
            # unchanged and only the level moves. Shifting one side would widen the
            # spread and confound the test with a depth effect.
            shifted = dc_replace(
                observation,
                cex_bids=[(p * scale, s) for p, s in observation.cex_bids],
                cex_asks=[(p * scale, s) for p, s in observation.cex_asks],
            )
            cex.set_observation(shifted)
            dex.set_observation(shifted)
            evaluated += 1
            try:
                opportunities = await detector.detect()
            except Exception as exc:
                # Surfaced, not swallowed. A control that silently skips every
                # observation reports zero and reads as "no opportunity" -- which is
                # precisely the confusion the control exists to remove.
                errors[f"{type(exc).__name__}: {exc}"] += 1
                continue
            for opportunity in opportunities:
                found += 1
                edge = getattr(opportunity, "edge_bps", None)
                if edge is not None:
                    edges.append(float(edge))

        total_found += found
        total_evaluated += evaluated
        median = sorted(edges)[len(edges) // 2] if edges else None
        label = f"{symbol} {chain} {fee}"
        print(f"{label:<34} {evaluated:>10,} {found:>8,} "
              f"{('-' if median is None else f'{median:.1f}'):>15}")
        for message, count in sorted(errors.items(), key=lambda kv: -kv[1])[:2]:
            print(f"      ! {count:,}x {message[:110]}")
        if not found and reasons:
            top = sorted(reasons.items(), key=lambda kv: -kv[1])[:3]
            print("      rejected: "
                  + ", ".join(f"{reason} x{count:,}" for reason, count in top))

    print()
    if total_found:
        print(f"The detector found {total_found:,} opportunities in "
              f"{total_evaluated:,} shifted observations. So a zero on unshifted data")
        print("is a statement about the market, not about the harness.")
    else:
        print(f"The detector found NOTHING even with {inject_bps} bps injected into")
        print(f"{total_evaluated:,} real observations. That is a defect in the pipeline,")
        print("and every zero it has reported is uninterpretable until it is found.")


asyncio.run(main())
