"""Turn recorded observations into the report.

Read-only against the store, safe to run while the recorder is still writing (WAL).

Every number is printed with the assumptions that produced it, and the refusal counts
are printed alongside the statistics rather than under them: "no edge found" and
"could not be evaluated" have opposite implications, and a report that blurs them is
worse than no report.
"""
import argparse
import json
from decimal import Decimal
from pathlib import Path

from research_config import research_config

from src.research.evaluate import CostModel
from src.research.observations import ObservationStore
from src.research.optimiser import geometric_size_grid
from src.research.report import (
    analyse_store,
    format_report,
    format_summary_table,
    group_key,
    scrambled_control,
)
from src.research.statistics import decay_profile, half_life_seconds

config = research_config("WARNING")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default="data/observations.sqlite3")
    parser.add_argument("--targets", default="targets.json")
    parser.add_argument("--min-notional", type=float, default=100.0)
    parser.add_argument("--max-notional", type=float, default=100_000.0)
    parser.add_argument("--points", type=int, default=10)
    parser.add_argument("--probe", type=float, default=1000.0)
    parser.add_argument("--control-offset", type=float, default=1800.0)
    parser.add_argument("--json-out", default=None)
    args = parser.parse_args()

    store = ObservationStore(args.db, run_id="analysis")
    total = store.count()
    span = store.time_span()
    print(f"store: {total:,} observations", end="")
    if span:
        print(f" over {(span[1] - span[0]) / 3600:.2f}h")
    else:
        print()
    if not total:
        print("nothing to analyse yet")
        return

    notionals = geometric_size_grid(
        Decimal(str(args.min_notional)), Decimal(str(args.max_notional)), args.points
    )
    costs = CostModel(
        taker_fee_bps=Decimal(str(config.strategy.taker_fee_bps)),
        cex_legs=1,
        gas_units=config.dex.swap_gas_estimate_units,
        rotation_cost_quote=Decimal("0"),
        floor_bps=Decimal(str(config.strategy.min_net_bps)),
    )
    print(f"\ncost model: taker {costs.taker_fee_bps} bps x{costs.cex_legs}, "
          f"gas {costs.gas_units} units, floor {costs.floor_bps} bps, "
          f"rotation {costs.rotation_cost_quote}")
    print(f"notional grid: {[f'{float(n):,.0f}' for n in notionals]}")
    print("NOTE: rotation cost is set to zero here. That is deliberate for a "
          "research read -- it isolates the market from the inventory model -- and "
          "it means every net figure below is an UPPER bound on what a rotating "
          "strategy would earn.\n")

    # Which side of each pool the base token sits on. Read from the recorded
    # snapshot rather than assumed: address ordering differs per pool, and getting it
    # wrong inverts the price by a factor of price squared.
    targets = {
        (t["cex_symbol"], t["chain"], t["fee"]): t
        for t in json.loads(Path(args.targets).read_text(encoding="utf-8"))
    }

    reports = []
    for key in sorted({group_key(o) for o in store.read_all()}):
        target = targets.get(key)
        if target is None:
            print(f"skipping {key}: not in the target list, so token order is unknown")
            continue
        sample = next(iter(store.read_all(cex_symbol=key[0], limit=1)), None)
        base_is_token0 = None
        for observation in store.read_all(cex_symbol=key[0]):
            if group_key(observation) == key:
                base_is_token0 = (
                    observation.pool.token0.lower()
                    == str(target["base_address"]).lower()
                )
                break
        if base_is_token0 is None:
            continue

        market = analyse_store(
            store, costs, notionals,
            base_is_token0=base_is_token0,
            probe_notional=Decimal(str(args.probe)),
            since=None, until=None,
        )
        for report in market:
            if (report.cex_symbol, report.chain, report.pool_fee) == key:
                reports.append((report, base_is_token0))

    seen = set()
    for report, base_is_token0 in reports:
        key = (report.cex_symbol, report.chain, report.pool_fee)
        if key in seen:
            continue
        seen.add(key)
        print(format_report(report))

        observations = [
            o for o in store.read_all(cex_symbol=report.cex_symbol)
            if group_key(o) == key
        ]
        try:
            control = scrambled_control(
                observations, costs, notionals,
                base_is_token0=base_is_token0,
                offset_seconds=args.control_offset,
            )
        except ValueError as exc:
            print(f"  control: not run ({exc})")
            print()
            continue
        true_p99 = control["true"].get("p99")
        print(f"  CONTROL   true p99 {_f(true_p99)} bps vs scrambled p99 "
              f"{_f(control['noise_bound_bps'])} bps "
              f"(offset {control['offset_seconds']:.0f}s, "
              f"{control['identical_pairs']} identical pairs)")
        if control.get("constant_offset"):
            print("            FIXED OFFSET: the gap's own range is under a quarter of")
            print("            its size, so it is a constant difference between the two")
            print("            things compared -- a bridged representation, a different")
            print("            asset, or a peg difference -- not a tradeable signal.")
        elif control.get("control_has_power") is False:
            print("            CONTROL HAS NO POWER on this pair: scrambling can only add")
            print("            the venue's own movement, and this level barely moves over")
            print("            the offset. Says nothing about the gap either way.")
        elif control["exceeds_noise"] is False:
            print("            the real tail is NOT heavier than a time-scrambled "
                  "one: apparent edge at this level is indistinguishable from noise")
        elif control["exceeds_noise"] is True:
            print("            the real tail IS heavier than a time-scrambled one")
        print()

    if reports:
        print("=" * 96)
        print("CROSS-MARKET SUMMARY")
        print("=" * 96)
        print(format_summary_table([r for r, _ in reports]))
        print()
        print("=" * 96)
        print("HOW FAST AN EDGE DECAYS -- the latency budget")
        print("=" * 96)
        print("An edge whose correlation is gone by 12s cannot be captured by a")
        print("system that settles on a 12s block, whatever its cost model.")
        print()
        print(f"{'market':<30} {'cadence':>8} {'half-life':>10}  correlation at lag")
        for report, base_is_token0 in reports:
            key = (report.cex_symbol, report.chain, report.pool_fee)
            cadence = report.median_cadence_seconds
            if not cadence:
                continue
            series = _gross_series(store, costs, notionals, key, base_is_token0)
            if len(series) < 30:
                continue
            profile = decay_profile(series, cadence)
            half = half_life_seconds(series, cadence)
            label = f"{report.cex_symbol} {report.chain} {report.pool_fee}"
            cells = "  ".join(
                f"{int(lag)}s:{('-' if rho is None else f'{rho:+.2f}')}"
                for lag, rho in sorted(profile.items())
            )
            print(f"{label:<30} {cadence:>7.1f}s "
                  f"{('-' if half is None else f'{half:.1f}s'):>10}  {cells}")
        print()

    if args.json_out:
        payload = [
            {
                "pair": r.cex_symbol, "chain": r.chain, "fee": r.pool_fee,
                "observations": r.observations, "uncostable": r.uncostable,
                "unpriceable": r.unpriceable,
                "span_seconds": r.span_seconds,
                "cadence_seconds": r.median_cadence_seconds,
                "gross": r.gross_bps, "gross_interval": r.gross_interval,
                "net": r.net_bps,
                "exceedance_net": {str(k): v for k, v in r.exceedance_net.items()},
                "tradeable": r.tradeable_observations,
                "median_lifetime_seconds": r.median_lifetime_seconds,
                "probe_understatement_bps": r.probe_understatement_bps,
                "latency": {str(k): v for k, v in r.latency.items()},
            }
            for r, _ in reports
        ]
        Path(args.json_out).write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"json written to {args.json_out}")


def _gross_series(store, costs, notionals, key, base_is_token0):
    """The best-gross series for one market, for the decay analysis.

    Recomputed rather than carried out of the report because the report keeps
    distributions, not the ordered series -- and order is the entire content of an
    autocorrelation.
    """
    from src.research.evaluate import evaluate_observation

    series = []
    for observation in store.read_all(cex_symbol=key[0]):
        if group_key(observation) != key:
            continue
        result = evaluate_observation(
            observation, costs, notionals, base_is_token0=base_is_token0
        )
        if result.best_gross_bps is not None:
            series.append(float(result.best_gross_bps))
    return series


def _f(value, spec=".2f"):
    return "-" if value is None else format(value, spec)


main()
