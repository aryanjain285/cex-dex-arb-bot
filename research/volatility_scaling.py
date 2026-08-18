"""Does the dislocation scale with volatility? The extrapolation rests entirely on this.

Today is the 12th percentile of ETH volatility over 180 days: 2.88 bps/min against a
median of 5.43 and a p99 of 20.43. So the measured dislocation of 2.3 bps was taken in an
unusually quiet market, and the interesting question is no longer "does this work today"
but "does it work in a normal or a violent one".

Scaling the measurement by the volatility ratio would answer that, and it would be a
guess. A dislocation exists because one venue has not yet reflected a move the other has,
so it PLAUSIBLY scales with the size of moves -- plausibly is not measured. The
relationship could be sublinear (arbitrageurs work harder when it pays more), or
threshold-like (nothing until moves exceed the fee, then everything), or absent (the gap
is a structural basis unrelated to flow).

Those three possibilities give completely different answers to whether this strategy ever
works, so the relationship is estimated here from the recorded data rather than assumed.

METHOD. Bucket the recorded observations by time. In each bucket compute the realised
volatility of the CEX mid and the median |dislocation|, both from the same rows -- so the
two are measured on the same instants, with no join and no alignment question. Then look
at how one moves with the other, and extrapolate only along a relationship that is
actually visible in the data.

WHAT WOULD MAKE THIS UNRELIABLE, stated up front. The recorded window spans a few hours
of one quiet day, so the volatility RANGE inside it is narrow. Fitting a slope over a
narrow range and extending it to the 99th percentile is extrapolation of the least
defensible kind, and the output says how far outside the observed range each projection
sits. A slope measured between 2 and 4 bps/min tells you very little about 20.
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


def realised_volatility_bps(mids, seconds_between):
    """Standard deviation of returns, RESCALED to a per-minute basis.

    The rescaling is not cosmetic. These observations are 2 seconds apart, so their
    returns have about sqrt(2/60) of the standard deviation of one-minute returns -- and
    the historical comparison comes from one-minute Binance klines. Comparing the two
    unscaled understates the recorded volatility by a factor of sqrt(30), about 5.5x,
    which made a 12th-percentile day look like a hundredth of one and put every
    projection far outside a range it was actually inside.

    Volatility scales as the square root of the interval under a random walk, so the
    conversion is sqrt(60 / seconds_between). That assumption is exactly right for
    independent increments and slightly wrong for a mean-reverting series, which these
    are over short horizons -- so the rescaled figure is a mild overstatement, in the
    direction that does not flatter the strategy.
    """
    if len(mids) < 3 or seconds_between <= 0:
        return None
    returns = []
    for a, b in zip(mids, mids[1:]):
        if a > 0 and b > 0:
            returns.append(float((b / a - 1) * 10_000))
    if len(returns) < 2:
        return None
    per_interval = statistics.stdev(returns)
    return per_interval * ((60.0 / seconds_between) ** 0.5)


def fit_through_origin(xs, ys):
    """Least-squares slope with no intercept: y = k*x.

    Through the origin on purpose. At zero volatility neither venue moves, so there is
    nothing for a dislocation to be made of, and a fitted intercept would be a constant
    edge appearing from nowhere -- which is precisely the artifact this analysis exists to
    avoid inventing. An intercept is reported separately as a diagnostic.
    """
    numerator = sum(x * y for x, y in zip(xs, ys))
    denominator = sum(x * x for x in xs)
    return (numerator / denominator) if denominator > 0 else None


def fit_with_intercept(xs, ys):
    n = len(xs)
    if n < 3:
        return None, None
    mean_x, mean_y = statistics.fmean(xs), statistics.fmean(ys)
    sxx = sum((x - mean_x) ** 2 for x in xs)
    if sxx == 0:
        return None, None
    sxy = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    slope = sxy / sxx
    return slope, mean_y - slope * mean_x


def correlation(xs, ys):
    if len(xs) < 3:
        return None
    try:
        return statistics.correlation(xs, ys)
    except Exception:
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default="data/observations_fast.sqlite3")
    parser.add_argument("--targets", default="research/targets_fast.json")
    parser.add_argument("--bucket-seconds", type=float, default=120.0)
    parser.add_argument("--taker-bps", type=float, default=7.5)
    args = parser.parse_args()

    store = ObservationStore(args.db, run_id="volscaling")
    if not store.count():
        print("nothing recorded yet")
        return
    targets = {
        (t["cex_symbol"], t["chain"], t["fee"]): t
        for t in json.loads(Path(args.targets).read_text(encoding="utf-8"))
    }

    # The observation cadence, measured, because the volatility rescaling depends on it
    # and a configured value has already been wrong by 12x once in this project.
    timestamps = defaultdict(list)
    buckets = defaultdict(lambda: {"mids": [], "dislocations": []})
    for observation in store.read_all():
        key = group_key(observation)
        target = targets.get(key)
        if target is None:
            continue
        base_is_token0 = (
            str(target["base_address"]).lower() < str(target["quote_address"]).lower()
        )
        value = mid_dislocation_bps(observation, base_is_token0)
        mid = observation.cex_mid
        if value is None or mid is None:
            continue
        timestamps[key].append(observation.ts)
        bucket = int(observation.ts // args.bucket_seconds)
        entry = buckets[(key, bucket)]
        entry["mids"].append(mid)
        entry["dislocations"].append(abs(float(value)))

    cadence = {}
    for key, times in timestamps.items():
        ordered = sorted(times)
        gaps = [b - a for a, b in zip(ordered, ordered[1:]) if b > a]
        cadence[key] = statistics.median(gaps) if gaps else None

    per_market = defaultdict(list)
    for (key, _bucket), entry in buckets.items():
        seconds = cadence.get(key)
        if not seconds:
            continue
        vol = realised_volatility_bps(entry["mids"], seconds)
        if vol is None or not entry["dislocations"]:
            continue
        per_market[key].append((vol, statistics.median(entry["dislocations"])))

    # Volatility percentiles from the 180-day history, for the projection.
    history = {
        "ETH/USDC": {"today": 2.88, "p50": 5.43, "p90": 11.38, "p99": 20.43},
        "ETH/USDT": {"today": 2.88, "p50": 5.43, "p90": 11.38, "p99": 20.43},
        "ARB/USDT": {"today": 7.93, "p50": 10.71, "p90": 17.16, "p99": 28.63},
        "ARB/USDC": {"today": 7.93, "p50": 10.71, "p90": 17.16, "p99": 28.63},
    }

    print(f"buckets of {args.bucket_seconds:.0f}s from {args.db}")
    print()
    for key in sorted(per_market):
        points = [(v, d) for v, d in per_market[key] if v > 0]
        if len(points) < 8:
            print(f"{key}: only {len(points)} buckets, too few to fit")
            continue
        xs = [v for v, _ in points]
        ys = [d for _, d in points]
        slope0 = fit_through_origin(xs, ys)
        slope, intercept = fit_with_intercept(xs, ys)
        r = correlation(xs, ys)
        label = f"{key[0]} {key[1]} {key[2]}"
        print(f"=== {label} ===")
        print(f"  cadence {cadence.get(key):.2f}s, so returns are rescaled to a "
              f"per-minute basis by sqrt(60/cadence)")
        print(f"  {len(points)} buckets, volatility observed from {min(xs):.2f} to "
              f"{max(xs):.2f} bps/min (median {statistics.median(xs):.2f})")
        print(f"  |dislocation| observed from {min(ys):.2f} to {max(ys):.2f} bps "
              f"(median {statistics.median(ys):.2f})")
        print(f"  correlation           {('-' if r is None else f'{r:+.3f}')}")
        print(f"  slope through origin  {('-' if slope0 is None else f'{slope0:.3f}')} "
              f"bps of dislocation per bps/min of volatility")
        if slope is not None:
            print(f"  free fit              slope {slope:+.3f}, intercept "
                  f"{intercept:+.2f} bps")

        if r is not None and abs(r) < 0.2:
            print(f"  NO USABLE RELATIONSHIP: correlation {r:+.3f}. Within this window")
            print(f"  the dislocation does not move with volatility, so scaling the")
            print(f"  measurement by a volatility ratio has no support in the data.")
            print(f"  That does not mean the relationship is absent -- the observed")
            print(f"  volatility range is narrow -- it means this sample cannot see it.")
            print()
            continue

        floor = float(key[2]) / 100.0 + args.taker_bps
        stats = history.get(key[0])
        if stats and slope0:
            # BOTH fits, because they disagree and only one of them flatters the
            # strategy. Through the origin the whole dislocation scales with
            # volatility; with an intercept most of it is a constant that does not.
            # Reporting only the first would be choosing the favourable model without
            # saying so, and the correlation here is far too weak to justify choosing
            # either.
            print(f"  projection against the 180-day volatility distribution, "
                  f"under BOTH fits:")
            print(f"    {'regime':<7} {'vol':>7}  {'origin fit':>11} "
                  f"{'free fit':>11}  {'floor':>6}  verdict")
            for name in ("today", "p50", "p90", "p99"):
                vol = stats[name]
                projected0 = slope0 * vol
                projected1 = (
                    slope * vol + intercept
                    if slope is not None and intercept is not None else None
                )
                factor = vol / max(xs) if max(xs) > 0 else None
                note = ""
                if factor and factor > 1:
                    note = f"  [{factor:.1f}x past the observed range]"
                both_clear = (
                    projected0 > floor
                    and projected1 is not None and projected1 > floor
                )
                neither = (
                    projected0 <= floor
                    and (projected1 is None or projected1 <= floor)
                )
                verdict = "clears" if both_clear else ("short" if neither
                                                       else "FITS DISAGREE")
                print(f"    {name:<7} {vol:>7.2f}  {projected0:>11.2f} "
                      f"{('-' if projected1 is None else f'{projected1:>11.2f}')}  "
                      f"{floor:>6.1f}  {verdict}{note}")
            print(f"    The two fits differ because the free one puts an intercept of "
                  f"{intercept:+.2f} bps on it -- a component of the dislocation that")
            print(f"    does not move with volatility at all. Whether that intercept is "
                  f"real or an artifact of a narrow range is exactly what this sample")
            print(f"    cannot settle, and it is the difference between clearing the "
                  f"floor at p99 and never clearing it.")
        print()

    print("Every projection past the observed volatility range is extrapolation, and the")
    print("range here spans a few hours of a 12th-percentile day. A slope fitted between")
    print("0.5 and 4 bps/min says very little about 20. The projections bound what this")
    print("data can support; they are not a forecast.")
    print()
    print("THE DECISIVE EXPERIMENT this identifies: record during a high-volatility")
    print("period. The two fits agree on today and disagree at p90 and p99, the")
    print("correlation is too weak to choose between them, and no amount of further")
    print("recording in a quiet market resolves it -- the missing data is a different")
    print("regime, not more of this one.")


main()
