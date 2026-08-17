"""Reading the audit trail: the queries that turn recorded rows into decisions.

Every number in the first readiness review came from ad-hoc SQL. That is fine
once and useless as a practice: if consulting the dataset takes a hand-written
query, it stops being consulted and the decisions revert to intuition.

Five summaries, each answering a question that was actually asked:

    edge_distribution     is there an edge, how big, per pair and direction?
    placebo_comparison    is it an edge, or a staleness artefact?
    cost_decomposition    where does the money go?
    direction_balance     is the flow one-directional -- is rotation cost real?
    rejection_reasons     is the market quiet, or is the bot broken?

Two conventions run through all of them.

MISSING IS NOT ZERO. A rejection that happened before the economics were
computable has no edge. Counting it as zero would drag every median toward
break-even, which is the direction that flatters the strategy.

UNTRADEABLE IS EXCLUDED BY DEFAULT. A denylist-mode measurement run deliberately
observes tokens it would never trade, and mixing them into one number is how a
fee-on-transfer token's transfer tax gets reported as an edge. `policy_verdict`
exists for exactly this, and `tradeable_only=False` opts out deliberately.
"""
from __future__ import annotations

import statistics
from decimal import Decimal
from typing import Dict, List, Optional, Sequence

__all__ = [
    "edge_distribution",
    "placebo_comparison",
    "cost_decomposition",
    "direction_balance",
    "rejection_reasons",
]

TEN_THOUSAND = Decimal("10000")

# Above this share of identical live/placebo pairs, the control is suspect: it
# means the delay is shorter than the DEX's own update interval, so the two arms
# are comparing a quote to itself. Not a hard threshold -- a genuinely still
# market produces identical pairs too -- which is why it produces a caveat rather
# than a verdict.
IDENTICAL_SUSPICION_RATIO = Decimal("0.4")


def _rows(store, tradeable_only: bool = True) -> List[dict]:
    rows = store.all_rows()
    if tradeable_only:
        # None is kept: rows written before policy_verdict existed are not
        # evidence of an untradeable token, and silently dropping them would
        # shrink a historical dataset without saying so.
        rows = [r for r in rows if r.get("policy_verdict") in (None, "allowed")]
    return rows


def _decimals(rows: Sequence[dict], column: str) -> List[Decimal]:
    out = []
    for row in rows:
        value = row.get(column)
        if value is None or value == "":
            continue
        out.append(Decimal(str(value)))
    return out


def _percentile(values: Sequence[Decimal], fraction: float) -> Decimal:
    ordered = sorted(values)
    index = int(fraction * (len(ordered) - 1))
    return ordered[index]


def edge_distribution(store, tradeable_only: bool = True) -> List[dict]:
    """Net edge per pair and direction, as a shape rather than an average.

    Percentiles rather than a mean because the shape IS the question: an edge
    present 2% of the time is a different strategy from one present always, and a
    mean cannot tell them apart.
    """
    grouped: Dict[tuple, List[Decimal]] = {}
    for row in _rows(store, tradeable_only):
        if row.get("net_bps") in (None, ""):
            continue
        key = (row["cex_symbol"], row["direction"])
        grouped.setdefault(key, []).append(Decimal(str(row["net_bps"])))

    results = []
    for (symbol, direction), values in sorted(grouped.items()):
        results.append({
            "cex_symbol": symbol,
            "direction": direction,
            "count": len(values),
            "median_bps": statistics.median(sorted(values)),
            "p10_bps": _percentile(values, 0.10),
            "p90_bps": _percentile(values, 0.90),
            "best_bps": max(values),
            "worst_bps": min(values),
        })
    return results


def placebo_comparison(store, tradeable_only: bool = True) -> dict:
    """Live versus the same book priced against a deliberately stale DEX quote.

    Under the null hypothesis -- the measured edge is a staleness artefact -- the
    two distributions coincide. The verdict string is deliberately cautious: a
    high identical rate is at least as likely to mean the delay is too short as it
    is to mean the null holds, and that mistake has already been made once here.
    """
    rows = _rows(store, tradeable_only)
    paired = [
        (Decimal(str(r["net_bps"])), Decimal(str(r["placebo_net_bps"])))
        for r in rows
        if r.get("net_bps") not in (None, "")
        and r.get("placebo_net_bps") not in (None, "")
    ]

    if not paired:
        return {
            "paired": 0,
            "live_median_bps": None,
            "placebo_median_bps": None,
            "median_difference_bps": None,
            "identical": 0,
            "live_better": 0,
            "verdict": (
                "No paired observations yet. The placebo needs a run longer than "
                "its configured delay before it can say anything, and a zero "
                "difference from no data would read as support for the null."
            ),
        }

    live = [p[0] for p in paired]
    placebo = [p[1] for p in paired]
    diffs = [a - b for a, b in paired]
    identical = sum(1 for d in diffs if d == 0)
    live_better = sum(1 for d in diffs if d > 0)
    ratio = Decimal(identical) / Decimal(len(paired))

    if ratio >= IDENTICAL_SUSPICION_RATIO:
        verdict = (
            f"{identical} of {len(paired)} pairs are IDENTICAL "
            f"({float(ratio):.0%}). Before reading this as support for the null, "
            f"check that strategy.placebo.delay_seconds exceeds the block time of "
            f"the chains being quoted: within one block every DEX quote is the "
            f"same number, so a short delay compares a quote to itself."
        )
    else:
        median_diff = statistics.median(sorted(diffs))
        verdict = (
            f"Live beats placebo in {live_better} of {len(paired)} pairs, median "
            f"difference {float(median_diff):+.2f} bps. Compare that against the "
            f"distance to break-even: if the difference is small relative to the "
            f"deficit, latency is not what is missing."
        )

    return {
        "paired": len(paired),
        "live_median_bps": statistics.median(sorted(live)),
        "placebo_median_bps": statistics.median(sorted(placebo)),
        "median_difference_bps": statistics.median(sorted(diffs)),
        "p10_difference_bps": _percentile(diffs, 0.10),
        "p90_difference_bps": _percentile(diffs, 0.90),
        "identical": identical,
        "live_better": live_better,
        "verdict": verdict,
    }


def cost_decomposition(store, tradeable_only: bool = True,
                       by_pair: bool = False):
    """Average cost per trade, in basis points of notional.

    Basis points rather than currency because that is the unit the floor is
    expressed in, and a reader comparing a 2.00 charge against a 5 bps floor has
    to do the division in their head -- which is where the significance of the
    rotation charge went unnoticed.

    `by_pair=True` returns one decomposition per pair, and is usually what you
    want. Averaging across pairs mixes gas regimes -- Arbitrum's gas is a fraction
    of Ethereum's -- and one pair with an 800 bps dislocation drags the combined
    gross into meaninglessness. The first run of this reported a -196 bps average
    gross that described no pair that existed.
    """
    rows = [
        r for r in _rows(store, tradeable_only)
        if r.get("notional_quote") not in (None, "")
    ]
    if by_pair:
        grouped: Dict[str, List[dict]] = {}
        for row in rows:
            grouped.setdefault(row["cex_symbol"], []).append(row)
        return {
            symbol: _decompose(group) for symbol, group in sorted(grouped.items())
        }
    return _decompose(rows)


def _decompose(rows: Sequence[dict]) -> dict:
    if not rows:
        return {"rows": 0}

    notionals = _decimals(rows, "notional_quote")
    notional = sum(notionals) / Decimal(len(notionals))

    def bps(column: str) -> Decimal:
        values = _decimals(rows, column)
        if not values:
            return Decimal(0)
        mean = sum(values) / Decimal(len(values))
        return mean / notional * TEN_THOUSAND

    costs = {
        "cex_fee": bps("cex_fee_quote"),
        "gas": bps("gas_quote"),
        "rotation": bps("rotation_cost_quote"),
    }
    largest = max(costs, key=lambda k: costs[k]) if costs else None

    return {
        "rows": len(rows),
        "notional_quote": notional,
        "gross_bps": bps("gross_quote"),
        "cex_fee_bps": costs["cex_fee"],
        "gas_bps": costs["gas"],
        "rotation_bps": costs["rotation"],
        "net_bps": bps("net_quote"),
        "total_cost_bps": costs["cex_fee"] + costs["gas"] + costs["rotation"],
        "largest_cost": largest,
    }


def direction_balance(store, tolerance_seconds: float = 0.5,
                      tradeable_only: bool = True) -> List[dict]:
    """Which direction won each cycle, per pair.

    This decides whether the rotation charge is real. A balanced flow means
    inventory mean-reverts and rotation is rare; a one-sided flow strands
    inventory every time and the full charge applies.

    Both directions of one cycle are written milliseconds apart, so they are
    paired on proximity in time rather than on an id -- there is no cycle id in
    the schema, and adding one would be the cleaner fix if this ever needs to be
    exact.
    """
    by_pair: Dict[str, List[dict]] = {}
    for row in _rows(store, tradeable_only):
        if row.get("net_bps") in (None, "") or not row.get("direction"):
            continue
        by_pair.setdefault(row["cex_symbol"], []).append(row)

    results = []
    for symbol, rows in sorted(by_pair.items()):
        rows.sort(key=lambda r: r["ts"])
        c2d = d2c = 0
        index = 0
        while index < len(rows) - 1:
            first, second = rows[index], rows[index + 1]
            same_cycle = (
                abs(first["ts"] - second["ts"]) <= tolerance_seconds
                and first["direction"] != second["direction"]
            )
            if not same_cycle:
                index += 1
                continue
            better = (
                first if Decimal(str(first["net_bps"]))
                > Decimal(str(second["net_bps"])) else second
            )
            if better["direction"] == "CEX_to_DEX":
                c2d += 1
            else:
                d2c += 1
            index += 2

        cycles = c2d + d2c
        if not cycles:
            continue
        # 0 when perfectly balanced, 1 when entirely one-sided.
        imbalance = abs(Decimal(c2d - d2c)) / Decimal(cycles)
        results.append({
            "cex_symbol": symbol,
            "cycles": cycles,
            "cex_to_dex_better": c2d,
            "dex_to_cex_better": d2c,
            "imbalance": imbalance,
        })
    return results


def rejection_reasons(store, tradeable_only: bool = False) -> Dict[str, int]:
    """How many evaluations ended each way.

    Defaults to including untradeable tokens: this is an operational health
    summary, not an edge measurement, and a denied pair is still information about
    what the bot spent its cycle doing.
    """
    counts: Dict[str, int] = {}
    for row in _rows(store, tradeable_only):
        key = row.get("reason") or row.get("outcome") or "unknown"
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: kv[1], reverse=True))
