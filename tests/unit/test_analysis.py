"""Turning the audit trail into answers.

Every number in today's report came from ad-hoc queries against
`data/evaluations.sqlite3`. That is fine once and useless as a practice: the
measurement loop has to be self-service, or the dataset stops being consulted and
the decisions go back to being intuitions.

Four summaries, each of which answers a question that was actually asked today:

    edge distribution     is there an edge, and how big, per pair and direction?
    placebo comparison    is what we see an edge, or a staleness artefact?
    cost decomposition    where does the money go?
    direction balance     is the flow one-directional, i.e. is rotation real?

Plus rejection reasons, which is how a quiet market is told apart from a broken
one -- and now also how a throttled node is told apart from an empty pool.
"""
from decimal import Decimal

import pytest

from src.infra.analysis import (
    cost_decomposition, direction_balance, edge_distribution,
    placebo_comparison, rejection_reasons,
)
from src.infra.evaluation_store import EvaluationRecord, EvaluationStore


def D(x) -> Decimal:
    return Decimal(str(x))


def _row(**overrides) -> EvaluationRecord:
    fields = dict(
        ts=1000.0, cex_symbol="ETH/USDT", base="WETH", quote_cex="USDT",
        dex_chain="ethereum", dex_pool_fee=500, is_synthetic=False,
        outcome="rejected", direction="CEX_to_DEX", reason="below_floor",
        size_base=D("0.5"), notional_quote=D(1000),
        cex_price=D(2000), cex_best_bid=D(1999), cex_best_ask=D(2000),
        dex_price=D(2001),
        gross_quote=D("0.50"), cex_fee_quote=D("0.75"), gas_quote=D("0.02"),
        rotation_cost_quote=D(2), net_quote=D("-2.27"), net_bps=D("-22.70"),
        placebo_net_bps=D("-23.00"), policy_verdict="allowed",
        cex_legs=1, book_age_s=0.1, depth_levels_used=1,
        min_net_bps=D(5), taker_fee_bps=D("7.5"),
    )
    fields.update(overrides)
    return EvaluationRecord(**fields)


@pytest.fixture
def store(tmp_path):
    s = EvaluationStore(tmp_path / "e.sqlite3", run_id="test-run")
    yield s
    s.close()


# --- edge distribution ---------------------------------------------------


def test_the_distribution_reports_a_median_per_pair_and_direction(store):
    for i, net in enumerate(["-30", "-20", "-10"]):
        store.record(_row(ts=1000.0 + i, net_bps=D(net)))
    store.record(_row(ts=2000.0, direction="DEX_to_CEX", net_bps=D("-5")))

    rows = edge_distribution(store)

    by_key = {(r["cex_symbol"], r["direction"]): r for r in rows}
    assert by_key[("ETH/USDT", "CEX_to_DEX")]["median_bps"] == D("-20")
    assert by_key[("ETH/USDT", "CEX_to_DEX")]["count"] == 3
    assert by_key[("ETH/USDT", "DEX_to_CEX")]["median_bps"] == D("-5")


def test_the_distribution_reports_percentiles_not_just_a_mean(store):
    """A mean hides the shape, and the shape is the question: an edge that exists
    2% of the time is a different strategy from one that exists always."""
    for i in range(10):
        store.record(_row(ts=1000.0 + i, net_bps=D(-i)))

    row = edge_distribution(store)[0]

    assert row["p10_bps"] < row["median_bps"] < row["p90_bps"]
    assert row["best_bps"] == D(0)


def test_the_distribution_excludes_untradeable_tokens_by_default(store):
    """A denylist-mode measurement run deliberately observes tokens it would never
    trade. Mixing them into one number is how a fee-on-transfer token's transfer
    tax gets reported as an edge."""
    store.record(_row(net_bps=D("-20")))
    store.record(_row(ts=1001.0, cex_symbol="LINGO/USDT",
                      policy_verdict="denied", net_bps=D(500)))

    rows = edge_distribution(store)

    assert [r["cex_symbol"] for r in rows] == ["ETH/USDT"]

    rows_all = edge_distribution(store, tradeable_only=False)
    assert len(rows_all) == 2


def test_rows_without_an_edge_are_not_counted_as_zero(store):
    """A rejection before the economics were computable has no edge, and counting
    it as zero would drag every median toward break-even."""
    store.record(_row(net_bps=D("-20")))
    store.record(_row(ts=1001.0, reason="no_book", net_bps=None, net_quote=None))

    row = edge_distribution(store)[0]

    assert row["count"] == 1


# --- the placebo --------------------------------------------------------


def test_the_placebo_comparison_reports_the_paired_difference(store):
    store.record(_row(net_bps=D("-20"), placebo_net_bps=D("-25")))
    store.record(_row(ts=1001.0, net_bps=D("-30"), placebo_net_bps=D("-30")))

    result = placebo_comparison(store)

    assert result["paired"] == 2
    assert result["live_median_bps"] == D("-25")
    assert result["placebo_median_bps"] == D("-27.5")
    assert result["identical"] == 1
    assert result["live_better"] == 1


def test_the_placebo_comparison_says_when_it_has_nothing_to_say(store):
    """Before the delay has elapsed there are no pairs, and reporting a difference
    of zero would read as support for the null."""
    store.record(_row(placebo_net_bps=None))

    result = placebo_comparison(store)

    assert result["paired"] == 0
    assert result["live_median_bps"] is None
    assert result["verdict"], "it must still say something a reader can act on"


def test_a_high_identical_rate_is_called_out(store):
    """The failure that made the first placebo useless: if the two arms agree
    almost always, the delay is shorter than the DEX's own update interval and the
    control is comparing a quote to itself."""
    for i in range(10):
        store.record(_row(ts=1000.0 + i, net_bps=D(-20), placebo_net_bps=D(-20)))

    result = placebo_comparison(store)

    assert result["identical"] == 10
    assert "delay" in result["verdict"].lower() or "block" in result["verdict"].lower()


# --- costs --------------------------------------------------------------


def test_the_cost_decomposition_is_expressed_in_basis_points(store):
    store.record(_row())

    result = cost_decomposition(store)

    assert result["notional_quote"] == D(1000)
    # 0.75 on 1000 is 7.5 bps; 2.00 is 20 bps.
    assert result["cex_fee_bps"] == D("7.5")
    assert result["rotation_bps"] == D(20)
    assert result["gas_bps"] == D("0.2")
    assert result["gross_bps"] == D(5)


def test_the_decomposition_names_the_largest_cost(store):
    """Twenty of the twenty-eight basis points were rotation, and that is the
    number a reader should leave with."""
    store.record(_row())

    result = cost_decomposition(store)

    assert result["largest_cost"] == "rotation"


def test_the_decomposition_can_be_grouped_by_pair(store):
    """Averaging across pairs mixes different gas regimes and different markets.

    Gas on Arbitrum is a fraction of Ethereum's, and one pair with an 800 bps
    dislocation drags a combined gross average into meaninglessness -- which is
    exactly what the first run of this produced: a -196 bps "average gross" that
    described no pair that existed.
    """
    store.record(_row(cex_symbol="ETH/USDT", dex_chain="ethereum",
                      gas_quote=D("0.02"), gross_quote=D("0.50")))
    store.record(_row(ts=1001.0, cex_symbol="ARB/USDT", dex_chain="arbitrum",
                      gas_quote=D("0.008"), gross_quote=D("-80")))

    combined = cost_decomposition(store)
    per_pair = cost_decomposition(store, by_pair=True)

    assert isinstance(per_pair, dict)
    assert set(per_pair) == {"ETH/USDT", "ARB/USDT"}
    assert per_pair["ETH/USDT"]["gas_bps"] == D("0.2")
    assert per_pair["ARB/USDT"]["gas_bps"] == D("0.08")
    assert per_pair["ETH/USDT"]["gross_bps"] == D(5)
    assert per_pair["ARB/USDT"]["gross_bps"] == D(-800)
    # The combined figure is the average of two unlike things, and is why the
    # per-pair form exists.
    assert combined["gross_bps"] == D("-397.5")


def test_the_decomposition_averages_over_rows_rather_than_picking_one(store):
    store.record(_row(gas_quote=D("0.01")))
    store.record(_row(ts=1001.0, gas_quote=D("0.03")))

    result = cost_decomposition(store)

    assert result["gas_bps"] == D("0.2")
    assert result["rows"] == 2


# --- direction balance --------------------------------------------------


def test_direction_balance_counts_which_side_won_each_cycle(store):
    # One cycle: both directions, CEX_to_DEX better.
    store.record(_row(ts=1000.0, direction="CEX_to_DEX", net_bps=D(-10)))
    store.record(_row(ts=1000.01, direction="DEX_to_CEX", net_bps=D(-20)))
    # Another cycle: DEX_to_CEX better.
    store.record(_row(ts=1001.0, direction="CEX_to_DEX", net_bps=D(-30)))
    store.record(_row(ts=1001.01, direction="DEX_to_CEX", net_bps=D(-5)))

    rows = direction_balance(store)

    row = rows[0]
    assert row["cycles"] == 2
    assert row["cex_to_dex_better"] == 1
    assert row["dex_to_cex_better"] == 1


def test_an_unpaired_evaluation_is_not_counted_as_a_cycle(store):
    """One direction alone is not a comparison."""
    store.record(_row(ts=1000.0, direction="CEX_to_DEX", net_bps=D(-10)))

    rows = direction_balance(store)

    assert rows == [] or rows[0]["cycles"] == 0


def test_a_one_sided_flow_is_reported_as_such(store):
    """Whether rotation cost is real depends on this: a 50/50 flow self-balances
    and rotation is rare; a 0/100 flow strands inventory every time."""
    for i in range(5):
        store.record(_row(ts=1000.0 + i, direction="CEX_to_DEX", net_bps=D(-30)))
        store.record(_row(ts=1000.0 + i + 0.01, direction="DEX_to_CEX",
                          net_bps=D(-5)))

    row = direction_balance(store)[0]

    assert row["cycles"] == 5
    assert row["dex_to_cex_better"] == 5
    assert row["imbalance"] == D(1), "a fully one-sided flow is an imbalance of 1"


# --- rejection reasons --------------------------------------------------


def test_rejection_reasons_are_counted(store):
    store.record(_row(reason="below_floor"))
    store.record(_row(ts=1001.0, reason="below_floor"))
    store.record(_row(ts=1002.0, reason="rpc_error", net_bps=None))

    counts = rejection_reasons(store)

    assert counts["below_floor"] == 2
    assert counts["rpc_error"] == 1


def test_an_rpc_error_is_visible_as_its_own_reason(store):
    """The whole point of separating it from no_dex_quote: an operator has to be
    able to see that the bot is being throttled rather than looking at an empty
    market."""
    store.record(_row(reason="rpc_error", net_bps=None))
    store.record(_row(ts=1001.0, reason="no_dex_quote", net_bps=None))

    counts = rejection_reasons(store)

    assert counts["rpc_error"] == 1
    assert counts["no_dex_quote"] == 1
