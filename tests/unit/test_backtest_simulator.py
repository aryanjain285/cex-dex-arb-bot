"""The backtest could not execute a single row.

Every interface it spoke had been replaced by the detector rewrite:

* `BacktestCexClient` implemented `get_quote(pair)`, but the detector calls
  `get_book(pair)` and needs a depth ladder -- so the first cycle raised
  AttributeError.
* `BacktestDexClient.get_quote` returned a `Quote(side=...)`; `Quote` has no
  `side` field, so pydantic raised, and the detector needs a `DexQuote` with
  `gas_cost_quote` besides.
* The run loop read `summary.is_complete_success` and `summary.net_pnl_quote`,
  and the report read `market_pair.quote`. None of those three exists.

So the whole module was dead in four independent ways, and nothing noticed
because it had no tests and the CLI wrapper catches every exception and prints it
as text -- which is indistinguishable from "the backtest found nothing".

It also carried its own cost model: a hardcoded `slippage = Decimal("0.001")`
applied to the DEX price. That is a third cost model beside the detector's and
the spike screen's, and it is wrong twice over -- the number is invented, and the
DEX quote it adjusts is already net of price impact.

Two limitations of the data are now explicit rather than hidden, because a
backtest that silently assumes them would overstate the strategy:

* the CSV carries no depth, so the book is synthesised at a configured size per
  level, and any trade larger than that is refused by the same depth check the
  live detector applies;
* gas must be present in the data. A missing gas cost is a hard error, not a
  zero: zero gas is the single easiest way to make an unprofitable strategy look
  profitable.
"""
import textwrap
from decimal import Decimal

import pytest

from backtest.datasets import load_dataset
from backtest.simulator import Simulator
from src.core.config import load_config

D = Decimal


def _csv(tmp_path, rows: str, header="timestamp,cex_bid_price,cex_ask_price,dex_price,gas_quote"):
    path = tmp_path / "data.csv"
    path.write_text(header + "\n" + textwrap.dedent(rows).strip() + "\n", encoding="utf-8")
    return load_dataset(str(path))


def _config(**strategy_overrides):
    config = load_config()
    for key, value in strategy_overrides.items():
        setattr(config.strategy, key, value)
    return config


async def test_a_crossing_row_produces_a_trade(tmp_path):
    """End to end over three rows: the module must actually run."""
    data = _csv(tmp_path, """
        2026-01-01T00:00:00Z,1000,1000.1,1000.2,0.10
        2026-01-01T00:00:01Z,1000,1000.1,1100.0,0.10
        2026-01-01T00:00:02Z,1000,1000.1,1000.2,0.10
    """)

    sim = Simulator(_config(), data)
    await sim.run()

    assert sim.results, "a 10% dislocation must produce at least one trade"
    assert all(s.pnl_quote is not None for s in sim.results)
    sim.report()  # must not raise


async def test_no_crossing_produces_no_trades(tmp_path):
    """The negative control. A backtest that trades on everything is worthless."""
    data = _csv(tmp_path, """
        2026-01-01T00:00:00Z,1000,1000.1,1000.05,0.10
        2026-01-01T00:00:01Z,1000,1000.1,1000.05,0.10
    """)

    sim = Simulator(_config(), data)
    await sim.run()

    assert sim.results == []


async def test_the_cex_client_serves_a_book_the_detector_can_use(tmp_path):
    data = _csv(tmp_path, """
        2026-01-01T00:00:00Z,999.5,1000.5,1000.0,0.10
    """)
    sim = Simulator(_config(), data)
    sim.cex_client.set_tick(data.iloc[0])

    book = await sim.cex_client.get_book(sim.market_pair)

    assert book is not None
    assert book.best_bid == D("999.5")
    assert book.best_ask == D("1000.5")
    assert book.bids and book.asks, "the detector walks a ladder, not a top of book"
    # The replay's feed is live by definition; a historical feed timestamp would
    # make every book fail the staleness gate and the backtest would find nothing.
    assert book.feed_age_seconds(book.feed_timestamp) == 0


async def test_the_dex_client_returns_a_dex_quote_with_gas(tmp_path):
    data = _csv(tmp_path, """
        2026-01-01T00:00:00Z,1000,1000.1,1234.5,0.42
    """)
    sim = Simulator(_config(), data)
    sim.dex_client.set_tick(data.iloc[0])

    quote = await sim.dex_client.get_quote(
        sim.market_pair, size=D(1), side="sell", estimate_gas=True
    )

    assert quote.price == D("1234.5"), "the recorded price must be used as recorded"
    assert quote.gas_cost_quote == D("0.42")


async def test_a_missing_gas_cost_is_a_hard_error(tmp_path):
    """Zero gas is the easiest way to make a losing strategy look profitable."""
    data = _csv(
        tmp_path,
        """
        2026-01-01T00:00:00Z,1000,1000.1,1100.0
        """,
        header="timestamp,cex_bid_price,cex_ask_price,dex_price",
    )

    sim = Simulator(_config(), data)
    with pytest.raises(ValueError, match="gas"):
        await sim.run()


async def test_the_synthesised_depth_is_finite_and_enforced(tmp_path):
    """The CSV has no depth, so the book is synthesised at a stated size.

    A trade larger than that size must be refused by the same depth check the
    live detector applies -- otherwise the backtest would report fills that the
    live system could never achieve.
    """
    data = _csv(tmp_path, """
        2026-01-01T00:00:00Z,1000,1000.1,1100.0,0.10
    """)

    # Depth of 0.001 base against a 1000-notional target: the size the detector
    # wants cannot be filled.
    config = _config(target_notional_usd=1000)
    sim = Simulator(config, data, depth_per_level_base=D("0.001"))
    await sim.run()

    assert sim.results == [], (
        "a trade far larger than the available depth must not report a fill"
    )


def test_no_invented_cost_constant_remains_in_the_module():
    """Guards against the third cost model coming back.

    The module previously applied `slippage = Decimal("0.001")` to the DEX price:
    an invented number, applied to a quote that is already net of price impact,
    so it double-counted the very thing it was invented for.

    Checked over the AST rather than the source text, so the module can keep
    describing the defect in prose without the guard tripping on its own
    explanation -- and so the check is about executable code, which is what
    matters.
    """
    import ast
    import inspect

    from backtest import simulator

    tree = ast.parse(inspect.getsource(simulator))

    # Any assignment or binding whose name suggests a locally-modelled cost.
    forbidden_names = {"slippage", "slippage_pct", "impact", "impact_bps",
                       "cost_buffer", "cost_buffer_bps", "fee_pct"}
    bound = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            bound.add(node.id)
        elif isinstance(node, ast.arg):
            bound.add(node.arg)
    assert not (bound & forbidden_names), (
        f"the backtest is modelling costs itself: {sorted(bound & forbidden_names)}"
    )

    # And no small fractional literal, which is what such a model looks like
    # even when its variable is named innocuously.
    literals = [
        node.value for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float))
        and not isinstance(node.value, bool)
    ]
    suspicious = [v for v in literals if 0 < float(v) < 0.5]
    assert not suspicious, (
        f"suspicious fractional constants in the backtest, which is how an "
        f"invented cost model looks: {suspicious}"
    )


async def test_the_report_survives_an_empty_result_set(tmp_path):
    data = _csv(tmp_path, """
        2026-01-01T00:00:00Z,1000,1000.1,1000.05,0.10
    """)
    sim = Simulator(_config(), data)
    await sim.run()
    sim.report()  # must not raise on zero trades
