"""The user-data-stream path, which is how the bot learns about fills.

`create_order` was fixed earlier today. `_process_execution_report` -- the handler
for Binance's asynchronous executionReport events -- was not, and it carries the
same defect cluster plus two worse ones. This is the path that matters for
reconciliation: it is how a fill that happens after the REST response is learned
about at all.

Five defects, in order of how much money they can cost:

1. `filled_size=last_filled` reads event['l'], the quantity of the LAST fill, not
   event['z'], the cumulative. A partial sequence of 0.2, 0.3, 0.5 therefore
   reports 0.5 on the final event rather than 1.0. `cumulative_filled` is even
   computed on the line above and then not used. Position accounting built on this
   under-reports every partially-filled order, so the bot believes it holds less
   than it does and hedges the wrong quantity.

2. `"new": "partially_filled"` -- a resting order with nothing filled claims a
   partial fill. Same lie as create_order had.

3. `status_map.get(raw, "partially_filled")` -- an unrecognised status also claims
   a partial fill.

4. `avg_fill_price = Decimal('0')` as the default, so an event with no price
   information reports a fill price of zero rather than None. Zero flows into PnL
   as a real price.

5. The map omits PENDING_NEW, PENDING_CANCEL and EXPIRED_IN_MATCH, all of which
   Binance sends, and each of which therefore falls through to defect 3.
"""
from decimal import Decimal

import pytest

from src.core.config import CexConfig, SecretsConfig
from src.exchange.binance import BinanceCexClient
from tests.fakes import make_pair


def D(x) -> Decimal:
    return Decimal(str(x))


def _client() -> BinanceCexClient:
    config = CexConfig(
        name="binance", base_url="https://x", ws_url="wss://x/ws",
        api_key_env="A", api_secret_env="B", recv_window_ms=3000,
    )
    secrets = SecretsConfig(
        binance_api_key="k", binance_api_secret="s",
        dex_wallet_private_key="0x" + "11" * 32,
    )
    client = BinanceCexClient(config, secrets, [make_pair()])
    client._captured = []

    async def capture(event, update, pair):
        client._captured.append(update)

    client._publish_dashboard_fill = capture
    return client


def _report(**overrides) -> dict:
    """A Binance executionReport. Field names are Binance's single letters."""
    event = {
        "e": "executionReport", "E": 1700000000000, "s": "ETHUSDT",
        "i": 111, "X": "PARTIALLY_FILLED", "x": "TRADE",
        "l": "0.30000000",      # last executed quantity
        "z": "0.50000000",      # CUMULATIVE filled quantity
        "L": "1894.00000000",   # last executed price
        "Z": "947.00000000",    # cumulative quote quantity
        "r": "NONE",
    }
    event.update(overrides)
    return event


# --- cumulative versus last fill ----------------------------------------


async def test_filled_size_is_cumulative_not_the_last_fill():
    """The defect that under-reports every partially filled order.

    A hedge sized from the last fill instead of the cumulative fill leaves the
    difference unhedged, and the bot does not know it.
    """
    client = _client()

    await client._process_execution_report(
        _report(l="0.30000000", z="0.50000000")
    )

    update = client._captured[0]
    assert update.filled_size == D("0.5"), (
        f"reported {update.filled_size}; 0.3 is the last fill, 0.5 is what is held"
    )


async def test_a_sequence_of_partials_ends_at_the_full_quantity():
    """The property that matters, stated over a realistic sequence."""
    client = _client()

    for last, cumulative in (("0.2", "0.2"), ("0.3", "0.5"), ("0.5", "1.0")):
        await client._process_execution_report(
            _report(l=last, z=cumulative, Z=str(Decimal(cumulative) * 1894))
        )

    assert [u.filled_size for u in client._captured] == [D("0.2"), D("0.5"), D("1.0")]


# --- status -------------------------------------------------------------


async def test_a_new_order_is_reported_as_new():
    client = _client()

    await client._process_execution_report(
        _report(X="NEW", x="NEW", l="0", z="0", Z="0")
    )

    assert client._captured[0].status == "new"


@pytest.mark.parametrize("raw,expected", [
    ("NEW", "new"),
    ("PENDING_NEW", "new"),
    ("PARTIALLY_FILLED", "partially_filled"),
    ("FILLED", "filled"),
    ("CANCELED", "canceled"),
    ("PENDING_CANCEL", "canceled"),
    ("EXPIRED", "canceled"),
    ("EXPIRED_IN_MATCH", "canceled"),
    ("REJECTED", "rejected"),
])
async def test_every_binance_status_maps_faithfully(raw, expected):
    client = _client()

    await client._process_execution_report(_report(X=raw))

    assert client._captured[0].status == expected


async def test_an_unknown_status_is_never_a_fill_claim():
    client = _client()

    await client._process_execution_report(_report(X="SOMETHING_NEW"))

    update = client._captured[0]
    assert update.status not in ("filled", "partially_filled")
    assert update.reason and "SOMETHING_NEW" in update.reason


# --- price --------------------------------------------------------------


async def test_the_price_is_the_cumulative_average():
    """947.00 over 0.5 filled is 1894.00."""
    client = _client()

    await client._process_execution_report(
        _report(z="0.50000000", Z="947.00000000")
    )

    assert client._captured[0].avg_fill_price == D("1894")


async def test_an_unfilled_event_reports_no_price_rather_than_zero():
    """Zero is a number that flows into PnL and values the position at nothing."""
    client = _client()

    await client._process_execution_report(
        _report(X="NEW", x="NEW", l="0", z="0", Z="0", L="0")
    )

    assert client._captured[0].avg_fill_price is None


async def test_the_last_price_is_used_when_no_cumulative_quote_is_present():
    """Some events carry L without Z. Better than nothing, and better than zero."""
    client = _client()

    await client._process_execution_report(
        _report(z="0.30000000", Z="0", L="1895.50000000", l="0.30000000")
    )

    assert client._captured[0].avg_fill_price == D("1895.5")


async def test_the_reason_is_preserved_for_a_rejection():
    client = _client()

    await client._process_execution_report(
        _report(X="REJECTED", r="INSUFFICIENT_BALANCE", l="0", z="0", Z="0")
    )

    update = client._captured[0]
    assert update.status == "rejected"
    assert update.reason == "INSUFFICIENT_BALANCE"


# --- the two paths must agree -------------------------------------------


def test_both_order_paths_share_one_status_translator():
    """`create_order` and `_process_execution_report` translated status
    independently, which is why one was fixed and the other was not. One
    translator, used twice, cannot drift.
    """
    import ast
    import inspect

    from src.exchange import binance

    tree = ast.parse(inspect.getsource(binance))
    literal_maps = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Dict):
            continue
        keys = [k.value for k in node.keys if isinstance(k, ast.Constant)]
        if any(str(k).upper() in ("NEW", "FILLED", "PARTIALLY_FILLED") for k in keys):
            literal_maps.append(node.lineno)

    assert len(literal_maps) <= 1, (
        f"more than one order-status map exists (lines {literal_maps}); they will "
        f"drift, and one of them already did"
    )
