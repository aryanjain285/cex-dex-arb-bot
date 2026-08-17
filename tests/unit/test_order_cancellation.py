"""Cancellation claimed success without sending anything.

    async def cancel_order(self, order_id, pair) -> bool:
        # TODO: implement order cancellation
        logger.info(f"Simulated order cancellation: {order_id}")
        return True

A stub returning False would be an obvious gap. Returning True is a lie in the
dangerous direction, and it has two live consequences:

* `risk.cancel_all_on_start` and `cancel_all_on_shutdown` are configured, and
  app.py's call site was commented out. An operator who sets them believes stale
  orders are cleared before trading resumes. Nothing was cleared, and a resting
  order from a previous run can fill into a market that has moved.
* Cancelling the unfilled leg is the standard unwind for a half-executed
  arbitrage. A caller told the cancel succeeded stops tracking the order -- and
  the order then fills, unhedged, with nothing watching it.

The tests assert what was actually SENT, not what the method returned, because
returning the right value was never the problem.
"""
import json
from decimal import Decimal

import pytest

from src.core.config import CexConfig, SecretsConfig
from src.exchange.binance import BinanceCexClient
from tests.fakes import make_pair


class FakeResponse:
    def __init__(self, payload, status=200):
        self._payload = payload
        self.status = status
        self.headers = {}

    async def text(self):
        return self._payload

    async def json(self, **kwargs):
        return json.loads(self._payload)

    def raise_for_status(self):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


class FakeSession:
    """Records every request and returns a canned payload per HTTP method."""

    def __init__(self, payloads=None, status=200):
        self.payloads = payloads or {}
        self.status = status
        self.requests = []

    def _respond(self, method, url, kwargs):
        self.requests.append((method, url, kwargs))
        return FakeResponse(self.payloads.get(method, "{}"), self.status)

    def get(self, url, **kwargs):
        return self._respond("GET", url, kwargs)

    def post(self, url, **kwargs):
        return self._respond("POST", url, kwargs)

    def delete(self, url, **kwargs):
        return self._respond("DELETE", url, kwargs)


def _client(payloads=None, status=200) -> BinanceCexClient:
    config = CexConfig(
        name="binance", base_url="https://x", ws_url="wss://x/ws",
        api_key_env="A", api_secret_env="B", recv_window_ms=3000,
    )
    secrets = SecretsConfig(
        binance_api_key="k", binance_api_secret="s",
        dex_wallet_private_key="0x" + "11" * 32,
    )
    client = BinanceCexClient(config, secrets, [make_pair()])
    client._session = FakeSession(payloads, status)
    return client


CANCELLED = json.dumps({
    "symbol": "ETHUSDT", "origClientOrderId": "x", "orderId": 111,
    "status": "CANCELED", "executedQty": "0.00000000",
    "cummulativeQuoteQty": "0.00000000",
})

CANCEL_ALL = json.dumps([
    {"symbol": "ETHUSDT", "orderId": 111, "status": "CANCELED"},
    {"symbol": "ETHUSDT", "orderId": 112, "status": "CANCELED"},
])


# --- a single order ------------------------------------------------------


async def test_cancelling_actually_sends_a_request():
    """The whole defect in one assertion."""
    client = _client({"DELETE": CANCELLED})

    result = await client.cancel_order("111", make_pair())

    assert client._session.requests, "cancel_order sent nothing at all"
    method, url, _ = client._session.requests[0]
    assert method == "DELETE"
    assert url.endswith("/api/v3/order")
    assert result is True


async def test_the_request_identifies_the_order_and_symbol():
    client = _client({"DELETE": CANCELLED})

    await client.cancel_order("111", make_pair())

    _, _, kwargs = client._session.requests[0]
    params = kwargs.get("params") or kwargs.get("data")
    assert params["symbol"] == "ETHUSDT"
    assert str(params["orderId"]) == "111"


async def test_the_request_is_signed_and_carries_the_recv_window():
    client = _client({"DELETE": CANCELLED})

    await client.cancel_order("111", make_pair())

    _, _, kwargs = client._session.requests[0]
    params = dict(kwargs.get("params") or kwargs.get("data"))
    signature = params.pop("signature")
    assert int(params["recvWindow"]) == 3000
    assert signature == client._get_signature(params), (
        "the signature does not match the parameters sent"
    )


async def test_a_failed_cancellation_reports_failure():
    """An order that could not be cancelled must not be reported as cancelled:
    the caller's next move is to stop watching it."""
    client = _client({"DELETE": '{"code":-2011,"msg":"Unknown order sent."}'},
                     status=400)

    result = await client.cancel_order("111", make_pair())

    assert result is False


async def test_a_network_failure_reports_failure_rather_than_raising():
    """Cancellation runs on the unwind path, often already handling an error. It
    must not raise there, and it must not claim success."""
    class Broken(FakeSession):
        def delete(self, url, **kwargs):
            raise ConnectionError("network down")

    client = _client()
    client._session = Broken()

    assert await client.cancel_order("111", make_pair()) is False


# --- all open orders -----------------------------------------------------


async def test_cancel_all_sends_one_request_per_configured_symbol():
    """Binance's DELETE /openOrders is per symbol, so "all" means a request each.
    A single call would silently cancel only the first pair's orders."""
    client = _client({"DELETE": CANCEL_ALL})
    client.pairs = [make_pair("ETH/USDT"), make_pair("ARB/USDT", base="ARB")]

    cancelled = await client.cancel_all_orders()

    symbols = []
    for _, url, kwargs in client._session.requests:
        assert url.endswith("/api/v3/openOrders")
        params = kwargs.get("params") or kwargs.get("data")
        symbols.append(params["symbol"])
    assert sorted(symbols) == ["ARBUSDT", "ETHUSDT"]
    assert cancelled == 4, f"expected 2 cancellations per symbol, got {cancelled}"


async def test_cancel_all_continues_past_one_failing_symbol():
    """One symbol with no open orders returns an error; the others still need
    clearing, and stopping there would leave them resting."""
    calls = {"n": 0}

    class Flaky(FakeSession):
        def delete(self, url, **kwargs):
            calls["n"] += 1
            self.requests.append(("DELETE", url, kwargs))
            if calls["n"] == 1:
                return FakeResponse('{"code":-2011,"msg":"Unknown order sent."}', 400)
            return FakeResponse(CANCEL_ALL, 200)

    client = _client()
    client._session = Flaky()
    client.pairs = [make_pair("ETH/USDT"), make_pair("ARB/USDT", base="ARB")]

    cancelled = await client.cancel_all_orders()

    assert len(client._session.requests) == 2, "it stopped after the first failure"
    assert cancelled == 2


# --- the config flag must do something ----------------------------------


def test_the_startup_cancel_flag_is_wired_to_a_real_call():
    """`cancel_all_on_start` had its only call site commented out, so the flag was
    a promise the system did not keep. Checked by AST so a commented-out call
    cannot satisfy it."""
    import ast
    import inspect

    from src import app

    tree = ast.parse(inspect.getsource(app))
    found = False
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and node.attr == "cancel_all_orders":
            found = True
    assert found, (
        "app.py never calls cancel_all_orders, so risk.cancel_all_on_start "
        "cannot do anything"
    )
