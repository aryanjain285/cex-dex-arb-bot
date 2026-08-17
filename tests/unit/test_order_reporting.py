"""What `create_order` reports back about an order it placed.

This is a LIVE path today: `Rebalancer.run_rebalance_check(paper_run=False)` calls
it. Five defects, each of which makes the returned OrderUpdate say something that
is not true.

1. `"new": "partially_filled"`. A resting order with zero filled was reported as
   partially filled, so a caller reading `filled_size` alongside a
   partially_filled status believes it holds inventory it does not hold. On an
   arbitrage the response is to hedge the other leg -- against a position that
   does not exist.

2. Unknown statuses ALSO defaulted to `partially_filled`
   (`status_map.get(raw, "partially_filled")`). A status Binance adds later would
   silently become a claim about inventory.

3. `avg_fill_price` read `data.get('avgPrice') or data.get('price')`. Binance spot
   returns neither as an achieved average: `avgPrice` does not exist on the spot
   order response at all, and `price` is the LIMIT price -- "0.00000000" for a
   market order. So the achieved price of a market order was reported as zero, and
   the achieved price of a limit order was reported as the price asked for rather
   than the one obtained. PnL computed from either is fiction.

4. `timeInForce` was hardcoded to GTC while `CexOrder.tif` already existed and
   already defaulted to IOC. For arbitrage a resting order is a liability: it can
   fill minutes later, after the opportunity is gone, leaving an unhedged
   position. GTC is the worst of the three choices and the field that said so was
   ignored.

5. `recvWindow` was never sent, although `cex.recv_window_ms` is configured and
   validated. Binance then applies its own 5000ms default, so the setting did
   nothing.
"""
from decimal import Decimal

import pytest

from src.core.config import CexConfig, SecretsConfig
from src.core.types import CexOrder, MarketPair
from src.exchange.binance import BinanceCexClient
from tests.fakes import make_pair


def D(x) -> Decimal:
    return Decimal(str(x))


class FakeResponse:
    def __init__(self, payload_text, status=200):
        self._text = payload_text
        self.status = status
        self.headers = {}

    async def text(self):
        return self._text

    async def json(self, **kwargs):
        import json
        return json.loads(self._text)

    def raise_for_status(self):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


class FakeSession:
    def __init__(self, payload_text):
        self.payload_text = payload_text
        self.posts = []

    def post(self, url, **kwargs):
        self.posts.append((url, kwargs))
        return FakeResponse(self.payload_text)

    def get(self, url, **kwargs):
        return FakeResponse(self.payload_text)


def _client(payload_text) -> BinanceCexClient:
    config = CexConfig(
        name="binance", base_url="https://x", ws_url="wss://x/ws",
        api_key_env="A", api_secret_env="B", recv_window_ms=3000,
    )
    secrets = SecretsConfig(
        binance_api_key="k", binance_api_secret="s",
        dex_wallet_private_key="0x" + "11" * 32,
    )
    client = BinanceCexClient(config, secrets, [make_pair()])
    client._session = FakeSession(payload_text)
    return client


def _order(order_type="MARKET", tif="IOC", price="1000") -> CexOrder:
    return CexOrder(
        order_id="local-1", pair=make_pair(), side="buy", type=order_type,
        price=D(price), size=D("0.5"), tif=tif, ts=0.0,
    )


# A real Binance spot MARKET response: no avgPrice, price is zero, the achieved
# price is only recoverable from fills[] or cummulativeQuoteQty (Binance's own
# spelling, with the doubled m).
MARKET_FILLED = """{
  "symbol": "ETHUSDT", "orderId": 111, "clientOrderId": "x",
  "transactTime": 1700000000000, "price": "0.00000000",
  "origQty": "0.50000000", "executedQty": "0.50000000",
  "cummulativeQuoteQty": "947.25000000", "status": "FILLED",
  "timeInForce": "IOC", "type": "MARKET", "side": "BUY",
  "fills": [
    {"price": "1894.00000000", "qty": "0.30000000", "commission": "0.0", "commissionAsset": "USDT"},
    {"price": "1894.75000000", "qty": "0.20000000", "commission": "0.0", "commissionAsset": "USDT"}
  ]
}"""

LIMIT_RESTING = """{
  "symbol": "ETHUSDT", "orderId": 222, "clientOrderId": "y",
  "transactTime": 1700000000000, "price": "1800.00000000",
  "origQty": "0.50000000", "executedQty": "0.00000000",
  "cummulativeQuoteQty": "0.00000000", "status": "NEW",
  "timeInForce": "GTC", "type": "LIMIT", "side": "BUY", "fills": []
}"""

UNKNOWN_STATUS = """{
  "symbol": "ETHUSDT", "orderId": 333, "transactTime": 1700000000000,
  "price": "0.00000000", "origQty": "0.50000000", "executedQty": "0.00000000",
  "cummulativeQuoteQty": "0.00000000", "status": "SOMETHING_NEW",
  "type": "MARKET", "side": "BUY", "fills": []
}"""

PARTIAL = """{
  "symbol": "ETHUSDT", "orderId": 444, "transactTime": 1700000000000,
  "price": "0.00000000", "origQty": "0.50000000", "executedQty": "0.20000000",
  "cummulativeQuoteQty": "378.80000000", "status": "PARTIALLY_FILLED",
  "type": "MARKET", "side": "BUY",
  "fills": [{"price": "1894.00000000", "qty": "0.20000000"}]
}"""


# --- the achieved price -------------------------------------------------


async def test_the_achieved_price_comes_from_the_fills():
    """0.3 at 1894.00 and 0.2 at 1894.75 is a weighted average of 1894.30."""
    client = _client(MARKET_FILLED)

    update = await client.create_order(_order())

    assert update.avg_fill_price == D("1894.30"), (
        f"got {update.avg_fill_price}; a market order's price is only in fills[] "
        f"or cummulativeQuoteQty, never in `price`"
    )


async def test_a_market_order_never_reports_a_zero_price():
    """The exact old defect: `price` is "0.00000000" on a market order, and it was
    being reported as the achieved price."""
    client = _client(MARKET_FILLED)

    update = await client.create_order(_order())

    assert update.avg_fill_price is not None
    assert update.avg_fill_price > 0


async def test_the_price_falls_back_to_the_quote_quantity_when_fills_are_absent():
    """`newOrderRespType=ACK` omits fills[], and cummulativeQuoteQty/executedQty
    is then the only correct source: 378.80 / 0.20 = 1894."""
    import json

    payload = json.loads(PARTIAL)
    del payload["fills"]
    client = _client(json.dumps(payload))

    update = await client.create_order(_order())

    assert update.avg_fill_price == D("1894")


async def test_an_unfilled_order_reports_no_price_rather_than_zero():
    """Zero is a number; None is the truth. A zero would flow into PnL as a real
    price and value the position at nothing."""
    client = _client(LIMIT_RESTING)

    update = await client.create_order(_order(order_type="LIMIT"))

    assert update.filled_size == 0
    assert update.avg_fill_price is None


# --- status -------------------------------------------------------------


async def test_a_resting_order_is_reported_as_new_not_partially_filled():
    client = _client(LIMIT_RESTING)

    update = await client.create_order(_order(order_type="LIMIT"))

    assert update.status == "new", (
        "a resting order with nothing filled must not claim a partial fill"
    )
    assert update.filled_size == 0


async def test_an_unknown_status_is_never_reported_as_a_fill():
    """A status Binance adds later must not silently become a claim about
    inventory."""
    client = _client(UNKNOWN_STATUS)

    update = await client.create_order(_order())

    assert update.status not in ("filled", "partially_filled"), (
        f"an unrecognised exchange status became {update.status!r}"
    )
    assert update.reason and "SOMETHING_NEW" in update.reason, (
        "the unrecognised status must be preserved for diagnosis"
    )


async def test_a_real_partial_fill_is_reported_as_such():
    client = _client(PARTIAL)

    update = await client.create_order(_order())

    assert update.status == "partially_filled"
    assert update.filled_size == D("0.2")


async def test_a_filled_order_is_reported_as_filled():
    client = _client(MARKET_FILLED)

    update = await client.create_order(_order())

    assert update.status == "filled"
    assert update.filled_size == D("0.5")


# --- request parameters -------------------------------------------------


async def test_the_orders_own_time_in_force_is_sent():
    """`CexOrder.tif` already defaulted to IOC and was ignored in favour of a
    hardcoded GTC. For arbitrage a resting order can fill after the opportunity
    is gone, leaving an unhedged position."""
    client = _client(MARKET_FILLED)

    await client.create_order(_order(order_type="LIMIT", tif="IOC"))

    _, kwargs = client._session.posts[0]
    assert kwargs["data"]["timeInForce"] == "IOC"


async def test_a_deliberate_gtc_is_still_honoured():
    """The field is respected, not overridden in the other direction either."""
    client = _client(MARKET_FILLED)

    await client.create_order(_order(order_type="LIMIT", tif="GTC"))

    _, kwargs = client._session.posts[0]
    assert kwargs["data"]["timeInForce"] == "GTC"


async def test_a_market_order_sends_no_time_in_force():
    """Binance rejects timeInForce on a MARKET order."""
    client = _client(MARKET_FILLED)

    await client.create_order(_order(order_type="MARKET"))

    _, kwargs = client._session.posts[0]
    assert "timeInForce" not in kwargs["data"]


async def test_the_configured_recv_window_is_sent():
    """`cex.recv_window_ms` is configured and validated, and was never sent -- so
    Binance applied its own 5000ms default and the setting did nothing."""
    client = _client(MARKET_FILLED)

    await client.create_order(_order())

    _, kwargs = client._session.posts[0]
    assert int(kwargs["data"]["recvWindow"]) == 3000


async def test_the_signature_covers_every_parameter():
    """A signature computed before a parameter is added is invalid, and the error
    Binance returns for that is not obviously about ordering."""
    client = _client(MARKET_FILLED)

    await client.create_order(_order())

    _, kwargs = client._session.posts[0]
    data = dict(kwargs["data"])
    signature = data.pop("signature")
    assert signature == client._get_signature(data), (
        "the signature does not match the parameters actually sent"
    )
