import asyncio
import orjson
import random
import time
import hmac
import hashlib
from decimal import Decimal
from typing import Optional, Dict, List, TYPE_CHECKING

import aiohttp
import websockets
from loguru import logger

from .rate_limit import WeightGovernor, governed_request
from ..core import clock
from ..core.config import CexConfig, SecretsConfig
from ..core.types import BookSnapshot, MarketPair, Quote
from .cex_base import CexClient, CexOrder, OrderUpdate

# Binance's own status vocabulary, mapped to ours. Exhaustive rather than
# defaulted: the previous version mapped BOTH "NEW" and anything unrecognised to
# "partially_filled", so a resting order and a status Binance had not yet invented
# both reported a partial fill -- a claim about inventory that did not exist.
_BINANCE_ORDER_STATUS = {
    "NEW": "new",
    "PENDING_NEW": "new",
    "PARTIALLY_FILLED": "partially_filled",
    "FILLED": "filled",
    "CANCELED": "canceled",
    "PENDING_CANCEL": "canceled",
    "EXPIRED": "canceled",
    # An IOC or FOK order that could not be matched. Distinct from CANCELED only
    # in cause, and both mean no further fill is coming.
    "EXPIRED_IN_MATCH": "canceled",
    "REJECTED": "rejected",
}


def _translate_order_status(raw: str):
    """Our status for an exchange status, plus a reason when it is unrecognised.

    An unknown status maps to "unknown", never to a fill or a rejection: the state
    is genuinely indeterminate, and the only safe reading is that it must be
    reconciled. The raw value is preserved so the log says what actually arrived.
    """
    mapped = _BINANCE_ORDER_STATUS.get(raw)
    if mapped is None:
        return "unknown", f"unrecognised exchange status {raw!r}"
    return mapped, None


def _achieved_price(data: dict, filled_size: Decimal):
    """The average price actually obtained, or None if nothing filled.

    Binance's spot order response has no `avgPrice` field, and `price` is the
    LIMIT price -- "0.00000000" for a market order. The previous code read
    `data.get('avgPrice') or data.get('price')`, so a market order's achieved price
    was reported as zero and a limit order's as the price asked for rather than the
    one obtained. PnL from either is fiction.

    Two correct sources, in order of preference:

    * `fills[]`, a size-weighted average of the actual matches. Present when
      newOrderRespType is FULL, which is the default for MARKET and LIMIT.
    * `cummulativeQuoteQty / executedQty` -- Binance's own spelling, with the
      doubled m. Exact, and available even on an ACK response.
    """
    if filled_size <= 0:
        # None, not zero: a zero would flow into PnL as a real price and value the
        # position at nothing.
        return None

    fills = data.get('fills') or []
    if fills:
        total_quote = Decimal("0")
        total_base = Decimal("0")
        for fill in fills:
            qty = Decimal(str(fill.get('qty', '0')))
            price = Decimal(str(fill.get('price', '0')))
            total_base += qty
            total_quote += qty * price
        if total_base > 0:
            return total_quote / total_base

    quote_qty = data.get('cummulativeQuoteQty')
    if quote_qty is not None:
        quote_total = Decimal(str(quote_qty))
        if quote_total > 0:
            return quote_total / filled_size

    logger.error(
        f"Order reports {filled_size} filled but neither fills[] nor "
        f"cummulativeQuoteQty gives an achieved price. The fill price is unknown; "
        f"reconcile from the exchange before relying on any PnL for it."
    )
    return None
from ..infra import metrics

if TYPE_CHECKING:  # pragma: no cover
    from ..infra.dashboard import DashboardPublisher

USER_STREAM_KEEPALIVE_SECONDS = 25 * 60  # under Binance's 30-minute requirement


class BinanceCexClient(CexClient):
    def __init__(
        self,
        config: CexConfig,
        secrets: SecretsConfig,
        pairs: List[MarketPair],
        dashboard_publisher: Optional["DashboardPublisher"] = None,
        governor: Optional[WeightGovernor] = None,
    ):
        # The exchange meters REST by weight per minute per IP, and a 418 ban
        # blocks the market-data WebSocket too -- so the client that places
        # orders is the one that most needs a budget. A governor passed in from
        # the application is shared with the scanners, which is what the per-IP
        # limit actually requires; one is created here only so no construction
        # path can end up unmetered.
        self.governor = governor or WeightGovernor(
            max_weight_per_minute=config.max_request_weight_per_minute,
            safety_fraction=config.request_weight_safety_fraction,
        )
        self.config = config
        self.secrets = secrets
        self.pairs = pairs
        self.base_url = config.base_url.rstrip('/')
        self.ws_url = config.ws_url.rstrip('/')
        self._session: Optional[aiohttp.ClientSession] = None
        self._ws_conn: Optional[websockets.WebSocketClientProtocol] = None
        self._ws_listener_task: Optional[asyncio.Task] = None
        self._user_stream_task: Optional[asyncio.Task] = None
        self._listen_key_keepalive_task: Optional[asyncio.Task] = None
        self.orderbooks: Dict[str, Dict[str, Decimal]] = {p.cex_symbol.replace('/', ''): {'bids': {}, 'asks': {}} for p in self.pairs}
        self.last_update_ids: Dict[str, int] = {}
        # Unix epoch seconds when each book last received a snapshot or diff.
        # Lets a stalled stream be detected instead of read as a quiet market.
        self._book_synced_at: Dict[str, float] = {}
        # Reconnect backoff bounds, in seconds. A fixed retry interval
        # turns a transient outage into a rate-limit ban.
        # When the feed last delivered ANY frame. Distinct from per-symbol
        # freshness: Binance suppresses unchanged books, so a quiet symbol
        # is not a stale feed.
        self._last_frame_at: float = 0.0
        self._reconnect_backoff_initial = 1.0
        self._reconnect_backoff_max = 60.0
        self._listen_key: Optional[str] = None
        self._closing = False
        self._order_pair_cache: Dict[str, MarketPair] = {}
        self.dashboard_publisher = dashboard_publisher

    async def connect(self):
        logger.info("Connecting to Binance...")
        self._closing = False
        self._session = aiohttp.ClientSession(
            headers={
                'X-MBX-APIKEY': self.secrets.binance_api_key.get_secret_value()
            }
        )
        # start the WebSocket listener task
        self._ws_listener_task = asyncio.create_task(self._ws_listener())
        await self._start_user_stream()
        logger.info("Binance client connected; WebSocket listener started.")

    async def close(self):
        self._closing = True
        if self._ws_listener_task:
            self._ws_listener_task.cancel()
        if self._user_stream_task:
            self._user_stream_task.cancel()
        if self._listen_key_keepalive_task:
            self._listen_key_keepalive_task.cancel()
        if self._listen_key:
            await self._delete_listen_key()
        if self._ws_conn:
            await self._ws_conn.close()
        if self._session:
            await self._session.close()
        logger.info("Binance connection closed.")

    def _get_signature(self, params: dict) -> str:
        query_string = '&'.join([f"{k}={v}" for k, v in params.items()])
        return hmac.new(self.secrets.binance_api_secret.get_secret_value().encode('utf-8'), query_string.encode('utf-8'), hashlib.sha256).hexdigest()

    async def _start_user_stream(self) -> None:
        if not self._session:
            logger.warning("User stream init failed: session not yet created.")
            return
        try:
            self._listen_key = await self._create_listen_key()
            if not self._listen_key:
                logger.error("Could not obtain a Binance listenKey; skipping the user data stream.")
                return
            logger.info("Obtained Binance listenKey; starting the user data stream.")
            self._user_stream_task = asyncio.create_task(self._user_stream_listener())
            self._listen_key_keepalive_task = asyncio.create_task(self._keepalive_user_stream())
        except Exception as exc:
            logger.error(f"Failed to initialise the Binance user data stream: {exc}")

    async def _create_listen_key(self) -> Optional[str]:
        assert self._session is not None
        async with self._session.post(f"{self.base_url}/api/v3/userDataStream") as response:
            body = await response.text()
            if response.status >= 400:
                logger.error(f"Failed to request a listenKey [{response.status}]: {body}")
                return None
            data = orjson.loads(body)
            return data.get('listenKey')

    async def _keepalive_user_stream(self) -> None:
        if not self._session:
            return
        while not self._closing and self._listen_key:
            await asyncio.sleep(USER_STREAM_KEEPALIVE_SECONDS)
            try:
                async with self._session.put(
                    f"{self.base_url}/api/v3/userDataStream",
                    params={'listenKey': self._listen_key}
                ) as response:
                    if response.status >= 400:
                        body = await response.text()
                        logger.warning(f"listenKey keepalive failed [{response.status}]: {body}")
                    else:
                        logger.debug("Binance listenKey kept alive.")
            except asyncio.CancelledError:  # pragma: no cover
                break
            except Exception as exc:  # pragma: no cover
                logger.warning(f"Exception during listenKey keepalive: {exc}")

    async def _delete_listen_key(self) -> None:
        if not self._session or not self._listen_key:
            return
        try:
            async with self._session.delete(
                f"{self.base_url}/api/v3/userDataStream",
                params={'listenKey': self._listen_key}
            ) as response:
                if response.status >= 400:
                    body = await response.text()
                    logger.warning(f"Failed to delete listenKey [{response.status}]: {body}")
        except Exception as exc:  # pragma: no cover
            logger.warning(f"Exception while deleting listenKey: {exc}")
        finally:
            self._listen_key = None

    async def _user_stream_listener(self) -> None:
        if not self._listen_key:
            return
        url = f"{self.ws_url}/{self._listen_key}"
        while not self._closing:
            try:
                async with websockets.connect(url, ping_interval=20) as ws:
                    logger.info("Connected to the Binance user data stream.")
                    async for message in ws:
                        data = orjson.loads(message)
                        event_type = data.get('e')
                        if event_type == 'executionReport':
                            await self._process_execution_report(data)
            except asyncio.CancelledError:  # pragma: no cover
                break
            except Exception as exc:
                if not self._closing:
                    logger.error(f"User data stream error: {exc}; retrying in 5s.")
                    await asyncio.sleep(5)

    async def _process_execution_report(self, event: dict) -> None:
        order_id = str(event.get('i'))
        status_raw = str(event.get('X', '')).lower()
        status_map = {
            "new": "partially_filled",
            "partially_filled": "partially_filled",
            "partial_fill": "partially_filled",
            "filled": "filled",
            "canceled": "canceled",
            "expired": "canceled",
            "rejected": "rejected",
        }
        status = status_map.get(status_raw, "partially_filled")

        last_filled = Decimal(str(event.get('l', '0')))
        cumulative_filled = Decimal(str(event.get('z', '0')))
        cumulative_quote = Decimal(str(event.get('Z', '0')))

        avg_fill_price = Decimal('0')
        if cumulative_filled > 0 and cumulative_quote > 0:
            avg_fill_price = cumulative_quote / cumulative_filled
        elif last_filled > 0:
            avg_fill_price = Decimal(str(event.get('L', '0')))

        reason = event.get('r')
        if reason and reason.upper() == 'NONE':
            reason = None

        ts_ms = event.get('E') or event.get('T') or int(time.time() * 1000)
        ts = float(ts_ms) / 1000 if ts_ms else time.time()

        update = OrderUpdate(
            order_id=order_id,
            status=status,
            filled_size=last_filled,
            avg_fill_price=avg_fill_price,
            reason=reason,
            ts=ts,
        )

        pair = self._order_pair_cache.get(order_id)
        if status in {"filled", "canceled", "rejected"}:
            self._order_pair_cache.pop(order_id, None)

        await self._publish_dashboard_fill(event, update, pair)

    async def _publish_dashboard_fill(self, event: dict, update: OrderUpdate, pair: Optional[MarketPair]) -> None:
        if not self.dashboard_publisher:
            return
        try:
            payload = {
                "type": "cex_fill",
                "source": "binance",
                "data": {
                    "order_id": update.order_id,
                    "symbol": pair.cex_symbol if pair else event.get('s'),
                    "side": event.get('S'),
                    "status": update.status,
                    "fill_size": float(update.filled_size),
                    "avg_fill_price": float(update.avg_fill_price) if update.avg_fill_price is not None else 0.0,
                    "cumulative_filled": float(Decimal(str(event.get('z', '0')))),
                    "cumulative_quote": float(Decimal(str(event.get('Z', '0')))),
                    "event_time": event.get('E'),
                }
            }
            await self.dashboard_publisher.publish(payload)
        except Exception as exc:  # pragma: no cover
            logger.error(f"Failed to publish fill information to the dashboard: {exc}")

    def _stream_url(self) -> str:
        """Build a combined *partial book depth* stream URL.

        Two Binance subtleties are load-bearing here:

        - `/ws/<name>` returns raw, unwrapped frames while `/stream?streams=`
          wraps each payload as {"stream": ..., "data": {...}}. Requesting
          several streams from `/ws/` yields unwrapped frames, so a consumer
          filtering on the "stream" key silently discards every update -- which
          is exactly how the book previously froze after its first snapshot.

        - `@depth` is a 1000ms *diff* stream; `@depth<N>@100ms` is a 100ms
          *snapshot* stream. Measured, the difference is 1 update/second
          versus 10, against a detector that polls five times a second.
        """
        base = self.ws_url
        if base.endswith("/ws"):
            base = base[: -len("/ws")]
        levels = self.config.book_depth_levels
        interval = self.config.book_update_ms
        streams = [
            f"{p.cex_symbol.replace('/', '').lower()}@depth{levels}@{interval}ms"
            for p in self.pairs
        ]
        return f"{base}/stream?streams={'/'.join(streams)}"

    async def _ws_listener(self):
        """Maintain local books from partial-depth snapshots.

        There is deliberately no REST snapshot and no resync path. Every frame
        is self-contained, so a missed or malformed frame costs 100ms of
        freshness and nothing else -- it cannot desynchronise a book. That
        removes the failure mode where a swallowed snapshot error left the
        book permanently stale while a per-event resync loop burned ~3000
        weight/minute per stuck symbol until the exchange banned the IP.
        """
        backoff = self._reconnect_backoff_initial
        while not self._closing:
            url = self._stream_url()
            try:
                async with websockets.connect(url, ping_interval=20) as ws:
                    self._ws_conn = ws
                    logger.info(f"Connected to the Binance WebSocket: {url}")
                    backoff = self._reconnect_backoff_initial
                    async for message in ws:
                        data = orjson.loads(message)
                        # Combined streams wrap the payload; tolerate raw frames.
                        payload = data.get("data") if "stream" in data else data
                        if payload:
                            await self._handle_ws_message(
                                payload, stream=data.get("stream")
                            )
            except asyncio.CancelledError:  # pragma: no cover
                break
            except Exception as e:
                if self._closing:
                    break
                logger.warning(
                    f"Binance WebSocket error: {e}. Reconnecting in {backoff:.1f}s."
                )

            if self._closing:
                break
            # Exponential backoff with jitter. A fixed short retry produced a
            # reconnect storm: N pairs x a snapshot each, repeated every few
            # seconds, is a straight path to a rate-limit ban.
            await asyncio.sleep(backoff * (1.0 + 0.25 * random.random()))
            backoff = min(backoff * 2, self._reconnect_backoff_max)

    async def _handle_ws_message(
        self, data: dict, stream: Optional[str] = None
    ) -> None:
        """Apply one partial-depth snapshot, replacing the book wholesale.

        A partial-depth stream never sends deletions, so merging into the
        existing book would leave orphaned levels below the top N indefinitely.
        Replacement is the only correct semantics.

        Partial-depth payloads carry NO symbol field, so `stream` -- the key
        from the combined-stream wrapper -- is the only routing information
        available. Without it, nothing beyond a single-pair setup could ever
        be updated.
        """
        symbol = self._resolve_symbol(data, stream)
        if symbol is None:
            return
        if symbol not in self.orderbooks:
            logger.debug(f"Ignoring frame for unsubscribed symbol {symbol}.")
            return
        if "bids" not in data or "asks" not in data:
            logger.debug(f"Ignoring non-depth frame for {symbol}.")
            return

        try:
            bids = {Decimal(price): Decimal(qty) for price, qty in data["bids"]}
            asks = {Decimal(price): Decimal(qty) for price, qty in data["asks"]}
        except (TypeError, ValueError, ArithmeticError) as exc:
            logger.warning(f"Malformed depth frame for {symbol}: {exc}")
            return

        # An empty frame must not blank a usable book and stamp it fresh.
        if not bids or not asks:
            logger.debug(f"Ignoring empty depth frame for {symbol}.")
            return

        book = self.orderbooks[symbol]
        book["bids"] = bids
        book["asks"] = asks
        now = clock.now()
        self.last_update_ids[symbol] = int(data.get("lastUpdateId", 0))
        self._book_synced_at[symbol] = now
        self._last_frame_at = now

    def _resolve_symbol(self, data: dict, stream: Optional[str]) -> Optional[str]:
        """Determine which book a frame belongs to.

        Preference order: an explicit symbol field (diff streams carry `s`),
        then the combined-stream name, then -- only when exactly one pair is
        configured -- that pair. The single-pair fallback exists so a
        minimal setup works without the wrapper; it must never be relied on
        for multi-pair routing, which is why an unsubscribed symbol is
        rejected by the caller rather than silently written somewhere.
        """
        if data.get("s"):
            return str(data["s"]).upper()
        if stream:
            return stream.split("@", 1)[0].upper()
        if len(self.pairs) == 1:
            return self.pairs[0].cex_symbol.replace('/', '')
        return None

    async def get_book(self, pair: MarketPair) -> Optional[BookSnapshot]:
        """Return the locally maintained depth ladder for this pair.

        Sourced entirely from the WebSocket-maintained book. There is
        deliberately no REST fallback: a REST depth call costs weight 50
        against a 6000/minute budget, so using it on the hot path would
        breach the limit well before the pair count became interesting.

        `_book_synced_at` is stamped by the snapshot and by every applied
        diff, so a stalled stream shows up as an ageing book rather than
        being silently mistaken for a quiet market.
        """
        symbol = pair.cex_symbol.replace('/', '')
        book = self.orderbooks.get(symbol)
        if not book or not book['bids'] or not book['asks']:
            return None

        synced_at = self._book_synced_at.get(symbol)
        if synced_at is None:
            return None

        # Sorted best-first: asks ascending, bids descending.
        asks = sorted(book['asks'].items())
        bids = sorted(book['bids'].items(), reverse=True)
        return BookSnapshot(
            pair=pair, bids=bids, asks=asks, timestamp=synced_at,
            feed_timestamp=self._last_frame_at,
        )

    async def get_quote(self, pair: MarketPair) -> Optional[Quote]:
        symbol = pair.cex_symbol.replace('/', '')
        book = self.orderbooks.get(symbol)

        if book and book['bids'] and book['asks']:
            # prefer the locally maintained order book
            best_bid = max(book['bids'].keys())
            best_ask = min(book['asks'].keys())
            bid_size = book['bids'][best_bid]
            ask_size = book['asks'][best_ask]
        else:
            # fall back to a direct API query if the local book is unavailable
            logger.info(f"Local order book for {symbol} unavailable; fetching a quote via the API...")
            if not self._session:
                logger.error("Session not initialized for API quote fetch.")
                return None
            
            params = {'symbol': symbol}
            try:
                data = await governed_request(
                    self._session, self.governor, "GET",
                    f"{self.base_url}/api/v3/ticker/bookTicker", params=params,
                )
                best_bid = Decimal(data['bidPrice'])
                best_ask = Decimal(data['askPrice'])
                bid_size = Decimal(data['bidQty'])
                ask_size = Decimal(data['askQty'])
                logger.info(f"Fetched {symbol} quote via the API: Bid={best_bid}, Ask={best_ask}")
            except aiohttp.ClientError as e:
                logger.error(f"Failed to fetch the {symbol} quote via the API: {e}")
                return None
            except Exception as e:
                logger.error(f"Unknown error while handling the API quote: {e}", exc_info=True)
                return None

        return Quote(
            pair=pair,
            price=(best_bid + best_ask) / 2,
            bid_price=best_bid,
            ask_price=best_ask,
            bid_size=bid_size,
            ask_size=ask_size,
            side="buy", # placeholder
            size=Decimal(0), # placeholder
            venue="CEX",
            timestamp=clock.now(),
        )

    async def create_order(self, order: CexOrder) -> OrderUpdate:
        if not self._session:
            raise ConnectionError("Session not initialized.")

        params = {
            'symbol': order.pair.cex_symbol.replace('/', ''),
            'side': order.side.upper(),
            'type': order.type.upper(),
            'quantity': f"{order.size:.{order.pair.base_precision}f}",
            # The one process clock, not a second call to time.time().
            'timestamp': clock.now_ms(),
            # Was never sent, so Binance applied its own 5000ms default and the
            # configured, validated value did nothing.
            'recvWindow': self.config.recv_window_ms,
        }
        if order.type == 'LIMIT':
            params['price'] = f"{order.price:.{order.pair.quote_precision}f}"
            # From the order, not hardcoded. `CexOrder.tif` already existed and
            # already defaulted to IOC, and this line said GTC -- the worst of the
            # three for arbitrage, because a resting order can fill minutes later
            # once the opportunity is gone, leaving an unhedged position.
            params['timeInForce'] = order.tif
        # Binance rejects timeInForce on a MARKET order, so it is set only above.

        params['signature'] = self._get_signature(params)
        
        try:
            async with self._session.post(f"{self.base_url}/api/v3/order", data=params) as response:
                body = await response.text()
                if response.status >= 400:
                    logger.error(f"Binance order placement failed [{response.status}]: {body}")
                    raise aiohttp.ClientResponseError(
                        response.request_info,
                        response.history,
                        status=response.status,
                        message=body,
                        headers=response.headers,
                    )
                
                data = orjson.loads(body)
                logger.info("CEX order created: %s", data)

                status_raw = str(data.get('status', '')).upper()
                status, reason = _translate_order_status(status_raw)

                filled_size = Decimal(str(data.get('executedQty', '0')))
                avg_fill_price = _achieved_price(data, filled_size)

                ts_ms = (
                    data.get('transactTime') or data.get('updateTime')
                    or clock.now_ms()
                )
                ts = float(ts_ms) / 1000

                order_id = str(data['orderId'])
                self._order_pair_cache[order_id] = order.pair

                if status == "unknown":
                    logger.error(
                        f"Binance returned an unrecognised order status "
                        f"{status_raw!r} for order {order_id}. Treating the state "
                        f"as indeterminate rather than as a fill; reconcile before "
                        f"trading this pair again."
                    )

                return OrderUpdate(
                    order_id=order_id,
                    status=status,
                    avg_fill_price=avg_fill_price,
                    filled_size=filled_size,
                    reason=reason,
                    ts=ts,
                )
        except aiohttp.ClientResponseError as e:
            logger.error("Failed to create the CEX order: [%s] %s", e.status, e.message)
            raise

    async def get_balance(self, asset: str) -> Decimal:
        """Return the available balance for a given asset."""
        # Binance uses ETH rather than WETH
        query_asset = "ETH" if asset.upper() == "WETH" else asset
        
        logger.debug(f"Fetching CEX balance for {asset} (queried as {query_asset})...")
        params = {
            'timestamp': int(time.time() * 1000),
            'recvWindow': self.config.recv_window_ms
        }
        params['signature'] = self._get_signature(params)
        try:
            async with self._session.get(f"{self.base_url}/api/v3/account", params=params) as response:
                data = await response.json(loads=orjson.loads)
                response.raise_for_status()
                
                for balance in data['balances']:
                    if balance['asset'] == query_asset:
                        return Decimal(balance['free'])
                return Decimal("0") # asset absent from the balance list
        except Exception as e:
            logger.error(f"Failed to fetch the CEX balance: {e}")
            return Decimal("-1") # a negative value signals a query error

    async def cancel_order(self, order_id: str, pair: MarketPair) -> bool:
        """Cancel one order. Returns whether the exchange confirmed it.

        This was a stub that logged "Simulated order cancellation" and returned
        True. Returning False would have been an obvious gap; returning True was a
        lie in the dangerous direction, because cancelling the unfilled leg is the
        standard unwind for a half-executed arbitrage -- and a caller told the
        cancel succeeded stops tracking an order that then fills, unhedged.

        Never raises. This runs on the unwind path, usually while already handling
        a failure, and an exception there would replace a recoverable position with
        an unhandled one.
        """
        if not self._session:
            logger.error(f"Cannot cancel order {order_id}: no session.")
            return False

        symbol = pair.cex_symbol.replace('/', '')
        params = {
            'symbol': symbol,
            'orderId': order_id,
            'timestamp': clock.now_ms(),
            'recvWindow': self.config.recv_window_ms,
        }
        params['signature'] = self._get_signature(params)

        try:
            async with self._session.delete(
                f"{self.base_url}/api/v3/order", params=params
            ) as response:
                body = await response.text()
                self.governor.observe_headers(response.headers)
                if response.status >= 400:
                    logger.error(
                        f"Failed to cancel order {order_id} on {symbol} "
                        f"[{response.status}]: {body}"
                    )
                    return False
                logger.info(f"Cancelled order {order_id} on {symbol}.")
                return True
        except Exception as exc:
            logger.error(
                f"Error cancelling order {order_id} on {symbol}: "
                f"{type(exc).__name__}: {exc}"
            )
            return False

    async def cancel_all_orders(self) -> int:
        """Cancel every open order on every configured symbol.

        Returns how many the exchange reported cancelling, so the caller can log a
        number rather than an assumption.

        Binance's DELETE /api/v3/openOrders is per symbol, so "all" is one request
        each. A single call would have cleared only the first pair and left the
        rest resting -- which is exactly the state `cancel_all_on_start` exists to
        prevent.

        One symbol failing does not stop the others: a symbol with no open orders
        returns an error, and treating that as fatal would leave every later
        symbol uncleared.
        """
        if not self._session:
            logger.error("Cannot cancel open orders: no session.")
            return 0

        total = 0
        for symbol in sorted({p.cex_symbol.replace('/', '') for p in self.pairs}):
            params = {
                'symbol': symbol,
                'timestamp': clock.now_ms(),
                'recvWindow': self.config.recv_window_ms,
            }
            params['signature'] = self._get_signature(params)
            try:
                async with self._session.delete(
                    f"{self.base_url}/api/v3/openOrders", params=params
                ) as response:
                    body = await response.text()
                    self.governor.observe_headers(response.headers)
                    if response.status >= 400:
                        # -2011 "Unknown order sent" is what an empty book returns.
                        logger.info(
                            f"No open orders cancelled on {symbol} "
                            f"[{response.status}]: {body}"
                        )
                        continue
                    cancelled = orjson.loads(body)
                    count = len(cancelled) if isinstance(cancelled, list) else 1
                    total += count
                    logger.info(f"Cancelled {count} open order(s) on {symbol}.")
            except Exception as exc:
                logger.error(
                    f"Error cancelling open orders on {symbol}: "
                    f"{type(exc).__name__}: {exc}"
                )
        return total
