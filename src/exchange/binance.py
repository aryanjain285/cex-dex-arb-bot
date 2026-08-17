import asyncio
import orjson
import time
import hmac
import hashlib
from decimal import Decimal
from typing import Optional, Dict, List, TYPE_CHECKING

import aiohttp
import websockets
from loguru import logger

from ..core.config import CexConfig, SecretsConfig
from ..core.types import MarketPair, Quote
from .cex_base import CexClient, CexOrder, OrderUpdate
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
    ):
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

    def _combined_stream_url(self, streams: List[str]) -> str:
        """
        Build a Binance *combined* stream URL.

        The two endpoints behave differently and are not interchangeable:
          - `/ws/<name>`            -> a single raw stream, payloads unwrapped
          - `/stream?streams=a/b/c` -> many streams, each payload wrapped as
                                       {"stream": ..., "data": {...}}

        Requesting several streams from `/ws/` yields raw, unwrapped frames, so
        any consumer that filters on the "stream" key silently discards every
        update and the local order book is never advanced past its snapshot.
        """
        base = self.ws_url
        if base.endswith("/ws"):
            base = base[: -len("/ws")]
        return f"{base}/stream?streams={'/'.join(streams)}"

    async def _ws_listener(self):
        streams = [f"{p.cex_symbol.replace('/', '').lower()}@depth" for p in self.pairs]
        url = self._combined_stream_url(streams)

        while True:
            try:
                async with websockets.connect(url) as ws:
                    self._ws_conn = ws
                    logger.info(f"Connected to the Binance WebSocket: {url}")
                    # after connecting, sync the order book for every pair
                    await self._sync_all_orderbooks()

                    async for message in ws:
                        data = orjson.loads(message)
                        # combined streams wrap the payload; tolerate raw frames too
                        payload = data.get("data") if "stream" in data else data
                        if payload:
                            await self._handle_ws_message(payload)
            except websockets.exceptions.ConnectionClosed as e:
                logger.warning(f"Binance WebSocket connection closed: {e}. Reconnecting in 5s...")
            except Exception as e:
                logger.error(f"WebSocket listener error: {e}. Reconnecting in 5s...", exc_info=True)
            
            await asyncio.sleep(5)

    async def _sync_all_orderbooks(self):
        for pair in self.pairs:
            await self._sync_orderbook(pair)

    async def _sync_orderbook(self, pair: MarketPair):
        symbol = pair.cex_symbol.replace('/', '')
        logger.debug(f"Syncing the order book snapshot for {symbol}...")
        if not self._session:
            raise ConnectionError("Session not initialized.")

        params = {'symbol': symbol, 'limit': 1000}
        try:
            async with self._session.get(f"{self.base_url}/api/v3/depth", params=params) as response:
                response.raise_for_status()
                data = await response.json(loads=orjson.loads)
                
                self.last_update_ids[symbol] = data['lastUpdateId']
                self.orderbooks[symbol]['bids'] = {Decimal(price): Decimal(qty) for price, qty in data['bids']}
                self.orderbooks[symbol]['asks'] = {Decimal(price): Decimal(qty) for price, qty in data['asks']}
                logger.info(f"{symbol} order book synced. LastUpdateId: {data['lastUpdateId']}")
        except aiohttp.ClientError as e:
            logger.error(f"Failed to fetch {symbol} order book snapshot failed: {e}")

    async def _handle_ws_message(self, data: dict):
        if data.get('e') != 'depthUpdate':
            return

        symbol = data['s']
        first_update_id = data['U']
        final_update_id = data['u']

        if symbol not in self.last_update_ids:
            logger.warning(f"Received {symbol}  WebSocket update received, but the order book is not yet synced.")
            return

        # per the Binance docs, resync when first_update_id > last_update_id + 1
        if first_update_id > self.last_update_ids[symbol] + 1:
            logger.warning(f"{symbol} WebSocket update gap detected; resyncing...")
            await self._sync_orderbook(next(p for p in self.pairs if p.cex_symbol.replace('/','') == symbol))
            return

        # only apply the update when final_update_id > last_update_id
        if final_update_id <= self.last_update_ids[symbol]:
            return

        # apply the update to the order book
        for price, qty in data['b']: # Bids
            p, q = Decimal(price), Decimal(qty)
            if q == 0:
                self.orderbooks[symbol]['bids'].pop(p, None)
            else:
                self.orderbooks[symbol]['bids'][p] = q
        
        for price, qty in data['a']: # Asks
            p, q = Decimal(price), Decimal(qty)
            if q == 0:
                self.orderbooks[symbol]['asks'].pop(p, None)
            else:
                self.orderbooks[symbol]['asks'][p] = q
        
        self.last_update_ids[symbol] = final_update_id

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
                async with self._session.get(f"{self.base_url}/api/v3/ticker/bookTicker", params=params) as response:
                    response.raise_for_status()
                    data = await response.json(loads=orjson.loads)
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
            timestamp=asyncio.get_running_loop().time()
        )

    async def create_order(self, order: CexOrder) -> OrderUpdate:
        if not self._session:
            raise ConnectionError("Session not initialized.")

        params = {
            'symbol': order.pair.cex_symbol.replace('/', ''),
            'side': order.side.upper(),
            'type': order.type.upper(),
            'quantity': f"{order.size:.{order.pair.base_precision}f}",
            'timestamp': int(time.time() * 1000)
        }
        if order.type == 'LIMIT':
            params['price'] = f"{order.price:.{order.pair.quote_precision}f}"
            params['timeInForce'] = 'GTC'

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

                status_raw = str(data.get('status', '')).lower()
                status_map = {
                    "new": "partially_filled",
                    "partially_filled": "partially_filled",
                    "partial_fill": "partially_filled",
                    "partial_filled": "partially_filled",
                    "filled": "filled",
                    "canceled": "canceled",
                    "expired": "canceled",
                    "rejected": "rejected",
                }
                status = status_map.get(status_raw, "partially_filled")

                filled_size = Decimal(str(data.get('executedQty', '0')))
                avg_price_value = data.get('avgPrice') or data.get('price') or '0'
                avg_fill_price = Decimal(str(avg_price_value))
                ts_ms = data.get('transactTime') or data.get('updateTime') or int(time.time() * 1000)
                ts = float(ts_ms) / 1000 if ts_ms else time.time()

                order_id = str(data['orderId'])
                self._order_pair_cache[order_id] = order.pair

                return OrderUpdate(
                    order_id=str(data['orderId']),
                    status=status,
                    avg_fill_price=avg_fill_price,
                    filled_size=filled_size,
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
        # TODO: implement order cancellation
        logger.info(f"Simulated order cancellation: {order_id}")
        return True
