"""Order book freshness and the feed protocol.

Measured live: the configured `@depth` stream delivers one update per SECOND
while the detector polls every 200ms. So the book was evaluated five times per
update at ~500ms average age, worst case 1000ms+. ETH moves 2-4 bps/second
and `min_net_bps` is 5 -- the staleness was the same magnitude as the entire
edge threshold, and biased: an edge appears precisely when the stale side is
wrong in your favour.

The fix is `@depth<N>@100ms`, a *partial book depth* stream. Verified live:
each frame carries keys {bids, asks, lastUpdateId} and is a COMPLETE snapshot
of the top N levels at 100ms cadence, 20 levels covering ~$111k of ETHUSDT
bid-side notional. That deletes the REST snapshot, the sequence arithmetic,
the resync path and its rate-limit storm -- and makes "time since last
message" a precise staleness measure, because a 500ms gap on a 100ms stream
is unambiguously abnormal.
"""
from decimal import Decimal

import pytest

from src.core import clock
from src.core.config import CexConfig, SecretsConfig, TokenPolicyConfig
from src.core.types import MarketPair


def _cex_config(**kw) -> CexConfig:
    defaults = dict(
        name="binance", base_url="https://example.invalid",
        ws_url="wss://example.invalid/ws", api_key_env="A", api_secret_env="B",
        recv_window_ms=5000,
    )
    defaults.update(kw)
    return CexConfig(**defaults)


def _secrets() -> SecretsConfig:
    return SecretsConfig(binance_api_key="k", binance_api_secret="s",
                         dex_wallet_private_key="0x" + "11" * 32)


@pytest.fixture
def pair() -> MarketPair:
    return MarketPair(base="WETH", quote_cex="USDT", quote_dex="USDT",
                      cex_symbol="ETH/USDT", dex_chain="ethereum", dex_pool_fee=500)


def client(pair, **cfg):
    from src.exchange.binance import BinanceCexClient
    return BinanceCexClient(_cex_config(**cfg), _secrets(), [pair])


# --------------------------------------------------------------------------

def test_stream_url_requests_a_fast_partial_depth_stream(pair):
    """The 1 Hz default was the defect. Cadence and depth must be explicit."""
    c = client(pair)
    url = c._stream_url()

    assert "@depth20@100ms" in url, f"expected a 100ms partial-depth stream, got {url}"
    assert "/stream?streams=" in url, "must use the combined-stream endpoint"
    assert "@depth/" not in url and not url.endswith("@depth"), (
        "must not use the 1000ms diff stream"
    )


def test_stream_cadence_and_depth_are_configurable(pair):
    c = client(pair, book_depth_levels=10, book_update_ms=1000)
    assert "@depth10@1000ms" in c._stream_url()


def test_invalid_depth_or_cadence_is_rejected():
    """Binance accepts only specific values; a typo must fail at load, not
    silently produce a stream that never delivers."""
    with pytest.raises(Exception):
        _cex_config(book_depth_levels=7)
    with pytest.raises(Exception):
        _cex_config(book_update_ms=250)


async def test_partial_depth_snapshot_replaces_the_book(pair):
    """Each frame is a complete snapshot, so it replaces rather than merges.

    Merging a snapshot into a stale book would leave orphaned levels below the
    top N forever, since a partial stream never sends deletions.
    """
    c = client(pair)
    c.orderbooks["ETHUSDT"]["bids"] = {Decimal("1"): Decimal("1")}   # stale junk
    c.orderbooks["ETHUSDT"]["asks"] = {Decimal("9"): Decimal("1")}

    await c._handle_ws_message({
        "lastUpdateId": 100,
        "bids": [["1900.10", "2.0"], ["1900.00", "3.0"]],
        "asks": [["1900.20", "1.5"], ["1900.30", "4.0"]],
    })

    book = await c.get_book(pair)
    assert book is not None
    assert book.best_bid == Decimal("1900.10")
    assert book.best_ask == Decimal("1900.20")
    assert Decimal("1") not in dict(book.bids), "stale level must be gone"
    assert Decimal("9") not in dict(book.asks), "stale level must be gone"


async def test_book_is_ordered_best_first(pair):
    c = client(pair)
    await c._handle_ws_message({
        "lastUpdateId": 1,
        "bids": [["1900.00", "1"], ["1900.10", "1"]],   # deliberately unsorted
        "asks": [["1900.30", "1"], ["1900.20", "1"]],
    })
    book = await c.get_book(pair)

    bid_prices = [p for p, _ in book.bids]
    ask_prices = [p for p, _ in book.asks]
    assert bid_prices == sorted(bid_prices, reverse=True), "bids must descend"
    assert ask_prices == sorted(ask_prices), "asks must ascend"


async def test_freshness_is_stamped_on_every_frame(pair):
    c = client(pair)
    before = clock.now()
    await c._handle_ws_message({
        "lastUpdateId": 1, "bids": [["1900", "1"]], "asks": [["1901", "1"]],
    })
    book = await c.get_book(pair)

    assert before <= book.timestamp <= clock.now()
    assert book.age_seconds(clock.now()) < 1.0


async def test_a_frame_with_no_levels_does_not_blank_a_good_book(pair):
    """Defensive: an empty or malformed frame must be ignored rather than
    destroying a usable book and stamping it fresh."""
    c = client(pair)
    await c._handle_ws_message({
        "lastUpdateId": 1, "bids": [["1900", "1"]], "asks": [["1901", "1"]],
    })
    good = await c.get_book(pair)

    await c._handle_ws_message({"lastUpdateId": 2, "bids": [], "asks": []})
    after = await c.get_book(pair)

    assert after is not None, "a bad frame must not blank the book"
    assert after.best_bid == good.best_bid


async def test_get_book_returns_none_before_any_frame_arrives(pair):
    c = client(pair)
    assert await c.get_book(pair) is None


def test_default_max_book_age_is_tight_enough_to_detect_a_stall():
    """5.0s permitted five consecutive missed updates on the old 1 Hz stream.
    On a 100ms stream the guard must be far tighter to mean anything."""
    from src.core.config import StrategyConfig, TokenPolicyConfig

    assert StrategyConfig().max_book_age_seconds <= 1.0


# --------------------------------------------------------------------------
# Multi-pair routing. Partial-depth payloads carry NO symbol field, so the
# stream name in the combined-stream wrapper is the only routing key. A
# handler that cannot use it silently updates nothing once there is more
# than one pair -- which is every real deployment.
# --------------------------------------------------------------------------

def _pair(symbol, base):
    return MarketPair(base=base, quote_cex="USDT", quote_dex="USDT",
                      cex_symbol=symbol, dex_chain="ethereum", dex_pool_fee=500)


async def test_frames_are_routed_by_stream_name_with_multiple_pairs():
    from src.exchange.binance import BinanceCexClient

    eth, arb = _pair("ETH/USDT", "WETH"), _pair("ARB/USDT", "ARB")
    c = BinanceCexClient(_cex_config(), _secrets(), [eth, arb])

    await c._handle_ws_message(
        {"lastUpdateId": 1, "bids": [["1900", "1"]], "asks": [["1901", "1"]]},
        stream="ethusdt@depth20@100ms",
    )
    await c._handle_ws_message(
        {"lastUpdateId": 2, "bids": [["0.40", "10"]], "asks": [["0.41", "10"]]},
        stream="arbusdt@depth20@100ms",
    )

    eth_book = await c.get_book(eth)
    arb_book = await c.get_book(arb)
    assert eth_book is not None and arb_book is not None
    assert eth_book.best_bid == Decimal("1900")
    assert arb_book.best_bid == Decimal("0.40")


async def test_a_frame_for_an_unknown_stream_is_ignored():
    from src.exchange.binance import BinanceCexClient

    eth = _pair("ETH/USDT", "WETH")
    c = BinanceCexClient(_cex_config(), _secrets(), [eth])

    await c._handle_ws_message(
        {"lastUpdateId": 1, "bids": [["1", "1"]], "asks": [["2", "1"]]},
        stream="dogeusdt@depth20@100ms",
    )
    assert await c.get_book(eth) is None, "must not write into an unrelated book"


# --------------------------------------------------------------------------
# Staleness: feed liveness vs market quiet.
#
# Measured over 15s on one connection: ethusdt 150 frames (median gap 100ms,
# max 200ms), arbusdt 46 (max 1500ms), dogeusdt 30 (max 2600ms). Every frame
# carried a unique lastUpdateId, so Binance SUPPRESSES unchanged books rather
# than republishing them.
#
# Per-symbol frame age therefore measures MARKET QUIET, not FEED STALLED. A
# guard that rejects on it would refuse to trade precisely the illiquid pairs
# this strategy exists to trade. Connection liveness is the real signal: if no
# frame has arrived for ANY symbol, the feed is down and every book is suspect.
# --------------------------------------------------------------------------

async def test_a_quiet_book_is_still_valid():
    """An unchanged price is current, not stale. DOGE went 2.6s between
    frames while its quoted price remained correct."""
    from src.exchange.binance import BinanceCexClient

    quiet = _pair("DOGE/USDT", "DOGE")
    liquid = _pair("ETH/USDT", "WETH")
    c = BinanceCexClient(_cex_config(), _secrets(), [quiet, liquid])

    await c._handle_ws_message(
        {"lastUpdateId": 1, "bids": [["0.10", "1000"]], "asks": [["0.11", "1000"]]},
        stream="dogeusdt@depth20@100ms")
    # 3 seconds pass with no DOGE change, but the feed stays live via ETH
    c._book_synced_at["DOGEUSDT"] -= 3.0
    await c._handle_ws_message(
        {"lastUpdateId": 2, "bids": [["1900", "1"]], "asks": [["1901", "1"]]},
        stream="ethusdt@depth20@100ms")

    book = await c.get_book(quiet)
    assert book is not None
    assert book.feed_age_seconds(clock.now()) < 1.0, (
        "the FEED is live -- a quiet symbol must not look like a dead feed"
    )
    assert book.age_seconds(clock.now()) > 2.0, (
        "per-symbol age should still report the true quiet interval"
    )


async def test_a_dead_feed_invalidates_every_book():
    """If no frame has arrived for any symbol, all books are suspect."""
    from src.exchange.binance import BinanceCexClient

    eth = _pair("ETH/USDT", "WETH")
    c = BinanceCexClient(_cex_config(), _secrets(), [eth])
    await c._handle_ws_message(
        {"lastUpdateId": 1, "bids": [["1900", "1"]], "asks": [["1901", "1"]]},
        stream="ethusdt@depth20@100ms")

    c._last_frame_at -= 30.0          # connection went silent
    book = await c.get_book(eth)
    assert book.feed_age_seconds(clock.now()) > 29.0


async def test_detector_rejects_on_feed_staleness_not_symbol_quiet():
    """The behaviour that matters: a quiet illiquid pair must still trade,
    while a dead feed must stop everything."""
    from src.core.config import StrategyConfig
    from src.core.types import BookSnapshot
    from src.strategy.detector import OpportunityDetector
    from tests.fakes import FakeDex, flat_book

    pair = _pair("DOGE/USDT", "DOGE")
    bids, asks = flat_book(bid=100, ask=100)

    class Cex:
        def __init__(self, symbol_age, feed_age):
            self.symbol_age, self.feed_age = symbol_age, feed_age
        async def get_book(self, p):
            now = clock.now()
            return BookSnapshot(pair=p, bids=bids, asks=asks,
                                timestamp=now - self.symbol_age,
                                feed_timestamp=now - self.feed_age)

    cfg = StrategyConfig(target_notional_usd=1000, taker_fee_bps=Decimal("7.5"),
                         min_net_bps=Decimal(5), max_book_age_seconds=0.5,
                         # This test is about feed staleness, not the token
                         # policy, and its ticker is a placeholder.
                         token_policy=TokenPolicyConfig(mode="denylist"))
    dex = FakeDex(sell_price=110, buy_price=110)

    quiet = await OpportunityDetector(cfg, Cex(symbol_age=3.0, feed_age=0.1), dex, [pair]).detect()
    assert quiet, "a quiet symbol on a live feed must still produce opportunities"

    dead = await OpportunityDetector(cfg, Cex(symbol_age=0.1, feed_age=30.0), dex, [pair]).detect()
    assert not dead, "a dead feed must reject every book"
