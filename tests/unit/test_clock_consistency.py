"""Every timestamp in the pipeline must sit on one comparable time base.

Regression tests for a real defect: quote timestamps were produced from
`asyncio.get_running_loop().time()` (a monotonic clock with an arbitrary
epoch) while execution timestamps used `time.time()` (unix epoch). The two
are not comparable, so staleness checks and markout measurement are both
impossible to express correctly on top of them.
"""
import time
from decimal import Decimal

import pytest

from src.core.config import CexConfig, SecretsConfig
from src.core.types import MarketPair


def _cex_config() -> CexConfig:
    return CexConfig(
        name="binance",
        base_url="https://example.invalid",
        ws_url="wss://example.invalid/ws",
        api_key_env="BINANCE_API_KEY",
        api_secret_env="BINANCE_API_SECRET",
        recv_window_ms=5000,
    )


def _secrets() -> SecretsConfig:
    return SecretsConfig(
        binance_api_key="test-key",
        binance_api_secret="test-secret",
        dex_wallet_private_key="0x" + "11" * 32,
    )


@pytest.fixture
def pair() -> MarketPair:
    return MarketPair(
        base="WETH",
        quote_cex="USDT",
        quote_dex="USDT",
        cex_symbol="ETH/USDT",
        dex_chain="ethereum",
        dex_pool_fee=500,
    )


async def test_cex_quote_timestamp_is_unix_epoch_seconds(pair: MarketPair):
    """A Quote's timestamp must be unix epoch seconds.

    Would fail if get_quote returns loop time: on any host with meaningful
    uptime the monotonic clock differs from the unix epoch by ~1.7e9.
    """
    from src.exchange.binance import BinanceCexClient

    client = BinanceCexClient(_cex_config(), _secrets(), [pair])
    book = client.orderbooks["ETHUSDT"]
    book["bids"] = {Decimal("1900"): Decimal("5")}
    book["asks"] = {Decimal("1901"): Decimal("5")}

    quote = await client.get_quote(pair)

    assert quote is not None
    assert abs(quote.timestamp - time.time()) < 5.0, (
        f"quote.timestamp={quote.timestamp} is not on the unix epoch "
        f"(time.time()={time.time()})"
    )


async def test_opportunity_valid_until_is_comparable_to_wall_clock(pair: MarketPair):
    """valid_until must be an absolute unix timestamp in the near future.

    Would fail if built from loop time: comparing it against time.time()
    would then either always pass or always fail depending on process uptime.
    """
    from src.core.config import StrategyConfig
    from src.strategy.detector import OpportunityDetector
    from tests.fakes import FakeCex, FakeDex, flat_book

    cex = FakeCex({"ETH/USDT": flat_book(bid=1900, ask=1900)})
    dex = FakeDex(sell_price=1960, buy_price=1960, gas=Decimal("0.01"))

    detector = OpportunityDetector(
        StrategyConfig(
            target_notional_usd=1000,
            taker_fee_bps=Decimal("7.5"),
            min_net_bps=Decimal("5"),
        ),
        cex,
        dex,
        [pair],
    )
    opportunities = await detector.detect()

    assert opportunities, "fixture should produce an opportunity"
    valid_until = opportunities[0].valid_until
    now = time.time()
    assert now < valid_until < now + 60, (
        f"valid_until={valid_until} is not a near-future unix timestamp "
        f"(now={now})"
    )
