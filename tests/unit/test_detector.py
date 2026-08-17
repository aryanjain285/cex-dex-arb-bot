import asyncio
import pytest
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock

from src.core.types import MarketPair, CexQuote
from src.core.config import PairConfig, StrategyConfig
from src.strategy.detector import OpportunityDetector

# mock pair configuration
@pytest.fixture
def mock_pair_config() -> PairConfig:
    return PairConfig(
        base="WETH",
        quote="USDT",
        cex_symbol="ETH/USDT",
        min_edge_bps=10,
        max_slippage_bps=5,
        max_size_quote=5000,
        dex_chain="ethereum",
        dex_pool_fee=500,
        edge_safety_multiplier=Decimal("1.2"),
        base_precision=4,
        quote_precision=2,
    )

# mock MarketPair
@pytest.fixture
def market_pair() -> MarketPair:
    return MarketPair(
        base="WETH",
        quote_cex="USDT",
        quote_dex="USDT",
        cex_symbol="ETH/USDT",
        dex_chain="ethereum",
        dex_pool_fee=500,
        base_precision=4,
        quote_precision=2,
    )

# mock StrategyConfig
@pytest.fixture
def mock_strategy_config() -> StrategyConfig:
    return StrategyConfig(target_notional_usd=10, min_edge_bps=5, max_slippage_bps=15)

# mock CEX and DEX clients
@pytest.fixture
def mock_clients():
    cex_client = AsyncMock()
    dex_client = AsyncMock()
    return cex_client, dex_client

def test_no_opportunity(mock_clients, mock_strategy_config, market_pair):
    """No arbitrage opportunity available."""
    cex_client, dex_client = mock_clients

    # set the CEX quote
    cex_client.get_quote.return_value = CexQuote(
        bid_price=Decimal("2999.5"), ask_price=Decimal("3000.5"), ts=0
    )
    # set the DEX quote
    async def dex_quote_side_effect(pair, size, side, estimate_gas):
        if side == "buy": # DEX ask
            return MagicMock(price=Decimal("3001"), gas_cost_quote=Decimal("0.01"))
        else: # DEX bid
            return MagicMock(price=Decimal("2999"), gas_cost_quote=Decimal("0.01"))
    dex_client.get_quote.side_effect = dex_quote_side_effect

    detector = OpportunityDetector(mock_strategy_config, cex_client, dex_client, [market_pair])
    opportunities = asyncio.run(detector.detect())

    assert len(opportunities) == 0

def test_cex_to_dex_opportunity(mock_clients, mock_strategy_config, market_pair):
    """Buy on the CEX, sell on the DEX."""
    cex_client, dex_client = mock_clients

    # CEX cheap, DEX expensive.
    # The detector's bar is dynamic: (taker_fee + slippage + gas_bps) * safety.
    # At a $10 target notional a $0.01 gas cost is ~10 bps, so the effective
    # threshold here is (7.5 + 10 + 10) * 1.2 = 33 bps. The DEX sell price
    # below clears that with room to spare.
    cex_client.get_quote.return_value = CexQuote(
        bid_price=Decimal("2999.5"), ask_price=Decimal("3000.5"), ts=0
    )
    async def dex_quote_side_effect(pair, size, side, estimate_gas):
        if side == "buy":
            return MagicMock(price=Decimal("3019"), gas_cost_quote=Decimal("0.01"))
        else:
            return MagicMock(price=Decimal("3020"), gas_cost_quote=Decimal("0.01"))
    dex_client.get_quote.side_effect = dex_quote_side_effect

    detector = OpportunityDetector(mock_strategy_config, cex_client, dex_client, [market_pair])
    opportunities = asyncio.run(detector.detect())

    assert len(opportunities) == 1
    opp = opportunities[0]
    assert opp.direction == "CEX_to_DEX"
    assert opp.cex_price == Decimal("3000.5") # CEX ask price
    assert opp.dex_price == Decimal("3020")   # DEX bid price
    assert opp.edge_bps > Decimal("33")       # cleared the dynamic threshold


def test_dex_to_cex_opportunity(mock_clients, mock_strategy_config, market_pair):
    """Buy on the DEX, sell on the CEX."""
    cex_client, dex_client = mock_clients

    # DEX cheap, CEX expensive. Same ~33 bps dynamic threshold as above;
    # the DEX buy price is set low enough to clear it.
    cex_client.get_quote.return_value = CexQuote(
        bid_price=Decimal("3009.5"), ask_price=Decimal("3010.5"), ts=0
    )
    async def dex_quote_side_effect(pair, size, side, estimate_gas):
        if side == "buy":
            return MagicMock(price=Decimal("2990"), gas_cost_quote=Decimal("0.01"))
        else:
            return MagicMock(price=Decimal("2989"), gas_cost_quote=Decimal("0.01"))
    dex_client.get_quote.side_effect = dex_quote_side_effect

    detector = OpportunityDetector(mock_strategy_config, cex_client, dex_client, [market_pair])
    opportunities = asyncio.run(detector.detect())

    assert len(opportunities) == 1
    opp = opportunities[0]
    assert opp.direction == "DEX_to_CEX"
    assert opp.cex_price == Decimal("3009.5") # CEX bid price
    assert opp.dex_price == Decimal("2990")   # DEX ask price
    assert opp.edge_bps > Decimal("33")       # cleared the dynamic threshold


def test_opportunity_below_threshold(mock_clients, mock_strategy_config, market_pair):
    """Spread exists but is below the minimum threshold."""
    cex_client, dex_client = mock_clients

    # a spread exists, but it is small
    cex_client.get_quote.return_value = CexQuote(
        bid_price=Decimal("2999.9"), ask_price=Decimal("3000.1"), ts=0
    )
    async def dex_quote_side_effect(pair, size, side, estimate_gas):
        if side == "buy":
            return MagicMock(price=Decimal("3000.2"), gas_cost_quote=Decimal("0.01"))
        else:
            return MagicMock(price=Decimal("3000"), gas_cost_quote=Decimal("0.01"))
    dex_client.get_quote.side_effect = dex_quote_side_effect

    detector = OpportunityDetector(mock_strategy_config, cex_client, dex_client, [market_pair])
    opportunities = asyncio.run(detector.detect())

    assert len(opportunities) == 0


def test_opportunity_filtered_by_dynamic_threshold(mock_clients, mock_strategy_config, market_pair):
    """An edge between the static and dynamic thresholds should be filtered out."""
    cex_client, dex_client = mock_clients

    cex_client.get_quote.return_value = CexQuote(
        bid_price=Decimal("2999.5"), ask_price=Decimal("3000.5"), ts=0
    )

    async def dex_quote_side_effect(pair, size, side, estimate_gas):
        if side == "buy":
            return MagicMock(price=Decimal("3001.0"), gas_cost_quote=Decimal("0.01"))
        else:
            return MagicMock(price=Decimal("3003.1"), gas_cost_quote=Decimal("0.01"))

    dex_client.get_quote.side_effect = dex_quote_side_effect

    detector = OpportunityDetector(mock_strategy_config, cex_client, dex_client, [market_pair])
    opportunities = asyncio.run(detector.detect())

    assert len(opportunities) == 0
