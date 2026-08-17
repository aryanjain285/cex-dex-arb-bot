import asyncio
import pytest
from decimal import Decimal

from loguru import logger

from src.core.config import PairConfig
from src.core.types import MarketPair, Opportunity
from src.strategy.executor import TransactionExecutor, PaperExecutor

CEX_TAKER_FEE_BPS = Decimal("7.5")
TEN_THOUSAND = Decimal("10000")


@pytest.fixture
def pair_config() -> PairConfig:
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
        price_floor_quote=Decimal("100"),
        price_ceiling_quote=Decimal("10000"),
        max_edge_bps=50000,
        base_precision=4,
        quote_precision=2,
    )


@pytest.fixture
def market_pair(pair_config: PairConfig) -> MarketPair:
    return MarketPair(
        base=pair_config.base,
        quote_cex=pair_config.quote,
        quote_dex=pair_config.quote,
        cex_symbol=pair_config.cex_symbol,
        dex_chain=pair_config.dex_chain,
        dex_pool_fee=pair_config.dex_pool_fee,
        base_precision=pair_config.base_precision,
        quote_precision=pair_config.quote_precision,
    )


def make_opportunity(
    market_pair: MarketPair,
    pair_config: PairConfig,
    direction: str,
    cex_price: Decimal,
    dex_price: Decimal,
    size: Decimal = Decimal("0.1"),
    gas_cost: Decimal = Decimal("0"),
) -> Opportunity:
    slippage_bps = Decimal(pair_config.max_slippage_bps)
    cex_fee_quote = (cex_price * size * CEX_TAKER_FEE_BPS) / TEN_THOUSAND

    if direction == "CEX_to_DEX":
        buy_price = cex_price
        sell_price = dex_price
    else:
        buy_price = dex_price
        sell_price = cex_price

    edge_bps = (sell_price / buy_price - Decimal(1)) * TEN_THOUSAND
    gross = size * (sell_price - buy_price)
    mid_price = (sell_price + buy_price) / Decimal(2)
    slippage_cost = (mid_price * size * slippage_bps) / TEN_THOUSAND
    expected_pnl = gross - cex_fee_quote - slippage_cost - gas_cost

    return Opportunity(
        pair=market_pair,
        direction=direction,
        size=size,
        cex_price=cex_price,
        dex_price=dex_price,
        dex_chain=market_pair.dex_chain,
        dex_pool_fee=market_pair.dex_pool_fee,
        edge_bps=edge_bps,
        slippage_bps=slippage_bps,
        gas_cost_quote=gas_cost,
        cex_fee_quote=cex_fee_quote,
        expected_pnl_quote=expected_pnl,
        valid_until=0.0,
    )


def test_dex_to_cex_legs(pair_config: PairConfig, market_pair: MarketPair):
    executor = TransactionExecutor(None, None, None, [pair_config])
    opp = make_opportunity(
        market_pair,
        pair_config,
        direction="DEX_to_CEX",
        cex_price=Decimal("4500"),
        dex_price=Decimal("4000"),
    )

    summary = asyncio.run(executor.run(opp))

    assert [leg.venue for leg in summary.legs] == ["DEX", "CEX"]
    assert [leg.side for leg in summary.legs] == ["buy", "sell"]
    assert summary.legs[0].fees_quote == Decimal("0")
    assert summary.legs[1].fees_quote == Decimal("0.3375")
    assert summary.pnl_quote == Decimal("49.45")
    assert summary.edge_bps == Decimal("1250")
    assert summary.edge_bps > 0


def test_cex_to_dex_legs(pair_config: PairConfig, market_pair: MarketPair):
    executor = TransactionExecutor(None, None, None, [pair_config])
    opp = make_opportunity(
        market_pair,
        pair_config,
        direction="CEX_to_DEX",
        cex_price=Decimal("4000"),
        dex_price=Decimal("4500"),
    )

    summary = asyncio.run(executor.run(opp))

    assert [leg.venue for leg in summary.legs] == ["CEX", "DEX"]
    assert [leg.side for leg in summary.legs] == ["buy", "sell"]
    assert summary.legs[0].fees_quote == Decimal("0.3")
    assert summary.legs[1].fees_quote == Decimal("0")
    assert summary.pnl_quote == Decimal("49.4875")
    assert summary.edge_bps == Decimal("1250")
    assert summary.edge_bps > 0


def test_invalid_price_rejected(pair_config: PairConfig, market_pair: MarketPair):
    executor = TransactionExecutor(None, None, None, [pair_config])
    opp = make_opportunity(
        market_pair,
        pair_config,
        direction="DEX_to_CEX",
        cex_price=Decimal("4500"),
        dex_price=Decimal("0.0002"),
    )

    summary = asyncio.run(executor.run(opp))

    assert summary.legs == []
    assert summary.pnl_quote == Decimal("0")
    assert summary.edge_bps == Decimal("0")


def test_paper_executor_log_format(pair_config: PairConfig, market_pair: MarketPair):
    paper = PaperExecutor([pair_config])
    opp = make_opportunity(
        market_pair,
        pair_config,
        direction="DEX_to_CEX",
        cex_price=Decimal("4500"),
        dex_price=Decimal("4000"),
    )

    captured = []
    handler_id = logger.add(lambda msg: captured.append(str(msg)), format="{message}")
    try:
        summary = asyncio.run(paper.run(opp))
    finally:
        logger.remove(handler_id)

    assert summary.legs

    paper_logs = [msg for msg in captured if "[PAPER MODE] Opportunity detected" in msg]
    assert paper_logs, "should have captured the paper-trading opportunity log"
    assert all("%s" not in log_msg and "%.4f" not in log_msg for log_msg in paper_logs)
