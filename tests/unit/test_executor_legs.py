"""Leg construction and economics passthrough.

The executor must not re-derive trade economics. It previously recomputed PnL
with its own slippage deduction, which double-counted price impact already
inside the DEX quote -- so for the same trade the executor and the detector
disagreed about what it was worth. The detector owns the economics now; the
executor's job is sanity gating, leg normalisation, and reporting.
"""
from decimal import Decimal

import asyncio

import pytest
from loguru import logger

from src.core import clock
from src.core.config import PairConfig
from src.core.types import MarketPair, Opportunity
from src.strategy.costs import evaluate_trade
from src.strategy.executor import PaperExecutor, TransactionExecutor

TAKER_FEE_BPS = Decimal("7.5")


@pytest.fixture
def pair_config() -> PairConfig:
    return PairConfig(
        base="WETH",
        quote="USDT",
        cex_symbol="ETH/USDT",
        max_slippage_bps=5,
        max_size_quote=5000,
        dex_chain="ethereum",
        dex_pool_fee=500,
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
        max_slippage_bps=pair_config.max_slippage_bps,
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
    """Build an Opportunity the way the detector does -- via evaluate_trade.

    Using the production cost function here rather than a hand-rolled formula
    is deliberate: it keeps the fixture from drifting away from the real model.
    """
    econ = evaluate_trade(
        direction=direction,
        size_base=size,
        cex_price=cex_price,
        dex_price=dex_price,
        taker_fee_bps=TAKER_FEE_BPS,
        gas_quote=gas_cost,
    )
    return Opportunity(
        pair=market_pair,
        direction=direction,
        size=size,
        cex_price=cex_price,
        dex_price=dex_price,
        dex_chain=market_pair.dex_chain,
        dex_pool_fee=market_pair.dex_pool_fee,
        edge_bps=econ.net_bps,
        slippage_bps=Decimal(pair_config.max_slippage_bps),
        gas_cost_quote=gas_cost,
        cex_fee_quote=econ.cex_fee_quote,
        expected_pnl_quote=econ.net_quote,
        # A live deadline. This was 0.0 while `valid_until` was written by the
        # detector and read by nobody -- which is precisely the hazard of an
        # unenforced field: no fixture author ever had to think about it. The
        # deadline itself is covered by test_opportunity_expiry.py.
        valid_until=clock.now() + 60.0,
    )


def test_dex_to_cex_legs(pair_config: PairConfig, market_pair: MarketPair):
    executor = TransactionExecutor(None, None, None, [pair_config])
    opp = make_opportunity(
        market_pair, pair_config, direction="DEX_to_CEX",
        cex_price=Decimal("4500"), dex_price=Decimal("4000"),
    )

    summary = asyncio.run(executor.run(opp))

    assert [leg.venue for leg in summary.legs] == ["DEX", "CEX"]
    assert [leg.side for leg in summary.legs] == ["buy", "sell"]
    assert summary.legs[0].fees_quote == Decimal("0")
    assert summary.legs[1].fees_quote == Decimal("0.3375")  # 4500*0.1*0.00075


def test_cex_to_dex_legs(pair_config: PairConfig, market_pair: MarketPair):
    executor = TransactionExecutor(None, None, None, [pair_config])
    opp = make_opportunity(
        market_pair, pair_config, direction="CEX_to_DEX",
        cex_price=Decimal("4000"), dex_price=Decimal("4500"),
    )

    summary = asyncio.run(executor.run(opp))

    assert [leg.venue for leg in summary.legs] == ["CEX", "DEX"]
    assert [leg.side for leg in summary.legs] == ["buy", "sell"]
    assert summary.legs[0].fees_quote == Decimal("0.3")     # 4000*0.1*0.00075
    assert summary.legs[1].fees_quote == Decimal("0")


@pytest.mark.parametrize("direction", ["CEX_to_DEX", "DEX_to_CEX"])
def test_executor_reports_the_detectors_pnl_without_recomputing_it(
    direction, pair_config: PairConfig, market_pair: MarketPair
):
    """The regression guard.

    slippage_bps is deliberately non-zero. A recomputing executor would
    subtract it and disagree with the opportunity it was handed.
    """
    executor = TransactionExecutor(None, None, None, [pair_config])
    opp = make_opportunity(
        market_pair, pair_config, direction=direction,
        cex_price=Decimal("4500") if direction == "DEX_to_CEX" else Decimal("4000"),
        dex_price=Decimal("4000") if direction == "DEX_to_CEX" else Decimal("4500"),
        gas_cost=Decimal("0.25"),
    )
    assert opp.slippage_bps > 0, "fixture must carry a tolerance to be meaningful"

    summary = asyncio.run(executor.run(opp))

    assert summary.pnl_quote == opp.expected_pnl_quote
    assert summary.edge_bps == opp.edge_bps
    assert summary.gas_quote == opp.gas_cost_quote


def test_invalid_price_rejected(pair_config: PairConfig, market_pair: MarketPair):
    executor = TransactionExecutor(None, None, None, [pair_config])
    opp = make_opportunity(
        market_pair, pair_config, direction="DEX_to_CEX",
        cex_price=Decimal("4500"), dex_price=Decimal("0.0002"),
    )

    summary = asyncio.run(executor.run(opp))

    assert summary.legs == []
    assert summary.pnl_quote == Decimal("0")
    assert summary.edge_bps == Decimal("0")


def test_paper_executor_logs_and_reports_the_same_economics(
    pair_config: PairConfig, market_pair: MarketPair
):
    paper = PaperExecutor([pair_config])
    opp = make_opportunity(
        market_pair, pair_config, direction="DEX_to_CEX",
        cex_price=Decimal("4500"), dex_price=Decimal("4000"),
    )

    captured = []
    handler_id = logger.add(lambda msg: captured.append(str(msg)), format="{message}")
    try:
        summary = asyncio.run(paper.run(opp))
    finally:
        logger.remove(handler_id)

    assert summary.legs
    assert summary.pnl_quote == opp.expected_pnl_quote

    paper_logs = [m for m in captured if "[PAPER MODE] Opportunity detected" in m]
    assert paper_logs, "should have captured the paper-trading opportunity log"
    assert all("%s" not in m and "%.4f" not in m for m in paper_logs)
