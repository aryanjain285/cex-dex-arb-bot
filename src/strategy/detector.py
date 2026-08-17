from typing import List, Optional, Tuple, Dict
from decimal import Decimal
import asyncio
import time

from loguru import logger

from src.core.config import StrategyConfig
from src.core.types import MarketPair, Opportunity, CexQuote
from src.exchange.cex_base import CexClient
from src.exchange.dex_base import DexClient
from src.infra.metrics import opportunities_found

# --- Constants ---
TEN_THOUSAND = Decimal(10000)
ZERO = Decimal(0)
CEX_TAKER_FEE_BPS = Decimal(7.5) # Example: 0.075%
DEFAULT_SAFETY_MULTIPLIER = Decimal("1.2")

class OpportunityDetector:
    def __init__(self, strategy_config: StrategyConfig, cex_client: CexClient, dex_client: DexClient, pairs: List[MarketPair]):
        self.strategy_config = strategy_config
        self.cex_client = cex_client
        self.dex_client = dex_client
        self.pairs = pairs
        self._intermediate_price_cache: Dict[str, Tuple[Tuple[Decimal, Decimal], float]] = {}
        logger.info(f"Opportunity detector initialised, monitoring {len(pairs)} pairs.")

    async def detect(self) -> List[Opportunity]:
        tasks = [self.check_pair(p) for p in self.pairs]
        results = await asyncio.gather(*tasks)
        opportunities = [opp for opp in results if opp]
        return opportunities

    async def check_pair(self, pair: MarketPair) -> Optional[Opportunity]:
        if pair.is_synthetic:
            return await self._check_synthetic_pair(pair)
        else:
            return await self._check_direct_pair(pair)

    async def _check_direct_pair(self, pair: MarketPair) -> Optional[Opportunity]:
        cex_quote = await self.cex_client.get_quote(pair)
        if not (cex_quote and cex_quote.bid_price and cex_quote.ask_price and cex_quote.ask_price > 0):
            return None

        trade_size = Decimal(str(self.strategy_config.target_notional_usd)) / cex_quote.ask_price
        
        dex_buy_task = self.dex_client.get_quote(pair, size=trade_size, side="buy", estimate_gas=True)
        dex_sell_task = self.dex_client.get_quote(pair, size=trade_size, side="sell", estimate_gas=True)
        dex_buy_quote, dex_sell_quote = await asyncio.gather(dex_buy_task, dex_sell_task)

        if not (dex_buy_quote and dex_sell_quote):
            return None

        return self._evaluate_prices(
            pair, 
            cex_quote.ask_price, cex_quote.bid_price, 
            dex_buy_quote.price, dex_sell_quote.price, 
            dex_buy_quote.gas_cost_quote, dex_sell_quote.gas_cost_quote, 
            trade_size
        )

    async def _get_cached_intermediate_price(self, pair_config: MarketPair) -> Optional[Tuple[Decimal, Decimal]]:
        now = time.time()
        record = self._intermediate_price_cache.get(pair_config.cex_symbol)
        if record:
            (cached_ask, cached_bid), ts = record
            if now - ts < 2:
                return (cached_ask, cached_bid)

        quote = await self.cex_client.get_quote(pair_config)
        if not (quote and quote.ask_price and quote.bid_price):
            return None

        prices = (quote.ask_price, quote.bid_price)
        self._intermediate_price_cache[pair_config.cex_symbol] = (prices, now)
        return prices

    async def _check_synthetic_pair(self, pair: MarketPair) -> Optional[Opportunity]:
        intermediate_symbol = pair.intermediate_symbol
        if not intermediate_symbol: return None

        intermediate_cex_pair_symbol = f"{intermediate_symbol}/{pair.quote_cex}"
        intermediate_cex_pair = next((p for p in self.pairs if p.cex_symbol == intermediate_cex_pair_symbol), None)
        if not intermediate_cex_pair:
            logger.warning(f"Cannot check synthetic pair {pair.cex_symbol}: Intermediate pair {intermediate_cex_pair_symbol} not configured.")
            return None

        # --- Step 1: Fetch all required prices concurrently ---
        cex_target_q_task = self.cex_client.get_quote(pair)
        intermediate_prices_task = self._get_cached_intermediate_price(intermediate_cex_pair)
        
        # We need a rough CEX price to estimate trade size first
        cex_target_q_prelim = await self.cex_client.get_quote(pair)
        if not (cex_target_q_prelim and cex_target_q_prelim.ask_price > 0):
            return None
        trade_size = Decimal(str(self.strategy_config.target_notional_usd)) / cex_target_q_prelim.ask_price

        dex_sell_q_task = self.dex_client.get_quote(pair, size=trade_size, side="sell", estimate_gas=True)
        dex_buy_q_task = self.dex_client.get_quote(pair, size=trade_size, side="buy", estimate_gas=True)

        results = await asyncio.gather(cex_target_q_task, intermediate_prices_task, dex_sell_q_task, dex_buy_q_task, return_exceptions=True)
        cex_target_q, intermediate_prices, dex_sell_q, dex_buy_q = results

        if any(isinstance(r, Exception) for r in results) or not all(results):
            logger.debug(f"Could not fetch all required quotes for synthetic pair {pair.cex_symbol}")
            return None

        # --- Step 2: Assertions and Sanity Checks ---
        assert intermediate_cex_pair.base_decimals == 18, f"Intermediate token {intermediate_symbol} decimals should be 18"
        cex_intermediate_ask, cex_intermediate_bid = intermediate_prices
        if dex_sell_q.price <= 0 or dex_buy_q.price <= 0 or cex_intermediate_ask <= 0 or cex_intermediate_bid <= 0:
            logger.debug(f"Invalid quotes for synthetic pair {pair.cex_symbol}")
            return None

        # --- Step 3: Currency Alignment (The Core Fix) ---
        synthetic_dex_sell_price = dex_sell_q.price * cex_intermediate_bid
        synthetic_dex_buy_price  = dex_buy_q.price  * cex_intermediate_ask

        logger.debug(
            f"Synthetic Price for {pair.cex_symbol}: "
            f"DEX({pair.base}/{pair.quote_dex})={dex_sell_q.price:.4f} * CEX({intermediate_symbol}/{pair.quote_cex})={cex_intermediate_bid:.2f} -> Sell at {synthetic_dex_sell_price:.4f} {pair.quote_cex}"
        )

        # --- Step 4: Pass aligned prices and costs to evaluation ---
        # For synthetic trades, we approximate total gas cost by doubling the DEX leg cost
        total_gas_cost_buy = dex_buy_q.gas_cost_quote * 2
        total_gas_cost_sell = dex_sell_q.gas_cost_quote * 2

        return self._evaluate_prices(
            pair, 
            cex_target_q.ask_price, cex_target_q.bid_price, 
            synthetic_dex_buy_price, synthetic_dex_sell_price,
            total_gas_cost_buy, total_gas_cost_sell,
            trade_size
        )

    def _evaluate_prices(
        self, 
        pair: MarketPair, 
        cex_ask_price: Decimal, 
        cex_bid_price: Decimal, 
        dex_buy_price: Decimal, 
        dex_sell_price: Decimal,
        dex_buy_gas_quote: Decimal,
        dex_sell_gas_quote: Decimal,
        size: Decimal
    ) -> Optional[Opportunity]:
        # Sanity check all incoming prices
        for v_name, v in [("cex_ask_price", cex_ask_price), ("cex_bid_price", cex_bid_price), ("dex_buy_price", dex_buy_price), ("dex_sell_price", dex_sell_price)]:
            if v is None or v <= 0:
                logger.debug(f"Invalid {v_name} for {pair.cex_symbol} in evaluation")
                return None

        # Bounded BPS check to prevent pollution from abnormal data
        def _bounded_bps(n: Decimal) -> bool:
            return abs(n) <= Decimal(100_000)  # 1000% edge limit

        edge_bps_1_pre_fee = (dex_sell_price - cex_ask_price) / cex_ask_price * TEN_THOUSAND
        edge_bps_2_pre_fee = (cex_bid_price - dex_buy_price) / dex_buy_price * TEN_THOUSAND

        if not (_bounded_bps(edge_bps_1_pre_fee) and _bounded_bps(edge_bps_2_pre_fee)):
            logger.warning(f"Abnormal edge for {pair.cex_symbol}: edge1={edge_bps_1_pre_fee:.2f}, edge2={edge_bps_2_pre_fee:.2f}. Skip.")
            return None

        slippage_bps = Decimal(pair.max_slippage_bps or 10)
        safety_multiplier = pair.edge_safety_multiplier if pair.edge_safety_multiplier is not None else DEFAULT_SAFETY_MULTIPLIER
        static_threshold = Decimal(pair.min_edge_bps or 0)

        # Direction 1: CEX_to_DEX
        gas_cost_1 = dex_sell_gas_quote
        notional_1 = size * cex_ask_price
        gas_bps_1 = (gas_cost_1 / notional_1 * TEN_THOUSAND) if notional_1 > 0 else ZERO
        dynamic_min_edge_1 = (CEX_TAKER_FEE_BPS + slippage_bps + gas_bps_1) * safety_multiplier
        effective_min_edge_1 = max(static_threshold, dynamic_min_edge_1)

        if edge_bps_1_pre_fee >= effective_min_edge_1:
            opp = self._create_opportunity(pair=pair, direction="CEX_to_DEX", cex_price=cex_ask_price, dex_price=dex_sell_price, edge_bps=edge_bps_1_pre_fee, size=size, slippage_bps=slippage_bps, gas_cost_quote=gas_cost_1)
            opportunities_found.labels(pair=pair.cex_symbol, direction="CEX_to_DEX").inc()
            return opp

        # Direction 2: DEX_to_CEX
        gas_cost_2 = dex_buy_gas_quote
        notional_2 = size * dex_buy_price
        gas_bps_2 = (gas_cost_2 / notional_2 * TEN_THOUSAND) if notional_2 > 0 else ZERO
        dynamic_min_edge_2 = (CEX_TAKER_FEE_BPS + slippage_bps + gas_bps_2) * safety_multiplier
        effective_min_edge_2 = max(static_threshold, dynamic_min_edge_2)

        if edge_bps_2_pre_fee >= effective_min_edge_2:
            opp = self._create_opportunity(pair=pair, direction="DEX_to_CEX", cex_price=cex_bid_price, dex_price=dex_buy_price, edge_bps=edge_bps_2_pre_fee, size=size, slippage_bps=slippage_bps, gas_cost_quote=gas_cost_2)
            opportunities_found.labels(pair=pair.cex_symbol, direction="DEX_to_CEX").inc()
            return opp
            
        return None

    def _create_opportunity(self, pair: MarketPair, direction: str, cex_price: Decimal, dex_price: Decimal, edge_bps: Decimal, size: Decimal, slippage_bps: Decimal, gas_cost_quote: Decimal) -> Opportunity:
        cex_fee_quote = (cex_price * size * CEX_TAKER_FEE_BPS) / TEN_THOUSAND
        if direction == "CEX_to_DEX":
            buy_price = cex_price
            sell_price = dex_price
        else:
            buy_price = dex_price
            sell_price = cex_price

        gross_pnl = (sell_price - buy_price) * size
        mid_price = (sell_price + buy_price) / Decimal(2)
        slippage_cost = (mid_price * size * slippage_bps) / TEN_THOUSAND
        net_pnl = gross_pnl - cex_fee_quote - slippage_cost - gas_cost_quote

        return Opportunity(
            pair=pair,
            direction=direction,
            size=size,
            cex_price=cex_price,
            dex_price=dex_price,
            dex_chain=pair.dex_chain,
            dex_pool_fee=pair.dex_pool_fee,
            edge_bps=edge_bps,
            slippage_bps=slippage_bps,
            gas_cost_quote=gas_cost_quote,
            cex_fee_quote=cex_fee_quote,
            expected_pnl_quote=net_pnl,
            valid_until=asyncio.get_running_loop().time() + 2
        )