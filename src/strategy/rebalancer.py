import asyncio
from decimal import Decimal
from loguru import logger

from src.core.config import AppConfig
from src.core.types import CexOrder, MarketPair
from src.exchange.cex_base import CexClient

class Rebalancer:
    def __init__(self, config: AppConfig, cex_client: CexClient):
        self.config = config.inventory.rebalance
        self.cex_client = cex_client
        self.pairs_config = {p.cex_symbol: p for p in config.pairs}
        logger.info("Inventory rebalancer initialised.")

    async def run_rebalance_check(self, paper_run: bool = False):
        if not self.config.enable:
            logger.info("Inventory rebalancing is disabled.")
            return

        logger.info("Checking CEX inventory ratio...")
        
        if not self.pairs_config:
            logger.warning("No trading pairs configured; cannot rebalance.")
            return
        
        pair_config = list(self.pairs_config.values())[0]
        base_asset = pair_config.base
        quote_asset = pair_config.quote_cex # Use the CEX quote symbol
        market_pair = MarketPair(
            base=base_asset,
            quote_cex=quote_asset,
            quote_dex=quote_asset, # Rebalancing happens on CEX, so they are the same
            cex_symbol=pair_config.cex_symbol,
            dex_chain=pair_config.dex_chain,
            dex_pool_fee=pair_config.dex_pool_fee,
            base_precision=pair_config.base_precision if pair_config.base_precision is not None else 8,
            quote_precision=pair_config.quote_precision if pair_config.quote_precision is not None else 8,
        )

        # 1. fetch balances and price
        base_balance, quote_balance, quote = await asyncio.gather(
            self.cex_client.get_balance(base_asset),
            self.cex_client.get_balance(quote_asset),
            self.cex_client.get_quote(market_pair)
        )

        if base_balance < 0 or quote_balance < 0 or not quote:
            logger.error("Could not fetch complete balance or price data; skipping this rebalance.")
            return

        # 2. compute the current ratio
        base_value = base_balance * quote.price
        total_value = base_value + quote_balance
        if total_value == 0:
            logger.warning("Total asset value is zero; cannot compute a ratio.")
            return
        
        current_ratio = base_value / total_value
        target_ratio = Decimal(str(self.config.target_ratio))
        trigger_threshold = Decimal(str(self.config.trigger_bps)) / 10000

        logger.info(f"Asset: {base_asset}, current ratio: {current_ratio:.2%}, target ratio: {target_ratio:.2%}")

        # 3. decide whether a rebalance is needed
        if abs(current_ratio - target_ratio) > trigger_threshold:
            logger.warning(f"Asset ratio has drifted too far; triggering a rebalance.")
            # 4. compute the quantity to trade
            target_base_value = total_value * target_ratio
            value_to_trade = target_base_value - base_value
            amount_to_trade = abs(value_to_trade) / quote.price
            side = "buy" if value_to_trade > 0 else "sell"

            logger.info(f"Planning a CEX market {side} of {amount_to_trade:.6f} {base_asset}")

            if paper_run:
                logger.warning("Paper run enabled; skipping the actual trade.")
                return

            # 5. execute the trade
            try:
                order = CexOrder(pair=market_pair, side=side, type="MARKET", size=amount_to_trade, price=Decimal(0), order_id="", ts=0)
                update = await self.cex_client.create_order(order)
                if update.status == "FILLED":
                    logger.success("Rebalance trade succeeded.")
                else:
                    logger.error(f"Rebalance trade failed, order status: {update.status}")
            except Exception as e:
                logger.error(f"Error while executing the rebalance trade: {e}")
        else:
            logger.info("Asset ratio is within tolerance; no rebalance needed.")
