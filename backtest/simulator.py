import asyncio
from decimal import Decimal
from typing import Optional, List
import pandas as pd
from loguru import logger

from src.core.types import MarketPair, Quote, CexOrder, OrderUpdate, DexSwapParams, DexTxReceipt, ExecutionSummary
from src.exchange.cex_base import CexClient
from src.exchange.dex_base import DexClient
from src.strategy.detector import OpportunityDetector
from src.strategy.router import SimpleRouter
from src.strategy.executor import TransactionExecutor
from src.risk.limits import RiskManager
from src.core.config import AppConfig

# --- mock clients ---

class BacktestCexClient(CexClient):
    """A mock CEX client that reads from a dataset."""
    def __init__(self, pair: MarketPair):
        self.pair = pair
        self._current_tick: Optional[pd.Series] = None

    def set_tick(self, tick: pd.Series):
        self._current_tick = tick

    async def connect(self): pass
    async def close(self): pass

    async def get_quote(self, pair: MarketPair) -> Optional[Quote]:
        if self._current_tick is None:
            return None
        return Quote(
            pair=self.pair,
            price=(self._current_tick['cex_bid_price'] + self._current_tick['cex_ask_price']) / 2,
            size=Decimal(1), # mock size
            venue="CEX",
            timestamp=self._current_tick.name.timestamp(),
            bid_price=Decimal(str(self._current_tick['cex_bid_price'])),
            ask_price=Decimal(str(self._current_tick['cex_ask_price']))
        )

    async def create_order(self, order: CexOrder) -> OrderUpdate:
        # simulate a market order filling immediately at the current tick price
        fill_price = self._current_tick['cex_ask_price'] if order.side == 'buy' else self._current_tick['cex_bid_price']
        return OrderUpdate(
            order_id=f"backtest_cex_{self._current_tick.name.timestamp()}",
            status="FILLED",
            avg_fill_price=Decimal(str(fill_price)),
            filled_size=order.size
        )

class BacktestDexClient(DexClient):
    """A mock DEX client that reads from a dataset."""
    def __init__(self, pair: MarketPair):
        self.pair = pair
        self._current_tick: Optional[pd.Series] = None

    def set_tick(self, tick: pd.Series):
        self._current_tick = tick

    async def get_quote(self, pair: MarketPair, size: Decimal, side: str) -> Optional[Quote]:
        if self._current_tick is None:
            return None
        # simplified model: assume fixed slippage
        price = self._current_tick['dex_price']
        slippage = Decimal("0.001") # 0.1%
        if side == 'buy':
            price *= (1 + slippage)
        else:
            price *= (1 - slippage)
        return Quote(pair=pair, side=side, price=Decimal(str(price)), size=size, venue="DEX", timestamp=self._current_tick.name.timestamp())

    async def execute_swap(self, params: DexSwapParams) -> DexTxReceipt:
        # simulate an immediate fill
        quote = await self.get_quote(params.pair, params.size, params.side)
        return DexTxReceipt(
            tx_hash=f"0x_backtest_dex_{self._current_tick.name.timestamp()}",
            status=True,
            block_number=0,
            gas_used=150000,
            effective_gas_price=int(self._current_tick.get('gas_price_gwei', 50) * 1e9),
            avg_fill_price=quote.price,
            filled_size=params.size
        )

# --- simulator ---

class Simulator:
    def __init__(self, config: AppConfig, data: pd.DataFrame):
        self.config = config
        self.data = data
        self.results: List[ExecutionSummary] = []
        
        # the backtest only covers the first configured pair
        self.pair_config = self.config.pairs[0]
        self.market_pair = MarketPair(
            base=self.pair_config.base, 
            quote_cex=self.pair_config.quote, 
            quote_dex=self.pair_config.quote, # Backtest assumes direct pairs
            cex_symbol=self.pair_config.cex_symbol,
            dex_chain=self.pair_config.dex_chain,
            dex_pool_fee=self.pair_config.dex_pool_fee,
            base_precision=self.pair_config.base_precision if self.pair_config.base_precision is not None else 8,
            quote_precision=self.pair_config.quote_precision if self.pair_config.quote_precision is not None else 8,
        )

        # initialise the mock clients
        self.cex_client = BacktestCexClient(self.market_pair)
        self.dex_client = BacktestDexClient(self.market_pair)

        # initialise the strategy components
        self.detector = OpportunityDetector(self.config.strategy, self.cex_client, self.dex_client, [self.market_pair])
        self.router = SimpleRouter()
        self.risk_manager = RiskManager(self.config.risk)
        self.executor = TransactionExecutor(self.cex_client, self.dex_client, self.risk_manager)

    async def run(self):
        logger.info("Backtest starting...")
        for timestamp, tick in self.data.iterrows():
            # push the current market data into the mock clients
            self.cex_client.set_tick(tick)
            self.dex_client.set_tick(tick)

            # run the detect -> plan -> execute cycle
            opportunities = await self.detector.detect()
            if not opportunities:
                continue

            for opp in opportunities:
                logger.info(f"[{timestamp}] Opportunity found: {opp.direction} | edge: {opp.edge_bps:.2f} bps")
                plan = self.router.plan(opp)
                if self.risk_manager.is_trade_allowed(plan):
                    summary = await self.executor.run(plan)
                    if summary.is_complete_success:
                        self.results.append(summary)
                        logger.success(f"  └──> Trade succeeded, PnL: {summary.net_pnl_quote:.4f} {self.market_pair.quote}")
                    else:
                        logger.warning("  └──> Trade failed or was hedged")
        logger.success("Backtest complete.")

    def report(self):
        if not self.results:
            print("\n--- Backtest Report ---")
            print("No successful trades.")
            return

        total_pnl = sum(s.net_pnl_quote for s in self.results)
        trade_count = len(self.results)
        win_trades = [s for s in self.results if s.net_pnl_quote > 0]
        loss_trades = [s for s in self.results if s.net_pnl_quote <= 0]
        win_rate = len(win_trades) / trade_count if trade_count > 0 else 0

        print("\n--- Backtest Report ---")
        print(f"Period: {self.data.index.min()} to {self.data.index.max()}")
        print(f"Total trades: {trade_count}")
        print(f"Total PnL: {total_pnl:.4f} {self.market_pair.quote}")
        print(f"Win rate: {win_rate:.2%}")
        if win_trades:
            avg_win = sum(t.net_pnl_quote for t in win_trades) / len(win_trades)
            print(f"Average win: {avg_win:.4f}")
        if loss_trades:
            avg_loss = sum(t.net_pnl_quote for t in loss_trades) / len(loss_trades)
            print(f"Average loss: {avg_loss:.4f}")
        print("------------------")
