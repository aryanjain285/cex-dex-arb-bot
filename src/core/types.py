from __future__ import annotations
from typing import List, Optional, Literal
from decimal import Decimal
from pydantic import BaseModel
from dataclasses import dataclass

ZERO = Decimal("0")

__all__ = [
    "MarketPair",
    "BookSnapshot",
    "Quote",
    "Opportunity",
    "CexOrder",
    "OrderUpdate",
    "ExecutionLeg",
    "ExecutionSummary",
    "DexSwapParams",
    "DexTxReceipt",
]

# --- Core Data Structures ---

class MarketPair(BaseModel):
    base: str
    quote_cex: str  # The quote currency for CEX and user-facing representation (e.g., USDT)
    quote_dex: str  # The quote currency on the DEX (e.g., WETH for synthetic pairs)
    cex_symbol: str  # e.g., "ETHUSDT"
    dex_chain: str
    dex_pool_fee: int
    base_precision: int = 8
    quote_precision: int = 8
    # --- Optional fields for dynamically discovered pairs ---
    base_address: Optional[str] = None
    quote_address: Optional[str] = None # This address corresponds to quote_dex
    base_decimals: Optional[int] = None
    quote_decimals: Optional[int] = None # These decimals correspond to quote_dex
    # --- Optional fields for triangular arbitrage ---
    is_synthetic: bool = False
    intermediate_symbol: Optional[str] = None
    # --- Optional fields for strategy parameters ---
    # Per-pair override of StrategyConfig.min_net_bps: the minimum net edge,
    # after taker fee and gas, required to act on this pair.
    min_net_bps: Optional[Decimal] = None
    # Execution slippage tolerance, used only to derive amountOutMinimum for
    # the on-chain swap. This is NOT a cost and never enters the economics.
    max_slippage_bps: Optional[int] = None
    max_size_quote: Optional[int] = None
    price_floor_quote: Optional[Decimal] = None
    price_ceiling_quote: Optional[Decimal] = None
    max_edge_bps: Optional[int] = None

@dataclass
class BookSnapshot:
    """A point-in-time order book ladder, deep enough to price a real trade.

    `bids` are ordered best-first (descending price) and `asks` best-first
    (ascending price), matching what `costs.walk_book` expects. `timestamp`
    is unix epoch seconds from `core.clock`, so staleness is checkable.

    This exists because pricing from top-of-book alone is only valid for
    trades smaller than the top level, which is not the case for the thin
    pools this strategy targets.
    """

    pair: "MarketPair"
    bids: List[tuple]
    asks: List[tuple]
    # When THIS symbol's book last changed. Grows in a quiet market, which is
    # a legitimate state -- an unchanged price is current, not stale.
    timestamp: float
    # When the FEED last delivered any frame, for any symbol. This is the
    # staleness signal: Binance suppresses unchanged books, so per-symbol age
    # measures market quiet while connection age measures a dead feed.
    # Defaults to `timestamp` for callers that supply only one clock.
    feed_timestamp: Optional[float] = None

    @property
    def best_bid(self) -> Optional[Decimal]:
        return self.bids[0][0] if self.bids else None

    @property
    def best_ask(self) -> Optional[Decimal]:
        return self.asks[0][0] if self.asks else None

    def age_seconds(self, now: float) -> float:
        """Seconds since this symbol's book last changed."""
        return now - self.timestamp

    def feed_age_seconds(self, now: float) -> float:
        """Seconds since the feed last delivered anything.

        This is what a staleness guard must use. Measured on Binance:
        ethusdt produced 150 frames in 15s while dogeusdt produced 30, with
        gaps up to 2.6s -- every frame carrying a distinct lastUpdateId, so
        the sparse ones were genuinely unchanged books rather than lost data.
        """
        stamp = self.timestamp if self.feed_timestamp is None else self.feed_timestamp
        return now - stamp


class Quote(BaseModel):
    pair: MarketPair
    price: Decimal
    size: Decimal
    venue: Literal["CEX", "DEX"]
    timestamp: float
    side: Optional[Literal["buy", "sell"]] = None
    bid_price: Optional[Decimal] = None
    ask_price: Optional[Decimal] = None
    bid_size: Optional[Decimal] = None
    ask_size: Optional[Decimal] = None

class Opportunity(BaseModel):
    pair: MarketPair
    direction: Literal["CEX_to_DEX", "DEX_to_CEX"]
    size: Decimal
    cex_price: Decimal
    dex_price: Decimal
    dex_chain: str
    dex_pool_fee: int
    edge_bps: Decimal
    slippage_bps: Decimal
    gas_cost_quote: Decimal
    cex_fee_quote: Decimal
    expected_pnl_quote: Decimal
    valid_until: float

# --- DEX Specific Types ---

class DexSwapParams(BaseModel):
    chain: str
    token_in_address: str
    token_in_decimals: int
    token_out_address: str
    fee: int
    amount_in: Decimal
    slippage_bps: int

class DexTxReceipt(BaseModel):
    tx_hash: str
    block_number: Optional[int] = None
    gas_used: Optional[int] = None
    status: Optional[int] = None
    timestamp: Optional[float] = None

# --- CEX & Execution Data Structures (User Provided) ---

class CexOrder(BaseModel):
    order_id: str
    pair: "MarketPair"
    side: Literal["buy", "sell"]
    type: Literal["MARKET", "LIMIT"] = "LIMIT"
    price: Decimal
    size: Decimal
    tif: Literal["IOC", "FOK", "GTC"] = "IOC"
    status: Literal["new","partially_filled","filled","canceled","rejected"] = "new"
    ts: float

class OrderUpdate(BaseModel):
    order_id: str
    status: Literal["partially_filled","filled","canceled","rejected"]
    filled_size: Decimal
    avg_fill_price: Optional[Decimal] = None
    reason: Optional[str] = None
    ts: float

class ExecutionLeg(BaseModel):
    venue: Literal["CEX","DEX"]
    side: Literal["buy","sell"]
    price_quote: Decimal
    size: Decimal
    fees_quote: Decimal
    tx_hash: Optional[str] = None
    order_id: Optional[str] = None

class ExecutionSummary(BaseModel):
    pair: "MarketPair"
    direction: Literal["CEX_to_DEX","DEX_to_CEX"]
    size: Decimal
    legs: List[ExecutionLeg]
    gas_quote: Decimal
    pnl_quote: Decimal
    edge_bps: Decimal
    hedged: bool
    started_ts: float
    completed_ts: float

@dataclass
class CexQuote:
    bid_price: Decimal
    ask_price: Decimal
    ts: float

@dataclass
class DexQuote:
    price: Decimal
    gas_cost_quote: Decimal = ZERO