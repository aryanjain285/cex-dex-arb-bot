from __future__ import annotations
from typing import List, Optional, Literal
from decimal import Decimal
from pydantic import BaseModel
from dataclasses import dataclass

ZERO = Decimal("0")

__all__ = [
    "MarketPair",
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
    min_edge_bps: Optional[int] = None
    max_slippage_bps: Optional[int] = None
    max_size_quote: Optional[int] = None
    price_floor_quote: Optional[Decimal] = None
    price_ceiling_quote: Optional[Decimal] = None
    max_edge_bps: Optional[int] = None
    edge_safety_multiplier: Optional[Decimal] = None

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