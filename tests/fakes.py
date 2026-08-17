"""Shared in-memory venue doubles for tests.

These are real objects implementing the client protocols, not mocks. The
detector's use of order book depth and of the buy/sell size conventions is
precisely what needs testing, so a mock that returns whatever it is asked for
would assert nothing.
"""
from __future__ import annotations

from decimal import Decimal
from typing import Dict, List, Optional, Tuple

from src.core import clock
from src.core.types import BookSnapshot, DexQuote, MarketPair

Level = Tuple[Decimal, Decimal]


def D(x) -> Decimal:
    return Decimal(str(x))


def flat_book(bid, ask, depth=D(10_000)) -> Tuple[List[Level], List[Level]]:
    """A book with a single deep level each side -- no depth effects."""
    return ([(D(bid), D(depth))], [(D(ask), D(depth))])


def make_pair(symbol: str = "ETH/USDT", **overrides) -> MarketPair:
    defaults = dict(
        base="WETH",
        quote_cex="USDT",
        quote_dex="USDT",
        cex_symbol=symbol,
        dex_chain="ethereum",
        dex_pool_fee=500,
    )
    defaults.update(overrides)
    return MarketPair(**defaults)


class FakeCex:
    """Serves fixed depth ladders keyed by cex_symbol."""

    def __init__(self, books: Dict[str, Tuple[List[Level], List[Level]]]):
        self.books = books
        self.book_calls = 0

    async def get_book(self, pair: MarketPair) -> Optional[BookSnapshot]:
        self.book_calls += 1
        entry = self.books.get(pair.cex_symbol)
        if entry is None:
            return None
        bids, asks = entry
        return BookSnapshot(pair=pair, bids=bids, asks=asks, timestamp=clock.now())


class FakeDex:
    """Returns a fixed price per side.

    Records the sizes it was asked for, so tests can assert the buy leg was
    given a quote-currency amount rather than a base amount.
    """

    def __init__(self, sell_price, buy_price, gas=D(0)):
        self.sell_price = D(sell_price)
        self.buy_price = D(buy_price)
        self.gas = D(gas)
        self.requests: List[Tuple[str, Decimal]] = []

    async def get_quote(self, pair, size, side, estimate_gas=False):
        self.requests.append((side, size))
        price = self.sell_price if side == "sell" else self.buy_price
        return DexQuote(price=price, gas_cost_quote=self.gas)
