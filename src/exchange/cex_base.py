from typing import Protocol, Optional
from decimal import Decimal

from ..core.types import BookSnapshot, MarketPair, Quote, CexOrder, OrderUpdate


# --- Abstract Base Class / Protocol for CEX clients ---

class CexClient(Protocol):
    """
    The standard interface for interacting with a centralized exchange (CEX).
    """

    async def connect(self) -> None:
        """Connect to the exchange, e.g. open the WebSocket streams."""
        ...

    async def close(self) -> None:
        """Close the connection to the exchange."""
        ...

    async def get_quote(self, pair: MarketPair) -> Optional[Quote]:
        """
        Return the current quote (best bid and ask) for the given pair.
        In a complete implementation this is driven by the WebSocket feed.

        Top-of-book only. Use `get_book` for anything that must be priced
        for a specific trade size.
        """
        ...

    async def get_book(self, pair: MarketPair) -> Optional[BookSnapshot]:
        """
        Return a depth ladder for the pair, or None if unavailable.

        `bids` must be ordered best-first (descending) and `asks` best-first
        (ascending). This is what the detector prices trades against; pricing
        from `get_quote` alone is only valid for trades smaller than the top
        level.
        """
        ...

    async def create_order(self, order: CexOrder) -> OrderUpdate:
        """
        Create a new order.
        Returns an order update containing the order ID and initial status.
        """
        ...

    async def cancel_order(self, order_id: str, pair: MarketPair) -> bool:
        """
        Cancel an existing order.
        Returns True if the cancellation succeeded.
        """
        ...

    async def get_balance(self, asset: str) -> Decimal:
        """Return the available balance for a given asset."""
        ...
