from typing import Protocol, Optional
from decimal import Decimal

from ..core.types import MarketPair, Quote, CexOrder, OrderUpdate


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
