from typing import Protocol, Optional
from decimal import Decimal

from ..core.types import MarketPair, DexQuote, DexSwapParams, DexTxReceipt

# --- Abstract Base Class / Protocol for DEX clients ---

class DexClient(Protocol):
    """
    The standard interface for interacting with a decentralized exchange (DEX).
    """

    async def get_quote(
        self,
        pair: MarketPair,
        size: Decimal,
        side: str,
        estimate_gas: bool = False,
    ) -> Optional[DexQuote]:
        """
        Return the quote for trading a given amount in a specific pool.
        This normally calls the DEX's Quoter contract.

        For `side="sell"`, `size` is an amount of the base token.
        For `side="buy"`, `size` is an amount of the quote token to spend.
        """
        ...

    async def execute_swap(self, params: DexSwapParams) -> DexTxReceipt:
        """
        Execute a swap.
        Builds, signs, and broadcasts a transaction to the chain.
        """
        ...

    async def estimate_gas(self, params: DexSwapParams) -> int:
        """
        Estimate the gas cost of executing a swap.
        """
        ...

    async def get_balance(self, asset: str, chain: str) -> Decimal:
        """Return the wallet balance of a token on a specific chain."""
        ...
