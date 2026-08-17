"""Pure trade economics. No I/O, no clients, no config objects.

Everything here is a deterministic function of prices, sizes, and fee rates,
which is what makes the arithmetic testable in isolation. The strategy layer
supplies the inputs; this module decides what a trade is actually worth.
"""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import List, Optional, Sequence, Tuple

__all__ = ["BookFill", "walk_book", "TradeEconomics", "evaluate_trade"]

ZERO = Decimal("0")
TEN_THOUSAND = Decimal("10000")
DIRECTIONS = ("CEX_to_DEX", "DEX_to_CEX")

# A price level as (price, quantity_in_base).
Level = Tuple[Decimal, Decimal]


@dataclass(frozen=True)
class BookFill:
    """The result of consuming order book levels to fill a base quantity.

    `complete` is the field callers must check. When it is False the book
    could not supply `requested_base`, and `vwap` describes only the portion
    that `filled_base` covers -- treating it as the price for the full
    requested size is exactly the error this type exists to prevent.
    """

    requested_base: Decimal
    filled_base: Decimal
    vwap: Optional[Decimal]
    complete: bool

    @property
    def shortfall_base(self) -> Decimal:
        return self.requested_base - self.filled_base


def walk_book(
    levels: Sequence[Level],
    requested_base: Decimal,
    ascending: Optional[bool] = None,
) -> BookFill:
    """Walk `levels` in the order given, filling `requested_base`.

    Returns the volume-weighted average price of what was actually filled.

    `levels` must already be sorted best-price-first: ascending for asks
    (cheapest first), descending for bids (highest first). Pass `ascending`
    to have that assumption verified rather than assumed -- an unsorted or
    wrong-side book would otherwise produce a plausible but wrong VWAP.
    """
    if requested_base < ZERO:
        raise ValueError(f"requested_base must be non-negative, got {requested_base}")

    if ascending is not None:
        prices = [price for price, _ in levels]
        for previous, current in zip(prices, prices[1:]):
            if ascending and current < previous:
                raise ValueError(
                    f"levels must ascend for asks, got {previous} then {current}"
                )
            if not ascending and current > previous:
                raise ValueError(
                    f"levels must descend for bids, got {previous} then {current}"
                )

    if requested_base == ZERO:
        return BookFill(ZERO, ZERO, None, True)

    remaining = requested_base
    quote_spent = ZERO
    filled = ZERO

    for price, available in levels:
        if remaining <= ZERO:
            break
        taken = available if available < remaining else remaining
        quote_spent += taken * price
        filled += taken
        remaining -= taken

    if filled == ZERO:
        return BookFill(requested_base, ZERO, None, False)

    return BookFill(
        requested_base=requested_base,
        filled_base=filled,
        vwap=quote_spent / filled,
        complete=filled >= requested_base,
    )


@dataclass(frozen=True)
class TradeEconomics:
    """What a trade is actually worth, with every cost counted exactly once.

    `net_quote` is always `gross_quote - cex_fee_quote - gas_quote`. That
    identity is deliberate and is enforced by test: the DEX quote already
    carries the pool fee and the price impact for the requested size, so
    there is no further impact or slippage term to subtract. `slippage_bps`
    is a *tolerance* used to derive `amountOutMinimum` at execution time --
    it is never a cost, and it never appears here.
    """

    direction: str
    size_base: Decimal
    buy_price: Decimal
    sell_price: Decimal
    notional_quote: Decimal
    gross_quote: Decimal
    cex_fee_quote: Decimal
    gas_quote: Decimal
    net_quote: Decimal
    net_bps: Decimal
    cex_legs: int

    @property
    def is_profitable(self) -> bool:
        return self.net_quote > ZERO


def evaluate_trade(
    *,
    direction: str,
    size_base: Decimal,
    cex_price: Decimal,
    dex_price: Decimal,
    taker_fee_bps: Decimal,
    gas_quote: Decimal,
    cex_legs: int = 1,
) -> TradeEconomics:
    """Net economics of a two-venue trade.

    Args:
        direction: "CEX_to_DEX" (buy base on the CEX) or "DEX_to_CEX".
        size_base: trade size in base units.
        cex_price: effective CEX price for this size -- a depth-weighted VWAP
            from `walk_book`, not top of book. Fee-exclusive.
        dex_price: effective DEX price for this size, as returned by the
            quoter. Already net of the pool fee and price impact.
        taker_fee_bps: CEX taker fee in basis points.
        gas_quote: on-chain gas cost, denominated in the quote currency.
        cex_legs: number of CEX orders the round trip requires. 1 for a
            direct pair; 2 for a synthetic pair, where the intermediate
            asset must also be traded on the CEX. Gas is charged once
            regardless, because only one on-chain swap occurs.
    """
    if direction not in DIRECTIONS:
        raise ValueError(f"direction must be one of {DIRECTIONS}, got {direction!r}")
    if size_base <= ZERO:
        raise ValueError(f"size_base must be positive, got {size_base}")
    if cex_price <= ZERO:
        raise ValueError(f"cex_price must be positive, got {cex_price}")
    if dex_price <= ZERO:
        raise ValueError(f"dex_price must be positive, got {dex_price}")
    if taker_fee_bps < ZERO:
        raise ValueError(f"taker_fee_bps must be non-negative, got {taker_fee_bps}")
    if gas_quote < ZERO:
        raise ValueError(f"gas_quote must be non-negative, got {gas_quote}")
    if cex_legs < 1:
        raise ValueError(f"cex_legs must be at least 1, got {cex_legs}")

    if direction == "CEX_to_DEX":
        buy_price, sell_price = cex_price, dex_price
    else:
        buy_price, sell_price = dex_price, cex_price

    # Capital committed is on the buy leg, whichever venue that is.
    notional_quote = size_base * buy_price
    gross_quote = (sell_price - buy_price) * size_base

    # The taker fee applies to the quote value transacted on each CEX leg.
    # For a synthetic pair both legs move approximately the same notional,
    # so one leg's fee scaled by the leg count is accurate to rounding.
    cex_fee_quote = (cex_price * size_base * taker_fee_bps / TEN_THOUSAND) * cex_legs

    net_quote = gross_quote - cex_fee_quote - gas_quote
    net_bps = net_quote / notional_quote * TEN_THOUSAND

    return TradeEconomics(
        direction=direction,
        size_base=size_base,
        buy_price=buy_price,
        sell_price=sell_price,
        notional_quote=notional_quote,
        gross_quote=gross_quote,
        cex_fee_quote=cex_fee_quote,
        gas_quote=gas_quote,
        net_quote=net_quote,
        net_bps=net_bps,
        cex_legs=cex_legs,
    )
