"""Pure trade economics. No I/O, no clients, no config objects.

Everything here is a deterministic function of prices, sizes, and fee rates,
which is what makes the arithmetic testable in isolation. The strategy layer
supplies the inputs; this module decides what a trade is actually worth.
"""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import List, Optional, Sequence, Tuple

__all__ = [
    "BookFill", "walk_book", "TradeEconomics", "evaluate_trade",
    "amortised_rotation_cost",
]

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


def amortised_rotation_cost(
    *,
    withdrawal_fee_quote: Decimal,
    bridge_gas_quote: Decimal,
    float_quote: Decimal,
    notional_quote: Decimal,
) -> Decimal:
    """Per-trade share of the FEES paid to move inventory between venues.

    A CEX<->DEX arb is an inventory rotation, not a round trip. `CEX_to_DEX` buys
    base on the exchange and sells base from the on-chain wallet: two different
    assets in two custody locations. Trading repeatedly in one direction requires
    physically moving inventory back, which costs money.

    Two costs, both of which genuinely leave the account:

    - `withdrawal_fee_quote`: what the exchange charges to withdraw.
    - `bridge_gas_quote`: on-chain cost of the transfer or bridge.

    Amortised over `float_quote / notional_quote` trades -- how many trades the
    float supports before a rotation is needed. A larger float amortises further.

    WHAT IS NO LONGER HERE, AND WHY

    This function also charged `float_quote * transfer_risk_bps / 10000` and called
    it, in its own docstring, the "expected adverse price move on the in-transit
    float". That is the error stated out loud: under a zero-drift assumption

        E[dP] = 0        while       Var(dP) > 0

    Exposure while inventory is in transit is variance, not a negative mean.
    Subtracting it made `net_bps` -- the number recorded on every row of the audit
    trail -- not the expected value of anything, and depressed every measured edge
    by the full `transfer_risk_bps`.

    Worse, it corrupted the measurement rather than the decision: changing the risk
    assumption silently rewrote what history's edges had "been". The exposure is
    real and still charged, as a threshold: see `rotation_risk_bps` and
    `required_net_bps`. Measurement stays honest; risk appetite moves to the floor.

    Worked: a $4 withdrawal plus $1 bridge gas funding a $5,000 float at $1,000
    notional supports 5 trades, so fees contribute $1.00/trade -- 10 bps.
    """
    if withdrawal_fee_quote < ZERO:
        raise ValueError("withdrawal_fee_quote must be non-negative")
    if bridge_gas_quote < ZERO:
        raise ValueError("bridge_gas_quote must be non-negative")
    if notional_quote <= ZERO:
        raise ValueError("notional_quote must be positive")
    if float_quote < notional_quote:
        raise ValueError(
            f"float_quote ({float_quote}) is smaller than one trade's notional "
            f"({notional_quote}): the strategy cannot run at this size"
        )

    trades_per_rotation = float_quote / notional_quote
    return (withdrawal_fee_quote + bridge_gas_quote) / trades_per_rotation


def rotation_risk_bps(
    *,
    float_quote: Decimal,
    notional_quote: Decimal,
    transfer_risk_bps: Decimal,
) -> Decimal:
    """Per-trade risk charge for unhedged inventory in transit, in basis points.

    This is deliberately NOT a cost. It raises the edge a trade must clear rather
    than reducing the edge it is measured to have -- so `net_bps` remains an
    expected value and stays comparable across changes in risk policy.

    Note what falls out of the arithmetic: the exposure is on the whole float but
    is incurred once per rotation, and a rotation covers `float / notional` trades.
    Per trade that is exactly `transfer_risk_bps`, independent of float size. The
    old form multiplied by the float AND divided by trades-per-rotation, so the
    float cancelled anyway -- the scaling was decorative.

    Proper treatment of the variance -- required Sharpe, expected shortfall, a
    capital charge, or hedging the exposure with a perp instead of transferring --
    is a portfolio question this function does not attempt. It only makes the
    threshold reflect that the risk exists.
    """
    if transfer_risk_bps < ZERO:
        raise ValueError("transfer_risk_bps must be non-negative")
    if notional_quote <= ZERO:
        raise ValueError("notional_quote must be positive")
    if float_quote < notional_quote:
        raise ValueError(
            f"float_quote ({float_quote}) is smaller than one trade's notional "
            f"({notional_quote})"
        )
    return transfer_risk_bps


def required_net_bps(
    *,
    base_floor_bps: Decimal,
    risk_charge_bps: Decimal,
) -> Decimal:
    """The edge a trade must clear: the configured floor plus the risk charge.

    Keeping these separate is what lets the audit trail record both an honest
    expected edge and the threshold it was judged against, so a later reader can
    re-decide the same rows under a different risk policy without re-measuring.
    """
    if base_floor_bps < ZERO:
        raise ValueError("base_floor_bps must be non-negative")
    if risk_charge_bps < ZERO:
        raise ValueError(
            f"risk_charge_bps must be non-negative, got {risk_charge_bps}. A "
            f"negative charge would LOWER the required edge, which is not a risk "
            f"policy anyone means to express."
        )
    return base_floor_bps + risk_charge_bps


@dataclass(frozen=True)
class TradeEconomics:
    """What a trade is actually worth, with every cost counted exactly once.

    `net_quote` is always `gross_quote - cex_fee_quote - gas_quote -
    rotation_cost_quote`. That
    identity is deliberate and is enforced by test: the DEX quote already
    carries the pool fee and the price impact for the requested size, so
    there is no further impact or slippage term to subtract. `slippage_bps`
    is a *tolerance* used to derive `amountOutMinimum` at execution time --
    it is never a cost, and it never appears here.
    """

    direction: str
    size_base: Decimal
    # Direction-relative, for the PnL arithmetic.
    buy_price: Decimal
    sell_price: Decimal
    # Venue-attributed, so an audit record is self-describing and no consumer
    # has to re-derive which venue a price came from -- a re-derivation that
    # invites a sign error at every call site.
    cex_price: Decimal
    dex_price: Decimal
    notional_quote: Decimal
    gross_quote: Decimal
    cex_fee_quote: Decimal
    gas_quote: Decimal
    # Per-trade share of moving inventory back between venues. Unmodelled
    # originally, and large enough at realistic float sizes to invert the sign
    # of a marginal trade.
    rotation_cost_quote: Decimal
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
    rotation_cost_quote: Decimal = ZERO,
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
        rotation_cost_quote: per-trade share of moving inventory back between
            venues, from `amortised_rotation_cost`. Defaults to zero so the
            function stays usable in isolation, but a live configuration that
            leaves it at zero is asserting that inventory rotation is free.
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
    if rotation_cost_quote < ZERO:
        raise ValueError(
            f"rotation_cost_quote must be non-negative, got {rotation_cost_quote}"
        )
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

    net_quote = gross_quote - cex_fee_quote - gas_quote - rotation_cost_quote
    net_bps = net_quote / notional_quote * TEN_THOUSAND

    return TradeEconomics(
        direction=direction,
        size_base=size_base,
        buy_price=buy_price,
        sell_price=sell_price,
        cex_price=cex_price,
        dex_price=dex_price,
        notional_quote=notional_quote,
        gross_quote=gross_quote,
        cex_fee_quote=cex_fee_quote,
        gas_quote=gas_quote,
        rotation_cost_quote=rotation_cost_quote,
        net_quote=net_quote,
        net_bps=net_bps,
        cex_legs=cex_legs,
    )
