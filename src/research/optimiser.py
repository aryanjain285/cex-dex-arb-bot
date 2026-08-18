"""The profit curve over size, and the size that maximises it.

The system has only ever asked "is $1,000 profitable?". That was forced by the RPC
path -- one size cost one eth_call -- and it is the wrong question. With local pool
math twenty sizes are free, so the right one becomes answerable:

    Pi(q) = P_sell(q) * q - P_buy(q) * q - C(q)          q* = argmax_q Pi(q)

Pi is NOT monotonic, and that is the whole point. Gas is fixed per trade, so small
sizes cannot cover it. Price impact grows with size, so large sizes give the edge
back. The optimum is interior, and a fixed probe finds it only by luck.

Measured on a live ETH/USDT pool, CEX_to_DEX, 2026-08-18:

        notional   gross bps     net bps
             211       -5.68      -61.62
             966       -5.81      -23.87     <- the $1,000 probe
           4,434       -6.43      -16.23
           9,498       -7.32      -15.90     <- the actual optimum
          20,346       -9.24      -17.24
         200,003      -41.13      -48.68

The fixed probe understated the best achievable net edge by 8 bps -- larger than the
5 bps floor the strategy trades on. It does not make this pair profitable, but a
measurement that is 8 bps pessimistic cannot be used to decide whether anything is.

PARAMETERISED BY NOTIONAL, NOT BASE SIZE

Because the two directions consume different tokens on the DEX:

    CEX_to_DEX   sell base on the DEX   -> the pool input is BASE
    DEX_to_CEX   buy  base on the DEX   -> the pool input is QUOTE

An earlier version passed a base amount for both. Against a live ETH pool that
priced a swap of 0.5 USDT instead of 0.5 WETH and reported 36,117,456,820 bps of
gross edge -- the same buy-leg units defect the detector once had. Notional is well
defined for both directions, so it is the parameter, and each direction converts it
to the input its own leg actually takes.

BOTH VENUES ARE WALKED PROPERLY

The CEX side goes through its book ladder via `walk_book`; the DEX side through the
pool's tick liquidity via the local swap math. Neither is a top-of-book
approximation -- a curve built on top of book is a straight line with no interior
optimum, so it would silently answer a different question.

REFUSALS ARE MARKED, NOT DROPPED

A notional the CEX book cannot fill, or one that would leave the pool's recorded
tick range, has no price. Both are recorded as such: dropping them would make a
truncated curve look complete, and `argmax` over it would return the largest size
that happened to be answerable rather than the largest that is tradeable. That
distinction is what capacity means.
"""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import List, Optional, Sequence

from ..strategy.costs import evaluate_trade, walk_book

__all__ = [
    "SizePoint",
    "SizeCurve",
    "geometric_size_grid",
    "optimise_size",
]

ZERO = Decimal("0")
TEN_THOUSAND = Decimal("10000")


@dataclass(frozen=True)
class SizePoint:
    """One point on the curve. `net_bps is None` means the size was unpriceable."""

    notional_requested: Decimal
    size_base: Optional[Decimal] = None
    notional_quote: Optional[Decimal] = None
    gross_bps: Optional[Decimal] = None
    net_bps: Optional[Decimal] = None
    net_quote: Optional[Decimal] = None
    cex_price: Optional[Decimal] = None
    dex_price: Optional[Decimal] = None
    cex_levels_used: Optional[int] = None
    # Why this size could not be priced. Present exactly when net_bps is None.
    reason: Optional[str] = None


@dataclass(frozen=True)
class SizeCurve:
    """The full curve, plus the best tradeable point if there is one."""

    direction: str
    curve: List[SizePoint]
    # The best point clearing the floor, or None. Deliberately None rather than the
    # least-bad point: returning that would read as an opportunity downstream.
    best: Optional[SizePoint] = None
    # The best GROSS dislocation at any priceable size. This is the research signal
    # and it survives every cost assumption, so it is reported even when nothing is
    # tradeable -- which is the situation this system is actually in.
    best_gross_bps: Optional[Decimal] = None
    best_gross_notional: Optional[Decimal] = None

    def priceable(self) -> List[SizePoint]:
        return [p for p in self.curve if p.net_bps is not None]


def geometric_size_grid(
    minimum: Decimal, maximum: Decimal, points: int
) -> List[Decimal]:
    """Sizes spaced by a constant ratio.

    Geometric, not linear: the interesting region spans three orders of magnitude,
    and a linear grid either misses the small end entirely or spends most of its
    points where the answer is already known.
    """
    if points < 1:
        raise ValueError("points must be at least 1")
    if minimum <= 0:
        raise ValueError(
            f"minimum must be positive, got {minimum}; geometric spacing has no "
            f"meaning from zero, and clamping would silently change the span"
        )
    if maximum < minimum:
        raise ValueError(f"maximum ({maximum}) is below minimum ({minimum})")
    if points == 1 or maximum == minimum:
        return [minimum]

    ratio = (maximum / minimum) ** (Decimal(1) / Decimal(points - 1))
    grid = [minimum * (ratio ** Decimal(i)) for i in range(points)]
    # Pin the endpoints exactly: repeated multiplication drifts, and a grid whose
    # ends are not what was asked for is confusing in a report.
    grid[0], grid[-1] = minimum, maximum
    return grid


def optimise_size(
    *,
    pool,
    direction: str,
    cex_bids: Sequence,
    cex_asks: Sequence,
    notionals: Sequence[Decimal],
    taker_fee_bps: Decimal,
    gas_quote: Decimal,
    base_is_token0: bool,
    rotation_cost_quote: Decimal = ZERO,
    cex_legs: int = 1,
    floor_bps: Decimal = ZERO,
    dex_price_scale: Decimal = Decimal(1),
) -> SizeCurve:
    """Evaluate Pi(q) over a grid of NOTIONALS and return the curve plus the best.

    `base_is_token0` says which side of the pool the base token is on, which is
    determined by address order and therefore differs between pools. Getting it
    wrong inverts the price -- a factor of price^2 -- so it is a required argument
    rather than something inferred.

    `dex_price_scale` converts a synthetic route's DEX quote through an intermediate
    asset, the same way the detector does.
    """
    if not notionals:
        raise ValueError("notionals must not be empty")
    if direction not in ("CEX_to_DEX", "DEX_to_CEX"):
        raise ValueError(f"unknown direction {direction!r}")

    selling_base_on_dex = direction == "CEX_to_DEX"
    # Which pool token the DEX leg SPENDS. Selling base spends base; buying base
    # spends quote. This is the distinction that produced a 36-billion-bps reading
    # when it was collapsed into one case.
    if selling_base_on_dex:
        zero_for_one = base_is_token0
    else:
        zero_for_one = not base_is_token0

    points: List[SizePoint] = []
    for notional in notionals:
        if notional <= 0:
            points.append(SizePoint(
                notional_requested=notional, reason="non-positive notional"
            ))
            continue

        if selling_base_on_dex:
            point = _price_cex_to_dex(
                pool, notional, cex_asks, zero_for_one, dex_price_scale,
                taker_fee_bps, gas_quote, cex_legs, rotation_cost_quote,
            )
        else:
            point = _price_dex_to_cex(
                pool, notional, cex_bids, zero_for_one, dex_price_scale,
                taker_fee_bps, gas_quote, cex_legs, rotation_cost_quote,
            )
        points.append(point)

    priceable = [p for p in points if p.net_bps is not None]
    clearing = [p for p in priceable if p.net_bps > floor_bps]
    best = max(clearing, key=lambda p: p.net_quote) if clearing else None
    best_gross = max(priceable, key=lambda p: p.gross_bps) if priceable else None

    return SizeCurve(
        direction=direction,
        curve=points,
        best=best,
        best_gross_bps=best_gross.gross_bps if best_gross else None,
        best_gross_notional=best_gross.notional_requested if best_gross else None,
    )


def _price_cex_to_dex(
    pool, notional, cex_asks, zero_for_one, dex_price_scale,
    taker_fee_bps, gas_quote, cex_legs, rotation_cost_quote,
) -> SizePoint:
    """Buy base on the CEX, sell base on the DEX.

    Sizing starts from the CEX ask, because that is the leg whose price is known
    before the size is chosen: the notional buys a base amount at the touch, and
    then the ladder is walked for that amount.
    """
    if not cex_asks:
        return SizePoint(notional_requested=notional, reason="no CEX asks")

    size_base = notional / cex_asks[0][0]
    fill = walk_book(cex_asks, size_base, ascending=True)
    if not fill.complete or fill.vwap is None:
        return SizePoint(
            notional_requested=notional, size_base=size_base,
            reason=f"insufficient CEX depth: filled {fill.filled_base} of {size_base}",
        )

    # The DEX leg SELLS base, so the pool input is a base amount.
    dex_price = pool.price_for_amount_in(fill.filled_base, zero_for_one=zero_for_one)
    if dex_price is None or dex_price <= 0:
        return SizePoint(
            notional_requested=notional, size_base=size_base, cex_price=fill.vwap,
            reason=(
                "no DEX price at this size: the pool is empty, or the quote would "
                "leave the recorded tick range"
            ),
        )
    dex_price = dex_price * dex_price_scale

    return _finish(
        notional, fill.filled_base, fill.vwap, dex_price, "CEX_to_DEX",
        taker_fee_bps, gas_quote, cex_legs, rotation_cost_quote, cex_asks, fill,
    )


def _price_dex_to_cex(
    pool, notional, cex_bids, zero_for_one, dex_price_scale,
    taker_fee_bps, gas_quote, cex_legs, rotation_cost_quote,
) -> SizePoint:
    """Buy base on the DEX, sell base on the CEX.

    The DEX leg SPENDS QUOTE, so the pool input is the notional itself -- not a base
    amount. `price_for_amount_in` returns quote-per-base for this direction, from
    which the base received follows.
    """
    if not cex_bids:
        return SizePoint(notional_requested=notional, reason="no CEX bids")

    received_per_spent = pool.price_for_amount_in(notional, zero_for_one=zero_for_one)
    if received_per_spent is None or received_per_spent <= 0:
        return SizePoint(
            notional_requested=notional,
            reason=(
                "no DEX price at this size: the pool is empty, or the quote would "
                "leave the recorded tick range"
            ),
        )
    # `price_for_amount_in` returns OUT per IN. Here IN is quote and OUT is base,
    # so it returns base-per-quote -- and every price in the cost model is
    # quote-per-base. Without this reciprocal an ETH pool reported 0.000526 as its
    # price and the gross edge came out at 36,190,121,923 bps.
    dex_price = (Decimal(1) / received_per_spent) * dex_price_scale

    # What the notional actually buys on chain, at the price it actually gets.
    size_base = notional / dex_price
    fill = walk_book(cex_bids, size_base, ascending=False)
    if not fill.complete or fill.vwap is None:
        return SizePoint(
            notional_requested=notional, size_base=size_base, dex_price=dex_price,
            reason=f"insufficient CEX depth: filled {fill.filled_base} of {size_base}",
        )

    return _finish(
        notional, size_base, fill.vwap, dex_price, "DEX_to_CEX",
        taker_fee_bps, gas_quote, cex_legs, rotation_cost_quote, cex_bids, fill,
    )


def _finish(
    notional_requested, size_base, cex_price, dex_price, direction,
    taker_fee_bps, gas_quote, cex_legs, rotation_cost_quote, levels, fill,
) -> SizePoint:
    econ = evaluate_trade(
        direction=direction, size_base=size_base,
        cex_price=cex_price, dex_price=dex_price,
        taker_fee_bps=taker_fee_bps, gas_quote=gas_quote,
        cex_legs=cex_legs, rotation_cost_quote=rotation_cost_quote,
    )
    if econ is None:
        return SizePoint(
            notional_requested=notional_requested, size_base=size_base,
            cex_price=cex_price, dex_price=dex_price,
            reason="economics not computable",
        )

    return SizePoint(
        notional_requested=notional_requested,
        size_base=size_base,
        notional_quote=econ.notional_quote,
        gross_bps=econ.gross_quote / econ.notional_quote * TEN_THOUSAND,
        net_bps=econ.net_bps,
        net_quote=econ.net_quote,
        cex_price=cex_price,
        dex_price=dex_price,
        cex_levels_used=_levels_consumed(levels, fill.filled_base),
    )


def _levels_consumed(levels: Sequence, filled_base: Decimal) -> int:
    """How many book levels the fill touched -- the realised depth requirement."""
    remaining, used = filled_base, 0
    for _, available in levels:
        if remaining <= ZERO:
            break
        used += 1
        remaining -= available
    return used
