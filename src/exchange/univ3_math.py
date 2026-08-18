"""Uniswap v3 swap arithmetic, computed locally.

Three things this unlocks, in ascending order of importance:

LATENCY. QuoterV2 costs an eth_call. Measured on 2026-08-17: 0.31s median on
Ethereum, 0.78s on Arbitrum, 0.83s on Base, with a 2.2s p90 on Ethereum that
exceeds the 2.0s opportunity TTL. This is microseconds.

THE SIZE CURVE. Asking "is $1,000 profitable?" costs one RPC call; asking "what is
the OPTIMAL size?" needs twenty. That is why the system has only ever evaluated one
fixed notional -- a limitation of the RPC path rather than a considered choice.
Locally, twenty sizes cost nothing, so `Pi(q)` and `argmax_q Pi(q)` become
available.

OFFLINE BACKTESTING. A recorded pool state can be re-quoted at any size, under any
cost assumption, months later. A recorded QuoterV2 answer can only be re-read at
the one size it was asked about. This is the difference between having a backtest
and having a log.

CORRECTNESS BAR

Exact agreement with the deployed QuoterV2. Uniswap's maths is integer fixed-point
throughout and this mirrors it: `//` everywhere, no floats in the swap loop. An
approximation in floats would carry an error of roughly the size this strategy
trades on, which is the one place a rounding difference actually matters.

Rounding follows the contract's convention: against the trader. Amounts owed to the
pool round UP, amounts paid out round DOWN. Getting that backwards produces a local
quote fractionally better than reality -- the direction that manufactures edge.

Reference: Uniswap v3-core `SqrtPriceMath`, `SwapMath` and `TickMath`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from typing import Iterable, List, Optional, Sequence, Tuple

__all__ = [
    "Q96",
    "SwapResult",
    "MIN_TICK",
    "MAX_TICK",
    "TickInfo",
    "V3Pool",
    "sqrt_price_x96_from_tick",
    "tick_from_sqrt_price_x96",
    "price_from_sqrt_price_x96",
    "amount0_delta",
    "amount1_delta",
]

# Uniswap's fixed-point scale for sqrt prices: Q64.96.
Q96 = 2 ** 96
Q192 = Q96 * Q96

# The representable tick range. Beyond it the contract's own fixed-point type
# overflows, so accepting a wider value here would diverge from the chain.
MIN_TICK = -887272
MAX_TICK = 887272

MIN_SQRT_RATIO = 4295128739
MAX_SQRT_RATIO = 1461446703485210103287273052203988822378723970342

# Fee is in hundredths of a basis point: 500 = 0.05%.
FEE_DENOMINATOR = 1_000_000


def sqrt_price_x96_from_tick(tick: int) -> int:
    """sqrt(1.0001^tick) * 2^96, by the contract's exact algorithm.

    Bit-by-bit multiplication of precomputed constants, not `sqrt(1.0001**tick)`.
    The float version drifts by several wei at large ticks, and a few wei of sqrt
    price is a real difference in output amount on a large pool.
    """
    if not MIN_TICK <= tick <= MAX_TICK:
        raise ValueError(
            f"tick {tick} is outside the representable range "
            f"[{MIN_TICK}, {MAX_TICK}]; the contract's fixed-point type would "
            f"overflow, so a value accepted here would not match the chain"
        )

    abs_tick = abs(tick)
    ratio = 0x100000000000000000000000000000000

    # Each constant is 2^128 / 1.0001^(2^i), from v3-core TickMath.
    for bit, constant in (
        (0x1, 0xfffcb933bd6fad37aa2d162d1a594001),
        (0x2, 0xfff97272373d413259a46990580e213a),
        (0x4, 0xfff2e50f5f656932ef12357cf3c7fdcc),
        (0x8, 0xffe5caca7e10e4e61c3624eaa0941cd0),
        (0x10, 0xffcb9843d60f6159c9db58835c926644),
        (0x20, 0xff973b41fa98c081472e6896dfb254c0),
        (0x40, 0xff2ea16466c96a3843ec78b326b52861),
        (0x80, 0xfe5dee046a99a2a811c461f1969c3053),
        (0x100, 0xfcbe86c7900a88aedcffc83b479aa3a4),
        (0x200, 0xf987a7253ac413176f2b074cf7815e54),
        (0x400, 0xf3392b0822b70005940c7a398e4b70f3),
        (0x800, 0xe7159475a2c29b7443b29c7fa6e889d9),
        (0x1000, 0xd097f3bdfd2022b8845ad8f792aa5825),
        (0x2000, 0xa9f746462d870fdf8a65dc1f90e061e5),
        (0x4000, 0x70d869a156d2a1b890bb3df62baf32f7),
        (0x8000, 0x31be135f97d08fd981231505542fcfa6),
        (0x10000, 0x9aa508b5b7a84e1c677de54f3e99bc9),
        (0x20000, 0x5d6af8dedb81196699c329225ee604),
        (0x40000, 0x2216e584f5fa1ea926041bedfe98),
        (0x80000, 0x48a170391f7dc42444e8fa2),
    ):
        if abs_tick & bit:
            ratio = (ratio * constant) >> 128

    if tick > 0:
        ratio = (2 ** 256 - 1) // ratio

    # Q128.128 -> Q64.96, rounding up so the result is never below the true value.
    sqrt_price = (ratio >> 32) + (0 if ratio % (1 << 32) == 0 else 1)
    return sqrt_price


def tick_from_sqrt_price_x96(sqrt_price_x96: int) -> int:
    """The greatest tick whose sqrt price does not exceed this one.

    Binary search rather than the contract's log2 approximation: this runs off the
    hot path (once per state update, not once per quote), and an exact search cannot
    disagree with `sqrt_price_x96_from_tick` -- which the log approximation can, at
    the boundaries, by one tick.
    """
    if not MIN_SQRT_RATIO <= sqrt_price_x96 < MAX_SQRT_RATIO:
        raise ValueError(f"sqrt price {sqrt_price_x96} is outside the valid range")

    low, high = MIN_TICK, MAX_TICK
    while low < high:
        mid = (low + high + 1) // 2
        if sqrt_price_x96_from_tick(mid) <= sqrt_price_x96:
            low = mid
        else:
            high = mid - 1
    return low


def price_from_sqrt_price_x96(
    sqrt_price_x96: int, decimals0: int, decimals1: int
) -> Decimal:
    """Price of token0 in token1, adjusted for both tokens' decimals.

    The decimals adjustment is part of the calculation rather than an afterthought:
    a 6-decimal token against an 18-decimal one shifts the price by 10^12, which is
    exactly the error a missing decimals field produced in the pool dataset.
    """
    raw = (Decimal(sqrt_price_x96) ** 2) / Decimal(Q192)
    return raw * (Decimal(10) ** (decimals0 - decimals1))


def _mul_div_rounding_up(a: int, b: int, denominator: int) -> int:
    if denominator == 0:
        raise ZeroDivisionError("mulDivRoundingUp with a zero denominator")
    product = a * b
    result = product // denominator
    return result + 1 if product % denominator else result


def amount0_delta(
    sqrt_price_a_x96: int, sqrt_price_b_x96: int, liquidity: int,
    round_up: bool = True,
) -> int:
    """Amount of token0 between two prices, for a given liquidity.

    Rounds UP by default: token0 is what the trader owes when swapping
    zero-for-one, and the contract rounds amounts owed to the pool against the
    trader. Rounding the other way makes a local quote fractionally better than
    reality -- the direction that invents edge.
    """
    if sqrt_price_a_x96 > sqrt_price_b_x96:
        sqrt_price_a_x96, sqrt_price_b_x96 = sqrt_price_b_x96, sqrt_price_a_x96
    if liquidity == 0 or sqrt_price_a_x96 == sqrt_price_b_x96:
        return 0

    numerator1 = liquidity << 96
    numerator2 = sqrt_price_b_x96 - sqrt_price_a_x96

    if round_up:
        inner = _mul_div_rounding_up(numerator1, numerator2, sqrt_price_b_x96)
        return _mul_div_rounding_up(inner, 1, sqrt_price_a_x96)
    return (numerator1 * numerator2 // sqrt_price_b_x96) // sqrt_price_a_x96


def amount1_delta(
    sqrt_price_a_x96: int, sqrt_price_b_x96: int, liquidity: int,
    round_up: bool = True,
) -> int:
    """Amount of token1 between two prices, for a given liquidity."""
    if sqrt_price_a_x96 > sqrt_price_b_x96:
        sqrt_price_a_x96, sqrt_price_b_x96 = sqrt_price_b_x96, sqrt_price_a_x96
    if liquidity == 0 or sqrt_price_a_x96 == sqrt_price_b_x96:
        return 0

    delta = sqrt_price_b_x96 - sqrt_price_a_x96
    if round_up:
        return _mul_div_rounding_up(liquidity, delta, Q96)
    return liquidity * delta // Q96


def _next_sqrt_price_from_amount0_in(
    sqrt_price_x96: int, liquidity: int, amount: int
) -> int:
    """Price after adding `amount` of token0. Price falls (zero-for-one)."""
    if amount == 0:
        return sqrt_price_x96
    numerator1 = liquidity << 96
    product = amount * sqrt_price_x96
    denominator = numerator1 + product
    if denominator >= numerator1:
        return _mul_div_rounding_up(numerator1, sqrt_price_x96, denominator)
    # Overflow path from the contract: fall back to the division form.
    return _mul_div_rounding_up(numerator1, 1, (numerator1 // sqrt_price_x96) + amount)


def _next_sqrt_price_from_amount1_in(
    sqrt_price_x96: int, liquidity: int, amount: int
) -> int:
    """Price after adding `amount` of token1. Price rises (one-for-zero)."""
    if amount == 0:
        return sqrt_price_x96
    return sqrt_price_x96 + (amount << 96) // liquidity


@dataclass(frozen=True)
class SwapResult:
    """The outcome of a local swap, including whether the data ran out.

    `range_exhausted` exists because a snapshot only holds the ticks within the
    range it scanned. A swap large enough to reach that edge stops with input left
    over, and the partial output UNDER-states the price -- safe in direction, but it
    makes two states indistinguishable:

        the pool really is that thin       -> drop the pair
        we only scanned 50 spacings        -> re-read with a wider range

    Those call for opposite actions, so the flag is not optional detail.
    """

    amount_out: int
    amount_in_consumed: int
    range_exhausted: bool
    ticks_crossed: int = 0
    final_sqrt_price_x96: int = 0


@dataclass(frozen=True)
class TickInfo:
    """One initialised tick and the liquidity delta on crossing it upward."""

    tick: int
    liquidity_net: int


@dataclass(frozen=True)
class V3Pool:
    """An immutable snapshot of a v3 pool, quotable at any size.

    Frozen, not merely by convention. `swap_exact_in` works on local copies of the
    price and liquidity precisely so twenty sizes can be quoted against one recorded
    state; a mutating swap would make the second quote of a size curve wrong, and
    the error would look exactly like price impact.

    Immutable in use: `swap_exact_in` never mutates it, because the entire point is
    to quote twenty sizes against one recorded state. A swap that moved the state
    would make the second quote of a size curve wrong, and the error would look
    like price impact.
    """

    sqrt_price_x96: int
    liquidity: int
    tick: int
    fee: int
    tick_spacing: int
    ticks: Sequence[TickInfo]
    decimals0: int
    decimals1: int
    # For the record, so a stored snapshot says which block it describes.
    block_number: Optional[int] = None
    address: Optional[str] = None

    def __post_init__(self):
        if self.fee < 0 or self.fee >= FEE_DENOMINATOR:
            raise ValueError(f"fee {self.fee} is outside (0, 1_000_000)")
        if self.liquidity < 0:
            raise ValueError("liquidity must be non-negative")
        for name, value in (("decimals0", self.decimals0), ("decimals1", self.decimals1)):
            if not 0 <= value <= 36:
                raise ValueError(f"{name} is {value}, outside 0..36")
        # Sorted once so the swap loop can walk them in order.
        object.__setattr__(
            self, "ticks", tuple(sorted(self.ticks, key=lambda t: t.tick))
        )

    # ------------------------------------------------------------------

    def spot_price(self) -> Decimal:
        return price_from_sqrt_price_x96(
            self.sqrt_price_x96, self.decimals0, self.decimals1
        )

    def _next_initialised_tick(self, current: int, zero_for_one: bool):
        """The next initialised tick in the direction of travel, or None."""
        if zero_for_one:
            candidates = [t for t in self.ticks if t.tick < current]
            return candidates[-1] if candidates else None
        candidates = [t for t in self.ticks if t.tick > current]
        return candidates[0] if candidates else None

    def swap_exact_in(self, amount_in: int, zero_for_one: bool) -> int:
        """Output amount for an exact input, in integer token units.

        The simple interface, for callers that only want the number -- the
        differential test against QuoterV2 among them. Use
        `swap_exact_in_detailed` when it matters whether the recorded tick range
        was sufficient.
        """
        return self.swap_exact_in_detailed(amount_in, zero_for_one).amount_out

    def swap_exact_in_detailed(
        self, amount_in: int, zero_for_one: bool
    ) -> SwapResult:
        """Output amount plus whether the recorded tick range was enough.

        Mirrors the contract's swap loop: step to the next initialised tick, fill
        what the current range can absorb, cross and pick up the next range's
        liquidity, repeat. Returns what could actually be filled -- a swap larger
        than the pool holds returns the partial amount rather than looping, because
        an unbounded loop in a hot path is worse than an imprecise answer.
        """
        if amount_in < 0:
            raise ValueError(f"amount_in must be non-negative, got {amount_in}")
        if amount_in == 0 or self.liquidity == 0:
            # An empty pool is a real state -- the survey found several -- and must
            # not raise ZeroDivisionError inside the detector.
            return SwapResult(
                amount_out=0, amount_in_consumed=0, range_exhausted=False,
                final_sqrt_price_x96=self.sqrt_price_x96,
            )

        sqrt_price = self.sqrt_price_x96
        liquidity = self.liquidity
        tick = self.tick
        remaining = amount_in
        amount_out_total = 0
        crossed = 0
        exhausted = False
        # A bound rather than `while True`: a malformed tick list must not hang the
        # detector. 1024 crossings is far beyond any real pool at any real size.
        for _ in range(1024):
            if remaining <= 0 or liquidity <= 0:
                break

            next_tick = self._next_initialised_tick(tick, zero_for_one)
            if next_tick is None:
                target_sqrt_price = (
                    MIN_SQRT_RATIO + 1 if zero_for_one else MAX_SQRT_RATIO - 1
                )
                target_tick = None
            else:
                target_sqrt_price = sqrt_price_x96_from_tick(next_tick.tick)
                target_tick = next_tick

            # How much input this range can absorb before reaching the boundary.
            if zero_for_one:
                max_in = amount0_delta(
                    target_sqrt_price, sqrt_price, liquidity, round_up=True
                )
            else:
                max_in = amount1_delta(
                    sqrt_price, target_sqrt_price, liquidity, round_up=True
                )

            # The fee is taken from the input, before it reaches the curve.
            fee_on_max = _mul_div_rounding_up(
                max_in, self.fee, FEE_DENOMINATOR - self.fee
            ) if max_in > 0 else 0

            if remaining >= max_in + fee_on_max and max_in > 0:
                # This range is fully consumed; step to the boundary and cross.
                amount_in_step = max_in
                next_sqrt_price = target_sqrt_price
                remaining -= (max_in + fee_on_max)
            else:
                # The swap ends inside this range.
                amount_in_step = remaining - _mul_div_rounding_up(
                    remaining, self.fee, FEE_DENOMINATOR
                )
                if zero_for_one:
                    next_sqrt_price = _next_sqrt_price_from_amount0_in(
                        sqrt_price, liquidity, amount_in_step
                    )
                else:
                    next_sqrt_price = _next_sqrt_price_from_amount1_in(
                        sqrt_price, liquidity, amount_in_step
                    )
                remaining = 0

            # Output for the step, rounded DOWN: amounts paid out round against
            # the trader.
            if zero_for_one:
                amount_out_total += amount1_delta(
                    next_sqrt_price, sqrt_price, liquidity, round_up=False
                )
            else:
                amount_out_total += amount0_delta(
                    sqrt_price, next_sqrt_price, liquidity, round_up=False
                )

            sqrt_price = next_sqrt_price

            if target_tick is not None and sqrt_price == target_sqrt_price:
                # Crossing: liquidity_net is signed for an upward crossing, so it
                # is subtracted when travelling down.
                liquidity += (
                    -target_tick.liquidity_net if zero_for_one
                    else target_tick.liquidity_net
                )
                tick = target_tick.tick - 1 if zero_for_one else target_tick.tick
                crossed += 1
            elif remaining <= 0:
                break
            else:
                # Input left over with no further initialised tick: the recorded
                # range has been walked to its edge. The caller needs to know.
                exhausted = True
                break

        if remaining > 0:
            exhausted = True

        return SwapResult(
            amount_out=amount_out_total,
            amount_in_consumed=amount_in - max(remaining, 0),
            range_exhausted=exhausted,
            ticks_crossed=crossed,
            final_sqrt_price_x96=sqrt_price,
        )

    # ------------------------------------------------------------------

    def price_for_amount_in(
        self, amount_in: Decimal, zero_for_one: bool
    ) -> Optional[Decimal]:
        """Effective price for a human-scale input amount.

        The decimals conversion lives here rather than at each call site, where a
        mistake would be a factor of 10^n rather than a rounding difference.
        """
        if amount_in <= 0:
            return None
        decimals_in = self.decimals0 if zero_for_one else self.decimals1
        decimals_out = self.decimals1 if zero_for_one else self.decimals0

        raw_in = int(amount_in * (Decimal(10) ** decimals_in))
        result = self.swap_exact_in_detailed(raw_in, zero_for_one=zero_for_one)
        if result.amount_out <= 0:
            return None
        if result.range_exhausted:
            # A price from a partial fill is not the price of the requested size.
            # Returning it would understate the edge -- safe -- but a caller could
            # not tell it from a genuine thin-pool price, and those need opposite
            # responses. Refuse, so the snapshot can be re-read with a wider range.
            return None

        out = Decimal(result.amount_out) / (Decimal(10) ** decimals_out)
        return out / amount_in

    def price_curve(
        self, sizes: Iterable[Decimal], zero_for_one: bool
    ) -> List[Tuple[Decimal, Optional[Decimal]]]:
        """Effective price at each size: the input to `argmax_q Pi(q)`.

        This is the function the RPC path could not afford. Twenty sizes against one
        recorded state cost nothing here and twenty eth_calls there, which is why
        the system has only ever asked about one fixed notional.
        """
        return [(size, self.price_for_amount_in(size, zero_for_one)) for size in sizes]
