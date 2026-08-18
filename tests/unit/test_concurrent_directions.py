"""Both directions of a pair must be measured at the same instant.

`_evaluate_direct` returned:

    return [
        await self._eval_cex_to_dex(...),
        await self._eval_dex_to_cex(...),
    ]

Those awaits are sequential, so the second direction's DEX quote is taken after
the first one's has completed. With a measured 0.31-0.83s median RPC latency and
a gas call per direction, the two directions were sampled up to a second apart:

    t=0ms      CEX book snapshot  (shared, so this side is consistent)
    t=310ms    DEX quote, direction A
    t=620ms    gas
    t=930ms    DEX quote, direction B
    t=1240ms   gas

The CEX side is fine -- one snapshot is taken and both directions use it. The DEX
side is not: the two directions are compared against pool states up to a second
apart, and `_decide` then picks "the better direction" from two observations that
were never simultaneous. On a moving pool that comparison is partly a coin flip on
which direction happened to be measured first.

Fixed with gather. The same fix roughly halves the pair's contribution to cycle
time, which matters independently: the measured cadence is 2.32s against a
configured 0.2s.
"""
import asyncio
from decimal import Decimal

import pytest

from src.core.config import RotationConfig, StrategyConfig, TokenPolicyConfig
from src.core.types import DexQuote
from src.strategy.detector import OpportunityDetector
from tests.fakes import FakeCex, flat_book, make_pair


def D(x) -> Decimal:
    return Decimal(str(x))


class TimedDex:
    """Records when each quote was requested, and how many were concurrent."""

    def __init__(self, delay=0.05, price=1100):
        self.delay = delay
        self.price = D(price)
        self.in_flight = 0
        self.peak_in_flight = 0
        self.request_times = []

    async def get_pool_address(self, *a, **k):
        return None

    async def get_quote(self, pair, size, side, estimate_gas=False):
        self.in_flight += 1
        self.peak_in_flight = max(self.peak_in_flight, self.in_flight)
        self.request_times.append(asyncio.get_running_loop().time())
        try:
            await asyncio.sleep(self.delay)
            return DexQuote(price=self.price, gas_cost_quote=D(0))
        finally:
            self.in_flight -= 1


def _strategy(**kw) -> StrategyConfig:
    fields = dict(
        target_notional_usd=1000, taker_fee_bps=D("7.5"), min_net_bps=D(5),
        rotation=RotationConfig(enabled=False),
        token_policy=TokenPolicyConfig(mode="denylist"),
        dex_routing={"enabled": False},
    )
    fields.update(kw)
    return StrategyConfig(**fields)


async def test_the_two_directions_are_quoted_concurrently():
    """The property: both DEX quotes in flight at once, not one after the other."""
    dex = TimedDex(delay=0.05)
    det = OpportunityDetector(
        _strategy(), FakeCex({"ETH/USDT": flat_book(bid=1000, ask=1000)}),
        dex, [make_pair()],
    )

    await det.detect()

    assert dex.peak_in_flight == 2, (
        f"peak concurrency was {dex.peak_in_flight}; the two directions are still "
        f"sequential, so they measure the pool up to a full RPC latency apart"
    )


async def test_the_quotes_are_taken_within_a_few_milliseconds_of_each_other():
    """The consequence that matters: they describe the same instant."""
    dex = TimedDex(delay=0.05)
    det = OpportunityDetector(
        _strategy(), FakeCex({"ETH/USDT": flat_book(bid=1000, ask=1000)}),
        dex, [make_pair()],
    )

    await det.detect()

    assert len(dex.request_times) == 2
    skew = abs(dex.request_times[1] - dex.request_times[0])
    assert skew < dex.delay / 2, (
        f"the two directions were sampled {skew * 1000:.0f}ms apart against a "
        f"{dex.delay * 1000:.0f}ms quote latency"
    )


async def test_a_pair_costs_one_quote_latency_not_two():
    """Halving the per-pair cycle contribution. The measured cadence is 2.32s
    against a configured 0.2s, so this is not a micro-optimisation."""
    dex = TimedDex(delay=0.1)
    det = OpportunityDetector(
        _strategy(), FakeCex({"ETH/USDT": flat_book(bid=1000, ask=1000)}),
        dex, [make_pair()],
    )

    loop = asyncio.get_running_loop()
    started = loop.time()
    await det.detect()
    elapsed = loop.time() - started

    assert elapsed < dex.delay * 1.8, (
        f"one cycle took {elapsed:.3f}s for a {dex.delay:.3f}s quote; the two "
        f"directions are still serialised"
    )


async def test_one_failing_direction_does_not_lose_the_other():
    """Concurrency must not turn a single-direction failure into a lost pair --
    the whole point of the per-pair isolation elsewhere in the detector."""
    class HalfBroken(TimedDex):
        async def get_quote(self, pair, size, side, estimate_gas=False):
            if side == "sell":
                raise RuntimeError("quoter reverted")
            return DexQuote(price=self.price, gas_cost_quote=D(0))

    rows = []

    class Rec:
        def record(self, r):
            rows.append(r)
            return len(rows)

    det = OpportunityDetector(
        _strategy(), FakeCex({"ETH/USDT": flat_book(bid=1000, ask=1000)}),
        HalfBroken(), [make_pair()], store=Rec(),
    )

    await det.detect()

    directions = {r.direction for r in rows}
    assert "DEX_to_CEX" in directions, (
        f"the working direction was lost when the other failed; recorded "
        f"{directions}"
    )


async def test_both_directions_still_share_one_book_snapshot():
    """Concurrency on the DEX side must not accidentally introduce a second CEX
    fetch -- the two directions comparing different books would be a worse version
    of the defect being fixed."""
    cex = FakeCex({"ETH/USDT": flat_book(bid=1000, ask=1000)})
    det = OpportunityDetector(
        _strategy(), cex, TimedDex(), [make_pair()],
    )

    await det.detect()

    assert cex.book_calls == 1, (
        f"the book was fetched {cex.book_calls} times for one pair"
    )
