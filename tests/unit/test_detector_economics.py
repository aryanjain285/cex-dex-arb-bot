"""The detector must decide on net economics, from depth-weighted prices.

Three defects are pinned here:

- the CEX taker fee was a hardcoded module constant, so no config change
  could correct it for a different fee tier;
- prices came from top-of-book regardless of trade size, while the fetched
  depth went unused, so the signal was wrong on exactly the thin pairs the
  strategy targets;
- one pair raising an exception took down the detection cycle for every
  other pair, because `asyncio.gather` was called without
  `return_exceptions=True`.
"""
from decimal import Decimal
from typing import List, Optional, Tuple

import pytest

from src.core import clock
from src.core.config import StrategyConfig
from src.core.types import BookSnapshot, DexQuote, MarketPair
from src.strategy.detector import OpportunityDetector


def D(x) -> Decimal:
    return Decimal(str(x))


def make_pair(symbol="ETH/USDT", **kw) -> MarketPair:
    defaults = dict(
        base="WETH", quote_cex="USDT", quote_dex="USDT", cex_symbol=symbol,
        dex_chain="ethereum", dex_pool_fee=500,
    )
    defaults.update(kw)
    return MarketPair(**defaults)


def strategy(**kw) -> StrategyConfig:
    defaults = dict(target_notional_usd=1000, taker_fee_bps=D("7.5"), min_net_bps=D(5))
    defaults.update(kw)
    return StrategyConfig(**defaults)


class FakeCex:
    """Serves a fixed book. Real object, not a mock -- the detector's use of
    depth is exactly what is under test."""

    def __init__(self, books: dict):
        self.books = books
        self.calls = 0

    async def get_book(self, pair: MarketPair) -> Optional[BookSnapshot]:
        self.calls += 1
        entry = self.books.get(pair.cex_symbol)
        if entry is None:
            return None
        bids, asks = entry
        return BookSnapshot(pair=pair, bids=bids, asks=asks, timestamp=clock.now())


class FakeDex:
    def __init__(self, sell_price, buy_price, gas=D(0)):
        self.sell_price, self.buy_price, self.gas = sell_price, buy_price, gas

    async def get_quote(self, pair, size, side, estimate_gas=False):
        price = self.sell_price if side == "sell" else self.buy_price
        return DexQuote(price=price, gas_cost_quote=self.gas)


def flat_book(bid, ask, depth=D(1000)):
    return ([(bid, depth)], [(ask, depth)])


# --------------------------------------------------------------------------

async def test_taker_fee_is_read_from_config_not_hardcoded():
    """Setting the fee to zero must remove the fee from the economics."""
    pair = make_pair()
    cex = FakeCex({"ETH/USDT": flat_book(D(1000), D(1000))})
    dex = FakeDex(sell_price=D(1010), buy_price=D(1010))

    detector = OpportunityDetector(strategy(taker_fee_bps=D(0)), cex, dex, [pair])
    opps = await detector.detect()

    assert opps, "a 100 bps gross edge with zero fees must be detected"
    assert opps[0].cex_fee_quote == D(0)


async def test_detection_requires_net_bps_above_the_configured_floor():
    pair = make_pair()
    cex = FakeCex({"ETH/USDT": flat_book(D(1000), D(1000))})
    # 10 bps gross, and the 7.5 bps taker fee leaves ~2.5 bps net
    dex = FakeDex(sell_price=D(1001), buy_price=D(1001))

    permissive = OpportunityDetector(strategy(min_net_bps=D(1)), cex, dex, [pair])
    strict = OpportunityDetector(strategy(min_net_bps=D(50)), cex, dex, [pair])

    assert await permissive.detect(), "2.5 bps net should clear a 1 bps floor"
    assert not await strict.detect(), "2.5 bps net must not clear a 50 bps floor"


async def test_price_is_depth_weighted_not_top_of_book():
    """The regression guard for depth-blindness.

    Top of book is 1000 but only covers a sliver of the trade; the rest fills
    at 1100. Pricing at top-of-book would show a large edge that cannot be
    achieved. The detector must use the VWAP.
    """
    pair = make_pair()
    thin_asks = [(D(1000), D("0.01")), (D(1100), D(1000))]
    cex = FakeCex({"ETH/USDT": ([(D(999), D(1000))], thin_asks)})
    dex = FakeDex(sell_price=D(1050), buy_price=D(1050))

    detector = OpportunityDetector(strategy(min_net_bps=D(5)), cex, dex, [pair])
    opps = await detector.detect()

    # size ~= 1000/vwap; vwap is far above 1000, so selling at 1050 is a loss
    assert not opps, (
        "buying at a ~1100 VWAP to sell at 1050 is a loss; only top-of-book "
        "pricing would make this look profitable"
    )


async def test_insufficient_cex_depth_skips_the_pair():
    pair = make_pair()
    # only 0.001 base available: nowhere near a $1000 notional
    cex = FakeCex({"ETH/USDT": ([(D(999), D("0.001"))], [(D(1000), D("0.001"))])})
    dex = FakeDex(sell_price=D(2000), buy_price=D(2000))

    detector = OpportunityDetector(strategy(), cex, dex, [pair])
    assert not await detector.detect(), "must not trade on depth that cannot fill"


async def test_missing_book_skips_the_pair_without_rest_fallback():
    """Detection requires a live book.

    Falling back to a REST top-of-book query would be both depth-blind and,
    at scale, a rate-limit violation: at 50 pairs on a 200ms loop the weight
    cost is ~5x the 6000/min budget.
    """
    pair = make_pair()
    cex = FakeCex({})  # no book for this pair
    dex = FakeDex(sell_price=D(2000), buy_price=D(2000))

    detector = OpportunityDetector(strategy(), cex, dex, [pair])
    assert not await detector.detect()


async def test_synthetic_pair_is_charged_two_cex_legs():
    synthetic = make_pair(
        symbol="ALT/USDT", base="ALT", quote_dex="ETH",
        is_synthetic=True, intermediate_symbol="ETH",
    )
    direct = make_pair(symbol="ALT/USDT", base="ALT")
    books = {"ALT/USDT": flat_book(D(1000), D(1000)), "ETH/USDT": flat_book(D(1), D(1))}
    dex = FakeDex(sell_price=D(1010), buy_price=D(1010))

    syn_det = OpportunityDetector(strategy(min_net_bps=D(1)), FakeCex(books), dex, [synthetic])
    dir_det = OpportunityDetector(strategy(min_net_bps=D(1)), FakeCex(books), dex, [direct])

    syn = await syn_det.detect()
    dct = await dir_det.detect()
    assert syn and dct
    assert syn[0].cex_fee_quote == dct[0].cex_fee_quote * 2


async def test_stale_book_is_rejected():
    """A stalled feed must not be mistaken for a quiet market."""
    pair = make_pair()
    cex = FakeCex({"ETH/USDT": flat_book(D(1000), D(1000))})
    dex = FakeDex(sell_price=D(1050), buy_price=D(1050))

    detector = OpportunityDetector(
        strategy(min_net_bps=D(1), max_book_age_seconds=0.001), cex, dex, [pair]
    )
    # force the snapshot to look old
    original = cex.get_book

    async def stale(pair):
        book = await original(pair)
        book.timestamp -= 60.0
        return book

    cex.get_book = stale
    assert not await detector.detect()


async def test_one_failing_pair_does_not_block_the_others():
    """Error isolation. Previously a single raising pair aborted the whole
    gather, so main_loop caught it and slept -- detecting nothing at all."""
    good = make_pair(symbol="ETH/USDT")
    bad = make_pair(symbol="BAD/USDT", base="BAD")

    class ExplodingDex(FakeDex):
        async def get_quote(self, pair, size, side, estimate_gas=False):
            if pair.base == "BAD":
                raise RuntimeError("simulated RPC failure")
            return await super().get_quote(pair, size, side, estimate_gas)

    cex = FakeCex({
        "ETH/USDT": flat_book(D(1000), D(1000)),
        "BAD/USDT": flat_book(D(1000), D(1000)),
    })
    dex = ExplodingDex(sell_price=D(1050), buy_price=D(1050))

    detector = OpportunityDetector(strategy(min_net_bps=D(1)), cex, dex, [bad, good])
    opps = await detector.detect()

    assert len(opps) == 1
    assert opps[0].pair.cex_symbol == "ETH/USDT"
