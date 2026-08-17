"""A placebo arm, to separate real edge from latency illusion.

The quant audit's central methodological objection: markout computed from
later rows re-samples BOTH venues, so it answers "would the detector still
fire?" not "was the trade worth anything?" If the entire apparent edge were a
stale CEX book, the measured decay curve would look identical to a real,
decaying arbitrage. The methodology could not tell the two apart.

The cheapest fix with the highest information value: alongside each live
evaluation, evaluate the SAME CEX book against a DEX quote from N cycles ago.
Under the null hypothesis that the "edge" is really just data staleness, the
placebo distribution matches the live one. If they diverge, the live edge
contains something the delay does not explain.

This costs no extra RPC calls -- the delayed quote is one already fetched.
"""
from decimal import Decimal

import pytest

from src.strategy.placebo import DelayedQuoteBuffer


def D(x) -> Decimal:
    return Decimal(str(x))


def test_a_cold_buffer_offers_no_delayed_quote():
    """Nothing to compare against until enough history exists, and it must say
    so rather than substituting the live quote."""
    buf = DelayedQuoteBuffer(delay_cycles=3)
    buf.push("ETH/USDT", "sell", D(1000))
    assert buf.delayed("ETH/USDT", "sell") is None


def test_it_returns_the_quote_from_exactly_n_cycles_ago():
    buf = DelayedQuoteBuffer(delay_cycles=3)
    for price in (1000, 1001, 1002, 1003):
        buf.push("ETH/USDT", "sell", D(price))

    # with delay 3, the 4th push makes the 1st available
    assert buf.delayed("ETH/USDT", "sell") == D(1000)

    buf.push("ETH/USDT", "sell", D(1004))
    assert buf.delayed("ETH/USDT", "sell") == D(1001)


def test_series_are_isolated_by_pair_and_side():
    """A placebo that mixed sides or pairs would compare unrelated numbers and
    produce a meaningless null distribution."""
    buf = DelayedQuoteBuffer(delay_cycles=1)
    buf.push("ETH/USDT", "sell", D(1000))
    buf.push("ETH/USDT", "buy", D(2000))
    buf.push("ARB/USDT", "sell", D(3000))
    buf.push("ETH/USDT", "sell", D(1001))
    buf.push("ETH/USDT", "buy", D(2001))
    buf.push("ARB/USDT", "sell", D(3001))

    assert buf.delayed("ETH/USDT", "sell") == D(1000)
    assert buf.delayed("ETH/USDT", "buy") == D(2000)
    assert buf.delayed("ARB/USDT", "sell") == D(3000)


def test_the_buffer_is_bounded():
    """This runs for weeks in a hot loop; unbounded history is a leak."""
    buf = DelayedQuoteBuffer(delay_cycles=2)
    for i in range(10_000):
        buf.push("ETH/USDT", "sell", D(i))

    assert buf.size("ETH/USDT", "sell") <= 3, "must keep only what the delay needs"
    assert buf.delayed("ETH/USDT", "sell") == D(9997)


def test_zero_delay_is_rejected():
    """A zero-cycle delay is not a placebo, it is the live arm."""
    with pytest.raises(ValueError):
        DelayedQuoteBuffer(delay_cycles=0)


def test_a_negative_delay_is_rejected():
    with pytest.raises(ValueError):
        DelayedQuoteBuffer(delay_cycles=-1)


# --------------------------------------------------------------------------
# integration: the placebo must be recorded alongside the live evaluation
# --------------------------------------------------------------------------

async def test_placebo_net_bps_is_recorded_once_history_exists():
    from src.core.config import PlaceboConfig, RotationConfig, StrategyConfig
    from src.strategy.detector import OpportunityDetector
    from tests.fakes import FakeCex, FakeDex, flat_book, make_pair

    pair = make_pair()
    cex = FakeCex({"ETH/USDT": flat_book(bid=1000, ask=1000)})
    dex = FakeDex(sell_price=1050, buy_price=1050)

    class Rec:
        def __init__(self): self.rows = []
        def record(self, r): self.rows.append(r); return len(self.rows)

    rec = Rec()
    det = OpportunityDetector(
        StrategyConfig(
            target_notional_usd=1000, taker_fee_bps=D("7.5"), min_net_bps=D(5),
            rotation=RotationConfig(enabled=False),
            placebo=PlaceboConfig(enabled=True, delay_cycles=2)),
        cex, dex, [pair], store=rec)

    # first two cycles cannot have a placebo yet
    await det.detect()
    await det.detect()
    assert all(r.placebo_net_bps is None for r in rec.rows), "no history yet"

    await det.detect()
    later = [r for r in rec.rows if r.placebo_net_bps is not None]
    assert later, "a placebo value must appear once the buffer is warm"


async def test_a_constant_market_makes_the_placebo_match_the_live_arm():
    """The null hypothesis, made concrete. With prices that never move, a
    delayed quote is identical to the live one, so the placebo edge must equal
    the live edge. Any divergence here would mean the plumbing is wrong."""
    from src.core.config import PlaceboConfig, RotationConfig, StrategyConfig
    from src.strategy.detector import OpportunityDetector
    from tests.fakes import FakeCex, FakeDex, flat_book, make_pair

    pair = make_pair()
    cex = FakeCex({"ETH/USDT": flat_book(bid=1000, ask=1000)})
    dex = FakeDex(sell_price=1050, buy_price=1050)

    class Rec:
        def __init__(self): self.rows = []
        def record(self, r): self.rows.append(r); return len(self.rows)

    rec = Rec()
    det = OpportunityDetector(
        StrategyConfig(
            target_notional_usd=1000, taker_fee_bps=D("7.5"), min_net_bps=D(5),
            rotation=RotationConfig(enabled=False),
            placebo=PlaceboConfig(enabled=True, delay_cycles=1)),
        cex, dex, [pair], store=rec)

    for _ in range(4):
        await det.detect()

    paired = [r for r in rec.rows
              if r.placebo_net_bps is not None and r.net_bps is not None]
    assert paired, "expected paired live/placebo observations"
    for row in paired:
        assert row.placebo_net_bps == row.net_bps, (
            "in a constant market the delayed quote equals the live one"
        )


async def test_placebo_disabled_records_nothing():
    from src.core.config import PlaceboConfig, RotationConfig, StrategyConfig
    from src.strategy.detector import OpportunityDetector
    from tests.fakes import FakeCex, FakeDex, flat_book, make_pair

    pair = make_pair()
    cex = FakeCex({"ETH/USDT": flat_book(bid=1000, ask=1000)})

    class Rec:
        def __init__(self): self.rows = []
        def record(self, r): self.rows.append(r); return len(self.rows)

    rec = Rec()
    det = OpportunityDetector(
        StrategyConfig(target_notional_usd=1000, min_net_bps=D(5),
                       rotation=RotationConfig(enabled=False),
                       placebo=PlaceboConfig(enabled=False)),
        cex, FakeDex(1050, 1050), [pair], store=rec)
    for _ in range(3):
        await det.detect()

    assert all(r.placebo_net_bps is None for r in rec.rows)
