"""A placebo arm for the edge measurement -- delayed by TIME, not by cycles.

The methodological objection this answers: markout computed from later rows
re-samples *both* venues, so it measures whether the detector would still fire,
not whether the trade was worth anything. If the entire apparent edge were a
stale CEX book, the decay curve would look exactly like a real, decaying
arbitrage. Nothing in the data would distinguish them.

The control: alongside each live evaluation, evaluate the same CEX book against
a DEX quote from N seconds ago. Under the null hypothesis -- the measured edge is
an artefact of staleness rather than genuine mispricing -- the placebo
distribution matches the live one.

WHY SECONDS AND NOT CYCLES. The first version of this delayed by a count of
detection cycles: 5 cycles at a 0.2s loop, about one second. Run live, it
produced 94 paired observations whose live and placebo values were IDENTICAL in
69% of cases and had a median difference of 0.00 bps -- which looks like decisive
evidence for the null and is in fact evidence of nothing at all.

The reason is that a Uniswap v3 quote only changes when a block lands. On
Ethereum that is roughly 12 seconds, so every quote taken within the same block
is the same number by construction, and a one-second delay compares a quote to
itself. The control was measuring the block time, not the market.

So the delay is in seconds and must exceed the block time of the slowest chain
being quoted. That is validated rather than documented: a placebo shorter than a
block cannot answer the question it exists to answer, and it fails in the
direction that manufactures a comforting result.
"""
from decimal import Decimal

import pytest

from src.strategy.placebo import (
    CHAIN_BLOCK_SECONDS, DelayedQuoteBuffer, min_delay_seconds_for,
)


def D(x) -> Decimal:
    return Decimal(str(x))


class FakeClock:
    def __init__(self, t=1000.0):
        self.t = t

    def __call__(self):
        return self.t

    def advance(self, seconds):
        self.t += seconds


# --- the buffer ----------------------------------------------------------


def test_a_cold_buffer_offers_no_delayed_quote():
    """Nothing to compare against until enough history exists, and it must say
    so rather than substituting the live quote -- which would make the control
    silently agree with the live arm."""
    clock = FakeClock()
    buf = DelayedQuoteBuffer(delay_seconds=30, now_fn=clock)
    buf.push("ETH/USDT", "sell", D(1000))

    assert buf.delayed("ETH/USDT", "sell") is None


def test_a_quote_younger_than_the_delay_is_not_served():
    clock = FakeClock()
    buf = DelayedQuoteBuffer(delay_seconds=30, now_fn=clock)
    buf.push("ETH/USDT", "sell", D(1000))

    clock.advance(29.9)
    buf.push("ETH/USDT", "sell", D(1001))

    assert buf.delayed("ETH/USDT", "sell") is None


def test_the_oldest_quote_at_least_the_delay_old_is_served():
    clock = FakeClock()
    buf = DelayedQuoteBuffer(delay_seconds=30, now_fn=clock)
    buf.push("ETH/USDT", "sell", D(1000))

    clock.advance(30.0)
    buf.push("ETH/USDT", "sell", D(1001))

    assert buf.delayed("ETH/USDT", "sell") == D(1000)


def test_the_most_recent_eligible_quote_is_served_not_the_oldest():
    """With a 30s delay and quotes at 0, 10, 40 and 50 seconds, the right answer
    at t=50 is the quote from t=10: the newest one that is still at least 30s
    old. Serving the t=0 quote would delay by 50s, not 30."""
    clock = FakeClock()
    buf = DelayedQuoteBuffer(delay_seconds=30, now_fn=clock)

    buf.push("ETH/USDT", "sell", D(1000))   # t=1000
    clock.advance(10)
    buf.push("ETH/USDT", "sell", D(1001))   # t=1010
    clock.advance(30)
    buf.push("ETH/USDT", "sell", D(1002))   # t=1040
    clock.advance(10)
    buf.push("ETH/USDT", "sell", D(1003))   # t=1050

    assert buf.delayed("ETH/USDT", "sell") == D(1001)


def test_series_are_isolated_by_pair_and_side():
    """A placebo that mixed sides or pairs would compare unrelated numbers and
    produce a meaningless null distribution."""
    clock = FakeClock()
    buf = DelayedQuoteBuffer(delay_seconds=10, now_fn=clock)
    buf.push("ETH/USDT", "sell", D(1000))
    buf.push("ETH/USDT", "buy", D(2000))
    buf.push("ARB/USDT", "sell", D(3000))

    clock.advance(10)
    for symbol, side, price in (("ETH/USDT", "sell", D(1001)),
                                ("ETH/USDT", "buy", D(2001)),
                                ("ARB/USDT", "sell", D(3001))):
        buf.push(symbol, side, price)

    assert buf.delayed("ETH/USDT", "sell") == D(1000)
    assert buf.delayed("ETH/USDT", "buy") == D(2000)
    assert buf.delayed("ARB/USDT", "sell") == D(3000)


def test_the_buffer_is_bounded_by_time_not_by_count():
    """This runs for weeks in a hot loop. Entries older than what the delay needs
    must be discarded, and at a 0.2s loop interval a 30s window is 150 entries
    per series -- so the bound has to be enforced, not assumed."""
    clock = FakeClock()
    buf = DelayedQuoteBuffer(delay_seconds=30, now_fn=clock)

    for i in range(10_000):
        clock.advance(0.2)
        buf.push("ETH/USDT", "sell", D(i))

    # One 30s window at 0.2s per push is 150 entries, plus the one being served.
    assert buf.size("ETH/USDT", "sell") <= 160, (
        f"the buffer holds {buf.size('ETH/USDT', 'sell')} entries"
    )
    assert buf.delayed("ETH/USDT", "sell") is not None


def test_a_zero_delay_is_rejected():
    """A zero delay is not a placebo, it is the live arm."""
    with pytest.raises(ValueError):
        DelayedQuoteBuffer(delay_seconds=0)


def test_a_negative_delay_is_rejected():
    with pytest.raises(ValueError):
        DelayedQuoteBuffer(delay_seconds=-1)


# --- the delay must exceed a block ---------------------------------------


def test_the_block_time_table_covers_the_configured_chains():
    for chain in ("ethereum", "arbitrum", "base", "bsc"):
        assert chain in CHAIN_BLOCK_SECONDS


def test_the_minimum_delay_is_driven_by_the_slowest_chain():
    """Quoting Ethereum and Base together, Ethereum's 12s block sets the floor."""
    slow = min_delay_seconds_for(["base", "ethereum"])
    fast = min_delay_seconds_for(["base"])

    assert slow > fast
    assert slow >= CHAIN_BLOCK_SECONDS["ethereum"]


def test_the_minimum_delay_exceeds_a_block_rather_than_equalling_it():
    """Equal to the block time is not enough: two samples exactly one block apart
    can still land in the same block, which is the failure this prevents."""
    assert min_delay_seconds_for(["ethereum"]) > CHAIN_BLOCK_SECONDS["ethereum"]


def test_an_unknown_chain_is_treated_pessimistically():
    """A chain with no known block time must not silently reduce the floor."""
    assert min_delay_seconds_for(["something-new"]) >= min_delay_seconds_for(
        ["ethereum"]
    )


def test_no_chains_still_yields_a_usable_floor():
    assert min_delay_seconds_for([]) > 0


# --- integration ---------------------------------------------------------


async def test_placebo_net_bps_is_recorded_once_history_exists(monkeypatch):
    from src.core.config import PlaceboConfig, RotationConfig, StrategyConfig
    from src.strategy.detector import OpportunityDetector
    from tests.fakes import FakeCex, FakeDex, flat_book, make_pair

    clock = FakeClock()
    monkeypatch.setattr("src.core.clock.now", clock)

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
            placebo=PlaceboConfig(enabled=True, delay_seconds=30)),
        cex, dex, [pair], store=rec)

    await det.detect()
    assert all(r.placebo_net_bps is None for r in rec.rows), "no history yet"

    clock.advance(30)
    await det.detect()

    later = [r for r in rec.rows if r.placebo_net_bps is not None]
    assert later, "a placebo value must appear once the delay has elapsed"


async def test_a_constant_market_makes_the_placebo_match_the_live_arm(monkeypatch):
    """The null hypothesis, made concrete. With prices that never move, a delayed
    quote is identical to the live one, so the placebo edge must equal the live
    edge. Any divergence here would mean the plumbing is wrong."""
    from src.core.config import PlaceboConfig, RotationConfig, StrategyConfig
    from src.strategy.detector import OpportunityDetector
    from tests.fakes import FakeCex, FakeDex, flat_book, make_pair

    clock = FakeClock()
    monkeypatch.setattr("src.core.clock.now", clock)

    pair = make_pair()
    rec_rows = []

    class Rec:
        def record(self, r): rec_rows.append(r); return len(rec_rows)

    det = OpportunityDetector(
        StrategyConfig(
            target_notional_usd=1000, taker_fee_bps=D("7.5"), min_net_bps=D(5),
            rotation=RotationConfig(enabled=False),
            placebo=PlaceboConfig(enabled=True, delay_seconds=30)),
        FakeCex({"ETH/USDT": flat_book(bid=1000, ask=1000)}),
        FakeDex(sell_price=1050, buy_price=1050), [pair], store=Rec())

    for _ in range(4):
        await det.detect()
        clock.advance(30)

    paired = [r for r in rec_rows
              if r.placebo_net_bps is not None and r.net_bps is not None]
    assert paired, "expected paired live/placebo observations"
    for row in paired:
        assert row.placebo_net_bps == row.net_bps, (
            "in a constant market the delayed quote equals the live one"
        )


async def test_a_moving_market_makes_the_placebo_diverge(monkeypatch):
    """The other half, and the one the live run could not produce: when the DEX
    price actually moves between the two samples, the two arms must differ.

    Without this test, a placebo that always returned the live quote would pass
    every other test in this file.
    """
    from src.core.config import PlaceboConfig, RotationConfig, StrategyConfig
    from src.strategy.detector import OpportunityDetector
    from tests.fakes import FakeCex, FakeDex, flat_book, make_pair

    clock = FakeClock()
    monkeypatch.setattr("src.core.clock.now", clock)

    pair = make_pair()
    dex = FakeDex(sell_price=1050, buy_price=1050)
    rows = []

    class Rec:
        def record(self, r): rows.append(r); return len(rows)

    det = OpportunityDetector(
        StrategyConfig(
            target_notional_usd=1000, taker_fee_bps=D("7.5"), min_net_bps=D(5),
            rotation=RotationConfig(enabled=False),
            placebo=PlaceboConfig(enabled=True, delay_seconds=30)),
        FakeCex({"ETH/USDT": flat_book(bid=1000, ask=1000)}), dex, [pair],
        store=Rec())

    await det.detect()
    clock.advance(30)
    # The DEX moves by 1% between the two samples.
    dex.sell_price = D(1060)
    dex.buy_price = D(1060)
    await det.detect()

    diverged = [
        r for r in rows
        if r.placebo_net_bps is not None and r.net_bps is not None
        and r.placebo_net_bps != r.net_bps
    ]
    assert diverged, (
        "the placebo tracked the live quote even though the market moved -- the "
        "control is inert"
    )


async def test_placebo_disabled_records_nothing():
    from src.core.config import PlaceboConfig, RotationConfig, StrategyConfig
    from src.strategy.detector import OpportunityDetector
    from tests.fakes import FakeCex, FakeDex, flat_book, make_pair

    rows = []

    class Rec:
        def record(self, r): rows.append(r); return len(rows)

    det = OpportunityDetector(
        StrategyConfig(target_notional_usd=1000, min_net_bps=D(5),
                       rotation=RotationConfig(enabled=False),
                       placebo=PlaceboConfig(enabled=False)),
        FakeCex({"ETH/USDT": flat_book(bid=1000, ask=1000)}),
        FakeDex(1050, 1050), [make_pair()], store=Rec())
    for _ in range(3):
        await det.detect()

    assert all(r.placebo_net_bps is None for r in rows)
