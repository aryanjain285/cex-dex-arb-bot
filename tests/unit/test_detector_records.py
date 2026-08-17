"""The detector must emit an audit record for every evaluation.

Three audits independently found that `EvaluationStore` was built, tested, and
imported by no production code -- so a paper run would produce a scrollback
buffer rather than a dataset, and none of the questions the run exists to
answer could be answered.

Wiring it is not a one-line change. `detect()` returned `List[Opportunity]`
and rejections returned `None`, so the near-miss distribution -- the entire
point of the exercise -- was unreachable through the return type. The detector
now produces an `EvaluationRecord` per direction attempted, carrying the
rejection reason, and derives opportunities from the taken ones.
"""
from decimal import Decimal

import pytest

from src.core.config import StrategyConfig
from src.core.types import BookSnapshot, MarketPair
from src.strategy.detector import OpportunityDetector, RejectionReason
from tests.fakes import D, FakeCex, FakeDex, flat_book, make_pair


def strategy(**kw) -> StrategyConfig:
    defaults = dict(target_notional_usd=1000, taker_fee_bps=D("7.5"), min_net_bps=D(5))
    defaults.update(kw)
    return StrategyConfig(**defaults)


class Recorder:
    """Collects records instead of writing SQLite, so the detector's emission
    is what is under test rather than the store's persistence."""

    def __init__(self):
        self.records = []

    def record(self, record):
        self.records.append(record)
        return len(self.records)


def detector(cex, dex, pairs, recorder=None, **cfg):
    return OpportunityDetector(strategy(**cfg), cex, dex, pairs, store=recorder)


# --------------------------------------------------------------------------

async def test_both_directions_are_recorded_for_a_taken_opportunity():
    pair = make_pair()
    rec = Recorder()
    cex = FakeCex({"ETH/USDT": flat_book(bid=1000, ask=1000)})
    dex = FakeDex(sell_price=1050, buy_price=1050)

    opps = await detector(cex, dex, [pair], rec).detect()

    assert len(opps) == 1
    assert len(rec.records) == 2, "one record per direction evaluated"
    outcomes = {r.outcome for r in rec.records}
    assert outcomes == {"taken", "rejected"}


async def test_a_near_miss_is_recorded_with_its_edge_and_reason():
    """The whole point: a rejection below the floor must be persisted with the
    number that missed, so the threshold can be calibrated from data."""
    pair = make_pair()
    rec = Recorder()
    cex = FakeCex({"ETH/USDT": flat_book(bid=1000, ask=1000)})
    dex = FakeDex(sell_price=1000.4, buy_price=1000.4)   # ~4 bps gross

    opps = await detector(cex, dex, [pair], rec, min_net_bps=D(50)).detect()

    assert not opps
    below = [r for r in rec.records if r.reason == RejectionReason.BELOW_FLOOR]
    assert below, f"expected a below_floor record, got {[r.reason for r in rec.records]}"
    assert below[0].net_bps is not None, "the near-miss edge must be recorded"
    assert below[0].min_net_bps == D(50), "the floor in force must be recorded"


async def test_rejection_reasons_are_specific_not_generic():
    """Each early exit needs its own reason, or the dataset cannot distinguish
    'no liquidity' from 'no feed' from 'edge too small'."""
    rec = Recorder()

    # no book at all
    await detector(FakeCex({}), FakeDex(1050, 1050), [make_pair()], rec).detect()
    assert rec.records[-1].reason == RejectionReason.NO_BOOK

    # book present, but nowhere near deep enough to fill $1000
    rec2 = Recorder()
    thin = ([(D(999), D("0.0001"))], [(D(1000), D("0.0001"))])
    await detector(FakeCex({"ETH/USDT": thin}), FakeDex(2000, 2000),
                   [make_pair()], rec2).detect()
    assert RejectionReason.INSUFFICIENT_DEPTH in {r.reason for r in rec2.records}

    # absurd edge -> bad data, not an opportunity
    rec3 = Recorder()
    await detector(FakeCex({"ETH/USDT": flat_book(1000, 1000)}),
                   FakeDex(10_000_000, 10_000_000), [make_pair()], rec3,
                   max_net_bps_sanity=D(1000)).detect()
    assert RejectionReason.ABOVE_SANITY in {r.reason for r in rec3.records}


async def test_records_carry_the_inputs_needed_to_re_derive_the_decision():
    pair = make_pair()
    rec = Recorder()
    cex = FakeCex({"ETH/USDT": flat_book(bid=1000, ask=1000)})
    dex = FakeDex(sell_price=1050, buy_price=1050, gas=D("0.5"))

    await detector(cex, dex, [pair], rec).detect()
    taken = next(r for r in rec.records if r.outcome == "taken")

    # every component of the arithmetic, so a third party can reproduce it
    for field in ("size_base", "notional_quote", "cex_price", "cex_best_bid",
                  "cex_best_ask", "dex_price", "gross_quote", "cex_fee_quote",
                  "gas_quote", "net_quote", "net_bps", "cex_legs",
                  "taker_fee_bps", "min_net_bps"):
        assert getattr(taken, field) is not None, f"{field} must be recorded"

    recomputed = taken.gross_quote - taken.cex_fee_quote - taken.gas_quote
    assert recomputed == taken.net_quote, "the row must be self-consistent"


async def test_top_of_book_is_recorded_alongside_the_vwap_used():
    """Makes the cost of depth-blindness measurable after the fact."""
    pair = make_pair()
    rec = Recorder()
    thin_asks = [(D(1000), D("0.01")), (D(1100), D(1000))]
    cex = FakeCex({"ETH/USDT": ([(D(999), D(1000))], thin_asks)})
    dex = FakeDex(sell_price=1200, buy_price=1200)

    await detector(cex, dex, [pair], rec).detect()
    c2d = next(r for r in rec.records if r.direction == "CEX_to_DEX")

    assert c2d.cex_best_ask == D(1000)
    assert c2d.cex_price > c2d.cex_best_ask, "VWAP must be worse than top of book"


async def test_a_failing_pair_still_produces_a_record():
    """Error isolation must not mean silent omission -- an exception is itself
    a data point, and a pair that always errors must be visible in the data."""
    good, bad = make_pair("ETH/USDT"), make_pair("BAD/USDT", base="BAD")
    rec = Recorder()

    class Exploding(FakeDex):
        async def get_quote(self, pair, size, side, estimate_gas=False):
            if pair.base == "BAD":
                raise RuntimeError("simulated RPC failure")
            return await super().get_quote(pair, size, side, estimate_gas)

    cex = FakeCex({"ETH/USDT": flat_book(1000, 1000), "BAD/USDT": flat_book(1000, 1000)})
    opps = await detector(cex, Exploding(1050, 1050), [bad, good], rec).detect()

    assert len(opps) == 1
    errors = [r for r in rec.records if r.reason == RejectionReason.ERROR]
    assert errors, "the failing pair must leave a record"
    assert errors[0].cex_symbol == "BAD/USDT"


async def test_the_detector_works_without_a_store():
    """Recording is optional, so the store cannot become a hard dependency of
    detection -- but when absent, nothing silently swallows the records."""
    pair = make_pair()
    cex = FakeCex({"ETH/USDT": flat_book(bid=1000, ask=1000)})
    opps = await detector(cex, FakeDex(1050, 1050), [pair], None).detect()
    assert len(opps) == 1


async def test_a_store_failure_does_not_stop_trading():
    """Telemetry must never be able to halt the strategy."""
    pair = make_pair()

    class Broken:
        def record(self, record):
            raise RuntimeError("disk full")

    cex = FakeCex({"ETH/USDT": flat_book(bid=1000, ask=1000)})
    opps = await detector(cex, FakeDex(1050, 1050), [pair], Broken()).detect()
    assert len(opps) == 1, "a broken store must not suppress opportunities"
