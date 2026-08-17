"""The volume-spike screen: a dead code path with its own cost model.

Two defects, found by an AST sweep for constructor keywords that are not model
fields rather than by reading:

1. It could not run at all. `_evaluate_symbol` built
   `MarketPair(quote=..., symbol=...)`, and MarketPair has neither field -- they
   are `quote_cex`/`quote_dex` and `cex_symbol`. Pydantic raises on unexpected
   keywords, so the first pool it found killed the whole scan. Nothing caught it
   because this path had no tests and the CLI wrapper reports failures as text.

2. It carried a SECOND cost model. Edge was the raw spread minus a
   `cost_buffer_bps: 15.0` fudge, with no taker fee, no gas, and no depth. That
   is the "assembled threshold" model the detector rewrite deleted, and keeping a
   rival copy is worse than having no screen: it reports signals the detector
   would reject, and the first person to compare the two numbers concludes the
   detector is broken.

The screen now prices through `costs.evaluate_trade` -- the single place costs
are summed -- so its number is directly comparable to `min_net_bps`.

What it still cannot do is see depth: it prices from a single ticker, so it
overstates achievable edge on any size. That is inherent to a screen and is
recorded on every signal rather than hidden.
"""
from decimal import Decimal

import pytest

from src.core.config import load_config
from src.scanner.spike import SpikeArbitrageEvaluator, VolumeSpike


def D(x) -> Decimal:
    return Decimal(str(x))


class FakePublicClient:
    def __init__(self, price):
        self.price = price

    async def fetch_ticker_price(self, symbol):
        return self.price


class FakeDexClient:
    """Enough of UniV3DexClient for the screen: a token registry and a quote."""

    def __init__(self, price, chains=("ethereum",)):
        self.price = D(price)
        self.tokens_config = {
            "WETH": {c: object() for c in chains},
            "USDT": {c: object() for c in chains},
        }
        self.pairs_quoted = []

    async def get_pool_address(self, base, quote, chain, fee):
        return "0x" + "ab" * 20

    async def get_quote(self, pair, size, side="sell", estimate_gas=False):
        from src.core.types import DexQuote

        self.pairs_quoted.append(pair)
        return DexQuote(price=self.price, gas_cost_quote=D("1.50"))


def _spike(symbol="ETH/USDT", base="WETH", quote="USDT") -> VolumeSpike:
    import datetime

    return VolumeSpike(
        symbol=symbol, base=base, quote=quote,
        current_volume=1_000_000.0,
        previous_volume=100_000.0,
        ratio=10.0,
        closed_at=datetime.datetime(2026, 8, 17, tzinfo=datetime.timezone.utc),
        base_precision=4,
        quote_precision=2,
    )


def _evaluator():
    return SpikeArbitrageEvaluator(load_config())


async def test_the_screen_can_build_a_market_pair_at_all():
    """The regression guard for the ValidationError.

    A screen that raises on the first pool it finds is indistinguishable from a
    market with no opportunities.
    """
    evaluator = _evaluator()
    dex = FakeDexClient(price=1100)

    signal = await evaluator._evaluate_symbol(
        FakePublicClient(1000.0), dex, _spike()
    )

    assert dex.pairs_quoted, "no MarketPair was ever successfully constructed"
    pair = dex.pairs_quoted[0]
    assert pair.cex_symbol == "ETH/USDT"
    assert pair.quote_cex == "USDT"
    assert pair.quote_dex == "USDT"
    assert signal is not None, "a 1000 vs 1100 spread must screen positive"


async def test_the_screen_prices_through_the_shared_cost_model():
    """The number must be net of the taker fee and gas, so it is comparable to
    strategy.min_net_bps rather than to a fudge factor."""
    from src.strategy.costs import evaluate_trade

    config = load_config()
    evaluator = SpikeArbitrageEvaluator(config)
    dex = FakeDexClient(price=1100)

    signal = await evaluator._evaluate_symbol(
        FakePublicClient(1000.0), dex, _spike()
    )

    # CEX 1000 / DEX 1100 -> buy on the CEX, sell on the DEX.
    econ = evaluate_trade(
        direction="CEX_to_DEX",
        size_base=D(config.scanner.spike.arbitrage.probe_size_base),
        cex_price=D(1000),
        dex_price=D(1100),
        taker_fee_bps=config.strategy.taker_fee_bps,
        gas_quote=D("1.50"),
    )

    assert signal is not None
    assert signal.direction == "CEX_to_DEX"
    assert Decimal(str(signal.net_bps)) == pytest.approx(
        Decimal(str(econ.net_bps)), rel=Decimal("1e-9")
    ), "the screen must not have its own cost model"


async def test_gross_and_net_are_both_reported():
    """Keeping both makes the cost of trading visible on the signal itself,
    which is the number an operator needs to judge whether a screen hit is
    interesting."""
    signal = await _evaluator()._evaluate_symbol(
        FakePublicClient(1000.0), FakeDexClient(price=1100), _spike()
    )

    assert signal is not None
    assert signal.gross_bps > signal.net_bps, (
        "net must be strictly worse than gross once fees and gas are charged"
    )


async def test_a_spread_that_does_not_cover_costs_is_refused():
    """1 bps of gross spread cannot pay a 7.5 bps round trip plus gas."""
    signal = await _evaluator()._evaluate_symbol(
        FakePublicClient(1000.0), FakeDexClient(price=1000.1), _spike()
    )

    assert signal is None


async def test_the_signal_records_that_it_is_depth_blind():
    """The screen prices from a single ticker, so it overstates achievable edge
    on any real size. Recording that on the signal keeps a screen hit from being
    read as a tradeable opportunity."""
    signal = await _evaluator()._evaluate_symbol(
        FakePublicClient(1000.0), FakeDexClient(price=1100), _spike()
    )

    assert signal is not None
    payload = signal.as_dict()
    assert payload.get("depth_aware") is False
    assert "net_bps" in payload and "gross_bps" in payload


async def test_a_denied_token_is_not_screened():
    """The screen feeds a human's attention, and attention spent on a token that
    can never be traded is wasted. It also spends RPC calls."""
    evaluator = _evaluator()
    dex = FakeDexClient(price=1100)
    dex.tokens_config = {"LINGO": {"base": object()}, "USDT": {"base": object()}}

    signal = await evaluator._evaluate_symbol(
        FakePublicClient(1000.0), dex,
        _spike(symbol="LINGO/USDT", base="LINGO", quote="USDT"),
    )

    assert signal is None
    assert dex.pairs_quoted == [], "a denied token was quoted anyway"


async def test_no_cex_price_is_handled():
    signal = await _evaluator()._evaluate_symbol(
        FakePublicClient(None), FakeDexClient(price=1100), _spike()
    )
    assert signal is None
