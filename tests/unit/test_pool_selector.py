"""Which Uniswap v3 fee tier to quote.

Each pair in pairs.yaml names ONE `dex_pool_fee` and the detector quotes only
that pool. Uniswap v3 lists the same asset pair at up to four fee tiers with
independent liquidity, so a static choice is a standing bet that one tier is
always best. Measured live on 2026-08-17 at a 1000 notional:

    ETH/USDT  configured tier 500 -> 1892.49    tier 100 -> 1893.49   (5.3 bps)
    ETH/USDC  configured tier 500 -> 1891.05    tier 100 -> 1891.74   (3.7 bps)

Against a 5 bps net floor, the tier choice alone is larger than the edge the
strategy is trying to capture. This is not a refinement; it is the difference
between trading and not trading.

Two constraints shape the design:

* QuoterV2's price is already net of the pool fee AND of price impact for the
  size quoted, so comparing tiers at the intended size compares exactly the right
  thing -- the executable price. No fee arithmetic is needed or wanted here.
* Quoting four tiers on both sides of three pairs every 200ms would be 120 RPC
  calls a second. So selection is refreshed on a TTL and the hot loop quotes only
  the chosen tier.
"""
from decimal import Decimal

import pytest

from src.core.types import DexQuote
from src.exchange.pool_selector import PoolSelector
from tests.fakes import make_pair


def D(x) -> Decimal:
    return Decimal(str(x))


class FakeClock:
    def __init__(self, t=1000.0):
        self.t = t

    def __call__(self):
        return self.t

    def advance(self, seconds):
        self.t += seconds


class FakeDex:
    """Prices keyed by fee tier, and a record of every call made.

    `pools` lists which tiers exist, so an absent tier can be distinguished from
    a tier that quotes badly -- they need different handling and the difference
    is easy to conflate.
    """

    def __init__(self, prices_by_fee, pools=None):
        self.prices_by_fee = {int(k): D(v) for k, v in prices_by_fee.items()}
        self.pools = set(pools) if pools is not None else set(self.prices_by_fee)
        self.quote_calls = []
        self.pool_calls = []

    async def get_pool_address(self, base, quote, chain, fee):
        self.pool_calls.append(fee)
        return ("0x" + f"{fee:040x}") if fee in self.pools else None

    async def get_quote(self, pair, size, side, estimate_gas=False):
        self.quote_calls.append((pair.dex_pool_fee, side, size))
        price = self.prices_by_fee.get(pair.dex_pool_fee)
        if price is None:
            return None
        # A buyer spends quote units for base, so a LOWER number is better on the
        # buy side; the selector must not simply maximise.
        return DexQuote(price=price, gas_cost_quote=D(0))


TIERS = [100, 500, 3000, 10000]


# --- selection ------------------------------------------------------------


async def test_the_best_selling_tier_is_the_highest_price():
    """Selling base for quote: more quote received is better."""
    dex = FakeDex({100: 1893.49, 500: 1892.49, 3000: 1891.38, 10000: 1857.54})
    selector = PoolSelector(dex, TIERS, refresh_seconds=300)

    fee = await selector.best_fee(make_pair(), side="sell", size=D("0.5"))

    assert fee == 100


async def test_the_best_buying_tier_is_the_lowest_price():
    """Buying base with quote: paying less per base is better. A selector that
    maximised unconditionally would pick the worst pool on this side, and the
    error would be invisible -- it would just look like a thin market."""
    dex = FakeDex({100: 1894.23, 500: 1894.45, 3000: 1902.79, 10000: 1928.22})
    selector = PoolSelector(dex, TIERS, refresh_seconds=300)

    fee = await selector.best_fee(make_pair(), side="buy", size=D(1000))

    assert fee == 100


async def test_the_real_measured_numbers_pick_the_measured_winner():
    """The live ETH/USDT observation, as a regression fixture."""
    dex = FakeDex({100: 1893.48542322, 500: 1892.49079867,
                   3000: 1891.38264538, 10000: 1857.53660176})
    selector = PoolSelector(dex, TIERS, refresh_seconds=300)

    assert await selector.best_fee(make_pair(), "sell", D("0.5")) == 100


async def test_a_tier_with_no_pool_is_skipped():
    dex = FakeDex({500: 1892.49, 3000: 1891.38}, pools=[500, 3000])
    selector = PoolSelector(dex, TIERS, refresh_seconds=300)

    fee = await selector.best_fee(make_pair(), "sell", D("0.5"))

    assert fee == 500


async def test_a_tier_that_cannot_quote_is_skipped():
    """A pool can exist and still fail to quote -- zero liquidity, or a revert."""
    dex = FakeDex({100: 0, 500: 1892.49}, pools=[100, 500])
    selector = PoolSelector(dex, TIERS, refresh_seconds=300)

    assert await selector.best_fee(make_pair(), "sell", D("0.5")) == 500


async def test_no_quotable_tier_falls_back_to_the_configured_one():
    """Falling back rather than raising: the configured tier is the operator's
    stated intent, and a selector that cannot improve on it must not prevent
    trading. The detector's own no-quote handling then applies."""
    dex = FakeDex({}, pools=[])
    pair = make_pair(dex_pool_fee=3000)
    selector = PoolSelector(dex, TIERS, refresh_seconds=300)

    assert await selector.best_fee(pair, "sell", D("0.5")) == 3000


# --- cost control ---------------------------------------------------------

async def test_the_selection_is_cached_within_the_ttl():
    """Quoting four tiers on every cycle would be 120 RPC calls a second."""
    clock = FakeClock()
    dex = FakeDex({100: 1893.49, 500: 1892.49})
    selector = PoolSelector(dex, TIERS, refresh_seconds=300, now_fn=clock)
    pair = make_pair()

    await selector.best_fee(pair, "sell", D("0.5"))
    calls_after_first = len(dex.quote_calls)

    for _ in range(10):
        await selector.best_fee(pair, "sell", D("0.5"))

    assert len(dex.quote_calls) == calls_after_first, (
        "the selector re-quoted every tier inside its own TTL"
    )


async def test_the_selection_is_refreshed_after_the_ttl():
    """Liquidity migrates between tiers, so a selection made once and kept
    forever is the static choice with extra steps."""
    clock = FakeClock()
    dex = FakeDex({100: 1893.49, 500: 1892.49})
    selector = PoolSelector(dex, TIERS, refresh_seconds=300, now_fn=clock)
    pair = make_pair()

    assert await selector.best_fee(pair, "sell", D("0.5")) == 100

    # Liquidity moves: the 500 tier is now better.
    dex.prices_by_fee = {100: D("1880"), 500: D("1893")}
    clock.advance(301)

    assert await selector.best_fee(pair, "sell", D("0.5")) == 500


async def test_sides_are_selected_independently():
    """The best pool to buy in is not necessarily the best pool to sell in, and
    the live measurements show exactly that asymmetry."""
    clock = FakeClock()
    dex = FakeDex({100: 1893.49, 500: 1892.49})
    selector = PoolSelector(dex, TIERS, refresh_seconds=300, now_fn=clock)
    pair = make_pair()

    sell = await selector.best_fee(pair, "sell", D("0.5"))
    buy = await selector.best_fee(pair, "buy", D(1000))

    assert sell == 100      # highest price when selling
    assert buy == 500       # lowest price when buying


async def test_pairs_are_selected_independently():
    clock = FakeClock()
    dex = FakeDex({100: 1893.49, 500: 1892.49})
    selector = PoolSelector(dex, TIERS, refresh_seconds=300, now_fn=clock)

    await selector.best_fee(make_pair("ETH/USDT"), "sell", D("0.5"))
    before = len(dex.quote_calls)
    await selector.best_fee(make_pair("ARB/USDT", base="ARB"), "sell", D("0.5"))

    assert len(dex.quote_calls) > before, (
        "a second pair reused the first pair's selection"
    )


async def test_the_selection_is_reported_so_it_can_be_audited():
    """A choice the operator cannot see is a choice they cannot check."""
    clock = FakeClock()
    dex = FakeDex({100: 1893.49, 500: 1892.49})
    selector = PoolSelector(dex, TIERS, refresh_seconds=300, now_fn=clock)
    pair = make_pair()

    await selector.best_fee(pair, "sell", D("0.5"))

    snapshot = selector.describe()
    assert "ETH/USDT" in snapshot
    assert "100" in snapshot


async def test_a_single_candidate_tier_costs_no_extra_calls():
    """With one candidate there is nothing to compare, so the selector must not
    spend a quote to discover that."""
    dex = FakeDex({500: 1892.49})
    selector = PoolSelector(dex, [500], refresh_seconds=300)

    fee = await selector.best_fee(make_pair(dex_pool_fee=500), "sell", D("0.5"))

    assert fee == 500
    assert dex.quote_calls == [], "the selector quoted a tier it had no choice about"


async def test_an_empty_candidate_list_is_rejected():
    with pytest.raises(ValueError):
        PoolSelector(FakeDex({}), [], refresh_seconds=300)


async def test_a_non_positive_refresh_interval_is_rejected():
    """Zero would re-quote every tier on every cycle."""
    with pytest.raises(ValueError):
        PoolSelector(FakeDex({}), TIERS, refresh_seconds=0)


async def test_a_selector_failure_falls_back_rather_than_breaking_detection():
    """Selection is an optimisation. If it raises, the configured tier is used and
    trading continues -- an improvement that can stop the bot is not one."""
    class Broken:
        async def get_pool_address(self, *a, **k):
            raise RuntimeError("rpc down")

        async def get_quote(self, *a, **k):
            raise RuntimeError("rpc down")

    selector = PoolSelector(Broken(), TIERS, refresh_seconds=300)
    pair = make_pair(dex_pool_fee=3000)

    assert await selector.best_fee(pair, "sell", D("0.5")) == 3000
