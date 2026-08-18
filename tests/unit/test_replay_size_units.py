"""`size` means different things by side, and getting it wrong is invisible.

The production convention, from `detector._evaluate_cex_to_dex` and
`_evaluate_dex_to_cex`:

    side="sell"   a BASE amount   -- the leg spends base for quote
    side="buy"    a QUOTE amount  -- the leg spends quote for base, and the detector
                                     passes the target notional directly

A first version of the replay adapter treated `size` as a base amount in both branches
and converted it to a notional internally for the buy leg. Since the detector already
passes the notional, the conversion happened twice and every buy-side swap was priced
roughly 1,900x too large on an ETH pair.

It survived its own unit tests, which asserted only that a buy quote exceeds a sell
quote -- true whichever units are used. What found it was a positive control: 200 bps
injected into real recorded books produced zero opportunities on nine of twenty-two
markets, with the detector reporting `no_dex_quote` and `below_floor`. At 200 bps
against a 5 bps floor the arithmetic could not explain that, which is the only reason
anyone looked.

This is the third appearance of the same defect class in this project -- the two DEX
legs consume different tokens, and any code that treats them alike produces a price
wrong by a factor of the price itself. Hence tests that pin the units rather than an
ordering.
"""
from decimal import Decimal, getcontext

import pytest

from src.exchange.pool_state import PoolSnapshot
from src.exchange.univ3_math import TickInfo
from src.research.observations import Observation

from backtest.observation_replay import ObservationReplayDex, build_market_pair

PRICE = Decimal("1900")


def _pool(price=PRICE, liquidity=10 ** 25, base_is_token0=True):
    getcontext().prec = 60
    if base_is_token0:
        decimals0, decimals1 = 18, 6
        raw = price * (Decimal(10) ** decimals1) / (Decimal(10) ** decimals0)
        token0, token1 = "0x" + "11" * 20, "0x" + "22" * 20
    else:
        decimals0, decimals1 = 6, 18
        raw = (Decimal(1) / price) * (Decimal(10) ** decimals1) / (Decimal(10) ** decimals0)
        token0, token1 = "0x" + "22" * 20, "0x" + "11" * 20
    return PoolSnapshot(
        sqrt_price_x96=int(Decimal(2 ** 96) * raw.sqrt()),
        liquidity=liquidity, tick=0, fee=500, tick_spacing=10,
        ticks=[TickInfo(tick=-600000, liquidity_net=liquidity),
               TickInfo(tick=600000, liquidity_net=-liquidity)],
        decimals0=decimals0, decimals1=decimals1, block_number=1,
        address="0x" + "ab" * 20, token0=token0, token1=token1,
        chain="ethereum", known_lower_tick=-600000, known_upper_tick=600000,
    )


def _obs(base_is_token0=True):
    spread = Decimal("0.00005")
    return Observation(
        ts=0.0, cex_symbol="ETH/USDT", base="WETH", quote="USDT", chain="ethereum",
        pool_fee=500, pool_address="0x" + "ab" * 20,
        cex_bids=[(PRICE * (1 - spread), Decimal("1000"))],
        cex_asks=[(PRICE * (1 + spread), Decimal("1000"))],
        cex_feed_ts=0.0, pool=_pool(base_is_token0=base_is_token0),
        gas_price_wei=10 ** 9, native_price_quote=PRICE,
    )


def _dex(observation):
    pair = build_market_pair(observation)
    dex = ObservationReplayDex(pair, gas_units=200_000)
    dex.set_observation(observation)
    return pair, dex


@pytest.mark.asyncio
async def test_a_buy_size_is_a_quote_amount():
    observation = _obs()
    pair, dex = _dex(observation)
    quote = await dex.get_quote(pair, Decimal("1000"), "buy")
    assert quote is not None
    assert Decimal("1890") < quote.price < Decimal("1912"), (
        f"spending 1,000 quote priced at {quote.price} per base; under the old "
        f"double conversion this would be the price of a 1.9m swap"
    )


@pytest.mark.asyncio
async def test_a_sell_size_is_a_base_amount():
    observation = _obs()
    pair, dex = _dex(observation)
    quote = await dex.get_quote(pair, Decimal("0.5"), "sell")
    assert quote is not None
    assert Decimal("1888") < quote.price < Decimal("1900")


@pytest.mark.asyncio
async def test_the_same_economic_size_brackets_spot_tightly():
    """The check that cannot pass under mismatched units. 0.5 base and 950 quote are the
    same trade at 1,900, so the two quotes must sit either side of spot and close to it.
    Doubling the conversion on one leg moves it far away."""
    observation = _obs()
    pair, dex = _dex(observation)
    sell = await dex.get_quote(pair, Decimal("0.5"), "sell")
    buy = await dex.get_quote(pair, Decimal("950"), "buy")
    assert sell is not None and buy is not None
    assert sell.price < buy.price
    assert (buy.price - sell.price) / sell.price < Decimal("0.01"), (
        f"sell {sell.price} and buy {buy.price} differ by more than 1% on the same "
        f"economic size; one leg is priced in the wrong units"
    )


@pytest.mark.asyncio
async def test_the_units_hold_with_the_base_as_token1():
    """Half of all pools order the tokens the other way, and the buy leg's
    `zero_for_one` flips with it."""
    observation = _obs(base_is_token0=False)
    pair, dex = _dex(observation)
    sell = await dex.get_quote(pair, Decimal("0.5"), "sell")
    buy = await dex.get_quote(pair, Decimal("950"), "buy")
    assert sell is not None and buy is not None
    assert Decimal("1888") < sell.price < Decimal("1900")
    assert Decimal("1890") < buy.price < Decimal("1912")


@pytest.mark.asyncio
async def test_a_larger_buy_notional_prices_worse():
    observation = _obs()
    pair, dex = _dex(observation)
    small = await dex.get_quote(pair, Decimal("1000"), "buy")
    large = await dex.get_quote(pair, Decimal("5000000"), "buy")
    assert small is not None and large is not None
    assert large.price > small.price, "buying more base must cost more per base"


@pytest.mark.asyncio
async def test_a_round_trip_at_matched_sizes_costs_about_two_pool_fees():
    """The strongest form: sell base, then buy it back with the proceeds. The pair of
    quotes must differ by about twice the pool fee -- 10 bps at this tier -- and nothing
    else. A units error on either leg moves this by orders of magnitude.
    """
    observation = _obs()
    pair, dex = _dex(observation)
    size_base = Decimal("0.5")
    sell = await dex.get_quote(pair, size_base, "sell")
    assert sell is not None
    proceeds = sell.price * size_base
    buy = await dex.get_quote(pair, proceeds, "buy")
    assert buy is not None
    round_trip_bps = (buy.price - sell.price) / sell.price * Decimal(10000)
    assert Decimal("8") < round_trip_bps < Decimal("14"), (
        f"round trip cost {round_trip_bps} bps against about 10 for two 5 bps fees"
    )
