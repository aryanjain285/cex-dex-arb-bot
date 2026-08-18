"""Replay recorded observations through the PRODUCTION detector.

Two different things are being tested by two different tools, and conflating them is
how a backtest comes to prove nothing:

  The research stack (src/research) evaluates the MARKET. It computes size curves,
  distributions and latency costs from recorded state, using its own optimiser. It says
  whether an opportunity exists.

  This replays recorded state through the ACTUAL detector, router, risk manager and
  executor. It says whether the bot would have found and acted on one. A market
  conclusion drawn from the research stack tells you nothing about whether the shipped
  code path agrees, and the shipped code path is what would trade.

The existing CSV replay already drives the production components, which is right. What
it cannot do is vary size, because a row carries one scalar `dex_price` -- a quote taken
at one size, under one fee tier -- and it synthesises a one-level book at an assumed
depth. So every fill is an assumption the data cannot support.

Replaying the observation store fixes both, because the store holds re-quotable state:

    the DEX quote comes from the recorded pool snapshot through the local swap math,
    so a quote at any size is the same arithmetic the deployed QuoterV2 performs
    (verified: 44/44 exact at the recorded block)

    the CEX book is the recorded ladder, walked, rather than one synthesised level

    gas comes from the recorded gas price under an explicit limit, and an observation
    with no gas price is refused rather than treated as free

What this still cannot tell you: whether an order would have filled. No dataset of
quotes can. That limit belongs in the report, not in a footnote.
"""
from decimal import Decimal

import pytest

from src.exchange.pool_state import PoolSnapshot
from src.exchange.univ3_math import TickInfo
from src.research.observations import Observation, ObservationStore

from backtest.observation_replay import (
    ObservationReplayCex,
    ObservationReplayDex,
    build_market_pair,
)


def _pool(price=Decimal("1900"), liquidity=10 ** 24):
    from decimal import getcontext
    getcontext().prec = 60
    raw = price * (Decimal(10) ** 6) / (Decimal(10) ** 18)
    return PoolSnapshot(
        sqrt_price_x96=int(Decimal(2 ** 96) * raw.sqrt()),
        liquidity=liquidity, tick=0, fee=500, tick_spacing=10,
        ticks=[TickInfo(tick=-500000, liquidity_net=liquidity),
               TickInfo(tick=500000, liquidity_net=-liquidity)],
        decimals0=18, decimals1=6, block_number=1234,
        address="0x" + "ab" * 20, token0="0x" + "11" * 20,
        token1="0x" + "22" * 20, chain="ethereum",
        known_lower_tick=-500000, known_upper_tick=500000,
    )


def _obs(ts=0.0, cex=Decimal("1900"), dex=Decimal("1900"), gas_wei=10 ** 9,
         levels=5, level_size=Decimal("50")):
    spread = Decimal("0.00005")
    return Observation(
        ts=ts, cex_symbol="ETH/USDT", base="WETH", quote="USDT", chain="ethereum",
        pool_fee=500, pool_address="0x" + "ab" * 20,
        cex_bids=[(cex * (1 - spread) * (1 - Decimal("0.00001") * i), level_size)
                  for i in range(levels)],
        cex_asks=[(cex * (1 + spread) * (1 + Decimal("0.00001") * i), level_size)
                  for i in range(levels)],
        cex_feed_ts=ts, pool=_pool(dex),
        gas_price_wei=gas_wei, native_price_quote=cex,
        rpc_endpoint="test", run_id="t",
    )


class TestTheCexSideServesTheRecordedLadder:
    @pytest.mark.asyncio
    async def test_the_whole_ladder_is_served_not_just_the_touch(self):
        """The CSV path synthesises one level at an assumed depth, so every fill is an
        assumption. A recorded ladder is evidence."""
        observation = _obs(levels=5)
        pair = build_market_pair(observation)
        cex = ObservationReplayCex(pair)
        cex.set_observation(observation)
        book = await cex.get_book(pair)
        assert len(book.bids) == 5
        assert len(book.asks) == 5
        assert book.bids[0][0] > book.bids[1][0], "bids must descend"
        assert book.asks[0][0] < book.asks[1][0], "asks must ascend"

    @pytest.mark.asyncio
    async def test_the_book_is_stamped_with_replay_time_not_history(self):
        """The detector rejects a book whose feed is stale against the process clock.
        Historical stamps would make every book stale and the replay would find nothing
        -- reported as "no opportunities", which is a claim about the market."""
        from src.core import clock

        observation = _obs(ts=1_600_000_000.0)
        pair = build_market_pair(observation)
        cex = ObservationReplayCex(pair)
        cex.set_observation(observation)
        book = await cex.get_book(pair)
        assert abs(book.feed_timestamp - clock.now()) < 60
        # And the historical instant is still available, because the report prints it.
        assert cex.historical_ts == 1_600_000_000.0

    @pytest.mark.asyncio
    async def test_no_observation_means_no_book(self):
        pair = build_market_pair(_obs())
        cex = ObservationReplayCex(pair)
        assert await cex.get_book(pair) is None


class TestTheDexSidePricesTheRecordedPool:
    @pytest.mark.asyncio
    async def test_the_quote_varies_with_size(self):
        """The whole point. A scalar dex_price cannot do this, so the CSV replay can
        only ever answer one size."""
        observation = _obs()
        pair = build_market_pair(observation)
        dex = ObservationReplayDex(pair, gas_units=200_000)
        dex.set_observation(observation)

        small = await dex.get_quote(pair, Decimal("0.1"), "sell")
        large = await dex.get_quote(pair, Decimal("1000"), "sell")
        assert small is not None and large is not None
        assert large.price < small.price, (
            "selling more base must achieve a worse price; the pool quote is not "
            "responding to size"
        )

    @pytest.mark.asyncio
    async def test_gas_comes_from_the_recorded_gas_price(self):
        observation = _obs(gas_wei=10 ** 10)
        pair = build_market_pair(observation)
        dex = ObservationReplayDex(pair, gas_units=200_000)
        dex.set_observation(observation)
        quote = await dex.get_quote(pair, Decimal("1"), "sell")
        expected = observation.gas_quote(200_000)
        assert quote.gas_cost_quote == expected
        assert quote.gas_cost_quote > 0

    @pytest.mark.asyncio
    async def test_an_observation_without_gas_yields_no_quote(self):
        """Not a free one. A zero gas cost is the single easiest way to make this
        strategy look profitable, since its edge and its gas are the same size."""
        observation = _obs(gas_wei=None)
        pair = build_market_pair(observation)
        dex = ObservationReplayDex(pair, gas_units=200_000)
        dex.set_observation(observation)
        assert await dex.get_quote(pair, Decimal("1"), "sell") is None

    @pytest.mark.asyncio
    async def test_a_size_beyond_the_observed_window_yields_no_quote(self):
        """The pool snapshot refuses to price past the liquidity it observed, and that
        refusal has to survive into the replay -- otherwise the backtest fills sizes the
        research stack declines to quote."""
        observation = _obs()
        thin = Observation(**{**observation.__dict__,
                             "pool": _pool(liquidity=10 ** 12)})
        pair = build_market_pair(thin)
        dex = ObservationReplayDex(pair, gas_units=200_000)
        dex.set_observation(thin)
        assert await dex.get_quote(pair, Decimal("100000"), "sell") is None

    @pytest.mark.asyncio
    async def test_both_directions_are_priced(self):
        observation = _obs()
        pair = build_market_pair(observation)
        dex = ObservationReplayDex(pair, gas_units=200_000)
        dex.set_observation(observation)
        sell = await dex.get_quote(pair, Decimal("1"), "sell")
        buy = await dex.get_quote(pair, Decimal("1"), "buy")
        assert sell is not None and buy is not None
        assert buy.price > sell.price, (
            "buying base on the DEX must cost more than selling it does; the two "
            "directions are not distinguished"
        )


class TestTheMarketPairComesFromTheObservation:
    def test_identity_and_decimals_are_taken_from_the_record(self):
        observation = _obs()
        pair = build_market_pair(observation)
        assert pair.cex_symbol == "ETH/USDT"
        assert pair.base == "WETH"
        assert pair.quote_cex == "USDT"
        assert pair.dex_chain == "ethereum"
        assert pair.dex_pool_fee == 500

    def test_the_fee_tier_is_the_recorded_one_not_a_configured_one(self):
        """A replay against the wrong tier would price a different pool. The tier is a
        property of the observation, so it comes from there."""
        observation = _obs()
        rewritten = Observation(**{**observation.__dict__, "pool_fee": 3000})
        assert build_market_pair(rewritten).dex_pool_fee == 3000


class TestStoreDriven:
    @pytest.mark.asyncio
    async def test_replaying_a_store_evaluates_every_observation(self, tmp_path):
        from backtest.observation_replay import replay_store

        store = ObservationStore(tmp_path / "obs.sqlite3", run_id="replay")
        for i in range(10):
            store.record(_obs(ts=float(i)))

        result = await replay_store(store, gas_units=200_000, taker_fee_bps=Decimal("7.5"),
                                    target_notional=Decimal("1000"),
                                    min_net_bps=Decimal("5"))
        assert result["observations"] == 10
        assert result["evaluated"] == 10

    @pytest.mark.asyncio
    async def test_a_dislocated_market_produces_opportunities(self, tmp_path):
        from backtest.observation_replay import replay_store

        store = ObservationStore(tmp_path / "obs.sqlite3", run_id="replay")
        for i in range(10):
            store.record(_obs(ts=float(i), cex=Decimal("1900"), dex=Decimal("1960")))

        result = await replay_store(store, gas_units=200_000, taker_fee_bps=Decimal("7.5"),
                                    target_notional=Decimal("1000"),
                                    min_net_bps=Decimal("5"))
        assert result["opportunities"] > 0

    @pytest.mark.asyncio
    async def test_a_fair_market_produces_none(self, tmp_path):
        from backtest.observation_replay import replay_store

        store = ObservationStore(tmp_path / "obs.sqlite3", run_id="replay")
        for i in range(10):
            store.record(_obs(ts=float(i)))

        result = await replay_store(store, gas_units=200_000, taker_fee_bps=Decimal("7.5"),
                                    target_notional=Decimal("1000"),
                                    min_net_bps=Decimal("5"))
        assert result["opportunities"] == 0
        # And that is reported as a market statement rather than a failure to run.
        assert result["evaluated"] == 10

    @pytest.mark.asyncio
    async def test_uncostable_observations_are_counted_not_skipped(self, tmp_path):
        from backtest.observation_replay import replay_store

        store = ObservationStore(tmp_path / "obs.sqlite3", run_id="replay")
        for i in range(10):
            store.record(_obs(ts=float(i), gas_wei=10 ** 9 if i % 2 else None))

        result = await replay_store(store, gas_units=200_000, taker_fee_bps=Decimal("7.5"),
                                    target_notional=Decimal("1000"),
                                    min_net_bps=Decimal("5"))
        assert result["observations"] == 10
        assert result["uncostable"] == 5


class TestOneMarketPerReplay:
    """A market is (pair, chain, fee tier), not a pair.

    Filtering on the CEX symbol alone pools six pools -- three chains times two tiers --
    under a MarketPair built from whichever observation was earliest. Pricing stays
    correct, since the DEX side reads each observation's own pool, but the MarketPair the
    detector sees then describes a different pool from the state it is handed, and every
    per-market count is a mixture.

    Caught by a positive control rather than by inspection: with 60 bps injected into
    real recorded books, ETH/USDC found 148 opportunities and ETH/USDT found none. Both
    should have found roughly the same number, and the difference was entirely an
    artifact of which observation came first.
    """

    @pytest.mark.asyncio
    async def test_the_replay_can_be_restricted_to_one_chain_and_tier(self, tmp_path):
        from backtest.observation_replay import replay_store

        store = ObservationStore(tmp_path / "obs.sqlite3", run_id="replay")
        for i in range(6):
            store.record(_obs(ts=float(i)))
        for i in range(4):
            arb = Observation(**{**_obs(ts=float(100 + i)).__dict__,
                                 "chain": "arbitrum", "pool_fee": 3000})
            store.record(arb)

        both = await replay_store(
            store, gas_units=200_000, taker_fee_bps=Decimal("7.5"),
            target_notional=Decimal("1000"), min_net_bps=Decimal("5"),
        )
        assert both["observations"] == 10

        one = await replay_store(
            store, gas_units=200_000, taker_fee_bps=Decimal("7.5"),
            target_notional=Decimal("1000"), min_net_bps=Decimal("5"),
            chain="ethereum", pool_fee=500,
        )
        assert one["observations"] == 6, (
            "the replay must be restrictable to a single market, or its per-market "
            "counts mix pools"
        )

    @pytest.mark.asyncio
    async def test_filtering_by_tier_alone_separates_the_tiers(self, tmp_path):
        from backtest.observation_replay import replay_store

        store = ObservationStore(tmp_path / "obs.sqlite3", run_id="replay")
        for i in range(5):
            store.record(_obs(ts=float(i)))
        for i in range(3):
            store.record(Observation(**{**_obs(ts=float(50 + i)).__dict__,
                                        "pool_fee": 3000}))

        low = await replay_store(
            store, gas_units=200_000, taker_fee_bps=Decimal("7.5"),
            target_notional=Decimal("1000"), min_net_bps=Decimal("5"),
            pool_fee=500,
        )
        high = await replay_store(
            store, gas_units=200_000, taker_fee_bps=Decimal("7.5"),
            target_notional=Decimal("1000"), min_net_bps=Decimal("5"),
            pool_fee=3000,
        )
        assert low["observations"] == 5
        assert high["observations"] == 3
