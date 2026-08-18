"""Replay recorded observations through the PRODUCTION detector.

Two different tools answer two different questions, and conflating them is how a
backtest comes to prove nothing:

  `src/research` evaluates the MARKET. It computes size curves, distributions, latency
  costs and confidence intervals from recorded state, using its own optimiser. It says
  whether an opportunity existed.

  This module replays the same recorded state through the ACTUAL detector, router, risk
  manager and executor. It says whether the shipped code path would have found and acted
  on one. A market conclusion says nothing about whether the code agrees, and the code is
  what would trade.

WHAT THIS FIXES ABOUT THE CSV REPLAY. `backtest/simulator.py` already drives the
production components, which is the right shape. What it cannot do is vary size: a CSV
row carries one scalar `dex_price`, a quote taken at one size under one fee tier, and it
synthesises a one-level book at an assumed depth. Every fill is therefore an assumption
the data cannot support, and the answer to "would $5,000 have worked" does not exist in
the file.

The observation store holds re-quotable state, so:

  the DEX quote is computed from the recorded pool snapshot through the local swap math,
  which reproduces the deployed QuoterV2 exactly at the recorded block (verified 44/44
  across 22 markets), at ANY size;

  the CEX book is the recorded ladder, walked by the production cost code, rather than
  one synthesised level;

  gas comes from the recorded gas PRICE under an explicit limit, so the limit is a stated
  assumption, and an observation with no gas price is refused rather than treated as free.

WHAT IT STILL CANNOT TELL YOU. Whether an order would have filled. No dataset of quotes
can: a quote is what the venue offered, not what it would have honoured to a taker
arriving a moment later with size. That limit belongs in the report rather than a
footnote, and `resolve_with_latency` in the research stack is the closest available
answer -- it re-prices a frozen decision against a later observation.
"""
from __future__ import annotations

from decimal import Decimal
from typing import Any, Dict, Optional

from loguru import logger

from src.core import clock
from src.core.types import BookSnapshot, DexQuote, MarketPair
from src.research.observations import Observation, ObservationStore

__all__ = [
    "ObservationReplayCex",
    "ObservationReplayDex",
    "build_market_pair",
    "replay_store",
]


def build_market_pair(observation: Observation) -> MarketPair:
    """A MarketPair describing exactly what was recorded.

    Identity, chain, fee tier and decimals all come from the observation rather than
    from config. A replay against a configured fee tier would price a different pool
    than the one whose state is in the row, and nothing about the output would look
    wrong.
    """
    pool = observation.pool
    base_is_token0 = _base_is_token0(observation)
    return MarketPair(
        base=observation.base,
        quote_cex=observation.quote,
        quote_dex=observation.quote,
        cex_symbol=observation.cex_symbol,
        dex_chain=observation.chain,
        dex_pool_fee=int(observation.pool_fee),
        base_address=pool.token0 if base_is_token0 else pool.token1,
        quote_address=pool.token1 if base_is_token0 else pool.token0,
        base_decimals=pool.decimals0 if base_is_token0 else pool.decimals1,
        quote_decimals=pool.decimals1 if base_is_token0 else pool.decimals0,
    )


def _base_is_token0(observation: Observation) -> bool:
    """Which side of the pool the base sits on, from the recorded price.

    Derived rather than configured. The pool's spot price is token0-in-token1; the CEX
    mid is quote-per-base. Whichever orientation puts the two within an order of
    magnitude of each other is the right one, and the wrong one is out by price squared
    -- which for any real pair is several orders of magnitude, so the test is not close.
    """
    mid = observation.cex_mid
    spot = observation.pool.spot_price()
    if mid is None or spot is None or spot <= 0 or mid <= 0:
        return True
    direct = abs((spot / mid) - 1)
    inverted = abs(((Decimal(1) / spot) / mid) - 1)
    return direct <= inverted


class ObservationReplayCex:
    """Serves the recorded CEX ladder.

    Deliberately not a subclass of the CEX client base: that declares an
    order-placement surface this object has no business implementing, and inheriting it
    once hid the fact that the method the detector actually calls -- `get_book` -- was
    missing entirely.
    """

    def __init__(self, pair: MarketPair):
        self.pair = pair
        self._observation: Optional[Observation] = None
        # The historical instant, kept separately from the replay stamp below.
        self.historical_ts: Optional[float] = None

    def set_observation(self, observation: Observation) -> None:
        self._observation = observation
        self.historical_ts = observation.ts

    async def connect(self) -> None:
        pass

    async def close(self) -> None:
        pass

    async def get_book(self, pair: MarketPair) -> Optional[BookSnapshot]:
        if self._observation is None:
            return None
        observation = self._observation
        if not observation.cex_bids or not observation.cex_asks:
            return None
        # Both stamps are REPLAY time. The detector rejects a book whose feed is stale
        # against the process clock, so historical stamps would make every book stale
        # and the run would report zero opportunities -- a claim about the market rather
        # than about the harness. The historical instant is carried on the object.
        now = clock.now()
        return BookSnapshot(
            pair=pair,
            bids=list(observation.cex_bids),
            asks=list(observation.cex_asks),
            timestamp=now,
            feed_timestamp=now,
        )

    async def create_order(self, order) -> Any:
        raise NotImplementedError(
            "a replay does not place orders; use the research stack's "
            "resolve_with_latency to ask what a decision would have been worth"
        )


class ObservationReplayDex:
    """Serves a quote computed from the recorded pool snapshot, at any size."""

    def __init__(self, pair: MarketPair, gas_units: int):
        if gas_units <= 0:
            raise ValueError(
                f"gas_units must be positive, got {gas_units}; a zero limit is a zero "
                f"gas cost by another name, and gas is the same order of magnitude as "
                f"the edge being measured"
            )
        self.pair = pair
        self.gas_units = gas_units
        self._observation: Optional[Observation] = None

    def set_observation(self, observation: Observation) -> None:
        self._observation = observation

    async def get_quote(
        self, pair: MarketPair, size: Decimal, side: str,
        estimate_gas: bool = False,
    ) -> Optional[DexQuote]:
        """Effective quote-per-base for one DEX leg.

        `size` MEANS DIFFERENT THINGS BY SIDE, and this is the production convention
        rather than a choice made here -- see detector._evaluate_cex_to_dex and
        _evaluate_dex_to_cex:

            side="sell"   `size` is a BASE amount.  The leg spends base for quote.
            side="buy"    `size` is a QUOTE amount. The leg spends quote for base.

        A first version of this adapter treated `size` as a base amount in both
        branches and converted it to a notional internally for the buy leg. The
        detector already passes the notional, so the conversion happened twice and
        every buy-side swap was priced roughly 1,900x too large on an ETH pair. It
        surfaced as no_dex_quote and below_floor on nine of twenty-two markets in a
        positive control with 200 bps injected -- too large for the arithmetic to
        explain, which is the only reason it was found. The same defect class,
        collapsing the two legs units, once produced a 36-billion-bps reading in the
        optimiser.

        Its unit test passed throughout, because it asserted only that a buy quote
        exceeds a sell quote, and that holds whichever units are used.
        """
        if self._observation is None:
            return None
        observation = self._observation
        gas_quote = observation.gas_quote(self.gas_units)
        if gas_quote is None:
            # Not zero. See the module docstring.
            return None
        if size <= 0:
            return None

        base_is_token0 = _base_is_token0(observation)
        pool = observation.pool

        if side == "sell":
            # Base in, quote out. price_for_amount_in returns out-per-in, which is
            # already quote-per-base.
            rate = pool.price_for_amount_in(size, zero_for_one=base_is_token0)
            if rate is None or rate <= 0:
                return None
            price = rate
        else:
            # Quote in, base out. `size` IS the quote amount -- no conversion. The
            # rate returned is base-per-quote here, so quote-per-base is its
            # reciprocal.
            rate = pool.price_for_amount_in(size, zero_for_one=not base_is_token0)
            if rate is None or rate <= 0:
                return None
            price = Decimal(1) / rate

        return DexQuote(price=price, gas_cost_quote=gas_quote)

    async def execute_swap(self, params) -> Any:
        raise NotImplementedError(
            "a replay does not execute swaps; a recorded quote is what the venue "
            "offered, not what it would have honoured to a taker arriving with size"
        )


async def replay_store(
    store: ObservationStore,
    *,
    gas_units: int,
    taker_fee_bps: Decimal,
    target_notional: Decimal,
    min_net_bps: Decimal,
    cex_symbol: Optional[str] = None,
    chain: Optional[str] = None,
    pool_fee: Optional[int] = None,
    limit: Optional[int] = None,
) -> Dict[str, Any]:
    """Walk one MARKET through the production detector, observation by observation.

    A market is (pair, chain, fee tier), not a pair. Filtering on the CEX symbol alone
    pools six different pools -- three chains times two tiers -- under a single
    MarketPair built from whichever observation happened to be earliest. The pricing
    stays correct, because the DEX side reads each observation's own pool, but the
    MarketPair the detector sees describes a different pool from the state it is being
    given, and every per-market count is then a mixture.

    Caught by a positive control: with 60 bps injected into real recorded books, ETH/USDC
    and ARB/USDC found opportunities while ETH/USDT and ARB/USDT found none at all. Not a
    market fact -- an artifact of the earliest observation for those symbols coming from a
    pool whose configuration disagreed with the rest.

    Returns counts rather than a PnL. A count of opportunities is a joint statement about
    the code and the market; a PnL would additionally need a fill assumption, and a
    dataset of quotes contains no evidence for one.
    """
    from src.core.config import StrategyConfig
    from src.strategy.detector import OpportunityDetector

    observations = [
        o for o in store.read_all(cex_symbol=cex_symbol, limit=limit)
        if (chain is None or o.chain == chain)
        and (pool_fee is None or int(o.pool_fee) == int(pool_fee))
    ]
    result = {
        "observations": len(observations),
        "evaluated": 0,
        "uncostable": 0,
        "opportunities": 0,
        "by_direction": {},
        "best_net_bps": None,
    }
    if not observations:
        return result

    pair = build_market_pair(observations[0])
    cex = ObservationReplayCex(pair)
    dex = ObservationReplayDex(pair, gas_units=gas_units)

    strategy = StrategyConfig(
        target_notional_usd=int(target_notional),
        min_net_bps=min_net_bps,
        max_net_bps_sanity=Decimal("1000"),
        taker_fee_bps=taker_fee_bps,
        opportunity_ttl_seconds=30,
        loop_interval_seconds=1.0,
        intermediate_price_cache_seconds=5.0,
        max_book_age_seconds=30.0,
        error_backoff_seconds=1.0,
        shutdown_drain_seconds=1.0,
        max_consecutive_errors=10,
    )
    detector = OpportunityDetector(strategy, cex, dex, [pair])

    for observation in observations:
        if observation.gas_quote(gas_units) is None:
            result["uncostable"] += 1
            continue
        cex.set_observation(observation)
        dex.set_observation(observation)
        result["evaluated"] += 1
        try:
            opportunities = await detector.detect()
        except Exception as exc:  # noqa: BLE001 - a replay must survive one bad row
            logger.debug(f"replay: detector raised on ts={observation.ts}: {exc}")
            continue
        for opportunity in opportunities:
            result["opportunities"] += 1
            direction = getattr(opportunity, "direction", "unknown")
            result["by_direction"][direction] = (
                result["by_direction"].get(direction, 0) + 1
            )
            edge = getattr(opportunity, "edge_bps", None)
            if edge is not None and (
                result["best_net_bps"] is None or edge > result["best_net_bps"]
            ):
                result["best_net_bps"] = edge

    return result
