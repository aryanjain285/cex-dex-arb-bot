"""Evaluating recorded observations, without peeking at the future.

Two properties define whether this module is worth anything.

NO LOOK-AHEAD. A decision at time t must use only the observation at t. The
temptation is structural rather than careless: the natural way to write a latency
model is to fetch the later observation and compute the best trade "available"
then, which silently re-optimises size and direction with information the bot could
not have had. That does not produce a slightly optimistic backtest; it produces a
strategy that trades only when it already knows the outcome, and it will show
profit on any data at all, including noise. So size and direction are frozen at t
and only prices advance.

LATENCY IS PRICED, NOT ASSUMED AWAY. The existing simulator fills at the recorded
touch, instantly and completely. For a strategy whose entire edge is a few basis
points and whose measured detection loop takes 2.32 seconds against 12-second
blocks, instantaneous execution is not a small simplification -- it removes the
dominant cost. An unresolvable trade (no successor observation close enough to the
intended delay) must be reported as UNRESOLVED, never as filled: dropping those
rows would select precisely the periods when the recorder kept up, which correlate
with calm markets.
"""
from decimal import Decimal

import pytest

from src.exchange.pool_state import PoolSnapshot
from src.exchange.univ3_math import TickInfo, sqrt_price_x96_from_tick
from src.research.evaluate import (
    CostModel,
    evaluate_observation,
    resolve_with_latency,
)
from src.research.observations import Observation


def _pool(price=Decimal("1900"), decimals0=18, decimals1=6) -> PoolSnapshot:
    from decimal import getcontext
    getcontext().prec = 60
    raw = price * (Decimal(10) ** decimals1) / (Decimal(10) ** decimals0)
    sqrt_price = int(Decimal(2 ** 96) * raw.sqrt())
    liquidity = 10 ** 24
    return PoolSnapshot(
        sqrt_price_x96=sqrt_price,
        liquidity=liquidity,
        tick=0,
        fee=500,
        tick_spacing=10,
        ticks=[TickInfo(tick=-500000, liquidity_net=liquidity),
               TickInfo(tick=500000, liquidity_net=-liquidity)],
        decimals0=decimals0,
        decimals1=decimals1,
        block_number=1,
        address="0x" + "ab" * 20,
        token0="0x" + "11" * 20,   # base is token0
        token1="0x" + "22" * 20,
        chain="ethereum",
        known_lower_tick=-500000,
        known_upper_tick=500000,
    )


def _obs(ts=0.0, cex=Decimal("1900"), dex=Decimal("1900"), spread=Decimal("0.0001")):
    """A one-instant observation. `cex` is the mid; the ladder is deep."""
    bid = cex * (1 - spread)
    ask = cex * (1 + spread)
    return Observation(
        ts=ts,
        cex_symbol="ETHUSDT", base="ETH", quote="USDT", chain="ethereum",
        pool_fee=500, pool_address="0x" + "ab" * 20,
        cex_bids=[(bid * (1 - Decimal("0.00001") * i), Decimal("1000"))
                  for i in range(5)],
        cex_asks=[(ask * (1 + Decimal("0.00001") * i), Decimal("1000"))
                  for i in range(5)],
        cex_feed_ts=ts,
        pool=_pool(dex),
        gas_price_wei=10 ** 9,
        native_price_quote=cex,
        rpc_endpoint="test",
        run_id="test",
    )


COSTS = CostModel(
    taker_fee_bps=Decimal("7.5"),
    cex_legs=1,
    gas_units=200_000,
    rotation_cost_quote=Decimal("0"),
    floor_bps=Decimal("5"),
)

NOTIONALS = [Decimal("100"), Decimal("1000"), Decimal("10000")]


class TestEvaluatingOneObservation:
    def test_a_fair_market_shows_no_tradeable_edge(self):
        result = evaluate_observation(_obs(), COSTS, NOTIONALS, base_is_token0=True)
        assert result.best is None
        # Gross is still reported: it is the research signal and survives every
        # cost assumption.
        assert result.best_gross_bps is not None

    def test_a_dislocated_market_shows_the_right_direction(self):
        """DEX above CEX: buy on the CEX, sell on the DEX -- CEX_to_DEX."""
        result = evaluate_observation(
            _obs(cex=Decimal("1900"), dex=Decimal("1920")),
            COSTS, NOTIONALS, base_is_token0=True,
        )
        assert result.best is not None
        assert result.best.direction == "CEX_to_DEX"
        assert result.best.net_bps > COSTS.floor_bps

    def test_the_other_direction_is_found_too(self):
        result = evaluate_observation(
            _obs(cex=Decimal("1920"), dex=Decimal("1900")),
            COSTS, NOTIONALS, base_is_token0=True,
        )
        assert result.best is not None
        assert result.best.direction == "DEX_to_CEX"

    def test_both_curves_are_kept_not_just_the_winner(self):
        """The losing direction's curve is data: a market that is always -6 bps one
        way and -4 the other is a fee story, and only both curves show that."""
        result = evaluate_observation(_obs(), COSTS, NOTIONALS, base_is_token0=True)
        assert set(result.curves) == {"CEX_to_DEX", "DEX_to_CEX"}
        for curve in result.curves.values():
            assert len(curve.curve) == len(NOTIONALS)

    def test_the_fixed_probe_is_reported_alongside_the_optimum(self):
        """The reviewer's claim, made measurable on every observation: what a single
        fixed size would have concluded, next to what the best size achieves."""
        result = evaluate_observation(
            _obs(cex=Decimal("1900"), dex=Decimal("1920")),
            COSTS, NOTIONALS, base_is_token0=True,
            probe_notional=Decimal("1000"),
        )
        assert result.probe_net_bps is not None
        assert result.best is not None
        assert result.best.net_bps >= result.probe_net_bps

    def test_gas_absent_means_the_observation_cannot_be_costed(self):
        """A missing gas price must not silently become a zero gas cost."""
        obs = _obs()
        obs = Observation(**{**obs.__dict__, "gas_price_wei": None})
        result = evaluate_observation(obs, COSTS, NOTIONALS, base_is_token0=True)
        assert result.best is None
        assert result.reason is not None and "gas" in result.reason.lower()

    def test_the_cost_model_is_recorded_with_the_result(self):
        """Every number must carry the assumptions that produced it, or two runs
        cannot be compared."""
        result = evaluate_observation(_obs(), COSTS, NOTIONALS, base_is_token0=True)
        assert result.costs == COSTS


class TestLatency:
    def _decision(self):
        result = evaluate_observation(
            _obs(ts=0.0, cex=Decimal("1900"), dex=Decimal("1920")),
            COSTS, NOTIONALS, base_is_token0=True,
        )
        assert result.best is not None
        return result

    def test_zero_delay_reproduces_the_instantaneous_result(self):
        decision = self._decision()
        resolved = resolve_with_latency(
            decision, [_obs(ts=0.0, cex=Decimal("1900"), dex=Decimal("1920"))],
            delay_seconds=0.0, tolerance_seconds=0.5,
            base_is_token0=True,
        )
        assert resolved.realised_net_bps == pytest.approx(
            float(decision.best.net_bps), abs=0.01
        )

    def test_an_edge_that_closes_costs_the_trade(self):
        """The dislocation is gone by the time the order lands."""
        decision = self._decision()
        resolved = resolve_with_latency(
            decision,
            [_obs(ts=2.0, cex=Decimal("1900"), dex=Decimal("1900"))],
            delay_seconds=2.0, tolerance_seconds=0.5,
            base_is_token0=True,
        )
        assert resolved.realised_net_bps is not None
        assert resolved.realised_net_bps < float(decision.best.net_bps)

    def test_the_size_and_direction_are_frozen_at_decision_time(self):
        """The look-ahead guard. If resolution re-optimised, a later observation
        with a BETTER opposite-direction edge would flip the trade -- which the bot
        could not have done, and which would make any noise profitable."""
        decision = self._decision()
        assert decision.best.direction == "CEX_to_DEX"
        # Later, the dislocation has inverted and is larger the other way.
        resolved = resolve_with_latency(
            decision,
            [_obs(ts=2.0, cex=Decimal("1960"), dex=Decimal("1900"))],
            delay_seconds=2.0, tolerance_seconds=0.5,
            base_is_token0=True,
        )
        assert resolved.direction == "CEX_to_DEX", (
            "resolution changed the direction; it is re-optimising with future "
            "information"
        )
        assert resolved.size_base == decision.best.size_base
        # And the frozen trade should now be a loss, since the edge inverted.
        assert resolved.realised_net_bps < 0

    def test_no_successor_within_tolerance_is_unresolved_not_filled(self):
        decision = self._decision()
        resolved = resolve_with_latency(
            decision, [_obs(ts=60.0)],
            delay_seconds=2.0, tolerance_seconds=0.5,
            base_is_token0=True,
        )
        assert resolved.realised_net_bps is None
        assert resolved.unresolved_reason is not None

    def test_an_empty_future_is_unresolved(self):
        decision = self._decision()
        resolved = resolve_with_latency(
            decision, [], delay_seconds=2.0, tolerance_seconds=0.5,
            base_is_token0=True,
        )
        assert resolved.realised_net_bps is None

    def test_the_nearest_successor_to_the_target_time_is_used(self):
        """Not the first one past the delay: with irregular cadence, "first past" is
        systematically later than the delay asked for, so the model would measure a
        longer latency than it reports."""
        decision = self._decision()
        candidates = [
            _obs(ts=1.0, cex=Decimal("1900"), dex=Decimal("1919")),
            _obs(ts=2.1, cex=Decimal("1900"), dex=Decimal("1910")),
            _obs(ts=5.0, cex=Decimal("1900"), dex=Decimal("1901")),
        ]
        resolved = resolve_with_latency(
            decision, candidates, delay_seconds=2.0, tolerance_seconds=1.0,
            base_is_token0=True,
        )
        assert resolved.resolved_at == 2.1

    def test_the_realised_delay_is_reported_not_the_requested_one(self):
        """The achieved delay is what the number actually measures."""
        decision = self._decision()
        resolved = resolve_with_latency(
            decision, [_obs(ts=2.4, cex=Decimal("1900"), dex=Decimal("1910"))],
            delay_seconds=2.0, tolerance_seconds=1.0,
            base_is_token0=True,
        )
        assert resolved.realised_delay_seconds == pytest.approx(2.4)


class TestCostModel:
    def test_a_negative_fee_is_rejected(self):
        with pytest.raises(ValueError):
            CostModel(taker_fee_bps=Decimal("-1"), cex_legs=1, gas_units=200_000,
                      rotation_cost_quote=Decimal(0), floor_bps=Decimal(5))

    def test_zero_gas_units_is_rejected(self):
        """A zero gas limit is a zero gas cost by another name."""
        with pytest.raises(ValueError):
            CostModel(taker_fee_bps=Decimal("7.5"), cex_legs=1, gas_units=0,
                      rotation_cost_quote=Decimal(0), floor_bps=Decimal(5))

    def test_it_is_hashable_so_it_can_label_a_result_set(self):
        assert hash(COSTS) == hash(CostModel(
            taker_fee_bps=Decimal("7.5"), cex_legs=1, gas_units=200_000,
            rotation_cost_quote=Decimal("0"), floor_bps=Decimal("5"),
        ))
