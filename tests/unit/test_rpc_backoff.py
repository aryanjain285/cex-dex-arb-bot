"""A 429 must change the rate, not just be counted.

The limiter paces requests at a configured rate. That rate is a GUESS: providers
publish no per-method cost, meter differently by plan, and the same public endpoint
tolerates different loads at different times of day. Measured today: 8 req/s drew
`429 Too Many Requests` from Base while Arbitrum served the same load happily.

Before this, a 429 was classified as a transport failure, counted, and the next
request went out at exactly the rate the endpoint had just refused. So the response
to being told "too fast" was to continue at the same speed, and a throttled endpoint
degraded into a mostly-failing one -- which, in a recording run, looks like a market
with no data rather than an endpoint that needs slowing down.

The correction is the standard one for an unknown limit: multiplicative decrease on
refusal, gradual additive recovery afterwards. What matters for honesty is that the
CURRENT effective rate is observable, so a run's throughput can be explained rather
than guessed at.
"""
import pytest

from src.exchange.rpc_limit import RpcLimiter


def _limiter(rate=8.0, **kwargs):
    return RpcLimiter(
        requests_per_second=rate, max_concurrency=4,
        now_fn=_FakeClock(), **kwargs
    )


class _FakeClock:
    """Monotonic time under test control, so backoff windows do not need sleeps."""

    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t

    def advance(self, seconds):
        self.t += seconds


class TestThrottleReducesTheRate:
    def test_a_refusal_lowers_the_effective_rate(self):
        limiter = _limiter(rate=8.0)
        before = limiter.effective_rate("base")
        limiter.throttled("base")
        assert limiter.effective_rate("base") < before

    def test_repeated_refusals_keep_lowering_it(self):
        limiter = _limiter(rate=8.0)
        limiter.throttled("base")
        once = limiter.effective_rate("base")
        limiter.throttled("base")
        assert limiter.effective_rate("base") < once

    def test_the_rate_never_reaches_zero(self):
        """A rate of zero would deadlock the caller rather than slow it, turning a
        throttled endpoint into a hung recorder."""
        limiter = _limiter(rate=8.0)
        for _ in range(50):
            limiter.throttled("base")
        assert limiter.effective_rate("base") > 0

    def test_throttling_one_chain_does_not_slow_another(self):
        """Rate limits are per endpoint. Slowing Arbitrum because Base complained
        would waste most of the budget -- and it is the sort of coupling that shows
        up as an unexplained drop in throughput."""
        limiter = _limiter(rate=8.0)
        limiter.throttled("base")
        assert limiter.effective_rate("arbitrum") == pytest.approx(8.0)


class TestRecovery:
    def test_the_rate_recovers_over_time(self):
        clock = _FakeClock()
        limiter = RpcLimiter(requests_per_second=8.0, max_concurrency=4, now_fn=clock)
        limiter.throttled("base")
        throttled_rate = limiter.effective_rate("base")
        clock.advance(120.0)
        assert limiter.effective_rate("base") > throttled_rate

    def test_recovery_stops_at_the_configured_rate(self):
        """Recovery restores the configured rate; it does not discover a higher one.
        Probing upward past a configured ceiling would make the setting meaningless."""
        clock = _FakeClock()
        limiter = RpcLimiter(requests_per_second=8.0, max_concurrency=4, now_fn=clock)
        limiter.throttled("base")
        clock.advance(10_000.0)
        assert limiter.effective_rate("base") == pytest.approx(8.0)

    def test_recovery_is_gradual_not_immediate(self):
        """Jumping straight back to the refused rate would produce a cycle of 429s
        at a predictable period, which is worse than a steady lower rate."""
        clock = _FakeClock()
        limiter = RpcLimiter(requests_per_second=8.0, max_concurrency=4, now_fn=clock)
        limiter.throttled("base")
        clock.advance(1.0)
        assert limiter.effective_rate("base") < 8.0


class TestObservability:
    def test_the_effective_rate_is_reportable_per_chain(self):
        limiter = _limiter(rate=8.0)
        limiter.throttled("base")
        stats = limiter.stats()
        assert "base" in stats
        assert stats["base"]["effective_rate"] < 8.0
        assert stats["base"]["configured_rate"] == pytest.approx(8.0)
        assert stats["base"]["throttle_events"] == 1

    def test_a_chain_never_throttled_reports_zero_events(self):
        limiter = _limiter(rate=8.0)
        limiter.effective_rate("ethereum")
        assert limiter.stats()["ethereum"]["throttle_events"] == 0

    def test_the_description_mentions_throttling_when_it_happened(self):
        """A run's log has to explain its own throughput. A recorder that collected
        half the expected observations must say why in its own output, not leave it
        to be inferred."""
        limiter = _limiter(rate=8.0)
        limiter.throttled("base")
        assert "throttl" in limiter.describe().lower()


class TestPacingStillHappens:
    @pytest.mark.asyncio
    async def test_a_throttled_chain_is_paced_more_slowly(self):
        """The point of the whole exercise: the lowered rate has to actually reach
        the pacing decision, not just the stats."""
        slept = []
        clock = _FakeClock()

        async def _record(seconds):
            # Advance the clock, as a real sleep would. Without this the token
            # bucket can never refill and the limiter spins.
            slept.append(seconds)
            clock.advance(seconds)

        limiter = RpcLimiter(
            requests_per_second=100.0, max_concurrency=4,
            now_fn=clock, sleep_fn=_record,
        )
        # Drain the bucket at the configured rate.
        for _ in range(5):
            async with limiter.acquire("base"):
                pass
        baseline = sum(slept)
        slept.clear()

        for _ in range(10):
            limiter.throttled("base")
        for _ in range(5):
            async with limiter.acquire("base"):
                pass
        assert sum(slept) > baseline


class TestSubOneRateDoesNotDeadlock:
    """A rate below 1/s used to hang the process, not slow it.

    The token bucket's capacity was its rate, and `acquire` needs a whole token, so
    at 0.25/s the bucket refilled to 0.25 and the caller waited forever. Two things
    made that reachable: `min_requests_per_second` is below 1 by design, and every
    effective rate produced by backing off a throttled endpoint passes through the
    same path. It presented as a hung recorder.
    """

    @pytest.mark.asyncio
    async def test_a_configured_sub_one_rate_still_serves_requests(self):
        slept = []
        clock = _FakeClock()

        async def _record(seconds):
            slept.append(seconds)
            clock.advance(seconds)

        limiter = RpcLimiter(
            requests_per_second=0.25, max_concurrency=2,
            now_fn=clock, sleep_fn=_record,
        )
        for _ in range(3):
            async with limiter.acquire("base"):
                pass
        # It paced them -- roughly one every four seconds -- rather than hanging.
        assert sum(slept) > 0

    @pytest.mark.asyncio
    async def test_a_heavily_throttled_chain_still_serves_requests(self):
        slept = []
        clock = _FakeClock()

        async def _record(seconds):
            slept.append(seconds)
            clock.advance(seconds)

        limiter = RpcLimiter(
            requests_per_second=8.0, max_concurrency=2,
            now_fn=clock, sleep_fn=_record,
        )
        for _ in range(30):
            limiter.throttled("base")
        assert limiter.effective_rate("base") < 1.0
        for _ in range(2):
            async with limiter.acquire("base"):
                pass

    def test_the_bucket_capacity_is_never_below_one_token(self):
        from src.exchange.rpc_limit import _Bucket

        assert _Bucket(0.1, 0.0).capacity() >= 1.0
        assert _Bucket(50.0, 0.0).capacity() == 50.0


class TestTheWiringExists:
    """Backoff that is never invoked is decoration.

    The limiter is only told about a refusal if the RPC chokepoint tells it, and that
    call site is one line in one method. Checked structurally because the alternative
    is a live 429, which cannot be arranged on demand.
    """

    def test_the_rpc_chokepoint_reports_rate_limits(self):
        import inspect

        from src.exchange.univ3 import UniV3DexClient

        source = inspect.getsource(UniV3DexClient._rpc)
        assert "is_rate_limited" in source, (
            "_rpc must detect a rate-limit refusal; without it the limiter never "
            "learns and keeps requesting at the refused rate"
        )
        assert "throttled" in source

    def test_a_rate_limit_is_distinguished_from_a_plain_timeout(self):
        """Ratcheting the rate down on every timeout would cripple a slow endpoint
        that was never complaining about volume."""
        from src.exchange.errors import classify_rpc_failure, is_rate_limited

        rate_limited = RuntimeError("429 Client Error: Too Many Requests for url")
        timeout = TimeoutError("Read timed out")
        revert = ValueError("execution reverted: STF")

        assert is_rate_limited(rate_limited) is True
        assert is_rate_limited(timeout) is False
        assert is_rate_limited(revert) is False
        # Both a 429 and a timeout mean the node did not answer.
        assert classify_rpc_failure(rate_limited) is True
        assert classify_rpc_failure(timeout) is True
        assert classify_rpc_failure(revert) is False
