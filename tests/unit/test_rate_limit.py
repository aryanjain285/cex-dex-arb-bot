"""Nothing accounted for the exchange's rate limit.

Binance meters REST by request weight per minute per IP. Exceed it and you get
429; keep going and you get 418, which is an IP ban lasting from two minutes to
three days. For a bot that is supposed to stay connected, a ban is not a slow
request -- it is an outage, and it takes the WebSocket feed with it.

Today five endpoints are called from four modules, each with its own
aiohttp.ClientSession and no shared accounting:

    exchangeInfo (weight 20), klines (2), ticker/price (2), ticker/bookTicker (2),
    account (20), order (1), userDataStream (2)

The volume scanner fetches klines for the whole symbol universe at concurrency
20. Several hundred symbols is a burst of over a thousand weight with nothing
watching, and a retry storm multiplies it. Nobody read `X-MBX-USED-WEIGHT-1M`,
which is the exchange telling us exactly how much budget is left.

The governor is a single chokepoint that owns the request, so accounting cannot
be bypassed by adding a call site.
"""
import asyncio
from decimal import Decimal

import pytest

from src.exchange.rate_limit import (
    IpBannedError,
    RateLimitedError,
    WeightGovernor,
)


class FakeClock:
    def __init__(self, t=1000.0):
        self.t = t

    def __call__(self):
        return self.t

    def advance(self, seconds):
        self.t += seconds


class SleepSpy:
    """Records sleeps and advances the clock instead of really waiting.

    A test that actually slept would either be slow or would assert nothing --
    the interesting property is how long the governor decided to wait.
    """

    def __init__(self, clock):
        self.clock = clock
        self.calls = []

    async def __call__(self, seconds):
        self.calls.append(seconds)
        self.clock.advance(seconds)

    @property
    def total(self):
        return sum(self.calls)


def _governor(clock, sleeper, **kw):
    kw.setdefault("max_weight_per_minute", 100)
    kw.setdefault("safety_fraction", 1.0)
    return WeightGovernor(now_fn=clock, sleep_fn=sleeper, **kw)


# --- the budget ----------------------------------------------------------


async def test_requests_below_the_ceiling_do_not_wait():
    clock = FakeClock()
    sleeper = SleepSpy(clock)
    gov = _governor(clock, sleeper)

    for _ in range(10):
        await gov.acquire(5)  # 50 of 100

    assert sleeper.calls == [], "the governor throttled inside its own budget"


async def test_the_ceiling_forces_a_wait_until_the_window_rolls():
    clock = FakeClock()
    sleeper = SleepSpy(clock)
    gov = _governor(clock, sleeper)

    await gov.acquire(100)  # the whole minute's budget at t=1000
    await gov.acquire(1)    # must wait for the first to age out

    assert sleeper.total >= 60.0, (
        f"expected a wait of about a minute, got {sleeper.calls}"
    )


async def test_weight_ages_out_of_the_window():
    clock = FakeClock()
    sleeper = SleepSpy(clock)
    gov = _governor(clock, sleeper)

    await gov.acquire(100)
    clock.advance(61)
    await gov.acquire(100)

    assert sleeper.calls == [], "old weight was still being counted"


async def test_the_safety_fraction_reserves_headroom():
    """The bot must not ride its own limit: a burst it did not predict, or
    another process on the same IP, has to have somewhere to go."""
    clock = FakeClock()
    sleeper = SleepSpy(clock)
    gov = _governor(clock, sleeper, max_weight_per_minute=100, safety_fraction=0.5)

    assert gov.ceiling == 50
    await gov.acquire(50)
    await gov.acquire(1)

    assert sleeper.total >= 60.0


async def test_a_single_request_larger_than_the_ceiling_is_a_configuration_error():
    """Waiting forever would be worse: the request can never fit, and a bot that
    hangs silently is harder to diagnose than one that refuses to start."""
    clock = FakeClock()
    gov = _governor(clock, SleepSpy(clock), max_weight_per_minute=10)

    with pytest.raises(ValueError, match="never fit|ceiling"):
        await gov.acquire(11)


# --- what the exchange says ---------------------------------------------


async def test_the_servers_number_overrides_the_local_estimate():
    """`X-MBX-USED-WEIGHT-1M` is ground truth. The local ledger cannot see other
    processes on the same IP, retries inside a client library, or weights that
    differ from the documented value -- so where the two disagree, the server
    wins."""
    clock = FakeClock()
    sleeper = SleepSpy(clock)
    gov = _governor(clock, sleeper)

    await gov.acquire(5)
    gov.observe_headers({"X-MBX-USED-WEIGHT-1M": "99"})
    # 99 + 2 exceeds the ceiling of 100; a weight of 1 would fit exactly, and
    # asserting a wait on that would be asserting the wrong thing.
    await gov.acquire(2)

    assert sleeper.total >= 60.0, (
        "the governor ignored the exchange's own usage figure"
    )


async def test_a_lower_server_figure_does_not_relax_the_local_ledger():
    """Trust the server when it says we are closer to the limit, not when it
    says we are further away: the reported window may simply have just rolled,
    and treating that as free budget is how a burst becomes a ban."""
    clock = FakeClock()
    sleeper = SleepSpy(clock)
    gov = _governor(clock, sleeper)

    await gov.acquire(100)
    gov.observe_headers({"X-MBX-USED-WEIGHT-1M": "0"})
    await gov.acquire(1)

    assert sleeper.total >= 60.0


async def test_a_missing_header_is_not_an_error():
    """Not every endpoint returns it, and a response with no header must not
    break the request that carried it."""
    clock = FakeClock()
    gov = _governor(clock, SleepSpy(clock))
    gov.observe_headers({})
    gov.observe_headers({"X-MBX-USED-WEIGHT-1M": "not-a-number"})


# --- the two failure statuses -------------------------------------------


async def test_429_waits_for_retry_after_and_raises():
    """The caller must learn it was throttled -- silently returning nothing would
    look like an empty market."""
    clock = FakeClock()
    sleeper = SleepSpy(clock)
    gov = _governor(clock, sleeper)

    with pytest.raises(RateLimitedError):
        await gov.handle_status(429, {"Retry-After": "7"})

    assert sleeper.total >= 7.0


async def test_429_without_a_retry_after_still_backs_off():
    clock = FakeClock()
    sleeper = SleepSpy(clock)
    gov = _governor(clock, sleeper)

    with pytest.raises(RateLimitedError):
        await gov.handle_status(429, {})

    assert sleeper.total > 0, "a 429 with no header must still back off"


async def test_418_is_fatal_and_does_not_retry():
    """418 means the IP is already banned. Retrying extends the ban, so the only
    correct response is to stop and surface it."""
    clock = FakeClock()
    sleeper = SleepSpy(clock)
    gov = _governor(clock, sleeper)

    with pytest.raises(IpBannedError):
        await gov.handle_status(418, {"Retry-After": "120"})

    assert gov.banned_until is not None
    assert sleeper.calls == [], "a banned IP must not be slept against and retried"


async def test_a_banned_governor_refuses_further_acquisitions():
    clock = FakeClock()
    gov = _governor(clock, SleepSpy(clock))

    with pytest.raises(IpBannedError):
        await gov.handle_status(418, {"Retry-After": "120"})

    with pytest.raises(IpBannedError):
        await gov.acquire(1)


async def test_the_ban_expires():
    clock = FakeClock()
    gov = _governor(clock, SleepSpy(clock))

    with pytest.raises(IpBannedError):
        await gov.handle_status(418, {"Retry-After": "120"})

    clock.advance(121)
    await gov.acquire(1)  # must not raise


async def test_a_normal_status_is_a_no_op():
    clock = FakeClock()
    gov = _governor(clock, SleepSpy(clock))
    await gov.handle_status(200, {})
    await gov.handle_status(400, {})  # a bad request is not a rate-limit event


# --- concurrency ---------------------------------------------------------


async def test_concurrent_callers_share_one_budget():
    """The whole point. Four modules with four sessions previously each believed
    they had the full budget.

    Uses the real event loop and real (tiny) sleeps, because the property under
    test is mutual exclusion between coroutines -- which a fake sleep that never
    yields would hide.
    """
    gov = WeightGovernor(max_weight_per_minute=10, safety_fraction=1.0,
                         window_seconds=0.5)

    async def one():
        await gov.acquire(1)

    await asyncio.gather(*(one() for _ in range(10)))

    # An 11th inside the same window has to wait for the window to roll.
    assert gov.used_weight() == 10
    await gov.acquire(1)
    assert gov.used_weight() <= 10, (
        "the ledger kept stale weight after the window rolled"
    )


async def test_the_ledger_does_not_grow_without_bound():
    """This runs for weeks. Entries outside the window must be discarded, not
    merely ignored."""
    clock = FakeClock()
    gov = _governor(clock, SleepSpy(clock), max_weight_per_minute=1_000_000)

    for i in range(5000):
        clock.advance(1)
        await gov.acquire(1)

    assert gov.entry_count() <= 61, (
        f"the ledger holds {gov.entry_count()} entries for a 60s window"
    )
