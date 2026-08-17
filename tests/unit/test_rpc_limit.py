"""Pacing the chain side, which had no budget at all.

The exchange side has a request-weight governor. The chain side had nothing, and
the universe survey demonstrated the consequence: sustained 429s from public
endpoints, and before the attribution fix every one of them was reported upward as
"no pool".

RPC providers meter differently from Binance. There is no weight header and no
published cost per method; the limits are requests per second and concurrent
requests, and they vary per provider and per plan. So this is a token bucket plus
a concurrency cap, per chain -- Ethereum and Base are different endpoints with
different budgets, and a shared limiter would let a busy chain starve a quiet one.

Two properties matter more than throughput:

  * It must never LIE. A limiter that silently drops a call would turn a paced
    request into a missing quote, which is the failure the attribution fix exists
    to prevent. It only ever delays.
  * It must be per chain. One budget across chains would make a survey of Base
    throttle detection on Ethereum, coupling two things that share nothing.
"""
import asyncio

import pytest

from src.exchange.rpc_limit import RpcLimiter


class FakeClock:
    def __init__(self, t=1000.0):
        self.t = t

    def __call__(self):
        return self.t

    def advance(self, seconds):
        self.t += seconds


class SleepSpy:
    def __init__(self, clock):
        self.clock = clock
        self.calls = []

    async def __call__(self, seconds):
        self.calls.append(seconds)
        self.clock.advance(seconds)

    @property
    def total(self):
        return sum(self.calls)


def _limiter(clock, sleeper, **kw):
    kw.setdefault("requests_per_second", 5.0)
    kw.setdefault("max_concurrency", 4)
    return RpcLimiter(now_fn=clock, sleep_fn=sleeper, **kw)


# --- the rate budget -----------------------------------------------------


async def test_calls_within_the_budget_do_not_wait():
    clock = FakeClock()
    sleeper = SleepSpy(clock)
    limiter = _limiter(clock, sleeper, requests_per_second=5.0)

    for _ in range(5):
        async with limiter.acquire("ethereum"):
            pass

    assert sleeper.calls == [], "the limiter paced inside its own budget"


async def test_exceeding_the_budget_waits():
    clock = FakeClock()
    sleeper = SleepSpy(clock)
    limiter = _limiter(clock, sleeper, requests_per_second=5.0)

    for _ in range(6):
        async with limiter.acquire("ethereum"):
            pass

    assert sleeper.total > 0, "a sixth call in one second must be delayed"


async def test_the_budget_refills_over_time():
    clock = FakeClock()
    sleeper = SleepSpy(clock)
    limiter = _limiter(clock, sleeper, requests_per_second=5.0)

    for _ in range(5):
        async with limiter.acquire("ethereum"):
            pass
    clock.advance(2.0)
    before = sleeper.total
    for _ in range(5):
        async with limiter.acquire("ethereum"):
            pass

    assert sleeper.total == before, "the bucket did not refill"


async def test_chains_have_independent_budgets():
    """A survey of Base must not throttle detection on Ethereum. The two are
    different endpoints with different limits and share nothing."""
    clock = FakeClock()
    sleeper = SleepSpy(clock)
    limiter = _limiter(clock, sleeper, requests_per_second=2.0)

    for _ in range(2):
        async with limiter.acquire("base"):
            pass
    before = sleeper.total

    for _ in range(2):
        async with limiter.acquire("ethereum"):
            pass

    assert sleeper.total == before, "one chain's spending delayed another's"


async def test_a_per_chain_override_is_honoured():
    """A paid endpoint on one chain and a public one on another is the normal
    case, not the exception."""
    clock = FakeClock()
    sleeper = SleepSpy(clock)
    limiter = _limiter(
        clock, sleeper, requests_per_second=1.0,
        per_chain_requests_per_second={"ethereum": 50.0},
    )

    for _ in range(20):
        async with limiter.acquire("ethereum"):
            pass
    assert sleeper.calls == []

    for _ in range(3):
        async with limiter.acquire("base"):
            pass
    assert sleeper.total > 0


# --- concurrency ---------------------------------------------------------


async def test_concurrency_is_capped():
    """Providers cap concurrent requests separately from rate, and exceeding it
    produces the same 429."""
    limiter = RpcLimiter(requests_per_second=1000.0, max_concurrency=2)
    peak = 0
    current = 0

    async def call():
        nonlocal peak, current
        async with limiter.acquire("ethereum"):
            current += 1
            peak = max(peak, current)
            await asyncio.sleep(0.01)
            current -= 1

    await asyncio.gather(*(call() for _ in range(10)))

    assert peak <= 2, f"{peak} calls were in flight against a cap of 2"


async def test_the_slot_is_released_even_when_the_call_raises():
    """An RPC failure is the common case, not the exception. A limiter that leaked
    a slot on failure would wedge after N errors -- and would look like a hung
    bot rather than a limiter bug."""
    limiter = RpcLimiter(requests_per_second=1000.0, max_concurrency=1)

    for _ in range(3):
        with pytest.raises(RuntimeError):
            async with limiter.acquire("ethereum"):
                raise RuntimeError("node said no")

    # If a slot had leaked, this would block forever.
    async with limiter.acquire("ethereum"):
        pass


# --- configuration -------------------------------------------------------


def test_a_non_positive_rate_is_rejected():
    """Zero would block forever, which presents as a hung bot rather than as a
    configuration error."""
    with pytest.raises(ValueError):
        RpcLimiter(requests_per_second=0)


def test_a_non_positive_concurrency_is_rejected():
    with pytest.raises(ValueError):
        RpcLimiter(requests_per_second=5.0, max_concurrency=0)


def test_the_limiter_reports_its_settings():
    limiter = RpcLimiter(
        requests_per_second=5.0, max_concurrency=4,
        per_chain_requests_per_second={"ethereum": 25.0},
    )

    description = limiter.describe()

    assert "5" in description and "4" in description and "ethereum" in description


# --- it only ever delays -------------------------------------------------


def test_no_chain_call_bypasses_the_chokepoint():
    """The single-chokepoint guarantee, checked over the AST.

    Sixteen call sites in univ3.py issue chain calls. A limiter applied at each
    one can be bypassed by adding a seventeenth -- which is exactly how the
    exchange side went unmetered for so long. Every `asyncio.to_thread` must be
    inside `_rpc` or the explicitly-named `_rpc_unpaced`.
    """
    import ast
    import inspect

    from src.exchange import univ3

    tree = ast.parse(inspect.getsource(univ3))

    # Which function does each to_thread call live in?
    offenders = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if node.name in ("_rpc", "_rpc_unpaced"):
            continue
        for inner in ast.walk(node):
            if not isinstance(inner, ast.Call):
                continue
            func = inner.func
            if (isinstance(func, ast.Attribute) and func.attr == "to_thread"
                    and isinstance(func.value, ast.Name)
                    and func.value.id == "asyncio"):
                offenders.append(f"{node.name} at line {inner.lineno}")

    assert not offenders, (
        "these chain calls bypass the RPC limiter: " + ", ".join(offenders)
    )


def test_a_receipt_wait_is_deliberately_unpaced():
    """A receipt poll runs as long as the chain takes. Holding a concurrency slot
    for a minute would let one pending transaction starve every quote, which is
    why the exception exists and is named rather than implicit."""
    import inspect

    from src.exchange import univ3

    source = inspect.getsource(univ3)
    assert "_rpc_unpaced(\n                w3.eth.wait_for_transaction_receipt" in source \
        or "_rpc_unpaced(\n            w3.eth.wait_for_transaction_receipt" in source, (
        "the receipt wait should use the unpaced path"
    )


async def test_the_limiter_never_drops_a_call():
    """The property that matters most. A dropped call would become a missing
    quote, and a missing quote is indistinguishable from an empty market -- which
    is the exact confusion the RpcError attribution work exists to remove."""
    clock = FakeClock()
    sleeper = SleepSpy(clock)
    limiter = _limiter(clock, sleeper, requests_per_second=1.0)
    completed = 0

    for _ in range(10):
        async with limiter.acquire("ethereum"):
            completed += 1

    assert completed == 10
