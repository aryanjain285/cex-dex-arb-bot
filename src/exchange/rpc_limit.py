"""Pacing chain requests, per chain.

The exchange side has a request-weight governor. The chain side had nothing, and
the universe survey demonstrated the consequence: sustained 429s from public
endpoints, and -- before the attribution fix -- every one reported upward as "no
pool", so a throttled bot looked like an empty market.

RPC providers meter differently from Binance. There is no weight header and no
published cost per method: the limits are requests per second and concurrent
requests, and both vary by provider and plan. So this is a token bucket plus a
concurrency cap.

PER CHAIN, not global. Ethereum and Base are different endpoints with different
budgets, and one shared limiter would let a survey of Base throttle detection on
Ethereum -- coupling two things that share nothing.

IT ONLY EVER DELAYS. A limiter that dropped a call would turn a paced request into
a missing quote, and a missing quote is indistinguishable from an empty market:
exactly the confusion the RpcError work exists to remove. The only failure mode
here is latency.
"""
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from typing import Callable, Dict, Mapping, Optional

from loguru import logger

from ..core import clock

__all__ = ["RpcLimiter"]


class _Bucket:
    """One chain's token bucket. Not thread-safe; one event loop only."""

    def __init__(self, rate: float, now: float):
        self.rate = rate
        # Starts full: a process that has just started has not spent anything, and
        # making it wait for its first call would add latency for no protection.
        self.tokens = rate
        self.updated = now

    def capacity(self) -> float:
        """Burst size: at least one token, whatever the rate.

        `acquire` needs a whole token to proceed, so a capacity equal to the rate
        deadlocks any rate below 1/s -- the bucket refills to `rate`, which is less
        than 1, and the caller waits forever. That presents as a hung process, and
        it made every sub-1/s rate unusable: the configured floor, and every
        effective rate produced by backing off a throttled endpoint.

        A rate of 0.25/s means one request every four seconds, not none.
        """
        return max(1.0, self.rate)

    def refill(self, now: float) -> None:
        elapsed = now - self.updated
        if elapsed <= 0:
            return
        self.tokens = min(self.capacity(), self.tokens + elapsed * self.rate)
        self.updated = now


class RpcLimiter:
    def __init__(
        self,
        requests_per_second: float = 10.0,
        max_concurrency: int = 8,
        per_chain_requests_per_second: Optional[Mapping[str, float]] = None,
        now_fn: Optional[Callable[[], float]] = None,
        sleep_fn: Optional[Callable[[float], "asyncio.Future"]] = None,
        throttle_decay: float = 0.5,
        throttle_recovery_seconds: float = 120.0,
        min_requests_per_second: float = 0.25,
    ):
        if requests_per_second <= 0:
            raise ValueError(
                f"requests_per_second must be positive, got {requests_per_second}. "
                f"Zero would block forever, which presents as a hung bot rather "
                f"than as a configuration error."
            )
        if max_concurrency <= 0:
            raise ValueError(
                f"max_concurrency must be positive, got {max_concurrency}"
            )

        self.requests_per_second = float(requests_per_second)
        self.max_concurrency = int(max_concurrency)
        self.per_chain = {
            str(k).lower(): float(v)
            for k, v in (per_chain_requests_per_second or {}).items()
        }
        for chain, rate in self.per_chain.items():
            if rate <= 0:
                raise ValueError(
                    f"requests_per_second for {chain} must be positive, got {rate}"
                )

        # Adaptive backoff. The configured rate is a GUESS -- providers publish no
        # per-method cost and meter differently by plan -- so a 429 is information,
        # not just a failure. Measured: 8 req/s drew 429s from Base while Arbitrum
        # served the same load. Without this, the response to "too fast" was to
        # continue at the same speed, and a throttled endpoint degraded into a
        # mostly-failing one; in a recording run that looks like a market with no
        # data rather than an endpoint that needs slowing down.
        self.throttle_decay = float(throttle_decay)
        self.throttle_recovery_seconds = float(throttle_recovery_seconds)
        self.min_requests_per_second = float(min_requests_per_second)
        if not 0 < self.throttle_decay < 1:
            raise ValueError("throttle_decay must be in (0, 1)")
        if self.throttle_recovery_seconds <= 0:
            raise ValueError("throttle_recovery_seconds must be positive")
        if self.min_requests_per_second <= 0:
            raise ValueError(
                "min_requests_per_second must be positive; a floor of zero would "
                "block forever, presenting as a hung process rather than a slow one"
            )
        # chain -> {"factor": float, "updated": float, "events": int}
        self._throttle: Dict[str, Dict[str, float]] = {}

        # Monotonic: a wall-clock jump would otherwise refill or freeze the bucket.
        self._now_fn = now_fn
        self._sleep_fn = sleep_fn
        self._buckets: Dict[str, _Bucket] = {}
        self._semaphores: Dict[str, asyncio.Semaphore] = {}
        self._locks: Dict[str, asyncio.Lock] = {}

    def _now(self) -> float:
        return clock.monotonic() if self._now_fn is None else self._now_fn()

    async def _sleep(self, seconds: float) -> None:
        if self._sleep_fn is None:
            await asyncio.sleep(seconds)
        else:
            await self._sleep_fn(seconds)

    def rate_for(self, chain: str) -> float:
        """The CONFIGURED rate. See `effective_rate` for what is actually used."""
        return self.per_chain.get(str(chain).lower(), self.requests_per_second)

    def throttled(self, chain: str) -> float:
        """Record that this chain refused a request, and slow it down.

        Multiplicative decrease: the correct response to an unknown limit that has
        just been exceeded, because the overshoot could be by any factor. Returns
        the new effective rate.
        """
        key = str(chain).lower()
        state = self._throttle_state(key)
        configured = self.rate_for(key)
        floor = min(1.0, self.min_requests_per_second / configured)
        state["factor"] = max(floor, state["factor"] * self.throttle_decay)
        state["updated"] = self._now()
        state["events"] += 1
        rate = configured * state["factor"]
        logger.warning(
            f"RPC limiter: {key} refused a request; rate {configured:g}/s -> "
            f"{rate:.2f}/s (throttle event {int(state['events'])})"
        )
        # The bucket caches its rate, so it has to be told. Refill clamps the token
        # count to the new rate, which makes the slowdown take effect immediately
        # rather than after the existing tokens are spent.
        bucket = self._buckets.get(key)
        if bucket is not None:
            bucket.rate = rate
            # Clamp the banked tokens explicitly. `refill` returns early when no
            # time has elapsed, so relying on it here left a full burst of
            # old-rate tokens available at the exact moment the endpoint asked us
            # to slow down -- the worst possible time to spend them.
            bucket.tokens = min(bucket.tokens, bucket.capacity())
        return rate

    def effective_rate(self, chain: str) -> float:
        """The rate currently in use: configured, reduced by any active throttle.

        Recovery is additive over `throttle_recovery_seconds` and stops at the
        configured rate -- it restores a known-good setting rather than probing for a
        higher one, which would make the configured value meaningless.
        """
        key = str(chain).lower()
        configured = self.rate_for(key)
        # Registered on first look, so `stats()` can report a chain that has been
        # used and never throttled. A chain absent from the report is
        # indistinguishable from one that was never touched.
        state = self._throttle_state(key)
        if state["factor"] >= 1.0:
            return configured
        now = self._now()
        elapsed = max(0.0, now - state["updated"])
        if elapsed > 0 and state["factor"] < 1.0:
            state["factor"] = min(
                1.0, state["factor"] + elapsed / self.throttle_recovery_seconds
            )
            state["updated"] = now
        return configured * state["factor"]

    def _throttle_state(self, key: str) -> Dict[str, float]:
        state = self._throttle.get(key)
        if state is None:
            state = {"factor": 1.0, "updated": self._now(), "events": 0}
            self._throttle[key] = state
        return state

    def stats(self) -> Dict[str, Dict[str, float]]:
        """Per-chain pacing state, so a run can explain its own throughput."""
        chains = set(self._buckets) | set(self._throttle) | set(self.per_chain)
        out = {}
        for chain in sorted(chains):
            state = self._throttle.get(chain, {})
            out[chain] = {
                "configured_rate": self.rate_for(chain),
                "effective_rate": self.effective_rate(chain),
                "throttle_events": int(state.get("events", 0)),
            }
        return out

    def _bucket(self, chain: str) -> _Bucket:
        key = str(chain).lower()
        bucket = self._buckets.get(key)
        if bucket is None:
            bucket = _Bucket(self.effective_rate(key), self._now())
            self._buckets[key] = bucket
        else:
            # Follow the effective rate, which drifts back up as a throttle decays.
            bucket.rate = self.effective_rate(key)
        return bucket

    def _semaphore(self, chain: str) -> asyncio.Semaphore:
        key = str(chain).lower()
        semaphore = self._semaphores.get(key)
        if semaphore is None:
            semaphore = asyncio.Semaphore(self.max_concurrency)
            self._semaphores[key] = semaphore
        return semaphore

    def _lock(self, chain: str) -> asyncio.Lock:
        key = str(chain).lower()
        lock = self._locks.get(key)
        if lock is None:
            lock = asyncio.Lock()
            self._locks[key] = lock
        return lock

    @asynccontextmanager
    async def acquire(self, chain: str):
        """Hold one slot on this chain for the duration of the block.

        The slot is released on the way out whichever way that happens. An RPC
        failure is the common case, not the exception, and a leaked slot would
        wedge the limiter after N errors -- presenting as a hung bot rather than
        as a limiter bug.
        """
        # Rate first, then concurrency: waiting for a token while holding a slot
        # would make the concurrency cap the binding constraint at any rate.
        async with self._lock(chain):
            bucket = self._bucket(chain)
            while True:
                bucket.refill(self._now())
                if bucket.tokens >= 1:
                    bucket.tokens -= 1
                    break
                needed = (1 - bucket.tokens) / bucket.rate
                logger.debug(
                    f"RPC limiter: pacing {chain} for {needed:.3f}s "
                    f"({bucket.rate}/s budget)"
                )
                await self._sleep(max(needed, 0.001))

        semaphore = self._semaphore(chain)
        await semaphore.acquire()
        try:
            yield
        finally:
            semaphore.release()

    def describe(self) -> str:
        overrides = (
            ", per-chain: " + ", ".join(
                f"{chain}={rate:g}/s" for chain, rate in sorted(self.per_chain.items())
            )
            if self.per_chain else ""
        )
        # Throttled chains are named explicitly. A run that collected half the
        # observations it expected has to explain that in its own output rather than
        # leave it to be inferred from the failure count.
        throttled = [
            f"{chain}={info['effective_rate']:.2f}/s "
            f"({info['throttle_events']} events)"
            for chain, info in self.stats().items()
            if info["throttle_events"]
        ]
        throttling = (
            ", throttled: " + ", ".join(throttled) if throttled else ""
        )
        return (
            f"RPC limiter: {self.requests_per_second:g} req/s per chain, "
            f"max {self.max_concurrency} concurrent{overrides}{throttling}"
        )
