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

    def refill(self, now: float) -> None:
        elapsed = now - self.updated
        if elapsed <= 0:
            return
        self.tokens = min(self.rate, self.tokens + elapsed * self.rate)
        self.updated = now


class RpcLimiter:
    def __init__(
        self,
        requests_per_second: float = 10.0,
        max_concurrency: int = 8,
        per_chain_requests_per_second: Optional[Mapping[str, float]] = None,
        now_fn: Optional[Callable[[], float]] = None,
        sleep_fn: Optional[Callable[[float], "asyncio.Future"]] = None,
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
        return self.per_chain.get(str(chain).lower(), self.requests_per_second)

    def _bucket(self, chain: str) -> _Bucket:
        key = str(chain).lower()
        bucket = self._buckets.get(key)
        if bucket is None:
            bucket = _Bucket(self.rate_for(key), self._now())
            self._buckets[key] = bucket
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
        return (
            f"RPC limiter: {self.requests_per_second:g} req/s per chain, "
            f"max {self.max_concurrency} concurrent{overrides}"
        )
