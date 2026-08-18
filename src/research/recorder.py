"""Collect raw market observations at cadence, and be honest about the gaps.

This is the data-collection engine the research rests on. Everything downstream --
the size curves, the edge distribution, the capacity estimate, the significance
test -- is computed from what this writes, so its failure modes matter more than
its throughput.

Three of them are silent, and each turns an infrastructure problem into a false
statement about the market:

  * ONE FAILING CHAIN. If a per-target failure aborted the cycle, a week of
    recording would quietly become a week of recording whichever pair sits first
    in the list on whichever endpoint stayed up. The cross-pair comparison would
    then be a comparison of endpoint reliability. So each target is isolated.

  * UNCOUNTED FAILURES. A gap in a time series is indistinguishable from a quiet
    market. Failures are counted per pair, and the achieved cadence is measured
    rather than assumed -- this system has already been caught configuring 0.2s
    and achieving 2.32s, a 12x error in every per-second figure derived from it.

  * HALF OBSERVATIONS. An observation with a CEX book and no pool state, or the
    reverse, would be read as a valid instant by anything computing a spread. A
    target that cannot produce both sides produces nothing.

Gas is treated differently from the two venues on purpose: it is a cost input, not
part of the market state, so losing it costs the cost model rather than the
observation. The row then records the gas price as ABSENT, which is not the same as
zero -- a zero gas cost is the single easiest way to make this strategy look
profitable, since its edge and its gas are the same order of magnitude.
"""
from __future__ import annotations

import asyncio
import statistics
import time
import uuid
from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Dict, List, Optional, Sequence

from loguru import logger

from ..exchange.errors import RpcError
from .observations import Observation, ObservationStore

__all__ = ["Recorder", "RecorderTarget", "GasReader"]


@dataclass(frozen=True)
class RecorderTarget:
    """One pair on one pool. The pool address is resolved once, at setup.

    Resolved once rather than per cycle because the factory lookup is an RPC call
    that returns the same answer every time -- and at five-second cadence over a
    week that is a hundred thousand pointless calls competing with the reads that
    carry information.
    """

    pair: Any
    pool_address: str


class GasReader:
    """Current gas price and native-token price for a chain.

    Separate from the pool reader because it is refreshed on a different clock: gas
    moves on block time, the native price on the exchange's feed, and neither needs
    to be re-read as often as a pool's price. It also fails independently, and its
    failure must not cost an observation.
    """

    def __init__(self, dex_client, cex_client=None, ttl_seconds: float = 15.0):
        self._dex = dex_client
        self._cex = cex_client
        self._ttl = ttl_seconds
        self._cache: Dict[str, tuple] = {}

    async def read(self, chain: str):
        """(gas_price_wei, native_price_quote). Raises on failure."""
        now = time.monotonic()
        cached = self._cache.get(chain)
        if cached is not None and now - cached[0] < self._ttl:
            return cached[1], cached[2]

        w3 = self._dex._get_w3(chain)
        gas_price = await self._dex._rpc(chain, lambda: w3.eth.gas_price)
        native_price = await self._native_price(chain)
        self._cache[chain] = (now, int(gas_price), native_price)
        return int(gas_price), native_price

    async def _native_price(self, chain: str) -> Optional[Decimal]:
        getter = getattr(self._dex, "get_native_price_quote", None)
        if getter is None:
            return None
        return await getter(chain)


class Recorder:
    def __init__(
        self,
        *,
        store: ObservationStore,
        cex,
        pools,
        gas,
        targets: Sequence[RecorderTarget],
        interval_seconds: float = 5.0,
        run_id: Optional[str] = None,
    ):
        if not targets:
            raise ValueError(
                "no targets to record. A recorder with nothing to read would run "
                "for a week and produce an empty file, which reads as 'no "
                "opportunities' rather than 'misconfigured'."
            )
        self.store = store
        self.cex = cex
        self.pools = pools
        self.gas = gas
        self.targets = list(targets)
        self.interval_seconds = interval_seconds
        self.run_id = run_id or store.run_id or uuid.uuid4().hex[:12]

        self._stop = False
        self._cycles = 0
        self._recorded = 0
        self._failed = 0
        self._failures_by_pair: Dict[str, int] = {}
        self._failures_by_reason: Dict[str, int] = {}
        # Cycle start times, for the achieved cadence. Bounded: a week at five
        # seconds is 120k entries, and the median of the last few hundred is a
        # better description of current behaviour than the mean of all of them.
        self._cycle_starts: List[float] = []
        self._cadence_window = 500

    # -- one cycle ---------------------------------------------------------

    async def cycle(self) -> int:
        """Record every target once. Returns how many observations were written."""
        started = time.monotonic()
        self._cycle_starts.append(started)
        if len(self._cycle_starts) > self._cadence_window:
            del self._cycle_starts[0]

        # return_exceptions: one target's failure must not cancel its siblings.
        # Without this, `gather` propagates the first exception and cancels the
        # rest -- so a single bad endpoint silently reduces the recording to
        # whichever pairs come before it.
        results = await asyncio.gather(
            *(self._observe(target) for target in self.targets),
            return_exceptions=True,
        )

        observations: List[Observation] = []
        for target, result in zip(self.targets, results):
            symbol = target.pair.cex_symbol
            if isinstance(result, BaseException):
                self._count_failure(symbol, result)
                continue
            if result is None:
                # `_observe` already counted and explained it.
                continue
            observations.append(result)

        if observations:
            self.store.record_many(observations)
            self._recorded += len(observations)
        self._cycles += 1
        return len(observations)

    async def _observe(self, target: RecorderTarget) -> Optional[Observation]:
        pair = target.pair
        symbol = pair.cex_symbol

        # Both venues concurrently: the point of an observation is that its two
        # halves describe the same instant, so reading them in sequence would put
        # the RPC latency between them. On a public endpoint that is seconds, which
        # is longer than the edge survives.
        book_task = self._read_book(pair)
        pool_task = self.pools.get(
            pair.dex_chain, target.pool_address,
            decimals0=None, decimals1=None,
        )
        book, pool = await asyncio.gather(book_task, pool_task)

        if book is None or not book.bids or not book.asks:
            self._count_failure(symbol, ValueError("no two-sided CEX book"))
            return None
        if pool is None:
            self._count_failure(symbol, ValueError("no pool snapshot"))
            return None

        # Gas is allowed to be missing. See the module docstring.
        gas_price_wei, native_price = None, None
        try:
            gas_price_wei, native_price = await self.gas.read(pair.dex_chain)
        except Exception as exc:  # noqa: BLE001 - any failure here is non-fatal
            logger.debug(f"{symbol}: gas unavailable this cycle ({exc})")

        return Observation(
            ts=time.time(),
            cex_symbol=symbol,
            base=pair.base,
            quote=pair.quote,
            chain=pair.dex_chain,
            pool_fee=int(pair.dex_pool_fee),
            pool_address=target.pool_address,
            cex_bids=list(book.bids),
            cex_asks=list(book.asks),
            cex_feed_ts=getattr(book, "feed_timestamp", None),
            pool=pool,
            gas_price_wei=gas_price_wei,
            native_price_quote=native_price,
            rpc_endpoint=self._endpoint_for(pair.dex_chain),
            run_id=self.run_id,
        )

    async def _read_book(self, pair):
        return await self.cex.get_book(pair)

    def _endpoint_for(self, chain: str) -> Optional[str]:
        """Which endpoint answered, so numbers from different runs are comparable.

        Best-effort: the reader may not expose it. Recording None is honest;
        recording a guess would be worse than recording nothing.
        """
        client = getattr(self.pools, "reader", None)
        client = getattr(client, "client", None)
        urls = getattr(getattr(client, "net_config", None), "rpc_urls", None)
        if isinstance(urls, dict):
            return urls.get(chain)
        return None

    def _count_failure(self, symbol: str, exc: BaseException) -> None:
        self._failed += 1
        self._failures_by_pair[symbol] = self._failures_by_pair.get(symbol, 0) + 1
        reason = type(exc).__name__ if not isinstance(exc, RpcError) else "RpcError"
        self._failures_by_reason[reason] = self._failures_by_reason.get(reason, 0) + 1
        logger.debug(f"{symbol}: observation failed ({reason}: {exc})")

    # -- the loop ----------------------------------------------------------

    async def run(self, max_cycles: Optional[int] = None) -> None:
        """Record until stopped, or for a fixed number of cycles.

        The interval is measured from the START of one cycle to the start of the
        next, so a slow cycle eats its own sleep rather than adding to it. Sleeping
        a fixed amount AFTER the work makes the achieved cadence the sum of the
        interval and the work, which is how a configured 0.2s became a measured
        2.32s.
        """
        self._stop = False
        while not self._stop:
            started = time.monotonic()
            try:
                await self.cycle()
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 - the loop must outlive a cycle
                logger.error(f"recorder cycle failed entirely: {exc}")

            if max_cycles is not None and self._cycles >= max_cycles:
                break
            if self._stop:
                break

            elapsed = time.monotonic() - started
            remaining = self.interval_seconds - elapsed
            if remaining > 0:
                await asyncio.sleep(remaining)
            elif self.interval_seconds > 0:
                # Not a warning per cycle -- that would flood the log at exactly
                # the moment the log matters. The measured cadence in stats() is
                # the durable record.
                logger.debug(
                    f"cycle took {elapsed:.2f}s, longer than the "
                    f"{self.interval_seconds:.2f}s interval"
                )

    def stop(self) -> None:
        self._stop = True

    # -- reporting ---------------------------------------------------------

    def measured_cadence_seconds(self) -> Optional[float]:
        """Median seconds between cycle starts, or None before two cycles.

        Median rather than mean: one 30-second stall on a public endpoint would
        drag a mean far from anything the recorder typically achieves, and the
        typical value is what per-second statistics need.
        """
        if len(self._cycle_starts) < 2:
            return None
        gaps = [b - a for a, b in zip(self._cycle_starts, self._cycle_starts[1:])]
        return statistics.median(gaps)

    def stats(self) -> Dict[str, Any]:
        attempted = self._recorded + self._failed
        return {
            "run_id": self.run_id,
            "cycles": self._cycles,
            "targets": len(self.targets),
            "recorded": self._recorded,
            "failed": self._failed,
            # The number an analysis needs in order to know whether a gap is the
            # market or the endpoint.
            "failure_rate": (self._failed / attempted) if attempted else 0.0,
            "failures_by_pair": dict(self._failures_by_pair),
            "failures_by_reason": dict(self._failures_by_reason),
            "configured_interval_seconds": self.interval_seconds,
            "measured_cadence_seconds": self.measured_cadence_seconds(),
        }
