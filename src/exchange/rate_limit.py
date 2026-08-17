"""One shared budget for every REST request to the exchange.

Binance meters REST by request weight per minute per IP. Exceeding it returns
429; continuing returns 418, which is an IP ban lasting from two minutes to three
days. For a bot that is meant to stay connected that is an outage, and it takes
the market-data WebSocket with it.

Before this module, five endpoints were called from four modules, each with its
own aiohttp session and no shared accounting, and nothing read
`X-MBX-USED-WEIGHT-1M` -- the header in which the exchange states exactly how
much budget is left. The volume scanner fetches klines for the whole symbol
universe at concurrency 20, which is a burst of over a thousand weight with
nothing watching it, and a retry storm multiplies that.

Three design choices worth stating:

SLIDING WINDOW, not a fixed minute. Binance resets on wall-clock minute
boundaries, so a sliding window is strictly more conservative: it can wait when
the exchange would have allowed the request, but never the reverse. Being early
costs a little latency; being late costs the IP.

THE SERVER WINS UPWARD ONLY. Where the observed header is higher than the local
ledger, the ledger is corrected up: another process on the same IP, a retry
inside a client library, or an endpoint whose real weight differs from the
documented one are all invisible locally. Where the header is LOWER, it is
ignored -- the reported window may simply have just rolled, and treating that as
free budget is exactly how a burst becomes a ban.

418 IS FATAL. A banned IP cannot be waited out by retrying; retries extend the
ban. The governor refuses further work until the stated expiry and lets the error
reach the caller.
"""
from __future__ import annotations

import asyncio
from collections import deque
from typing import Callable, Deque, Mapping, Optional, Tuple

from loguru import logger

from ..core import clock

__all__ = [
    "WeightGovernor",
    "governed_request",
    "get_shared_governor",
    "reset_shared_governor",
    "weight_for_path",
    "DEFAULT_ENDPOINT_WEIGHT",
    "RateLimitedError",
    "IpBannedError",
    "USED_WEIGHT_HEADER",
    "ENDPOINT_WEIGHTS",
]

# Binance sends this on every /api/v3 response. "1M" is one minute.
USED_WEIGHT_HEADER = "X-MBX-USED-WEIGHT-1M"

# Documented request weights, so no call site has to guess. Keyed by path
# suffix. Values are the published spot weights; where an endpoint's weight
# varies with parameters, the LARGER value is used, because underestimating is
# what gets an IP banned.
ENDPOINT_WEIGHTS = {
    "/api/v3/exchangeInfo": 20,
    "/api/v3/klines": 2,
    "/api/v3/ticker/price": 4,       # 2 for one symbol, 4 for all
    "/api/v3/ticker/bookTicker": 4,  # 2 for one symbol, 4 for all
    "/api/v3/depth": 50,             # worst case at limit=5000
    "/api/v3/account": 20,
    "/api/v3/order": 1,
    "/api/v3/openOrders": 6,         # 3 for one symbol, 6 for all... 40 in some
    "/api/v3/userDataStream": 2,
    "/api/v3/myTrades": 20,
}

# Fallback for an endpoint not listed above. Deliberately pessimistic: an
# unknown endpoint should cost more than a known cheap one, so adding a call
# site cannot silently understate the budget.
DEFAULT_ENDPOINT_WEIGHT = 20

# How long to back off on a 429 that carries no Retry-After. Binance's window is
# a minute, so waiting out the rest of it is the only response that reliably
# clears the condition.
DEFAULT_RETRY_AFTER_SECONDS = 60.0


class RateLimitedError(RuntimeError):
    """A 429 was returned. The governor has already backed off.

    Raised rather than swallowed so the caller cannot mistake a throttled
    request for an empty market -- which is what a silent None would look like
    to the volume scanner.
    """


class IpBannedError(RuntimeError):
    """A 418 was returned: this IP is banned until a stated time.

    Fatal by design. Retrying extends the ban, and the ban also blocks the
    market-data WebSocket, so the correct behaviour is to stop and surface it.
    """


class WeightGovernor:
    def __init__(
        self,
        max_weight_per_minute: int = 6000,
        safety_fraction: float = 0.5,
        window_seconds: float = 60.0,
        now_fn: Callable[[], float] = clock.monotonic,
        sleep_fn: Callable[[float], "asyncio.Future"] = asyncio.sleep,
    ):
        if max_weight_per_minute <= 0:
            raise ValueError("max_weight_per_minute must be positive")
        if not 0 < safety_fraction <= 1:
            raise ValueError("safety_fraction must be in (0, 1]")
        if window_seconds <= 0:
            raise ValueError("window_seconds must be positive")

        self.max_weight_per_minute = max_weight_per_minute
        self.safety_fraction = safety_fraction
        self.window_seconds = window_seconds
        # Monotonic by default: a wall-clock jump (NTP, DST, a VM resuming)
        # would otherwise expire or freeze the window.
        self._now = now_fn
        self._sleep = sleep_fn

        self.ceiling = int(max_weight_per_minute * safety_fraction)
        if self.ceiling <= 0:
            raise ValueError(
                f"safety_fraction {safety_fraction} leaves no budget at "
                f"{max_weight_per_minute} weight/minute"
            )

        self._entries: Deque[Tuple[float, int]] = deque()
        self._used = 0
        self.banned_until: Optional[float] = None
        # One lock so concurrent callers cannot each read the budget, each find
        # it sufficient, and each spend it.
        self._lock = asyncio.Lock()

    # ------------------------------------------------------------------

    def _expire(self) -> None:
        """Drop entries outside the window. Bounded memory for a long-lived run."""
        cutoff = self._now() - self.window_seconds
        while self._entries and self._entries[0][0] <= cutoff:
            _, weight = self._entries.popleft()
            self._used -= weight
        if not self._entries:
            # Guards against drift if the ledger was corrected upward by a
            # header observation and then fully aged out.
            self._used = 0

    def used_weight(self) -> int:
        self._expire()
        return self._used

    def entry_count(self) -> int:
        self._expire()
        return len(self._entries)

    def _check_ban(self) -> None:
        if self.banned_until is None:
            return
        remaining = self.banned_until - self._now()
        if remaining <= 0:
            logger.warning("Exchange IP ban has expired; resuming REST requests.")
            self.banned_until = None
            return
        raise IpBannedError(
            f"this IP is banned by the exchange for another {remaining:.0f}s. "
            f"Retrying extends the ban, so no request will be issued."
        )

    # ------------------------------------------------------------------

    async def acquire(self, weight: int) -> None:
        """Block until `weight` fits in the window, then record it."""
        if weight <= 0:
            raise ValueError(f"weight must be positive, got {weight}")
        if weight > self.ceiling:
            raise ValueError(
                f"a request of weight {weight} can never fit under the ceiling "
                f"of {self.ceiling} ({self.max_weight_per_minute} x "
                f"{self.safety_fraction}). Waiting would hang forever; raise "
                f"cex.max_request_weight_per_minute or the safety fraction."
            )

        async with self._lock:
            self._check_ban()
            while True:
                self._expire()
                if self._used + weight <= self.ceiling:
                    self._entries.append((self._now(), weight))
                    self._used += weight
                    return

                # Wait for the oldest entry to leave the window. If the ledger
                # was corrected upward by a header and holds no entries, wait a
                # whole window -- there is nothing to age out.
                if self._entries:
                    wait = (self._entries[0][0] + self.window_seconds) - self._now()
                else:
                    wait = self.window_seconds
                wait = max(wait, 0.001)
                logger.debug(
                    f"Rate limit: {self._used}/{self.ceiling} weight used, "
                    f"waiting {wait:.2f}s for {weight} more."
                )
                await self._sleep(wait)

    def observe_headers(self, headers: Mapping[str, str]) -> None:
        """Correct the ledger from the exchange's own usage figure.

        Upward only -- see the module docstring. Never raises: a malformed or
        absent header must not break the request that carried it.
        """
        raw = None
        for key, value in (headers or {}).items():
            if key.upper() == USED_WEIGHT_HEADER.upper():
                raw = value
                break
        if raw is None:
            return
        try:
            reported = int(str(raw).strip())
        except (TypeError, ValueError):
            logger.debug(f"Unparseable {USED_WEIGHT_HEADER}: {raw!r}")
            return

        if reported > self._used:
            logger.debug(
                f"Rate limit: exchange reports {reported} weight used, local "
                f"ledger had {self._used}. Trusting the exchange."
            )
            # Stamped now, so the correction ages out of the window normally
            # rather than persisting forever.
            self._entries.append((self._now(), reported - self._used))
            self._used = reported

    async def handle_status(self, status: int, headers: Mapping[str, str]) -> None:
        """Interpret a response status. Raises on 429 and 418.

        Called for every response, including successful ones, so the two failure
        statuses cannot be handled in only some call sites.
        """
        if status == 418:
            retry_after = self._retry_after(headers, default=DEFAULT_RETRY_AFTER_SECONDS)
            self.banned_until = self._now() + retry_after
            logger.error(
                f"Exchange returned 418: this IP is BANNED for {retry_after:.0f}s. "
                f"No further REST requests will be issued. Retrying would extend "
                f"the ban, and the ban also blocks the market-data WebSocket."
            )
            raise IpBannedError(f"IP banned for {retry_after:.0f}s")

        if status == 429:
            retry_after = self._retry_after(headers, default=DEFAULT_RETRY_AFTER_SECONDS)
            # Treat the whole window as spent: the exchange has told us we are
            # over, so the local estimate was wrong by an unknown amount.
            self._entries.append((self._now(), max(self.ceiling - self._used, 0)))
            self._used = max(self._used, self.ceiling)
            logger.warning(
                f"Exchange returned 429 (rate limited). Backing off "
                f"{retry_after:.0f}s. Continuing to send would produce a 418 IP "
                f"ban."
            )
            await self._sleep(retry_after)
            raise RateLimitedError(f"rate limited; backed off {retry_after:.0f}s")

    @staticmethod
    def _retry_after(headers: Mapping[str, str], default: float) -> float:
        for key, value in (headers or {}).items():
            if key.lower() == "retry-after":
                try:
                    return max(float(str(value).strip()), 0.0)
                except (TypeError, ValueError):
                    break
        return default


_shared: Optional["WeightGovernor"] = None


def get_shared_governor(
    max_weight_per_minute: int = 6000,
    safety_fraction: float = 0.5,
) -> "WeightGovernor":
    """The one governor for this process.

    A per-client governor is only half a fix: the exchange's limit is per IP, so
    three clients with three private budgets of 3000 can spend 9000 against a
    6000 ceiling and get the IP banned while each one believes it stayed inside
    its allowance. A process has one outbound address, so process scope is
    exactly the scope of the constraint.

    Conflicting configurations resolve to the STRICTER ceiling, and weight
    already spent is preserved. Adopting a later, larger ceiling would raise the
    limit after an earlier caller had already sized its behaviour to the smaller
    one; discarding spent weight would hand a late-starting component a budget it
    has not earned.
    """
    global _shared
    if _shared is None:
        _shared = WeightGovernor(
            max_weight_per_minute=max_weight_per_minute,
            safety_fraction=safety_fraction,
        )
        logger.info(
            f"REST weight governor: {_shared.ceiling} of "
            f"{max_weight_per_minute} weight/minute available to this process "
            f"({safety_fraction:.0%} of the exchange limit)."
        )
        return _shared

    requested = int(max_weight_per_minute * safety_fraction)
    if requested < _shared.ceiling:
        logger.warning(
            f"Tightening the shared REST weight ceiling from {_shared.ceiling} "
            f"to {requested}: a component asked for a stricter budget, and the "
            f"stricter of two disagreeing limits is the safe one."
        )
        _shared.max_weight_per_minute = max_weight_per_minute
        _shared.safety_fraction = safety_fraction
        _shared.ceiling = requested
    elif requested > _shared.ceiling:
        logger.warning(
            f"Ignoring a request to raise the shared REST weight ceiling from "
            f"{_shared.ceiling} to {requested}. The limit is per IP, and the "
            f"stricter figure already in force is the safe one."
        )
    return _shared


def reset_shared_governor() -> None:
    """Discard the process governor. For tests; never call this in production.

    Resetting in a live process would hand it a fresh budget while the exchange's
    own window still holds the weight already spent -- which is how a burst
    becomes a ban.
    """
    global _shared
    _shared = None


async def governed_request(
    session,
    governor: "WeightGovernor",
    method: str,
    url: str,
    weight: Optional[int] = None,
    **kwargs,
):
    """Issue one REST request through the shared budget and return its JSON.

    The single chokepoint. Every REST call in the codebase goes through here, so
    the accounting cannot be bypassed by adding a call site -- which is exactly
    how the previous arrangement failed: four modules, four sessions, four
    independent beliefs about how much budget was available, and one shared IP.

    Weight is charged BEFORE the request, from the documented table, and the
    response's own usage header is fed back afterwards. Charging first is what
    makes a burst wait rather than discover the limit by being banned.
    """
    charge = weight_for_path(url) if weight is None else weight
    await governor.acquire(charge)

    request = getattr(session, method.lower())
    async with request(url, **kwargs) as response:
        # Read the accounting before anything can raise, so a throttled or
        # failed request still updates the budget.
        governor.observe_headers(response.headers)
        await governor.handle_status(response.status, response.headers)
        response.raise_for_status()
        return await response.json()


def weight_for_path(path: str) -> int:
    """The documented weight for an endpoint, or a pessimistic default.

    Matching on suffix so a full URL or a bare path both work, and so a new call
    site cannot understate its cost by passing the wrong form.
    """
    for endpoint, weight in ENDPOINT_WEIGHTS.items():
        if path.endswith(endpoint) or endpoint in path:
            return weight
    logger.debug(
        f"No documented weight for {path}; charging the pessimistic default "
        f"{DEFAULT_ENDPOINT_WEIGHT}."
    )
    return DEFAULT_ENDPOINT_WEIGHT
