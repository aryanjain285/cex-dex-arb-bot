"""Every REST call must go through the governor.

A governor that exists but is bypassed by one call site protects nothing: the
budget is per IP, so a single unaccounted burst bans the whole process --
including the market-data WebSocket.

These tests do not mock the governor. They give it a budget of one unit and then
check that the call site actually blocks, which is the only observable proof that
the request went through the chokepoint rather than around it.
"""
import asyncio

import pytest

from src.exchange.rate_limit import (
    ENDPOINT_WEIGHTS, DEFAULT_ENDPOINT_WEIGHT, IpBannedError, WeightGovernor,
    weight_for_path,
)


class FakeResponse:
    def __init__(self, payload, status=200, headers=None):
        self._payload = payload
        self.status = status
        self.headers = headers or {}

    async def json(self, **kwargs):
        return self._payload

    def raise_for_status(self):
        if self.status >= 400:
            raise AssertionError(f"unexpected status {self.status}")

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


class RecordingSession:
    """Counts requests and returns a canned payload."""

    def __init__(self, payload, status=200, headers=None):
        self.payload = payload
        self.status = status
        self.headers = headers or {}
        self.requests = []

    def get(self, url, **kwargs):
        self.requests.append(("GET", url, kwargs))
        return FakeResponse(self.payload, self.status, self.headers)

    def post(self, url, **kwargs):
        self.requests.append(("POST", url, kwargs))
        return FakeResponse(self.payload, self.status, self.headers)


# --- the weight table ----------------------------------------------------


def test_documented_weights_cover_the_endpoints_actually_called():
    """Every endpoint this codebase calls must have a stated weight.

    An endpoint that falls through to the default is charged pessimistically,
    which is safe but wasteful -- and if the default were ever lowered, an
    unlisted endpoint would silently understate its cost.
    """
    called = [
        "/api/v3/exchangeInfo",
        "/api/v3/klines",
        "/api/v3/ticker/price",
        "/api/v3/ticker/bookTicker",
        "/api/v3/account",
        "/api/v3/order",
        "/api/v3/userDataStream",
    ]
    missing = [e for e in called if e not in ENDPOINT_WEIGHTS]
    assert not missing, f"no documented weight for {missing}"


def test_a_full_url_resolves_to_the_same_weight_as_a_bare_path():
    assert weight_for_path("/api/v3/klines") == ENDPOINT_WEIGHTS["/api/v3/klines"]
    assert weight_for_path("https://api.binance.com/api/v3/klines") == (
        ENDPOINT_WEIGHTS["/api/v3/klines"]
    )


def test_an_unknown_endpoint_is_charged_pessimistically():
    """Cheap-by-default would let a new call site understate its cost."""
    weight = weight_for_path("/api/v3/somethingNew")
    assert weight == DEFAULT_ENDPOINT_WEIGHT
    assert weight >= max(ENDPOINT_WEIGHTS["/api/v3/klines"],
                         ENDPOINT_WEIGHTS["/api/v3/order"])


# --- the governed request helper -----------------------------------------


async def test_the_helper_charges_the_endpoints_weight():
    from src.exchange.rate_limit import governed_request

    gov = WeightGovernor(max_weight_per_minute=1000, safety_fraction=1.0)
    session = RecordingSession({"ok": True})

    await governed_request(session, gov, "GET", "https://x/api/v3/klines")

    assert gov.used_weight() == ENDPOINT_WEIGHTS["/api/v3/klines"]
    assert len(session.requests) == 1


async def test_the_helper_feeds_the_response_header_back_to_the_governor():
    from src.exchange.rate_limit import governed_request

    gov = WeightGovernor(max_weight_per_minute=1000, safety_fraction=1.0)
    session = RecordingSession({"ok": True},
                               headers={"X-MBX-USED-WEIGHT-1M": "500"})

    await governed_request(session, gov, "GET", "https://x/api/v3/klines")

    assert gov.used_weight() == 500, (
        "the exchange's usage figure was read but not applied"
    )


async def test_the_helper_surfaces_a_ban_rather_than_returning_data():
    from src.exchange.rate_limit import governed_request

    gov = WeightGovernor(max_weight_per_minute=1000, safety_fraction=1.0)
    session = RecordingSession({"ok": True}, status=418,
                               headers={"Retry-After": "120"})

    with pytest.raises(IpBannedError):
        await governed_request(session, gov, "GET", "https://x/api/v3/klines")


async def test_the_helper_blocks_when_the_budget_is_exhausted():
    """The proof that the chokepoint is real: with a budget below one call's
    weight, the request cannot be issued at all."""
    from src.exchange.rate_limit import governed_request

    gov = WeightGovernor(max_weight_per_minute=1, safety_fraction=1.0)
    session = RecordingSession({"ok": True})

    with pytest.raises(ValueError, match="never fit|ceiling"):
        await governed_request(session, gov, "GET", "https://x/api/v3/klines")

    assert session.requests == [], "a request was issued despite no budget"


async def test_a_second_call_waits_for_the_window():
    """Two callers, one budget."""
    from src.exchange.rate_limit import governed_request

    gov = WeightGovernor(max_weight_per_minute=2, safety_fraction=1.0,
                         window_seconds=0.05)
    session = RecordingSession({"ok": True})

    await governed_request(session, gov, "GET", "https://x/api/v3/klines")
    await governed_request(session, gov, "GET", "https://x/api/v3/klines")

    assert len(session.requests) == 2
    # The second could only have proceeded after the window rolled.
    assert gov.used_weight() <= 2


# --- the clients hold a governor ----------------------------------------


def test_the_live_cex_client_owns_a_governor():
    """The client that places orders is the one that must never be banned."""
    import inspect

    from src.exchange import binance

    source = inspect.getsource(binance)
    assert "WeightGovernor" in source or "governed_request" in source, (
        "the live Binance client does not use the rate-limit governor"
    )


def test_the_scanners_route_through_the_governor():
    """The scanners generate the bursts, so they matter most.

    Checked by source inspection rather than by call, because constructing them
    requires a live session; the behavioural coverage is on the governor itself.
    """
    import inspect

    from src.scanner import autodiscovery, spike, volume

    for module in (spike, volume, autodiscovery):
        source = inspect.getsource(module)
        assert "governed_request" in source or "governor" in source, (
            f"{module.__name__} issues REST requests outside the governor"
        )
