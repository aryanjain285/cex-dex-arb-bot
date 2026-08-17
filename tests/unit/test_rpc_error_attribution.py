"""An RPC failure must not be recorded as an absent market.

`get_quote` ended in `except Exception: return None`, and the detector records a
`None` quote as `no_dex_quote`. So four completely different situations produced
the same audit row:

    the pool genuinely has no liquidity
    the node returned 429 Too Many Requests
    the node timed out
    the ABI is wrong

Discovered by hitting it: a universe survey against public RPC endpoints drew
sustained 429s, and every one was logged at debug level and reported upward as "no
pool". Under RPC pressure a live bot would quietly stop finding opportunities
while its own dataset said the market was empty -- and the dataset is the thing
we intend to make decisions from.

The distinction is not cosmetic. "No liquidity here" means stop watching this
pair. "We are being throttled" means slow down, or pay for a better node. Acting
on the first when the second is true is how a working strategy gets abandoned.
"""
from decimal import Decimal

import pytest

from src.core.config import RotationConfig, StrategyConfig, TokenPolicyConfig
from src.exchange.errors import RpcError, classify_rpc_failure
from src.strategy.detector import OpportunityDetector, RejectionReason
from tests.fakes import FakeCex, flat_book, make_pair


def D(x) -> Decimal:
    return Decimal(str(x))


# --- classification ------------------------------------------------------


@pytest.mark.parametrize("message", [
    "429 Client Error: Too Many Requests for url: https://mainnet.base.org/",
    "Too Many Requests",
    "503 Server Error: Service Unavailable",
    "502 Bad Gateway",
    "504 Gateway Timeout",
    "HTTPSConnectionPool(host='x', port=443): Read timed out.",
    "Max retries exceeded with url",
    "Connection aborted",
    "Cannot connect to host mainnet.base.org:443",
])
def test_transport_failures_are_classified_as_rpc_problems(message):
    assert classify_rpc_failure(Exception(message)) is True, message


@pytest.mark.parametrize("message", [
    "execution reverted",
    "execution reverted: STF",
    "Could not decode contract function call",
    "insufficient funds for gas * price + value",
    "abi is missing a function",
])
def test_contract_level_failures_are_not_rpc_problems(message):
    """A revert is information about the chain, not about our connection to it.
    Misclassifying it would hide a genuinely empty pool behind a retry."""
    assert classify_rpc_failure(Exception(message)) is False, message


def test_a_timeout_type_is_recognised_without_a_message():
    assert classify_rpc_failure(TimeoutError()) is True


def test_an_unknown_failure_is_not_assumed_to_be_transport():
    """Defaulting to "RPC problem" would relabel every genuine revert as
    infrastructure and hide real data issues behind an operational excuse."""
    assert classify_rpc_failure(ValueError("something unexpected")) is False


# --- the detector distinguishes them ------------------------------------


class ThrottledDex:
    """A DEX client whose node is rate-limiting."""

    def __init__(self):
        self.calls = 0

    async def get_pool_address(self, *a, **k):
        return None

    async def get_quote(self, pair, size, side, estimate_gas=False):
        self.calls += 1
        raise RpcError(
            "429 Client Error: Too Many Requests for url: https://mainnet.base.org/"
        )


class EmptyDex:
    """A DEX client whose pool genuinely has nothing in it."""

    async def get_pool_address(self, *a, **k):
        return None

    async def get_quote(self, pair, size, side, estimate_gas=False):
        return None


class Recorder:
    def __init__(self):
        self.rows = []

    def record(self, r):
        self.rows.append(r)
        return len(self.rows)


def _strategy() -> StrategyConfig:
    return StrategyConfig(
        target_notional_usd=1000, taker_fee_bps=D("7.5"), min_net_bps=D(5),
        rotation=RotationConfig(enabled=False),
        token_policy=TokenPolicyConfig(mode="denylist"),
        dex_routing={"enabled": False},
    )


async def test_a_throttled_rpc_is_recorded_as_an_rpc_error():
    rec = Recorder()
    det = OpportunityDetector(
        _strategy(), FakeCex({"ETH/USDT": flat_book(bid=1000, ask=1000)}),
        ThrottledDex(), [make_pair()], store=rec,
    )

    found = await det.detect()

    assert found == []
    assert rec.rows, "the failure must still be recorded"
    reasons = {r.reason for r in rec.rows}
    assert reasons == {RejectionReason.RPC_ERROR}, (
        f"a throttled RPC was recorded as {reasons}"
    )


async def test_an_empty_pool_is_still_recorded_as_no_dex_quote():
    """The other half: the two must remain distinguishable in both directions."""
    rec = Recorder()
    det = OpportunityDetector(
        _strategy(), FakeCex({"ETH/USDT": flat_book(bid=1000, ask=1000)}),
        EmptyDex(), [make_pair()], store=rec,
    )

    await det.detect()

    reasons = {r.reason for r in rec.rows}
    assert reasons == {RejectionReason.NO_DEX_QUOTE}, reasons


async def test_an_rpc_failure_does_not_stop_the_other_pairs():
    """One chain being throttled must not take down pairs on another chain."""
    rec = Recorder()

    class PerChain:
        async def get_pool_address(self, *a, **k):
            return None

        async def get_quote(self, pair, size, side, estimate_gas=False):
            from src.core.types import DexQuote
            if pair.dex_chain == "base":
                raise RpcError("429 Too Many Requests")
            return DexQuote(price=D(1100), gas_cost_quote=D(0))

    det = OpportunityDetector(
        _strategy(),
        FakeCex({
            "ETH/USDT": flat_book(bid=1000, ask=1000),
            "ARB/USDT": flat_book(bid=1000, ask=1000),
        }),
        PerChain(),
        [make_pair("ETH/USDT"),
         make_pair("ARB/USDT", base="ARB", dex_chain="base")],
        store=rec,
    )

    found = await det.detect()

    assert found, "the healthy chain's pair must still produce an opportunity"
    by_pair = {}
    for row in rec.rows:
        by_pair.setdefault(row.cex_symbol, set()).add(row.reason)
    assert RejectionReason.RPC_ERROR in by_pair["ARB/USDT"]
    assert RejectionReason.RPC_ERROR not in by_pair.get("ETH/USDT", set())


async def test_the_rpc_error_reason_is_a_distinct_metric_label():
    """Prometheus must be able to separate them too: the alert for "we are being
    throttled" is not the alert for "this pair is dead"."""
    from src.infra import metrics

    def read(reason):
        return float(
            metrics.evaluations_total.labels(
                pair="ETH/USDT", direction="CEX_to_DEX",
                outcome="rejected", reason=reason,
            )._value.get()
        )

    before = read(RejectionReason.RPC_ERROR)

    det = OpportunityDetector(
        _strategy(), FakeCex({"ETH/USDT": flat_book(bid=1000, ask=1000)}),
        ThrottledDex(), [make_pair()],
    )
    await det.detect()

    assert read(RejectionReason.RPC_ERROR) == before + 1
