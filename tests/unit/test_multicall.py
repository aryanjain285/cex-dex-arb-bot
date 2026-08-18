"""Batched eth_call, where the only thing that really matters is ORDER.

Why this exists: a full pool read is roughly one RPC call per initialised tick.
Measured against public endpoints, that is 170-215 calls taking 60-160 seconds --
1 to 3 requests per second achieved against 8 configured. The pool cache also
re-reads ticks every 120 seconds to catch mints and burns, so 22 pools at that cost
would spend the entire run doing full reads and record almost nothing. Batching
turns ~200 calls into ~3.

The danger it introduces is silent. Each result is matched to its request by
POSITION, and the requests are `ticks(t)` for a list of ticks. Drop one result,
reorder them, or mis-handle a partial batch, and every tick is paired with another
tick's liquidityNet. Nothing raises. The pool still prices. Small swaps stay right
because they never leave the current range, so a smoke test passes -- and every
large-size quote, which is the whole point of having the tick data, is wrong.

So these tests are mostly about position: order within a batch, order across batch
boundaries, and what a failed sub-call does to the alignment of everything after it.
"""
import pytest

from src.exchange.multicall import (
    MULTICALL3_ADDRESS,
    Multicall,
    chunk_calls,
)


class FakeW3:
    """Enough web3 surface to exercise batching without a chain."""

    class _Eth:
        def __init__(self, outer):
            self._outer = outer

        def get_code(self, address, block_identifier=None):
            return self._outer.code

    def __init__(self, code=b"\x60\x60", handler=None):
        self.code = code
        self.eth = FakeW3._Eth(self)
        self.handler = handler


class FakeClient:
    """Records every batch it was asked to send, in order."""

    def __init__(self, code=b"\x60\x60", results=None, fail_batches=()):
        self._w3 = FakeW3(code=code)
        self.batches = []
        self.results = results
        self.fail_batches = set(fail_batches)
        self.rpc_calls = 0

    def _get_w3(self, chain):
        return self._w3

    async def _rpc(self, chain, fn):
        self.rpc_calls += 1
        return fn()


def _calls(n, target="0x" + "11" * 20):
    """n distinct calls whose payload encodes their own index, so a reordering is
    detectable rather than merely suspected."""
    return [(target, i.to_bytes(32, "big")) for i in range(n)]


class TestChunking:
    def test_calls_are_split_into_batches_of_at_most_the_limit(self):
        batches = list(chunk_calls(_calls(250), 100))
        assert [len(b) for b in batches] == [100, 100, 50]

    def test_chunking_preserves_order_exactly(self):
        original = _calls(250)
        rejoined = [call for batch in chunk_calls(original, 100) for call in batch]
        assert rejoined == original

    def test_an_empty_list_produces_no_batches(self):
        assert list(chunk_calls([], 100)) == []

    def test_a_batch_size_below_one_is_rejected(self):
        with pytest.raises(ValueError):
            list(chunk_calls(_calls(5), 0))


class TestAvailability:
    @pytest.mark.asyncio
    async def test_a_chain_with_the_contract_deployed_is_available(self):
        multicall = Multicall(FakeClient(code=b"\x60\x60\x60"))
        assert await multicall.available("ethereum") is True

    @pytest.mark.asyncio
    async def test_a_chain_without_the_contract_is_unavailable(self):
        """Empty code at the address. Reporting available and then decoding an empty
        return would produce zeros -- a pool with no liquidity anywhere, which is a
        plausible-looking and completely wrong answer."""
        multicall = Multicall(FakeClient(code=b""))
        assert await multicall.available("ethereum") is False

    @pytest.mark.asyncio
    async def test_availability_is_checked_once_per_chain(self):
        client = FakeClient(code=b"\x60")
        multicall = Multicall(client)
        await multicall.available("ethereum")
        await multicall.available("ethereum")
        assert client.rpc_calls == 1, (
            "the deployment check is a constant; repeating it per pool read would "
            "spend the budget this class exists to save"
        )

    @pytest.mark.asyncio
    async def test_each_chain_is_checked_separately(self):
        client = FakeClient(code=b"\x60")
        multicall = Multicall(client)
        await multicall.available("ethereum")
        await multicall.available("arbitrum")
        assert client.rpc_calls == 2

    @pytest.mark.asyncio
    async def test_a_failed_check_reports_unavailable_rather_than_raising(self):
        class Broken(FakeClient):
            async def _rpc(self, chain, fn):
                raise RuntimeError("endpoint down")

        multicall = Multicall(Broken())
        assert await multicall.available("ethereum") is False


class TestOrdering:
    """The section that protects the tick data."""

    @pytest.mark.asyncio
    async def test_results_come_back_in_request_order(self, monkeypatch):
        client = FakeClient()
        multicall = Multicall(client, max_batch=1000)
        # The stub echoes each call's payload back as its result, so position can be
        # verified rather than assumed.
        multicall._send_batch = _echo_batch(multicall)

        calls = _calls(10)
        results = await multicall.aggregate("ethereum", calls)
        assert results == [payload for _, payload in calls]

    @pytest.mark.asyncio
    async def test_order_is_preserved_across_batch_boundaries(self):
        client = FakeClient()
        multicall = Multicall(client, max_batch=7)
        multicall._send_batch = _echo_batch(multicall)

        calls = _calls(30)
        results = await multicall.aggregate("ethereum", calls)
        assert results == [payload for _, payload in calls], (
            "results were reordered across batches; every tick would be paired "
            "with another tick's liquidity"
        )

    @pytest.mark.asyncio
    async def test_a_failed_sub_call_becomes_None_in_place(self):
        """Not omitted. An omission shifts every subsequent result by one, which is
        the exact corruption this file is about."""
        client = FakeClient()
        multicall = Multicall(client, max_batch=1000)
        multicall._send_batch = _echo_batch(multicall, fail_indices={3})

        calls = _calls(6)
        results = await multicall.aggregate("ethereum", calls)
        assert len(results) == 6
        assert results[3] is None
        assert results[4] == calls[4][1], "results after a failure shifted position"

    @pytest.mark.asyncio
    async def test_a_short_batch_response_is_an_error_not_a_silent_truncation(self):
        """A node returning fewer results than requested must not be interpreted as
        'the rest failed' -- that is indistinguishable from a misalignment, and the
        safe response is to refuse the whole read."""
        client = FakeClient()
        multicall = Multicall(client, max_batch=1000)

        async def _short(chain, batch, block_number=None):
            return [(True, b"\x01")] * (len(batch) - 1)

        multicall._send_batch = _short
        with pytest.raises(ValueError, match="returned"):
            await multicall.aggregate("ethereum", _calls(5))

    @pytest.mark.asyncio
    async def test_no_calls_means_no_rpc_at_all(self):
        client = FakeClient()
        multicall = Multicall(client)
        assert await multicall.aggregate("ethereum", []) == []
        assert client.rpc_calls == 0


class TestBlockPinning:
    @pytest.mark.asyncio
    async def test_the_block_is_passed_to_every_batch(self):
        """All calls in one pool read must see the same block, or the snapshot mixes
        a new price with old ticks -- a state the pool never had."""
        seen = []
        client = FakeClient()
        multicall = Multicall(client, max_batch=3)

        async def _capture(chain, batch, block_number=None):
            seen.append(block_number)
            return [(True, payload) for _, payload in batch]

        multicall._send_batch = _capture
        await multicall.aggregate("ethereum", _calls(10), block_number=25_779_036)
        assert seen == [25_779_036] * 4


def _echo_batch(multicall, fail_indices=frozenset()):
    """A stub `_send_batch` that echoes payloads, optionally failing some positions.

    Indices are GLOBAL across batches, so a batching bug that resets them shows up.
    """
    state = {"offset": 0}

    async def _send(chain, batch, block_number=None):
        out = []
        for i, (_, payload) in enumerate(batch):
            index = state["offset"] + i
            out.append((False, b"") if index in fail_indices else (True, payload))
        state["offset"] += len(batch)
        return out

    return _send
