"""A batched pool read must produce exactly what the unbatched one produced.

Batching is a pure performance change, so the only acceptable outcome is an
identical snapshot. The test is therefore a differential one: read the same fake
chain twice, once through multicall and once call-by-call, and require the two
snapshots to be equal AND to price identically.

The failure this guards against is not "batching is broken" -- that would be
obvious. It is "batching works, but the tick list is off by one position", which
leaves small swaps exact (they never leave the current range) and every large swap
wrong. Equality of the tick list is checked directly for that reason.
"""
from decimal import Decimal

import pytest

from src.exchange.pool_state import fetch_pool_state


class FakeChain:
    """A pool with a known tick layout, served either singly or in batches.

    The point of encoding real values here rather than returning constants is that
    a positional bug produces a WRONG tick list rather than an obviously empty one.
    """

    TICKS = {
        -600: 111,
        -300: 222,
        -100: 333,
        100: -333,
        300: -222,
        600: -111,
        # An initialised tick with zero net liquidity: legal on chain, and it must be
        # dropped from the snapshot rather than stored as a no-op crossing.
        900: 0,
    }

    def __init__(self, tick_spacing=100, current_tick=0, use_multicall=True):
        self.tick_spacing = tick_spacing
        self.current_tick = current_tick
        self.use_multicall = use_multicall
        self.single_calls = 0
        self.batch_calls = 0

    # -- the surface fetch_pool_state uses ------------------------------

    @property
    def erc20_abi(self):
        return [{"inputs": [], "name": "decimals",
                 "outputs": [{"name": "", "type": "uint8"}],
                 "stateMutability": "view", "type": "function"}]

    def _get_w3(self, chain):
        return _FakeW3(self)

    async def _rpc(self, chain, fn):
        self.single_calls += 1
        return fn()


class _FakeEth:
    def __init__(self, chain):
        self._chain = chain
        self.block_number = 1234

    def contract(self, address=None, abi=None):
        return _FakeContract(self._chain, abi)

    def get_code(self, address, block_identifier=None):
        return b"\x60\x60" if self._chain.use_multicall else b""


class _FakeCodec:
    """The fake encodes calls as (kind, argument) tuples rather than ABI bytes, so
    decoding is the identity. That keeps this test about batching and alignment
    instead of about eth-abi, which has its own tests."""

    @staticmethod
    def decode(types, data):
        if data and data[0] == "bitmap":
            return (data[2],)
        if data and data[0] == "ticks":
            return (0, data[2], 0, 0, 0, 0, 0, True)
        raise AssertionError(f"unexpected payload {data!r}")


class _FakeW3:
    def __init__(self, chain):
        self.eth = _FakeEth(chain)
        self.codec = _FakeCodec()

    @staticmethod
    def to_checksum_address(address):
        return address


class _FakeFunctions:
    def __init__(self, chain, abi):
        self._chain = chain
        self._abi = abi

    def slot0(self):
        # sqrtPriceX96 for tick 0, tick 0.
        return _Callable((79228162514264337593543950336, self._chain.current_tick,
                          0, 0, 0, 0, True))

    def liquidity(self):
        return _Callable(10 ** 20)

    def fee(self):
        return _Callable(500)

    def tickSpacing(self):
        return _Callable(self._chain.tick_spacing)

    def token0(self):
        return _Callable("0x" + "11" * 20)

    def token1(self):
        return _Callable("0x" + "22" * 20)

    def decimals(self):
        return _Callable(18)

    def tickBitmap(self, word):
        return _Callable(self._chain_bitmap(word))

    def _chain_bitmap(self, word):
        spacing = self._chain.tick_spacing
        bits = 0
        for tick in self._chain.TICKS:
            if tick % spacing:
                continue
            compressed = tick // spacing
            if compressed >> 8 == word:
                bits |= 1 << (compressed & 0xFF)
        return bits

    def ticks(self, tick):
        net = self._chain.TICKS.get(tick, 0)
        return _Callable((0, net, 0, 0, 0, 0, 0, tick in self._chain.TICKS))

    def aggregate3(self, payload):
        """Serve a batch by decoding each call's marker and answering it.

        The fake encodes calls as (kind, argument) rather than real ABI bytes, which
        keeps the test about position and alignment instead of about eth-abi.
        """
        self._chain.batch_calls += 1
        out = []
        for _target, _allow, data in payload:
            kind, argument = data
            if kind == "ticks":
                net = self._chain.TICKS.get(argument, 0)
                out.append((True, ("ticks", argument, net)))
            elif kind == "bitmap":
                out.append((True, ("bitmap", argument, self._chain_bitmap(argument))))
            else:  # pragma: no cover - a new call kind must not pass silently
                out.append((False, b""))
        return _Callable(out)


class _FakeContract:
    def __init__(self, chain, abi):
        self.functions = _FakeFunctions(chain, abi)
        self.address = "0x" + "ab" * 20

    def encode_abi(self, abi_element_identifier=None, args=None):
        """Calldata as a marker tuple. A real encoding would be opaque to the fake
        chain, which has to answer the call."""
        return (
            "bitmap" if abi_element_identifier == "tickBitmap" else "ticks",
            args[0],
        )


class _Callable:
    def __init__(self, value):
        self._value = value

    def call(self, block_identifier=None):
        return self._value

    def __call__(self, *args, **kwargs):
        return self._value


@pytest.mark.asyncio
async def test_batched_and_unbatched_reads_agree_exactly():
    batched = FakeChain(use_multicall=True)
    single = FakeChain(use_multicall=False)

    a = await fetch_pool_state(batched, "ethereum", "0x" + "ab" * 20,
                              decimals0=18, decimals1=6, tick_range=10)
    b = await fetch_pool_state(single, "ethereum", "0x" + "ab" * 20,
                               decimals0=18, decimals1=6, tick_range=10)

    assert [(t.tick, t.liquidity_net) for t in a.ticks] == \
           [(t.tick, t.liquidity_net) for t in b.ticks], (
        "the batched read produced a different tick list; a positional bug here "
        "leaves small swaps exact and every large swap wrong"
    )
    assert (a.known_lower_tick, a.known_upper_tick) == \
           (b.known_lower_tick, b.known_upper_tick)
    assert a.liquidity == b.liquidity
    assert a.sqrt_price_x96 == b.sqrt_price_x96

    for size in (Decimal("0.001"), Decimal("1"), Decimal("100")):
        assert (a.price_for_amount_in(size, zero_for_one=True)
                == b.price_for_amount_in(size, zero_for_one=True))


@pytest.mark.asyncio
async def test_batching_uses_far_fewer_round_trips():
    """The reason for the change. If the saving is not real, the risk is not worth
    taking."""
    batched = FakeChain(use_multicall=True)
    single = FakeChain(use_multicall=False)

    await fetch_pool_state(batched, "ethereum", "0x" + "ab" * 20,
                           decimals0=18, decimals1=6, tick_range=10)
    await fetch_pool_state(single, "ethereum", "0x" + "ab" * 20,
                           decimals0=18, decimals1=6, tick_range=10)

    assert batched.single_calls < single.single_calls


@pytest.mark.asyncio
async def test_a_zero_net_tick_is_dropped_by_both_paths():
    """Tick 900 is initialised with zero net liquidity. Storing it would add a
    crossing that changes nothing, which is harmless -- but the two paths must agree,
    or the batched snapshot is not a drop-in replacement."""
    batched = FakeChain(use_multicall=True)
    a = await fetch_pool_state(batched, "ethereum", "0x" + "ab" * 20,
                               decimals0=18, decimals1=6, tick_range=10)
    assert 900 not in [t.tick for t in a.ticks]


@pytest.mark.asyncio
async def test_a_chain_without_multicall_still_reads():
    """The fallback must be real, not theoretical: a chain without Multicall3
    deployed has to keep working, slowly."""
    single = FakeChain(use_multicall=False)
    snapshot = await fetch_pool_state(single, "ethereum", "0x" + "ab" * 20,
                                      decimals0=18, decimals1=6, tick_range=10)
    assert len(snapshot.ticks) == 6
    assert single.batch_calls == 0
