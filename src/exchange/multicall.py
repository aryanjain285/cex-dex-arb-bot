"""Batched eth_call via Multicall3, because a full pool read is ~200 round trips.

Measured, not assumed: a full read of a deep pool is one RPC call per initialised
tick plus one per bitmap word -- 170 to 215 calls, taking 60 to 160 seconds against
public endpoints. That is 1 to 3 requests per second achieved against 8 configured,
so the limiter is not the constraint; the endpoint is. Twenty-two pools at that cost
cannot be recorded at all, and the pool cache's 120-second tick re-read makes it
worse: the recorder would spend its entire run doing full reads.

Batching turns ~200 calls into ~3.

THE RISK IT INTRODUCES IS SILENT. Each result is matched to its request by POSITION,
and the requests are `ticks(t)` for a list of ticks. Drop a result, reorder them, or
mishandle a partial batch, and every tick is paired with another tick's liquidityNet.
Nothing raises. The pool still prices, and small swaps stay exactly right because
they never leave the current range -- so a smoke test passes while every large-size
quote, the entire reason for reading ticks, is wrong.

Hence: a failed sub-call becomes None IN PLACE rather than being omitted, and a
response with the wrong length is an error rather than a truncation. Refusing a read
costs one snapshot. Misaligning one corrupts the dataset.

Multicall3 is deployed at the same address on Ethereum, Arbitrum, Base and BSC, but
deployment is verified per chain before use -- decoding an empty return from a
missing contract yields zeros, which reads as a pool with no liquidity anywhere:
plausible-looking and entirely wrong.
"""
from __future__ import annotations

from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

from loguru import logger

__all__ = [
    "MULTICALL3_ADDRESS",
    "MULTICALL3_ABI",
    "Multicall",
    "chunk_calls",
    "DEFAULT_MAX_BATCH",
]

# Deterministic-deployment address, identical across Ethereum, Arbitrum, Base, BSC
# and most EVM chains.
MULTICALL3_ADDRESS = "0xcA11bde05977b3631167028862bE2a173976CA11"

# aggregate3 only: it allows per-call failure, which `aggregate` does not. A single
# reverting `ticks()` call must not take the whole batch down, because an uninitialised
# tick is a normal thing to ask about.
MULTICALL3_ABI = [
    {
        "inputs": [{
            "components": [
                {"name": "target", "type": "address"},
                {"name": "allowFailure", "type": "bool"},
                {"name": "callData", "type": "bytes"},
            ],
            "name": "calls",
            "type": "tuple[]",
        }],
        "name": "aggregate3",
        "outputs": [{
            "components": [
                {"name": "success", "type": "bool"},
                {"name": "returnData", "type": "bytes"},
            ],
            "name": "returnData",
            "type": "tuple[]",
        }],
        "stateMutability": "payable",
        "type": "function",
    }
]

# Calls per batch. Bounded by the node's response size and gas limit for an
# eth_call, not by the contract. 200 x 32-byte returns is ~7KB, comfortably inside
# any provider's limit, and keeps a single failed batch cheap to retry.
DEFAULT_MAX_BATCH = 200

Call = Tuple[str, bytes]


def chunk_calls(calls: Sequence[Call], max_batch: int) -> Iterator[List[Call]]:
    """Split into batches, preserving order exactly."""
    if max_batch < 1:
        raise ValueError(f"max_batch must be at least 1, got {max_batch}")
    for start in range(0, len(calls), max_batch):
        yield list(calls[start:start + max_batch])


class Multicall:
    def __init__(
        self,
        client,
        address: str = MULTICALL3_ADDRESS,
        max_batch: int = DEFAULT_MAX_BATCH,
    ):
        self.client = client
        self.address = address
        self.max_batch = max_batch
        # Deployment status per chain. A constant, so checked once -- repeating it
        # per pool read would spend the budget this class exists to save.
        self._available: Dict[str, bool] = {}

    async def available(self, chain: str) -> bool:
        """Is Multicall3 deployed on this chain?

        Any failure answers False. A check that raised would make the caller choose
        between crashing and assuming, and assuming is how a missing contract
        becomes a pool with no liquidity.
        """
        cached = self._available.get(chain)
        if cached is not None:
            return cached
        try:
            w3 = self.client._get_w3(chain)
            code = await self.client._rpc(
                chain, lambda: w3.eth.get_code(self._checksum(w3, self.address))
            )
            ok = bool(code) and len(code) > 0
        except Exception as exc:  # noqa: BLE001 - unavailability is the answer
            logger.debug(f"multicall unavailable on {chain}: {exc}")
            ok = False
        self._available[chain] = ok
        if ok:
            logger.debug(f"multicall available on {chain} at {self.address}")
        return ok

    @staticmethod
    def _checksum(w3, address: str):
        try:
            return w3.to_checksum_address(address)
        except Exception:
            return address

    async def aggregate(
        self,
        chain: str,
        calls: Sequence[Call],
        block_number: Optional[int] = None,
    ) -> List[Optional[bytes]]:
        """Return one raw return-value per call, in request order.

        None marks a sub-call that reverted. Never omitted: an omission shifts every
        subsequent result by one position, which is exactly the misalignment that
        would pair each tick with another tick's liquidity.
        """
        if not calls:
            return []

        results: List[Optional[bytes]] = []
        for batch in chunk_calls(calls, self.max_batch):
            raw = await self._send_batch(chain, batch, block_number=block_number)
            if len(raw) != len(batch):
                # Not treated as "the rest failed". A short response is
                # indistinguishable from a misalignment, and the only safe reading
                # is that this batch cannot be trusted at all.
                raise ValueError(
                    f"multicall on {chain} returned {len(raw)} results for "
                    f"{len(batch)} calls; refusing the read rather than guessing "
                    f"which results belong to which calls"
                )
            for success, data in raw:
                # Passed through unchanged. web3 already decodes aggregate3's
                # `bytes` output to bytes, and coercing here only breaks callers
                # that substitute the codec.
                results.append(data if success else None)
        return results

    async def _send_batch(
        self,
        chain: str,
        batch: Sequence[Call],
        block_number: Optional[int] = None,
    ) -> List[Tuple[bool, bytes]]:
        w3 = self.client._get_w3(chain)
        contract = w3.eth.contract(
            address=self._checksum(w3, self.address), abi=MULTICALL3_ABI
        )
        payload = [
            (self._checksum(w3, target), True, data) for target, data in batch
        ]
        call = contract.functions.aggregate3(payload)
        if block_number is not None:
            result = await self.client._rpc(
                chain, lambda: call.call(block_identifier=block_number)
            )
        else:
            result = await self.client._rpc(chain, call.call)
        return [(bool(item[0]), item[1]) for item in result]
