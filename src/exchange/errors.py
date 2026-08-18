"""Telling "the market is empty" apart from "our node is unhappy".

`UniV3DexClient.get_quote` ended in `except Exception: return None`, and the
detector records a `None` quote as `no_dex_quote`. Four unrelated situations
therefore produced the same audit row:

    the pool genuinely has no liquidity
    the node returned 429 Too Many Requests
    the node timed out
    the ABI does not match the deployed contract

Found by hitting it: surveying a wide token universe against public RPC endpoints
drew sustained 429s, each logged at debug level and reported upward as "no pool".
A live bot under RPC pressure would quietly stop finding opportunities while its
own dataset recorded an empty market -- and that dataset is what decisions are
made from.

The distinction drives different actions. "No liquidity" means stop watching the
pair. "Throttled" means slow down, or pay for a better node. Acting on the first
while the second is true is how a working strategy gets abandoned on the evidence
of its own instrumentation.
"""
from __future__ import annotations

__all__ = ["RpcError", "classify_rpc_failure"]


class RpcError(RuntimeError):
    """A transport-level failure talking to a chain: throttling, timeout, outage.

    Deliberately NOT raised for a contract revert. A revert is information about
    the chain's state -- an empty pool, an unsupported path -- and retrying it or
    blaming infrastructure would hide a real data problem behind an operational
    excuse.
    """


# Substrings that indicate the request never got a meaningful answer from the
# node. Matched case-insensitively against the exception's string form, because
# web3 wraps provider errors in several different types depending on the
# transport and none of them is stable enough to match on class alone.
_TRANSPORT_MARKERS = (
    "429",
    "too many requests",
    "rate limit",
    "500 server error",
    "502",
    "bad gateway",
    "503",
    "service unavailable",
    "504",
    "gateway timeout",
    "timed out",
    "timeout",
    "connection aborted",
    "connection reset",
    "connection refused",
    "cannot connect",
    "max retries exceeded",
    "remote end closed connection",
    "server disconnected",
    "temporary failure in name resolution",
)

# Types that are transport failures whatever their message says.
_TRANSPORT_TYPES = (TimeoutError, ConnectionError)


def classify_rpc_failure(exc: BaseException) -> bool:
    """True when this exception means the node did not answer.

    Unknown failures return False. Defaulting the other way would relabel every
    genuine revert as infrastructure, which is the more comfortable answer and
    the less true one.
    """
    if isinstance(exc, _TRANSPORT_TYPES):
        return True
    text = f"{type(exc).__name__}: {exc}".lower()
    # A revert is decisive: the node answered, and the answer was "no".
    if "revert" in text:
        return False
    return any(marker in text for marker in _TRANSPORT_MARKERS)


class ReadOnlyWalletError(RuntimeError):
    """Raised when a read-only client is asked to sign.

    Its own type, not a bare RuntimeError: an operator seeing a failed execution
    needs to distinguish "the wallet key is missing from this process" from "the
    chain rejected the transaction". They have completely different fixes.
    """
