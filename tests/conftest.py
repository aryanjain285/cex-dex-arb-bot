"""Test-wide setup.

`load_config()` raises unless three credentials are present in the environment,
so until now no test could call it -- and the shipped YAML (default.yaml,
pairs.yaml, tokens.yaml) was therefore never validated by the test suite. Only
the CI "verify config loads" step covered it, which meant a malformed config
file could only be discovered by pushing.

Placeholders are set unconditionally rather than only when absent. `load_dotenv`
does not override variables already in the environment, so setting them here
guarantees two things: the suite behaves identically on a machine with a real
.env and on one without, and a developer's real keys never enter a test process.
No test makes an authenticated call; these values only have to be non-empty and
well-formed.
"""
import os

import pytest

# 32 bytes of 0x11 -- a syntactically valid secp256k1 private key that is not
# anyone's. web3 parses it; nothing signs with it.
PLACEHOLDER_PRIVATE_KEY = "0x" + "11" * 32


@pytest.fixture(scope="session", autouse=True)
def placeholder_credentials():
    os.environ["BINANCE_API_KEY"] = "test-placeholder-key"
    os.environ["BINANCE_API_SECRET"] = "test-placeholder-secret"
    os.environ["DEX_WALLET_PRIVATE_KEY"] = PLACEHOLDER_PRIVATE_KEY
    yield
