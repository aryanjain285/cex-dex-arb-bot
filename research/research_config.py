"""One place every research script gets its config from.

Read-only and explicitly pointed at `.env.research`, so no observation process
holds a signing key and none of them can be confused about which endpoints
produced their numbers.
"""
import os
import sys

sys.path.insert(0, ".")

from loguru import logger

from src.core.config import load_config


def research_config(log_level: str = "INFO"):
    # The RPC limiter logs every pacing decision at DEBUG. That is right for
    # diagnosing a stall and wrong for a measurement run, where it buries the
    # measurement itself.
    logger.remove()
    logger.add(sys.stderr, level=log_level,
               format="{time:HH:mm:ss} | {level: <7} | {message}")

    config = load_config(env_file=".env.research", require_signing_key=False)
    assert config.secrets.dex_wallet_private_key is None, (
        "a research process must not hold a signing key"
    )
    return config
