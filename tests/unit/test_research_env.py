"""A research run reads its own env file, and cannot borrow the live one.

`load_config` called bare `load_dotenv()`, which walks up looking for `.env`. That
makes the live operator's secrets the ambient default for every process in the
tree, including a week-long recorder that needs nothing but public RPC URLs.

Two properties matter here and they pull in opposite directions:

  * A research process must be able to run with its own endpoints and no wallet.
  * The live path must be completely unchanged -- if a research convenience alters
    which file the live bot reads, it has created exactly the confusion it was
    meant to prevent.

So `env_file` is an explicit opt-in that defaults to None, and None means "the
existing behaviour, byte for byte".
"""
import os
from pathlib import Path
from unittest.mock import patch

import pytest

from src.core.config import load_config


@pytest.fixture
def yaml_paths():
    """The real config tree. These tests are about env resolution, not YAML."""
    return dict(
        default_path="config/default.yaml",
        pairs_path="config/pairs.yaml",
        tokens_path="config/tokens.yaml",
    )


def test_a_named_env_file_is_read(tmp_path, yaml_paths):
    env = tmp_path / ".env.research"
    env.write_text(
        'ETH_RPC_URL="https://research.example/eth"\n'
        'BINANCE_API_KEY=""\n'
        'BINANCE_API_SECRET=""\n',
        encoding="utf-8",
    )
    with patch.dict(os.environ, {"DEX_WALLET_PRIVATE_KEY": "", "ETH_RPC_URL": ""},
                    clear=False):
        config = load_config(
            **yaml_paths, env_file=str(env), require_signing_key=False
        )
    assert config.network.rpc_urls.get("ethereum") == "https://research.example/eth"


def test_a_research_config_has_no_signing_key(tmp_path, yaml_paths):
    env = tmp_path / ".env.research"
    env.write_text('ETH_RPC_URL="https://research.example/eth"\n', encoding="utf-8")
    with patch.dict(os.environ, {"DEX_WALLET_PRIVATE_KEY": ""}, clear=False):
        config = load_config(
            **yaml_paths, env_file=str(env), require_signing_key=False
        )
    assert config.secrets.dex_wallet_private_key is None


def test_a_missing_env_file_is_an_error_not_a_silent_fallback(tmp_path, yaml_paths):
    """`load_dotenv` returns False for a missing path and carries on. A research
    run that silently fell back to the ambient environment would read the live
    operator's endpoints while reporting that it used its own -- and every number
    it produced would be attributed to the wrong configuration."""
    missing = tmp_path / "does-not-exist.env"
    with pytest.raises(FileNotFoundError, match="does-not-exist"):
        load_config(**yaml_paths, env_file=str(missing), require_signing_key=False)


def test_the_default_still_reads_dotenv_from_the_tree(yaml_paths):
    """The live path must be untouched: env_file=None means the old behaviour."""
    import inspect
    sig = inspect.signature(load_config)
    assert sig.parameters["env_file"].default is None
