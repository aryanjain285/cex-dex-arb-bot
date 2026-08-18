"""Research must not require the operator's signing key.

Every observation script -- the recorder, the size-curve measurement, the survey,
the backtest -- reads chain state and never signs anything. Yet `load_config`
refused to return without `DEX_WALLET_PRIVATE_KEY`, so each of those processes had
to hold real signing material to do read-only work.

That is the wrong default in two directions at once:

  * A long-running recorder holds a key it has no use for. Any bug that reaches a
    signing path can spend real funds from a process that was never meant to be
    able to.
  * Because research would not start without it, the path of least resistance is
    to put SOMETHING in .env -- and a placeholder key in a file the live bot also
    reads is worse than no file at all.

So a read-only config is a first-class thing. The important half of these tests is
not that it loads; it is that what loads back CANNOT sign. A config that returns a
None key but leaves the signing path reachable has moved the failure later,
which for a private key means moving it to the worst possible moment.

The default is unchanged: ask for a config without saying read-only and a missing
key is still fatal.
"""
import os
from unittest.mock import patch

import pytest

from src.core.config import SecretsConfig


class TestTheDefaultIsUnchanged:
    """A relaxation that also relaxes the default is not a relaxation, it is a hole."""

    def test_a_missing_key_is_still_fatal_by_default(self):
        with patch.dict(os.environ, {"DEX_WALLET_PRIVATE_KEY": ""}, clear=False):
            with pytest.raises(ValueError, match="DEX_WALLET_PRIVATE_KEY"):
                SecretsConfig.load()

    def test_read_only_must_be_asked_for_explicitly(self):
        """No truthy-by-accident default: the parameter is keyword-only and False."""
        import inspect
        sig = inspect.signature(SecretsConfig.load)
        param = sig.parameters["require_signing_key"]
        assert param.default is True, (
            "requiring the signing key must be the default; a research convenience "
            "must never become the live default"
        )
        assert param.kind is inspect.Parameter.KEYWORD_ONLY


class TestReadOnlyLoads:
    def test_a_read_only_config_loads_without_a_key(self):
        with patch.dict(os.environ, {"DEX_WALLET_PRIVATE_KEY": ""}, clear=False):
            secrets = SecretsConfig.load(require_signing_key=False)
        assert secrets.dex_wallet_private_key is None

    def test_read_only_does_not_invent_a_placeholder_key(self):
        """A zero key or a dummy key would load, then fail at signing time with an
        obscure error. None is the only honest representation of 'absent'."""
        with patch.dict(os.environ, {"DEX_WALLET_PRIVATE_KEY": ""}, clear=False):
            secrets = SecretsConfig.load(require_signing_key=False)
        assert secrets.dex_wallet_private_key is None, (
            "a placeholder key makes an unsigned config look signable"
        )

    def test_a_key_that_is_present_is_still_loaded_in_read_only_mode(self):
        """Read-only means 'a key is not required', not 'a key is discarded'. The
        same code path serves live runs, so silently dropping a provided key would
        break execution in the mode that matters."""
        key = "0x" + "11" * 32
        with patch.dict(os.environ, {"DEX_WALLET_PRIVATE_KEY": key}, clear=False):
            secrets = SecretsConfig.load(require_signing_key=False)
        assert secrets.dex_wallet_private_key is not None
        assert secrets.dex_wallet_private_key.get_secret_value() == key


class TestSigningIsUnreachableWithoutAKey:
    """The half that actually protects anything."""

    def _read_only_client(self):
        from src.core.config import DexConfig, DexContracts, NetworkConfig
        from src.exchange.univ3 import UniV3DexClient

        with patch.dict(os.environ, {"DEX_WALLET_PRIVATE_KEY": ""}, clear=False):
            secrets = SecretsConfig.load(require_signing_key=False)

        network = NetworkConfig(
            default_chain="ethereum",
            rpc_urls={"ethereum": "https://example.invalid"},
            native_token={"ethereum": "ETH"},
            max_pending_seconds=60,
            gas_estimation_chain="ethereum",
            priority_fee_gwei=1.0,
            max_fee_gwei=50.0,
        )
        dex = DexConfig(uniswap_v3={"ethereum": DexContracts(
            router="0x" + "11" * 20, quoter_v2="0x" + "22" * 20,
            weth="0x" + "33" * 20)})
        return UniV3DexClient(dex, network, secrets, {})

    def test_a_read_only_client_constructs(self):
        """Construction derived user_address from the key, so a read-only client
        could not even be built -- which is why research needed a key at all."""
        client = self._read_only_client()
        assert client.user_address is None

    def test_signing_a_swap_without_a_key_raises_before_any_rpc(self):
        client = self._read_only_client()
        with pytest.raises(Exception) as exc:
            client._sign_transaction({"to": "0x" + "22" * 20, "value": 0})
        assert "read-only" in str(exc.value).lower(), (
            f"the refusal must name the cause, got: {exc.value}"
        )

    def test_the_refusal_is_not_a_generic_attribute_error(self):
        """`None.get_secret_value()` would also 'fail safe'. It would also be
        indistinguishable from a bug, and an operator debugging a failed execution
        at 3am should not have to work out which."""
        client = self._read_only_client()
        with pytest.raises(Exception) as exc:
            client._sign_transaction({"to": "0x" + "22" * 20, "value": 0})
        assert not isinstance(exc.value, AttributeError)
