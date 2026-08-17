"""The token policy has to be enforced, not merely configured.

A policy object nobody consults is documentation. There are three places a
token can reach capital, and each needs its own gate:

1. `pairs.yaml` -- a human writes a pair by hand. Caught at config load, so the
   process refuses to start rather than trading it.
2. the detector -- a pair arrives at runtime (the volume scanner writes
   candidates that later become config). Caught per evaluation, and the
   rejection is persisted so the denial is visible in the dataset rather than
   being a silent absence.
3. `env: prod` -- denylist mode is a measurement tool. With real capital,
   "anything not on the hazard list is fine" is the wrong default, so prod
   requires default-deny.
"""
import pytest

from decimal import Decimal

from src.core.config import (
    AppConfig, CexConfig, DashboardConfig, DexConfig, DexContracts,
    InventoryConfig, ObservabilityConfig, PairConfig, RebalanceConfig,
    RiskConfig, RotationConfig, SecretsConfig, StrategyConfig,
    TokenPolicyConfig,
)


def _app(**overrides):
    base = dict(
        env="dev",
        network=dict(default_chain="ethereum", max_pending_seconds=30,
                     gas_estimation_chain="ethereum", priority_fee_gwei=2.0,
                     max_fee_gwei=60.0, native_token={"ethereum": "ETH"}),
        dex=DexConfig(uniswap_v3={"ethereum": DexContracts(
            router="0x" + "11" * 20, quoter_v2="0x" + "22" * 20,
            weth="0x" + "33" * 20)}),
        cex=CexConfig(name="binance", base_url="https://x", ws_url="wss://x/ws",
                      api_key_env="A", api_secret_env="B", recv_window_ms=5000),
        risk=RiskConfig(max_notional_per_leg_quote=1200, max_position_per_asset=2.0,
                        circuit_breaker_bps=250, cancel_all_on_start=False,
                        cancel_all_on_shutdown=False, max_daily_loss_quote=250.0),
        inventory=InventoryConfig(rebalance=RebalanceConfig(
            enable=False, target_ratio=0.5, trigger_bps=500, method="on_cex")),
        observability=ObservabilityConfig(metrics_port=9000, log_level="INFO",
                                         redact_keys=[]),
        dashboard=DashboardConfig(enabled=False),
        strategy=StrategyConfig(target_notional_usd=1000),
        pairs=[PairConfig(base="WETH", quote="USDT", cex_symbol="ETH/USDT",
                          max_slippage_bps=30, max_size_quote=5000,
                          dex_chain="ethereum", dex_pool_fee=500)],
        tokens={},
        secrets=SecretsConfig(binance_api_key="k", binance_api_secret="s",
                              dex_wallet_private_key="0x" + "11" * 32),
    )
    base.update(overrides)
    return AppConfig(**base)


# --------------------------------------------------------------------------
# 1. config load
# --------------------------------------------------------------------------

def test_a_configured_pair_on_a_denied_token_refuses_to_load():
    """LINGO is fee-on-transfer. A hand-written pairs.yaml entry for it must
    stop the process, not produce a bot that loses 125 bps per trade while its
    own arithmetic reports a profit."""
    with pytest.raises(Exception, match="LINGO|denylist"):
        _app(pairs=[PairConfig(base="LINGO", quote="WETH", cex_symbol="LINGO/ETH",
                               max_slippage_bps=30, max_size_quote=5000,
                               dex_chain="ethereum", dex_pool_fee=3000)])


def test_a_configured_pair_outside_the_allowlist_refuses_to_load():
    with pytest.raises(Exception, match="allowlist|NEWCOIN"):
        _app(pairs=[PairConfig(base="NEWCOIN", quote="USDT",
                               cex_symbol="NEWCOIN/USDT",
                               max_slippage_bps=30, max_size_quote=5000,
                               dex_chain="ethereum", dex_pool_fee=3000)])


def test_the_quote_side_is_checked_too():
    """A clean base against a hazardous quote loses exactly as much money."""
    with pytest.raises(Exception, match="LINGO|denylist"):
        _app(pairs=[PairConfig(base="WETH", quote="LINGO", cex_symbol="ETH/LINGO",
                               max_slippage_bps=30, max_size_quote=5000,
                               dex_chain="ethereum", dex_pool_fee=3000)])


def test_the_dex_quote_asset_is_checked_for_a_synthetic_pair():
    """A synthetic pair trades base against a third asset on the DEX -- WETH,
    say -- and that asset is just as capable of being fee-on-transfer as the two
    in the pair's name. It is the easiest one to overlook precisely because the
    pair is called ETH/USDT while the on-chain leg touches something else.

    PairConfig has no synthetic fields, so the config-load gate covers base and
    quote; the DEX-side asset is checked in the detector, where the MarketPair
    that carries it is actually evaluated (see the synthetic test below).
    """
    from src.strategy.token_policy import TokenPolicy

    policy = TokenPolicy(
        mode="allowlist", allowed=["WETH", "USDT"],
        denied={"LINGO": {"risks": ["fee_on_transfer"], "note": "1.25%"}},
    )
    # base and quote alone would pass; the third asset is what fails.
    assert policy.check("WETH", "USDT").allowed
    assert not policy.check("WETH", "USDT", "LINGO").allowed


def test_denylist_mode_is_refused_in_prod():
    with pytest.raises(Exception, match="denylist|allowlist"):
        _app(env="prod",
             strategy=StrategyConfig(
                 target_notional_usd=1000,
                 token_policy=TokenPolicyConfig(mode="denylist")))


def test_denylist_mode_is_permitted_outside_prod():
    """Measurement runs need to observe the whole market."""
    cfg = _app(strategy=StrategyConfig(
        target_notional_usd=1000,
        token_policy=TokenPolicyConfig(mode="denylist")))
    assert cfg.strategy.token_policy.mode == "denylist"


# --------------------------------------------------------------------------
# 2. the detector
# --------------------------------------------------------------------------

async def test_the_detector_rejects_and_records_a_denied_pair():
    """Defence in depth for a pair that reaches the detector anyway.

    Recording matters as much as rejecting: a silently skipped pair is
    indistinguishable in the dataset from a pair that never had an opportunity,
    which would make the run history misleading.
    """
    from src.strategy.detector import OpportunityDetector, RejectionReason
    from tests.fakes import FakeCex, FakeDex, flat_book, make_pair

    pair = make_pair("LINGO/USDT", base="LINGO")
    cex = FakeCex({"LINGO/USDT": flat_book(bid=1000, ask=1000)})
    dex = FakeDex(sell_price=1100, buy_price=1100)  # a large fake edge

    class Rec:
        def __init__(self): self.rows = []
        def record(self, r): self.rows.append(r); return len(self.rows)

    rec = Rec()
    det = OpportunityDetector(
        StrategyConfig(target_notional_usd=1000, min_net_bps=Decimal(5),
                       rotation=RotationConfig(enabled=False)),
        cex, dex, [pair], store=rec)

    found = await det.detect()

    assert found == [], "a denied token must never produce an opportunity"
    assert rec.rows, "the denial must be persisted, not silently skipped"
    assert all(r.outcome == "rejected" for r in rec.rows)
    assert any(r.reason == RejectionReason.TOKEN_DENIED for r in rec.rows), (
        f"expected a token_denied rejection, got {[r.reason for r in rec.rows]}"
    )


async def test_the_detector_does_not_call_the_dex_for_a_denied_pair():
    """The gate must come before the RPC call. Quoting a token we will never
    trade spends rate limit and adds latency for every other pair."""
    from src.strategy.detector import OpportunityDetector
    from tests.fakes import FakeCex, FakeDex, flat_book, make_pair

    pair = make_pair("LINGO/USDT", base="LINGO")
    dex = FakeDex(sell_price=1100, buy_price=1100)
    det = OpportunityDetector(
        StrategyConfig(target_notional_usd=1000, min_net_bps=Decimal(5),
                       rotation=RotationConfig(enabled=False)),
        FakeCex({"LINGO/USDT": flat_book(bid=1000, ask=1000)}), dex, [pair])

    await det.detect()

    assert dex.requests == [], (
        f"the DEX was quoted {len(dex.requests)} times for a denied pair"
    )


async def test_the_detector_checks_the_dex_side_asset_of_a_synthetic_pair():
    """The asset that appears nowhere in the pair's name.

    A synthetic ETH/USDT pair trading against a hazardous asset on-chain has
    clean base and quote symbols, so only a check on the DEX-side asset catches
    it. Same for the intermediate symbol used to convert the synthetic price.
    """
    from src.strategy.detector import OpportunityDetector, RejectionReason
    from tests.fakes import FakeCex, FakeDex, flat_book, make_pair

    pair = make_pair("ETH/USDT", quote_dex="LINGO", is_synthetic=True,
                     intermediate_symbol="LINGO/USDT")
    dex = FakeDex(sell_price=1100, buy_price=1100)

    class Rec:
        def __init__(self): self.rows = []
        def record(self, r): self.rows.append(r); return len(self.rows)

    rec = Rec()
    det = OpportunityDetector(
        StrategyConfig(target_notional_usd=1000, min_net_bps=Decimal(5),
                       rotation=RotationConfig(enabled=False)),
        FakeCex({"ETH/USDT": flat_book(bid=1000, ask=1000)}), dex, [pair],
        store=rec)

    found = await det.detect()

    assert found == []
    assert dex.requests == [], "the gate must precede the RPC call"
    assert any(r.reason == RejectionReason.TOKEN_DENIED for r in rec.rows)


async def test_an_allowed_pair_still_evaluates_normally():
    """The negative control: the gate must not block everything."""
    from src.strategy.detector import OpportunityDetector
    from tests.fakes import FakeCex, FakeDex, flat_book, make_pair

    pair = make_pair()  # WETH/USDT
    det = OpportunityDetector(
        StrategyConfig(target_notional_usd=1000, min_net_bps=Decimal(5),
                       rotation=RotationConfig(enabled=False)),
        FakeCex({"ETH/USDT": flat_book(bid=1000, ask=1000)}),
        FakeDex(sell_price=1100, buy_price=1100), [pair])

    found = await det.detect()

    assert found, "an allowlisted pair with a real edge must still be found"
