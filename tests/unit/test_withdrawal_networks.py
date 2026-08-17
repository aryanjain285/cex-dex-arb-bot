"""A token existing on a chain is not the same as being able to move it there.

The survey found the one plausible result of the day: canonical bridged LINK on
Arbitrum prices 30-53 bps BELOW Binance's bid, in pools holding roughly $1.2m,
and the price barely moves between a $100 and a $1,000 buy -- so it is not price
impact, the pool is genuinely priced there.

    CEX  LINK/USDT bid                     9.5000
    DEX  tier 500,  $1000 buy   9.4495     +53.4 bps
    DEX  tier 3000, $1000 buy   9.4713     +30.3 bps
    pool 500   holds 10,459 LINK and  97.7 WETH
    pool 3000  holds 60,915 LINK and 179.5 WETH

A persistent discount on one of the most heavily arbitraged tokens in existence
has an explanation, and the likely one is settlement: capturing it requires moving
LINK between Arbitrum and Binance, and if Binance does not credit LINK deposits on
the Arbitrum network then the round trip needs a bridge with its own fee and delay.
The discount would then be the market's price for that bridge -- real, and not
capturable by this strategy.

The rotation cost model assumes a direct CEX withdrawal onto the chain being
traded. That holds for ETH and ARB on Arbitrum. Whether it holds for LINK is a
FACTUAL question about Binance's supported networks, answerable from
`/sapi/v1/capital/config/getall` -- which is a signed endpoint.

So this module adds the missing distinction. `withdraw_networks` records, per
token, the chains on which the CEX will actually settle it. A pair whose
`dex_chain` is not on that list cannot be traded, however good its price looks,
because the inventory cannot get to or from the venue.
"""
import pytest

from src.strategy.token_policy import TokenPolicy, TokenPolicyError, TokenRisk


def _policy(**kwargs) -> TokenPolicy:
    defaults = dict(mode="allowlist", allowed=["WETH", "ETH", "USDT", "ARB", "LINK"])
    defaults.update(kwargs)
    return TokenPolicy(**defaults)


# --- the check itself ----------------------------------------------------


def test_a_token_with_no_recorded_networks_is_unconstrained():
    """Absence of data is not evidence of absence, and inventing a constraint from
    silence would block every token before the data has been gathered."""
    policy = _policy()

    assert policy.check_chain("LINK", "arbitrum").allowed


def test_a_chain_on_the_list_is_permitted():
    policy = _policy(withdraw_networks={"LINK": ["ethereum", "arbitrum"]})

    assert policy.check_chain("LINK", "arbitrum").allowed


def test_a_chain_absent_from_the_list_is_refused():
    """The LINK case: the price is real and the settlement is not available."""
    policy = _policy(withdraw_networks={"LINK": ["ethereum"]})

    verdict = policy.check_chain("LINK", "arbitrum")

    assert not verdict.allowed
    assert "arbitrum" in verdict.reason
    assert TokenRisk.WITHDRAWAL_SUSPENDED in verdict.risks


def test_the_reason_says_what_the_alternative_is():
    """An operator reading this needs to know where the token CAN settle, or the
    message just says no."""
    policy = _policy(withdraw_networks={"LINK": ["ethereum", "bsc"]})

    verdict = policy.check_chain("LINK", "arbitrum")

    assert "ethereum" in verdict.reason and "bsc" in verdict.reason


def test_chain_names_are_matched_case_insensitively():
    policy = _policy(withdraw_networks={"link": ["Arbitrum"]})

    assert policy.check_chain("LINK", "arbitrum").allowed


def test_an_empty_network_list_means_the_token_cannot_settle_anywhere():
    """Distinct from an absent entry: an explicit empty list is a recorded fact --
    somebody looked, and there is no network. That must block, where silence does
    not."""
    policy = _policy(withdraw_networks={"LINK": []})

    verdict = policy.check_chain("LINK", "arbitrum")

    assert not verdict.allowed
    assert "no network" in verdict.reason.lower()


def test_a_denied_token_is_still_denied_regardless_of_networks():
    """The hazard check comes first: a fee-on-transfer token with perfect
    settlement is still a fee-on-transfer token."""
    policy = _policy(
        denied={"LINGO": {"risks": ["fee_on_transfer"], "note": "1.25%"}},
        withdraw_networks={"LINGO": ["base"]},
    )

    verdict = policy.check_chain("LINGO", "base")

    assert not verdict.allowed
    assert "denylist" in verdict.reason


def test_a_token_outside_the_allowlist_fails_before_the_network_check():
    """Ordering matters for the message: "not reviewed" is more actionable than
    "wrong network" for a token nobody has looked at."""
    policy = _policy(withdraw_networks={"NEWCOIN": ["ethereum"]})

    verdict = policy.check_chain("NEWCOIN", "ethereum")

    assert not verdict.allowed
    assert "allowlist" in verdict.reason


# --- configuration -------------------------------------------------------


def test_an_unknown_chain_name_in_the_config_is_rejected():
    """A typo would silently make a token look unsettleable on the chain it
    actually works on -- failing closed, but for a fictional reason."""
    with pytest.raises(TokenPolicyError, match="unknown chain|arbitum"):
        _policy(withdraw_networks={"LINK": ["arbitum"]})


def test_the_networks_map_is_reported_in_the_description():
    policy = _policy(withdraw_networks={"LINK": ["ethereum"]})

    assert "withdraw" in policy.describe().lower()


# --- the detector applies it --------------------------------------------


async def test_the_detector_refuses_a_pair_on_an_unsettleable_chain():
    """The full point: a good price on a chain we cannot settle on is not an
    opportunity, and the audit trail must say which of the two problems it was."""
    from decimal import Decimal

    from src.core.config import (
        RotationConfig, StrategyConfig, TokenPolicyConfig,
    )
    from src.strategy.detector import OpportunityDetector, RejectionReason
    from tests.fakes import FakeCex, FakeDex, flat_book, make_pair

    class Rec:
        def __init__(self): self.rows = []
        def record(self, r): self.rows.append(r); return len(self.rows)

    rec = Rec()
    pair = make_pair("LINK/USDT", base="LINK", dex_chain="arbitrum")
    det = OpportunityDetector(
        StrategyConfig(
            target_notional_usd=1000, min_net_bps=Decimal(5),
            rotation=RotationConfig(enabled=False),
            dex_routing={"enabled": False},
            token_policy=TokenPolicyConfig(
                mode="allowlist",
                allowed=["LINK", "USDT", "WETH", "ETH"],
                withdraw_networks={"LINK": ["ethereum"]},
            ),
        ),
        FakeCex({"LINK/USDT": flat_book(bid=9.5, ask=9.5)}),
        FakeDex(sell_price=10, buy_price=10), [pair], store=rec,
    )

    found = await det.detect()

    assert found == [], "a 5% edge on an unsettleable chain is not an opportunity"
    assert any(r.reason == RejectionReason.TOKEN_DENIED for r in rec.rows)


async def test_the_detector_still_trades_a_settleable_chain():
    from decimal import Decimal

    from src.core.config import (
        RotationConfig, StrategyConfig, TokenPolicyConfig,
    )
    from src.strategy.detector import OpportunityDetector
    from tests.fakes import FakeCex, FakeDex, flat_book, make_pair

    pair = make_pair("LINK/USDT", base="LINK", dex_chain="arbitrum")
    det = OpportunityDetector(
        StrategyConfig(
            target_notional_usd=1000, min_net_bps=Decimal(5),
            rotation=RotationConfig(enabled=False),
            dex_routing={"enabled": False},
            token_policy=TokenPolicyConfig(
                mode="allowlist",
                allowed=["LINK", "USDT", "WETH", "ETH"],
                withdraw_networks={"LINK": ["ethereum", "arbitrum"],
                                   "USDT": ["ethereum", "arbitrum"]},
            ),
        ),
        FakeCex({"LINK/USDT": flat_book(bid=9.5, ask=9.5)}),
        FakeDex(sell_price=10, buy_price=10), [pair],
    )

    assert await det.detect(), "a settleable chain with a real edge must trade"
