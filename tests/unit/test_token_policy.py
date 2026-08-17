"""A token must be cleared before capital touches it.

The blockchain audit found three classes of token that break this strategy's
core assumption -- that the amount you send is the amount that arrives, and the
amount you hold is the amount you can withdraw:

* FEE-ON-TRANSFER. The highest-volume Base pool in the scanned dataset is
  LINGO/WETH, and LINGO takes 1.25% on transfer. QuoterV2 does not model it, so
  the quoted output is not the received amount. Every trade would lose 125 bps
  that the economics say is profit -- and 125 bps is an order of magnitude
  larger than the 5 bps edge the system trades on.

* REBASING. sOHM-style balances change without a transfer. Position accounting
  reads a balance that moved on its own, so reconciliation cannot distinguish a
  rebase from a missing fill.

* UNEXITABLE. A token can trade on the CEX while withdrawals are suspended
  (UST, LUNA, FEI, renZEC). The arbitrage completes on paper and the inventory
  is then stranded on one venue -- a total loss of the float, not a bad trade.

None of these is detectable from a price. They are properties of the token that
must be checked once, by a human, and then enforced mechanically. This module is
the enforcement.
"""
import pytest

from src.strategy.token_policy import (
    TokenPolicy,
    TokenPolicyError,
    TokenRisk,
)


# --------------------------------------------------------------------------
# allowlist mode: default deny
# --------------------------------------------------------------------------

def test_allowlist_mode_permits_only_listed_tokens():
    policy = TokenPolicy(mode="allowlist", allowed=["WETH", "USDT"])

    assert policy.check("WETH", "USDT").allowed

    verdict = policy.check("WETH", "LINGO")
    assert not verdict.allowed
    assert "LINGO" in verdict.reason


def test_allowlist_mode_rejects_an_unknown_token_even_with_no_denylist():
    """The point of default-deny: a token nobody has looked at is not tradeable.

    A denylist can only ever contain the hazards someone already knows about.
    The dangerous token is the one discovered by the scanner at 3am.
    """
    policy = TokenPolicy(mode="allowlist", allowed=["WETH"])

    assert not policy.check("SOMETHING_NEW").allowed


def test_an_empty_allowlist_is_rejected_at_construction():
    """An empty allowlist in allowlist mode denies everything, which reads as a
    broken bot rather than as a policy. Fail loudly at startup instead."""
    with pytest.raises(TokenPolicyError):
        TokenPolicy(mode="allowlist", allowed=[])


# --------------------------------------------------------------------------
# denylist mode: for measurement, never for capital
# --------------------------------------------------------------------------

def test_denylist_mode_permits_anything_not_explicitly_denied():
    policy = TokenPolicy(
        mode="denylist",
        denied={"LINGO": {"risks": ["fee_on_transfer"], "note": "1.25% on transfer"}},
    )

    assert policy.check("WETH", "SOMETHING_NEW").allowed
    assert not policy.check("LINGO").allowed


def test_a_denied_token_reports_its_risks_and_the_human_note():
    """The rejection has to say WHY, or the next operator deletes the entry."""
    policy = TokenPolicy(
        mode="denylist",
        denied={
            "LINGO": {
                "risks": ["fee_on_transfer"],
                "note": "1.25% transfer fee, invisible to QuoterV2",
            }
        },
    )

    verdict = policy.check("LINGO")

    assert not verdict.allowed
    assert TokenRisk.FEE_ON_TRANSFER in verdict.risks
    assert "1.25%" in verdict.reason
    assert "QuoterV2" in verdict.reason


def test_the_denylist_wins_over_the_allowlist():
    """If a token appears in both, the hazard is the operative fact.

    Two people editing two lists must not be able to produce a permit.
    """
    policy = TokenPolicy(
        mode="allowlist",
        allowed=["WETH", "LINGO"],
        denied={"LINGO": {"risks": ["fee_on_transfer"], "note": "1.25%"}},
    )

    assert not policy.check("LINGO").allowed
    assert policy.check("WETH").allowed


def test_an_unknown_risk_label_is_rejected_rather_than_ignored():
    """A typo in a risk label must not silently become an unclassified entry.

    A denylist whose reasons are unparseable cannot be reviewed, and an entry
    nobody can review is an entry someone eventually deletes.
    """
    with pytest.raises(TokenPolicyError):
        TokenPolicy(
            mode="denylist",
            denied={"X": {"risks": ["fee_on_transfr"], "note": "typo"}},
        )


def test_a_denied_token_must_carry_a_note():
    """An entry with no explanation cannot be audited or safely removed."""
    with pytest.raises(TokenPolicyError):
        TokenPolicy(mode="denylist", denied={"X": {"risks": ["rebasing"], "note": ""}})


def test_an_unknown_mode_is_rejected():
    with pytest.raises(TokenPolicyError):
        TokenPolicy(mode="permissive", allowed=["WETH"])


# --------------------------------------------------------------------------
# case and whitespace
# --------------------------------------------------------------------------

def test_symbols_are_matched_case_insensitively():
    """Binance says ETHUSDT, tokens.yaml says WETH, subgraphs return mixed case.
    A policy that could be bypassed by case would be worse than no policy."""
    policy = TokenPolicy(
        mode="allowlist",
        allowed=["weth"],
        denied={"lingo": {"risks": ["fee_on_transfer"], "note": "1.25%"}},
    )

    assert policy.check("WETH").allowed
    assert not policy.check("LiNgO").allowed
    assert not policy.check(" LINGO ").allowed, "whitespace must not bypass"


def test_checking_no_symbols_is_a_programming_error():
    """An empty call returning "allowed" would make a forgotten argument look
    like a pass."""
    policy = TokenPolicy(mode="allowlist", allowed=["WETH"])
    with pytest.raises(TokenPolicyError):
        policy.check()


# --------------------------------------------------------------------------
# the shipped policy must actually cover the known hazards
# --------------------------------------------------------------------------

def test_the_shipped_default_policy_denies_the_known_hazards():
    """A guard on the configuration itself, not just on the mechanism.

    These specific tokens were identified as hazards in the audit. A refactor
    that empties the denylist would otherwise pass every test above.
    """
    from src.core.config import load_config

    policy = load_config().strategy.token_policy.build()

    for symbol, why in [
        ("LINGO", "fee-on-transfer, 1.25%"),
        ("UST", "withdrawals suspended"),
        ("LUNA", "withdrawals suspended"),
        ("sOHM", "rebasing"),
    ]:
        verdict = policy.check(symbol)
        assert not verdict.allowed, f"{symbol} must be denied ({why})"
        assert verdict.risks, f"{symbol} must carry a classified risk"


def test_an_operator_can_add_a_denial_without_touching_code():
    """A token found to be hazardous at 2am must be blockable from YAML.

    The reviewed registry stays in code -- where it is version-controlled, code
    reviewed, and covered by the test above -- while urgent additions go in
    config. Editing a Python file under time pressure is the more dangerous of
    the two operations.
    """
    from src.core.config import TokenPolicyConfig

    policy = TokenPolicyConfig(
        denied_extra={
            "URGENT": {"risks": ["fee_on_transfer"], "note": "found taking 2%"}
        }
    ).build()

    assert not policy.check("URGENT").allowed
    # and the code-level registry is unaffected
    assert not policy.check("LINGO").allowed


def test_an_operator_addition_cannot_remove_a_registered_hazard():
    """The asymmetry that makes the split safe: config can only add.

    If the whole denylist lived in YAML, a careless edit -- or a merge that
    dropped a key -- would silently re-permit a known hazard, and nothing would
    fail until money moved.
    """
    from src.core.config import TokenPolicyConfig

    policy = TokenPolicyConfig(denied_extra={}).build()

    assert not policy.check("LINGO").allowed


def test_the_shipped_default_policy_permits_the_configured_pairs():
    """The other half: a policy that denied the pairs we actually trade would
    stop the bot dead, and this test says so at build time rather than in
    production."""
    from src.core.config import load_config

    config = load_config()
    policy = config.strategy.token_policy.build()

    for pair in config.pairs:
        verdict = policy.check(pair.base, pair.quote)
        assert verdict.allowed, (
            f"configured pair {pair.cex_symbol} is blocked by the token "
            f"policy: {verdict.reason}"
        )
