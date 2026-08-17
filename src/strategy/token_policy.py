"""Which tokens capital is permitted to touch.

The strategy's arithmetic rests on two assumptions that are false for a
meaningful fraction of tokens on any DEX:

1. The amount sent is the amount that arrives. False for fee-on-transfer
   tokens. QuoterV2 does not model a transfer tax, so the quote overstates the
   received amount by the tax. The tax is typically 100-500 bps against a target
   edge of 5 bps, so the sign of the trade is wrong, not merely the magnitude.

2. A balance changes only when you trade. False for rebasing tokens, whose
   supply is adjusted for every holder. Position accounting then cannot tell a
   rebase from an unfilled leg.

A third hazard is not about arithmetic at all: a token can be perfectly
well-behaved on-chain and still be un-withdrawable from the CEX. The trade
completes, the inventory lands on one venue, and the float is stranded. That is
a loss of principal rather than a losing trade.

None of these is visible in a price feed, so none can be detected by the
detector. They are static properties of a token, established once by a human and
then enforced mechanically. This module is that enforcement.

The default mode is `allowlist` -- default-deny. A denylist can only contain the
hazards someone has already found, and the token that costs money is the one the
volume scanner discovers at 3am. Denylist mode exists for measurement runs where
the point is to observe the whole market; `env: prod` refuses it.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple

__all__ = [
    "TokenRisk",
    "TokenVerdict",
    "TokenPolicy",
    "TokenPolicyError",
    "DeniedToken",
]

MODES = ("allowlist", "denylist")


class TokenPolicyError(ValueError):
    """A policy that cannot be enforced as written.

    Raised at construction, never at check time. A misconfigured policy must
    stop the process at startup rather than fail open on the first trade.
    """


class TokenRisk(str, Enum):
    """Why a token is unsafe. Fixed vocabulary so the denylist is reviewable."""

    # Takes a cut on transfer. The quoter cannot see it.
    FEE_ON_TRANSFER = "fee_on_transfer"
    # Balances move without a transfer, so accounting cannot be reconciled.
    REBASING = "rebasing"
    # Tradeable on the CEX but not withdrawable: inventory strands.
    WITHDRAWAL_SUSPENDED = "withdrawal_suspended"
    # Owner can pause, blacklist, or otherwise block a transfer mid-trade.
    TRANSFER_RESTRICTED = "transfer_restricted"
    # Supply or price can be moved arbitrarily by a privileged address.
    UPGRADEABLE_OR_MINTABLE = "upgradeable_or_mintable"
    # Looked at, found unclear, deliberately parked. Distinct from "unknown",
    # which is what an absence from the allowlist already means.
    UNVERIFIED = "unverified"


@dataclass(frozen=True)
class DeniedToken:
    symbol: str
    risks: Tuple[TokenRisk, ...]
    note: str


@dataclass(frozen=True)
class TokenVerdict:
    """The outcome of a check, with enough detail to act on and to audit."""

    allowed: bool
    # Empty for a token that is merely absent from the allowlist: "nobody has
    # cleared this" is not the same claim as "this has a known defect".
    risks: Tuple[TokenRisk, ...]
    reason: str


def _normalise(symbol: str) -> str:
    """Uppercase and strip.

    Binance reports ETHUSDT, tokens.yaml says WETH, subgraphs return whatever
    the token contract declares. A policy bypassable by case or a stray space
    would be worse than no policy, because it would be believed.
    """
    return symbol.strip().upper()


class TokenPolicy:
    def __init__(
        self,
        mode: str = "allowlist",
        allowed: Optional[Iterable[str]] = None,
        denied: Optional[Mapping[str, Mapping[str, object]]] = None,
    ):
        if mode not in MODES:
            raise TokenPolicyError(
                f"unknown token policy mode {mode!r}; expected one of {MODES}"
            )
        self.mode = mode

        self.allowed = {_normalise(s) for s in (allowed or ())}
        if mode == "allowlist" and not self.allowed:
            raise TokenPolicyError(
                "allowlist mode with an empty allowlist denies every token, "
                "which presents as a broken bot rather than as a policy. List "
                "the tokens that have been cleared, or switch to denylist mode "
                "for a measurement run."
            )

        self.denied: Dict[str, DeniedToken] = {}
        for symbol, spec in (denied or {}).items():
            self.denied[_normalise(symbol)] = self._parse_denied(symbol, spec)

    # ------------------------------------------------------------------

    @staticmethod
    def _parse_denied(symbol: str, spec: Mapping[str, object]) -> DeniedToken:
        raw_risks = spec.get("risks") or ()
        if isinstance(raw_risks, str):
            raw_risks = (raw_risks,)
        risks = []
        for label in raw_risks:
            try:
                risks.append(TokenRisk(str(label)))
            except ValueError as exc:
                valid = ", ".join(r.value for r in TokenRisk)
                raise TokenPolicyError(
                    f"token {symbol!r} lists an unknown risk {label!r}. A "
                    f"denylist whose reasons cannot be parsed cannot be "
                    f"reviewed, and an entry nobody can review is one someone "
                    f"eventually deletes. Valid risks: {valid}."
                ) from exc
        if not risks:
            raise TokenPolicyError(
                f"token {symbol!r} is denied without a classified risk. State "
                f"which property makes it unsafe."
            )

        note = str(spec.get("note") or "").strip()
        if not note:
            raise TokenPolicyError(
                f"token {symbol!r} is denied without a note. The next person to "
                f"read this list needs the evidence, or they will remove the "
                f"entry as unexplained."
            )
        return DeniedToken(
            symbol=_normalise(symbol), risks=tuple(risks), note=note
        )

    # ------------------------------------------------------------------

    def check(self, *symbols: str) -> TokenVerdict:
        """Verdict for a set of symbols -- typically a pair's base and quote.

        Both sides matter: a well-behaved base against a fee-on-transfer quote
        loses exactly as much money as the reverse.
        """
        if not symbols:
            raise TokenPolicyError(
                "check() called with no symbols. Returning 'allowed' for an "
                "empty check would make a forgotten argument look like a pass."
            )

        # Denials first, and unconditionally: if a symbol appears in both lists
        # the hazard is the operative fact. Two people editing two lists must
        # not be able to combine into a permit.
        for symbol in symbols:
            entry = self.denied.get(_normalise(symbol))
            if entry is not None:
                risks = ", ".join(r.value for r in entry.risks)
                return TokenVerdict(
                    allowed=False,
                    risks=entry.risks,
                    reason=(
                        f"{entry.symbol} is on the token denylist "
                        f"[{risks}]: {entry.note}"
                    ),
                )

        if self.mode == "allowlist":
            for symbol in symbols:
                if _normalise(symbol) not in self.allowed:
                    return TokenVerdict(
                        allowed=False,
                        risks=(),
                        reason=(
                            f"{_normalise(symbol)} is not on the token "
                            f"allowlist. Under default-deny a token must be "
                            f"reviewed for transfer fees, rebasing and CEX "
                            f"withdrawal status before capital touches it."
                        ),
                    )

        return TokenVerdict(allowed=True, risks=(), reason="")

    # ------------------------------------------------------------------

    def classify(self, *symbols: str) -> str:
        """Mode-independent label for a set of symbols, for the audit trail.

        Returns "denied", "not_allowlisted", or "allowed". Recorded on every
        evaluation so a denylist-mode measurement run -- which deliberately
        observes tokens it would never trade -- still yields an honest tradeable
        subset. Without it, one edge distribution would silently mix real
        opportunities with a fee-on-transfer token's transfer tax showing up as
        an edge, which overstates the strategy: the exact failure paper trading
        exists to prevent.

        "denied" and "not_allowlisted" are kept distinct because they are
        different claims -- a known hazard versus a token nobody has examined --
        and the analysis will want to treat them differently.
        """
        if not symbols:
            raise TokenPolicyError("classify() called with no symbols")
        for symbol in symbols:
            if _normalise(symbol) in self.denied:
                return "denied"
        for symbol in symbols:
            if _normalise(symbol) not in self.allowed:
                return "not_allowlisted"
        return "allowed"

    def describe(self) -> str:
        """One line for the startup log, so the policy in force is on the record."""
        if self.mode == "allowlist":
            return (
                f"token policy: allowlist of {len(self.allowed)} "
                f"({', '.join(sorted(self.allowed))}), "
                f"{len(self.denied)} explicit denials"
            )
        return (
            f"token policy: DENYLIST mode -- any token not among the "
            f"{len(self.denied)} listed hazards is permitted"
        )

    def denied_symbols(self) -> Sequence[str]:
        return sorted(self.denied)
