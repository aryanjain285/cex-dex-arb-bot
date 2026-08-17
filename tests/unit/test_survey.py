"""The universe survey: does a tradeable spread exist anywhere?

This is gate 2, and it turned out not to need the Graph API key the pool dataset
implied. Everything the survey needs is directly knowable:

  * Binance exchangeInfo -- every spot symbol and its status.
  * CoinGecko's free /coins/list?include_platform=true -- canonical token
    addresses per chain, 2.9 MB, no key.
  * The Uniswap v3 factory -- does a pool exist, per fee tier.
  * QuoterV2 -- at what price for this size, net of pool fee and impact.

Three findings shaped this module and are pinned by the tests below.

DIRECT PAIRS ARE THE WRONG QUESTION. Quoting TOKEN/USDT for the top Binance
tokens reverted almost everywhere: altcoin liquidity on v3 pairs against WETH,
not against stablecoins. The survey therefore prices the SYNTHETIC route -- base
against WETH on chain, converted through the CEX's own ETH price -- which is the
route the strategy would actually take, and which costs TWO taker legs.

TICKER COLLISIONS ARE REAL. CoinGecko lists two coins with the ticker BNB. A
survey that took the first match would price whichever it happened to see, and
a ticker collision is exactly how the counterfeit WETH in the shipped pool
dataset would have entered.

A POSITIVE RESULT IS USUALLY A TRAP. The one positive hit in a 45-token survey
was BNB at +239 bps net, through a genuine ~$200k pool -- against the LEGACY
ERC-20 BNB at 0xb8c77482, which Binance does not withdraw (it settles BNB on BNB
Chain). The two legs are different assets in custody terms, so the trade cannot
settle, which is why the gap has been left standing. The survey must therefore
report the token policy's verdict beside every result rather than leaving the
reader to notice.
"""
from decimal import Decimal

import pytest

from src.core.config import load_config
from src.scanner.survey import (
    SurveyCandidate, TokenRegistry, evaluate_candidate, rank,
)


def D(x) -> Decimal:
    return Decimal(str(x))


# --- the token registry --------------------------------------------------


COINS = [
    {"symbol": "eth", "id": "ethereum",
     "platforms": {"ethereum": "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2"}},
    {"symbol": "link", "id": "chainlink",
     "platforms": {"ethereum": "0x514910771AF9Ca656af840dff83E8264EcF986CA"}},
    {"symbol": "bnb", "id": "binancecoin",
     "platforms": {"ethereum": "0xb8c77482e45f1f44de1745f52c74426c631bdd52"}},
    {"symbol": "bnb", "id": "anubis-bridged-bnb",
     "platforms": {"ethereum": "0x699d13487ed6b78953da2750887b58ef738f9636"}},
    {"symbol": "nochain", "id": "nochain", "platforms": {}},
]


def test_the_registry_resolves_an_unambiguous_ticker():
    registry = TokenRegistry.from_coingecko(COINS, chain="ethereum")

    assert registry.address("LINK") == "0x514910771AF9Ca656af840dff83E8264EcF986CA"


def test_a_ticker_claimed_by_two_coins_is_dropped_entirely():
    """CoinGecko really does list two BNBs. Taking the first match would price
    whichever entry happened to come first, and a ticker collision is exactly how
    a counterfeit token enters a universe."""
    registry = TokenRegistry.from_coingecko(COINS, chain="ethereum")

    assert registry.address("BNB") is None
    assert "BNB" in registry.ambiguous


def test_a_token_with_no_address_on_this_chain_is_absent():
    registry = TokenRegistry.from_coingecko(COINS, chain="ethereum")

    assert registry.address("NOCHAIN") is None
    assert "NOCHAIN" not in registry.ambiguous, (
        "absent is not the same as ambiguous, and the reasons differ"
    )


def test_the_registry_is_case_insensitive():
    registry = TokenRegistry.from_coingecko(COINS, chain="ethereum")

    assert registry.address("link") == registry.address("LINK")


# --- evaluating one candidate -------------------------------------------


class StubDex:
    """Quotes a fixed WETH-per-token price, or raises, or returns nothing."""

    def __init__(self, sell_price=None, buy_price=None, error=None, gas=D("0.02")):
        self.sell_price = sell_price
        self.buy_price = buy_price
        self.error = error
        self.gas = gas
        self.calls = []

    async def get_quote(self, pair, size, side, estimate_gas=False):
        from src.core.types import DexQuote
        self.calls.append((pair.dex_pool_fee, side, size))
        if self.error:
            raise self.error
        price = self.sell_price if side == "sell" else self.buy_price
        if price is None:
            return None
        return DexQuote(price=D(price), gas_cost_quote=self.gas)


def _candidate(**overrides) -> SurveyCandidate:
    fields = dict(
        cex_symbol="LINKUSDT", base="LINK", quote="USDT",
        base_address="0x514910771AF9Ca656af840dff83E8264EcF986CA",
        base_decimals=18, chain="ethereum",
        cex_bid=D("14.00"), cex_ask=D("14.01"),
    )
    fields.update(overrides)
    return SurveyCandidate(**fields)


async def test_a_fair_market_yields_a_negative_net_edge():
    """The base case, and the one that should hold nearly everywhere: with the
    two venues in line, two taker legs plus gas plus rotation put the net edge
    firmly under water. A survey that showed otherwise would be broken."""
    # 14.00 USDT per LINK at an ETH price of 1900 is 0.0073684 WETH per LINK.
    dex = StubDex(sell_price="0.00736842", buy_price="0.00736842")

    result = await evaluate_candidate(
        _candidate(), dex, eth_bid=D(1900), eth_ask=D(1900),
        notional=D(1000), taker_fee_bps=D("7.5"), rotation_quote=D(2),
        fee_tiers=(3000,),
    )

    assert result is not None
    assert result.net_bps < 0
    assert result.net_bps < result.gross_bps, "costs must make net worse than gross"


async def test_the_synthetic_route_is_charged_two_taker_legs():
    """base->quote on the CEX and the ETH conversion are both taker fills. One
    leg would understate the cost by 7.5 bps, which is larger than the floor."""
    dex = StubDex(sell_price="0.00736842", buy_price="0.00736842")

    two_legs = await evaluate_candidate(
        _candidate(), dex, eth_bid=D(1900), eth_ask=D(1900),
        notional=D(1000), taker_fee_bps=D("7.5"), rotation_quote=D(0),
        fee_tiers=(3000,),
    )
    one_leg = await evaluate_candidate(
        _candidate(), dex, eth_bid=D(1900), eth_ask=D(1900),
        notional=D(1000), taker_fee_bps=D("7.5"), rotation_quote=D(0),
        fee_tiers=(3000,), cex_legs=1,
    )

    assert one_leg.net_bps - two_legs.net_bps == pytest.approx(D("7.5"), abs=D("0.2"))


async def test_the_best_fee_tier_wins():
    """The survey must not report the first tier it tries."""
    class TieredDex(StubDex):
        async def get_quote(self, pair, size, side, estimate_gas=False):
            from src.core.types import DexQuote
            price = {3000: D("0.00736842"), 500: D("0.00800000")}[pair.dex_pool_fee]
            return DexQuote(price=price, gas_cost_quote=D("0.02"))

    result = await evaluate_candidate(
        _candidate(), TieredDex(), eth_bid=D(1900), eth_ask=D(1900),
        notional=D(1000), taker_fee_bps=D("7.5"), rotation_quote=D(0),
        fee_tiers=(3000, 500),
    )

    assert result.fee == 500, "the better-priced tier must be reported"


async def test_an_unquotable_candidate_returns_nothing():
    """Most of the universe is unquotable at a real size, and that has to be an
    absence rather than a zero -- 93 of 122 tier-checks in the live survey
    reverted."""
    result = await evaluate_candidate(
        _candidate(), StubDex(sell_price=None, buy_price=None),
        eth_bid=D(1900), eth_ask=D(1900), notional=D(1000),
        taker_fee_bps=D("7.5"), rotation_quote=D(2), fee_tiers=(3000,),
    )

    assert result is None


async def test_an_rpc_failure_is_distinguished_from_no_pool():
    """The distinction that a whole earlier fix exists for: being throttled is not
    the same fact as an empty market."""
    from src.exchange.errors import RpcError

    result = await evaluate_candidate(
        _candidate(), StubDex(error=RpcError("429 Too Many Requests")),
        eth_bid=D(1900), eth_ask=D(1900), notional=D(1000),
        taker_fee_bps=D("7.5"), rotation_quote=D(2), fee_tiers=(3000,),
    )

    assert result is not None
    assert result.rpc_failed is True
    assert result.net_bps is None, (
        "a throttled candidate has no measured edge, and must not report one"
    )


async def test_an_implausible_edge_is_flagged_rather_than_reported():
    """A survey of Base returned TURBOUSDT at +4,118,836 bps.

    That is not an opportunity; it is a token-identity error. CoinGecko lists
    exactly one coin with the ticker TURBO on Base, so the ambiguity filter passed
    it -- but a ticker with a single claimant can still be a DIFFERENT asset from
    the one Binance lists, and no price check can tell. Unambiguous within
    CoinGecko is not the same as "the same asset as the CEX".

    The detector has had a `max_net_bps_sanity` guard for exactly this reason: an
    edge beyond a plausible bound usually means a decimals or identity error, not
    a mispricing. The survey needs the same guard, or its output includes fantasies
    that a reader has to recognise unaided.
    """
    # A WETH-per-token price 1000x the truth, as a wrong-decimals error produces.
    dex = StubDex(sell_price="7.36842", buy_price="7.36842")

    result = await evaluate_candidate(
        _candidate(), dex, eth_bid=D(1900), eth_ask=D(1900),
        notional=D(1000), taker_fee_bps=D("7.5"), rotation_quote=D(2),
        fee_tiers=(3000,), max_plausible_bps=D(1000),
    )

    assert result is not None
    assert result.implausible is True
    assert result.net_bps is not None, (
        "the number is still reported -- suppressing it would hide the error "
        "rather than label it"
    )


async def test_a_plausible_edge_is_not_flagged():
    dex = StubDex(sell_price="0.00750000", buy_price="0.00750000")

    result = await evaluate_candidate(
        _candidate(), dex, eth_bid=D(1900), eth_ask=D(1900),
        notional=D(1000), taker_fee_bps=D("7.5"), rotation_quote=D(2),
        fee_tiers=(3000,), max_plausible_bps=D(1000),
    )

    assert result.implausible is False


async def test_an_implausible_result_is_never_counted_as_tradeable():
    """Belt and braces: the token policy caught the TURBO case, but a survey must
    not depend on a token happening to be off the allowlist."""
    from src.scanner.survey import summarise

    dex = StubDex(sell_price="7.36842", buy_price="7.36842")
    result = await evaluate_candidate(
        _candidate(base="ARB", cex_symbol="ARBUSDT"), dex,
        eth_bid=D(1900), eth_ask=D(1900), notional=D(1000),
        taker_fee_bps=D("7.5"), rotation_quote=D(2), fee_tiers=(3000,),
        token_policy=load_config().strategy.token_policy.build(),
        max_plausible_bps=D(1000),
    )

    assert result.tradeable is True, "ARB is allowlisted"
    assert result.implausible is True

    summary = summarise([result], floor_bps=D(5))
    assert summary["tradeable_above_floor"] == 0, (
        "an implausible number was counted as a tradeable opportunity"
    )
    assert summary["implausible"] == 1


async def test_every_result_carries_the_token_policy_verdict():
    """The survey's one positive hit was an asset-identity trap. A reader must not
    have to notice that themselves."""
    dex = StubDex(sell_price="0.00800000", buy_price="0.00800000")
    policy = load_config().strategy.token_policy.build()

    result = await evaluate_candidate(
        _candidate(base="BNB", cex_symbol="BNBUSDT"), dex,
        eth_bid=D(1900), eth_ask=D(1900), notional=D(1000),
        taker_fee_bps=D("7.5"), rotation_quote=D(2), fee_tiers=(3000,),
        token_policy=policy,
    )

    assert result is not None
    assert result.tradeable is False
    assert "allowlist" in result.policy_reason.lower()


async def test_an_allowlisted_token_is_marked_tradeable():
    dex = StubDex(sell_price="0.00736842", buy_price="0.00736842")
    policy = load_config().strategy.token_policy.build()

    result = await evaluate_candidate(
        _candidate(base="ARB", cex_symbol="ARBUSDT"), dex,
        eth_bid=D(1900), eth_ask=D(1900), notional=D(1000),
        taker_fee_bps=D("7.5"), rotation_quote=D(2), fee_tiers=(3000,),
        token_policy=policy,
    )

    assert result.tradeable is True


# --- ranking -------------------------------------------------------------


EXCHANGE_INFO = {"symbols": [
    {"symbol": "LINKUSDT", "baseAsset": "LINK", "quoteAsset": "USDT",
     "status": "TRADING"},
    {"symbol": "ETHUSDT", "baseAsset": "ETH", "quoteAsset": "USDT",
     "status": "TRADING"},
    {"symbol": "BNBUSDT", "baseAsset": "BNB", "quoteAsset": "USDT",
     "status": "TRADING"},
    {"symbol": "HALTEDUSDT", "baseAsset": "LINK", "quoteAsset": "USDT",
     "status": "BREAK"},
    {"symbol": "LINKBTC", "baseAsset": "LINK", "quoteAsset": "BTC",
     "status": "TRADING"},
]}
BOOKS = {
    "LINKUSDT": (D("14.00"), D("14.01")),
    "ETHUSDT": (D(1900), D(1901)),
    "BNBUSDT": (D(534), D(535)),
    "LINKBTC": (D("0.0002"), D("0.0002")),
}


def test_candidates_require_a_tradeable_symbol_and_a_known_address():
    from src.scanner.survey import build_candidates

    registry = TokenRegistry.from_coingecko(COINS, chain="ethereum")
    candidates = build_candidates(EXCHANGE_INFO, BOOKS, registry, chain="ethereum")

    symbols = [c.cex_symbol for c in candidates]
    assert symbols == ["LINKUSDT"], (
        "expected only LINK: ETH has no synthetic route, BNB's ticker is "
        "ambiguous, HALTEDUSDT is not TRADING, and LINKBTC is the wrong quote"
    )


def test_eth_itself_is_excluded():
    """The synthetic route converts through ETH, so for ETH the conversion leg is
    a no-op and the comparison is meaningless."""
    from src.scanner.survey import build_candidates

    registry = TokenRegistry.from_coingecko(COINS, chain="ethereum")
    candidates = build_candidates(EXCHANGE_INFO, BOOKS, registry, chain="ethereum")

    assert not any(c.base in ("ETH", "WETH") for c in candidates)


def test_candidates_are_ordered_by_prominence_not_alphabetically():
    """A survey has to stop somewhere, and it should stop at the illiquid end. The
    first attempt ran alphabetically and spent its whole RPC budget on tokens
    beginning with A."""
    from src.scanner.survey import build_candidates

    coins = [
        {"symbol": "zzz", "platforms": {"ethereum": "0x" + "11" * 20}},
        {"symbol": "aaa", "platforms": {"ethereum": "0x" + "22" * 20}},
    ]
    info = {"symbols": [
        {"symbol": "AAAUSDT", "baseAsset": "AAA", "quoteAsset": "USDT",
         "status": "TRADING"},
        {"symbol": "ZZZUSDT", "baseAsset": "ZZZ", "quoteAsset": "USDT",
         "status": "TRADING"},
    ]}
    books = {"AAAUSDT": (D(1), D(1)), "ZZZUSDT": (D(1), D(1))}
    registry = TokenRegistry.from_coingecko(coins, chain="ethereum")

    candidates = build_candidates(info, books, registry, chain="ethereum")

    assert [c.base for c in candidates] == ["ZZZ", "AAA"], (
        "ZZZ appears first in the source list, so it is the more prominent"
    )


def test_the_limit_keeps_the_most_prominent():
    from src.scanner.survey import build_candidates

    registry = TokenRegistry.from_coingecko(COINS, chain="ethereum")
    candidates = build_candidates(
        EXCHANGE_INFO, BOOKS, registry, chain="ethereum", limit=1
    )

    assert len(candidates) == 1


def test_base_decimals_are_left_unset_for_the_caller_to_read_on_chain():
    """Guessing them would be a 10^n price error. The sentinel makes a caller that
    forgets to fill them in fail loudly rather than quietly mis-price."""
    from src.scanner.survey import build_candidates

    registry = TokenRegistry.from_coingecko(COINS, chain="ethereum")
    candidates = build_candidates(EXCHANGE_INFO, BOOKS, registry, chain="ethereum")

    assert candidates[0].base_decimals == -1


def test_the_summary_separates_tradeable_from_merely_positive():
    """The one positive result in the first live survey was an untradeable
    asset-identity trap. A summary that merged the two would have reported an
    opportunity that cannot be taken."""
    from src.scanner.survey import SurveyResult, summarise

    rows = [
        SurveyResult(cex_symbol="TRAP", chain="ethereum", fee=10000,
                     direction="CEX_to_DEX", net_bps=D(239), gross_bps=D(274),
                     tradeable=False, policy_reason="not on the allowlist"),
        SurveyResult(cex_symbol="REAL", chain="ethereum", fee=500,
                     direction="DEX_to_CEX", net_bps=D(-44), gross_bps=D(-8),
                     tradeable=True, policy_reason=""),
    ]

    summary = summarise(rows, floor_bps=D(5))

    assert summary["above_floor"] == 1
    assert summary["tradeable_above_floor"] == 0
    assert summary["measured"] == 2


def test_the_summary_counts_rpc_failures_separately():
    from src.scanner.survey import SurveyResult, summarise

    rows = [
        SurveyResult(cex_symbol="X", chain="ethereum", fee=None, direction=None,
                     net_bps=None, gross_bps=None, tradeable=True,
                     policy_reason="", rpc_failed=True),
    ]

    summary = summarise(rows, floor_bps=D(5))

    assert summary["rpc_failed"] == 1
    assert summary["measured"] == 0
    assert summary["above_floor"] == 0


def test_ranking_puts_the_best_measured_edge_first():
    from src.scanner.survey import SurveyResult

    rows = [
        SurveyResult(cex_symbol="A", chain="ethereum", fee=3000,
                     direction="CEX_to_DEX", net_bps=D(-50), gross_bps=D(-15),
                     tradeable=True, policy_reason=""),
        SurveyResult(cex_symbol="B", chain="ethereum", fee=500,
                     direction="DEX_to_CEX", net_bps=D(-8), gross_bps=D(27),
                     tradeable=True, policy_reason=""),
    ]

    assert [r.cex_symbol for r in rank(rows)] == ["B", "A"]


def test_ranking_puts_unmeasured_candidates_last():
    """An RPC failure must not sort as if it were a good result."""
    from src.scanner.survey import SurveyResult

    rows = [
        SurveyResult(cex_symbol="FAILED", chain="ethereum", fee=None,
                     direction=None, net_bps=None, gross_bps=None,
                     tradeable=True, policy_reason="", rpc_failed=True),
        SurveyResult(cex_symbol="MEASURED", chain="ethereum", fee=500,
                     direction="DEX_to_CEX", net_bps=D(-99), gross_bps=D(-60),
                     tradeable=True, policy_reason=""),
    ]

    assert [r.cex_symbol for r in rank(rows)] == ["MEASURED", "FAILED"]
