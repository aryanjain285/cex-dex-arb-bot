"""Synthetic pair pricing must use the correct side of the intermediate book.

A synthetic trade routes through an intermediate asset: CEX buy base ->
on-chain swap base<->INT -> CEX trade INT. Converting the on-chain price into
the quote currency therefore requires the intermediate's BID when you will be
selling INT, and its ASK when you will be buying INT. Getting it backwards
biases every synthetic edge by the full intermediate spread -- and synthetic
pairs are 837 of the 1,062 pools in the shipped dataset.

These tests exist because a mutation test proved the previous suite could not
detect the inversion: the fixtures used a flat intermediate book (bid == ask
== 1), so swapping the two sides changed nothing. Every fixture here uses a
WIDE intermediate spread, which is the only way the assertion has teeth.
"""
from decimal import Decimal

import pytest

from src.core.config import StrategyConfig, TokenPolicyConfig
from src.strategy.detector import OpportunityDetector
from tests.fakes import D, FakeCex, FakeDex, flat_book, make_pair

# Deliberately wide so that bid and ask are distinguishable.
INT_BID = D(1900)
INT_ASK = D(1910)


def strategy(**kw) -> StrategyConfig:
    defaults = dict(target_notional_usd=1000, taker_fee_bps=D("7.5"), min_net_bps=D(5))
    defaults.update(kw)
    # Denylist mode: these tests use invented tickers as neutral
    # placeholders and are not about the token policy. Opting out here
    # keeps test symbols out of the production allowlist.
    defaults.setdefault("token_policy", TokenPolicyConfig(mode="denylist"))
    return StrategyConfig(**defaults)


def synthetic_pair():
    return make_pair(
        symbol="ALT/USDT", base="ALT", quote_dex="ETH",
        is_synthetic=True, intermediate_symbol="ETH",
    )


async def test_cex_to_dex_converts_at_the_intermediate_bid():
    """Selling base on the DEX yields INT, which is then SOLD on the CEX.

    Selling INT means receiving the bid. Using the ask would overstate the
    proceeds by the full intermediate spread.
    """
    dex_sell = D("0.53")            # INT per base
    cex = FakeCex({
        "ALT/USDT": flat_book(bid=1000, ask=1000),
        "ETH/USDT": flat_book(bid=INT_BID, ask=INT_ASK),
    })
    dex = FakeDex(sell_price=dex_sell, buy_price=dex_sell)

    opps = await OpportunityDetector(
        strategy(), cex, dex, [synthetic_pair()]
    ).detect()

    assert opps, "fixture should produce a profitable CEX_to_DEX synthetic"
    opp = opps[0]
    assert opp.direction == "CEX_to_DEX"
    assert opp.dex_price == dex_sell * INT_BID, (
        f"expected conversion at the intermediate BID ({INT_BID}); "
        f"got {opp.dex_price}, which is {opp.dex_price / dex_sell} per INT"
    )


async def test_dex_to_cex_converts_at_the_intermediate_ask():
    """Buying base on the DEX spends INT, which must first be BOUGHT on the
    CEX. Buying INT means paying the ask."""
    dex_buy = D("0.57")
    cex = FakeCex({
        "ALT/USDT": flat_book(bid=1100, ask=1100),
        "ETH/USDT": flat_book(bid=INT_BID, ask=INT_ASK),
    })
    dex = FakeDex(sell_price=dex_buy, buy_price=dex_buy)

    opps = await OpportunityDetector(
        strategy(), cex, dex, [synthetic_pair()]
    ).detect()

    assert opps, "fixture should produce a profitable DEX_to_CEX synthetic"
    opp = opps[0]
    assert opp.direction == "DEX_to_CEX"
    assert opp.dex_price == dex_buy * INT_ASK, (
        f"expected conversion at the intermediate ASK ({INT_ASK}); "
        f"got {opp.dex_price}, which is {opp.dex_price / dex_buy} per INT"
    )


async def test_dex_buy_leg_spends_intermediate_units_not_quote_units():
    """`side="buy"` consumes the DEX quote token, which for a synthetic pair
    is the intermediate asset -- so the notional must be converted into INT at
    the intermediate ask before being passed to the quoter."""
    cex = FakeCex({
        "ALT/USDT": flat_book(bid=1100, ask=1100),
        "ETH/USDT": flat_book(bid=INT_BID, ask=INT_ASK),
    })
    dex = FakeDex(sell_price=D("0.57"), buy_price=D("0.57"))

    await OpportunityDetector(strategy(), cex, dex, [synthetic_pair()]).detect()

    buys = [size for side, size in dex.requests if side == "buy"]
    assert buys, "the buy side should have been quoted"
    assert buys[0] == D(1000) / INT_ASK, (
        "the buy leg must spend notional/intermediate_ask INT units, "
        f"not raw quote units; got {buys[0]}"
    )


async def test_intermediate_spread_is_actually_paid():
    """Sanity check on the whole construction: widening the intermediate
    spread must reduce the CEX_to_DEX edge, because the spread is a real cost
    borne on the third leg."""
    dex_sell = D("0.53")

    async def edge_with(bid, ask):
        # ALT priced low enough that both spread cases stay profitable, so the
        # comparison is between two real edges rather than one absent one.
        cex = FakeCex({
            "ALT/USDT": flat_book(bid=980, ask=980),
            "ETH/USDT": flat_book(bid=bid, ask=ask),
        })
        dex = FakeDex(sell_price=dex_sell, buy_price=dex_sell)
        opps = await OpportunityDetector(
            strategy(min_net_bps=D(0)), cex, dex, [synthetic_pair()]
        ).detect()
        return opps[0].edge_bps if opps else None

    tight = await edge_with(D(1905), D(1906))
    wide = await edge_with(D(1880), D(1930))

    assert tight is not None and wide is not None
    assert tight > wide, "a wider intermediate spread must cost edge"
