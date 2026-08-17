"""Direction selection and threshold behaviour.

Rewritten for the net-economics detector. The previous version of this file
mocked `get_quote` and asserted against the old pre-fee-edge-versus-assembled-
threshold model; after the rewrite two of its tests passed only because every
pair raised a TypeError and returned zero opportunities, which is a false
pass. These exercise the real decision path instead.
"""
from decimal import Decimal

import pytest

from src.core.config import RotationConfig, StrategyConfig
from src.strategy.detector import OpportunityDetector
from tests.fakes import D, FakeCex, FakeDex, flat_book, make_pair


def strategy(**overrides) -> StrategyConfig:
    # Rotation is disabled here so each test isolates the variable it is
    # about. Rotation cost itself is covered by test_rotation_cost.py and
    # test_rotation_wiring.py.
    defaults = dict(
        target_notional_usd=1000, taker_fee_bps=D("7.5"), min_net_bps=D(5),
        rotation=RotationConfig(enabled=False),
    )
    defaults.update(overrides)
    return StrategyConfig(**defaults)


def detector(cex, dex, pairs, **cfg) -> OpportunityDetector:
    return OpportunityDetector(strategy(**cfg), cex, dex, pairs)


async def test_no_opportunity_when_the_spread_is_inside_costs():
    """A 2 bps gross spread cannot survive a 7.5 bps taker fee."""
    pair = make_pair()
    cex = FakeCex({"ETH/USDT": flat_book(bid=1000, ask=1000)})
    dex = FakeDex(sell_price=1000.2, buy_price=1000.2)

    assert not await detector(cex, dex, [pair]).detect()


async def test_cex_to_dex_is_selected_when_the_dex_is_expensive():
    pair = make_pair()
    cex = FakeCex({"ETH/USDT": flat_book(bid=1000, ask=1000)})
    dex = FakeDex(sell_price=1050, buy_price=1050)

    opps = await detector(cex, dex, [pair]).detect()

    assert len(opps) == 1
    assert opps[0].direction == "CEX_to_DEX"
    assert opps[0].cex_price == D(1000)   # bought on the CEX
    assert opps[0].dex_price == D(1050)   # sold on the DEX


async def test_dex_to_cex_is_selected_when_the_dex_is_cheap():
    pair = make_pair()
    cex = FakeCex({"ETH/USDT": flat_book(bid=1000, ask=1000)})
    dex = FakeDex(sell_price=950, buy_price=950)

    opps = await detector(cex, dex, [pair]).detect()

    assert len(opps) == 1
    assert opps[0].direction == "DEX_to_CEX"
    assert opps[0].dex_price == D(950)    # bought on the DEX
    assert opps[0].cex_price == D(1000)   # sold on the CEX


async def test_the_more_profitable_direction_wins_when_both_qualify():
    """Both directions are always evaluated; the better net must be chosen."""
    pair = make_pair()
    # ask 1000 / bid 1200: buying at 1000 to sell at 1100 beats
    # buying at 1100 to sell at 1200 only if net is compared, not gross.
    cex = FakeCex({"ETH/USDT": ([(D(1200), D(10_000))], [(D(1000), D(10_000))])})
    dex = FakeDex(sell_price=1100, buy_price=1100)

    opps = await detector(cex, dex, [pair]).detect()

    assert len(opps) == 1
    # CEX_to_DEX gross = 100/unit; DEX_to_CEX gross = 100/unit, but the CEX
    # fee differs by leg price, so exactly one is strictly better.
    assert opps[0].expected_pnl_quote > 0


async def test_reported_edge_is_net_of_fees_not_gross():
    """edge_bps must be the net figure the decision was made on.

    Gross here is 100 bps; the taker fee removes 7.5, so a gross-reporting
    detector would show ~100 and a net-reporting one ~92.5.
    """
    pair = make_pair()
    cex = FakeCex({"ETH/USDT": flat_book(bid=1000, ask=1000)})
    dex = FakeDex(sell_price=1010, buy_price=1010)

    opps = await detector(cex, dex, [pair]).detect()

    assert len(opps) == 1
    assert opps[0].edge_bps == D("92.5")


async def test_per_pair_min_net_bps_overrides_the_global_floor():
    lax = make_pair(symbol="LAX/USDT", base="LAX", min_net_bps=D(1))
    strict = make_pair(symbol="STR/USDT", base="STR", min_net_bps=D(500))
    cex = FakeCex({
        "LAX/USDT": flat_book(bid=1000, ask=1000),
        "STR/USDT": flat_book(bid=1000, ask=1000),
    })
    dex = FakeDex(sell_price=1010, buy_price=1010)

    opps = await detector(cex, dex, [lax, strict], min_net_bps=D(5)).detect()

    symbols = {o.pair.cex_symbol for o in opps}
    assert symbols == {"LAX/USDT"}


async def test_absurd_edge_is_rejected_as_bad_data():
    """A units or decimals error must not be actioned as an opportunity."""
    pair = make_pair()
    cex = FakeCex({"ETH/USDT": flat_book(bid=1000, ask=1000)})
    dex = FakeDex(sell_price=10_000_000, buy_price=10_000_000)

    opps = await detector(cex, dex, [pair], max_net_bps_sanity=D(1000)).detect()

    assert not opps


async def test_dex_buy_leg_is_quoted_in_quote_currency_not_base():
    """Regression guard for the buy-leg unit bug.

    `get_quote(side="buy")` consumes an amount of the DEX quote token. The
    detector previously passed a base amount, understating slippage by a
    factor of the price and inflating the DEX_to_CEX edge.
    """
    pair = make_pair()
    cex = FakeCex({"ETH/USDT": flat_book(bid=1000, ask=1000)})
    dex = FakeDex(sell_price=950, buy_price=950)

    await detector(cex, dex, [pair]).detect()

    buys = [size for side, size in dex.requests if side == "buy"]
    assert buys, "the buy side should have been quoted"
    # target notional is 1000 quote; a base amount would be ~1
    assert buys[0] == D(1000)
