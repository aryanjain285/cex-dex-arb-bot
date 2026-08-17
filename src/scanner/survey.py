"""Does a tradeable CEX-DEX spread exist anywhere, at the size we intend to trade?

This answers the question that comes before any execution work. It needs no
Graph API key, which the shipped pool dataset had implied: everything here is
directly knowable.

    Binance exchangeInfo          every spot symbol and its trading status
    CoinGecko /coins/list         canonical token addresses per chain, free
    Uniswap v3 factory            does a pool exist, per fee tier
    QuoterV2                      at what price for THIS size, net of fee+impact

THE SYNTHETIC ROUTE, NOT DIRECT PAIRS

The first version of this quoted TOKEN/USDT and TOKEN/USDC and reverted almost
everywhere. Altcoin liquidity on Uniswap v3 pairs against WETH, not against
stablecoins -- which is exactly why this codebase has synthetic pairs. So the
survey prices the route the strategy would actually take:

    DEX price of TOKEN in USDT = (WETH per TOKEN) x (ETH/USDT on the CEX)

That conversion is a second CEX taker leg, so the fee is charged twice. Fifteen
basis points before gas and rotation, not seven and a half.

WHAT THE NUMBERS ARE AND ARE NOT

The CEX side is TOP OF BOOK. It ignores depth, so every figure here is
OPTIMISTIC: a real opportunity would still have to survive walking the ladder.
This is a screen -- it says where to point the real measurement, and its output
must never be read as an achievable edge.

The DEX side is QuoterV2 at the full notional, so it already contains the pool
fee and the price impact of the actual size.

Costs are the production model, unsoftened: two taker legs, live gas, and the
same amortised rotation charge the detector applies.

A POSITIVE RESULT IS USUALLY A TRAP

Run over the 45 most prominent Binance tokens with Ethereum addresses, 29 had a
quotable TOKEN/WETH pool at a 1000 notional and exactly one showed a positive net
edge: BNB at +239 bps, through a genuine pool holding 185 BNB and 53 WETH.

That pool prices the LEGACY ERC-20 BNB at 0xb8c77482. Binance settles BNB on BNB
Chain, not as that ERC-20, so the two legs are different assets in custody terms
and the trade cannot settle -- which is why a 2.7% gap has been left standing by
everyone else too. Every result therefore carries the token policy's verdict, so
a reader does not have to notice that for themselves.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

from loguru import logger

from ..core.types import MarketPair
from ..exchange.errors import RpcError
from ..strategy.costs import evaluate_trade

__all__ = [
    "TokenRegistry",
    "SurveyCandidate",
    "SurveyResult",
    "evaluate_candidate",
    "rank",
    "COINGECKO_PLATFORMS",
    "WETH_ADDRESSES",
]

# CoinGecko platform id -> our chain name.
COINGECKO_PLATFORMS = {
    "ethereum": "ethereum",
    "arbitrum-one": "arbitrum",
    "base": "base",
}

WETH_ADDRESSES = {
    "ethereum": "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2",
    "arbitrum": "0x82aF49447D8a07e3bd95BD0d56f35241523fBab1",
    "base": "0x4200000000000000000000000000000000000006",
}

DEFAULT_FEE_TIERS = (3000, 500, 10000)
TEN_THOUSAND = Decimal("10000")


@dataclass(frozen=True)
class TokenRegistry:
    """Ticker -> address on one chain, with collisions removed rather than guessed."""

    addresses: Dict[str, str]
    ambiguous: Set[str]
    prominence: Dict[str, int]

    @classmethod
    def from_coingecko(cls, coins: Iterable[dict], chain: str) -> "TokenRegistry":
        platform = next(
            (cg for cg, ours in COINGECKO_PLATFORMS.items() if ours == chain), chain
        )
        claims: Dict[str, Set[str]] = defaultdict(set)
        prominence: Dict[str, int] = {}
        for index, coin in enumerate(coins):
            symbol = str(coin.get("symbol", "")).upper()
            address = (coin.get("platforms") or {}).get(platform)
            if not symbol or not address or not str(address).startswith("0x"):
                continue
            claims[symbol].add(str(address))
            prominence.setdefault(symbol, index)

        addresses, ambiguous = {}, set()
        for symbol, found in claims.items():
            # Compared case-insensitively: the same address in two casings is one
            # token, not a collision.
            distinct = {a.lower() for a in found}
            if len(distinct) == 1:
                addresses[symbol] = sorted(found)[0]
            else:
                # Two coins claiming one ticker cannot be told apart by price, and
                # a ticker collision is how a counterfeit token enters a universe.
                ambiguous.add(symbol)
        return cls(addresses=addresses, ambiguous=ambiguous, prominence=prominence)

    def address(self, symbol: str) -> Optional[str]:
        return self.addresses.get(str(symbol).upper())


@dataclass(frozen=True)
class SurveyCandidate:
    cex_symbol: str
    base: str
    quote: str
    base_address: str
    base_decimals: int
    chain: str
    cex_bid: Decimal
    cex_ask: Decimal


@dataclass
class SurveyResult:
    cex_symbol: str
    chain: str
    fee: Optional[int]
    direction: Optional[str]
    # None when nothing could be measured -- an RPC failure, or no quotable pool.
    # A zero here would sort as though it were a near-miss.
    net_bps: Optional[Decimal]
    gross_bps: Optional[Decimal]
    tradeable: bool
    policy_reason: str
    rpc_failed: bool = False


async def evaluate_candidate(
    candidate: SurveyCandidate,
    dex_client,
    *,
    eth_bid: Decimal,
    eth_ask: Decimal,
    notional: Decimal,
    taker_fee_bps: Decimal,
    rotation_quote: Decimal,
    fee_tiers: Sequence[int] = DEFAULT_FEE_TIERS,
    cex_legs: int = 2,
    token_policy=None,
) -> Optional[SurveyResult]:
    """Best net edge available on one token, across fee tiers and both directions.

    Returns None when no tier could be quoted at all, and a result with
    `net_bps=None, rpc_failed=True` when the node would not answer -- those are
    different facts and the caller acts differently on each.
    """
    weth = WETH_ADDRESSES.get(candidate.chain)
    if weth is None:
        logger.debug(f"No WETH address known for {candidate.chain}")
        return None

    tradeable, policy_reason = True, ""
    if token_policy is not None:
        verdict = token_policy.check(candidate.base, candidate.quote)
        tradeable, policy_reason = verdict.allowed, verdict.reason

    best: Optional[Tuple[Decimal, Decimal, int, str]] = None
    rpc_failed = False

    for fee in fee_tiers:
        pair = MarketPair(
            base=candidate.base, quote_cex=candidate.quote, quote_dex="WETH",
            cex_symbol=candidate.cex_symbol, dex_chain=candidate.chain,
            dex_pool_fee=fee,
            base_address=candidate.base_address, quote_address=weth,
            base_decimals=candidate.base_decimals, quote_decimals=18,
            is_synthetic=True, intermediate_symbol="ETH",
        )

        # CEX_to_DEX: buy base on the CEX at the ask, sell it on chain for WETH,
        # convert WETH to quote at the CEX ETH BID -- we are selling ETH there.
        size_base = notional / candidate.cex_ask
        try:
            sell = await dex_client.get_quote(
                pair, size=size_base, side="sell", estimate_gas=True
            )
        except RpcError:
            rpc_failed = True
            sell = None
        if sell is not None and sell.price > 0:
            econ = evaluate_trade(
                direction="CEX_to_DEX", size_base=size_base,
                cex_price=candidate.cex_ask, dex_price=sell.price * eth_bid,
                taker_fee_bps=taker_fee_bps, gas_quote=sell.gas_cost_quote,
                cex_legs=cex_legs, rotation_cost_quote=rotation_quote,
            )
            if econ is not None:
                gross = econ.gross_quote / econ.notional_quote * TEN_THOUSAND
                if best is None or econ.net_bps > best[0]:
                    best = (econ.net_bps, gross, fee, "CEX_to_DEX")

        # DEX_to_CEX: buy ETH on the CEX at the ask, spend WETH on chain for base,
        # sell base on the CEX at the bid.
        try:
            buy = await dex_client.get_quote(
                pair, size=notional / eth_ask, side="buy", estimate_gas=True
            )
        except RpcError:
            rpc_failed = True
            buy = None
        if buy is not None and buy.price > 0:
            dex_price_quote = buy.price * eth_ask
            econ = evaluate_trade(
                direction="DEX_to_CEX", size_base=notional / dex_price_quote,
                cex_price=candidate.cex_bid, dex_price=dex_price_quote,
                taker_fee_bps=taker_fee_bps, gas_quote=buy.gas_cost_quote,
                cex_legs=cex_legs, rotation_cost_quote=rotation_quote,
            )
            if econ is not None:
                gross = econ.gross_quote / econ.notional_quote * TEN_THOUSAND
                if best is None or econ.net_bps > best[0]:
                    best = (econ.net_bps, gross, fee, "DEX_to_CEX")

    if best is None:
        if rpc_failed:
            return SurveyResult(
                cex_symbol=candidate.cex_symbol, chain=candidate.chain, fee=None,
                direction=None, net_bps=None, gross_bps=None,
                tradeable=tradeable, policy_reason=policy_reason, rpc_failed=True,
            )
        return None

    net_bps, gross_bps, fee, direction = best
    return SurveyResult(
        cex_symbol=candidate.cex_symbol, chain=candidate.chain, fee=fee,
        direction=direction, net_bps=net_bps, gross_bps=gross_bps,
        tradeable=tradeable, policy_reason=policy_reason, rpc_failed=rpc_failed,
    )


def build_candidates(
    exchange_info: dict,
    books: Dict[str, Tuple[Decimal, Decimal]],
    registry: TokenRegistry,
    chain: str,
    quote_asset: str = "USDT",
    limit: Optional[int] = None,
) -> List[SurveyCandidate]:
    """Tradeable Binance symbols whose base token has an address on this chain.

    Ordered by CoinGecko prominence, because a survey that has to stop somewhere
    should stop at the illiquid end rather than the alphabetical one -- the first
    attempt at this ran alphabetically and spent its whole RPC budget on tokens
    beginning with A.

    Base decimals are NOT filled in here: they must be read from the ERC-20
    contract, and a survey that guessed them would produce 10^n price errors.
    """
    candidates: List[Tuple[int, SurveyCandidate]] = []
    for item in exchange_info.get("symbols", []):
        if item.get("status") != "TRADING":
            continue
        if item.get("quoteAsset") != quote_asset:
            continue
        base = item.get("baseAsset", "")
        # ETH itself has no synthetic route: base and intermediate would be the
        # same asset, so the "conversion" leg is a no-op and the comparison is
        # meaningless.
        if base in ("ETH", "WETH"):
            continue
        address = registry.address(base)
        if address is None:
            continue
        book = books.get(item["symbol"])
        if book is None:
            continue
        candidates.append((
            registry.prominence.get(base.upper(), 10**9),
            SurveyCandidate(
                cex_symbol=item["symbol"], base=base, quote=quote_asset,
                base_address=address, base_decimals=-1, chain=chain,
                cex_bid=book[0], cex_ask=book[1],
            ),
        ))
    candidates.sort(key=lambda pair: pair[0])
    ordered = [c for _, c in candidates]
    return ordered if limit is None else ordered[:limit]


def summarise(results: Sequence[SurveyResult], floor_bps: Decimal) -> dict:
    """Counts a reader needs before looking at any individual row.

    `tradeable_above_floor` is the number that matters, and it is deliberately
    separate from `above_floor`: the one positive result in the first live survey
    was an untradeable asset-identity trap, and a summary that merged them would
    have reported an opportunity.
    """
    measured = [r for r in results if r.net_bps is not None]
    positive = [r for r in measured if r.net_bps > 0]
    above = [r for r in measured if r.net_bps > floor_bps]
    return {
        "candidates": len(results),
        "measured": len(measured),
        "rpc_failed": sum(1 for r in results if r.rpc_failed),
        "positive": len(positive),
        "above_floor": len(above),
        "tradeable_above_floor": len([r for r in above if r.tradeable]),
        "best_gross_bps": (
            max((r.gross_bps for r in measured), default=None)
        ),
    }


def rank(results: Iterable[SurveyResult]) -> List[SurveyResult]:
    """Best measured edge first; unmeasured candidates last.

    An unmeasured candidate sorts last rather than as a zero, because a zero would
    place an RPC failure above every genuinely negative measurement -- and the
    reader would take it for a near-miss.
    """
    measured = [r for r in results if r.net_bps is not None]
    unmeasured = [r for r in results if r.net_bps is None]
    measured.sort(key=lambda r: r.net_bps, reverse=True)
    unmeasured.sort(key=lambda r: r.cex_symbol)
    return measured + unmeasured
