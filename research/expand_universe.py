"""Find tokens that trade on Binance AND have a Uniswap v3 pool, safely.

The one hypothesis left that could change the answer. The current universe shows a
+2.6 bps STANDING basis against a 12.5 bps cost floor -- the phenomenon exists, is
5x too small, and does not fluctuate. That is a statement about ETH and stablecoins
on three chains, which are the most heavily arbitraged pairs in the market. It says
nothing about a mid-cap token on a 0.30% pool.

The hazard in widening is identity, not economics. This project has already met a
counterfeit WETH and a legacy BNB contract. A wrong address does not fail loudly: it
produces a price, and therefore a dislocation, and therefore a finding. Three guards,
in increasing strength:

  AMBIGUITY IS REFUSAL. CoinGecko lists many distinct coins sharing a ticker. If more
  than one has an address on the chain in question, the symbol is ambiguous and the
  candidate is dropped. Picking the most plausible one is exactly how a counterfeit
  gets recorded.

  THE CHAIN MUST AGREE. The ERC-20 symbol() is read on-chain and must match the
  Binance ticker. That catches a mapping error at the source rather than downstream.

  THE POOL MUST CONTAIN IT. The factory is asked for the pool by address pair, so a
  pool that exists for a different token cannot be substituted.

Nothing here trades. A bad identity would corrupt research rather than lose money,
which is why widening is worth doing at all -- but a corrupted research conclusion is
what would authorise a trade later, so the guards are the ones execution would need.
"""
import asyncio
import json
from collections import defaultdict
from pathlib import Path

import aiohttp
from research_config import research_config
from web3 import Web3

from src.exchange.univ3 import UniV3DexClient

config = research_config("WARNING")

CG_PLATFORM = {"ethereum": "ethereum", "arbitrum": "arbitrum-one", "base": "base"}
QUOTES = ("USDT", "USDC")
FEES = (500, 3000, 10000)
OUT = Path(__file__).with_name("targets_wide.json")

# Already recorded, or a peg question rather than a dislocation question.
SKIP = {"ETH", "WETH", "USDC", "USDT", "ARB", "DAI", "FDUSD", "TUSD", "USDE", "PYUSD"}

ERC20_ABI = [
    {"inputs": [], "name": "symbol",
     "outputs": [{"name": "", "type": "string"}],
     "stateMutability": "view", "type": "function"},
    {"inputs": [], "name": "decimals",
     "outputs": [{"name": "", "type": "uint8"}],
     "stateMutability": "view", "type": "function"},
]

FACTORY_ABI = [
    {"inputs": [{"name": "tokenA", "type": "address"},
                {"name": "tokenB", "type": "address"},
                {"name": "fee", "type": "uint24"}],
     "name": "getPool",
     "outputs": [{"name": "pool", "type": "address"}],
     "stateMutability": "view", "type": "function"},
]

ZERO_ADDRESS = "0x0000000000000000000000000000000000000000"


async def binance_spot_bases():
    """Base assets with a live spot market against USDT or USDC."""
    url = "https://api.binance.com/api/v3/exchangeInfo"
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30)) as s:
        async with s.get(url) as r:
            r.raise_for_status()
            payload = await r.json()
    markets = defaultdict(set)
    for sym in payload["symbols"]:
        if sym.get("status") != "TRADING":
            continue
        if not sym.get("isSpotTradingAllowed"):
            continue
        if sym["quoteAsset"] in QUOTES:
            markets[sym["baseAsset"]].add(sym["quoteAsset"])
    return markets


def coingecko_addresses(coins, chain):
    """{SYMBOL: address} keeping only symbols with EXACTLY ONE token on this chain.

    Ambiguity is refusal. Choosing among several same-ticker coins is precisely how a
    counterfeit enters a dataset, and no tiebreak available here is safer than
    dropping the candidate.
    """
    platform = CG_PLATFORM[chain]
    by_symbol = defaultdict(set)
    for coin in coins:
        address = (coin.get("platforms") or {}).get(platform)
        if address and address.startswith("0x") and len(address) == 42:
            by_symbol[coin["symbol"].upper()].add(address.lower())
    unique, ambiguous = {}, 0
    for symbol, addresses in by_symbol.items():
        if len(addresses) == 1:
            unique[symbol] = next(iter(addresses))
        else:
            ambiguous += 1
    return unique, ambiguous


def factory_for(config, chain):
    contracts = config.dex.uniswap_v3.get(chain)
    if contracts is None:
        return None
    return getattr(contracts, "factory", None)


async def main():
    coins = json.loads(Path("data/coingecko_tokens.json").read_text(encoding="utf-8"))
    markets = await binance_spot_bases()
    print(f"Binance: {len(markets)} base assets with a USDT/USDC spot market")

    client = UniV3DexClient(config.dex, config.network, config.secrets, config.tokens)
    found = []
    checked = 0
    rejected = defaultdict(int)

    for chain in ("ethereum", "arbitrum", "base"):
        if not config.network.rpc_urls.get(chain):
            continue
        factory_address = factory_for(config, chain)
        if not factory_address:
            print(f"{chain}: no factory address configured, skipping")
            continue

        unique, ambiguous = coingecko_addresses(coins, chain)
        candidates = sorted(set(unique) & set(markets) - SKIP)
        print(f"\n{chain}: {len(unique)} unambiguous CoinGecko tokens, "
              f"{ambiguous} ambiguous symbols dropped, "
              f"{len(candidates)} intersect Binance")

        w3 = client._get_w3(chain)
        factory = w3.eth.contract(
            address=Web3.to_checksum_address(factory_address), abi=FACTORY_ABI
        )

        for symbol in candidates:
            address = Web3.to_checksum_address(unique[symbol])
            checked += 1
            try:
                token = w3.eth.contract(address=address, abi=ERC20_ABI)
                onchain_symbol = await client._rpc(chain, token.functions.symbol().call)
                decimals = int(await client._rpc(chain, token.functions.decimals().call))
            except Exception:
                rejected["identity_unreadable"] += 1
                continue

            # Leading W tolerated: WETH on chain is ETH on the exchange, and the same
            # applies to other wrapped natives. Nothing else is.
            if str(onchain_symbol).upper().lstrip("W") != symbol.upper().lstrip("W"):
                rejected["symbol_mismatch"] += 1
                print(f"  REJECT {symbol}: the chain calls it {onchain_symbol!r}")
                continue

            for quote in sorted(markets[symbol]):
                quote_token = config.tokens.get(quote, {}).get(chain)
                if quote_token is None:
                    continue
                quote_address = Web3.to_checksum_address(quote_token.address)
                for fee in FEES:
                    try:
                        pool = await client._rpc(
                            chain,
                            factory.functions.getPool(address, quote_address, fee).call,
                        )
                    except Exception:
                        rejected["factory_call_failed"] += 1
                        continue
                    if not pool or pool == ZERO_ADDRESS:
                        continue
                    found.append({
                        "base": symbol, "quote": quote,
                        "cex_symbol": f"{symbol}/{quote}",
                        "chain": chain, "fee": fee,
                        "pool_address": Web3.to_checksum_address(pool),
                        "base_address": address, "quote_address": quote_address,
                        "base_decimals": decimals,
                        "quote_decimals": quote_token.decimals,
                        "onchain_symbol": str(onchain_symbol),
                    })
                    print(f"  {symbol}/{quote:<5s} {chain:<9s} {fee:>5}  {pool}")

    OUT.write_text(json.dumps(found, indent=2), encoding="utf-8")
    print(f"\nchecked {checked} token identities")
    print(f"rejected: {dict(rejected)}")
    print(f"{len(found)} pools -> {OUT}")
    by_chain = defaultdict(int)
    for f in found:
        by_chain[f["chain"]] += 1
    print("by chain:", dict(by_chain))
    print(f"distinct assets: {len({f['base'] for f in found})}")


asyncio.run(main())
