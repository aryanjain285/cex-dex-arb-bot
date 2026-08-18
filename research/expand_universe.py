"""Find tokens that trade on Binance AND have a Uniswap v3 pool, safely and quickly.

The one hypothesis left that could change the answer. The current universe shows a
+2.6 bps STANDING basis against a 12.5 bps cost floor -- the phenomenon exists, is 5x
too small, and does not fluctuate. That is a statement about ETH and stablecoins on
three chains, which are the most heavily arbitraged pairs in the market. It says
nothing about a mid-cap token on a 0.30% pool.

THE HAZARD IN WIDENING IS IDENTITY, NOT ECONOMICS. This project has already met a
counterfeit WETH and a legacy BNB contract. A wrong address does not fail loudly: it
produces a price, therefore a dislocation, therefore a finding. Three guards, in
increasing strength:

  AMBIGUITY IS REFUSAL. CoinGecko lists many distinct coins sharing a ticker. If more
  than one has an address on the chain in question, the symbol is ambiguous and the
  candidate is dropped. Choosing the most plausible one is exactly how a counterfeit
  enters a dataset, and no tiebreak available here is safer than dropping it.

  THE CHAIN MUST AGREE. The ERC-20 symbol() is read on-chain and must match the
  Binance ticker, so a mapping error is caught at the source rather than downstream.

  THE FACTORY IS ASKED BY ADDRESS PAIR. A pool that exists for some other token cannot
  be substituted, because the query names both token addresses.

Nothing here trades. A bad identity would corrupt research rather than lose money --
which is why widening is worth doing at all -- but a corrupted research conclusion is
what would authorise a trade later, so the guards are the ones execution would need.

EVERYTHING IS BATCHED. A first version made six sequential factory calls per token and
managed 28 pools in twelve minutes against a contended public endpoint. Identity reads
and factory lookups are both pure view calls over a fixed list, which is the ideal
shape for Multicall3: 214 tokens x 6 tiers becomes 22 requests instead of 1,284.
"""
import asyncio
import json
from collections import defaultdict
from pathlib import Path

import aiohttp
from research_config import research_config
from web3 import Web3

from src.exchange.multicall import Multicall
from src.exchange.univ3 import UniV3DexClient

config = research_config("WARNING")

CG_PLATFORM = {"ethereum": "ethereum", "arbitrum": "arbitrum-one", "base": "base"}
QUOTES = ("USDT", "USDC")
FEES = (500, 3000, 10000)
OUT = Path("research/targets_wide.json")

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


def encode(contract, name, args):
    encoder = getattr(contract, "encode_abi", None)
    if encoder is not None:
        try:
            return encoder(abi_element_identifier=name, args=list(args))
        except TypeError:
            return encoder(name, list(args))
    return contract.encodeABI(fn_name=name, args=list(args))


async def binance_spot_bases():
    """Base assets with a live spot market against USDT or USDC."""
    url = "https://api.binance.com/api/v3/exchangeInfo"
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30)) as s:
        async with s.get(url) as r:
            r.raise_for_status()
            payload = await r.json()
    markets = defaultdict(set)
    for sym in payload["symbols"]:
        if sym.get("status") != "TRADING" or not sym.get("isSpotTradingAllowed"):
            continue
        if sym["quoteAsset"] in QUOTES:
            markets[sym["baseAsset"]].add(sym["quoteAsset"])
    return markets


def coingecko_addresses(coins, chain):
    """{SYMBOL: address} keeping only symbols with EXACTLY ONE token on this chain."""
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


def factory_for(chain):
    contracts = config.dex.uniswap_v3.get(chain)
    return getattr(contracts, "factory", None) if contracts else None


async def read_identities(client, multicall, chain, w3, symbols, unique):
    """{SYMBOL: (onchain_symbol, decimals)} for every candidate, in two batched passes."""
    addresses = {s: Web3.to_checksum_address(unique[s]) for s in symbols}
    template = w3.eth.contract(address=next(iter(addresses.values())), abi=ERC20_ABI)
    symbol_data = encode(template, "symbol", [])
    decimals_data = encode(template, "decimals", [])

    calls = []
    for symbol in symbols:
        calls.append((addresses[symbol], symbol_data))
        calls.append((addresses[symbol], decimals_data))

    if await multicall.available(chain):
        raw = await multicall.aggregate(chain, calls)
    else:
        raw = []
        for target, data in calls:
            contract = w3.eth.contract(address=target, abi=ERC20_ABI)
            name = "symbol" if data == symbol_data else "decimals"
            try:
                value = await client._rpc(
                    chain, getattr(contract.functions, name)().call
                )
                raw.append(value)
            except Exception:
                raw.append(None)

    out = {}
    for i, symbol in enumerate(symbols):
        sym_raw, dec_raw = raw[2 * i], raw[2 * i + 1]
        if sym_raw is None or dec_raw is None:
            continue
        try:
            if isinstance(sym_raw, (bytes, bytearray)):
                onchain = w3.codec.decode(["string"], sym_raw)[0]
                decimals = int(w3.codec.decode(["uint8"], dec_raw)[0])
            else:
                onchain, decimals = str(sym_raw), int(dec_raw)
        except Exception:
            continue
        out[symbol] = (str(onchain), decimals)
    return out


async def find_pools(client, multicall, chain, w3, queries):
    """queries: [(base_address, quote_address, fee)] -> [pool_address or None]."""
    factory_address = factory_for(chain)
    factory = w3.eth.contract(
        address=Web3.to_checksum_address(factory_address), abi=FACTORY_ABI
    )
    calls = [
        (Web3.to_checksum_address(factory_address),
         encode(factory, "getPool", [base, quote, fee]))
        for base, quote, fee in queries
    ]
    if await multicall.available(chain):
        raw = await multicall.aggregate(chain, calls)
        out = []
        for data in raw:
            if data is None:
                out.append(None)
                continue
            try:
                out.append(w3.codec.decode(["address"], data)[0])
            except Exception:
                out.append(None)
        return out

    out = []
    for base, quote, fee in queries:
        try:
            out.append(await client._rpc(
                chain, factory.functions.getPool(base, quote, fee).call
            ))
        except Exception:
            out.append(None)
    return out


async def main():
    coins = json.loads(Path("data/coingecko_tokens.json").read_text(encoding="utf-8"))
    markets = await binance_spot_bases()
    print(f"Binance: {len(markets)} base assets with a USDT/USDC spot market")

    client = UniV3DexClient(config.dex, config.network, config.secrets, config.tokens)
    multicall = Multicall(client)
    found = []
    rejected = defaultdict(int)
    checked = 0

    for chain in ("ethereum", "arbitrum", "base"):
        if not config.network.rpc_urls.get(chain) or not factory_for(chain):
            continue
        unique, ambiguous = coingecko_addresses(coins, chain)
        candidates = sorted((set(unique) & set(markets)) - SKIP)
        print(f"\n{chain}: {len(unique)} unambiguous CoinGecko tokens, "
              f"{ambiguous} ambiguous symbols dropped, "
              f"{len(candidates)} intersect Binance")
        if not candidates:
            continue

        w3 = client._get_w3(chain)
        identities = await read_identities(
            client, multicall, chain, w3, candidates, unique
        )
        checked += len(candidates)
        rejected["identity_unreadable"] += len(candidates) - len(identities)

        verified = {}
        for symbol, (onchain, decimals) in identities.items():
            # Leading W tolerated: WETH on chain is ETH on the exchange, and the same
            # holds for other wrapped natives. Nothing else is.
            if onchain.upper().lstrip("W") != symbol.upper().lstrip("W"):
                rejected["symbol_mismatch"] += 1
                print(f"  REJECT {symbol}: the chain calls it {onchain!r}")
                continue
            verified[symbol] = decimals
        print(f"  {len(verified)} identities confirmed on chain")

        queries, meta = [], []
        for symbol, decimals in sorted(verified.items()):
            base_address = Web3.to_checksum_address(unique[symbol])
            for quote in sorted(markets[symbol]):
                quote_token = config.tokens.get(quote, {}).get(chain)
                if quote_token is None:
                    continue
                quote_address = Web3.to_checksum_address(quote_token.address)
                for fee in FEES:
                    queries.append((base_address, quote_address, fee))
                    meta.append((symbol, decimals, base_address, quote,
                                 quote_address, quote_token.decimals, fee))

        print(f"  {len(queries)} factory lookups, batched")
        pools = await find_pools(client, multicall, chain, w3, queries)
        for (symbol, decimals, base_address, quote, quote_address,
             quote_decimals, fee), pool in zip(meta, pools):
            if not pool or str(pool) == ZERO_ADDRESS:
                continue
            found.append({
                "base": symbol, "quote": quote,
                "cex_symbol": f"{symbol}/{quote}",
                "chain": chain, "fee": fee,
                "pool_address": Web3.to_checksum_address(pool),
                "base_address": base_address, "quote_address": quote_address,
                "base_decimals": decimals, "quote_decimals": quote_decimals,
                "onchain_symbol": identities[symbol][0],
            })
        print(f"  {sum(1 for f in found if f['chain'] == chain)} pools on {chain}")

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
