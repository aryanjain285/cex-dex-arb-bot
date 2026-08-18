"""Where does mid-cap liquidity actually sit: against stablecoins, or against WETH?

The wide screen found nothing, and two of the reasons were mine rather than the market's.

First, it selected the LOWEST fee tier per asset as a depth proxy. That is backwards for
a volatile mid-cap: its liquidity concentrates in the 0.30% or 1.00% tier, because
providing at 0.05% against a token that moves 5% a day is a losing position. The tier
should be measured, not guessed, and depth is now measurable from slot0.

Second, and more fundamentally: on Uniswap, a mid-cap token trades against WETH, not
against USDC. TOKEN/USDC pools exist because anyone can create one, and they are mostly
abandoned -- which is exactly what the screen found. If the depth is in TOKEN/WETH, then
comparing a TOKEN/USDC pool with a Binance TOKEN/USDT book is measuring the wrong pool,
and the whole universe question has to be asked through a synthetic route:
TOKEN -> WETH on the DEX, priced against TOKEN/USDT and ETH/USDT on the exchange.

So before building that, establish whether the premise holds. This probes every tier of
TOKEN/WETH and TOKEN/USDC for the same assets and compares 1% depth. Batched: a few
hundred pools cost a handful of requests.

No prices, no CEX, no dislocation -- one question only, so the answer is unambiguous.
"""
import asyncio
import json
from collections import defaultdict
from decimal import Decimal
from pathlib import Path

from research_config import research_config
from web3 import Web3

from src.exchange.multicall import Multicall
from src.exchange.pool_state import POOL_ABI
from src.exchange.univ3 import UniV3DexClient
from src.exchange.univ3_math import V3Pool, notional_to_move_price

config = research_config("WARNING")

FEES = (500, 3000, 10000)
SLOT0_OUTPUTS = ["uint160", "int24", "uint16", "uint16", "uint16", "uint8", "bool"]

FACTORY_ABI = [
    {"inputs": [{"name": "tokenA", "type": "address"},
                {"name": "tokenB", "type": "address"},
                {"name": "fee", "type": "uint24"}],
     "name": "getPool",
     "outputs": [{"name": "pool", "type": "address"}],
     "stateMutability": "view", "type": "function"},
]
ZERO = "0x0000000000000000000000000000000000000000"


def encode(contract, name, args=()):
    encoder = getattr(contract, "encode_abi", None)
    if encoder is not None:
        try:
            return encoder(abi_element_identifier=name, args=list(args))
        except TypeError:
            return encoder(name, list(args))
    return contract.encodeABI(fn_name=name, args=list(args))


async def batched(client, multicall, chain, calls, decoder, block=None):
    if await multicall.available(chain):
        raw = await multicall.aggregate(chain, calls, block_number=block)
    else:
        raw = [None] * len(calls)
    out = []
    for data in raw:
        if data is None:
            out.append(None)
            continue
        try:
            out.append(decoder(data))
        except Exception:
            out.append(None)
    return out


async def main():
    wide = json.loads(Path("research/targets_wide.json").read_text(encoding="utf-8"))
    client = UniV3DexClient(config.dex, config.network, config.secrets, config.tokens)
    multicall = Multicall(client)

    # Assets and their verified addresses, per chain, from the expansion run.
    assets = defaultdict(dict)
    decimals = {}
    for t in wide:
        assets[t["chain"]][t["base"]] = t["base_address"]
        decimals[(t["chain"], t["base"])] = t["base_decimals"]

    results = defaultdict(dict)
    for chain, by_symbol in assets.items():
        weth = config.tokens.get("WETH", {}).get(chain)
        usdc = config.tokens.get("USDC", {}).get(chain)
        if weth is None or usdc is None:
            continue
        factory_address = getattr(config.dex.uniswap_v3.get(chain), "factory", None)
        if not factory_address:
            continue

        w3 = client._get_w3(chain)
        factory = w3.eth.contract(
            address=Web3.to_checksum_address(factory_address), abi=FACTORY_ABI
        )
        quotes = {
            "WETH": (Web3.to_checksum_address(weth.address), weth.decimals),
            "USDC": (Web3.to_checksum_address(usdc.address), usdc.decimals),
        }

        queries, meta = [], []
        for symbol, address in sorted(by_symbol.items()):
            if symbol in ("WETH", "USDC", "USDT"):
                continue
            base = Web3.to_checksum_address(address)
            for quote_name, (quote_address, quote_decimals) in quotes.items():
                for fee in FEES:
                    queries.append((
                        Web3.to_checksum_address(factory_address),
                        encode(factory, "getPool", [base, quote_address, fee]),
                    ))
                    meta.append((symbol, base, quote_name, quote_address,
                                 quote_decimals, fee))

        print(f"{chain}: {len(queries)} factory lookups for "
              f"{len(by_symbol)} assets x 2 quotes x {len(FEES)} tiers")
        pools = await batched(
            client, multicall, chain, queries,
            lambda d: w3.codec.decode(["address"], d)[0],
        )

        live = [(m, p) for m, p in zip(meta, pools) if p and str(p) != ZERO]
        print(f"  {len(live)} pools exist")
        if not live:
            continue

        block = await client._rpc(chain, lambda: w3.eth.block_number)
        template = w3.eth.contract(
            address=Web3.to_checksum_address(live[0][1]), abi=POOL_ABI
        )
        slot0_data = encode(template, "slot0")
        liquidity_data = encode(template, "liquidity")
        state_calls = []
        for _, pool in live:
            address = Web3.to_checksum_address(pool)
            state_calls.append((address, slot0_data))
            state_calls.append((address, liquidity_data))

        state = await batched(
            client, multicall, chain, state_calls, lambda d: d, block=block
        )
        for i, (m, pool) in enumerate(live):
            slot0_raw, liquidity_raw = state[2 * i], state[2 * i + 1]
            if slot0_raw is None or liquidity_raw is None:
                continue
            try:
                decoded = w3.codec.decode(SLOT0_OUTPUTS, slot0_raw)
                liquidity = int(w3.codec.decode(["uint128"], liquidity_raw)[0])
            except Exception:
                continue
            symbol, base, quote_name, quote_address, quote_decimals, fee = m
            base_is_token0 = base.lower() < quote_address.lower()
            snapshot = V3Pool(
                sqrt_price_x96=int(decoded[0]), liquidity=liquidity,
                tick=int(decoded[1]), fee=fee, tick_spacing=1, ticks=[],
                decimals0=(decimals[(chain, symbol)] if base_is_token0
                           else quote_decimals),
                decimals1=(quote_decimals if base_is_token0
                           else decimals[(chain, symbol)]),
            )
            depth = notional_to_move_price(snapshot, Decimal("0.01"))
            # Denominated in token1. When the base is token1 the figure is in base
            # units, which is not comparable across assets, so only quote-denominated
            # readings are ranked.
            quote_denominated = base_is_token0
            key = (chain, symbol, quote_name)
            best = results[key].get("depth", Decimal(-1))
            if quote_denominated and depth > best:
                results[key] = {
                    "depth": depth, "fee": fee, "pool": pool,
                    "liquidity": liquidity,
                }

    print()
    print("=" * 92)
    print("1% DEPTH BY QUOTE ASSET -- is mid-cap liquidity against WETH or stablecoins?")
    print("=" * 92)
    # WETH depth is in WETH; USDC depth is in USDC. Converted with a nominal ETH price
    # so the two are comparable; the conclusion is order-of-magnitude, not precise.
    eth_price = Decimal("1900")
    rows = []
    for (chain, symbol, quote_name), info in results.items():
        if not info:
            continue
        usd = info["depth"] * (eth_price if quote_name == "WETH" else Decimal(1))
        rows.append((chain, symbol, quote_name, info["fee"], usd))

    paired = defaultdict(dict)
    for chain, symbol, quote_name, fee, usd in rows:
        paired[(chain, symbol)][quote_name] = (fee, usd)

    both = [(k, v) for k, v in paired.items() if "WETH" in v and "USDC" in v]
    print(f"{len(both)} assets have a quote-denominated pool against BOTH WETH and USDC")
    print()
    print(f"{'asset':<20} {'WETH tier':>10} {'WETH depth $':>16} "
          f"{'USDC tier':>10} {'USDC depth $':>16} {'ratio':>10}")
    weth_deeper = 0
    for (chain, symbol), v in sorted(
        both, key=lambda kv: -(kv[1]["WETH"][1]), reverse=False
    )[:40]:
        wfee, wdepth = v["WETH"]
        ufee, udepth = v["USDC"]
        ratio = (wdepth / udepth) if udepth > 0 else Decimal("Infinity")
        if wdepth > udepth:
            weth_deeper += 1
        print(f"{symbol + ' ' + chain:<20} {wfee:>10} {float(wdepth):>16,.0f} "
              f"{ufee:>10} {float(udepth):>16,.0f} "
              f"{('inf' if udepth == 0 else f'{float(ratio):.1f}x'):>10}")
    print(f"\nWETH deeper in {weth_deeper} of {len(both)} assets")

    print()
    print("Deepest pools overall, any quote:")
    for chain, symbol, quote_name, fee, usd in sorted(rows, key=lambda r: -r[4])[:20]:
        print(f"  {symbol:<10} {chain:<9} vs {quote_name:<5} fee {fee:>5}  "
              f"1% depth ~${float(usd):>14,.0f}")

    Path("research/depth_probe.json").write_text(
        json.dumps(
            [
                {"chain": c, "asset": s, "quote": q, "fee": f, "depth_usd": float(u)}
                for c, s, q, f, u in rows
            ],
            indent=2,
        ),
        encoding="utf-8",
    )
    print("\nwritten to research/depth_probe.json")


asyncio.run(main())
