"""Record the deep WETH-quoted pools, through the synthetic route.

The depth probe established that the only pools in a 227-asset universe with real size
are quoted in WETH, not stablecoins: LDO/WETH at $16.1m of 1% depth, WBTC/WETH at
$1.9m, LINK/WETH at $398k. Everything stablecoin-quoted outside WBTC is under $50k, and
most is under $20k.

So the remaining question is whether a WETH-quoted pool shows a dislocation against the
exchange that clears its cost floor. Answering it needs a synthetic route, because the
exchange quotes TOKEN in USDT and the pool quotes TOKEN in WETH.

HOW THE TWO SIDES ARE MADE COMPARABLE. The pool's price is TOKEN in WETH. The exchange's
TOKEN/USDT ladder is divided through by the ETH/USDT mid, giving a synthetic TOKEN/WETH
ladder -- which is exactly what a taker would achieve by buying TOKEN with USDT and
buying WETH with USDT. Both sides then quote the same thing and the recorded observation
is internally consistent, with no join to a second dataset at analysis time.

TWO CONSEQUENCES, BOTH AGAINST THE STRATEGY, BOTH STATED RATHER THAN HIDDEN.

  The route costs TWO taker fees, not one: the TOKEN leg and the ETH leg. At 7.5 bps
  each on a 0.30% pool the floor is 30 + 15 = 45 bps, against 12.5 bps for a direct
  stablecoin pair. A synthetic route is a more expensive way to reach a deeper pool, not
  a cheaper one.

  The ETH leg is priced at the mid rather than walked. ETH/USDT depth is enormous
  relative to any size these pools can absorb, so the error is small -- but it is an
  approximation in the strategy's favour and it belongs in the record.
"""
import argparse
import asyncio
import json
import signal
import time
from decimal import Decimal
from pathlib import Path

from research_config import research_config

from src.core.types import MarketPair
from src.exchange.binance import BinanceCexClient
from src.exchange.pool_state import ChainPoolReader
from src.exchange.pool_state_cache import PoolStateCache
from src.exchange.univ3 import UniV3DexClient
from src.research.observations import Observation, ObservationStore

# Minimum 1% depth, in USD, for a pool to be worth recording. Below this the impact at
# any size the strategy would trade exceeds the entire cost budget, so a dislocation
# there is not an opportunity whatever its size.
MIN_DEPTH_USD = 5000.0

ETH_SYMBOL = "ETH/USDT"


def synthetic_ladder(levels, eth_price):
    """A TOKEN/USDT ladder restated in WETH.

    Each price is divided by the ETH price; sizes are unchanged, since they are already
    in base units. This is what a taker achieves by buying TOKEN with USDT and buying
    WETH with USDT -- two legs, hence two taker fees at analysis time.
    """
    return [(price / eth_price, size) for price, size in levels]


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default="data/observations_synthetic.sqlite3")
    parser.add_argument("--depth-probe", default="research/depth_probe.json")
    parser.add_argument("--targets", default="research/targets_wide.json")
    parser.add_argument("--interval", type=float, default=15.0)
    parser.add_argument("--cycles", type=int, default=None)
    parser.add_argument("--min-depth", type=float, default=MIN_DEPTH_USD)
    parser.add_argument("--max-pools", type=int, default=12)
    args = parser.parse_args()

    config = research_config("INFO")
    config.network.rpc_max_concurrency = 12
    config.network.rpc_requests_per_second_by_chain = {
        "ethereum": 6.0, "arbitrum": 8.0, "base": 3.0, "bsc": 5.0,
    }

    probe = json.loads(Path(args.depth_probe).read_text(encoding="utf-8"))
    wide = json.loads(Path(args.targets).read_text(encoding="utf-8"))
    identities = {}
    for t in wide:
        identities[(t["chain"], t["base"])] = (t["base_address"], t["base_decimals"])

    deep = [
        p for p in probe
        if p["quote"] == "WETH" and p["depth_usd"] >= args.min_depth
    ]
    deep.sort(key=lambda p: -p["depth_usd"])
    deep = deep[:args.max_pools]
    if not deep:
        print(f"no WETH-quoted pool has {args.min_depth:,.0f} of 1% depth")
        return

    print(f"{len(deep)} WETH-quoted pools with at least "
          f"{args.min_depth:,.0f} of 1% depth:")
    for p in deep:
        print(f"  {p['asset']:<8} {p['chain']:<9} fee {p['fee']:>5}  "
              f"1% depth ~${p['depth_usd']:>14,.0f}")

    dex = UniV3DexClient(config.dex, config.network, config.secrets, config.tokens)
    cache = PoolStateCache(ChainPoolReader(dex), full_reread_seconds=600.0)

    targets = []
    for p in deep:
        chain, asset = p["chain"], p["asset"]
        identity = identities.get((chain, asset))
        weth = config.tokens.get("WETH", {}).get(chain)
        if identity is None or weth is None:
            continue
        base_address, base_decimals = identity
        pool_address = await dex.get_pool_address_by_tokens(
            chain, base_address, weth.address, p["fee"]
        ) if hasattr(dex, "get_pool_address_by_tokens") else None
        targets.append({
            "asset": asset, "chain": chain, "fee": p["fee"],
            "base_address": base_address, "base_decimals": base_decimals,
            "weth_address": weth.address, "weth_decimals": weth.decimals,
            "depth_usd": p["depth_usd"],
            "pool_address": pool_address,
        })

    # Pool addresses come from the factory, by address pair, so no substitution is
    # possible. Resolved once at startup: it is a constant.
    from web3 import Web3
    FACTORY_ABI = [
        {"inputs": [{"name": "tokenA", "type": "address"},
                    {"name": "tokenB", "type": "address"},
                    {"name": "fee", "type": "uint24"}],
         "name": "getPool",
         "outputs": [{"name": "pool", "type": "address"}],
         "stateMutability": "view", "type": "function"},
    ]
    for target in targets:
        if target["pool_address"]:
            continue
        chain = target["chain"]
        factory_address = getattr(config.dex.uniswap_v3.get(chain), "factory", None)
        w3 = dex._get_w3(chain)
        factory = w3.eth.contract(
            address=Web3.to_checksum_address(factory_address), abi=FACTORY_ABI
        )
        target["pool_address"] = await dex._rpc(
            chain,
            factory.functions.getPool(
                Web3.to_checksum_address(target["base_address"]),
                Web3.to_checksum_address(target["weth_address"]),
                target["fee"],
            ).call,
        )

    targets = [t for t in targets if t["pool_address"]]
    print(f"\n{len(targets)} pools resolved")

    symbols = sorted({f"{t['asset']}/USDT" for t in targets} | {ETH_SYMBOL})
    pairs = {}
    for symbol in symbols:
        base = symbol.split("/")[0]
        chain = next((t["chain"] for t in targets if t["asset"] == base), "ethereum")
        pairs[symbol] = MarketPair(
            base=base, quote_cex="USDT", quote_dex="USDT",
            cex_symbol=symbol, dex_chain=chain, dex_pool_fee=3000,
        )
    cex = BinanceCexClient(config.cex, config.secrets, list(pairs.values()))
    await cex.connect()
    print(f"subscribed to {len(pairs)} symbols (including {ETH_SYMBOL} as the bridge)")
    await asyncio.sleep(15)

    run_id = f"syn-{int(time.time())}"
    store = ObservationStore(args.db, run_id=run_id)
    stopping = False

    def request_stop(*_):
        nonlocal stopping
        stopping = True
        print("\nstop requested")

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(sig, request_stop)
        except (ValueError, OSError):
            pass

    cycles = written = skipped_no_eth = 0
    print(f"run {run_id}: recording every {args.interval}s into {args.db}")
    try:
        while not stopping:
            cycle_start = time.monotonic()
            eth_book = await cex.get_book(pairs[ETH_SYMBOL])
            if eth_book is None or not eth_book.bids or not eth_book.asks:
                skipped_no_eth += 1
                await asyncio.sleep(args.interval)
                cycles += 1
                continue
            eth_mid = (eth_book.bids[0][0] + eth_book.asks[0][0]) / 2

            async def observe(target):
                symbol = f"{target['asset']}/USDT"
                pair = pairs[symbol]
                book = await cex.get_book(pair)
                if book is None or not book.bids or not book.asks:
                    return None
                try:
                    snapshot = await cache.refresh(
                        target["chain"], target["pool_address"],
                        decimals0=None, decimals1=None,
                    )
                except Exception:
                    return None
                return Observation(
                    ts=time.time(),
                    # Recorded as TOKEN/WETH: that is what both sides now quote, and
                    # labelling it TOKEN/USDT would invite the wrong cost model.
                    cex_symbol=f"{target['asset']}/WETH-synthetic",
                    base=target["asset"], quote="WETH", chain=target["chain"],
                    pool_fee=int(target["fee"]),
                    pool_address=target["pool_address"],
                    cex_bids=synthetic_ladder(book.bids, eth_mid),
                    cex_asks=synthetic_ladder(book.asks, eth_mid),
                    cex_feed_ts=getattr(book, "feed_timestamp", None),
                    pool=snapshot,
                    gas_price_wei=None, native_price_quote=None,
                    rpc_endpoint=config.network.rpc_urls.get(target["chain"]),
                    run_id=run_id,
                )

            results = await asyncio.gather(
                *(observe(t) for t in targets), return_exceptions=True
            )
            for result in results:
                if isinstance(result, BaseException) or result is None:
                    continue
                store.record(result)
                written += 1

            cycles += 1
            if args.cycles is not None and cycles >= args.cycles:
                break
            elapsed = time.monotonic() - cycle_start
            print(f"cycle {cycles}: {written:,} rows, {elapsed:.1f}s, "
                  f"ETH mid {float(eth_mid):.2f}")
            if args.interval - elapsed > 0:
                await asyncio.sleep(args.interval - elapsed)
    finally:
        print(f"\n--- synthetic recorder stopped ---")
        print(f"  cycles {cycles}, rows {written:,}, in store {store.count():,}")
        print(f"  cycles skipped for no ETH book: {skipped_no_eth}")
        print(f"  {dex._rpc_limiter.describe()}")
        store.close()
        await cex.close()


asyncio.run(main())
