"""A wide, cheap, block-consistent screen for raw dislocation.

The full recorder reads tick liquidity so that any SIZE can be priced later. That
costs 9-12 batched calls per pool and limits a run to a couple of dozen pools. But the
question a wide survey asks is not "what size is optimal" -- it is "does the pool price
ever differ from the exchange price by more than the fees", and that needs only the
pool's spot price.

Spot price is `slot0`, one call. Batched through Multicall3 at 60 per request, 180 pools
cost three calls per chain plus one for the block. Roughly 500x cheaper per pool than a
full read, which is what makes a several-hundred-pool universe observable at all.

WHAT THIS DELIBERATELY CANNOT DO. A screen observation carries no tick data, so it
cannot be priced at any size -- and it does not pretend to. The recorded snapshot has an
empty tick set and no observed window, so `price_for_amount_in` refuses, and the report
counts every screen row as unpriceable. That is the honest representation: the screen
measures the raw dislocation and nothing else. Anything it shortlists gets a full
block-pinned read from the real recorder before any claim about size or capacity.

The block IS pinned, so every pool on a chain in one cycle describes the same instant.
Without that, a cycle spanning several blocks would mix pool states from different
moments against a single CEX book, and the resulting "dislocation" would partly be the
chain moving underneath the measurement.
"""
import argparse
import asyncio
import json
import signal
import time
from collections import defaultdict
from decimal import Decimal
from pathlib import Path

from research_config import research_config
from web3 import Web3

from src.core.types import MarketPair
from src.exchange.binance import BinanceCexClient
from src.exchange.multicall import Multicall
from src.exchange.pool_state import POOL_ABI, PoolSnapshot
from src.exchange.univ3 import UniV3DexClient
from src.research.observations import Observation, ObservationStore

# Uniswap v3 slot0 return signature. Only sqrtPriceX96 and tick are used.
SLOT0_OUTPUTS = ["uint160", "int24", "uint16", "uint16", "uint16", "uint8", "bool"]


async def screen_chain(client, multicall, chain, targets, w3):
    """{pool_address: (sqrt_price_x96, tick, block)} for every target on one chain."""
    if not targets:
        return {}
    block = await client._rpc(chain, lambda: w3.eth.block_number)
    pool_contract = w3.eth.contract(
        address=Web3.to_checksum_address(targets[0]["pool_address"]), abi=POOL_ABI
    )
    encoder = getattr(pool_contract, "encode_abi", None)
    if encoder is not None:
        try:
            calldata = encoder(abi_element_identifier="slot0", args=[])
        except TypeError:
            calldata = encoder("slot0", [])
    else:
        calldata = pool_contract.encodeABI(fn_name="slot0", args=[])

    calls = [
        (Web3.to_checksum_address(t["pool_address"]), calldata) for t in targets
    ]
    if not await multicall.available(chain):
        # No batching: one call per pool, still only one call each.
        out = {}
        for target in targets:
            contract = w3.eth.contract(
                address=Web3.to_checksum_address(target["pool_address"]), abi=POOL_ABI
            )
            try:
                slot0 = await client._rpc(
                    chain,
                    lambda c=contract: c.functions.slot0().call(block_identifier=block),
                )
            except Exception:
                continue
            out[target["pool_address"].lower()] = (int(slot0[0]), int(slot0[1]), block)
        return out

    raw = await multicall.aggregate(chain, calls, block_number=block)
    out = {}
    for target, data in zip(targets, raw):
        if data is None:
            continue
        try:
            decoded = w3.codec.decode(SLOT0_OUTPUTS, data)
        except Exception:
            continue
        out[target["pool_address"].lower()] = (int(decoded[0]), int(decoded[1]), block)
    return out


def screen_snapshot(target, sqrt_price_x96, tick, block, base_is_token0):
    """A snapshot carrying spot price only.

    Empty ticks and no observed window, on purpose. That makes it unquotable at every
    size rather than quotable against invented liquidity, which is the exact failure
    already fixed once in the swap math -- and the report then counts every screen row
    as unpriceable, which is the honest description of what a screen measured.

    Token order comes from the addresses. Uniswap v3 orders a pool's tokens by address,
    so this is derivable rather than something to read from the chain per cycle; getting
    it wrong inverts the price by a factor of price squared.
    """
    return PoolSnapshot(
        sqrt_price_x96=sqrt_price_x96,
        liquidity=0,
        tick=tick,
        fee=int(target["fee"]),
        tick_spacing=1,
        ticks=[],
        decimals0=(target["base_decimals"] if base_is_token0
                   else target["quote_decimals"]),
        decimals1=(target["quote_decimals"] if base_is_token0
                   else target["base_decimals"]),
        block_number=block,
        address=target["pool_address"],
        token0=(target["base_address"] if base_is_token0
                else target["quote_address"]),
        token1=(target["quote_address"] if base_is_token0
                else target["base_address"]),
        chain=target["chain"],
        tick_range_scanned=0,
        observed_at=time.time(),
        known_lower_tick=None,
        known_upper_tick=None,
    )


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--targets", default="research/targets_wide.json")
    parser.add_argument("--db", default="data/screen.sqlite3")
    parser.add_argument("--interval", type=float, default=30.0)
    parser.add_argument("--cycles", type=int, default=None)
    parser.add_argument("--max-symbols", type=int, default=120,
                        help="cap on distinct CEX symbols subscribed")
    args = parser.parse_args()

    config = research_config("INFO")
    config.network.rpc_max_concurrency = 12
    config.network.rpc_requests_per_second_by_chain = {
        "ethereum": 6.0, "arbitrum": 8.0, "base": 3.0, "bsc": 5.0,
    }

    targets = json.loads(Path(args.targets).read_text(encoding="utf-8"))
    # One pool per (asset, chain): the deepest tier is unknown before reading, so the
    # lowest fee tier present is used as a proxy. The screen measures spot dislocation,
    # which does not depend on depth, so this only controls how many pools are read.
    best = {}
    for target in targets:
        key = (target["base"], target["quote"], target["chain"])
        if key not in best or target["fee"] < best[key]["fee"]:
            best[key] = target
    chosen = list(best.values())

    symbols = []
    seen = set()
    for target in chosen:
        if target["cex_symbol"] not in seen:
            seen.add(target["cex_symbol"])
            symbols.append(target["cex_symbol"])
    symbols = symbols[:args.max_symbols]
    chosen = [t for t in chosen if t["cex_symbol"] in set(symbols)]
    print(f"{len(chosen)} pools across {len(symbols)} CEX symbols")

    dex = UniV3DexClient(config.dex, config.network, config.secrets, config.tokens)
    multicall = Multicall(dex)

    pairs = {}
    for target in chosen:
        if target["cex_symbol"] in pairs:
            continue
        pairs[target["cex_symbol"]] = MarketPair(
            base=target["base"], quote_cex=target["quote"],
            quote_dex=target["quote"], cex_symbol=target["cex_symbol"],
            dex_chain=target["chain"], dex_pool_fee=target["fee"],
            base_address=target["base_address"],
            quote_address=target["quote_address"],
            base_decimals=target["base_decimals"],
            quote_decimals=target["quote_decimals"],
        )
    cex = BinanceCexClient(config.cex, config.secrets, list(pairs.values()))
    await cex.connect()
    print(f"waiting for {len(pairs)} books to sync...")
    await asyncio.sleep(20)

    by_chain = defaultdict(list)
    for target in chosen:
        by_chain[target["chain"]].append(target)

    # token0/token1 resolved once from the addresses, so the screen does not spend a
    # call per pool per cycle asking the chain which way round it is.
    orientation = {}
    for target in chosen:
        base = target["base_address"].lower()
        quote = target["quote_address"].lower()
        base_is_token0 = base < quote  # v3 orders tokens by address
        orientation[target["pool_address"].lower()] = base_is_token0

    run_id = f"screen-{int(time.time())}"
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

    cycles = 0
    written = 0
    started_run = time.time()
    print(f"run {run_id}: screening every {args.interval}s into {args.db}")
    try:
        while not stopping:
            cycle_start = time.monotonic()
            per_chain = {}
            for chain, chain_targets in by_chain.items():
                w3 = dex._get_w3(chain)
                try:
                    per_chain[chain] = await screen_chain(
                        dex, multicall, chain, chain_targets, w3
                    )
                except Exception as exc:
                    print(f"  {chain}: screen failed ({type(exc).__name__}: {exc})")
                    per_chain[chain] = {}

            now = time.time()
            for target in chosen:
                found = per_chain.get(target["chain"], {}).get(
                    target["pool_address"].lower()
                )
                if found is None:
                    continue
                sqrt_price_x96, tick, block = found
                pair = pairs.get(target["cex_symbol"])
                book = await cex.get_book(pair) if pair else None
                if book is None or not book.bids or not book.asks:
                    continue
                base_is_token0 = orientation[target["pool_address"].lower()]
                snapshot = screen_snapshot(
                    target, sqrt_price_x96, tick, block, base_is_token0
                )
                store.record(Observation(
                    ts=now, cex_symbol=target["cex_symbol"], base=target["base"],
                    quote=target["quote"], chain=target["chain"],
                    pool_fee=int(target["fee"]),
                    pool_address=target["pool_address"],
                    cex_bids=list(book.bids), cex_asks=list(book.asks),
                    cex_feed_ts=getattr(book, "feed_timestamp", None),
                    pool=snapshot,
                    gas_price_wei=None, native_price_quote=None,
                    rpc_endpoint=config.network.rpc_urls.get(target["chain"]),
                    run_id=run_id,
                ))
                written += 1

            cycles += 1
            if args.cycles is not None and cycles >= args.cycles:
                break
            elapsed = time.monotonic() - cycle_start
            print(f"cycle {cycles}: {written:,} rows total, {elapsed:.1f}s")
            if args.interval - elapsed > 0:
                await asyncio.sleep(args.interval - elapsed)
    finally:
        print(f"\n--- screen stopped after {(time.time() - started_run) / 60:.1f} min ---")
        print(f"  cycles {cycles}, rows {written:,}, in store {store.count():,}")
        print(f"  {dex._rpc_limiter.describe()}")
        store.close()
        await cex.close()


asyncio.run(main())
