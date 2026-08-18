"""Run the observation recorder.

Usage:  record.py [--cycles N] [--interval SECONDS] [--db PATH]

Records raw pool state and full CEX ladders for every discovered target, so the
analysis can re-quote any size under any cost model afterwards. Read-only: no
signing key is loaded, and the process is structurally unable to place an order.

Target selection keeps the fee tiers that plausibly hold liquidity rather than all
four. The 0.01% tier is the deep one for stablecoin pairs and near-empty for
volatile ones; 1.00% is a legacy tier that mostly holds dust. Recording near-empty
pools is not free -- it spends RPC budget on refusals -- so the tiers are chosen
per pair type. Which tiers were recorded is part of what the report states, since a
"no edge anywhere" conclusion means something different if the tiers that could
have carried it were never observed.
"""
import argparse
import asyncio
import json
import signal
import sys
import time
from pathlib import Path

from research_config import research_config

from src.core.types import MarketPair
from src.exchange.binance import BinanceCexClient
from src.exchange.pool_state import ChainPoolReader
from src.exchange.pool_state_cache import PoolStateCache
from src.exchange.univ3 import UniV3DexClient
from src.research.observations import ObservationStore
from src.research.recorder import GasReader, Recorder, RecorderTarget

STABLE_PAIRS = {"USDC/USDT"}
# Fee tiers worth the RPC budget, by pair type.
TIERS_STABLE = {100, 500}
TIERS_VOLATILE = {500, 3000}


def wanted(target: dict) -> bool:
    tiers = TIERS_STABLE if target["cex_symbol"] in STABLE_PAIRS else TIERS_VOLATILE
    return target["fee"] in tiers


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cycles", type=int, default=None)
    parser.add_argument("--interval", type=float, default=10.0)
    parser.add_argument("--db", default="data/observations.sqlite3")
    parser.add_argument("--targets", default="targets.json")
    args = parser.parse_args()

    config = research_config("INFO")
    all_targets = json.loads(Path(args.targets).read_text(encoding="utf-8"))
    chosen = [t for t in all_targets if wanted(t)]
    print(f"{len(chosen)} of {len(all_targets)} discovered pools selected")

    # Per-chain rates, measured rather than guessed. Base returned 429 at 8 req/s
    # today while Arbitrum served the same load; Ethereum's publicnode endpoint is
    # slow per call rather than volume-limited. The limiter now also backs off on a
    # refusal, so these are starting points and the run reports what it settled at.
    # Concurrency, not rate, is the binding constraint. Measured: a cheap refresh
    # cycle over 22 pools took 36s while no chain was anywhere near its rate limit,
    # because each call costs 1-3s of latency on a public endpoint and only 6 could
    # be in flight per chain. Raising this trades a higher chance of a 429 for real
    # throughput, which is the trade the adaptive backoff exists to make safe -- it
    # will find the endpoint's actual limit and report where it settled.
    config.network.rpc_max_concurrency = 12
    config.network.rpc_requests_per_second_by_chain = {
        "ethereum": 6.0,
        "arbitrum": 8.0,
        "base": 3.0,
        "bsc": 5.0,
    }
    dex = UniV3DexClient(config.dex, config.network, config.secrets, config.tokens)
    # Tick data is re-read on a timer to catch mints and burns. 120s was set when a
    # full read cost ~200 calls, which made it unaffordable for more than a couple of
    # pools; batching cut that to ~9, but the wall-clock cost of one batched read is
    # still 3-50s on a public endpoint, and a re-read blocks its own pool. 600s keeps
    # the active liquidity fresh every cycle -- that comes from slot0 and dominates
    # small sizes -- while limiting how often the out-of-range tick structure, which
    # only matters for large sizes, is refetched.
    cache = PoolStateCache(ChainPoolReader(dex), full_reread_seconds=600.0)

    pairs = [
        MarketPair(
            base=t["base"], quote_cex=t["quote"], quote_dex=t["quote"],
            cex_symbol=t["cex_symbol"], dex_chain=t["chain"],
            dex_pool_fee=t["fee"],
            base_address=t["base_address"], quote_address=t["quote_address"],
            base_decimals=t["base_decimals"], quote_decimals=t["quote_decimals"],
        )
        for t in chosen
    ]
    # One CEX subscription per distinct symbol; the same book serves every chain and
    # tier for that pair, which is precisely what makes the cross-chain comparison
    # clean -- identical CEX side, different DEX side.
    distinct = {}
    for pair in pairs:
        distinct.setdefault(pair.cex_symbol, pair)
    cex = BinanceCexClient(config.cex, config.secrets, list(distinct.values()))
    await cex.connect()
    print(f"subscribed to {len(distinct)} CEX symbols; waiting for books...")
    await asyncio.sleep(10)

    targets = [
        RecorderTarget(pair=pair, pool_address=t["pool_address"])
        for pair, t in zip(pairs, chosen)
    ]

    run_id = f"rec-{int(time.time())}"
    store = ObservationStore(args.db, run_id=run_id)
    recorder = Recorder(
        store=store, cex=cex, pools=cache,
        gas=GasReader(dex, cex), targets=targets,
        interval_seconds=args.interval, run_id=run_id,
    )

    stopping = False

    def request_stop(*_):
        nonlocal stopping
        if not stopping:
            stopping = True
            print("\nstop requested; finishing the current cycle...")
            recorder.stop()

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(sig, request_stop)
        except (ValueError, OSError):
            pass

    print(f"run {run_id}: recording {len(targets)} pools every "
          f"{args.interval}s into {args.db}")
    started = time.time()
    try:
        await recorder.run(max_cycles=args.cycles)
    finally:
        stats = recorder.stats()
        elapsed = time.time() - started
        print(f"\n--- recorder stopped after {elapsed / 60:.1f} min ---")
        for key, value in stats.items():
            print(f"  {key}: {value}")
        print(f"  observations in store: {store.count()}")
        span = store.time_span()
        if span:
            print(f"  span: {span[1] - span[0]:.0f}s")
        cache_stats = cache.stats()
        print(f"  pool cache: {cache_stats}")
        print(f"  {dex._rpc_limiter.describe()}")
        for chain, info in dex._rpc_limiter.stats().items():
            print(f"    {chain}: configured {info['configured_rate']:g}/s, "
                  f"effective {info['effective_rate']:.2f}/s, "
                  f"{info['throttle_events']} throttle events")
        store.close()
        await cex.close()


asyncio.run(main())
