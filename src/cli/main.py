import typer
import asyncio
from typing import Optional

try:
    import uvloop
except ImportError:  # uvloop does not support Windows
    uvloop = None


def install_event_loop_policy() -> None:
    """Install uvloop when available; a no-op on platforms without it."""
    if uvloop is not None:
        uvloop.install()

from src.app import ArbiBotApp
from src.core.config import load_config, TokenDetails
from src.core.types import MarketPair
from backtest.datasets import load_dataset
from backtest.simulator import Simulator
from src.strategy.rebalancer import Rebalancer
from src.exchange.binance import BinanceCexClient
from src.exchange.rate_limit import get_shared_governor
from src.scanner import VolumeScannerService, VolumeSpikeScanner, SpikeArbitrageEvaluator
from src.scanner.autodiscovery import AutoDiscoveryEngine

app = typer.Typer()

@app.command()
def autodiscover():
    """
    Run the AutoDiscovery engine: scan for CEX volume anomalies and find matching DEX pools.
    """
    typer.echo("Starting the AutoDiscovery engine...")
    install_event_loop_policy()
    
    # --- Enable DEBUG Logging ---
    import sys
    from loguru import logger
    logger.remove()
    logger.add(sys.stdout, level="DEBUG")
    # --- End DEBUG Logging ---

    config = load_config()
    
    async def main():
        engine = AutoDiscoveryEngine(config)
        await engine.run()

    try:
        asyncio.run(main())
        typer.secho(f"AutoDiscovery engine finished. Results saved to {config.scanner.auto_discovery.persist_path}", fg=typer.colors.GREEN)
    except Exception as e:
        typer.secho(f"Error while running the AutoDiscovery engine: {e}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from e

@app.command()
def run():
    """
    Start the bot in live trading mode.
    """
    typer.echo("Starting live trading mode...")
    install_event_loop_policy()
    
    config = load_config()
    bot_app = ArbiBotApp(config)
    
    try:
        asyncio.run(bot_app.start())
    except KeyboardInterrupt:
        typer.echo("Stop signal received; shutting down gracefully...")
    finally:
        typer.echo("Bot stopped.")


@app.command()
def paper():
    """
    Start the bot in paper trading mode (no orders placed).
    """
    typer.echo("Starting paper trading mode...")
    install_event_loop_policy()
    
    config = load_config()
    # initialise the app in paper mode
    bot_app = ArbiBotApp(config, mode='paper')
    
    try:
        asyncio.run(bot_app.start())
    except KeyboardInterrupt:
        typer.echo("Stop signal received; shutting down gracefully...")
    finally:
        typer.echo("Bot stopped.")


@app.command()
def rebalance(paper_run: bool = typer.Option(False, "--paper-run", help="Run in paper mode without placing orders")):
    """
    Run a one-off inventory rebalance check and action.
    """
    if paper_run:
        typer.echo("Running inventory rebalance (paper run)...")
    else:
        typer.echo("Running inventory rebalance...")
    
    install_event_loop_policy()
    config = load_config()
    
    async def main():
        # build a dedicated CEX client for the rebalance task
        pairs = [MarketPair(
            base=p.base,
            quote_cex=p.quote,
            quote_dex=p.quote, # Rebalancing is CEX-only, so they are the same
            cex_symbol=p.cex_symbol,
            dex_chain=p.dex_chain,
            dex_pool_fee=p.dex_pool_fee,
            base_precision=p.base_precision if p.base_precision is not None else 8,
            quote_precision=p.quote_precision if p.quote_precision is not None else 8,
        ) for p in config.pairs]
        cex_client = BinanceCexClient(
            config.cex, config.secrets, pairs,
            governor=get_shared_governor(
                config.cex.max_request_weight_per_minute,
                config.cex.request_weight_safety_fraction,
            ),
        )
        await cex_client.connect()
        
        rebalancer = Rebalancer(config, cex_client)
        await rebalancer.run_rebalance_check(paper_run=paper_run)
        
        await cex_client.close()

    try:
        asyncio.run(main())
    except Exception as e:
        typer.echo(f"Error while rebalancing: {e}", err=True)
    finally:
        typer.echo("Rebalance task finished.")


@app.command()
def check_dex_balance():
    """
    Show the balance of every configured token in the DEX wallet.
    """
    typer.echo("Querying DEX wallet balances...")
    install_event_loop_policy()
    config = load_config()

    # Dynamically import here to avoid circular dependency if main grows
    from src.exchange.univ3 import UniV3DexClient

    async def main():
        dex_client = UniV3DexClient(config.dex, config.network, config.secrets, config.tokens)
        
        typer.echo(f"Wallet address: {dex_client.user_address}")
        typer.echo("-" * 40)

        # iterate over every configured token and chain
        for token_symbol, chains in config.tokens.items():
            for chain in chains.keys():
                typer.echo(f"Querying {token_symbol} on {chain}...")
                balance = await dex_client.get_balance(token_symbol, chain)
                if balance >= 0:
                    typer.secho(f"  -> {balance:.8f} {token_symbol}", fg=typer.colors.GREEN)
                else:
                    typer.secho(f"  -> query failed", fg=typer.colors.RED)
        
        # also check the native token
        for chain, symbol in config.network.native_token.items():
             if symbol not in config.tokens:
                typer.echo(f"Querying native token {symbol} on {chain}...")
                balance = await dex_client.get_balance(symbol, chain)
                if balance >= 0:
                    typer.secho(f"  -> {balance:.8f} {symbol}", fg=typer.colors.GREEN)
                else:
                    typer.secho(f"  -> query failed", fg=typer.colors.RED)


    try:
        asyncio.run(main())
    except Exception as e:
        typer.echo(f"Error while querying balances: {e}", err=True)
    finally:
        typer.echo("DEX balance query complete.")


@app.command(name="discover_pairs")
def discover_pairs_command():
    """Discover new monitorable pairs via 1h volume anomaly scanning."""
    typer.echo("Running the volume scanner to discover new pairs...")
    install_event_loop_policy()
    config = load_config()
    scanner = VolumeScannerService(config)

    async def main():
        return await scanner.run()

    try:
        results = asyncio.run(main())
    except Exception as exc:
        typer.secho(f"Volume scanner failed: {exc}", fg=typer.colors.RED)
        raise typer.Exit(code=1) from exc

    if not results:
        typer.echo("No new candidate pairs found.")
        return

    typer.echo("Discovered the following monitorable pairs:")
    for item in results:
        pair = item.pair_config
        typer.secho(
            f" - {pair.cex_symbol} on {pair.dex_chain} (fee {pair.dex_pool_fee})",
            fg=typer.colors.GREEN,
        )
    typer.echo(f"Detailed configuration written to {config.scanner.volume.persist_path}")


# hyphenated alias, kept for backwards compatibility
@app.command(name="discover-pairs")
def discover_pairs_alias():
    return discover_pairs_command()


@app.command()
def lookup_pool(
    base: str = typer.Argument(..., help="Base currency symbol, e.g. REZ"),
    quote: str = typer.Argument(..., help="Quote currency symbol, e.g. USDC"),
    chain: str = typer.Option("ethereum", "--chain", help="Chain the pool is on, e.g. ethereum"),
    fee: int = typer.Option(3000, "--fee", help="Pool fee tier, e.g. 500, 3000, 10000"),
    base_address: Optional[str] = typer.Option(None, "--base-address", help="Base token address, if not configured in tokens.yaml"),
    base_decimals: int = typer.Option(18, "--base-decimals", help="Base token decimals, used with a custom address"),
    quote_address: Optional[str] = typer.Option(None, "--quote-address", help="Quote token address, if not configured in tokens.yaml"),
    quote_decimals: int = typer.Option(6, "--quote-decimals", help="Quote token decimals, used with a custom address"),
):
    """Manually check whether a Uniswap V3 pool exists."""
    typer.echo(f"Looking up {base}/{quote} on {chain} fee {fee}...")
    install_event_loop_policy()

    from src.exchange.univ3 import UniV3DexClient  # avoid a circular import

    config = load_config()

    async def main():
        client = UniV3DexClient(config.dex, config.network, config.secrets, config.tokens)
        if base_address:
            client.tokens_config.setdefault(base, {})[chain] = TokenDetails(address=base_address, decimals=base_decimals)
        if quote_address:
            client.tokens_config.setdefault(quote, {})[chain] = TokenDetails(address=quote_address, decimals=quote_decimals)
        return await client.get_pool_address(base, quote, chain, fee)

    try:
        pool = asyncio.run(main())
    except Exception as exc:
        typer.secho(f"Lookup failed: {exc}", fg=typer.colors.RED)
        raise typer.Exit(code=1) from exc

    if pool:
        typer.secho(f"Pool found at: {pool}", fg=typer.colors.GREEN)
    else:
        typer.echo("No matching pool found.")


@app.command()
def spike_run(
    evaluate: bool = typer.Option(True, "--evaluate/--no-evaluate", help="Evaluate arbitrage signals immediately after scanning"),
    use_existing: bool = typer.Option(False, "--use-existing", help="Use the existing JSON only; do not rescan"),
):
    """Run the Binance volume spike scan and optionally evaluate arbitrage."""
    install_event_loop_policy()
    config = load_config()
    scanner = VolumeSpikeScanner(config)
    evaluator = SpikeArbitrageEvaluator(config)

    async def main():
        spikes = scanner.load_spikes() if use_existing else await scanner.scan()
        signals = await evaluator.evaluate(spikes) if evaluate else []
        return spikes, signals

    try:
        spikes, signals = asyncio.run(main())
    except Exception as exc:
        typer.secho(f"spike_run failed: {exc}", fg=typer.colors.RED)
        raise typer.Exit(code=1) from exc

    typer.echo(f"Volume spike candidates: {len(spikes)}")
    if evaluate and signals:
        typer.secho("Detected the following arbitrage signals:", fg=typer.colors.GREEN)
        for signal in signals:
            data = signal.as_dict()
            # net_bps, not the old effective_edge_bps: that field was the raw
            # spread minus a flat fudge, and reading a key that no longer exists
            # would have raised here on the first signal the screen ever found.
            typer.echo(
                f" - {data['symbol']} | {data['direction']} | "
                f"net={data['net_bps']} bps (gross {data['gross_bps']}) | "
                f"CEX={data['cex_price']:.6g} | DEX={data['dex_price']:.6g} on "
                f"{data['dex_chain']} (fee {data['dex_fee_tier']}) | "
                f"depth-aware={data['depth_aware']}"
            )
    elif evaluate:
        typer.echo("No arbitrage signals currently meet the threshold.")


@app.command()
def backtest(dataset_path: str = typer.Option(..., "--dataset-path", help="Path to the historical dataset (.csv)")):
    """
    Run a backtest against historical data.
    """
    typer.echo(f"Preparing to backtest using dataset: {dataset_path}")
    install_event_loop_policy()
    
    try:
        # load configuration and data
        config = load_config()
        dataset = load_dataset(dataset_path)

        # initialise and run the simulator
        simulator = Simulator(config, dataset)
        asyncio.run(simulator.run())

        # produce the report
        simulator.report()

    except Exception as e:
        typer.echo(f"Error during the backtest: {e}", err=True)


@app.command()
def survey(
    chain: str = typer.Option("ethereum", "--chain", help="Chain to survey"),
    limit: int = typer.Option(40, "--limit", help="How many tokens, most prominent first"),
    quote_asset: str = typer.Option("USDT", "--quote-asset", help="CEX quote asset"),
    pace_seconds: float = typer.Option(0.1, "--pace", help="Delay between RPC calls"),
    coingecko_cache: str = typer.Option(
        "data/coingecko_tokens.json", "--coingecko-cache",
        help="Where to keep the CoinGecko token dump",
    ),
    output: str = typer.Option("data/survey.json", "--output", help="Where to write results"),
):
    """Ask whether a tradeable CEX-DEX spread exists at all, at the configured size.

    This is the question that comes before any execution work. It needs no paid
    API key: Binance's public endpoints give the symbols and the books, CoinGecko's
    free coin list gives canonical token addresses, and the chain itself answers
    which pools exist and at what price.

    The route priced is the SYNTHETIC one -- base against WETH on chain, converted
    through the CEX's own ETH price -- because that is where altcoin liquidity
    lives on Uniswap v3, and it costs two CEX taker legs rather than one.

    The CEX side is TOP OF BOOK, so every number is optimistic. This is a screen:
    it says where to point the real, depth-aware measurement.
    """
    import json
    import time
    from decimal import Decimal
    from pathlib import Path

    import aiohttp

    from src.exchange.errors import RpcError
    from src.exchange.rate_limit import governed_request
    from src.exchange.univ3 import UniV3DexClient
    from src.scanner.survey import (
        TokenRegistry, build_candidates, evaluate_candidate, rank, summarise,
    )
    from src.strategy.costs import amortised_rotation_cost

    install_event_loop_policy()
    config = load_config()

    if chain not in config.network.rpc_urls:
        typer.secho(
            f"No RPC URL configured for {chain}. Set the matching *_RPC_URL "
            f"environment variable.", fg=typer.colors.RED,
        )
        raise typer.Exit(code=1)

    notional = Decimal(config.strategy.target_notional_usd)
    rotation = (
        amortised_rotation_cost(
            withdrawal_fee_quote=Decimal(str(config.strategy.rotation.withdrawal_fee_quote)),
            bridge_gas_quote=Decimal(str(config.strategy.rotation.bridge_gas_quote)),
            float_quote=Decimal(str(config.strategy.rotation.float_quote)),
            notional_quote=notional,
            transfer_risk_bps=Decimal(str(config.strategy.rotation.transfer_risk_bps)),
        )
        if config.strategy.rotation.enabled else Decimal(0)
    )

    async def run():
        cache = Path(coingecko_cache)
        # Cached because it is 2.9 MB and changes slowly; refetched when stale so a
        # newly listed token is not invisible forever.
        stale = (
            not cache.exists()
            or (time.time() - cache.stat().st_mtime) > 24 * 3600
        )
        governor = get_shared_governor(
            config.cex.max_request_weight_per_minute,
            config.cex.request_weight_safety_fraction,
        )
        async with aiohttp.ClientSession() as session:
            if stale:
                typer.echo("Fetching the CoinGecko token list (free endpoint)...")
                async with session.get(
                    "https://api.coingecko.com/api/v3/coins/list?include_platform=true",
                    timeout=aiohttp.ClientTimeout(total=60),
                ) as response:
                    response.raise_for_status()
                    payload = await response.text()
                cache.parent.mkdir(parents=True, exist_ok=True)
                cache.write_text(payload, encoding="utf-8")

            coins = json.loads(cache.read_text(encoding="utf-8"))
            info = await governed_request(
                session, governor, "GET",
                f"{config.cex.base_url}/api/v3/exchangeInfo", timeout=30,
            )
            books_raw = await governed_request(
                session, governor, "GET",
                f"{config.cex.base_url}/api/v3/ticker/bookTicker", timeout=30,
            )

        books = {}
        for row in books_raw:
            bid, ask = Decimal(row["bidPrice"]), Decimal(row["askPrice"])
            if bid > 0 and ask > 0:
                books[row["symbol"]] = (bid, ask)

        eth_symbol = f"ETH{quote_asset}"
        if eth_symbol not in books:
            raise RuntimeError(
                f"no {eth_symbol} book; the synthetic leg cannot be priced"
            )
        eth_bid, eth_ask = books[eth_symbol]

        registry = TokenRegistry.from_coingecko(coins, chain=chain)
        candidates = build_candidates(
            info, books, registry, chain=chain,
            quote_asset=quote_asset, limit=limit,
        )
        typer.echo(
            f"{len(registry.addresses)} unambiguous {chain} tokens "
            f"({len(registry.ambiguous)} ambiguous tickers dropped); "
            f"surveying {len(candidates)} pairs at a {notional} notional"
        )
        typer.echo(
            f"cost floor: {config.strategy.taker_fee_bps * 2} bps taker "
            f"(two legs) + {rotation / notional * 10000:.2f} bps rotation"
        )

        client = UniV3DexClient(
            config.dex, config.network, config.secrets, config.tokens
        )
        policy = config.strategy.token_policy.build()
        decimals_cache = {}

        async def decimals_of(address: str) -> Optional[int]:
            key = address.lower()
            if key in decimals_cache:
                return decimals_cache[key]
            from web3 import Web3
            try:
                w3 = client._get_w3(chain)
                contract = w3.eth.contract(
                    address=Web3.to_checksum_address(address),
                    abi=client.erc20_abi,
                )
                value = int(await asyncio.to_thread(contract.functions.decimals().call))
            except Exception:
                # Never guessed: a wrong decimals value is a 10^n price error.
                value = None
            decimals_cache[key] = value
            return value

        results = []
        for candidate in candidates:
            await asyncio.sleep(pace_seconds)
            base_decimals = await decimals_of(candidate.base_address)
            if base_decimals is None:
                continue
            resolved = candidate.__class__(
                **{**candidate.__dict__, "base_decimals": base_decimals}
            )
            result = await evaluate_candidate(
                resolved, client,
                eth_bid=eth_bid, eth_ask=eth_ask, notional=notional,
                taker_fee_bps=config.strategy.taker_fee_bps,
                rotation_quote=rotation,
                token_policy=policy,
            )
            if result is None:
                continue
            results.append(result)
            if result.net_bps is None:
                typer.echo(f"  {result.cex_symbol:<12} (no measurement: RPC failure)")
            else:
                flag = ""
                if result.net_bps > config.strategy.min_net_bps:
                    flag = ("  <== ABOVE FLOOR" if result.tradeable
                            else "  <== above floor but NOT TRADEABLE")
                typer.echo(
                    f"  {result.cex_symbol:<12} tier {result.fee:<6} "
                    f"{result.direction:<11} net {float(result.net_bps):>9.2f} "
                    f"gross {float(result.gross_bps):>9.2f}{flag}"
                )
        return rank(results)

    try:
        results = asyncio.run(run())
    except Exception as exc:
        typer.secho(f"Survey failed: {exc}", fg=typer.colors.RED)
        raise typer.Exit(code=1) from exc

    from decimal import Decimal as _D
    summary = summarise(results, floor_bps=config.strategy.min_net_bps)
    typer.echo("")
    typer.echo(f"measured:               {summary['measured']} of {summary['candidates']}")
    typer.echo(f"RPC failures:           {summary['rpc_failed']}")
    typer.echo(f"positive net edge:      {summary['positive']}")
    typer.echo(f"above the {config.strategy.min_net_bps} bps floor:   {summary['above_floor']}")
    typer.secho(
        f"TRADEABLE above floor:  {summary['tradeable_above_floor']}",
        fg=(typer.colors.GREEN if summary["tradeable_above_floor"]
            else typer.colors.YELLOW),
    )
    if summary["best_gross_bps"] is not None:
        typer.echo(f"best gross dislocation: {float(summary['best_gross_bps']):.2f} bps")

    import json as _json
    from pathlib import Path as _Path

    out = _Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(_json.dumps([
        {
            "cex_symbol": r.cex_symbol, "chain": r.chain, "fee": r.fee,
            "direction": r.direction,
            "net_bps": None if r.net_bps is None else str(r.net_bps),
            "gross_bps": None if r.gross_bps is None else str(r.gross_bps),
            "tradeable": r.tradeable, "policy_reason": r.policy_reason,
            "rpc_failed": r.rpc_failed,
        }
        for r in results
    ], indent=2), encoding="utf-8")
    typer.echo(f"\nwrote {out}")
    typer.echo(
        "Reminder: the CEX side is top of book, so these figures are optimistic. "
        "A hit here is a candidate for depth-aware measurement, not an edge."
    )


if __name__ == "__main__":
    app()
