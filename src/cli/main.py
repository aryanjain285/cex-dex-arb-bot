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
            typer.echo(
                f" - {data['symbol']} | {data['direction']} | edge={data['effective_edge_bps']} bps | "
                f"CEX={data['cex_price']:.6g} | DEX={data['dex_price']:.6g} on {data['dex_chain']} (fee {data['dex_fee_tier']})"
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

if __name__ == "__main__":
    app()
