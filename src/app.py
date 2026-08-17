import asyncio
import json
import uuid
from typing import List, Optional
from pathlib import Path

try:
    import uvloop
except ImportError:  # uvloop does not support Windows
    uvloop = None

from .core import clock
from .core.config import load_config, AppConfig
from .core.types import MarketPair
from .infra.logging import setup_logging
from .infra.metrics import setup_metrics
from .infra.dashboard import DashboardPublisher
from .infra.evaluation_store import EvaluationStore
from .infra.lifecycle import ShutdownSignal, drain, install_signal_handlers
from .infra import metrics
from .exchange.binance import BinanceCexClient
from .exchange.univ3 import UniV3DexClient
from .strategy.detector import OpportunityDetector
from .strategy.router import SimpleRouter
from .strategy.executor import TransactionExecutor, PaperExecutor
from .risk.limits import RiskManager

# Install uvloop for improved async performance where it is available
if uvloop is not None:
    uvloop.install()

class ArbiBotApp:
    def __init__(self, config: AppConfig, mode: str = 'live'):
        self.config = config
        self.logger = setup_logging(config.observability)
        self.metrics_server = setup_metrics(config.observability)
        self.running = False
        self.mode = mode
        # Latched by SIGTERM/SIGINT so the loop exits between iterations
        # rather than the interpreter dying mid-await.
        self.shutdown_signal = ShutdownSignal()
        self._cycle_task: Optional[asyncio.Task] = None
        self.dashboard_publisher = None

        # 1. Load static pairs from config
        static_pairs: List[MarketPair] = []
        for p in config.pairs:
            # Authoritative lookup for decimals for static pairs
            base_token_details = config.tokens.get(p.base, {}).get(p.dex_chain)
            quote_token_details = config.tokens.get(p.quote, {}).get(p.dex_chain)

            pair = MarketPair(
                base=p.base,
                quote_cex=p.quote,
                quote_dex=p.quote, # For direct pairs, CEX and DEX quotes are the same
                cex_symbol=p.cex_symbol,
                dex_chain=p.dex_chain,
                dex_pool_fee=p.dex_pool_fee,
                base_address=base_token_details.address if base_token_details else None,
                quote_address=quote_token_details.address if quote_token_details else None,
                base_decimals=base_token_details.decimals if base_token_details else None,
                quote_decimals=quote_token_details.decimals if quote_token_details else None,
                base_precision=p.base_precision if p.base_precision is not None else 8,
                quote_precision=p.quote_precision if p.quote_precision is not None else 8,
                # Copy strategy params from static config
                min_net_bps=p.min_net_bps,
                max_slippage_bps=p.max_slippage_bps,
                max_size_quote=p.max_size_quote,
                price_floor_quote=p.price_floor_quote,
                price_ceiling_quote=p.price_ceiling_quote,
                max_edge_bps=p.max_edge_bps,
            )
            static_pairs.append(pair)

        # 2. Load dynamic pairs from auto-discovery JSON
        dynamic_pairs = self._load_discovered_pairs(config)

        # 3. Combine and de-duplicate pairs
        final_pairs = {p.cex_symbol: p for p in static_pairs}
        for p in dynamic_pairs:
            if p.cex_symbol not in final_pairs:
                final_pairs[p.cex_symbol] = p

        pairs = list(final_pairs.values())
        self.logger.info(f"Total monitored pairs: {len(pairs)}. (Static: {len(static_pairs)}, Dynamic: {len(dynamic_pairs)})")
        if not pairs:
            self.logger.warning("No trading pairs configured; the bot will not trade.")

        if config.dashboard.enabled:
            self.dashboard_publisher = DashboardPublisher(
                config.dashboard.redis_url,
                config.dashboard.channel,
            )
        self.cex_client = BinanceCexClient(config.cex, config.secrets, pairs, dashboard_publisher=self.dashboard_publisher)
        self.dex_client = UniV3DexClient(config.dex, config.network, config.secrets, config.tokens)
        self.risk_manager = RiskManager(config.risk)

        # Durable audit trail. `run_id` ties every row to this process, so runs
        # can be separated after the fact and a replay cannot be confused with
        # live data.
        self.run_id = f"{int(clock.now())}-{uuid.uuid4().hex[:8]}"
        self.evaluation_store: Optional[EvaluationStore] = None
        if config.observability.evaluation_store_enabled:
            self.evaluation_store = EvaluationStore(
                config.observability.evaluation_store_path, run_id=self.run_id
            )
            self.logger.info(f"Recording evaluations under run_id={self.run_id}.")
        else:
            self.logger.warning(
                "Evaluation store is DISABLED. This run will produce no audit "
                "trail and no measurable dataset."
            )

        self.detector = OpportunityDetector(
            config.strategy, self.cex_client, self.dex_client, pairs,
            store=self.evaluation_store,
        )
        self.router = SimpleRouter()

        # Select the executor based on the run mode
        if self.mode == 'paper':
            self.executor = PaperExecutor(pairs)
        else:
            self.executor = TransactionExecutor(self.cex_client, self.dex_client, self.risk_manager, pairs)
        self.logger.info(f"Application initialised in {self.mode.upper()} mode.")

    def _load_discovered_pairs(self, config: AppConfig) -> List[MarketPair]:
        """Load dynamically discovered pairs from auto_discovery.json."""
        if not config.scanner or not config.scanner.auto_discovery:
            return []

        discovery_path = Path(config.scanner.auto_discovery.persist_path)
        if not discovery_path.exists():
            self.logger.info("auto_discovery.json not found; skipping dynamic pair loading.")
            return []

        try:
            with discovery_path.open("r", encoding="utf-8") as f:
                data = json.load(f)

            discovered_pairs = []
            opportunities = data.get("opportunities", [])
            self.logger.info(f"Loaded {len(opportunities)} dynamic opportunities from {discovery_path}.")

            for opp in opportunities:
                if not opp.get("dex_candidates"):
                    continue

                candidate = opp["dex_candidates"][0]
                raw_pool = candidate.get("raw_pool_data")

                if not raw_pool:
                    self.logger.warning(f"Skipping opportunity {opp['symbol']} because raw_pool_data is missing.")
                    continue

                # --- Correctly identify token details from raw_pool_data ---
                token0 = raw_pool['token0']
                token1 = raw_pool['token1']

                # Normalize symbols to correctly identify base/quote
                token0_norm_sym = token0['symbol'].upper().replace('WETH', 'ETH').split('.')[0]
                token1_norm_sym = token1['symbol'].upper().replace('WETH', 'ETH').split('.')[0]

                base_details_raw, quote_details_raw = (token0, token1) if token0_norm_sym == opp["base"] else (token1, token0)

                # --- Authoritative Decimal Lookup ---
                is_synthetic = raw_pool.get('is_synthetic', False)
                intermediate_symbol = raw_pool.get('intermediate_symbol') if is_synthetic else None

                # For synthetic pairs, decimals MUST come from the raw pool data to match the on-chain addresses.
                if is_synthetic:
                    base_decimals = base_details_raw.get('decimals', 18)
                    quote_decimals = quote_details_raw.get('decimals', 18) # This is the intermediate token's decimals
                    quote_dex = intermediate_symbol
                else:
                    # For direct pairs, we can use the authoritative lookup.
                    base_token_config = config.tokens.get(opp["base"], {}).get(candidate["chain"])
                    quote_token_config = config.tokens.get(opp["quote"], {}).get(candidate["chain"])
                    base_decimals = base_token_config.decimals if base_token_config else base_details_raw.get('decimals', 18)
                    quote_decimals = quote_token_config.decimals if quote_token_config else quote_details_raw.get('decimals', 18)
                    quote_dex = opp["quote"]

                # Load default strategy params from config for dynamic pairs
                default_params = config.scanner.volume.default_pair_params
                pair = MarketPair(
                    base=opp["base"],
                    quote_cex=opp["quote"],
                    quote_dex=quote_dex,
                    cex_symbol=opp["symbol"],
                    dex_chain=candidate["chain"],
                    dex_pool_fee=candidate["fee"],
                    base_address=base_details_raw['address'],
                    quote_address=quote_details_raw['address'],
                    base_decimals=base_decimals,
                    quote_decimals=quote_decimals,
                    is_synthetic=is_synthetic,
                    intermediate_symbol=intermediate_symbol,
                    base_precision=8,
                    quote_precision=8,
                    min_net_bps=default_params.get("min_net_bps"),
                    max_slippage_bps=default_params.get("max_slippage_bps"),
                    max_size_quote=default_params.get("max_size_quote"),
                    price_floor_quote=default_params.get("price_floor_quote"),
                    price_ceiling_quote=default_params.get("price_ceiling_quote"),
                    max_edge_bps=default_params.get("max_edge_bps"),
                )
                discovered_pairs.append(pair)
            return discovered_pairs
        except Exception as e:
            self.logger.error(f"Error reading or parsing {discovery_path}: {e}", exc_info=True)
            return []

    async def start(self):
        self.logger.info("Bot starting...")
        self.running = True

        # SIGTERM is what systemd and docker send. Without a handler the
        # interpreter dies without unwinding, so shutdown() never runs.
        install_signal_handlers(asyncio.get_running_loop(), self.shutdown_signal)

        await self.cex_client.connect()

        if self.config.risk.cancel_all_on_start:
            self.logger.info("Cancelling all existing CEX orders (not yet implemented)...")
            # await self.cex_client.cancel_all_orders()

        self.logger.info("Bot started; waiting for the CEX order book to synchronise...")
        # Give the WebSocket order book a few seconds to perform its initial sync
        await asyncio.sleep(5)
        self.logger.info("Beginning market monitoring...")

        try:
            await self.main_loop()
        except asyncio.CancelledError:
            self.logger.info("Main loop cancelled.")
        finally:
            await self.shutdown()

    async def main_loop(self):
        while self.running and not self.shutdown_signal.requested:
            cycle_started = clock.monotonic()
            try:
                opportunities = await self.detector.detect()
                try:
                    metrics.cycle_duration_seconds.observe(
                        clock.monotonic() - cycle_started
                    )
                except Exception:  # pragma: no cover
                    pass
                if not opportunities:
                    await asyncio.sleep(self.config.strategy.loop_interval_seconds)
                    continue

                for opp in opportunities:
                    if self.shutdown_signal.requested:
                        self.logger.info(
                            "Shutdown requested mid-cycle; not starting further trades."
                        )
                        break
                    self.logger.info(f"Opportunity found: {opp.direction} | edge: {opp.edge_bps:.2f} bps | expected PnL: {opp.expected_pnl_quote:.4f}")

                    plan = self.router.plan(opp)
                    self.logger.debug(f"Execution plan generated: {plan}")

                    if not self.risk_manager.is_trade_allowed(plan):
                        self.logger.warning(f"Trade blocked by the risk manager: {plan.pair.cex_symbol}")
                        continue

                    exec_summary = await self.executor.run(plan)
                    self.logger.info(f"Execution complete: {exec_summary}")

                    self.risk_manager.update_state(exec_summary)

            except asyncio.CancelledError:
                raise
            except Exception as e:
                # A permanent fault would otherwise loop here forever without
                # exiting and without alerting. Count consecutive failures and
                # give up, so the supervisor can escalate.
                self._consecutive_errors = getattr(self, "_consecutive_errors", 0) + 1
                self.logger.error(
                    f"Unexpected error in the main loop "
                    f"(consecutive: {self._consecutive_errors}): {e}",
                    exc_info=True,
                )
                if self._consecutive_errors >= self.config.strategy.max_consecutive_errors:
                    self.logger.error(
                        f"Aborting: {self._consecutive_errors} consecutive loop "
                        f"failures reached the limit of "
                        f"{self.config.strategy.max_consecutive_errors}. Exiting so "
                        f"the supervisor can escalate rather than looping silently."
                    )
                    self.shutdown_signal.request("consecutive loop failures")
                    raise
                await asyncio.sleep(self.config.strategy.error_backoff_seconds)
            else:
                self._consecutive_errors = 0

    async def shutdown(self):
        reason = self.shutdown_signal.reason or "requested"
        self.logger.info(f"Bot shutting down ({reason})...")
        self.running = False

        # Give in-flight work a bounded chance to finish before the clients are
        # torn out from under it. An unbounded wait is a hang; no wait at all
        # abandons anything mid-flight.
        if self._cycle_task is not None:
            await drain([self._cycle_task],
                        timeout=self.config.strategy.shutdown_drain_seconds)
        await self.cex_client.close()
        # The Prometheus server runs as a daemon thread; no explicit stop is required.
        if self.dashboard_publisher:
            await self.dashboard_publisher.close()
        if self.evaluation_store:
            recorded = self.evaluation_store.count()
            self.evaluation_store.close()
            self.logger.info(
                f"Evaluation store closed: {recorded} rows recorded this run."
            )
        self.logger.info("Bot shut down cleanly.")
