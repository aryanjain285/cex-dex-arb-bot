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
from .scanner.dataset import DatasetError, require_decimals
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


# Symbols a token registry may spell differently from a pool. Wrapped-native
# tokens are the only real case: a pool says WETH, the CEX says ETH, and
# tokens.yaml keys the wrapped contract. Kept explicit rather than as a string
# replace so it cannot accidentally rewrite an unrelated ticker that merely
# contains these letters.
_SYMBOL_ALIASES = {
    "ETH": ("ETH", "WETH"),
    "WETH": ("WETH", "ETH"),
    "BNB": ("BNB", "WBNB"),
    "WBNB": ("WBNB", "BNB"),
    "MATIC": ("MATIC", "WMATIC"),
    "WMATIC": ("WMATIC", "MATIC"),
    "BTC": ("BTC", "WBTC"),
    "WBTC": ("WBTC", "BTC"),
}


def _normalise_pool_symbol(symbol: str) -> str:
    """Fold a pool's token symbol to the form used for comparison.

    Pools carry vendor suffixes (`USDC.e`) and the wrapped-native spelling, so
    the raw string cannot be compared to a CEX asset name directly.
    """
    folded = str(symbol or "").strip().upper().split(".")[0]
    return "ETH" if folded == "WETH" else folded


def _assign_pool_sides(token0: dict, token1: dict, expected_base: str):
    """Return (base_token, quote_token), or (None, None) if ambiguous.

    Ambiguity is a refusal rather than a coin flip. Two ways it arises, both
    present in real data: a pool that does not actually contain the expected
    base, and a pool whose two sides carry the same symbol -- which is what a
    counterfeit token paired against the real one looks like.
    """
    base = _normalise_pool_symbol(expected_base)
    sym0 = _normalise_pool_symbol(token0.get("symbol"))
    sym1 = _normalise_pool_symbol(token1.get("symbol"))

    if sym0 == base and sym1 != base:
        return token0, token1
    if sym1 == base and sym0 != base:
        return token1, token0
    return None, None


def _registry_address(config: AppConfig, symbol: str, chain: str) -> Optional[str]:
    """The address tokens.yaml registers for this symbol on this chain, if any.

    Returns None when the symbol is unregistered, in which case the token policy
    is the gate: under default-deny an unregistered token is refused anyway, so
    the two checks meet without a gap between them.
    """
    for candidate in _SYMBOL_ALIASES.get(str(symbol).upper(), (str(symbol).upper(),)):
        details = config.tokens.get(candidate, {}).get(chain)
        if details is not None and getattr(details, "address", None):
            return details.address.lower()
    return None


def _address_mismatch(config: AppConfig, chain: str, *sides) -> Optional[str]:
    """Describe the first symbol whose pool address contradicts the registry.

    Returns None when every registered symbol matches. The registry is
    authoritative: `config/tokens.yaml` is reviewed and version-controlled,
    while pool data is whatever a subgraph returned.
    """
    for symbol, token in sides:
        expected = _registry_address(config, symbol, chain)
        if expected is None:
            continue
        actual = str(token.get("address") or "").lower()
        if actual != expected:
            return (
                f"{symbol} on {chain} is registered at {expected} but this pool "
                f"uses {actual or '(none)'}. A matching ticker with a different "
                f"address is a different token; refusing to trade it."
            )
    return None

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

            # This file is written unattended by the volume scanner and its
            # contents go straight to the detector AND the executor, so it is an
            # auto-promotion path into live trading. The token policy and the
            # known-chain check are applied here as well as in load_config,
            # because that function gates a DIFFERENT file
            # (data/discovered_pairs.yaml) and a gate on one of two paths is not
            # a gate. The detector refuses a denied pair again per evaluation;
            # filtering here additionally keeps it out of the pair count an
            # operator reads as the trading universe, and away from the executor
            # entirely.
            token_policy = config.strategy.token_policy.build()
            known_chains = set(config.dex.uniswap_v3)

            skipped = 0
            blocked = 0
            for opp in opportunities:
                if not opp.get("dex_candidates"):
                    continue

                candidate = opp["dex_candidates"][0]
                raw_pool = candidate.get("raw_pool_data")

                if not raw_pool:
                    self.logger.warning(f"Skipping opportunity {opp['symbol']} because raw_pool_data is missing.")
                    continue

                # --- Identify which side of the pool is the base ---
                token0 = raw_pool['token0']
                token1 = raw_pool['token1']

                # The previous form of this line was:
                #
                #   base, quote = (token0, token1) \
                #       if token0_norm_sym == opp["base"] else (token1, token0)
                #
                # If NEITHER token matched the expected base, the else branch
                # silently declared token1 to be the base. The resulting pair
                # carried the wrong token's address and decimals, so every quote
                # for it priced a different asset than the CEX symbol it was
                # being compared against -- with nothing logged.
                #
                # Requiring an unambiguous match also happens to be the only
                # defence against an impostor token, and the shipped dataset
                # contains one: a Base pool pairing canonical WETH
                # (0x4200..0006) against a counterfeit token that is ALSO called
                # WETH (0x71b3..5860), showing $62m of 24h volume on $270k of
                # TVL. Both sides normalise to the same symbol, so the pool is
                # ambiguous and is refused here.
                base_details_raw, quote_details_raw = _assign_pool_sides(
                    token0, token1, opp["base"]
                )
                if base_details_raw is None:
                    self.logger.warning(
                        f"Skipping {opp['symbol']}: cannot identify the base "
                        f"token unambiguously in pool "
                        f"{candidate.get('pool_address', '?')} "
                        f"(token0={token0.get('symbol')} "
                        f"{token0.get('address')}, token1="
                        f"{token1.get('symbol')} {token1.get('address')}, "
                        f"expected base {opp['base']!r}). Either the pool does "
                        f"not contain the pair, or both sides share a symbol -- "
                        f"which is what an impostor token looks like."
                    )
                    blocked += 1
                    continue

                # A symbol is not an identity. Where tokens.yaml registers an
                # address for this symbol on this chain, the pool must use that
                # exact address; anything else is a different token wearing the
                # same ticker.
                mismatch = _address_mismatch(
                    config, candidate["chain"],
                    (opp["base"], base_details_raw),
                    (opp["quote"], quote_details_raw),
                )
                if mismatch is not None:
                    self.logger.warning(
                        f"Skipping {opp['symbol']}: {mismatch}"
                    )
                    blocked += 1
                    continue

                # --- Authoritative Decimal Lookup ---
                is_synthetic = raw_pool.get('is_synthetic', False)
                intermediate_symbol = raw_pool.get('intermediate_symbol') if is_synthetic else None

                # Token decimals are resolved here, and a missing value is a
                # hard error rather than a default of 18. On a 6-decimal token
                # that default is a 10^12 pricing error, and the shipped pool
                # dataset carries no decimals field at all.
                #
                # The failure is scoped to one pair: a single unusable entry
                # must not take down the whole dynamic pair set, but it must be
                # loudly summarised rather than silently priced.
                context = f"auto_discovery {opp['symbol']}"
                base_decimals = quote_decimals = None
                quote_dex = None
                try:
                    if is_synthetic:
                        # For synthetic pairs the decimals MUST come from the
                        # pool data, to match the on-chain addresses quoted.
                        base_decimals = require_decimals(base_details_raw, context)
                        quote_decimals = require_decimals(quote_details_raw, context)
                        quote_dex = intermediate_symbol
                    else:
                        # For direct pairs tokens.yaml is authoritative, falling
                        # back to pool data -- which must still carry decimals.
                        base_token_config = config.tokens.get(opp["base"], {}).get(candidate["chain"])
                        quote_token_config = config.tokens.get(opp["quote"], {}).get(candidate["chain"])
                        base_decimals = (
                            base_token_config.decimals if base_token_config
                            else require_decimals(base_details_raw, context)
                        )
                        quote_decimals = (
                            quote_token_config.decimals if quote_token_config
                            else require_decimals(quote_details_raw, context)
                        )
                        quote_dex = opp["quote"]
                except DatasetError as exc:
                    self.logger.warning(f"Skipping {opp['symbol']}: {exc}")

                if base_decimals is None or quote_decimals is None or quote_dex is None:
                    skipped += 1
                    continue

                # `quote_dex` is the asset actually touched on-chain, which for
                # a synthetic pair is neither the base nor the CEX quote -- and
                # is therefore the one most easily overlooked.
                verdict = token_policy.check(opp["base"], opp["quote"], quote_dex)
                if not verdict.allowed:
                    self.logger.warning(
                        f"Dropping discovered pair {opp['symbol']}: {verdict.reason}"
                    )
                    blocked += 1
                    continue

                if candidate["chain"] not in known_chains:
                    self.logger.warning(
                        f"Dropping discovered pair {opp['symbol']}: chain "
                        f"'{candidate['chain']}' has no dex.uniswap_v3 entry, so "
                        f"it can never be quoted. Known chains: "
                        f"{sorted(known_chains)}."
                    )
                    blocked += 1
                    continue

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

            if skipped:
                self.logger.error(
                    f"Skipped {skipped} discovered pair(s) with unusable token "
                    f"decimals. Regenerate data/target_pools_Dex.json -- the "
                    f"shipped snapshot predates the decimals field, and "
                    f"defaulting it to 18 is a 10^12 pricing error on any "
                    f"6-decimal token."
                )
            if blocked:
                self.logger.warning(
                    f"Blocked {blocked} discovered pair(s) on policy grounds "
                    f"(token not cleared for capital, or an unquotable chain). "
                    f"This is the scanner proposing pairs no human has reviewed, "
                    f"which is expected -- review them before adding them to "
                    f"config/pairs.yaml."
                )
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
