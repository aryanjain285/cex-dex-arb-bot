import os
import yaml
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Set

from pydantic import BaseModel, Field, SecretStr, ValidationError, model_validator
from dotenv import load_dotenv
from loguru import logger

# --- Helper function to load YAML with environment variable substitution ---
def load_yaml_with_env(path: str) -> dict:
    with open(path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Substitute ${VAR_NAME} with environment variables
    expanded_content = os.path.expandvars(content)
    return yaml.safe_load(expanded_content)

# --- Pydantic Models for Configuration ---
# All component models must be defined before the main AppConfig model

class TokenDetails(BaseModel):
    address: str
    decimals: int

class NetworkConfig(BaseModel):
    default_chain: str
    rpc_urls: Dict[str, str] = {}
    native_token: Dict[str, str] = {}
    max_pending_seconds: int
    gas_estimation_chain: str
    priority_fee_gwei: float
    max_fee_gwei: float

    @model_validator(mode='after')
    def load_rpc_urls_from_env(self) -> 'NetworkConfig':
        raw_urls = {
            "ethereum": os.getenv("ETH_RPC_URL", ""),
            "arbitrum": os.getenv("ARBITRUM_RPC_URL", ""),
            "bsc": os.getenv("BSC_RPC_URL", ""),
            "base": os.getenv("BASE_RPC_URL", ""),
        }
        # drop entries with empty URLs
        self.rpc_urls = {chain: url for chain, url in raw_urls.items() if url}
        return self

class DexContracts(BaseModel):
    router: str
    quoter_v2: str
    weth: str
    factory: Optional[str] = None

class DexConfig(BaseModel):
    uniswap_v3: Dict[str, DexContracts]

    # Gas units assumed for a single-hop V3 swap when pricing a quote. The
    # real figure is only knowable via estimate_gas after an approval exists,
    # so this is an explicit assumption rather than a hidden literal. Verify
    # it against actual receipts before trading size.
    swap_gas_estimate_units: int = 200_000

    # Deadline placed on a swap transaction, in seconds. Must be short: an
    # arbitrage swap that lands minutes late is a guaranteed loss, not a
    # late win. The original value was 600s.
    swap_deadline_seconds: int = 60

    # How long a fetched native-token USD price stays fresh, in seconds.
    native_price_ttl_seconds: float = 60.0

    # How long a stale native price may still be served after a failed
    # refresh, in seconds. Beyond this, gas cannot be priced and quotes are
    # declined rather than guessed.
    native_price_stale_grace_seconds: float = 120.0

    @model_validator(mode='after')
    def validate_dex(self) -> 'DexConfig':
        if self.swap_gas_estimate_units <= 0:
            raise ValueError("swap_gas_estimate_units must be positive")
        if not 0 < self.swap_deadline_seconds <= 120:
            raise ValueError(
                "swap_deadline_seconds must be in (0, 120]: a long deadline lets "
                "a stale arbitrage transaction land at a loss"
            )
        if self.native_price_ttl_seconds <= 0:
            raise ValueError("native_price_ttl_seconds must be positive")
        if self.native_price_stale_grace_seconds < 0:
            raise ValueError("native_price_stale_grace_seconds must be non-negative")
        return self

class CexConfig(BaseModel):
    name: str
    base_url: str
    ws_url: str
    api_key_env: str
    api_secret_env: str
    recv_window_ms: int

    # Order book source: a *partial book depth* stream, which pushes a
    # complete top-N snapshot on every frame. Measured live, the alternative
    # diff stream (@depth) delivers only one update per SECOND, so a detector
    # polling at 200ms evaluated a book averaging 500ms old -- the same
    # magnitude as the entire edge threshold, and biased in the direction that
    # manufactures false signals.
    #
    # Binance accepts only these values; anything else silently never
    # delivers, so they are validated rather than trusted.
    book_depth_levels: int = 20
    book_update_ms: int = 100

    # REST request-weight budget per minute. Binance's documented spot limit is
    # 6000 weight per minute per IP; exceeding it returns 429, and continuing
    # returns 418, which is an IP ban of two minutes to three days. A ban also
    # blocks the market-data WebSocket, so it is an outage rather than a delay.
    max_request_weight_per_minute: int = 6000

    # Fraction of that budget this process will use. Half by default, for two
    # reasons: the local weight table is an estimate that the exchange can
    # disagree with, and another process on the same IP (a manual query, a
    # second bot, a scanner run) shares the same limit. Riding the documented
    # ceiling means the first surprise is a ban.
    request_weight_safety_fraction: float = 0.5

    @model_validator(mode='after')
    def validate_cex(self) -> 'CexConfig':
        if self.max_request_weight_per_minute <= 0:
            raise ValueError("max_request_weight_per_minute must be positive")
        if not 0 < self.request_weight_safety_fraction <= 1:
            raise ValueError(
                f"request_weight_safety_fraction must be in (0, 1], got "
                f"{self.request_weight_safety_fraction}"
            )
        # The most expensive single call in use is /api/v3/depth at weight 50.
        # A ceiling below that would make the request impossible to issue at
        # all, and the governor would raise rather than hang -- better caught
        # here, at startup.
        ceiling = self.max_request_weight_per_minute * self.request_weight_safety_fraction
        if ceiling < 50:
            raise ValueError(
                f"the effective request-weight ceiling is {ceiling:.0f}, which is "
                f"below the weight of a single depth call (50). Raise "
                f"max_request_weight_per_minute or request_weight_safety_fraction."
            )
        if self.book_depth_levels not in (5, 10, 20):
            raise ValueError(
                f"book_depth_levels must be 5, 10 or 20 (Binance partial book "
                f"depth streams), got {self.book_depth_levels}"
            )
        if self.book_update_ms not in (100, 1000):
            raise ValueError(
                f"book_update_ms must be 100 or 1000, got {self.book_update_ms}"
            )
        if self.recv_window_ms <= 0 or self.recv_window_ms > 60000:
            raise ValueError("recv_window_ms must be in (0, 60000]")
        return self

class RiskConfig(BaseModel):
    max_notional_per_leg_quote: float
    max_position_per_asset: float
    circuit_breaker_bps: int
    cancel_all_on_start: bool
    cancel_all_on_shutdown: bool

    # Maximum cumulative realised loss for a UTC day, in the quote currency.
    # Breaching it halts trading and persists the halt, so a restart cannot
    # clear it. None disables the limit -- only appropriate for paper mode.
    max_daily_loss_quote: Optional[float] = None

    @model_validator(mode='after')
    def validate_risk(self) -> 'RiskConfig':
        if self.max_notional_per_leg_quote <= 0:
            raise ValueError("max_notional_per_leg_quote must be positive")
        if self.max_position_per_asset <= 0:
            raise ValueError("max_position_per_asset must be positive")
        if self.circuit_breaker_bps <= 0:
            raise ValueError("circuit_breaker_bps must be positive")
        if self.max_daily_loss_quote is not None and self.max_daily_loss_quote <= 0:
            raise ValueError(
                "max_daily_loss_quote must be a positive magnitude "
                "(it is applied as a negative bound)"
            )
        return self

class RebalanceConfig(BaseModel):
    enable: bool
    target_ratio: float
    trigger_bps: int
    method: Literal["on_cex", "on_dex", "cross"]

class InventoryConfig(BaseModel):
    rebalance: RebalanceConfig

class ObservabilityConfig(BaseModel):
    metrics_port: int
    log_level: str
    redact_keys: List[str]

    # Durable audit trail of EVERY evaluation, including rejections and their
    # reasons. Without it a run produces log scrollback rather than a dataset,
    # and none of the questions a paper run exists to answer can be answered.
    evaluation_store_enabled: bool = True
    evaluation_store_path: str = "data/evaluations.sqlite3"

class DashboardConfig(BaseModel):
    enabled: bool = False
    redis_url: str = "redis://localhost:6379/0"
    channel: str = "bot_dashboard_channel"


class VolumeScannerConfig(BaseModel):
    enabled: bool = True
    quote_assets: List[str] = Field(default_factory=lambda: ["USDT", "USDC"])
    lookback_hours: int = 24
    anomaly_multiplier: float = 3.0
    min_hourly_quote_volume: float = 500000.0
    max_candidates: int = 20
    preferred_chains: List[str] = Field(default_factory=lambda: ["ethereum", "arbitrum", "base"])
    preferred_fee_tiers: List[int] = Field(default_factory=lambda: [500, 3000, 10000])
    persist_path: str = "data/discovered_pairs.yaml"
    cooldown_hours: int = 6
    default_pair_params: Dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode='after')
    def validate_parameters(self) -> 'VolumeScannerConfig':
        if self.lookback_hours < 1:
            raise ValueError("lookback_hours must be >= 1")
        if self.anomaly_multiplier < 1:
            raise ValueError("anomaly_multiplier must be >= 1")
        if self.max_candidates < 1:
            raise ValueError("max_candidates must be >= 1")
        if self.min_hourly_quote_volume < 0:
            raise ValueError("min_hourly_quote_volume must not be negative")
        return self


class SpikeArbitrageConfig(BaseModel):
    # Minimum NET basis points -- after the taker fee and gas -- for a candidate
    # to be reported. None means "use strategy.min_net_bps", which is the right
    # default: a screen floor above the trading floor hides candidates the
    # strategy would actually take, and the previous hardcoded 30.0 was 6x the
    # 5 bps the strategy trades at. Set it explicitly only to screen deliberately
    # more strictly than the strategy trades.
    edge_threshold_bps: Optional[float] = None

    # `cost_buffer_bps` is gone. It was a flat fudge standing in for the taker
    # fee and gas, both of which are known exactly, and it meant the screen
    # carried a second cost model that disagreed with the detector's. The screen
    # now prices through costs.evaluate_trade, the single place costs are summed.
    probe_size_base: float = 1.0
    dex_chains: List[str] = Field(default_factory=lambda: ["ethereum", "arbitrum", "base"])
    dex_fee_tiers: List[int] = Field(default_factory=lambda: [500, 3000, 10000])

    @model_validator(mode='after')
    def validate_params(self) -> 'SpikeArbitrageConfig':
        if self.edge_threshold_bps is not None and self.edge_threshold_bps < 0:
            raise ValueError("edge_threshold_bps must not be negative")
        if self.probe_size_base <= 0:
            raise ValueError("probe_size_base must be > 0")
        if not self.dex_chains:
            raise ValueError("dex_chains requires at least one chain")
        if not self.dex_fee_tiers:
            raise ValueError("dex_fee_tiers requires at least one fee tier")
        return self


class AutoDiscoveryConfig(BaseModel):
    enabled: bool = True
    quote_asset: str = "USDT"
    lookback_hours: int = 2
    ratio_threshold: float = 1.5
    scan_interval_minutes: int = 30
    max_candidates: int = 10
    concurrency: int = 20
    persist_path: str = "data/auto_discovery.json"
    missing_tokens_path: str = "data/missing_tokens.json"
    dex_pool_dataset: str = "data/target_pools_Dex.json"
    target_quote_currencies: List[str] = Field(default_factory=lambda: ["USDT", "USDC"])
    arbitrage: SpikeArbitrageConfig = Field(default_factory=SpikeArbitrageConfig)


class VolumeSpikeConfig(BaseModel):
    enabled: bool = True
    quote_asset: str = "USDT"
    lookback_hours: int = 2
    ratio_threshold: float = 1.5
    scan_interval_minutes: int = 30
    persist_path: str = "data/volume_spikes.json"
    max_candidates: int = 10
    concurrency: int = 20
    arbitrage: SpikeArbitrageConfig = Field(default_factory=SpikeArbitrageConfig)

    @model_validator(mode='after')
    def validate_spike_config(self) -> 'VolumeSpikeConfig':
        if self.lookback_hours < 2:
            raise ValueError("lookback_hours must cover the current and previous hour; >= 2 recommended")
        if self.ratio_threshold <= 1:
            raise ValueError("ratio_threshold must be > 1")
        if self.scan_interval_minutes <= 0:
            raise ValueError("scan_interval_minutes must be > 0")
        if self.max_candidates < 1:
            raise ValueError("max_candidates must be > 0")
        if self.concurrency < 1:
            raise ValueError("concurrency must be > 0")
        return self


class ScannerConfig(BaseModel):
    volume: VolumeScannerConfig = Field(default_factory=VolumeScannerConfig)
    spike: VolumeSpikeConfig = Field(default_factory=VolumeSpikeConfig)
    auto_discovery: Optional[AutoDiscoveryConfig] = None


class DexRoutingConfig(BaseModel):
    """Which Uniswap v3 fee tier to quote for a pair.

    `pairs.yaml` names one `dex_pool_fee` per pair, and the detector quoted only
    that pool. Uniswap v3 lists the same asset pair at up to four fee tiers with
    independent liquidity, so a static choice is a standing bet that one tier is
    always best. Measured live at a 1000 notional on 2026-08-17:

        ETH/USDT  configured tier 500 -> 1892.49   tier 100 -> 1893.49  (5.3 bps)
        ETH/USDC  configured tier 500 -> 1891.05   tier 100 -> 1891.74  (3.7 bps)

    Against a 5 bps net floor the tier choice alone exceeds the edge being chased.

    Selection is refreshed on a TTL rather than per cycle: four tiers on both
    sides of three pairs at a 0.2s loop would be 120 RPC calls a second.
    """

    enabled: bool = True

    # The four tiers Uniswap v3 deploys by default. 100 (0.01%) exists mainly for
    # stable and correlated pairs, and is where the measured ETH improvement was.
    candidate_fee_tiers: List[int] = Field(
        default_factory=lambda: [100, 500, 3000, 10000]
    )

    # How long a selection stands. Liquidity migrates between tiers over hours,
    # not seconds, so five minutes is frequent enough to follow it and rare
    # enough that the extra RPC cost is negligible: 24 calls per interval against
    # 30 per second in the hot loop.
    refresh_seconds: float = 300.0

    @model_validator(mode='after')
    def validate_routing(self) -> 'DexRoutingConfig':
        if self.enabled and not self.candidate_fee_tiers:
            raise ValueError(
                "dex_routing.candidate_fee_tiers is empty while routing is "
                "enabled; there would be nothing to choose between"
            )
        for fee in self.candidate_fee_tiers:
            if fee <= 0:
                raise ValueError(f"fee tier {fee} must be positive")
        if len(set(self.candidate_fee_tiers)) != len(self.candidate_fee_tiers):
            raise ValueError(
                f"dex_routing.candidate_fee_tiers contains duplicates: "
                f"{self.candidate_fee_tiers}. Each duplicate costs a quote and "
                f"changes nothing."
            )
        if self.enabled and self.refresh_seconds <= 0:
            raise ValueError(
                "dex_routing.refresh_seconds must be positive; zero would "
                "re-quote every tier on every detection cycle"
            )
        return self


class PlaceboConfig(BaseModel):
    """A control arm for the edge measurement.

    Markout computed from later rows re-samples both venues, so it measures
    whether the detector would still fire rather than whether the trade was
    worth anything. If the whole apparent edge were a stale CEX book, the decay
    curve would look identical to a real, decaying arbitrage.

    The control: evaluate the same CEX book against a DEX quote from
    `delay_seconds` ago. Under the null -- the edge is a staleness artefact --
    the placebo distribution matches the live one. Divergence is the evidence
    that the edge is real.

    Costs nothing extra: the delayed quote was already fetched.

    THE DELAY MUST EXCEED A BLOCK. The first version of this counted detection
    cycles: 5 cycles at a 0.2s loop, about a second. Run live against Ethereum it
    produced 94 paired observations that were IDENTICAL in 69% of cases, median
    difference 0.00 bps -- which reads as decisive support for the null and is
    evidence of nothing. A Uniswap v3 quote changes only when a block lands, so
    inside one 12-second Ethereum block every quote is the same number and the
    control was comparing a quote to itself.

    `AppConfig.validate_coherence` therefore checks this delay against the block
    time of the slowest chain in `pairs`, because the failure mode is a control
    that silently confirms whatever it is pointed at.
    """

    enabled: bool = True

    # Seconds to delay the DEX quote by. The default spans two Ethereum blocks,
    # which is the slowest chain this system quotes; on an all-Arbitrum universe
    # a couple of seconds would do, and the validator will say so.
    delay_seconds: float = 24.0

    @model_validator(mode='after')
    def validate_placebo(self) -> 'PlaceboConfig':
        if self.enabled and self.delay_seconds <= 0:
            raise ValueError(
                "placebo.delay_seconds must be positive; a zero delay is the "
                "live arm, not a control"
            )
        return self


class RotationConfig(BaseModel):
    """Cost of moving inventory back between venues.

    A CEX<->DEX arb is an inventory rotation, not a round trip: it buys base on
    one venue and sells base on the other, so the two sides drain in opposite
    directions and must periodically be rebalanced by physically transferring
    assets. That transfer is neither free nor instant.

    Leaving this disabled asserts that inventory rotation costs nothing. That is
    defensible only in paper mode while measuring, and it must be an explicit
    choice rather than a default nobody noticed.
    """

    enabled: bool = True

    # Exchange withdrawal fee for one rotation, in the quote currency.
    # Verify the live figure: Binance ERC-20 ETH is typically ~0.0012 ETH.
    withdrawal_fee_quote: float = 4.0

    # On-chain cost of the transfer or bridge, in the quote currency.
    bridge_gas_quote: float = 1.0

    # Working capital held per venue, in the quote currency. Determines how many
    # trades one rotation funds, and therefore how far the fixed fees amortise.
    float_quote: float = 5000.0

    # Expected adverse price move on the in-transit float, in basis points.
    # Inventory is unhedged while it moves; over a 10-minute transfer of a
    # volatile alt this term dominates the fee.
    transfer_risk_bps: float = 10.0

    @model_validator(mode='after')
    def validate_rotation(self) -> 'RotationConfig':
        if self.withdrawal_fee_quote < 0:
            raise ValueError("withdrawal_fee_quote must be non-negative")
        if self.bridge_gas_quote < 0:
            raise ValueError("bridge_gas_quote must be non-negative")
        if self.transfer_risk_bps < 0:
            raise ValueError("transfer_risk_bps must be non-negative")
        if self.float_quote <= 0:
            raise ValueError("float_quote must be positive")
        return self


class TokenPolicyConfig(BaseModel):
    """Which tokens capital is permitted to touch.

    Three token properties break the strategy's arithmetic and none of them is
    visible in a price, so none can be caught by the detector:

    * fee-on-transfer -- QuoterV2 does not model a transfer tax, so the quote
      overstates what arrives. Taxes run 100-500 bps against a 5 bps target
      edge, which inverts the sign of the trade rather than trimming it.
    * rebasing -- balances move with no transfer, so a rebase and a missing
      fill look identical to reconciliation.
    * withdrawals suspended -- the token trades on the CEX but cannot leave it,
      so a completed arbitrage strands the float on one venue. That is a loss of
      principal, not a losing trade.

    Defaults to `allowlist` (default-deny) because a denylist can only hold the
    hazards someone has already found, and the expensive token is the one the
    volume scanner discovers unattended. `denylist` exists for measurement runs
    that deliberately observe the whole market; `env: prod` refuses it.
    """

    mode: str = "allowlist"

    # Tokens cleared for capital. Each has been checked for a transfer fee, for
    # rebasing, and for CEX withdrawal status.
    allowed: List[str] = Field(
        default_factory=lambda: ["WETH", "ETH", "USDT", "USDC", "ARB"]
    )

    # Known hazards, denied regardless of mode. `risks` uses the fixed
    # vocabulary in TokenRisk; `note` carries the evidence so a later reader can
    # audit the entry instead of deleting it as unexplained.
    denied: Dict[str, Dict[str, Any]] = Field(
        default_factory=lambda: {
            "LINGO": {
                "risks": ["fee_on_transfer"],
                "note": (
                    "1.25% transfer fee, invisible to QuoterV2. Sits in the "
                    "highest-volume Base pool in the scanned dataset, so it is "
                    "the single most likely token to be auto-discovered."
                ),
            },
            "UST": {
                "risks": ["withdrawal_suspended"],
                "note": "Binance withdrawals suspended; inventory cannot leave.",
            },
            "LUNA": {
                "risks": ["withdrawal_suspended"],
                "note": "Binance withdrawals suspended; inventory cannot leave.",
            },
            "LUNC": {
                "risks": ["withdrawal_suspended"],
                "note": "Post-collapse ticker; withdrawal status unreliable.",
            },
            "FEI": {
                "risks": ["withdrawal_suspended"],
                "note": "Protocol wound down; withdrawals unavailable.",
            },
            "RENZEC": {
                "risks": ["withdrawal_suspended", "transfer_restricted"],
                "note": (
                    "RenBridge shut down: the token cannot be redeemed, so any "
                    "inventory is permanently stranded."
                ),
            },
            "SOHM": {
                "risks": ["rebasing"],
                "note": (
                    "Rebases roughly every 8 hours. Balance changes with no "
                    "transfer, so reconciliation cannot distinguish a rebase "
                    "from an unfilled leg."
                ),
            },
            "AMPL": {
                "risks": ["rebasing"],
                "note": "Daily supply rebase; same accounting problem as sOHM.",
            },
            "STA": {
                "risks": ["fee_on_transfer"],
                "note": (
                    "Statera charges 1% on transfer -- the token that drained "
                    "Balancer pools in 2020 through exactly this mechanism."
                ),
            },
            "PAXG": {
                "risks": ["fee_on_transfer"],
                "note": (
                    "On-chain transfer fee set by the issuer and changeable "
                    "without notice, so it cannot be modelled from config."
                ),
            },
        }
    )

    # Operator additions, from YAML. Merged with `denied` above, which stays in
    # code so a careless config edit -- or a merge that drops a key -- cannot
    # silently re-permit a reviewed hazard. Config can add a denial; it cannot
    # remove one. Editing a Python file under time pressure is the more
    # dangerous of the two operations, so the urgent path is the YAML one.
    denied_extra: Dict[str, Dict[str, Any]] = Field(default_factory=dict)

    # Per token, the chains on which the exchange will actually settle it. A token
    # EXISTING on a chain is not the same as being able to move it there, and the
    # difference is worth real money: canonical LINK prices 30-53 bps below
    # Binance's bid on Arbitrum, in deep pools, size-independently. If Binance does
    # not credit LINK on the Arbitrum network then that discount is the price of a
    # bridge rather than an edge.
    #
    # An absent entry means "not established" and does not constrain -- silence is
    # not evidence. An explicit EMPTY list means somebody looked and found no
    # network, which does constrain.
    #
    # Populate from the signed endpoint /sapi/v1/capital/config/getall, which lists
    # withdraw networks per coin. It is not guessable from public data.
    withdraw_networks: Dict[str, List[str]] = Field(default_factory=dict)

    def build(self) -> 'TokenPolicy':
        """Construct the runtime policy, validating the lists as it goes."""
        from src.strategy.token_policy import TokenPolicy

        return TokenPolicy(
            mode=self.mode,
            allowed=self.allowed,
            denied={**self.denied, **self.denied_extra},
            withdraw_networks=self.withdraw_networks,
        )

    @model_validator(mode='after')
    def validate_token_policy(self) -> 'TokenPolicyConfig':
        # Build eagerly so a malformed list fails at config load rather than on
        # the first evaluation of the pair that happens to reference it.
        from src.strategy.token_policy import TokenPolicyError

        try:
            self.build()
        except TokenPolicyError as exc:
            raise ValueError(f"strategy.token_policy is unenforceable: {exc}") from exc
        return self


class StrategyConfig(BaseModel):
    """Global strategy parameters.

    Every value that affects a trading decision lives here rather than as a
    literal in the code. `min_net_bps` is the single knob that decides whether
    an opportunity is worth taking: it is compared against net basis points
    after the taker fee and gas have been deducted, so it means what it says.
    """

    # Target notional per trade, in the quote currency.
    target_notional_usd: int = 1000

    # CEX taker fee in basis points. Binance spot is 10.0 standard, or 7.5
    # with the BNB fee-burn discount enabled. Check yours under Wallet > Fees.
    taker_fee_bps: Decimal = Decimal("7.5")

    # Minimum net edge, in basis points, required to act. Net means after the
    # taker fee and gas. This replaces the old min_edge_bps/slippage pair,
    # which double-counted price impact already present in the DEX quote.
    min_net_bps: Decimal = Decimal("5")

    # Reject any computed net edge above this as bad data rather than acting
    # on it. Guards against unit and decimals errors reaching the executor.
    max_net_bps_sanity: Decimal = Decimal("1000")

    # How long a detected opportunity stays actionable, in seconds.
    opportunity_ttl_seconds: float = 2.0

    # Idle sleep between detection cycles when nothing was found, in seconds.
    loop_interval_seconds: float = 0.2

    # TTL for the cached intermediate-asset CEX price used by synthetic pairs.
    intermediate_price_cache_seconds: float = 2.0

    # Reject an order book older than this, in seconds. A stalled feed would
    # otherwise be indistinguishable from a quiet market.
    max_book_age_seconds: float = 0.5

    # Consecutive main-loop failures tolerated before the process exits.
    # A permanent fault previously looped forever, logging and sleeping,
    # without exiting and without alerting.
    max_consecutive_errors: int = 10

    # Sleep after a main-loop error, in seconds.
    error_backoff_seconds: float = 5.0

    # How long shutdown waits for in-flight work before cancelling it.
    shutdown_drain_seconds: float = 10.0

    # Inventory rotation cost, amortised into every trade's economics.
    rotation: RotationConfig = Field(default_factory=RotationConfig)

    # Control arm: the same CEX book against a deliberately stale DEX
    # quote, to distinguish a real edge from a latency artefact.
    placebo: PlaceboConfig = Field(default_factory=PlaceboConfig)

    # Which tokens capital may touch. Enforced at config load for the
    # configured pairs and again in the detector for anything discovered
    # at runtime.
    token_policy: TokenPolicyConfig = Field(default_factory=TokenPolicyConfig)

    # Which Uniswap v3 fee tier to quote per pair and side, measured rather
    # than assumed.
    dex_routing: DexRoutingConfig = Field(default_factory=DexRoutingConfig)

    @model_validator(mode='after')
    def validate_strategy(self) -> 'StrategyConfig':
        if self.target_notional_usd <= 0:
            raise ValueError("target_notional_usd must be positive")
        if self.taker_fee_bps < 0:
            raise ValueError("taker_fee_bps must be non-negative")
        if self.min_net_bps < 0:
            raise ValueError("min_net_bps must be non-negative")
        if self.max_net_bps_sanity <= self.min_net_bps:
            raise ValueError("max_net_bps_sanity must exceed min_net_bps")
        if self.max_consecutive_errors < 1:
            raise ValueError("max_consecutive_errors must be at least 1")
        for name in ("opportunity_ttl_seconds", "loop_interval_seconds",
                     "intermediate_price_cache_seconds", "max_book_age_seconds",
                     "error_backoff_seconds", "shutdown_drain_seconds"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        # Cross-check: a float smaller than one trade's notional cannot
        # support the strategy at all, and would silently mis-price every
        # evaluation instead of failing.
        if self.rotation.enabled and self.rotation.float_quote < self.target_notional_usd:
            raise ValueError(
                f"rotation.float_quote ({self.rotation.float_quote}) is smaller "
                f"than target_notional_usd ({self.target_notional_usd}): the "
                f"strategy cannot fund a single trade at this size"
            )
        return self

class PairConfig(BaseModel):
    base: str
    quote: str
    cex_symbol: str
    # Optional per-pair override of StrategyConfig.min_net_bps.
    min_net_bps: Optional[Decimal] = None
    # Execution slippage tolerance only -- used to derive amountOutMinimum.
    # It is not a cost and does not enter the trade economics.
    max_slippage_bps: int
    max_size_quote: int
    dex_chain: str
    dex_pool_fee: int
    price_floor_quote: Optional[Decimal] = None
    price_ceiling_quote: Optional[Decimal] = None
    max_edge_bps: Optional[int] = None
    base_precision: Optional[int] = None
    quote_precision: Optional[int] = None

class SecretsConfig(BaseModel):
    binance_api_key: SecretStr
    binance_api_secret: SecretStr
    dex_wallet_private_key: SecretStr

    @classmethod
    def load(cls):
        api_key = os.getenv("BINANCE_API_KEY", "")
        api_secret = os.getenv("BINANCE_API_SECRET", "")
        private_key = os.getenv("DEX_WALLET_PRIVATE_KEY", "")

        if not private_key:
            raise ValueError("Critical: DEX_WALLET_PRIVATE_KEY is missing or empty in your .env file.")
        if not api_key:
            logger.warning("BINANCE_API_KEY  is not set; CEX functionality will be limited.")
        if not api_secret:
            logger.warning("BINANCE_API_SECRET  is not set; CEX functionality will be limited.")

        return cls(
            binance_api_key=api_key,
            binance_api_secret=api_secret,
            dex_wallet_private_key=private_key,
        )

# Main AppConfig model
class AppConfig(BaseModel):
    env: str
    network: NetworkConfig
    dex: DexConfig
    cex: CexConfig
    risk: RiskConfig
    inventory: InventoryConfig
    observability: ObservabilityConfig
    dashboard: DashboardConfig
    strategy: StrategyConfig
    pairs: List[PairConfig]
    tokens: Dict[str, Dict[str, TokenDetails]]
    secrets: SecretsConfig = Field(default_factory=SecretsConfig.load)
    scanner: ScannerConfig = Field(default_factory=ScannerConfig)

    @model_validator(mode='after')
    def validate_coherence(self) -> 'AppConfig':
        """Reject configurations whose parts are individually valid but jointly
        incoherent.

        Every audit found at least one instance of the same pattern: a setting
        that looks like a limit, validates fine on its own, and cannot possibly
        fire given another setting. That is worse than an absent limit, because
        an operator reads it and believes they are protected.

        `env: "prod"` additionally requires the controls that only matter with
        real capital. It is the only place in this codebase where `env` changes
        behaviour -- previously it was read nowhere and `run` behaved
        identically under dev and prod.
        """
        notional = float(self.strategy.target_notional_usd)
        cap = float(self.risk.max_notional_per_leg_quote)

        # Trade size ALWAYS derives from target_notional_usd, so a cap far above
        # it is unreachable by construction. This was the live defect: a 10x gap
        # meant the only enforced risk gate could never fire.
        if cap > 5 * notional:
            raise ValueError(
                f"risk.max_notional_per_leg_quote ({cap}) is unreachable: trade "
                f"size always derives from strategy.target_notional_usd "
                f"({notional}), so a cap above ~1.2x it can never fire. Set it "
                f"near the target notional or raise the notional deliberately."
            )
        if cap < notional:
            raise ValueError(
                f"risk.max_notional_per_leg_quote ({cap}) is below "
                f"strategy.target_notional_usd ({notional}): every trade would "
                f"be rejected, and the bot would run indefinitely without "
                f"trading while appearing healthy."
            )

        loss_limit = self.risk.max_daily_loss_quote
        if loss_limit is not None and loss_limit < 0.05 * notional:
            raise ValueError(
                f"risk.max_daily_loss_quote ({loss_limit}) is small relative to "
                f"a {notional} notional: a single ordinary losing trade would "
                f"halt the system, which reads as a malfunction rather than a "
                f"risk control."
            )

        # The placebo delay must exceed the block time of the slowest chain being
        # quoted, or the control compares a DEX quote to itself: within one block
        # every quote is the same number. Measured live, a one-second delay
        # against Ethereum produced live and placebo values identical in 69% of
        # paired observations -- a control that confirms whatever it is pointed
        # at, which is worse than no control at all.
        if self.strategy.placebo.enabled and self.pairs:
            from src.strategy.placebo import min_delay_seconds_for

            floor = min_delay_seconds_for(p.dex_chain for p in self.pairs)
            if self.strategy.placebo.delay_seconds < floor:
                chains = sorted({p.dex_chain for p in self.pairs})
                raise ValueError(
                    f"strategy.placebo.delay_seconds is "
                    f"{self.strategy.placebo.delay_seconds}, below the "
                    f"{floor:.1f}s needed to span two blocks on the slowest "
                    f"configured chain (chains: {chains}). A DEX quote changes "
                    f"only when a block lands, so a shorter delay compares a "
                    f"quote to itself and the placebo silently agrees with the "
                    f"live arm."
                )

        # Every configured pair must be tradeable under the token policy. A
        # hand-written pairs.yaml entry for a fee-on-transfer token would
        # otherwise produce a bot whose own arithmetic reports a profit on every
        # trade while the transfer tax takes more than the entire edge.
        policy = self.strategy.token_policy.build()
        for pair in self.pairs:
            verdict = policy.check(pair.base, pair.quote)
            if not verdict.allowed:
                raise ValueError(
                    f"pair {pair.cex_symbol} is not tradeable: {verdict.reason}"
                )

        # Pairs must be quotable. A pair on a chain with no DEX contracts logs a
        # warning every cycle forever and never trades.
        known_chains = set(self.dex.uniswap_v3)
        for pair in self.pairs:
            if pair.dex_chain not in known_chains:
                raise ValueError(
                    f"pair {pair.cex_symbol} is configured for chain "
                    f"'{pair.dex_chain}', which has no dex.uniswap_v3 entry. "
                    f"Known chains: {sorted(known_chains)}."
                )

        if self.env == "prod":
            if self.risk.max_daily_loss_quote is None:
                raise ValueError(
                    "env: prod requires risk.max_daily_loss_quote. Running with "
                    "real capital and no daily loss limit is not a configuration "
                    "this system will accept."
                )
            if not self.strategy.rotation.enabled:
                raise ValueError(
                    "env: prod requires strategy.rotation.enabled. Disabling it "
                    "asserts that moving inventory between venues is free, which "
                    "is defensible only while measuring in paper mode."
                )
            if self.strategy.token_policy.mode != "allowlist":
                raise ValueError(
                    "env: prod requires strategy.token_policy.mode == "
                    "'allowlist'. Denylist mode permits every token nobody has "
                    "looked at yet, which is a reasonable stance for a "
                    "measurement run and an unreasonable one for capital: a "
                    "denylist can only contain hazards already discovered."
                )
            if not self.observability.evaluation_store_enabled:
                raise ValueError(
                    "env: prod requires observability.evaluation_store_enabled. "
                    "A run with no audit trail cannot be reconstructed "
                    "afterwards, so its results cannot be trusted or disputed."
                )
        return self



def load_config(
    default_path: str = "config/default.yaml",
    pairs_path: str = "config/pairs.yaml",
    tokens_path: str = "config/tokens.yaml",
    scanner_path: str = "config/scanner.yaml",
    discovered_pairs_path: str = "data/discovered_pairs.yaml"
) -> AppConfig:
    """Loads configuration from YAML files and environment variables."""
    load_dotenv()
    # Load YAML files
    default_cfg_data = load_yaml_with_env(default_path)
    pairs_cfg_data = load_yaml_with_env(pairs_path)
    tokens_cfg_data = load_yaml_with_env(tokens_path)
    scanner_cfg_data = load_yaml_with_env(scanner_path) if Path(scanner_path).exists() else {}

    # Combine data
    full_config_data = {**default_cfg_data, **pairs_cfg_data, **tokens_cfg_data}
    if scanner_cfg_data:
        # The content of scanner.yaml is expected to be nested under a 'scanner' key.
        # We merge this with any existing scanner config from the default yaml.
        default_scanner_conf = full_config_data.get('scanner', {})
        scanner_yaml_conf = scanner_cfg_data.get('scanner', {})
        default_scanner_conf.update(scanner_yaml_conf)
        full_config_data['scanner'] = default_scanner_conf

    # Parse with Pydantic
    config = AppConfig(**full_config_data)

    # Load dynamically discovered pairs, if any
    discovered_pairs_file = Path(discovered_pairs_path)
    if discovered_pairs_file.exists():
        try:
            discovered_data = load_yaml_with_env(str(discovered_pairs_file))
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning(f"Failed to load the dynamic pairs file: {exc}")
            return config

        if not discovered_data:
            return config

        raw_pairs = discovered_data.get('pairs', [])
        if not isinstance(raw_pairs, list):
            logger.warning("Dynamic pairs file has the wrong shape; expected a list.")
            return config

        # Discovered pairs are appended AFTER pydantic has finished validating
        # AppConfig, so none of the cross-checks in validate_coherence apply to
        # them -- and these are the pairs with the least human scrutiny, since
        # the volume scanner writes this file unattended and app.py feeds it
        # straight to the detector and the executor.
        #
        # The asymmetry against config/pairs.yaml is deliberate. A hand-written
        # pair is a statement of intent, so a hazard there stops the process. A
        # discovered pair is machine output, so a hazard is dropped with a
        # warning and the run continues: refusing to start would let one bad
        # discovery take the whole system offline.
        policy = config.strategy.token_policy.build()
        known_chains = set(config.dex.uniswap_v3)

        existing_symbols: Set[str] = {p.cex_symbol for p in config.pairs}
        for entry in raw_pairs:
            if not isinstance(entry, dict):
                logger.warning(f"Ignoring a malformed pair record: {entry}")
                continue
            pair_data = entry.get('config') if isinstance(entry, dict) else None
            if pair_data is None:
                pair_data = entry
            if not isinstance(pair_data, dict):
                logger.warning(f"Ignoring an unparseable pair configuration: {entry}")
                continue

            try:
                pair_cfg = PairConfig(**pair_data)
            except ValidationError as exc:
                logger.warning(f"Dynamic pair configuration failed validation: {exc}")
                continue

            if pair_cfg.cex_symbol in existing_symbols:
                continue

            verdict = policy.check(pair_cfg.base, pair_cfg.quote)
            if not verdict.allowed:
                logger.warning(
                    f"Dropping discovered pair {pair_cfg.cex_symbol}: "
                    f"{verdict.reason}"
                )
                continue

            if pair_cfg.dex_chain not in known_chains:
                logger.warning(
                    f"Dropping discovered pair {pair_cfg.cex_symbol}: chain "
                    f"'{pair_cfg.dex_chain}' has no dex.uniswap_v3 entry, so it "
                    f"can never be quoted. Known chains: {sorted(known_chains)}."
                )
                continue

            config.pairs.append(pair_cfg)
            existing_symbols.add(pair_cfg.cex_symbol)
            logger.info(
                f"Accepted discovered pair {pair_cfg.cex_symbol} on "
                f"{pair_cfg.dex_chain}."
            )

    return config

# Example usage:
if __name__ == '__main__':
    # This requires .env file to be present in the root directory
    from dotenv import load_dotenv
    load_dotenv()
    
    config = load_config()
    print("Config loaded successfully!")
    print(f"Environment: {config.env}")
    print(f"Default Chain: {config.network.default_chain}")
    print(f"CEX: {config.cex.name}")
    print(f"Monitored Pairs: {[p.cex_symbol for p in config.pairs]}")
    # Accessing secrets
    print(f"Binance Key (first 5 chars): {config.secrets.binance_api_key.get_secret_value()[:5]}...")
