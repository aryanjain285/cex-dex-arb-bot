from prometheus_client import start_http_server, Counter, Gauge, Histogram
from loguru import logger

from ..core.config import ObservabilityConfig

# --- metric definitions ---

# opportunities and trades
opportunities_found = Counter(
    "arb_opportunities_found_total",
    "Total arbitrage opportunities detected",
    ["pair", "direction"]
)
trades_executed = Counter(
    "arb_trades_executed_total",
    "Total arbitrage trades executed",
    ["pair", "direction", "status"] # status: success, failed, hedged
)

# profit and loss
# A Gauge, deliberately not a Counter. Counter.inc() raises ValueError on a
# negative argument, and that raise happened inside executor.run() BEFORE it
# returned -- so a losing trade propagated an exception past the risk manager's
# state update and the loss was discarded from PnL accounting entirely, while
# gains persisted. A PnL series must be able to fall.
pnl_quote = Gauge(
    "arb_pnl_quote_total",
    "Cumulative PnL in the quote currency (can decrease)",
    ["pair"]
)

# failures and hedges
failed_leg = Counter(
    "arb_failed_leg_total",
    "Failed execution leg count",
    ["pair", "venue", "reason"] # venue: CEX/DEX, reason: timeout, reverted, etc.
)
hedged_leg = Counter(
    "arb_hedged_leg_total",
    "Successfully hedged leg count",
    ["pair", "venue"]
)

# latency
latency_ms = Histogram(
    "latency_ms",
    "Per-stage processing latency in milliseconds",
    ["stage"], # stage: detection, execution, hedge
    buckets=[10, 25, 50, 75, 100, 150, 200, 300, 500, 1000]
)

# inventory and balances
asset_balance = Gauge(
    "asset_balance",
    "Asset balance held in a wallet or on an exchange",
    ["asset", "venue"] # venue: CEX/DEX
)

# risk
circuit_breaker_triggered = Counter(
    "risk_circuit_breaker_triggered_total",
    "Total circuit breaker activations",
    ["reason"]
)


def setup_metrics(config: ObservabilityConfig):
    """
    Start the Prometheus metrics server.
    """
    try:
        port = config.metrics_port
        start_http_server(port)
        logger.info(f"Prometheus metrics server started on port {port}.")
        return True
    except Exception as e:
        logger.error(f"Failed to start the Prometheus metrics server: {e}")
        return None
