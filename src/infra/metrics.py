"""Prometheus metrics.

Every series defined here MUST be emitted somewhere in `src/`, and there is a
test that enforces it. The reason is specific: an alert rule referencing a
series that is never created returns *no data* rather than firing, so a
dashboard stays green and an operator reads silence as health. Four of the five
documented alert rules previously referenced series that no code ever emitted
-- `arb_failed_leg_total`, `arb_hedged_leg_total`, `latency_ms`,
`asset_balance`, `risk_circuit_breaker_triggered_total`. Those are removed
below rather than left as decoration; they should return when the code that
emits them does.

The additions are aimed at one operator question that could not previously be
answered: is this bot working, or is it wedged? A cycle that evaluated 100
directions and rejected all of them looked identical to a cycle that did
nothing at all.
"""
from prometheus_client import Counter, Gauge, Histogram, start_http_server
from loguru import logger

from ..core.config import ObservabilityConfig

# --- opportunities and trades ---

opportunities_found = Counter(
    "arb_opportunities_found_total",
    "Total arbitrage opportunities detected",
    ["pair", "direction"]
)
trades_executed = Counter(
    "arb_trades_executed_total",
    "Total arbitrage trades executed",
    ["pair", "direction", "status"]
)

# Refusals at the execution gate, split by cause. `trades_executed{status=
# "invalid"}` lumped every refusal together, so "the price was out of range" and
# "we were too slow" were the same number.
#
# The expiry rate is the single most operationally important number for a
# latency-sensitive strategy: it separates edge lost to the market from edge lost
# to the plumbing. Emitted by the paper executor as well, which previously
# emitted nothing at all -- so the measurement run produced no telemetry about
# what it was discarding.
opportunities_rejected_total = Counter(
    "arb_opportunities_rejected_total",
    "Opportunities refused at the execution gate, by reason",
    ["pair", "direction", "reason"]
)

# --- profit and loss ---

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

# --- decision observability ---
#
# Every evaluation is counted by outcome and reason. This is what distinguishes
# a quiet market from a wedged loop: `rate(arb_evaluations_total[5m]) == 0`
# means the bot has stopped deciding, which no other signal reveals.

evaluations_total = Counter(
    "arb_evaluations_total",
    "Every evaluation performed, by outcome and rejection reason",
    ["pair", "direction", "outcome", "reason"]
)

cycle_duration_seconds = Histogram(
    "arb_cycle_duration_seconds",
    "Wall time for one full detection cycle across all pairs",
    buckets=[0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]
)

# Per-symbol book age. On a partial-depth feed this legitimately grows in a
# quiet market, so it is informational rather than an alert source.
book_age_seconds = Gauge(
    "arb_book_age_seconds",
    "Seconds since this symbol's order book last changed",
    ["pair"]
)

# Feed age is the real staleness signal: it grows only when the connection
# itself stops delivering, which invalidates every book at once.
feed_age_seconds = Gauge(
    "arb_feed_age_seconds",
    "Seconds since the market data feed last delivered any frame"
)

# --- risk ---

risk_halted = Gauge(
    "arb_risk_halted",
    "1 when the risk manager has halted trading, 0 otherwise"
)
daily_pnl_quote = Gauge(
    "arb_daily_pnl_quote",
    "Realised PnL for the current UTC day, in the quote currency"
)


def setup_metrics(config: ObservabilityConfig):
    """Start the Prometheus exporter.

    A bind failure is returned rather than raised, but it is logged at ERROR:
    losing the metrics endpoint means losing every alert, so it must be visible
    even though it is not fatal to trading.
    """
    try:
        port = config.metrics_port
        start_http_server(port)
        logger.info(f"Prometheus metrics server started on port {port}.")
        return True
    except Exception as e:
        logger.error(
            f"Failed to start the Prometheus metrics server on port "
            f"{config.metrics_port}: {e}. The process will continue, but every "
            f"alert rule is now blind."
        )
        return None
