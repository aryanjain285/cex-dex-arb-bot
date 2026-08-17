# Operations Guide

Guidance for deploying, monitoring, and maintaining the arbitrage bot.

## 1. Deployment

### Virtual environment — recommended for development

The most direct way to run the bot.

1. Make sure the host has Python 3.11+ installed. Several pinned dependencies do not build on newer interpreters.
2. Copy the project to the host.
3. Create and activate a virtual environment.
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    ```
4. Install dependencies.
    ```bash
    pip install -r requirements.txt
    ```
5. Create a `.env` file and fill in your production values.
6. Use `systemd` or `supervisor` to manage the process so it starts on boot and restarts after a crash.

**Example systemd unit (`arbi-bot.service`):**

```ini
[Unit]
Description=DEX-CEX Arbitrage Bot
After=network.target

[Service]
User=your_user
Group=your_group
WorkingDirectory=/path/to/your/arbi-bot
ExecStart=/path/to/your/arbi-bot/.venv/bin/python -m src.cli.main run
Restart=always
RestartSec=10
StandardOutput=append:/var/log/arbi-bot/output.log
StandardError=append:/var/log/arbi-bot/error.log

[Install]
WantedBy=multi-user.target
```

### Docker — recommended for production

Docker gives you a more consistent, isolated runtime.

1. **Create a `Dockerfile`:**
    ```dockerfile
    FROM python:3.11-slim

    WORKDIR /app

    COPY requirements.txt .
    RUN pip install --no-cache-dir -r requirements.txt

    COPY . .

    # add the src directory to PYTHONPATH
    ENV PYTHONPATH="${PYTHONPATH}:/app/src"

    # inject secrets at runtime, not here
    # ENV ETH_RPC_URL=...

    CMD ["python", "-m", "src.cli.main", "run"]
    ```

2. **Build the image:**
    ```bash
    docker build -t arbi-bot:latest .
    ```

3. **Run the container.** The simplest approach passes secrets with `--env-file`:
    ```bash
    docker run -d --name arbi-bot --env-file ./.env arbi-bot:latest
    ```
    In production, prefer Docker secrets or another dedicated secret-injection mechanism.

### Platform note

`uvloop` is imported unconditionally in `src/app.py` and `src/cli/main.py`, and it does not support Windows. Deploy on Linux, or guard the import:

```python
import sys
if sys.platform != "win32":
    import uvloop
    uvloop.install()
```

## 2. Monitoring

### Prometheus metrics

The bot exposes Prometheus metrics on a `/metrics` endpoint.

- **Default port:** `9000` (configurable in `config/default.yaml`).
- **Scrape config:** add a target to your `prometheus.yml`.
  ```yaml
  scrape_configs:
    - job_name: 'arbi-bot'
      static_configs:
        - targets: ['<your-bot-ip>:9000']
  ```
- **Every series the bot emits.** This list is exhaustive on purpose: an earlier
  version of this document named five metrics that no longer exist
  (`arb_failed_leg_total`, `arb_hedged_leg_total`, `latency_ms_bucket`,
  `risk_circuit_breaker_triggered_total`, `asset_balance`), and the alert rules
  below were written against them. An alert that can never fire is worse than no
  alert, because it is mistaken for coverage.

    - `arb_evaluations_total{pair,direction,outcome,reason}` — **the most
      important series.** Every evaluation, not just the profitable ones.
      `rate(...) == 0` means the bot has stopped *deciding*, which no other
      signal reveals: a quiet market and a wedged loop look identical in
      every other metric.
    - `arb_opportunities_found_total{pair,direction}` — evaluations that cleared
      the net floor.
    - `arb_opportunities_rejected_total{pair,direction,reason}` — refusals at the
      execution gate. `reason="opportunity_expired"` is the latency signal: it
      separates edge lost to the market from edge lost to the plumbing.
    - `arb_trades_executed_total{pair,direction,status}` — executions by status.
    - `arb_pnl_quote_total{pair}` — cumulative PnL. A **Gauge**, not a Counter,
      because it must be able to fall; `Counter.inc()` raises on a negative
      argument, and that raise previously discarded losses from PnL accounting
      while keeping gains.
    - `arb_cycle_duration_seconds` — wall time for one full detection cycle.
    - `arb_feed_age_seconds` — age of the last frame from the market-data feed.
      This is the staleness signal to alert on.
    - `arb_book_age_seconds{pair}` — per-symbol book age. Informational only:
      Binance suppresses unchanged books, so this legitimately grows in a quiet
      market and is **not** an alert source.
    - `arb_risk_halted` — 1 when the risk manager has halted trading.
    - `arb_daily_pnl_quote` — today's PnL as the risk manager sees it, which is
      what the daily loss limit is checked against.

### Structured logging

- **Format:** logs are intended to be emitted as JSON so aggregators such as ELK, Loki, or Datadog can parse and query them. See the note in `SECURITY.md` — the JSON sink is currently defined but inactive.
- **Location:** under `systemd`, logs go to the files named in the unit. Under Docker they go to `stdout`/`stderr` and are picked up by the logging driver.
- **Example query** in Grafana Loki or Kibana:
  `{job="arbi-bot"} | json | level="error" and pair="ETH/USDT"`

## 3. Alerting

- **Alertmanager:** use Prometheus Alertmanager to define alert rules on the
  metrics above. Every rule below is written against a series the bot actually
  emits; check that before adding your own.
- **Key rules:**
    - **Bot offline:** `up{job="arbi-bot"} == 0`
    - **Bot stopped deciding:** `rate(arb_evaluations_total[2m]) == 0` — the one
      rule that catches a wedged loop. Every other metric goes quiet in a quiet
      market too.
    - **Feed stalled:** `arb_feed_age_seconds > 5` — feed age, not per-symbol book
      age. A quiet illiquid symbol legitimately goes seconds between frames.
    - **Trading halted:** `arb_risk_halted == 1` — this requires a human to
      re-arm, so it should page rather than merely notify.
    - **Approaching the daily loss limit:**
      `arb_daily_pnl_quote < -0.7 * <risk.max_daily_loss_quote>` — substitute
      your configured limit. Alerting only on the halt means finding out after
      trading has already stopped.
    - **Latency is eating the edge:**
      `rate(arb_opportunities_rejected_total{reason="opportunity_expired"}[10m]) /
      rate(arb_opportunities_found_total[10m]) > 0.2` — a fifth of found
      opportunities expiring before execution means the deployment is too far from
      the venues, not that the market changed.
    - **Cycle time growing:**
      `histogram_quantile(0.95, rate(arb_cycle_duration_seconds_bucket[5m])) > 1`
    - **Sustained losses:** `rate(arb_pnl_quote_total[15m]) < 0`

## 4. Troubleshooting

- **Clock sync.** Keep the host clock synchronised via NTP. Drift against exchange servers causes API request failures, notably Binance `recvWindow` errors.
- **Nonce conflicts.** If you send transactions manually from the same wallet, the bot's nonce can go stale. Nonce handling here is basic and reads the current transaction count per transaction, so external interference will break it. If you see `nonce too low`, check whether another process is using the wallet.
- **Exchange rate limits.** Every REST call goes through one process-wide weight
  governor (`src/exchange/rate_limit.py`). Binance meters by request weight per
  minute **per IP**: 429 means throttled, 418 means the IP is banned for two
  minutes to three days — and a ban blocks the market-data WebSocket too, so it
  is an outage rather than a delay.
    - Configured by `cex.max_request_weight_per_minute` (6000, the documented spot
      limit) and `cex.request_weight_safety_fraction` (0.5). The default leaves
      half the budget unused, because the local weight table is an estimate and
      anything else sharing the IP — a manual query, a second bot, a scanner run
      — draws on the same limit.
    - `IpBannedError` is deliberately fatal. Do not retry through it: retries
      extend the ban. Wait out the stated interval, or move to another IP.
    - **Verified against the live API on 2026-08-17.** The header arrives
      lowercased as `x-mbx-used-weight-1m` (the lookup is case-insensitive; a
      case-sensitive one would have matched nothing and the governor would have
      been silently inert). Observed charges: `ticker/price` 2 against a table
      value of 4, `klines` 2 against 2, `ticker/bookTicker` 2 against 4. No
      endpoint charged more than the table, which is the direction that matters.
    - If the governor is throttling more than you expect, look at
      `arb_evaluations_total` first: the detector's hot loop uses the WebSocket
      feed and spends **no** REST weight, so sustained throttling means a scanner
      is running, not that trading is too fast.
- **RPC rate limits.** Separate budget, not covered by the governor. The hot loop
  issues two `eth_call`s per pair per iteration. If you see `RateLimitExceeded`
  from the RPC provider:
    - Increase `strategy.loop_interval_seconds`.
    - Check that the native-token price is being cached
      (`dex.native_price_ttl_seconds`) rather than fetched per quote.
    - Upgrade the RPC plan, or run your own node — for this strategy the node is
      also the latency-critical path, so the two arguments point the same way.
- **Failed DEX transactions:**
    - `transaction failed` — usually slippage or insufficient gas.
    - `transaction pending for too long` — network congestion and a gas price set too low. There is no automatic fee bump; consider adding one.

## 5. Pre-Production Checklist

Verify each of these before running with real funds:

- [ ] Order book updates are actually being applied — confirm `last_update_id` advances after startup.
- [ ] `amountOutMinimum` is derived from the quote and `max_slippage_bps`, not left at `0`.
- [ ] Token approvals are bounded rather than unlimited.
- [ ] The router ABI matches the deployed router version at the configured address.
- [ ] Daily loss limit, position cap, and circuit breaker are implemented and tested.
- [ ] Token decimals come from an authoritative source for every pair being traded.
- [ ] The full test suite passes.
- [ ] `env: prod` in `config/default.yaml`. It is not cosmetic: it makes the
      config validator require a daily loss limit, inventory rotation pricing, the
      evaluation audit trail, and `token_policy.mode: allowlist`. In `dev` all
      four are optional.
- [ ] Every token in `config/pairs.yaml` is on `strategy.token_policy.allowed`,
      and each has been checked for a transfer fee, for rebasing, and for CEX
      withdrawal status. A fee-on-transfer token loses 100–500 bps per trade
      against a 5 bps target edge, and the quoter cannot see it.
- [ ] `data/target_pools_Dex.json` has been regenerated. The shipped snapshot is
      from 2025-09-23 and `load_pool_dataset` refuses it as stale.
- [ ] The paper run's evaluation database (`data/evaluations.sqlite3`) shows a
      **positive** net edge distribution on rows where `policy_verdict='allowed'`,
      and the placebo arm (`placebo_net_bps`) is materially worse than the live
      arm. If the two distributions match, the measured edge is a latency
      artefact and no amount of execution work will make it profitable.
- [ ] `arb_opportunities_rejected_total{reason="opportunity_expired"}` is a small
      fraction of opportunities found. A large fraction means the deployment is in
      the wrong place, which is cheaper to fix before capital than after.
