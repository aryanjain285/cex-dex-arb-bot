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
- **Key metrics:**
    - `arb_opportunities_found_total` — arbitrage opportunities detected.
    - `arb_trades_executed_total` — arbitrage trades executed, labelled by status.
    - `arb_pnl_quote_total` — cumulative PnL in the quote currency.
    - `arb_failed_leg_total{venue="CEX|DEX", reason="..."}` — failed leg counts, useful for diagnosis.
    - `arb_hedged_leg_total` — successful hedges.
    - `latency_ms_bucket{stage="..."}` — latency distribution per stage (detection, execution).

### Structured logging

- **Format:** logs are intended to be emitted as JSON so aggregators such as ELK, Loki, or Datadog can parse and query them. See the note in `SECURITY.md` — the JSON sink is currently defined but inactive.
- **Location:** under `systemd`, logs go to the files named in the unit. Under Docker they go to `stdout`/`stderr` and are picked up by the logging driver.
- **Example query** in Grafana Loki or Kibana:
  `{job="arbi-bot"} | json | level="error" and pair="ETH/USDT"`

## 3. Alerting

- **Alertmanager:** use Prometheus Alertmanager to define alert rules on the metrics above.
- **Key rules:**
    - **Bot offline:** `up{job="arbi-bot"} == 0`
    - **Elevated trade failures:** `rate(arb_failed_leg_total[5m]) > 5`
    - **Circuit breaker fired:** `increase(risk_circuit_breaker_triggered_total[1m]) > 0`
    - **Sustained negative PnL:** `rate(arb_pnl_quote_total[15m]) < 0`
    - **Wallet balance low:** `asset_balance{asset="ETH"} < 0.1`

## 4. Troubleshooting

- **Clock sync.** Keep the host clock synchronised via NTP. Drift against exchange servers causes API request failures, notably Binance `recvWindow` errors.
- **Nonce conflicts.** If you send transactions manually from the same wallet, the bot's nonce can go stale. Nonce handling here is basic and reads the current transaction count per transaction, so external interference will break it. If you see `nonce too low`, check whether another process is using the wallet.
- **Rate limits.** Both the exchange and RPC providers enforce request limits. The hot loop currently issues two `eth_call`s per pair per iteration with no caching or backoff. If you see `RateLimitExceeded`:
    - Increase the polling interval in the main loop.
    - Cache the native-token price rather than fetching it per quote.
    - Upgrade your RPC plan.
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
