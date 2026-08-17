# Security Guide

Security is the highest priority in a high-frequency trading system. A single oversight can lead to serious loss of funds. Follow the practices below strictly.

## 1. Key Management

### Private Keys and API Keys
- **Never hard-code secrets.** No private key, API key, or password should ever appear in source code.
- **Use environment variables.** Manage secrets through a `.env` file in local development, make sure `.env` is listed in `.gitignore`, and never commit it to version control.
- **Principle of least privilege:**
    - **CEX API key:** grant only the permissions actually required. This bot generally needs no more than **spot trading**. **Always disable withdrawal permissions.**
    - **DEX wallet:** use a fresh hot wallet created solely for this bot. Fund it with the minimum required to run the strategy and never hold significant assets in it.
- **Production:** use a stronger key-management approach in production, such as:
    - **A cloud KMS** — AWS KMS, Google Cloud KMS, or Azure Key Vault.
    - **A hardware security module (HSM).**
    - **The OS keyring** — for example via the `keyring` Python library.

## 2. Log Redaction

Logs can capture sensitive information by accident. The logging layer (`src/infra/logging.py`) is designed to mask sensitive fields.
- **Redacted keys:** any field whose name contains `key`, `secret`, `token`, or `password` is replaced with `[REDACTED]`.
- **Address masking:** wallet addresses and transaction hashes are shown with only their leading and trailing characters, so they remain identifiable during debugging without being fully exposed.
- **Review regularly:** audit log output periodically to confirm nothing sensitive is leaking.

> **Note:** the structured JSON sink that performs this redaction is currently defined but not active — `setup_logging` falls back to plain stdout output. Enable the JSON sink before relying on redaction in production.

## 3. Dependency Management

- **Pin versions.** `requirements.txt` should specify exact versions for every dependency, so an upstream release cannot silently introduce a vulnerability or breaking change.
- **Scan regularly.** Use `pip-audit` or GitHub Dependabot to detect and patch known vulnerabilities promptly.
- **Upgrade process:**
    1. A tool reports a vulnerability.
    2. Upgrade the affected package in an isolated test environment.
    3. Run the full unit and integration test suite to confirm nothing regressed.
    4. Once verified, update `requirements.txt` and deploy.

## 4. Network Security

- **RPC nodes:** make sure the RPC endpoints you use are trustworthy. A malicious node can return incorrect on-chain data or log your request activity. Prefer reputable providers such as Alchemy or Infura.
- **Firewall:** run the bot behind a firewall and expose only the ports you need (for example the Prometheus metrics port and SSH).

## 5. Risk Control and Circuit Breakers

Beyond code-level security, the strategy's own risk controls are critical.
- **Circuit breaker:** the system should halt trading automatically when it detects abnormal conditions. Triggers should include:
    - N consecutive losing trades.
    - Cumulative daily loss exceeding a threshold.
    - Slippage or spread far outside expected ranges.
    - Wallet balance below a minimum level.
- **Order cancellation:** the bot should cancel all resting exchange orders on startup and shutdown, so no unmanaged "zombie orders" are left behind.

> **Note:** of the risk controls declared in `config/default.yaml`, only `max_notional_per_leg_quote` is currently enforced. `max_position_per_asset`, `circuit_breaker_bps`, and `cancel_all_on_start` are read from config but not acted on, and there is no daily loss limit. Implement these before running with real capital.
