# DexCex: High-Frequency Arbitrage Bot

This is a sophisticated high-frequency trading bot designed to execute arbitrage strategies between Centralized Exchanges (CEX), like Binance, and Decentralized Exchanges (DEX), like Uniswap v3.

The bot is architected to be modular, extensible, and performant, utilizing modern Python features and robust libraries.

## Core Features

- **CEX and DEX integration.** Binance spot via a 100 ms partial-book depth
  stream, and Uniswap v3 across Ethereum, Arbitrum and Base via QuoterV2 — whose
  price is already net of the pool fee and of the price impact for the size
  quoted.
- **Depth-aware detection.** Both directions of every pair, direct and synthetic
  (triangular), sized against the actual order-book ladder rather than the touch.
- **One cost model.** `costs.evaluate_trade` is the single place costs are summed:
  taker fee per leg, gas priced from the live gas price and native-token rate, and
  inventory rotation amortised per trade. Nothing else may compute an edge.
- **Fee-tier routing.** Uniswap v3 lists the same pair at up to four fee tiers
  with independent liquidity. The best is measured per side on a TTL rather than
  configured, which was worth a measured +6.14 bps on ETH/USDT.
- **An audit trail, not a log.** Every evaluation is persisted — not just the
  profitable ones — with its inputs, its rejection reason, a placebo comparison,
  the token-policy verdict and the fee tier actually used. `analyse` turns it back
  into answers.
- **A placebo arm.** Each evaluation is also priced against a deliberately stale
  DEX quote, so a real edge can be told apart from a latency artefact.
- **Token policy.** Default-deny allowlist plus a reviewed hazard registry
  (fee-on-transfer, rebasing, unwithdrawable), enforced at four separate entry
  points because there were four ways in.
- **Risk management.** Per-leg notional cap and a daily loss limit that persists
  across restart. Position cap and circuit breaker are declared in config and not
  yet enforced — see [Project Status](#project-status).
- **Rate limiting.** One process-wide request-weight governor for the exchange;
  418 is treated as fatal rather than retried.
- **Observability.** Prometheus metrics, and `OPERATIONS.md` alert rules written
  against series that actually exist.

---

## Project Status

Honest state of the codebase, so you know what you are building on.

### The result that matters

The engine measures correctly and the measurement says the trade is not there.
Across 676 live evaluations on three pairs, plus on-chain surveys of 87 candidates
across Ethereum, Arbitrum and Base:

| | |
|---|---|
| Structural cost per trade | **27.7 bps** — 20.0 rotation, 7.5 taker fee, 0.2 gas |
| Average gross dislocation, liquid pairs | **−1.5 to −2.0 bps** |
| Best single gross observation in 676 | **+3.28 bps** |
| Tightest gross of 60 measured survey candidates | **−3.41 bps** (WBTC on Arbitrum) |
| Positive net edges found | 3, all traps: wrong asset in custody, wrong chain for settlement, wrong token |
| Tradeable positive net edges found | **0** |

The placebo arm — the same order book priced against a deliberately 24-second-stale
DEX quote — differs from the live arm by under 1 bps at the 10th and 90th
percentiles. The gap to break-even is not being caused by latency.

Run `python -m src.cli.main analyse` after any `paper` run to reproduce every one
of those figures from the rows the run wrote itself.

**Do not deploy capital against the current configuration.** What would change the
answer is size, cost basis, or a token whose settlement path has been verified —
not more code.

### Working and verified live

| Area | Status |
|---|---|
| Uniswap v3 quoting (QuoterV2, multi-chain) | Verified against Ethereum, Arbitrum and Base pools |
| Binance order book | 100 ms partial-book depth stream (`@depth20@100ms`), zero REST weight. The previous diff stream delivered one update per second |
| Detection: depth-weighted VWAP, both directions, direct and synthetic | Working |
| Fee-tier routing | Measures all four Uniswap v3 tiers per side on a TTL and quotes the best. Worth a measured +6.14 bps on ETH/USDT against the hardcoded tier |
| Cost model | One function, `costs.evaluate_trade`, used by the detector, the spike screen and the backtest |
| Audit trail | Every evaluation persisted with its inputs, rejection reason, placebo value, policy verdict and the fee tier actually used |
| Token policy | Default-deny allowlist plus a reviewed hazard registry, enforced at four entry points |
| REST rate limiting | One process-wide weight governor; verified against the live API |
| Risk | Daily loss limit persisted across restart; per-mode state files |
| Paper trading | Working, and enforces the same deadline and sanity gates as live |
| Backtest | Rebuilt against the current interfaces; gas must be present in the data |
| Universe survey | `survey` answers "does a tradeable spread exist" without any paid API key |
| Test suite | 491 passing, 1 skipped. `pytest tests/ -q` |

### Not yet implemented

- **Live execution.** `TransactionExecutor` builds the two-leg plan, gates it and
  reports the expected outcome, but does not place orders — it never calls
  `create_order()` or `execute_swap()`. `run` and `paper` therefore behave
  identically. This is deliberate: the measurement says there is nothing to
  execute, so building the execution path first would be building for a trade that
  does not exist.
- **Unhedged-leg policy.** No handling for one leg filling and the other not.
  `HedgingError` is defined and never raised.
- **Nonce management.** `execute_swap` reads the transaction count per
  transaction, so two concurrent sends would collide.
- **WETH wrap/unwrap.** Most pools quote WETH while Binance withdraws native ETH.
- **Reconciliation.** `DexTxReceipt` returns placeholder fill values rather than
  parsing the swap event from the receipt logs.

### Known issues to address before live trading

1. **`dex.swap_deadline_seconds` cannot be enforced on the swap path.** Now
   verified rather than suspected. `ABI/router.json` is the SwapRouter02 ABI
   (`exactInputSingle`, selector `0x04e45aaf`, seven fields, no `deadline`), and
   the configured Ethereum, Arbitrum and Base routers all dispatch that selector —
   so the ABI and the chain agree. What disagreed was the caller, which built an
   eight-key struct including a `deadline` the struct does not have; that is fixed,
   and a test compares the built keys against the ABI.

   The consequence remains: SwapRouter02's `exactInputSingle` accepts no deadline,
   so deadline protection requires wrapping the call in
   `multicall(uint256 deadline, bytes[] data)` — which this ABI does not include.
   For an arbitrage swap that matters, since one landing late is a guaranteed loss
   rather than a late win. The code logs a warning on every swap saying so. **Wrap
   it in multicall before trading real size.** The BSC router
   (`0xB971eF87…`) is Uniswap's documented SwapRouter02 there but has not had its
   bytecode checked, because no BSC RPC was configured.
2. **No authenticated endpoint has been exercised.** Order placement,
   cancellation and balance reads are tested against recorded Binance response
   shapes, not against the exchange. They need a testnet run with real keys.
3. **`data/target_pools_Dex.json` is stale** — September 2025 — and
   `load_pool_dataset` correctly refuses it. The `survey` command does not need
   it, but `autodiscover` does.
4. **RPC rate limiting is not governed.** The exchange side has a weight
   governor; the chain side does not. Public endpoints throttle readily, and a
   throttled quote is now recorded as `reason=rpc_error` rather than silently as
   "no liquidity" — but nothing yet paces the calls.
5. **`withdraw_networks` is empty.** Until it is populated from
   `/sapi/v1/capital/config/getall`, the system cannot tell "this token exists on
   this chain" from "the exchange will settle this token on this chain". The
   survey found a 30–53 bps standing discount on LINK/Arbitrum that is probably
   exactly this distinction.

### Fixed, with tests that fail if they return

Recorded so nobody re-fixes them, and so the shape of each defect stays visible:

- `amountOutMinimum = 0` — an unprotected swap can no longer be *constructed*:
  `DexSwapParams.min_amount_out` is a required, must-be-positive field.
- Unlimited (`2**256 - 1`) token approvals — now bounded to the amount spent.
- Legacy `gasPrice` — now EIP-1559, from config values that nothing had read.
- A market order's fill price reported as **zero** (`avgPrice` does not exist on
  Binance's spot response; `price` is the limit price).
- A resting order reported as `partially_filled`, and unknown statuses too.
- `cancel_order` returning `True` without sending anything.
- `cancel_all_on_start` / `cancel_all_on_shutdown` having no call site.
- `Opportunity.valid_until` written and never read.
- `cex.recv_window_ms` configured, validated and never sent.
- `CexOrder.tif` defaulting to IOC and being overridden with a hardcoded GTC.
- Buy-leg size units on the DEX quote.
- Two rival cost models (`cost_buffer_bps` in the spike screen, a hardcoded
  `slippage = 0.001` in the backtest).
- A backtest that could not process a single row, in four independent ways.
- Risk state shared between paper, backtest and live, so a simulation could move
  the live daily loss allowance.
- Inclusive staleness windows, so `ttl_seconds=0` still served from cache.
- `spike.py` raising on the first pool it found (`MarketPair(quote=…, symbol=…)`).
- Every CLI command dead on a fresh install (typer 0.12.3 against click 8.4).
- `requirements.txt` uninstallable on Windows (uvloop).

---

## Data Files and How to Rebuild Them

Nothing in `data/` except the pool dataset is tracked, because these are all generated artifacts. Here is every path the code touches and where it comes from.

### Generated by the discovery pipeline

Run these in order — each depends on the ones above it.

| File | Command | Requires |
|---|---|---|
| `data/master_token_list.json` | `python -m src.scanner.token_address_builder` | `COINGECKO_API_KEY` |
| `data/target_pools_Dex.json` | `python src/scanner/dex_pool_scanner.py` | `THEGRAPH_API_KEY`. Not needed by `survey`, which queries the factory directly |
| `data/auto_discovery.json` | `python -m src.cli.main autodiscover` | both files above |
| `data/discovered_pairs.yaml` | `python -m src.cli.main discover-pairs` | optional; loaded if present, and gated by the token policy and the known-chain check on load |
| `data/volume_spikes.json` | `python -m src.cli.main spike-run` | nothing; the screen now runs and reports net bps through the shared cost model |

`data/target_pools_Dex.json` is committed as a convenience snapshot, but it dates from
September 2025 and predates the `decimals` field being added to the subgraph query.
**Re-run the pool scanner before relying on it** — without `decimals`, every
stablecoin-quoted pair prices incorrectly by a factor of 10^12.

### Created automatically at runtime

| File | Written by |
|---|---|
| `data/risk_state.json` | `RiskManager` — persists daily PnL across restarts |

### Declared but unused

These appear in config or code but nothing reads or writes them. Safe to remove.

| Path | Note |
|---|---|
| `data/missing_tokens.json` | `AutoDiscoveryConfig.missing_tokens_path` is never referenced |
| `data/state.json` | `src/infra/storage.py` is never imported; `RiskManager` has its own state handling |

### Required source files, not generated

These are committed and must stay committed. An overly broad `*.json` ignore rule
previously excluded all of them, which made the repository fail on a fresh clone:

- `ABI/quoter.json`, `ABI/router.json`, `ABI/erc20.json`, `ABI/factory.json` — minimal
  ABIs containing only the functions this codebase calls. `router.json` is shaped for
  SwapRouter02, matching the addresses in `config/default.yaml`.
- `dashboard/frontend/package.json`, `tsconfig.json`, `public/manifest.json`
- `dashboard/backend/package.json`
- both `package-lock.json` files

---

## Core Workflow: Automated Opportunity Discovery

The primary method for finding and executing trades is the automated discovery workflow. This process ensures that all opportunities are based on officially verified token contracts, thereby eliminating risks from fraudulent or "scam" tokens.

This is a three-step process.

### Step 1: Build the Authoritative Token List (One-time Setup)

This is the most critical step for ensuring data integrity. We will generate a local, authoritative list of official token contract addresses by cross-referencing Binance's spot market with CoinGecko's public data.

**Command:**
```bash
python -m src.scanner.token_address_builder
```

**What it does:**
1.  Fetches all active spot market symbols from Binance.
2.  For each symbol, queries the CoinGecko API to find its official contract addresses on various chains (e.g., Ethereum, Arbitrum, Base).
3.  Saves this verified data into `data/master_token_list.json`.

> **Note:** This script communicates with the public CoinGecko API, which has a rate limit. The script includes a delay, so the initial run may take 15-20 minutes. You only need to run this periodically (e.g., weekly) to update your list with new tokens.

### Step 2: Scan for Liquid DEX Pools

Next, we scan the DEX subgraphs to build a local database of all available and sufficiently liquid pools.

**Command:**
```bash
python src/scanner/dex_pool_scanner.py
```

**What it does:**
- Queries Uniswap v3-compatible subgraphs on supported chains (Ethereum, Arbitrum, Base).
- Fetches a comprehensive list of pools that meet a minimum liquidity threshold.
- Saves the raw pool data to `data/target_pools_Dex.json`.

### Step 3: Run the Auto-Discovery Engine

This is the main engine that finds arbitrage opportunities. It cross-references CEX volume spikes with our verified local DEX data.

**Command:**
```bash
python -m src.cli.main autodiscover
```

**What it does:**
1.  Scans Binance for assets with recent volume spikes.
2.  Loads the DEX pool data from `data/target_pools_Dex.json`.
3.  Loads the official address list from `data/master_token_list.json`.
4.  **Address Verification**: It filters the DEX pools, keeping only those whose token addresses match the official addresses in our master list. This is the key step where "scam" tokens are eliminated.
5.  For each valid CEX volume spike, it finds corresponding DEX pools (both direct and synthetic) and evaluates potential arbitrage opportunities.
6.  Saves any viable opportunities into `data/auto_discovery.json`.

---

## Getting Started

### 1. Installation

```bash
# Clone the repository
git clone <repository_url>
cd DexCex_bot

# Create and activate a virtual environment.
# Use Python 3.11 - several pinned dependencies do not build on newer interpreters.
python3.11 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# Install dependencies. Use the LOCKFILE, not requirements.txt.
pip install -r requirements.lock
```

> **Install from `requirements.lock`.** `requirements.txt` pins only direct
> dependencies, and an unpinned transitive one has already broken this project
> once: click 8.4 resolved against the pinned typer 0.12.3, and *every* CLI
> command — `--help` included — raised at import time. The same
> `requirements.txt` produced a working environment one month and a dead one the
> next. Regenerate the lock with:
>
> ```bash
> uv pip compile --universal --no-header --python-version 3.11 \
>   -o requirements.lock requirements.txt
> ```
>
> `uvloop` is Linux/macOS-only and marked `sys_platform != "win32"`, so the install
> completes on Windows (without the event-loop speedup). Before the marker existed,
> its setup script raised and took every other dependency down with it.

### 2. Configuration

- Copy the `.env.example` file to `.env`:
  ```bash
  cp .env.example .env
  ```
- Edit the `.env` file and fill in your API keys and wallet private key:
  - `BINANCE_API_KEY`: Your Binance API key.
  - `BINANCE_API_SECRET`: Your Binance API secret.
  - `DEX_WALLET_PRIVATE_KEY`: The private key of the wallet you will use for DEX trades. **(CRITICAL: Handle with extreme care.)** This one is mandatory — config loading fails without it, even for read-only commands.
  - RPC URLs for the chains you intend to use (e.g., `BASE_RPC_URL`).
  - `COINGECKO_API_KEY` and `THEGRAPH_API_KEY`: required by the two discovery
    scripts in the Core Workflow above. **Neither is needed for the `survey`
    command**, which goes straight to the Uniswap factory and uses CoinGecko's free
    coin list — that is the point of it.

- Review and adjust configuration files in the `config/` directory, especially:
  - `default.yaml`: Main application settings.
  - `pairs.yaml`: For statically configured trading pairs.
  - `tokens.yaml`: Pre-defined addresses and decimals for common tokens.

Set `env: prod` in `default.yaml` before running with capital. It is not
cosmetic — the config validator then *requires* a daily loss limit, inventory
rotation pricing, the evaluation audit trail, and `token_policy.mode: allowlist`.
In `dev` all four are optional.

### 3. Running the Bot

**Start here, before anything else:**

```bash
# Does a tradeable spread exist at all? No paid API key needed.
python -m src.cli.main survey --chain ethereum --limit 40
```

If that reports `TRADEABLE above floor: 0`, there is nothing to trade and no
amount of execution work will change it. That is the current state on Ethereum,
Arbitrum and Base — see [Project Status](#project-status).

Then, to measure a specific configured universe:

```bash
python -m src.cli.main paper       # writes data/evaluations.sqlite3
python -m src.cli.main analyse     # net edge, placebo, costs, direction balance
```

`analyse` opens the database read-only and reproduces every figure in the status
section above from the rows the run wrote itself.

The `autodiscover` workflow below is a different, older path that requires the two
API keys and the pool dataset. It is still wired, but `survey` answers the same
question with fewer moving parts.

**Run in Paper Trading Mode:**
```bash
python -m src.cli.main paper
```

**What it does:**
- Loads both static pairs from `config/pairs.yaml` and dynamically discovered pairs from `data/auto_discovery.json`.
- Connects to all exchanges and starts monitoring for the loaded opportunities.
- When an opportunity is found and passes all risk checks, it will log a simulated trade but **will not** execute any real orders.

**Run in Live Trading Mode:**
> **WARNING**: Live trading involves real funds. Ensure your configuration is thoroughly tested in paper mode before proceeding.

```bash
python -m src.cli.main run
```

> `run` currently behaves identically to `paper`: the executor builds and gates the
> plan but never calls `create_order()` or `execute_swap()`. See
> [Project Status](#project-status).

## CLI Commands

| Command | What it does |
|---|---|
| `survey` | **Start here.** Does a tradeable spread exist at all, on a whole chain, at the configured notional? No paid API key. |
| `paper` | Run the strategy without placing orders. Writes every evaluation to the audit trail. |
| `analyse` | Read the audit trail: net edge, placebo comparison, per-pair costs, direction balance, rejection reasons. Read-only. |
| `run` | Live mode. Identical to `paper` today — execution is not wired. |
| `backtest` | Replay a CSV of recorded bid/ask/DEX prices through the production detector and executor. Gas must be in the data. |
| `autodiscover` | The older discovery pipeline. Needs `COINGECKO_API_KEY`, `THEGRAPH_API_KEY` and the pool dataset. |
| `discover-pairs` | Volume-anomaly scan; writes candidates for review. |
| `spike-run` | Volume-spike screen. Reports gross and net bps through the shared cost model, and marks itself depth-blind. |
| `lookup-pool` | Does a Uniswap v3 pool exist for this pair and fee tier? |
| `rebalance` | One-off CEX inventory rebalance. **The only path that places a real order today.** |
| `check-dex-balance` | Balances of every configured token in the DEX wallet. |

Use `--help` on any command. Every one of them is smoke-tested in CI, because a
dependency resolution once left all of them raising at import time while the unit
suite stayed green.
