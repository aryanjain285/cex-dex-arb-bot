# Research toolchain

Read-only measurement of whether a CEX↔DEX price dislocation exists, how large it is,
how long it lasts, and whether any size can capture it. Nothing here can trade: every
script loads its config with `require_signing_key=False` and points at
`.env.research`, so the process holds no wallet key and `UniV3DexClient` raises
`ReadOnlyWalletError` if any code path reaches a signing call.

Run everything from the repository root, with `research/` on the path:

```
PYTHONPATH=research python research/<script>.py
```

## The pipeline

| step | script | what it does |
| --- | --- | --- |
| 1 | `discover_targets.py` | Pools for token identities already verified in `config/tokens.yaml`, across fee tiers and chains. Writes `targets.json`. |
| 1b | `expand_universe.py` | Widens to every Binance spot asset with an unambiguous CoinGecko address and a v3 pool. Writes `targets_wide.json`. |
| 2 | `record.py` | Records raw pool state and full CEX ladders into `data/observations.sqlite3`. |
| 3 | `analyse.py` | Statistics, exceedance curves, latency study, decay profile, negative control. |

Verification, which is the part worth keeping:

| script | what it proves |
| --- | --- |
| `validate_stored_vs_chain.py` | A stored observation, reloaded and priced locally, matches the deployed QuoterV2 **at the block it recorded**. 44/44 exact across 22 markets. |
| `validate_identity.py` | `gross(CEX→DEX) + gross(DEX→CEX) = −(2·pool_fee + spread + impact)` on recorded data. Catches every units, token-order and extrapolation defect at once. |
| `verify_batching_live.py` | The Multicall3 read reproduces the call-by-call read byte for byte, 19–20× cheaper. |
| `measure_window_cost.py` | What each tick-window width costs in RPC calls and buys in maximum priceable size. |

## Reading the output

The report deliberately separates numbers that are usually conflated.

**Raw dislocation** — pool mid against CEX mid, before any fee, spread, impact or
choice of direction. The only figure with nothing subtracted, and therefore the one
that says whether the phenomenon exists at all. Every other number is ambiguous
without it: a negative net edge can mean "the venues are at parity and the fees are
unavoidable" or "the venues disagree, just not enough", and those have opposite
implications.

**Standing basis vs fluctuating** — whether the dislocation's sign changes. A
persistent gap is a price, not an error: what the market charges for the asset being
on that chain rather than in that custodian. Capturing it once is an inventory move;
capturing it again needs the inventory bridged back, and the bridge costs the basis.
Reported as a per-trade edge it would be counted many times over.

**Effective n** — always printed beside the raw count. Observations seconds apart are
nearly the same fact recorded twice, so using the row count for a standard error
inflates every t-statistic by `sqrt(n/n_eff)`. Measured on this data: 236 observations
at a 2s cadence carry 11 independent facts, a factor of 21. Sampling *faster* makes
that worse, so the error rewards collecting more data.

**Uncostable and unpriceable counts** — printed next to the statistics, not under
them. An observation the simulator refused to price is not an observation of no edge.

**Negative control** — each market's true pairing against one where the CEX book is
matched to a pool snapshot from a distant time. The headline statistic is the best of
two directions, which is a max over two nearly-opposite quantities, so noise is
rectified rather than cancelled: two venues tracking each other perfectly show −5.5 bps
truly and +93 bps scrambled. A positive mean best-gross edge is not evidence of
anything on its own; only the excess over the scrambled tail is.

## Constraints these runs operate under

Public RPC endpoints. `mainnet.base.org` refused at 3 req/s and drove the limiter to
its floor within seconds; `base-rpc.publicnode.com` serves the same load. A full pool
read is 9–12 calls batched against 172–184 unbatched, but still 3–50s of wall clock on
Ethereum, so tick data is re-read on a 600s timer rather than the live default of 120s.
The recorder measures and reports its achieved cadence, its per-pair failure counts and
the effective rate each chain settled at — a gap in a time series is otherwise
indistinguishable from a quiet market.

Gas is stored as a **price**, never a cost, so any gas-limit assumption can be applied
at analysis time. A missing gas price makes an observation uncostable rather than free:
a zero gas cost is the single easiest way to make this strategy look profitable, since
its edge and its gas are the same order of magnitude.
