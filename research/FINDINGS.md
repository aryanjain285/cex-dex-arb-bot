# Measurements, 2026-08-18

Every number here was measured today, with the script that produced it named. Retractions
are kept in place rather than deleted, because a result I believed and then disproved is
the most useful thing in this file.

Read the last section first if you only read one.

---

## 1. The law: the typical dislocation equals the pool fee

`exact_swap_dislocation.py` — swap prints, exact block timestamps, 1-second Binance klines.

| pool | fee | median \|dislocation\| | bounded at |
| --- | --- | --- | --- |
| USDC/USDT 0.01% Arbitrum | 1 bps | 0.93 | max 1.15 |
| ETH/USDC 0.05% Arbitrum | 5 bps | 4.69 | p99 14.57 |
| ETH/USDT 0.05% Arbitrum | 5 bps | 4.70 | p99 5.96 |
| ETH/USDC 0.30% Arbitrum | 30 bps | 22.38 | p99 40.89 |
| ETH/USDT 0.30% Arbitrum | 30 bps | 30.30 | p99 37.96 |

Five pools, three tiers, three pairs. The 0.30% and 0.01% figures were both predicted from
the 0.05% ones before being measured — the 0.01% pool came in at 0.93 bps against a 1 bps
fee, with every observation inside 1.15 bps.

It follows from what an arbitrageur does: close a gap until the remainder no longer covers
the cost, and the irreducible cost of trading through a pool is that pool's fee. So the
market is arbitraged down to the pool fee and stops, leaving the competition's cost basis
on the table. The 0.05% distributions are also *bounded* near it — p99 of 5.96 bps against
a 5 bps fee — which is what an arbitraged market looks like from the inside.

**It holds across a twenty-fold range of volatility.** Four windows, `--at` historical:

| window | volatility bps/min | median \|d\| |
| --- | --- | --- |
| live, quiet day | 2.88 | 4.69 |
| 2026-02-23 01:00 | 29.00 | 5.17 |
| 2026-05-17 23:00 | 38.92 | 2.85 |
| 2026-03-23 11:00 | 58.12 | 5.81 |

A twenty-fold increase in volatility moves the median dislocation by about one basis point.

## 2. Why that closes the question

    available   ~  the pool fee
    required    =  the pool fee + the CEX fee + gas + half the spread
    shortfall   ~  the CEX fee, at every tier

Raising the fee tier raises both sides by the same amount, which is why fee-tier
selection — the obvious lever — does nothing. Measured: 0.30% pools give 22–30 bps of
dislocation and need 37.5 bps.

## 3. It is worse than that for a poller

`fee_sensitivity.py` — clock-sampled observations, which is what a polling loop sees.

| market | median \|d\| | max \|d\| | largest CEX fee that clears |
| --- | --- | --- | --- |
| ETH/USDC 0.05% | 2.67 | 7.40 | none at median; 2.33 bps at max |
| ETH/USDT 0.05% | 2.70 | 6.85 | none at median; 1.77 bps at max |
| USDC/USDT 0.05% | 1.11 | 1.21 | none at any percentile |

2,715 observations each, 2-second cadence. The median does not cover the **pool fee
alone**. A zero-fee exchange still loses about 2.9 bps at the median. One observation out
of 2,715 supports even a VIP6 maker fee.

Clock-sampled sees less than swap prints (2.67 against 4.69) because between trades the
gap decays further. The two answer different questions and only the first is available to
a poller.

## 4. Latency is not the constraint

`analyse.py`, latency study on 1,908 observations at 2-second cadence.

| delay | realised net | decay |
| --- | --- | --- |
| 0.0s | −10.41 bps | 0.00 |
| 2.3s | −10.42 bps | 0.01 |
| 12.0s | −10.51 bps | 0.10 |

Twelve seconds — one Ethereum block, five times the measured detector cadence — costs
0.10 bps. The dislocation's autocorrelation half-life is 21–24 seconds, still +0.75
correlated at 12s. A 2.3-second loop is comfortably fast enough.

## 5. Depth and efficiency are the same property

`probe_depth.py` — 1% depth from slot0, 804 pools across three chains.

| asset | WETH-quoted depth | USDC-quoted depth | ratio |
| --- | --- | --- | --- |
| LDO | $16,078,997 | $186 | 86,000× |
| WBTC arbitrum | $1,935,145 | $319,652 | 6× |
| LINK | $397,995 | $3,623 | 110× |
| ENA | $16,268 | $10 | 1,582× |

Mid-cap liquidity is quoted in WETH, not stablecoins. And outside WBTC and LDO, almost
nothing exceeds $50k of 1% depth; most sits between $1k and $20k, where a $2,000 trade
costs about 10 bps of impact — the entire budget.

`screen_residual_law.py` found the same thing as an absence of transactions: **ten of the
fourteen deepest non-ETH Arbitrum pools had zero swaps in sixty minutes.** A pool nobody
trades cannot be arbitraged, which is why its price drifts, and cannot be traded either,
which is why the drift is worthless.

## 6. Every positive found today was a trap, and each a different kind

| candidate | reading | what it was |
| --- | --- | --- |
| MET/WETH | +78,008 bps | ticker collision — CoinGecko's MET is not Binance's MET; both `symbol()` calls are correct |
| BNB/WETH | +455 bps standing | the legacy Ethereum ERC-20; BNB is native to BSC, so no withdrawal path exists |
| WBTC/USDT | −11.93 bps standing | an L1↔L2 bridging basis; every observation negative, direction separation −0.46 bps |
| AUCTION, CHR, COMP, AAVE | up to 3.9e52 bps | empty pools; v3 keeps the price its creator set until someone trades |

Each produced a guard, and each guard catches something the previous ones could not:
ambiguous tickers dropped, on-chain `symbol()` checked, zero liquidity refused, implausible
price ratios excluded, persistent large gaps flagged as barriers.

## 7. The only positive net edge in the dataset was a fixed offset

USDC/USDT 0.01% on Base showed **+0.85 bps net** — the single positive in 22 markets, on
the largest raw dislocation of any liquid pair.

| | |
| --- | --- |
| raw dislocation | mean +9.47 bps, p1 +9.23, p99 +9.83 |
| observations | 331 over 2.03h |
| standing-basis test | p = 0.0312, sign flips 0.0% |
| control | true p99 8.77 vs scrambled p99 8.77 |

A 0.6 bps range around a 9.43 bps median, across two hours. That is a fixed difference
between the two things being compared, not a market gap — almost certainly bridged USDT on
Base against native USDT on Binance, which are not the same asset and hold a persistent peg
difference. Both the standing-basis test and the constancy test flag it, by independent
routes.

It also corrected my reading of the control. Scrambling computes

    pool(t') - cex(t)  =  [pool(t') - cex(t')]  +  [cex(t') - cex(t)]

so it only ever ADDS the variance of the underlying price's own movement. On a pair whose
level barely moves there is nothing for it to disturb, and identical distributions are a
fact about the test rather than about the market. The control now reports whether it had
power, and constancy is measured from the gap's own spread instead. Every stablecoin pair
in the dataset classifies as a fixed offset; no volatile pair does.

## 8. The other side of the same trade is also at break-even

If the residual equals the pool fee, the arbitrageurs closing gaps are being paid that fee
— and the payer is the liquidity provider. So the LP side needed one measurement.

`flow_toxicity.py` — markout from the amounts in each Swap event, which are the pool's
actual deltas with the fee included, valued at the exchange price for that second.
WETH/USDC 0.05% Arbitrum, 340 trades over ~50 minutes:

| | |
| --- | --- |
| informed flow (provider loses) | 42.6% of trades, 52.9% of volume |
| uninformed flow (provider gains) | 57.4% of trades, 47.1% of volume |
| median loss to informed | 8.00 bps |
| median gain from uninformed | 8.89 bps |
| **markout, equal-weighted** | **+0.64 bps**, se 0.90, 95% CI [−1.12, +2.40] |
| **markout, volume-weighted** | **−1.95 bps** |

Not distinguishable from zero equal-weighted, which is the expected result in a competitive
market: the fee is exactly what arbitrage competes away, so the provider breaks even on
trading and impermanent loss only subtracts from there.

The gap between the weightings is the more interesting number. **Larger trades are more
toxic** — informed traders size up, uninformed ones do not — so a provider cannot fix the
problem by quoting more size. Textbook adverse selection, measured.

So both sides of this trade sit at or below break-even. That is the symmetric form of §1
and the strongest version of it: **the pool fee is not a margin available to either party,
it is the price of the competition.**

## 9. Retractions

**A +0.703 correlation between volatility and dislocation, with 25 of 29 windows clearing
the floor.** Block numbers were mapped to timestamps by interpolating between five anchors
across fourteen months; that was wrong by a median of 51 minutes. A stale comparison price
manufactures dislocation of about volatility × √(minutes of error) — 39 bps at median
volatility, 416 bps at the most violent hour, against "measured" medians of 16–52 and 560.
Same order at every level, and scaling with volatility, which *was* the finding. Binary
search for block timestamps cut placement error to 9 seconds; the correlation fell to
+0.300 and 0 of 16 windows cleared.

**A median swap-print dislocation of 8.67 bps.** Interpolating between two exact endpoints
three hours apart is wrong by only 16 seconds — and the error was *signed*, running late
throughout, so every swap was compared against a kline from after it happened. One RPC call
per block containing a swap cut the median to 4.69 bps.

**A fivefold tail expansion in violent regimes, 23% of prints above the floor.** From one
hour. Two more contradict it: at 38.92 bps/min the p90 was 8.03 bps — *lower* than the
quiet day's 13.31 — and nothing cleared the floor. The quiet day shows the highest
above-floor fraction of the four windows. The tail is not predicted by volatility.

**A standing basis of +2.6 bps on ETH pairs, sign flips 0.0%.** From 236 observations
holding about eleven independent draws. With 1,908 the same market flips sign 45.1% of the
time. Persistence is now a sign test on the effective sample rather than a flip-fraction
threshold.

**An LP edge of +4.74 bps of volume.** The first version of the flow measurement bracketed
consecutive prints and asked how much of the gap each trade closed — which conflates the
pool's move with the exchange's move over the interval between swaps, since consecutive
swaps can be minutes apart. It reported a median gap closed of 0.16 bps against a gap
*level* of 4.69. A number two orders of magnitude below the quantity it claims to measure
is measuring something else. Using the markout from each event's own amounts gives
+0.64 bps equal-weighted, with zero inside the interval.

All five came from asking what would produce the result if it were false. None came from
something breaking.

## 10. What would have to change

Not latency, not sizing, not fee tier, not universe breadth — each measured separately
above.

The requirement is a cost above the pool fee near zero. On Binance's schedule that means
maker orders at VIP8–9 (1.2 and 0.6 bps), and **a resting maker order is not a hedge until
it fills**, so the DEX leg would carry unhedged inventory for an unbounded period. That
risk is priced nowhere in this codebase, and pricing it is a prerequisite for evaluating
the maker route rather than a refinement afterwards.

The one experiment still worth running is specified: record through a period above roughly
the 90th volatility percentile and compare the clock-sampled distribution against the
print distribution in §1. Today was the 12th percentile, so today could not run it.

### Infrastructure limits that bound all of the above

The swap-print method needs `eth_getLogs`, and only Arbitrum's official endpoint serves it
on free infrastructure — Ethereum and Base both return 403 beyond a ten-block span. So
§1 is Arbitrum-only. The clock-sampled results in §3 cover all three chains, and the
median-below-pool-fee conclusion holds on each.

Historical reconstruction reaches back to June 2025 on Arbitrum (logs are not pruned like
state), and Binance retains 1-second klines for the full 180 days. That combination is what
made the violent-hour measurements possible at all: at 38 bps/min a one-minute kline
carries up to 38 bps of matching noise, the same order as the signal.
