"""Which pools actually exist for the token identities already in config.

The configured universe is three pairs on the venues that are arbitraged hardest in
crypto. Widening it properly needs new token identities, and every new identity is
an opportunity to record a counterfeit -- this project has already met a fake WETH
and a legacy BNB. So the first widening uses only identities already verified in
config, and varies the two things that cost nothing to vary:

  FEE TIER. A 0.30% pool on the same pair is a different market from the 0.05% one:
  less liquidity, less competition, and a 25 bps larger fee to overcome. Whether the
  extra dislocation exceeds the extra fee is a testable question and the answer is
  not obvious.

  CHAIN. WETH/USDC on Arbitrum and on Base are the same token identities and
  different markets. L2 pools are arbitraged by fewer participants against the same
  CEX book, and gas there is a hundredth of mainnet -- which changes the minimum
  viable size by two orders of magnitude.

Writes the discovered set to JSON so a restart does not re-run discovery.
"""
import asyncio
import json
import sys
from pathlib import Path

from research_config import research_config

from src.exchange.univ3 import UniV3DexClient

config = research_config("WARNING")

# (base, quote, cex_symbol) triples whose identities are already in config/tokens.
CANDIDATES = [
    ("WETH", "USDT", "ETH/USDT"),
    ("WETH", "USDC", "ETH/USDC"),
    ("ARB", "USDT", "ARB/USDT"),
    ("ARB", "USDC", "ARB/USDC"),
    ("USDC", "USDT", "USDC/USDT"),
]
CHAINS = ["ethereum", "arbitrum", "base"]
FEES = [100, 500, 3000, 10000]

OUT = Path("targets.json")


async def main():
    client = UniV3DexClient(config.dex, config.network, config.secrets, config.tokens)
    found = []
    for base, quote, symbol in CANDIDATES:
        for chain in CHAINS:
            if base not in config.tokens or chain not in config.tokens[base]:
                continue
            if quote not in config.tokens or chain not in config.tokens[quote]:
                continue
            for fee in FEES:
                try:
                    address = await client.get_pool_address(base, quote, chain, fee)
                except Exception as exc:
                    print(f"  {symbol:10s} {chain:9s} {fee:>5}  error: {type(exc).__name__}")
                    continue
                if not address:
                    continue
                found.append({
                    "base": base, "quote": quote, "cex_symbol": symbol,
                    "chain": chain, "fee": fee, "pool_address": address,
                    "base_address": config.tokens[base][chain].address,
                    "quote_address": config.tokens[quote][chain].address,
                    "base_decimals": config.tokens[base][chain].decimals,
                    "quote_decimals": config.tokens[quote][chain].decimals,
                })
                print(f"  {symbol:10s} {chain:9s} {fee:>5}  {address}")

    OUT.write_text(json.dumps(found, indent=2), encoding="utf-8")
    print(f"\n{len(found)} pools discovered -> {OUT}")
    by_chain = {}
    for f in found:
        by_chain[f["chain"]] = by_chain.get(f["chain"], 0) + 1
    print("by chain:", by_chain)


asyncio.run(main())
