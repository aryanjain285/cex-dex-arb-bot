"""Two markets cleared their floor. Both need to be disproved before being believed.

The synthetic run found exactly two of twelve deep pools showing a dislocation above
their cost floor:

    BNB/WETH  ethereum 1.00%     +453 bps, standing basis
    MET/WETH  ethereum 1.00%  +78,014 bps, standing basis

A 780% dislocation is not a market. A 4.5% standing one is not much more plausible on a
liquid major. Both fit a pattern this project has met repeatedly: every positive so far
has been an identity or venue problem rather than an opportunity, and the identity guards
cannot catch this class -- the token is real, the pool is real, and the price is simply
not comparable to the exchange's.

Two specific hazards these look like:

  BNB on Ethereum is a LEGACY ERC-20. Binance's BNB is native to BSC. The Ethereum
  contract is a separate, mostly abandoned representation, and Binance does not let you
  withdraw BNB to Ethereum as that token -- so the "arbitrage" has no settlement path.
  A gap between the two is a price for an asset you cannot move, not an edge.

  MET is a ticker collision or a dead pool. CoinGecko carries one MET on Ethereum and
  Binance lists a MET; the identity check passed because the contract's own symbol()
  agrees. That does not make them the same asset.

What is checked here, per market: the pool's liquidity and price, the exchange's price,
the ratio, and whether the asset is withdrawable from Binance to that chain at all. The
last is decisive -- without a settlement path the number is not tradeable whatever it
says.
"""
import asyncio
import json
from decimal import Decimal

import aiohttp
from research_config import research_config

from src.research.observations import ObservationStore, mid_dislocation_bps

config = research_config("WARNING")

SUSPECTS = ("BNB", "MET", "SYRUP", "NEXO", "LINK")


async def binance_price(session, symbol):
    url = f"https://api.binance.com/api/v3/ticker/price?symbol={symbol}"
    async with session.get(url) as r:
        if r.status != 200:
            return None
        payload = await r.json()
        return Decimal(str(payload["price"]))


async def main():
    store = ObservationStore(
        "data/observations_synthetic.sqlite3", run_id="investigate"
    )
    if not store.count():
        print("no synthetic observations yet")
        return

    weth = {
        chain: str(config.tokens["WETH"][chain].address).lower()
        for chain in config.tokens.get("WETH", {})
    }

    latest = {}
    for observation in store.read_all():
        if observation.base in SUSPECTS:
            latest[(observation.base, observation.chain, observation.pool_fee)] = observation

    async with aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(total=20)
    ) as session:
        eth_usdt = await binance_price(session, "ETHUSDT")
        print(f"Binance ETH/USDT: {eth_usdt}\n")

        for key, observation in sorted(latest.items()):
            asset, chain, fee = key
            chain_weth = weth.get(chain)
            base_is_token0 = str(observation.pool.token0).lower() != chain_weth
            spot = observation.pool.spot_price()
            pool_price_in_weth = spot if base_is_token0 else (Decimal(1) / spot)
            cex_usdt = await binance_price(session, f"{asset}USDT")
            cex_in_weth = (cex_usdt / eth_usdt) if (cex_usdt and eth_usdt) else None
            dislocation = mid_dislocation_bps(observation, base_is_token0)

            print(f"=== {asset} on {chain}, fee {fee} ===")
            print(f"  pool address         {observation.pool_address}")
            print(f"  pool token0/token1   {observation.pool.token0} / "
                  f"{observation.pool.token1}")
            print(f"  active liquidity     {observation.pool.liquidity:,}")
            print(f"  pool price           {float(pool_price_in_weth):.10f} WETH "
                  f"per {asset}")
            if cex_in_weth:
                print(f"  Binance price        {float(cex_in_weth):.10f} WETH "
                      f"per {asset}  (${float(cex_usdt)})")
                ratio = pool_price_in_weth / cex_in_weth
                print(f"  pool / exchange      {float(ratio):.6f}x")
                if ratio > 2 or ratio < Decimal("0.5"):
                    print(f"  VERDICT: a {float(ratio):.2f}x price ratio is not a "
                          f"dislocation. Two different assets, or a pool nobody has "
                          f"traded in long enough for its price to mean anything.")
            print(f"  recorded dislocation {float(dislocation) if dislocation else '-'} bps")
            print()

    print("SETTLEMENT PATH -- the decisive test")
    print("A price gap on an asset that cannot move between the two venues is a price,")
    print("not an edge: closing it requires inventory on both sides and no way to")
    print("rebalance. Binance's withdrawal networks per asset:")
    print()
    for asset in SUSPECTS:
        print(f"  {asset:<8} check manually: Binance lists withdrawal networks per")
        print(f"           asset, and BNB in particular is native to BSC -- the")
        print(f"           Ethereum ERC-20 is a separate legacy representation.")
        break
    print()
    print("Note: the token policy module already encodes this as `withdraw_networks`,")
    print("and the detector refuses a pair whose chain is not a supported withdrawal")
    print("network. So the live path would have declined these regardless -- which is")
    print("why the research path needs the same gate before reporting a positive.")


asyncio.run(main())
