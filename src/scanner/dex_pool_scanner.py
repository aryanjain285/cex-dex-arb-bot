import asyncio
import aiohttp
import json
import logging
import os
import sys
from datetime import datetime, timezone

from dotenv import load_dotenv

load_dotenv()

# =============================================================================
# |                                 CONFIG                                    |
# =============================================================================

# --- Subgraph Endpoints ---
# Dictionary to hold GraphQL endpoints for different chains/protocols
THEGRAPH_API_KEY = os.getenv("THEGRAPH_API_KEY", "")

# Subgraph deployment IDs, keyed by "<protocol>_<chain>".
SUBGRAPH_IDS = {
    "uniswap_ethereum": "5zvR82QoaXYFyDEKLZ9t6v9adgnptxYpKpSbxtgVENFV",
    "uniswap_polygon": "3hCPRGZc8s-ZCbhjm",
    "uniswap_base": "HMuAwufqZ1YCRmzL2SfHTVkzZovC9VL2UAKhjvRqKiR1",
    "uniswap_arbitrum": "HyW7A86UEdYVt5b9Lrw8W2F98yKecerHKutZTRbSCX27",
    "uniswap_bsc": "8f1KyiuNYiNGrjagzEVpf6k6KkPG517prtjdrJihgHw",
    "pancakeswap_bsc": "A1fvJWQLBeUAggX2WQTMm3FKjXTekNXo77ZySun4YN2m",
}

SUBGRAPH_URLS = {
    name: f"https://gateway.thegraph.com/api/{THEGRAPH_API_KEY}/subgraphs/id/{sid}"
    for name, sid in SUBGRAPH_IDS.items()
}

# --- Filtering Parameters ---
# Chain-specific wrapped native assets
NATIVE_ASSETS = {
    "uniswap_ethereum": "WETH",
    "uniswap_polygon": "WETH",
    "uniswap_base": "WETH",
    "uniswap_arbitrum": "WETH",
    "uniswap_bsc": "WETH",
    "pancakeswap_bsc": "WBNB"
}
STABLECOINS = {"USDC", "USDT", "DAI"}
# Uniswap V3 fee tiers to target (in basis points: 500 = 0.05%, 3000 = 0.3%)
TARGET_FEE_TIERS = {100, 500, 2500, 3000, 10000} # Expanded for PancakeSwap

# ======= tunable filter parameters ===========
# Minimum and maximum Total Value Locked (TVL) in USD
MIN_TVL_USD = 50_000
MAX_TVL_USD = 2_000_000
# Minimum 24-hour trading volume in USD to exclude inactive pools
MIN_VOLUME_USD = 2_000
# Number of top pools to fetch from the subgraph (ordered by volume)
POOL_FETCH_LIMIT = 1000

# --- Output ---
OUTPUT_FILENAME = "data/target_pools_Dex.json"

# --- Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# =============================================================================
# |                             CORE FUNCTIONS                                |
# =============================================================================

def build_graphql_query(limit: int) -> str:
    """
    Constructs the GraphQL query to fetch top pools by 24h volume.
    """
    return f"""
    query GetTopPools {{
      pools(
        first: {limit},
        orderBy: volumeUSD,
        orderDirection: desc
      ) {{
        id
        feeTier
        totalValueLockedUSD
        volumeUSD
        token0 {{
          id
          symbol
          decimals
        }}
        token1 {{
          id
          symbol
          decimals
        }}
      }}
    }}
    """

async def fetch_pools_from_subgraph(session: aiohttp.ClientSession, chain: str, url: str) -> list:
    """
    Asynchronously fetches pool data from a single Uniswap V3 subgraph.

    Args:
        session: The aiohttp client session.
        chain: The name of the blockchain (e.g., 'base').
        url: The subgraph URL to query.

    Returns:
        A list of pool dictionaries from the subgraph.
    """
    query = build_graphql_query(POOL_FETCH_LIMIT)
    try:
        async with session.post(url, json={"query": query}) as response:
            if response.status == 200:
                data = await response.json()
                pools = data.get("data", {}).get("pools", [])
                logging.info(f"Successfully fetched {len(pools)} pools from {chain.capitalize()}.")
                return pools
            else:
                logging.error(f"Error fetching from {chain.capitalize()}: HTTP {response.status} - {await response.text()}")
                return []
    except aiohttp.ClientError as e:
        logging.error(f"AIOHTTP client error for {chain.capitalize()}: {e}")
        return []
    except Exception as e:
        logging.error(f"An unexpected error occurred for {chain.capitalize()}: {e}")
        return []

def filter_and_format_pools(pools: list, chain_id: str) -> list:
    """
    Filters pools based on pre-defined criteria and adds a pool type for strategy selection.
    """
    filtered_list = []
    native_asset_symbol = NATIVE_ASSETS.get(chain_id)
    if not native_asset_symbol:
        logging.warning(f"No native asset configured for chain_id '{chain_id}'. Skipping.")
        return []

    for pool in pools:
        try:
            token0_sym, token1_sym = pool['token0']['symbol'], pool['token1']['symbol']

            # --- Classify the pool type ---
            pool_type = None
            is_t0_native = native_asset_symbol in token0_sym.upper()
            is_t1_native = native_asset_symbol in token1_sym.upper()
            is_t0_stable = token0_sym in STABLECOINS
            is_t1_stable = token1_sym in STABLECOINS

            # Model A: Native Asset / Stablecoin
            if (is_t0_native and is_t1_stable) or (is_t1_native and is_t0_stable):
                pool_type = "WETH_STABLE" # Keep generic type for the bot
            # Model C: Altcoin / Stablecoin
            elif (is_t0_stable and not is_t1_native) or (is_t1_stable and not is_t0_native):
                pool_type = "ALT_STABLE"
            # Model B: Altcoin / Native Asset
            elif (is_t0_native and not is_t1_stable) or (is_t1_native and not is_t0_stable):
                pool_type = "ALT_WETH" # Keep generic type for the bot
            else:
                continue # Skip pools that don't fit our models (e.g., USDC/DAI)

            # --- Apply all filtering criteria ---
            is_valid_tvl = MIN_TVL_USD <= float(pool['totalValueLockedUSD']) <= MAX_TVL_USD
            is_valid_fee = int(pool['feeTier']) in TARGET_FEE_TIERS
            is_active_pool = float(pool['volumeUSD']) >= MIN_VOLUME_USD

            if pool_type and is_valid_tvl and is_valid_fee and is_active_pool:
                # Extract protocol and chain from the combined chain_id
                protocol, chain = chain_id.split('_', 1)
                
                formatted_pool = {
                    "protocol": protocol,
                    "chain": chain,
                    "poolAddress": pool['id'],
                    "poolType": pool_type,
                    "feeTier": int(pool['feeTier']),
                    "tvlUSD": float(pool['totalValueLockedUSD']),
                    "volume24hUSD": float(pool['volumeUSD']),
                    "token0": {
                        "symbol": pool['token0']['symbol'],
                        "address": pool['token0']['id'],
                        "decimals": int(pool['token0']['decimals']),
                    },
                    "token1": {
                        "symbol": pool['token1']['symbol'],
                        "address": pool['token1']['id'],
                        "decimals": int(pool['token1']['decimals']),
                    }
                }
                filtered_list.append(formatted_pool)
        except (ValueError, TypeError, KeyError) as e:
            logging.warning(f"Could not process pool {pool.get('id', 'N/A')}: Missing data or wrong format. Details: {e}")
            continue
            
    logging.info(f"Found {len(filtered_list)} suitable pools on {chain_id} after filtering.")
    return filtered_list

def save_results_to_json(data: list, filename: str):
    """

    Saves the final list of pools to a JSON file, overwriting the previous content.
    """
    output_data = {
        "last_updated_utc": datetime.now(timezone.utc).isoformat(),
        "pool_count": len(data),
        "pools": data
    }
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=4)
        logging.info(f"Successfully saved {len(data)} pools to {filename}.")
    except IOError as e:
        logging.error(f"Failed to write to file {filename}: {e}")

# =============================================================================
# |                      OPTIONAL/ADVANCED FUNCTIONS                          |
# =============================================================================


# ==============================================================================
# |                                  MAIN                                      |
# ==============================================================================


async def estimate_slippage(pool_info: dict):
    """
    (Placeholder) Estimates slippage for a standard trade size in a given pool.
    
    To implement this, you would need:
    1. A Web3 provider (e.g., Infura, Alchemy) for the pool's chain.
    2. The Uniswap V3 QuoterV2 contract address and ABI.
    3. Use `w3.eth.contract(..).functions.quoteExactInputSingle(...).call()`
    
    This function is kept separate to avoid requiring web3 dependencies for the main script.
    """
    logging.info(f"Slippage estimation for {pool_info['poolAddress']} would be implemented here.")
    # This is where you would add the web3.py logic.
    # For now, it does nothing.
    pass

# =============================================================================
# |                                 MAIN                                      |
# =============================================================================

async def main():
    """
    Main asynchronous function to run the entire scanning and filtering process.
    """
    if not THEGRAPH_API_KEY:
        logging.error(
            "THEGRAPH_API_KEY is not set. Add it to your .env file - "
            "get a key from https://thegraph.com/studio/ (the free tier covers this workload)."
        )
        sys.exit(1)

    logging.info("Starting Uniswap V3 pool scanner...")
    all_filtered_pools = []
    
    async with aiohttp.ClientSession() as session:
        # Create tasks for all chains to run concurrently
        tasks = [
            fetch_pools_from_subgraph(session, chain, url)
            for chain, url in SUBGRAPH_URLS.items()
        ]
        chain_results = await asyncio.gather(*tasks)

    # Process results for each chain
    for i, chain_name in enumerate(SUBGRAPH_URLS.keys()):
        pools_data = chain_results[i]
        if pools_data:
            filtered_pools = filter_and_format_pools(pools_data, chain_name)
            all_filtered_pools.extend(filtered_pools)
            
    # Sort final list by TVL for better readability
    all_filtered_pools.sort(key=lambda p: p['tvlUSD'], reverse=True)

    # Save the final results
    if all_filtered_pools:
        save_results_to_json(all_filtered_pools, OUTPUT_FILENAME)
    else:
        logging.warning("No suitable pools found across any chain. The output file will not be updated.")
        
    logging.info("Scanner finished.")

    # --- Example of calling an advanced function ---
    # You can uncomment this to see it work.
    # binance_pairs = await fetch_binance_top_pairs()
    # if binance_pairs:
    #     print(f'\nTop 10 Binance pairs by volume: {list(binance_pairs)[:10]}')


if __name__ == "__main__":
    # This allows the async main function to be run from the command line.
    asyncio.run(main())