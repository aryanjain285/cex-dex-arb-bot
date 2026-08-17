import asyncio
import json
import os
from pathlib import Path
from typing import Dict, List, Any

import aiohttp
from loguru import logger
from dotenv import load_dotenv

# --- Constants ---
BINANCE_API_URL = "https://api.binance.com/api/v3/exchangeInfo"
CG_API_BASE_URL = "https://api.coingecko.com/api/v3" # Use Pro API endpoint
MASTER_LIST_PATH = Path("data/master_token_list.json")
TARGET_QUOTE_ASSETS = {"USDT", "USDC", "BUSD", "TUSD"}
TARGET_CHAINS = {"ethereum", "arbitrum-one", "base"}

# --- Main Builder Class ---

class MasterTokenListBuilder:
    def __init__(self, api_key: str):
        if not api_key:
            raise ValueError("CoinGecko API key is required. Please set it in your .env file.")
        self.api_key = api_key
        self.session: aiohttp.ClientSession
        self.binance_symbols: set[str] = set()
        self.cg_coin_list: List[Dict[str, Any]] = []
        self.master_token_list: Dict[str, Any] = {}

    async def run(self):
        logger.info("Building the authoritative token address list...")
        MASTER_LIST_PATH.parent.mkdir(exist_ok=True)

        headers = {"x-cg-pro-api-key": self.api_key}
        async with aiohttp.ClientSession(headers=headers) as self.session:
            await self._get_binance_spot_symbols()
            await self._get_coingecko_coin_list()
            self._build_master_list()
        
        self._save_master_list()
        logger.success(f"Authoritative token list complete, saved to {MASTER_LIST_PATH}")

    async def _get_binance_spot_symbols(self):
        logger.info("Fetching all spot-trading base assets from Binance...")
        try:
            # No API key needed for this Binance endpoint
            async with aiohttp.ClientSession() as temp_session:
                response = await temp_session.get(BINANCE_API_URL)
                response.raise_for_status()
                data = await response.json()
            
            FORBIDDEN_PERMISSIONS = {"MARGIN", "LEVERAGED", "TRD_GRP_002", "TRD_GRP_003"}
            for symbol_data in data.get("symbols", []):
                permissions = set(symbol_data.get("permissions", []))
                if symbol_data.get("status") == "TRADING" and symbol_data.get("quoteAsset") in TARGET_QUOTE_ASSETS and not permissions.intersection(FORBIDDEN_PERMISSIONS):
                    self.binance_symbols.add(symbol_data.get("baseAsset"))
            logger.info(f"Found {len(self.binance_symbols)} distinct base assets on Binance.")
        except Exception as e:
            logger.error(f"Error fetching symbol information from Binance: {e}")

    async def _get_coingecko_coin_list(self):
        logger.info("Fetching the full CoinGecko coin list with platform addresses...")
        params = {"include_platform": "true"}
        try:
            response = await self.session.get(f"{CG_API_BASE_URL}/coins/list", params=params)
            response.raise_for_status()
            self.cg_coin_list = await response.json()
            logger.info(f"Found {len(self.cg_coin_list)} coins on CoinGecko.")
        except Exception as e:
            logger.error(f"Error fetching the coin list from CoinGecko: {e}")

    def _build_master_list(self):
        logger.info("Walking the CoinGecko list to filter and build the authoritative list...")
        for coin in self.cg_coin_list:
            symbol = coin.get("symbol", "").upper()
            if symbol in self.binance_symbols:
                platforms = coin.get("platforms", {})
                if not platforms: continue

                addresses = {}
                for platform_id, address in platforms.items():
                    if address and platform_id in TARGET_CHAINS:
                        chain_name = "arbitrum" if platform_id == "arbitrum-one" else platform_id
                        addresses[chain_name] = address
                
                if addresses:
                    self.master_token_list[symbol] = {
                        "coingecko_id": coin.get("id"),
                        "name": coin.get("name"),
                        "addresses": addresses
                    }
        logger.info(f"Matched on-chain addresses for {len(self.master_token_list)} Binance assets.")

    def _save_master_list(self):
        with open(MASTER_LIST_PATH, "w", encoding="utf-8") as f:
            json.dump(self.master_token_list, f, indent=2)

# --- Main Execution ---

async def main():
    load_dotenv()
    api_key = os.getenv("COINGECKO_API_KEY")
    try:
        builder = MasterTokenListBuilder(api_key)
        await builder.run()
    except ValueError as e:
        logger.error(e)

if __name__ == "__main__":
    import sys
    logger.remove()
    logger.add(sys.stdout, level="INFO")
    asyncio.run(main())