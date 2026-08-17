
import asyncio
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Dict, List, Optional, Any

import aiohttp
from loguru import logger

from src.core.config import AppConfig, load_config
from src.core.types import MarketPair
from src.exchange.univ3 import UniV3DexClient

# --- Constants ---
TARGET_QUOTE_CURRENCIES = ["USDT", "USDC"]
MASTER_TOKEN_LIST_PATH = Path("data/master_token_list.json")

# --- Data Structures ---

@dataclass
class SymbolSnapshot:
    symbol: str
    base: str
    quote: str
    status: str
    permissions: List[str]

    @classmethod
    def from_exchange(cls, payload: Dict[str, Any]) -> "SymbolSnapshot":
        return cls(
            symbol=payload.get("symbol", ""),
            base=payload.get("baseAsset", ""),
            quote=payload.get("quoteAsset", ""),
            status=payload.get("status", ""),
            permissions=payload.get("permissions", []),
        )

@dataclass
class VolumeSpike:
    symbol: str
    base: str
    quote: str
    ratio: float

@dataclass
class DexCandidate:
    chain: str
    protocol: str
    fee: int
    pool_address: str
    dex_price: float = 0.0
    raw_pool_data: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ArbitrageOpportunity:
    symbol: str
    base: str
    quote: str
    ratio: float
    cex_price: float
    dex_candidates: List[DexCandidate] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "base": self.base,
            "quote": self.quote,
            "ratio": self.ratio,
            "cex_price": self.cex_price,
            "dex_candidates": [
                {
                    "chain": cand.chain,
                    "protocol": cand.protocol,
                    "fee": cand.fee,
                    "pool": cand.pool_address,
                    "dex_price": cand.dex_price,
                    "raw_pool_data": cand.raw_pool_data,
                }
                for cand in self.dex_candidates
            ],
        }

# --- CEX Client ---

class BinancePublicClient:
    def __init__(self, base_url: str, session: aiohttp.ClientSession, concurrency: int = 20):
        self.base_url = base_url.rstrip("/")
        self.session = session
        self.semaphore = asyncio.Semaphore(concurrency)

    async def fetch_exchange_info(self) -> List[SymbolSnapshot]:
        url = f"{self.base_url}/api/v3/exchangeInfo"
        try:
            async with aiohttp.ClientSession() as temp_session:
                async with temp_session.get(url, timeout=10) as response:
                    response.raise_for_status()
                    data = await response.json()
            return [SymbolSnapshot.from_exchange(item) for item in data.get("symbols", [])]
        except Exception as e:
            logger.error(f"Failed to fetch exchange info: {e}")
            return []

    async def fetch_hourly_klines(self, symbol: str, limit: int = 2) -> List[List[Any]]:
        params = {"symbol": symbol, "interval": "1h", "limit": limit}
        try:
            async with self.semaphore:
                async with self.session.get(f"{self.base_url}/api/v3/klines", params=params, timeout=10) as response:
                    if response.status != 200:
                        logger.warning(f"Failed to get klines for {symbol} [{response.status}]")
                        return []
                    return await response.json()
        except Exception as e:
            logger.error(f"Error fetching klines for {symbol}: {e}")
            return []

    async def fetch_ticker_price(self, symbol: str) -> Optional[float]:
        params = {"symbol": symbol}
        try:
            async with self.semaphore:
                async with self.session.get(f"{self.base_url}/api/v3/ticker/price", params=params, timeout=10) as response:
                    if response.status != 200:
                        logger.warning(f"Failed to get ticker price for {symbol} [{response.status}]")
                        return None
                    data = await response.json()
                    return float(data.get("price"))
        except Exception as e:
            logger.error(f"Error fetching ticker price for {symbol}: {e}")
            return None

# --- Core Logic ---

def _normalize_dex_symbol(symbol: str) -> str:
    normalized_symbol = symbol.upper().split('.')[0]
    normalized_symbol = normalized_symbol.replace('WETH', 'ETH')
    normalized_symbol = normalized_symbol.replace('WBTC', 'BTC')
    return normalized_symbol.strip()

def build_cex_to_dex_map(dex_pools: list, master_token_list: dict) -> dict:
    mapping: Dict[str, Dict[str, List[Any]]] = {}
    logger.info(f"Building the CEX-to-DEX map from {len(dex_pools)} DEX pools (with authoritative address verification)...")
    
    address_to_symbol_map: Dict[str, Dict[str, str]] = {}
    for symbol, data in master_token_list.items():
        for chain, address in data.get("addresses", {}).items():
            if chain not in address_to_symbol_map:
                address_to_symbol_map[chain] = {}
            address_to_symbol_map[chain][address.lower()] = symbol

    for pool in dex_pools:
        chain = pool.get("chain")
        if not chain or chain not in address_to_symbol_map:
            continue

        token0_addr = pool['token0']['address'].lower()
        token1_addr = pool['token1']['address'].lower()

        token0_official_sym = address_to_symbol_map[chain].get(token0_addr)
        token1_official_sym = address_to_symbol_map[chain].get(token1_addr)

        if not (token0_official_sym and token1_official_sym):
            logger.trace(f"Skipping pool {pool.get('poolAddress')} due to unofficial tokens.")
            continue

        alt_symbol, quote_symbol, intermediate_symbol = None, None, None
        is_t0_quote = token0_official_sym in TARGET_QUOTE_CURRENCIES
        is_t1_quote = token1_official_sym in TARGET_QUOTE_CURRENCIES
        is_t0_weth = token0_official_sym == 'ETH'
        is_t1_weth = token1_official_sym == 'ETH'

        if is_t0_quote and not (is_t1_quote or is_t1_weth):
            alt_symbol, quote_symbol = token1_official_sym, token0_official_sym
        elif is_t1_quote and not (is_t0_quote or is_t0_weth):
            alt_symbol, quote_symbol = token0_official_sym, token1_official_sym
        elif is_t0_weth and not (is_t1_quote or is_t1_weth):
            alt_symbol, intermediate_symbol = token1_official_sym, 'ETH'
        elif is_t1_weth and not (is_t0_quote or is_t0_weth):
            alt_symbol, intermediate_symbol = token0_official_sym, 'ETH'
        else:
            continue

        if alt_symbol:
            pool_info = pool.copy()
            if intermediate_symbol:
                pool_info['is_synthetic'] = True
                pool_info['intermediate_symbol'] = intermediate_symbol
                for q_currency in TARGET_QUOTE_CURRENCIES:
                    cex_symbol = f"{alt_symbol}{q_currency}"
                    mapping.setdefault(cex_symbol, {}).setdefault("synthetic", []).append(pool_info)
            else:
                pool_info['is_synthetic'] = False
                for q_currency in TARGET_QUOTE_CURRENCIES:
                    cex_symbol = f"{alt_symbol}{q_currency}"
                    mapping.setdefault(cex_symbol, {}).setdefault("direct", []).append(pool_info)

    logger.info(f"Map built. Found mappings for {len(mapping)} unique CEX symbols.")
    return mapping

class AutoDiscoveryEngine:
    def __init__(self, config: AppConfig):
        self.config = config
        self.scanner_config = config.scanner.auto_discovery
        self._dex_pools: List[Dict] = []
        self._master_token_list: Dict[str, Any] = {}
        self._cex_to_dex_map: Dict[str, List[Dict]] = {}
        self.cex_client: Optional[BinancePublicClient] = None
        self.dex_client = UniV3DexClient(config.dex, config.network, config.secrets, config.tokens)

    def _load_dex_pools(self):
        dataset_path = Path(self.scanner_config.dex_pool_dataset)
        if not dataset_path.exists():
            logger.error(f"DEX pool dataset not found at: {dataset_path}")
            return
        with dataset_path.open("r", encoding="utf-8") as f:
            self._dex_pools = json.load(f).get("pools", [])
        logger.info(f"Loaded {len(self._dex_pools)} DEX pools from {dataset_path}")

    def _load_master_token_list(self):
        if not MASTER_TOKEN_LIST_PATH.exists():
            logger.error(f"Master token list not found at: {MASTER_TOKEN_LIST_PATH}. Please run token_address_builder.py first.")
            return
        with MASTER_TOKEN_LIST_PATH.open("r", encoding="utf-8") as f:
            self._master_token_list = json.load(f)
        logger.info(f"Loaded {len(self._master_token_list)} tokens from master list.")

    async def run(self):
        logger.info("Starting the AutoDiscovery engine...")
        self._load_dex_pools()
        self._load_master_token_list()
        if not self._dex_pools or not self._master_token_list:
            logger.error("Missing the DEX pool set or the authoritative token list; cannot continue.")
            return

        self._cex_to_dex_map = build_cex_to_dex_map(self._dex_pools, self._master_token_list)
        logger.info(f"Sample CEX-to-DEX map keys (first five): {list(self._cex_to_dex_map.keys())[:5]}")

        async with aiohttp.ClientSession() as session:
            self.cex_client = BinancePublicClient(self.config.cex.base_url, session, self.scanner_config.concurrency)
            spiked_symbols = await self._scan_cex_for_spikes()
            logger.info(f"Anomalous CEX symbols: {[s.symbol for s in spiked_symbols]}")
            opportunities = await self._evaluate_opportunities(spiked_symbols)
        self._save_opportunities(opportunities)
        logger.info("AutoDiscovery engine finished.")

    async def _scan_cex_for_spikes(self) -> List[VolumeSpike]:
        if not self.cex_client: return []
        logger.info("Scanning the CEX for pairs with anomalous volume...")
        snapshots = await self.cex_client.fetch_exchange_info()
        
        target_quotes = set(self.scanner_config.target_quote_currencies)
        FORBIDDEN_PERMISSIONS = {"MARGIN", "LEVERAGED", "TRD_GRP_002", "TRD_GRP_003"}

        trading_symbols = [
            s for s in snapshots 
            if s.quote in target_quotes
            and s.status == "TRADING"
            and not set(s.permissions).intersection(FORBIDDEN_PERMISSIONS)
        ]
        logger.debug(f"Found {len(trading_symbols)} SPOT trading symbols with target quote assets.")
        
        tasks = [self._fetch_and_compute_spike(snap) for snap in trading_symbols]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        spikes = [r for r in results if r and not isinstance(r, Exception)]
        spikes.sort(key=lambda s: s.ratio, reverse=True)
        top_spikes = spikes[:self.scanner_config.max_candidates]
        logger.info(f"CEX scan complete. Found {len(top_spikes)} candidates.")
        return top_spikes

    async def _fetch_and_compute_spike(self, snapshot: SymbolSnapshot) -> Optional[VolumeSpike]:
        if not self.cex_client: return None
        klines = await self.cex_client.fetch_hourly_klines(snapshot.symbol, limit=self.scanner_config.lookback_hours)
        if len(klines) < 2: return None
        try:
            current_volume = float(klines[-1][5])
            previous_volume = float(klines[-2][5])
            if previous_volume <= 0: return None
            ratio = current_volume / previous_volume
            if ratio > self.scanner_config.ratio_threshold:
                return VolumeSpike(symbol=snapshot.symbol, base=snapshot.base, quote=snapshot.quote, ratio=ratio)
        except (TypeError, ValueError, IndexError): pass
        return None

    async def _evaluate_opportunities(self, spikes: List[VolumeSpike]) -> List[ArbitrageOpportunity]:
        if not self.cex_client: return []
        logger.info(f"Evaluating arbitrage opportunities for {len(spikes)} anomalous CEX pairs...")
        opportunities: List[ArbitrageOpportunity] = []
        for spike in spikes:
            if spike.symbol not in self._cex_to_dex_map:
                logger.debug(f"No DEX mapping found for CEX symbol {spike.symbol}")
                continue

            cex_price = await self.cex_client.fetch_ticker_price(spike.symbol)
            if not cex_price: continue

            logger.success(f"Found {spike.symbol} in the DEX map. CEX price: {cex_price}")
            opportunity = ArbitrageOpportunity(symbol=spike.symbol, base=spike.base, quote=spike.quote, ratio=spike.ratio, cex_price=cex_price)

            # Correctly combine direct and synthetic candidates
            candidate_groups = self._cex_to_dex_map[spike.symbol]
            dex_candidates = candidate_groups.get("direct", []) + candidate_groups.get("synthetic", [])

            for dex_pool in dex_candidates:
                try:
                    dex_price = None
                    token0, token1 = dex_pool['token0'], dex_pool['token1']

                    if dex_pool.get('is_synthetic'):
                        intermediate_symbol = dex_pool['intermediate_symbol']
                        if _normalize_dex_symbol(token0['symbol']) == intermediate_symbol:
                            intermediate_token_details, base_token_details = token0, token1
                        else:
                            intermediate_token_details, base_token_details = token1, token0
                        
                        # Flexible fetching of intermediate CEX price (e.g., ETH/USDC or ETH/USDT)
                        intermediate_price = await self.cex_client.fetch_ticker_price(f"{intermediate_symbol}{spike.quote}")
                        if not intermediate_price:
                            logger.debug(f"Could not fetch primary intermediate price {intermediate_symbol}{spike.quote}, falling back to USDT pair.")
                            intermediate_price = await self.cex_client.fetch_ticker_price(f"{intermediate_symbol}USDT")
                        
                        if not intermediate_price: 
                            logger.warning(f"Failed to fetch any valid intermediate price for {intermediate_symbol} against stablecoins.")
                            continue
                        
                        pair_dex = MarketPair(base=_normalize_dex_symbol(base_token_details['symbol']), quote_cex=spike.quote, quote_dex=intermediate_symbol, cex_symbol=f"{base_token_details['symbol']}/{intermediate_symbol}", dex_chain=dex_pool["chain"], dex_pool_fee=dex_pool["feeTier"], base_address=base_token_details['address'], quote_address=intermediate_token_details['address'], base_decimals=base_token_details.get('decimals', 18), quote_decimals=intermediate_token_details.get('decimals', 18))
                        probe_size = Decimal(str(self.scanner_config.arbitrage.probe_size_base))
                        quote_dex = await self.dex_client.get_quote(pair_dex, probe_size, side="sell")
                        if not quote_dex or not quote_dex.price > 0: continue
                        dex_price = float(quote_dex.price * Decimal(str(intermediate_price)))
                    else:
                        if _normalize_dex_symbol(token0['symbol']) in TARGET_QUOTE_CURRENCIES:
                            quote_token_details, base_token_details = token0, token1
                        else:
                            quote_token_details, base_token_details = token1, token0

                        pair_direct = MarketPair(base=_normalize_dex_symbol(base_token_details['symbol']), quote_cex=_normalize_dex_symbol(quote_token_details['symbol']), quote_dex=_normalize_dex_symbol(quote_token_details['symbol']), cex_symbol=f"{spike.base}/{spike.quote}", dex_chain=dex_pool["chain"], dex_pool_fee=dex_pool["feeTier"], base_address=base_token_details['address'], quote_address=quote_token_details['address'], base_decimals=base_token_details.get('decimals', 18), quote_decimals=quote_token_details.get('decimals', 18))
                        probe_size = Decimal(str(self.scanner_config.arbitrage.probe_size_base))
                        quote_direct = await self.dex_client.get_quote(pair_direct, probe_size, side="sell")
                        if not quote_direct or not quote_direct.price > 0: continue
                        dex_price = float(quote_direct.price)

                    if dex_price is not None:
                        candidate = DexCandidate(chain=dex_pool["chain"], protocol=dex_pool["protocol"], fee=dex_pool["feeTier"], pool_address=dex_pool["poolAddress"], dex_price=dex_price, raw_pool_data=dex_pool)
                        opportunity.dex_candidates.append(candidate)
                        logger.info(f"  -> DEX Candidate on {dex_pool['chain']} ({dex_pool['feeTier']}): Price = {dex_price:.4f} (Synthetic: {dex_pool.get('is_synthetic', False)})")
                except Exception as e:
                    logger.warning(f"Could not evaluate DEX pool {dex_pool.get('poolAddress')}: {e}", exc_info=True)
            
            if opportunity.dex_candidates:
                opportunities.append(opportunity)
        return opportunities

    def _save_opportunities(self, opportunities: List[ArbitrageOpportunity]):
        output_path = Path(self.scanner_config.persist_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_data = {"last_updated_utc": datetime.now(timezone.utc).isoformat(), "opportunity_count": len(opportunities), "opportunities": [opp.to_dict() for opp in opportunities]}
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2)
        logger.info(f"Saved {len(opportunities)} arbitrage opportunities to {output_path}")

async def main():
    import sys
    logger.remove()
    logger.add(sys.stdout, level="DEBUG")
    config = load_config()
    engine = AutoDiscoveryEngine(config)
    await engine.run()

if __name__ == "__main__":
    asyncio.run(main())
