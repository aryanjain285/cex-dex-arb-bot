"""Utilities for discovering new trading pairs based on CEX volume anomalies."""
from __future__ import annotations

import asyncio
import datetime as dt
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import aiohttp
import yaml
from loguru import logger

from src.core.config import AppConfig, PairConfig, VolumeScannerConfig
from src.exchange.rate_limit import (
    IpBannedError, WeightGovernor, get_shared_governor, governed_request,
)
from src.exchange.univ3 import UniV3DexClient


@dataclass
class SymbolInfo:
    """Simplified representation of a Binance symbol."""

    symbol: str
    base_asset: str
    quote_asset: str
    status: str
    filters: Dict[str, Dict[str, str]]
    cex_symbol: str
    base_precision: int
    quote_precision: int

    @staticmethod
    def _precision_from_step(step: str) -> int:
        step = step.strip()
        if step in ("0", "0.0", "0E-8"):
            return 0
        if "E" in step or "e" in step:
            normalized = f"{Decimal(step):f}"
        else:
            normalized = step
        if "." not in normalized:
            return 0
        fractional = normalized.rstrip("0").split(".")[-1]
        return max(len(fractional), 0)

    @classmethod
    def from_exchange_info(cls, data: Dict[str, object]) -> "SymbolInfo":
        filters = {f["filterType"]: f for f in data.get("filters", [])}
        lot_size = filters.get("LOT_SIZE", {}).get("stepSize", "1")
        price_filter = filters.get("PRICE_FILTER", {}).get("tickSize", "1")
        base_precision = cls._precision_from_step(lot_size)
        quote_precision = cls._precision_from_step(price_filter)
        base_asset = data.get("baseAsset", "")
        quote_asset = data.get("quoteAsset", "")
        symbol = data.get("symbol", "")
        cex_symbol = f"{base_asset}/{quote_asset}"
        return cls(
            symbol=symbol,
            base_asset=base_asset,
            quote_asset=quote_asset,
            status=data.get("status", ""),
            filters=filters,
            cex_symbol=cex_symbol,
            base_precision=base_precision,
            quote_precision=quote_precision,
        )


@dataclass
class VolumeAnomaly:
    symbol: SymbolInfo
    last_hour_quote_volume: float
    avg_quote_volume: float
    volume_ratio: float
    window_start: dt.datetime
    window_end: dt.datetime

    def to_metadata(self) -> Dict[str, object]:
        return {
            "source": "volume_scanner",
            "discovered_at": dt.datetime.utcnow().isoformat(timespec="seconds"),
            "last_hour_quote_volume": round(self.last_hour_quote_volume, 4),
            "avg_quote_volume": round(self.avg_quote_volume, 4),
            "volume_ratio": round(self.volume_ratio, 4),
            "window_start": self.window_start.isoformat(),
            "window_end": self.window_end.isoformat(),
        }


@dataclass
class ValidatedPair:
    pair_config: PairConfig
    metadata: Dict[str, object]


class BinanceDataSource:
    """Lightweight REST client for Binance public endpoints."""

    def __init__(
        self,
        base_url: str,
        lookback_hours: int,
        governor: Optional[WeightGovernor] = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.lookback_hours = lookback_hours
        # The semaphore bounds how many requests are in flight; the governor
        # bounds the WEIGHT they cost. The exchange limits the latter, per IP, so
        # only the governor can prevent the 418 ban that would also kill the
        # market-data WebSocket.
        self._semaphore = asyncio.Semaphore(5)
        self.governor = governor or WeightGovernor()

    async def fetch_exchange_info(self, session: aiohttp.ClientSession) -> List[SymbolInfo]:
        url = f"{self.base_url}/api/v3/exchangeInfo"
        data = await governed_request(
            session, self.governor, "GET", url,
            timeout=aiohttp.ClientTimeout(total=10),
        )
        symbols = data.get("symbols", [])
        return [SymbolInfo.from_exchange_info(item) for item in symbols]

    async def fetch_hourly_klines(self, session: aiohttp.ClientSession, symbol: SymbolInfo) -> List[List[str]]:
        params = {
            "symbol": symbol.symbol,
            "interval": "1h",
            "limit": min(self.lookback_hours + 1, 500),
        }
        url = f"{self.base_url}/api/v3/klines"
        async with self._semaphore:
            try:
                return await governed_request(
                    session, self.governor, "GET", url, params=params,
                    timeout=aiohttp.ClientTimeout(total=10),
                )
            except IpBannedError:
                # Fatal: stop the scan rather than extend the ban.
                raise
            except Exception as exc:
                logger.warning(f"Failed to fetch klines for {symbol.symbol}: {exc}")
                return []


class DiscoveredPairStore:
    """Persistence helper for dynamically discovered pairs."""

    def __init__(self, path: Path):
        self.path = path

    def load(self) -> Dict[str, object]:
        if not self.path.exists():
            return {"pairs": []}
        with self.path.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
        if "pairs" not in data:
            data["pairs"] = []
        return data

    def save(self, data: Dict[str, object]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(data, handle, sort_keys=False, allow_unicode=False)


class VolumeScannerService:
    def __init__(self, config: AppConfig):
        self.config = config
        self.volume_cfg: VolumeScannerConfig = config.scanner.volume
        self.datasource = BinanceDataSource(
            config.cex.base_url, self.volume_cfg.lookback_hours,
            governor=get_shared_governor(
                config.cex.max_request_weight_per_minute,
                config.cex.request_weight_safety_fraction,
            ),
        )
        self.store = DiscoveredPairStore(Path(self.volume_cfg.persist_path))

    async def run(self) -> List[ValidatedPair]:
        if not self.volume_cfg.enabled:
            logger.info("Volume scanner is disabled; skipping dynamic pair discovery.")
            return []

        async with aiohttp.ClientSession() as session:
            symbol_infos = await self.datasource.fetch_exchange_info(session)
            filtered_symbols = [
                s for s in symbol_infos
                if s.quote_asset in self.volume_cfg.quote_assets and s.status == "TRADING"
            ]
            anomalies = await self._find_volume_anomalies(session, filtered_symbols)

        if not anomalies:
            logger.info("No qualifying volume anomalies found.")
            return []

        dex_client = UniV3DexClient(
            self.config.dex,
            self.config.network,
            self.config.secrets,
            self.config.tokens,
        )
        validated = await self._validate_with_dex(dex_client, anomalies)
        if not validated:
            logger.info("No pairs also exist on Uniswap V3.")
            return []

        self._persist_results(validated)
        return validated

    async def _find_volume_anomalies(
        self,
        session: aiohttp.ClientSession,
        symbols: List[SymbolInfo],
    ) -> List[VolumeAnomaly]:
        tasks = [self.datasource.fetch_hourly_klines(session, sym) for sym in symbols]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        anomalies: List[VolumeAnomaly] = []
        for sym, klines in zip(symbols, results):
            if isinstance(klines, Exception):
                logger.warning(f"Error fetching kline data for {sym.symbol}: {klines}")
                continue
            if len(klines) < 2:
                continue

            # the last candle is the most recent closed 1h bar
            last_candle = klines[-1]
            history = klines[-(self.volume_cfg.lookback_hours + 1):-1] if len(klines) > 1 else []
            if not history:
                continue

            try:
                last_volume = float(last_candle[7])  # quote asset volume
            except (ValueError, TypeError):
                continue

            history_volumes: List[float] = []
            for candle in history:
                try:
                    history_volumes.append(float(candle[7]))
                except (ValueError, TypeError):
                    continue

            if not history_volumes:
                continue

            avg_volume = sum(history_volumes) / len(history_volumes)
            if avg_volume <= 0:
                continue

            ratio = last_volume / avg_volume if avg_volume else 0
            if last_volume < self.volume_cfg.min_hourly_quote_volume:
                continue
            if ratio < self.volume_cfg.anomaly_multiplier:
                continue

            open_ts = int(last_candle[0]) // 1000
            close_ts = int(last_candle[6]) // 1000
            anomaly = VolumeAnomaly(
                symbol=sym,
                last_hour_quote_volume=last_volume,
                avg_quote_volume=avg_volume,
                volume_ratio=ratio,
                window_start=dt.datetime.utcfromtimestamp(open_ts),
                window_end=dt.datetime.utcfromtimestamp(close_ts),
            )
            anomalies.append(anomaly)

        anomalies.sort(key=lambda a: (a.volume_ratio, a.last_hour_quote_volume), reverse=True)
        return anomalies[: self.volume_cfg.max_candidates]

    async def _validate_with_dex(
        self,
        dex_client: UniV3DexClient,
        anomalies: List[VolumeAnomaly],
    ) -> List[ValidatedPair]:
        validated: List[ValidatedPair] = []
        for anomaly in anomalies:
            result = await self._find_matching_pool(dex_client, anomaly)
            if not result:
                continue
            pair_cfg, metadata = result
            validated.append(ValidatedPair(pair_config=pair_cfg, metadata=metadata))
        return validated

    async def _find_matching_pool(
        self,
        dex_client: UniV3DexClient,
        anomaly: VolumeAnomaly,
    ) -> Optional[Tuple[PairConfig, Dict[str, object]]]:
        base = anomaly.symbol.base_asset
        quote = anomaly.symbol.quote_asset

        for chain in self.volume_cfg.preferred_chains:
            if not self._token_available_on_chain(dex_client, base, chain):
                logger.debug(
                    f"Skipping {anomaly.symbol.cex_symbol}: tokens.yaml has no address for {base} on {chain}."
                )
                continue
            if not self._token_available_on_chain(dex_client, quote, chain):
                logger.debug(
                    f"Skipping {anomaly.symbol.cex_symbol}: tokens.yaml has no address for {quote} on {chain}."
                )
                continue
            for fee in self.volume_cfg.preferred_fee_tiers:
                pool_address = await dex_client.get_pool_address(base, quote, chain, fee)
                if not pool_address:
                    continue

                pair_data = {
                    "base": base,
                    "quote": quote,
                    "cex_symbol": anomaly.symbol.cex_symbol,
                    "min_net_bps": self.volume_cfg.default_pair_params.get("min_net_bps"),
                    "max_slippage_bps": self.volume_cfg.default_pair_params.get("max_slippage_bps", 50),
                    "max_size_quote": self.volume_cfg.default_pair_params.get("max_size_quote", 1000),
                    "dex_chain": chain,
                    "dex_pool_fee": fee,
                    "price_floor_quote": self.volume_cfg.default_pair_params.get("price_floor_quote"),
                    "price_ceiling_quote": self.volume_cfg.default_pair_params.get("price_ceiling_quote"),
                    "max_edge_bps": self.volume_cfg.default_pair_params.get("max_edge_bps"),
                    "base_precision": anomaly.symbol.base_precision,
                    "quote_precision": anomaly.symbol.quote_precision,
                }

                pair_cfg = PairConfig(**pair_data)
                metadata = anomaly.to_metadata()
                metadata.update(
                    {
                        "dex_chain": chain,
                        "dex_fee_tier": fee,
                        "dex_pool_address": pool_address,
                    }
                )
                return pair_cfg, metadata

        logger.debug(
            f"Skipping {anomaly.symbol.cex_symbol}: no matching pool on the configured chains or fee tiers."
        )
        return None

    @staticmethod
    def _token_available_on_chain(dex_client: UniV3DexClient, symbol: str, chain: str) -> bool:
        return (
            symbol in dex_client.tokens_config
            and chain in dex_client.tokens_config[symbol]
        )

    def _persist_results(self, validated_pairs: List[ValidatedPair]) -> None:
        current_data = self.store.load()
        pairs: List[Dict[str, object]] = current_data.get("pairs", [])
        index = {self._extract_symbol(entry): i for i, entry in enumerate(pairs)}

        updated = False
        now = dt.datetime.utcnow()
        cooldown = dt.timedelta(hours=self.volume_cfg.cooldown_hours)

        for vp in validated_pairs:
            symbol = vp.pair_config.cex_symbol
            pair_payload = (
                vp.pair_config.model_dump(exclude_none=True)  # type: ignore[attr-defined]
                if hasattr(vp.pair_config, "model_dump")
                else vp.pair_config.dict(exclude_none=True)
            )
            entry = {
                "config": pair_payload,
                "metadata": vp.metadata,
            }

            existing_idx = index.get(symbol)
            if existing_idx is not None:
                existing_metadata = pairs[existing_idx].get("metadata", {})
                last_discovered = existing_metadata.get("discovered_at")
                if last_discovered:
                    try:
                        last_dt = dt.datetime.fromisoformat(str(last_discovered))
                    except ValueError:
                        last_dt = None
                    if last_dt and now - last_dt < cooldown:
                        logger.debug(f"{symbol} is within its cooldown window; skipping update.")
                        continue
                pairs[existing_idx] = entry
            else:
                pairs.append(entry)
                index[symbol] = len(pairs) - 1
            updated = True

        if updated:
            current_data["pairs"] = pairs
            self.store.save(current_data)
            logger.info(f"Updated the dynamic pair file with {len(validated_pairs)} candidates.")
        else:
            logger.info("No new dynamic pairs to write.")

    @staticmethod
    def _extract_symbol(entry: Dict[str, object]) -> Optional[str]:
        if not isinstance(entry, dict):
            return None
        config = entry.get("config") if isinstance(entry.get("config"), dict) else entry
        return config.get("cex_symbol") if isinstance(config, dict) else None
