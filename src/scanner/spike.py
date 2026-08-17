"""Binance volume spike scanner and DEX-CEX arbitrage evaluator."""
from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import aiohttp
from loguru import logger

from src.core.config import AppConfig, VolumeSpikeConfig
from src.core.types import MarketPair
from src.exchange.univ3 import UniV3DexClient
from src.strategy.costs import evaluate_trade


@dataclass
class SymbolSnapshot:
    symbol: str
    base: str
    quote: str
    status: str
    base_precision: int
    quote_precision: int

    @staticmethod
    def precision_from_step(step: str) -> int:
        step = step.strip()
        if not step or step in {"1", "1.0", "0"}:
            return 0
        try:
            normalized = f"{Decimal(step):f}"
        except Exception:
            normalized = step
        if "." not in normalized:
            return 0
        fractional = normalized.rstrip("0").split(".")[-1]
        return len(fractional)

    @classmethod
    def from_exchange(cls, payload: Dict[str, object]) -> "SymbolSnapshot":
        filters = {f["filterType"]: f for f in payload.get("filters", [])}
        lot_step = filters.get("LOT_SIZE", {}).get("stepSize", "1")
        price_step = filters.get("PRICE_FILTER", {}).get("tickSize", "1")
        base_precision = cls.precision_from_step(lot_step)
        quote_precision = cls.precision_from_step(price_step)
        return cls(
            symbol=payload.get("symbol", ""),
            base=payload.get("baseAsset", ""),
            quote=payload.get("quoteAsset", ""),
            status=payload.get("status", ""),
            base_precision=base_precision,
            quote_precision=quote_precision,
        )


@dataclass
class VolumeSpike:
    symbol: str
    base: str
    quote: str
    current_volume: float
    previous_volume: float
    ratio: float
    closed_at: datetime
    base_precision: int
    quote_precision: int

    def to_dict(self) -> Dict[str, object]:
        return {
            "symbol": self.symbol,
            "base": self.base,
            "quote": self.quote,
            "current_volume": self.current_volume,
            "previous_volume": self.previous_volume,
            "ratio": self.ratio,
            "closed_at": self.closed_at.isoformat(),
            "base_precision": self.base_precision,
            "quote_precision": self.quote_precision,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, object]) -> "VolumeSpike":
        return cls(
            symbol=str(data["symbol"]),
            base=str(data["base"]),
            quote=str(data["quote"]),
            current_volume=float(data["current_volume"]),
            previous_volume=float(data["previous_volume"]),
            ratio=float(data["ratio"]),
            closed_at=datetime.fromisoformat(str(data["closed_at"])),
            base_precision=int(data.get("base_precision", 8)),
            quote_precision=int(data.get("quote_precision", 8)),
        )


class VolumeSpikeStore:
    def __init__(self, path: Path):
        self.path = path

    def load(self) -> List[VolumeSpike]:
        if not self.path.exists():
            return []
        with self.path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        return [VolumeSpike.from_dict(item) for item in payload.get("spikes", [])]

    def save(self, spikes: List[VolumeSpike]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"updated_at": datetime.now(timezone.utc).isoformat(), "spikes": [s.to_dict() for s in spikes]}
        with self.path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)


class BinancePublicClient:
    def __init__(self, base_url: str, session: aiohttp.ClientSession, concurrency: int = 20):
        self.base_url = base_url.rstrip("/")
        self.session = session
        self.semaphore = asyncio.Semaphore(concurrency)

    async def fetch_exchange_info(self) -> List[SymbolSnapshot]:
        url = f"{self.base_url}/api/v3/exchangeInfo"
        async with self.session.get(url, timeout=10) as response:
            response.raise_for_status()
            data = await response.json()
        snapshots = [SymbolSnapshot.from_exchange(item) for item in data.get("symbols", [])]
        return snapshots

    async def fetch_hourly_klines(self, symbol: str, limit: int = 2) -> List[List[str]]:
        params = {"symbol": symbol, "interval": "1h", "limit": limit}
        async with self.semaphore:
            async with self.session.get(f"{self.base_url}/api/v3/klines", params=params, timeout=10) as response:
                if response.status != 200:
                    text = await response.text()
                    logger.warning(f"Failed to fetch klines for {symbol} [{response.status}]: {text}")
                    return []
                return await response.json()

    async def fetch_ticker_price(self, symbol: str) -> Optional[float]:
        params = {"symbol": symbol}
        async with self.session.get(f"{self.base_url}/api/v3/ticker/price", params=params, timeout=10) as response:
            if response.status != 200:
                text = await response.text()
                logger.warning(f"Failed to fetch the latest price for {symbol} [{response.status}]: {text}")
                return None
            data = await response.json()
            return float(data.get("price"))


class VolumeSpikeScanner:
    def __init__(self, config: AppConfig):
        self.config = config
        self.spike_cfg: VolumeSpikeConfig = config.scanner.spike
        self.store = VolumeSpikeStore(Path(self.spike_cfg.persist_path))

    async def scan(self) -> List[VolumeSpike]:
        if not self.spike_cfg.enabled:
            logger.info("Volume spike scanner is disabled.")
            return []

        async with aiohttp.ClientSession() as session:
            client = BinancePublicClient(self.config.cex.base_url, session, self.spike_cfg.concurrency)
            snapshots = await client.fetch_exchange_info()
            usdt_symbols = [s for s in snapshots if s.quote == self.spike_cfg.quote_asset and s.status == "TRADING"]
            tasks = [self._fetch_and_compute(client, snap) for snap in usdt_symbols]
            results = await asyncio.gather(*tasks, return_exceptions=True)

        spikes: List[VolumeSpike] = []
        for result in results:
            if isinstance(result, Exception):
                logger.warning(f"Error computing a volume spike: {result}")
                continue
            if result:
                spikes.append(result)

        spikes.sort(key=lambda s: s.ratio, reverse=True)
        top_spikes = spikes[: self.spike_cfg.max_candidates]
        self.store.save(top_spikes)
        logger.info(f"Volume spike scan complete; recorded {len(top_spikes)} candidates.")
        return top_spikes

    async def _fetch_and_compute(self, client: BinancePublicClient, snapshot: SymbolSnapshot) -> Optional[VolumeSpike]:
        klines = await client.fetch_hourly_klines(snapshot.symbol, limit=self.spike_cfg.lookback_hours)
        if len(klines) < 2:
            return None

        current = klines[-1]
        previous = klines[-2]
        try:
            current_volume = float(current[5])
            previous_volume = float(previous[5])
        except (TypeError, ValueError):
            return None
        if previous_volume <= 0:
            return None

        ratio = current_volume / previous_volume if previous_volume else 0
        if ratio <= self.spike_cfg.ratio_threshold:
            return None

        closed_at = datetime.fromtimestamp(int(current[6]) / 1000, tz=timezone.utc)
        return VolumeSpike(
            symbol=snapshot.symbol,
            base=snapshot.base,
            quote=snapshot.quote,
            current_volume=current_volume,
            previous_volume=previous_volume,
            ratio=ratio,
            closed_at=closed_at,
            base_precision=snapshot.base_precision,
            quote_precision=snapshot.quote_precision,
        )

    def load_spikes(self) -> List[VolumeSpike]:
        return self.store.load()


@dataclass
class ArbitrageSignal:
    """One screen hit.

    `gross_bps` is the raw venue-to-venue spread; `net_bps` is what remains after
    the taker fee and gas, computed by `costs.evaluate_trade` -- the same function
    the detector uses, so the two numbers are comparable. The previous version
    reported the raw spread minus a `cost_buffer_bps` fudge, which was a second
    and wrong cost model living beside the real one.

    `depth_aware` is always False and is recorded rather than assumed: the screen
    prices from a single ticker and a single probe size, so it cannot see the
    order book and therefore overstates achievable edge on any real size. A
    screen hit is a reason to look, not a tradeable opportunity.
    """

    symbol: str
    direction: str
    gross_bps: float
    net_bps: float
    cex_price: float
    dex_price: float
    dex_chain: str
    dex_fee_tier: int
    probe_size_base: float
    gas_quote: float
    taker_fee_bps: float
    depth_aware: bool = False

    def as_dict(self) -> Dict[str, object]:
        return {
            "symbol": self.symbol,
            "direction": self.direction,
            "gross_bps": round(self.gross_bps, 2),
            "net_bps": round(self.net_bps, 2),
            "cex_price": self.cex_price,
            "dex_price": float(self.dex_price),
            "dex_chain": self.dex_chain,
            "dex_fee_tier": self.dex_fee_tier,
            "probe_size_base": self.probe_size_base,
            "gas_quote": self.gas_quote,
            "taker_fee_bps": self.taker_fee_bps,
            "depth_aware": self.depth_aware,
        }


class SpikeArbitrageEvaluator:
    def __init__(self, config: AppConfig):
        self.config = config
        self.spike_cfg = config.scanner.spike
        self.store = VolumeSpikeStore(Path(self.spike_cfg.persist_path))
        # Built once: a validation error belongs at construction, not in a loop
        # over candidates where the only safe response is to keep going.
        self._token_policy = config.strategy.token_policy.build()

        # The screen reports against the SAME floor the strategy trades at unless
        # told otherwise, so a hit means "the detector would consider this"
        # rather than "this cleared a threshold nothing else in the system uses".
        configured = self.spike_cfg.arbitrage.edge_threshold_bps
        self._floor_bps = (
            config.strategy.min_net_bps if configured is None
            else Decimal(str(configured))
        )
        if self._floor_bps > config.strategy.min_net_bps:
            logger.warning(
                f"The spike screen floor ({self._floor_bps} bps) is above the "
                f"strategy's own floor ({config.strategy.min_net_bps} bps), so "
                f"the screen will hide candidates the strategy would take."
            )

    async def evaluate(self, spikes: Optional[List[VolumeSpike]] = None) -> List[ArbitrageSignal]:
        if spikes is None:
            spikes = self.store.load()
        if not spikes:
            logger.info("No volume spike candidates to evaluate.")
            return []

        async with aiohttp.ClientSession() as session:
            client = BinancePublicClient(self.config.cex.base_url, session)
            dex_client = UniV3DexClient(self.config.dex, self.config.network, self.config.secrets, self.config.tokens)
            tasks = [self._evaluate_symbol(client, dex_client, spike) for spike in spikes]
            results = await asyncio.gather(*tasks, return_exceptions=True)

        signals: List[ArbitrageSignal] = []
        for result in results:
            if isinstance(result, Exception):
                logger.warning(f"Error evaluating arbitrage: {result}")
                continue
            if result:
                signals.append(result)

        if signals:
            logger.success(f"Found {len(signals)} potential arbitrage signals.")
        else:
            logger.info("No arbitrage signals currently meet the threshold.")
        return signals

    async def _evaluate_symbol(
        self,
        client: BinancePublicClient,
        dex_client: UniV3DexClient,
        spike: VolumeSpike,
    ) -> Optional[ArbitrageSignal]:
        # The token gate first: a screen exists to direct human attention, and
        # attention spent on a token that can never be traded is wasted. It also
        # saves a ticker request and a pool lookup per candidate.
        verdict = self._token_policy.check(spike.base, spike.quote)
        if not verdict.allowed:
            logger.debug(f"Skipping {spike.symbol}: {verdict.reason}")
            return None

        cex_price = await client.fetch_ticker_price(spike.symbol)
        if not cex_price:
            return None

        arbitrage_cfg = self.spike_cfg.arbitrage
        probe_size = Decimal(str(arbitrage_cfg.probe_size_base))

        dex_price: Optional[Decimal] = None
        gas_quote: Decimal = Decimal("0")
        selected_chain: Optional[str] = None
        selected_fee: Optional[int] = None

        for chain in arbitrage_cfg.dex_chains:
            if spike.base not in dex_client.tokens_config or chain not in dex_client.tokens_config[spike.base]:
                continue
            if spike.quote not in dex_client.tokens_config or chain not in dex_client.tokens_config[spike.quote]:
                continue
            for fee in arbitrage_cfg.dex_fee_tiers:
                pool_addr = await dex_client.get_pool_address(spike.base, spike.quote, chain, fee)
                if not pool_addr:
                    continue
                # MarketPair's fields are quote_cex/quote_dex/cex_symbol. This
                # previously passed `quote=` and `symbol=`, so pydantic raised on
                # the first pool the screen found and the entire scan died -- a
                # failure indistinguishable from "no opportunities".
                pair = MarketPair(
                    base=spike.base,
                    quote_cex=spike.quote,
                    quote_dex=spike.quote,
                    cex_symbol=spike.symbol,
                    dex_chain=chain,
                    dex_pool_fee=fee,
                    base_precision=spike.base_precision,
                    quote_precision=spike.quote_precision,
                )
                # estimate_gas=True because gas is a real cost of the trade and
                # the screen's whole purpose is to say whether the spread covers
                # its costs.
                quote = await dex_client.get_quote(
                    pair, probe_size, side="sell", estimate_gas=True
                )
                if quote is None or quote.price <= 0:
                    continue
                dex_price = quote.price
                gas_quote = quote.gas_cost_quote
                selected_chain = chain
                selected_fee = fee
                break
            if dex_price is not None:
                break

        if dex_price is None or selected_chain is None or selected_fee is None:
            logger.debug(f"No valid DEX quote for {spike.symbol} on any available chain.")
            return None

        cex_price_dec = Decimal(str(cex_price))
        spread = cex_price_dec - dex_price
        if spread == 0:
            return None

        # Sell wherever the price is higher.
        direction = "DEX_to_CEX" if spread > 0 else "CEX_to_DEX"

        # Priced through the shared cost model rather than the raw spread minus a
        # buffer. `cost_buffer_bps` was a fudge standing in for the taker fee and
        # gas, which are both known exactly -- and a screen whose arithmetic
        # disagrees with the detector's is worse than no screen, because someone
        # will eventually trust it over the real model.
        econ = evaluate_trade(
            direction=direction,
            size_base=probe_size,
            cex_price=cex_price_dec,
            dex_price=dex_price,
            taker_fee_bps=self.config.strategy.taker_fee_bps,
            gas_quote=gas_quote,
        )
        if econ is None:
            return None

        gross_bps = abs(spread) / min(cex_price_dec, dex_price) * Decimal("10000")

        if econ.net_bps <= self._floor_bps:
            return None

        signal = ArbitrageSignal(
            symbol=spike.symbol,
            direction=direction,
            gross_bps=float(gross_bps),
            net_bps=float(econ.net_bps),
            cex_price=cex_price,
            dex_price=float(dex_price),
            dex_chain=selected_chain,
            dex_fee_tier=selected_fee,
            probe_size_base=float(probe_size),
            gas_quote=float(gas_quote),
            taker_fee_bps=float(self.config.strategy.taker_fee_bps),
        )
        logger.debug(f"Arbitrage signal: {signal.as_dict()}")
        return signal
