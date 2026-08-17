# backtest/datasets.py
from __future__ import annotations
import pandas as pd
from loguru import logger
from decimal import Decimal as D

def load_dataset(path: str) -> pd.DataFrame:
    """
    Load historical market data from a CSV file.

    Supported CSV columns:
    - required: timestamp (ISO-8601 or epoch seconds), cex_bid_price, cex_ask_price, dex_price
    - optional: gas_price_gwei
    """
    logger.info(f"Loading dataset from {path}...")
    try:
        df = pd.read_csv(path)

        # --- 1) validate required columns ---
        required_columns = ["timestamp", "cex_bid_price", "cex_ask_price", "dex_price"]
        missing = [c for c in required_columns if c not in df.columns]
        if missing:
            raise ValueError(f"Dataset is missing required columns: {missing}; required: {required_columns}")

        # --- 2) parse timestamp, accepting both ISO-8601 and epoch seconds ---
        ts = df["timestamp"]
        if pd.api.types.is_numeric_dtype(ts):
            # treat as epoch seconds (UTC)
            parsed = pd.to_datetime(ts, unit="s", utc=True, errors="coerce")
        else:
            # try ISO-8601 first
            parsed = pd.to_datetime(ts, utc=True, errors="coerce")
            # if most rows fail, fall back to parsing the strings as epoch seconds
            if parsed.isna().mean() > 0.5:
                as_num = pd.to_numeric(ts, errors="coerce")
                parsed = pd.to_datetime(as_num, unit="s", utc=True, errors="coerce")

        if parsed.isna().any():
            bad_idx = parsed[parsed.isna()].index[:5].tolist()
            raise ValueError(
                f"Failed to parse timestamp (first unconvertible indices: {bad_idx}). "
                f"Expected ISO-8601 (e.g. 2025-09-18T00:00:00Z) or epoch seconds."
            )

        df["timestamp"] = parsed

        # --- 3) strictly coerce numeric columns ---
        for col in ["cex_bid_price", "cex_ask_price", "dex_price"]:
            df[col] = pd.to_numeric(df[col], errors="raise")
            # important: route through str() so binary float error is not baked in
            df[col] = df[col].astype(str).map(D)

        # gas_price_gwei is optional; coerce it if present, otherwise skip
        if "gas_price_gwei" in df.columns:
            df["gas_price_gwei"] = pd.to_numeric(df["gas_price_gwei"], errors="coerce")
            df["gas_price_gwei"] = df["gas_price_gwei"].astype(str).map(lambda x: D(x) if x != "nan" else None)

        # --- 4) set the index and sort (UTC-aware) ---
        df = df.set_index("timestamp").sort_index()

        logger.success(f"Loaded {len(df)} rows.")
        return df

    except FileNotFoundError:
        logger.error(f"Dataset file not found: {path}")
        raise
    except Exception as e:
        logger.error(f"Error loading the dataset: {e}")
        raise
