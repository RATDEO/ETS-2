"""
Load daily carbon futures price data from Investing-style CSV exports.

The module name is legacy, but the loader is used for both EUA and UKA futures
files because the upstream schema is the same.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Union, Tuple
import logging
import re

logger = logging.getLogger(__name__)


def infer_futures_metadata(file_path: Union[str, Path]) -> dict[str, str]:
    """Infer instrument/source/currency metadata from the futures filename."""
    name = Path(file_path).name.lower()
    if "uk emissions allowances" in name or "(uka)" in name or "uka" in name:
        return {
            "instrument": "UKA_FUTURES",
            "source": "uka_futures",
            "currency": "GBP",
            "label": "UKA futures",
        }
    return {
        "instrument": "EUA_FUTURES",
        "source": "eua_futures",
        "currency": "EUR",
        "label": "EUA futures",
    }


def parse_volume(vol_str: str) -> Optional[float]:
    """
    Parse volume string with K/M suffixes.
    
    Args:
        vol_str: Volume string like "27.80K" or "1.5M"
        
    Returns:
        Volume as float or None if unparseable
    """
    if pd.isna(vol_str) or vol_str == "-" or vol_str == "":
        return None
    
    vol_str = str(vol_str).strip().upper()
    
    try:
        if vol_str.endswith("K"):
            return float(vol_str[:-1]) * 1_000
        elif vol_str.endswith("M"):
            return float(vol_str[:-1]) * 1_000_000
        elif vol_str.endswith("B"):
            return float(vol_str[:-1]) * 1_000_000_000
        else:
            return float(vol_str.replace(",", ""))
    except (ValueError, TypeError):
        return None


def parse_change_pct(change_str: str) -> Optional[float]:
    """
    Parse percentage change string.
    
    Args:
        change_str: Change string like "-1.20%" or "+0.50%"
        
    Returns:
        Change as decimal (e.g., -0.012) or None
    """
    if pd.isna(change_str) or change_str == "-" or change_str == "":
        return None
    
    change_str = str(change_str).strip()
    
    try:
        # Remove % sign and convert
        value = float(change_str.replace("%", "").replace(",", ""))
        return value / 100.0
    except (ValueError, TypeError):
        return None


def load_eua_futures_file(file_path: Union[str, Path]) -> pd.DataFrame:
    """
    Load a single carbon futures price data file.
    
    Expected CSV format:
    "Date","Price","Open","High","Low","Vol.","Change %"
    "05/01/2026","87.25","88.09","88.78","87.05","27.80K","-1.20%"
    
    Args:
        file_path: Path to the futures CSV file
        
    Returns:
        DataFrame with parsed futures data
    """
    file_path = Path(file_path)
    meta = infer_futures_metadata(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"Futures file not found: {file_path}")
    
    logger.info("Loading %s file: %s", meta["label"], file_path.name)
    
    # Read CSV - handle quoted strings
    df = pd.read_csv(file_path, encoding='utf-8')
    
    # Clean column names (remove quotes and whitespace)
    df.columns = df.columns.str.strip().str.strip('"')
    
    # Standardize column names
    rename_map = {
        "Date": "date",
        "Price": "close",
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Vol.": "volume_raw",
        "Change %": "change_pct_raw"
    }
    
    df = df.rename(columns=rename_map)
    
    # Parse dates (format: DD/MM/YYYY)
    df["date"] = pd.to_datetime(df["date"], format="%d/%m/%Y", errors="coerce")
    
    # Drop rows with invalid dates
    invalid_dates = df["date"].isna().sum()
    if invalid_dates > 0:
        logger.warning(f"Dropped {invalid_dates} rows with invalid dates")
        df = df.dropna(subset=["date"])
    
    # Parse numeric price columns (handle commas in numbers)
    for col in ["close", "open", "high", "low"]:
        if col in df.columns:
            # Remove commas and convert
            df[col] = df[col].astype(str).str.replace(",", "").astype(float)
    
    # Parse volume
    if "volume_raw" in df.columns:
        df["volume"] = df["volume_raw"].apply(parse_volume)
        df = df.drop(columns=["volume_raw"])
    
    # Parse change percentage
    if "change_pct_raw" in df.columns:
        df["change_pct"] = df["change_pct_raw"].apply(parse_change_pct)
        df = df.drop(columns=["change_pct_raw"])
    
    # Add metadata
    df["currency"] = meta["currency"]
    df["source"] = meta["source"]
    df["instrument"] = meta["instrument"]
    
    # Sort by date (ascending - oldest first)
    df = df.sort_values("date", ascending=True).reset_index(drop=True)
    
    logger.info(
        "Loaded %s: %d records from %s to %s",
        meta["label"],
        len(df),
        df["date"].min(),
        df["date"].max(),
    )
    
    return df


def load_eua_futures_data(data_dir: Union[str, Path]) -> pd.DataFrame:
    """
    Load and combine all carbon futures price data files.
    
    Args:
        data_dir: Path to Data/ directory
        
    Returns:
        Combined DataFrame with futures prices
    """
    data_dir = Path(data_dir)
    futures_dir = data_dir / "eua-futures"
    
    if not futures_dir.exists():
        raise FileNotFoundError(f"Futures directory not found: {futures_dir}")
    
    # Find all CSV files in the directory
    csv_files = list(futures_dir.glob("*.csv"))
    
    if not csv_files:
        raise FileNotFoundError(f"No futures CSV files found in {futures_dir}")
    
    dfs = []
    for file_path in csv_files:
        try:
            df = load_eua_futures_file(file_path)
            if not df.empty:
                dfs.append(df)
        except Exception as e:
            logger.error(f"Error loading {file_path.name}: {e}")
    
    if not dfs:
        raise ValueError("No futures data could be loaded")
    
    # Combine and deduplicate
    combined = pd.concat(dfs, ignore_index=True)
    combined = combined.sort_values("date").drop_duplicates(subset=["date"], keep="last")
    combined = combined.reset_index(drop=True)
    
    instrument = str(combined["instrument"].iloc[0]) if "instrument" in combined.columns and not combined.empty else "FUTURES"
    logger.info(
        "Combined %s data: %d records from %s to %s",
        instrument,
        len(combined),
        combined["date"].min(),
        combined["date"].max(),
    )
    
    return combined


def get_eua_futures_target_series(
    eua_df: pd.DataFrame,
    price_col: str = "close"
) -> pd.DataFrame:
    """
    Extract the primary target price series from a futures DataFrame.
    
    Args:
        eua_df: DataFrame from load_eua_futures_data()
        price_col: Which price column to use (close, open, high, low)
        
    Returns:
        DataFrame with date and close_eur columns
    """
    if eua_df.empty:
        return pd.DataFrame(columns=["date", "close_eur"])
    
    if price_col not in eua_df.columns:
        raise ValueError(f"Price column '{price_col}' not found in futures data")
    
    # Extract target series
    target = eua_df[["date", price_col]].copy()
    target = target.rename(columns={price_col: "close_eur"})
    
    # Drop NA prices
    target = target.dropna(subset=["close_eur"])
    
    # Sort by date
    target = target.sort_values("date").reset_index(drop=True)
    
    instrument = str(eua_df["instrument"].iloc[0]) if "instrument" in eua_df.columns else "FUTURES"
    currency = str(eua_df["currency"].iloc[0]) if "currency" in eua_df.columns else "native"
    logger.info(
        "%s target series: %d records, price range: %.2f - %.2f %s",
        instrument,
        len(target),
        target["close_eur"].min(),
        target["close_eur"].max(),
        currency,
    )
    
    return target


def create_eua_futures_features(
    eua_df: pd.DataFrame,
    calendar: pd.DatetimeIndex,
    compute_returns: bool = True,
    compute_range: bool = True
) -> pd.DataFrame:
    """
    Create additional features from EUA futures data.
    
    Args:
        eua_df: DataFrame from load_eua_futures_data()
        calendar: Master calendar dates to align to
        compute_returns: Whether to compute log returns
        compute_range: Whether to compute high-low range
        
    Returns:
        DataFrame with features aligned to calendar
    """
    if eua_df.empty:
        return pd.DataFrame({"date": calendar})
    
    features = pd.DataFrame({"date": calendar})
    
    # Join EUA data
    eua_aligned = eua_df.copy()
    eua_aligned = eua_aligned[["date", "close", "high", "low", "volume"]].copy()
    
    features = features.merge(eua_aligned, on="date", how="left")
    
    # Forward-fill missing values (up to 5 days)
    for col in ["close", "high", "low"]:
        if col in features.columns:
            features[col] = features[col].ffill(limit=5)
    
    # Volume: fill with 0 for missing days
    if "volume" in features.columns:
        features["volume"] = features["volume"].fillna(0)
    
    # Rename to avoid confusion with target
    features = features.rename(columns={
        "close": "eua_close",
        "high": "eua_high",
        "low": "eua_low",
        "volume": "eua_volume"
    })
    
    # Compute returns
    if compute_returns and "eua_close" in features.columns:
        features["eua_return"] = np.log(
            features["eua_close"] / features["eua_close"].shift(1)
        )
    
    # Compute high-low range (volatility proxy)
    if compute_range and "eua_high" in features.columns and "eua_low" in features.columns:
        features["eua_range"] = features["eua_high"] - features["eua_low"]
        features["eua_range_pct"] = (
            (features["eua_high"] - features["eua_low"]) / features["eua_close"]
        )
    
    return features


def get_data_summary(eua_df: pd.DataFrame) -> dict:
    """
    Get summary statistics for EUA futures data.
    
    Args:
        eua_df: DataFrame from load_eua_futures_data()
        
    Returns:
        Dictionary with summary statistics
    """
    if eua_df.empty:
        return {"error": "No data available"}
    
    return {
        "n_records": len(eua_df),
        "date_range": {
            "start": str(eua_df["date"].min()),
            "end": str(eua_df["date"].max())
        },
        "price_stats": {
            "min": float(eua_df["close"].min()),
            "max": float(eua_df["close"].max()),
            "mean": float(eua_df["close"].mean()),
            "std": float(eua_df["close"].std()),
            "current": float(eua_df.iloc[-1]["close"])
        },
        "volume_stats": {
            "mean": float(eua_df["volume"].mean()) if "volume" in eua_df.columns else None,
            "max": float(eua_df["volume"].max()) if "volume" in eua_df.columns else None
        },
        "currency": "EUR",
        "source": "EUA Yearly Futures"
    }
