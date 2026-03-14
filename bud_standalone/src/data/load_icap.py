"""
Load ICAP Allowance Price Explorer data (Secondary Market).

This is the primary EUA price series (secondary market trading).
Two files cover different periods: 2005-2018 and 2019-2025.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Union
import logging

logger = logging.getLogger(__name__)


def load_icap_file(file_path: Union[str, Path]) -> pd.DataFrame:
    """
    Load a single ICAP price data file.
    
    File format:
    - Row 1: Title
    - Row 2: Column headers
    - Columns: Date, Exchange rate EUR/EUR, Exchange rate EUR/USD, 
               Market Currency, Primary Market, Secondary Market
    
    Args:
        file_path: Path to ICAP CSV file
        
    Returns:
        DataFrame with parsed ICAP data
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"ICAP file not found: {file_path}")
    
    # Read with header on row 2 (0-indexed: skiprows=1)
    df = pd.read_csv(file_path, skiprows=1, encoding='utf-8')
    
    # Clean column names
    df.columns = df.columns.str.strip()
    
    # Rename columns
    rename_map = {
        "Date": "date",
        "Exchange rate EUR/EUR": "fx_eur_eur",
        "Exchange rate EUR/USD": "fx_eur_usd",
        "Market Currency": "currency",
        "Primary Market": "primary_market",
        "Secondary Market": "secondary_market"
    }
    
    df = df.rename(columns=rename_map)
    
    # Keep only relevant columns
    keep_cols = [c for c in ["date", "primary_market", "secondary_market", 
                             "fx_eur_usd", "currency"] if c in df.columns]
    df = df[keep_cols].copy()
    
    # Parse dates (format: YYYY-MM-DD)
    df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d", errors="coerce")
    
    # Drop invalid rows
    invalid = df["date"].isna().sum()
    if invalid > 0:
        logger.warning(f"ICAP {file_path.name}: Dropped {invalid} rows with invalid dates")
        df = df.dropna(subset=["date"])
    
    # Parse numeric columns
    for col in ["primary_market", "secondary_market", "fx_eur_usd"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    
    # Add source info
    df["source"] = "icap"
    
    # Sort by date
    df = df.sort_values("date").reset_index(drop=True)
    
    logger.info(f"Loaded ICAP file: {len(df)} records from {df['date'].min()} to {df['date'].max()}")
    
    return df


def load_icap_data(data_dir: Union[str, Path]) -> pd.DataFrame:
    """
    Load and combine all ICAP price data files.
    
    Args:
        data_dir: Path to Data/ directory
        
    Returns:
        Combined DataFrame with ICAP secondary market prices
    """
    data_dir = Path(data_dir)
    icap_dir = data_dir / "icap-allowance-price-explorer-secondary-market"
    
    if not icap_dir.exists():
        raise FileNotFoundError(f"ICAP directory not found: {icap_dir}")
    
    # Find ICAP CSV files (exclude images)
    icap_files = sorted(icap_dir.glob("icap-graph-price-data-*.csv"))
    
    if not icap_files:
        logger.warning(f"No ICAP files found in {icap_dir}")
        return pd.DataFrame()
    
    dfs = []
    for file_path in icap_files:
        try:
            df = load_icap_file(file_path)
            if not df.empty:
                dfs.append(df)
        except Exception as e:
            logger.error(f"Error loading {file_path.name}: {e}")
    
    if not dfs:
        return pd.DataFrame()
    
    # Combine and deduplicate
    combined = pd.concat(dfs, ignore_index=True)
    combined = combined.sort_values("date").drop_duplicates(subset=["date"], keep="last")
    combined = combined.reset_index(drop=True)
    
    logger.info(f"Combined ICAP data: {len(combined)} records from {combined['date'].min()} to {combined['date'].max()}")
    
    return combined


def get_icap_target_series(
    icap_df: pd.DataFrame,
    prefer_secondary: bool = True
) -> pd.DataFrame:
    """
    Extract the primary target price series from ICAP data.
    
    The secondary market is preferred as it represents active trading.
    Falls back to primary market when secondary is unavailable.
    
    Args:
        icap_df: DataFrame from load_icap_data()
        prefer_secondary: Whether to prefer secondary market prices
        
    Returns:
        DataFrame with date and close_eur columns
    """
    if icap_df.empty:
        return pd.DataFrame(columns=["date", "close_eur"])
    
    result = icap_df[["date"]].copy()
    
    if prefer_secondary and "secondary_market" in icap_df.columns:
        # Use secondary market, fall back to primary
        result["close_eur"] = icap_df["secondary_market"].combine_first(
            icap_df.get("primary_market", pd.Series(dtype=float))
        )
    elif "primary_market" in icap_df.columns:
        result["close_eur"] = icap_df["primary_market"]
    else:
        logger.warning("No price columns found in ICAP data")
        return pd.DataFrame(columns=["date", "close_eur"])
    
    # Drop rows without prices
    result = result.dropna(subset=["close_eur"])
    
    logger.info(f"ICAP target series: {len(result)} daily prices")
    
    return result


def create_icap_features(
    icap_df: pd.DataFrame,
    calendar: pd.DatetimeIndex,
    compute_returns: bool = True,
    rolling_windows: list = [5, 20]
) -> pd.DataFrame:
    """
    Create ICAP-based features aligned to a calendar.
    
    Args:
        icap_df: DataFrame from load_icap_data()
        calendar: Target date index
        compute_returns: Whether to compute returns
        rolling_windows: Windows for rolling statistics
        
    Returns:
        DataFrame with ICAP features
    """
    result = pd.DataFrame({"date": calendar})
    
    if icap_df.empty:
        return result
    
    # Get target series
    target = get_icap_target_series(icap_df)
    
    # Merge
    result = result.merge(target, on="date", how="left")
    
    # Rename for clarity
    result = result.rename(columns={"close_eur": "eua_price"})
    
    # Forward-fill (up to 5 days for non-trading days)
    result["eua_price"] = result["eua_price"].ffill(limit=5)
    
    if compute_returns:
        # Log returns
        result["eua_return"] = np.log(
            result["eua_price"] / result["eua_price"].shift(1)
        )
    
    # Rolling features
    for window in rolling_windows:
        result[f"eua_ma_{window}d"] = (
            result["eua_price"].rolling(window=window, min_periods=1).mean()
        )
        
        if compute_returns:
            result[f"eua_vol_{window}d"] = (
                result["eua_return"].rolling(window=window, min_periods=1).std()
            )
    
    return result


def standardize_icap_output(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert ICAP DataFrame to standardized format.
    """
    target = get_icap_target_series(df)
    
    return pd.DataFrame({
        "date": target["date"],
        "value": target["close_eur"],
        "series_name": "ICAP_EUA_SECONDARY",
        "currency": "EUR",
        "source": "icap"
    })
