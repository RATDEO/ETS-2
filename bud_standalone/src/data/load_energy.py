"""
Load energy benchmark data (Brent crude, Rotterdam coal).

Both series are USD-denominated and need FX conversion.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Union, Tuple
import logging

logger = logging.getLogger(__name__)


def load_brent_crude(file_path: Union[str, Path]) -> pd.DataFrame:
    """
    Load EU Brent Spot Price data.
    
    File format:
    - Header rows with metadata
    - Date format: "May 20, 1987"
    - Price in USD per barrel
    
    Args:
        file_path: Path to eu-brent-spot-usd.csv
        
    Returns:
        DataFrame with date, price_usd, source, currency
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"Brent file not found: {file_path}")
    
    # Read the file, skip metadata rows
    df = pd.read_csv(file_path, skiprows=2, header=None, names=["date", "price_usd"])
    
    # Parse dates (format: "May 20, 1987")
    df["date"] = pd.to_datetime(df["date"], format="%b %d, %Y", errors="coerce")
    
    # Drop invalid rows
    invalid = df["date"].isna().sum()
    if invalid > 0:
        logger.warning(f"Brent: Dropped {invalid} rows with invalid dates")
        df = df.dropna(subset=["date"])
    
    # Parse prices
    df["price_usd"] = pd.to_numeric(df["price_usd"], errors="coerce")
    
    # Add metadata
    df["series_name"] = "BRENT_SPOT"
    df["source"] = "eia"
    df["currency"] = "USD"
    
    # Sort and deduplicate
    df = df.sort_values("date").drop_duplicates(subset=["date"], keep="last")
    df = df.reset_index(drop=True)
    
    logger.info(f"Loaded Brent crude: {len(df)} records from {df['date'].min()} to {df['date'].max()}")
    
    return df


def load_rotterdam_coal(file_path: Union[str, Path]) -> pd.DataFrame:
    """
    Load Rotterdam Coal Futures data.
    
    File format:
    - Date format: "DD/MM/YYYY"
    - Price in USD
    - Volume with K suffix
    - Data is in reverse chronological order
    
    Args:
        file_path: Path to rotterdam-coal-futures-usd.csv
        
    Returns:
        DataFrame with date, price_usd, volume, source, currency
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"Coal file not found: {file_path}")
    
    df = pd.read_csv(file_path)
    
    # Rename columns
    df = df.rename(columns={
        "Date": "date",
        "Price": "price_usd",
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Vol.": "volume",
        "Change %": "change_pct"
    })
    
    # Parse dates (format: "DD/MM/YYYY")
    df["date"] = pd.to_datetime(df["date"], format="%d/%m/%Y", errors="coerce")
    
    # Drop invalid dates
    invalid = df["date"].isna().sum()
    if invalid > 0:
        logger.warning(f"Coal: Dropped {invalid} rows with invalid dates")
        df = df.dropna(subset=["date"])
    
    # Parse price
    df["price_usd"] = pd.to_numeric(df["price_usd"], errors="coerce")
    
    # Parse volume (remove 'K' suffix and convert)
    if "volume" in df.columns:
        df["volume"] = df["volume"].astype(str).str.replace("K", "").str.replace(",", "")
        df["volume"] = pd.to_numeric(df["volume"], errors="coerce") * 1000
    
    # Add metadata
    df["series_name"] = "COAL_FUTURES"
    df["source"] = "investing.com"
    df["currency"] = "USD"
    
    # Sort chronologically and deduplicate
    df = df.sort_values("date").drop_duplicates(subset=["date"], keep="last")
    df = df.reset_index(drop=True)
    
    logger.info(f"Loaded Rotterdam coal: {len(df)} records from {df['date'].min()} to {df['date'].max()}")
    
    return df


def load_energy_benchmarks(data_dir: Union[str, Path]) -> Dict[str, pd.DataFrame]:
    """
    Load all energy benchmark data.
    
    Args:
        data_dir: Path to Data/ directory
        
    Returns:
        Dictionary with 'brent' and 'coal' DataFrames
    """
    data_dir = Path(data_dir)
    energy_dir = data_dir / "energy-benchmarks"
    
    if not energy_dir.exists():
        raise FileNotFoundError(f"Energy benchmarks directory not found: {energy_dir}")
    
    result = {}
    
    # Load Brent
    brent_file = energy_dir / "eu-brent-spot-usd.csv"
    if brent_file.exists():
        try:
            result["brent"] = load_brent_crude(brent_file)
        except Exception as e:
            logger.error(f"Error loading Brent data: {e}")
    
    # Load Coal
    coal_file = energy_dir / "rotterdam-coal-futures-usd.csv"
    if coal_file.exists():
        try:
            result["coal"] = load_rotterdam_coal(coal_file)
        except Exception as e:
            logger.error(f"Error loading coal data: {e}")
    
    return result


def create_energy_features(
    energy_dict: Dict[str, pd.DataFrame],
    calendar: pd.DatetimeIndex,
    compute_returns: bool = True,
    rolling_windows: list = [5, 20]
) -> pd.DataFrame:
    """
    Create energy-based features aligned to a calendar.
    
    Args:
        energy_dict: Dictionary from load_energy_benchmarks()
        calendar: Target date index
        compute_returns: Whether to compute log returns
        rolling_windows: Windows for rolling statistics
        
    Returns:
        DataFrame with energy features
    """
    result = pd.DataFrame({"date": calendar})
    
    for name, df in energy_dict.items():
        if df.empty:
            continue
            
        # Get price column (may be price_usd or price_eur after conversion)
        price_col = "price_eur" if "price_eur" in df.columns else "price_usd"
        
        # Merge price
        temp = df[["date", price_col]].copy()
        temp = temp.rename(columns={price_col: f"{name}_price"})
        result = result.merge(temp, on="date", how="left")
        
        # Forward-fill up to 5 days
        result[f"{name}_price"] = result[f"{name}_price"].ffill(limit=5)
        
        if compute_returns:
            # Log returns
            result[f"{name}_return"] = np.log(
                result[f"{name}_price"] / result[f"{name}_price"].shift(1)
            )
        
        # Rolling volatility
        for window in rolling_windows:
            if compute_returns:
                result[f"{name}_vol_{window}d"] = (
                    result[f"{name}_return"].rolling(window=window, min_periods=1).std()
                )
    
    # Create spread features if both are available
    if "brent_price" in result.columns and "coal_price" in result.columns:
        result["coal_brent_ratio"] = result["coal_price"] / result["brent_price"]
    
    return result


def standardize_energy_output(df: pd.DataFrame, name: str) -> pd.DataFrame:
    """
    Convert energy DataFrame to standardized format.
    """
    price_col = "price_eur" if "price_eur" in df.columns else "price_usd"
    currency = "EUR" if "price_eur" in df.columns else "USD"
    
    return pd.DataFrame({
        "date": df["date"],
        "value": df[price_col],
        "series_name": f"ENERGY_{name.upper()}",
        "currency": currency,
        "source": df["source"].iloc[0] if "source" in df.columns else "energy"
    })
