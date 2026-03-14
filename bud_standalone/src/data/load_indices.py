"""
Load carbon market indices (KEUA, KRBN, GRN, KCCA, KSET).

These are ETF/index prices tracking carbon markets.
Data format: CSV with columns [Price, Close, High, Low, Open, Volume]
with ticker names in row 2 and dates starting from row 4.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Union
import logging

logger = logging.getLogger(__name__)


def load_single_index(file_path: Union[str, Path]) -> pd.DataFrame:
    """
    Load a single carbon market index file.
    
    Args:
        file_path: Path to the CSV file
        
    Returns:
        DataFrame with columns: date, close, high, low, open, volume, ticker
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"Index file not found: {file_path}")
    
    # Read header to get ticker name
    header_df = pd.read_csv(file_path, nrows=2, header=None)
    ticker = header_df.iloc[1, 1]  # Ticker is in row 2, column 2
    
    # Read the actual data (skip first 3 rows: header, ticker row, "Date" row)
    df = pd.read_csv(file_path, skiprows=3, header=None, 
                     names=["date", "close", "high", "low", "open", "volume"])
    
    # Parse dates
    df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d", errors="coerce")
    
    # Drop rows with invalid dates
    invalid_dates = df["date"].isna().sum()
    if invalid_dates > 0:
        logger.warning(f"{file_path.name}: Dropped {invalid_dates} rows with invalid dates")
        df = df.dropna(subset=["date"])
    
    # Convert numeric columns
    for col in ["close", "high", "low", "open", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    
    # Add metadata
    df["ticker"] = ticker
    df["source"] = "carbon_index"
    df["currency"] = "USD"  # These indices are USD-denominated
    
    # Sort by date and remove duplicates (keep last)
    df = df.sort_values("date").drop_duplicates(subset=["date"], keep="last")
    
    # Reset index
    df = df.reset_index(drop=True)
    
    logger.info(f"Loaded {ticker}: {len(df)} records from {df['date'].min()} to {df['date'].max()}")
    
    return df


def load_carbon_indices(
    data_dir: Union[str, Path],
    indices: Optional[List[str]] = None
) -> Dict[str, pd.DataFrame]:
    """
    Load all carbon market indices from the data directory.
    
    Args:
        data_dir: Path to Data/carbon-market-indices/
        indices: List of index tickers to load. If None, loads all available.
        
    Returns:
        Dictionary mapping ticker -> DataFrame
    """
    data_dir = Path(data_dir)
    indices_dir = data_dir / "carbon-market-indices"
    
    if not indices_dir.exists():
        raise FileNotFoundError(f"Carbon indices directory not found: {indices_dir}")
    
    # Available index files
    available_files = {
        "GRN": "GRN_history.csv",
        "KCCA": "KCCA_history.csv",
        "KEUA": "KEUA_history.csv",
        "KRBN": "KRBN_history.csv",
        "KSET": "KSET_history.csv",
    }
    
    if indices is None:
        indices = list(available_files.keys())
    
    result = {}
    for ticker in indices:
        if ticker not in available_files:
            logger.warning(f"Unknown index ticker: {ticker}")
            continue
            
        file_path = indices_dir / available_files[ticker]
        if file_path.exists():
            try:
                result[ticker] = load_single_index(file_path)
            except Exception as e:
                logger.error(f"Error loading {ticker}: {e}")
        else:
            logger.warning(f"File not found for {ticker}: {file_path}")
    
    logger.info(f"Loaded {len(result)} carbon market indices")
    return result


def get_indices_combined(
    indices_dict: Dict[str, pd.DataFrame],
    value_col: str = "close"
) -> pd.DataFrame:
    """
    Combine multiple index DataFrames into a single wide-format DataFrame.
    
    Args:
        indices_dict: Dictionary from load_carbon_indices()
        value_col: Column to use as value (close, high, low, open)
        
    Returns:
        DataFrame with date as index and each ticker as a column
    """
    dfs = []
    for ticker, df in indices_dict.items():
        temp = df[["date", value_col]].copy()
        temp = temp.rename(columns={value_col: ticker})
        temp = temp.set_index("date")
        dfs.append(temp)
    
    if not dfs:
        return pd.DataFrame()
    
    combined = pd.concat(dfs, axis=1)
    combined = combined.sort_index()
    
    return combined


def standardize_index_output(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """
    Convert index DataFrame to standardized format.
    
    Standardized format:
        date, value, series_name, currency, source
    """
    return pd.DataFrame({
        "date": df["date"],
        "value": df["close"],
        "series_name": f"IDX_{ticker}",
        "currency": df["currency"],
        "source": df["source"]
    })
