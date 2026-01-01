"""
Load VSTOXX volatility index data.

VSTOXX is the Euro STOXX 50 volatility index, used as a market volatility proxy.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Union
import logging

logger = logging.getLogger(__name__)


def load_vstoxx(file_path: Union[str, Path]) -> pd.DataFrame:
    """
    Load VSTOXX index data.
    
    File format (semicolon-separated):
    - Date: DD.MM.YYYY
    - Symbol: V2TX
    - Indexvalue: float
    
    Args:
        file_path: Path to vstoxx-index.txt
        
    Returns:
        DataFrame with date, vstoxx columns
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"VSTOXX file not found: {file_path}")
    
    # Read semicolon-separated file
    df = pd.read_csv(file_path, sep=';')
    
    # Rename columns
    df = df.rename(columns={
        'Date': 'date',
        'Symbol': 'symbol',
        'Indexvalue': 'vstoxx'
    })
    
    # Parse dates (format: DD.MM.YYYY)
    df['date'] = pd.to_datetime(df['date'], format='%d.%m.%Y', errors='coerce')
    
    # Drop invalid dates
    invalid = df['date'].isna().sum()
    if invalid > 0:
        logger.warning(f"VSTOXX: Dropped {invalid} rows with invalid dates")
        df = df.dropna(subset=['date'])
    
    # Parse numeric values
    df['vstoxx'] = pd.to_numeric(df['vstoxx'], errors='coerce')
    
    # Keep only relevant columns
    df = df[['date', 'vstoxx']].copy()
    
    # Add metadata
    df['source'] = 'stoxx'
    df['currency'] = 'INDEX'
    
    # Sort and deduplicate
    df = df.sort_values('date').drop_duplicates(subset=['date'], keep='last')
    df = df.reset_index(drop=True)
    
    logger.info(f"Loaded VSTOXX: {len(df)} records from {df['date'].min()} to {df['date'].max()}")
    
    return df


def create_vstoxx_features(
    vstoxx_df: pd.DataFrame,
    calendar: pd.DatetimeIndex,
    rolling_windows: list = [5, 20]
) -> pd.DataFrame:
    """
    Create VSTOXX-based features aligned to a calendar.
    
    Args:
        vstoxx_df: DataFrame from load_vstoxx()
        calendar: Target date index
        rolling_windows: Windows for rolling statistics
        
    Returns:
        DataFrame with VSTOXX features
    """
    result = pd.DataFrame({'date': calendar})
    
    if vstoxx_df.empty:
        result['vstoxx'] = np.nan
        return result
    
    # Merge
    result = result.merge(vstoxx_df[['date', 'vstoxx']], on='date', how='left')
    
    # Forward-fill (up to 5 days)
    result['vstoxx'] = result['vstoxx'].ffill(limit=5)
    
    # Rolling features
    for window in rolling_windows:
        result[f'vstoxx_ma_{window}d'] = (
            result['vstoxx'].rolling(window=window, min_periods=1).mean()
        )
        result[f'vstoxx_std_{window}d'] = (
            result['vstoxx'].rolling(window=window, min_periods=1).std()
        )
    
    # Volatility regime indicator (high if above 20-day mean)
    result['vstoxx_high_vol'] = (
        result['vstoxx'] > result['vstoxx_ma_20d']
    ).astype(int)
    
    return result


def standardize_vstoxx_output(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert VSTOXX DataFrame to standardized format.
    """
    return pd.DataFrame({
        'date': df['date'],
        'value': df['vstoxx'],
        'series_name': 'VSTOXX',
        'currency': 'INDEX',
        'source': 'stoxx'
    })
