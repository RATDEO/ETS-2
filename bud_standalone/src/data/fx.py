"""
FX conversion utilities for USD to EUR conversion.

Uses ECB reference exchange rates from the XML data file.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Union, Tuple
import logging
import xml.etree.ElementTree as ET

logger = logging.getLogger(__name__)


def load_eurusd_fx(file_path: Union[str, Path]) -> pd.DataFrame:
    """
    Load EUR/USD exchange rates from ECB XML data.
    
    The XML contains daily observations with:
    - TIME_PERIOD: date
    - OBS_VALUE: USD per EUR
    
    Args:
        file_path: Path to usd.xml file
        
    Returns:
        DataFrame with date and eurusd columns
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"FX file not found: {file_path}")
    
    records = []

    # Support CSV snapshots in addition to XML.
    if file_path.suffix.lower() == ".csv":
        df = pd.read_csv(file_path)
        date_col = None
        value_col = None
        for candidate in ["TIME_PERIOD", "observation_date", "date"]:
            if candidate in df.columns:
                date_col = candidate
                break
        for candidate in ["OBS_VALUE", "eurusd", "value"]:
            if candidate in df.columns:
                value_col = candidate
                break
        if date_col and value_col:
            records = [
                {"date": row[date_col], "eurusd": row[value_col]}
                for _, row in df.iterrows()
            ]
    else:
        # Parse XML formats:
        # 1) SDMX compact (TIME_PERIOD / OBS_VALUE attributes)
        # 2) SDMX generic (ObsDimension / ObsValue child nodes)
        # 3) ECB eurofxref-hist Cube format (time/currency/rate)
        tree = ET.parse(file_path)
        root = tree.getroot()

        # 1) SDMX compact format.
        for elem in root.iter():
            if elem.tag.endswith("Obs"):
                time_period = elem.get("TIME_PERIOD")
                obs_value = elem.get("OBS_VALUE")
                if time_period and obs_value:
                    records.append({"date": time_period, "eurusd": obs_value})

        # 2) SDMX generic format.
        if not records:
            for obs in root.iter():
                if not obs.tag.endswith("Obs"):
                    continue
                time_period = None
                obs_value = None
                for child in list(obs):
                    if child.tag.endswith("ObsDimension"):
                        time_period = child.attrib.get("value")
                    elif child.tag.endswith("ObsValue"):
                        obs_value = child.attrib.get("value")
                if time_period and obs_value:
                    records.append({"date": time_period, "eurusd": obs_value})

        # 3) ECB eurofxref-hist format.
        if not records:
            for day_cube in root.iter():
                day = day_cube.attrib.get("time")
                if not day:
                    continue
                for item in list(day_cube):
                    if item.attrib.get("currency") == "USD":
                        rate = item.attrib.get("rate")
                        if rate:
                            records.append({"date": day, "eurusd": rate})
                            break

    if not records:
        logger.warning("No FX observations found in FX file")
        return pd.DataFrame(columns=["date", "eurusd"])

    df = pd.DataFrame(records)
    
    # Parse dates
    df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d", errors="coerce")
    df["eurusd"] = pd.to_numeric(df["eurusd"], errors="coerce")
    
    # Drop invalid dates
    df = df.dropna(subset=["date", "eurusd"])
    
    # Sort and deduplicate
    df = df.sort_values("date").drop_duplicates(subset=["date"], keep="last")
    df = df.reset_index(drop=True)
    
    logger.info(f"Loaded EUR/USD FX rates: {len(df)} observations from {df['date'].min()} to {df['date'].max()}")
    
    return df


def convert_usd_to_eur(
    df_usd: pd.DataFrame,
    fx_df: pd.DataFrame,
    price_col: str = "price_usd",
    out_col: str = "price_eur",
    date_col: str = "date",
    max_ffill_days: int = 5
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Convert USD prices to EUR using ECB exchange rates.
    
    Conversion: EUR_price = USD_price / EURUSD_rate
    
    Args:
        df_usd: DataFrame with USD prices
        fx_df: DataFrame with EUR/USD rates from load_eurusd_fx()
        price_col: Column name containing USD prices
        out_col: Column name for EUR prices in output
        date_col: Column name for dates
        max_ffill_days: Maximum days to forward-fill FX rates
        
    Returns:
        Tuple of (converted DataFrame, audit DataFrame)
    """
    if df_usd.empty:
        return df_usd.copy(), pd.DataFrame()
    
    result = df_usd.copy()
    
    # Merge FX rates
    result = result.merge(
        fx_df[[date_col, 'eurusd']],
        on=date_col,
        how='left'
    )
    
    # Track coverage before forward-fill
    missing_before = result['eurusd'].isna().sum()
    
    # Forward-fill FX rates (for weekends/holidays)
    result['eurusd'] = result['eurusd'].ffill(limit=max_ffill_days)
    
    # Track coverage after forward-fill
    missing_after = result['eurusd'].isna().sum()
    
    # Convert: EUR = USD / EURUSD_rate
    result[out_col] = result[price_col] / result['eurusd']
    
    # Create audit log
    audit = pd.DataFrame({
        'metric': [
            'total_rows',
            'missing_fx_before_ffill',
            'missing_fx_after_ffill',
            'ffill_count',
            'conversion_failures',
            'date_range_start',
            'date_range_end'
        ],
        'value': [
            len(result),
            missing_before,
            missing_after,
            missing_before - missing_after,
            result[out_col].isna().sum(),
            str(result[date_col].min()),
            str(result[date_col].max())
        ]
    })
    
    # Log warnings for large gaps
    gap_pct = missing_after / len(result) * 100
    if gap_pct > 5:
        logger.warning(f"FX conversion: {gap_pct:.1f}% of rows still missing FX rate after forward-fill")
    
    logger.info(f"Converted {len(result) - missing_after} prices from USD to EUR")
    
    return result, audit


def get_fx_for_date_range(
    fx_df: pd.DataFrame,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    fill_method: str = "ffill"
) -> pd.DataFrame:
    """
    Get FX rates for a specific date range with a complete daily calendar.
    
    Args:
        fx_df: DataFrame from load_eurusd_fx()
        start_date: Start of date range
        end_date: End of date range
        fill_method: How to fill missing dates ('ffill', 'bfill', 'interpolate')
        
    Returns:
        DataFrame with complete daily calendar and FX rates
    """
    # Create full calendar
    calendar = pd.date_range(start=start_date, end=end_date, freq='D')
    result = pd.DataFrame({'date': calendar})
    
    # Merge with FX data
    result = result.merge(fx_df[['date', 'eurusd']], on='date', how='left')
    
    # Fill missing values
    if fill_method == 'ffill':
        result['eurusd'] = result['eurusd'].ffill()
    elif fill_method == 'bfill':
        result['eurusd'] = result['eurusd'].bfill()
    elif fill_method == 'interpolate':
        result['eurusd'] = result['eurusd'].interpolate(method='linear')
    
    return result
