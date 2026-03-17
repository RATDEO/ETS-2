"""
Load energy benchmark data.

Supports:
- Brent crude spot (USD)
- Rotterdam coal futures (USD)
- UK SAP gas (pence/kWh, ONS / National Gas)
- UK system electricity price (pence/kWh, ONS / Elexon)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Union
import logging

logger = logging.getLogger(__name__)


def _rolling_zscore(series: pd.Series, window: int) -> pd.Series:
    min_periods = min(window, 5)
    mean = series.rolling(window=window, min_periods=min_periods).mean()
    std = series.rolling(window=window, min_periods=min_periods).std()
    std = std.replace(0.0, np.nan)
    return (series - mean) / std


def _load_standard_energy_csv(
    file_path: Union[str, Path],
    *,
    price_col: str,
    series_name: str,
    source: str,
    currency: str,
) -> pd.DataFrame:
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Energy file not found: {file_path}")

    df = pd.read_csv(file_path)
    if "date" not in df.columns and "Date" in df.columns:
        df = df.rename(columns={"Date": "date"})
    if "date" not in df.columns:
        raise ValueError(f"Energy file missing date column: {file_path}")
    if price_col not in df.columns:
        raise ValueError(f"Energy file missing '{price_col}' column: {file_path}")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df[price_col] = pd.to_numeric(df[price_col], errors="coerce")
    df = df.dropna(subset=["date", price_col]).sort_values("date").drop_duplicates(subset=["date"], keep="last")

    if "rolling_avg_7d" in df.columns:
        df["rolling_avg_7d"] = pd.to_numeric(df["rolling_avg_7d"], errors="coerce")

    df["series_name"] = series_name
    df["source"] = source
    df["currency"] = currency
    return df.reset_index(drop=True)


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


def load_uk_sap_gas(file_path: Union[str, Path]) -> pd.DataFrame:
    """
    Load official UK SAP gas data published by ONS and sourced from National Gas.

    Expected normalized CSV format written by the UK bootstrap:
    - date
    - price_native (pence per kWh)
    - rolling_avg_7d
    """
    df = _load_standard_energy_csv(
        file_path,
        price_col="price_native",
        series_name="UK_SAP_GAS",
        source="ons_national_gas",
        currency="GBp/kWh",
    )
    logger.info("Loaded UK SAP gas: %s records from %s to %s", len(df), df["date"].min(), df["date"].max())
    return df


def load_uk_system_power(file_path: Union[str, Path]) -> pd.DataFrame:
    """
    Load official UK system electricity price data published by ONS and sourced from Elexon.

    Expected normalized CSV format written by the UK bootstrap:
    - date
    - price_native (pence per kWh)
    - rolling_avg_7d
    """
    df = _load_standard_energy_csv(
        file_path,
        price_col="price_native",
        series_name="UK_SYSTEM_POWER",
        source="ons_elexon",
        currency="GBp/kWh",
    )
    logger.info("Loaded UK system power: %s records from %s to %s", len(df), df["date"].min(), df["date"].max())
    return df


def load_energy_benchmarks(data_dir: Union[str, Path]) -> Dict[str, pd.DataFrame]:
    """
    Load all energy benchmark data.
    
    Args:
        data_dir: Path to Data/ directory
        
    Returns:
        Dictionary with available energy benchmark DataFrames
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

    uk_gas_file = energy_dir / "uk-sap-gas-pence-per-kwh.csv"
    if uk_gas_file.exists():
        try:
            result["uk_gas"] = load_uk_sap_gas(uk_gas_file)
        except Exception as e:
            logger.error(f"Error loading UK gas data: {e}")

    uk_power_file = energy_dir / "uk-system-electricity-price-pence-per-kwh.csv"
    if uk_power_file.exists():
        try:
            result["uk_power"] = load_uk_system_power(uk_power_file)
        except Exception as e:
            logger.error(f"Error loading UK power data: {e}")
    
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
            
        # Price column may be FX-converted, USD-native, or native local units.
        if "price_eur" in df.columns:
            price_col = "price_eur"
        elif "price_usd" in df.columns:
            price_col = "price_usd"
        elif "price_native" in df.columns:
            price_col = "price_native"
        else:
            logger.warning("Skipping energy series %s because no usable price column was found", name)
            continue
        
        # Merge price
        temp = df[["date", price_col]].copy()
        temp = temp.rename(columns={price_col: f"{name}_price"})
        result = result.merge(temp, on="date", how="left")
        
        # Forward-fill up to 5 days
        result[f"{name}_price"] = result[f"{name}_price"].ffill(limit=5)
        
        if compute_returns:
            # Log returns
            prev_price = result[f"{name}_price"].shift(1)
            valid = (result[f"{name}_price"] > 0) & (prev_price > 0)
            returns = pd.Series(np.nan, index=result.index, dtype=float)
            valid_idx = valid.fillna(False)
            returns.loc[valid_idx] = np.log(
                result.loc[valid_idx, f"{name}_price"] / prev_price.loc[valid_idx]
            )
            result[f"{name}_return"] = returns.to_numpy()
        
        # Rolling volatility
        for window in rolling_windows:
            if compute_returns:
                result[f"{name}_vol_{window}d"] = (
                    result[f"{name}_return"].rolling(window=window, min_periods=1).std()
                )
    
    # Create spread features if both are available
    if "brent_price" in result.columns and "coal_price" in result.columns:
        result["coal_brent_ratio"] = result["coal_price"] / result["brent_price"]
        result["coal_brent_ratio_z20"] = _rolling_zscore(result["coal_brent_ratio"], 20)
    if "brent_return" in result.columns and "coal_return" in result.columns:
        result["coal_brent_return_spread"] = result["coal_return"] - result["brent_return"]
    if "brent_vol_20d" in result.columns and "coal_vol_20d" in result.columns:
        denom = result["brent_vol_20d"].replace(0.0, np.nan)
        result["coal_brent_vol_ratio_20d"] = result["coal_vol_20d"] / denom
    if "uk_power_price" in result.columns and "uk_gas_price" in result.columns:
        denom = result["uk_gas_price"].replace(0.0, np.nan)
        result["uk_power_gas_spread"] = result["uk_power_price"] - result["uk_gas_price"]
        result["uk_power_gas_ratio"] = result["uk_power_price"] / denom
        result["uk_power_gas_ratio_z20"] = _rolling_zscore(result["uk_power_gas_ratio"], 20)
        result["uk_power_gas_spread_z20"] = _rolling_zscore(result["uk_power_gas_spread"], 20)
    if "uk_power_return" in result.columns and "uk_gas_return" in result.columns:
        result["uk_power_gas_return_spread"] = result["uk_power_return"] - result["uk_gas_return"]
    if "uk_power_vol_20d" in result.columns and "uk_gas_vol_20d" in result.columns:
        denom = result["uk_gas_vol_20d"].replace(0.0, np.nan)
        result["uk_power_gas_vol_ratio_20d"] = result["uk_power_vol_20d"] / denom

    return result


def standardize_energy_output(df: pd.DataFrame, name: str) -> pd.DataFrame:
    """
    Convert energy DataFrame to standardized format.
    """
    if "price_eur" in df.columns:
        price_col = "price_eur"
        currency = "EUR"
    elif "price_usd" in df.columns:
        price_col = "price_usd"
        currency = "USD"
    else:
        price_col = "price_native"
        currency = str(df.get("currency", pd.Series(["native"])).iloc[0])
    
    return pd.DataFrame({
        "date": df["date"],
        "value": df[price_col],
        "series_name": f"ENERGY_{name.upper()}",
        "currency": currency,
        "source": df["source"].iloc[0] if "source" in df.columns else "energy"
    })
