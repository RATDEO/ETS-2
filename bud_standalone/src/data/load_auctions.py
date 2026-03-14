"""
Load EU ETS primary market auction data.

Supports both legacy CSV snapshots and raw EEX Excel reports (.xls/.xlsx).
Each yearly file contains auction events with prices, volumes, and participant info.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Optional, Union
import logging
import re

logger = logging.getLogger(__name__)


def _find_header_row_from_dataframe(df: pd.DataFrame) -> Optional[int]:
    """Find the row index containing the auction table header."""
    for idx in range(len(df)):
        row = df.iloc[idx].tolist()
        values = [str(x).strip() for x in row if pd.notna(x)]
        if not values:
            continue
        joined = " ".join(values).lower()
        has_date = any(v.lower() == "date" for v in values)
        if has_date and "auction" in joined:
            return idx
    return None


def _load_auction_table(file_path: Path) -> pd.DataFrame:
    """
    Load raw auction table from CSV or Excel and return a dataframe with
    detected header columns applied.
    """
    suffix = file_path.suffix.lower()

    if suffix == ".csv":
        with open(file_path, "r", encoding="latin-1") as f:
            lines = f.readlines()

        header_row = None
        for i, line in enumerate(lines):
            if "Date" in line and "Auction" in line:
                header_row = i
                break

        if header_row is None:
            raise ValueError(f"Could not find auction header row in {file_path.name}")

        return pd.read_csv(
            file_path,
            skiprows=header_row,
            encoding="latin-1",
            on_bad_lines="skip",
            low_memory=False,
        )

    if suffix in {".xls", ".xlsx"}:
        raw = pd.read_excel(file_path, header=None)
        header_row = _find_header_row_from_dataframe(raw)
        if header_row is None:
            raise ValueError(f"Could not find auction header row in {file_path.name}")

        header_values = raw.iloc[header_row].tolist()
        columns = []
        for i, value in enumerate(header_values):
            if pd.isna(value):
                columns.append(f"col_{i}")
            else:
                columns.append(str(value).strip())

        table = raw.iloc[header_row + 1 :].copy()
        table.columns = columns
        table = table.dropna(how="all")
        return table

    raise ValueError(f"Unsupported auction file format: {file_path.name}")


def parse_auction_file(file_path: Union[str, Path]) -> pd.DataFrame:
    """
    Parse a single auction report CSV file.
    
    These files have a complex structure:
    - First few rows are headers/metadata
    - Actual data starts after the header rows
    - Key columns: Date, Auction Price, Auction Volume, Total Revenue
    
    Args:
        file_path: Path to the auction CSV file
        
    Returns:
        DataFrame with auction data
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"Auction file not found: {file_path}")
    
    # Read CSV or Excel and normalize into a tabular dataframe.
    try:
        df = _load_auction_table(file_path)
    except Exception as exc:
        logger.warning(f"Could not parse {file_path.name}: {exc}")
        return pd.DataFrame()
    
    # Clean column names
    df.columns = df.columns.str.strip()
    
    # Find relevant columns (they may have special characters)
    date_col = None
    price_col = None
    volume_col = None
    
    for col in df.columns:
        col_lower = col.lower()
        if col_lower == 'date':
            date_col = col
        elif 'auction price' in col_lower or 'price' in col_lower and 'co2' in col_lower:
            price_col = col
        elif 'auction volume' in col_lower and 'tco2' in col_lower.replace(' ', ''):
            volume_col = col
    
    if date_col is None:
        logger.warning(f"No date column found in {file_path.name}")
        return pd.DataFrame()
    
    # Extract relevant columns
    result_cols = {"date": date_col}
    if price_col:
        result_cols["auction_price_eur"] = price_col
    if volume_col:
        result_cols["auction_volume"] = volume_col
    
    result = df[list(result_cols.values())].copy()
    result.columns = list(result_cols.keys())
    
    # Parse dates (format: DD.MM.YYYY)
    result["date"] = pd.to_datetime(result["date"], format="%d.%m.%Y", errors="coerce")
    
    # Drop invalid rows
    result = result.dropna(subset=["date"])
    
    # Parse numeric columns
    for col in ["auction_price_eur", "auction_volume"]:
        if col in result.columns:
            # Remove thousand separators and convert
            result[col] = result[col].astype(str).str.replace(",", "").str.replace(" ", "")
            result[col] = pd.to_numeric(result[col], errors="coerce")
    
    # Add metadata
    year_match = re.search(r"(\d{4})", file_path.stem)
    result["year"] = int(year_match.group(1)) if year_match else None
    result["source"] = "eex_auction"
    result["currency"] = "EUR"
    
    logger.info(f"Parsed {file_path.name}: {len(result)} auction records")
    
    return result


def load_auction_data(data_dir: Union[str, Path]) -> pd.DataFrame:
    """
    Load all auction data files and combine them.
    
    Args:
        data_dir: Path to Data/ directory
        
    Returns:
        Combined DataFrame with all auction records
    """
    data_dir = Path(data_dir)
    auction_dir = data_dir / "emission-spot-primary-market-auction"
    
    if not auction_dir.exists():
        raise FileNotFoundError(f"Auction directory not found: {auction_dir}")
    
    # Prefer legacy CSV snapshots when available; fall back to Excel for years
    # that do not have a CSV counterpart.
    csv_files = sorted(auction_dir.glob("emission-spot-primary-market-auction-report-*.csv"))
    excel_files = sorted(
        p
        for p in auction_dir.glob("emission-spot-primary-market-auction-report-*.*")
        if p.suffix.lower() in {".xls", ".xlsx"}
    )

    def _extract_year(path: Path) -> Optional[int]:
        match = re.search(r"(\d{4})", path.stem)
        return int(match.group(1)) if match else None

    if csv_files:
        csv_years = {_extract_year(path) for path in csv_files}
        excel_only = [path for path in excel_files if _extract_year(path) not in csv_years]
        auction_files = csv_files + excel_only
    else:
        auction_files = excel_files
    
    if not auction_files:
        logger.warning(f"No auction files found in {auction_dir}")
        return pd.DataFrame()
    
    # Load and combine all files
    dfs = []
    for file_path in auction_files:
        try:
            df = parse_auction_file(file_path)
            if not df.empty:
                dfs.append(df)
        except Exception as e:
            logger.error(f"Error parsing {file_path.name}: {e}")
    
    if not dfs:
        return pd.DataFrame()
    
    combined = pd.concat(dfs, ignore_index=True)
    
    # Sort by date and handle duplicates (multiple auctions per day possible)
    # Keep the last auction of each day or aggregate
    combined = combined.sort_values("date")
    
    logger.info(f"Loaded {len(combined)} total auction records from {len(dfs)} files")
    logger.info(f"Date range: {combined['date'].min()} to {combined['date'].max()}")
    
    return combined


def aggregate_daily_auctions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate multiple auctions per day into daily summary.
    
    Args:
        df: Raw auction DataFrame
        
    Returns:
        Daily aggregated DataFrame
    """
    if df.empty:
        return df
    
    agg_dict = {
        "auction_price_eur": "mean",  # Average price if multiple auctions
        "auction_volume": "sum",  # Total volume
    }
    
    # Only aggregate columns that exist
    agg_dict = {k: v for k, v in agg_dict.items() if k in df.columns}
    
    if not agg_dict:
        return df.groupby("date").first().reset_index()
    
    daily = df.groupby("date").agg(agg_dict).reset_index()
    daily["is_auction_day"] = 1
    daily["source"] = "eex_auction"
    daily["currency"] = "EUR"
    
    return daily


def create_auction_features(
    auction_df: pd.DataFrame,
    calendar: pd.DatetimeIndex,
    lags: List[int] = [1, 5, 20]
) -> pd.DataFrame:
    """
    Create auction-based features aligned to a calendar.
    
    Features:
    - auction_price_eur: Price on auction days
    - auction_volume: Volume on auction days
    - is_auction_day: Binary indicator
    - auction_price_lag_X: Lagged auction prices
    - auction_price_rolling_mean_20: Rolling mean of auction prices
    
    Args:
        auction_df: Daily aggregated auction data
        calendar: Target date index to align to
        lags: List of lag periods
        
    Returns:
        DataFrame with auction features aligned to calendar
    """
    # Start with the calendar
    result = pd.DataFrame({"date": calendar})
    
    if auction_df.empty:
        result["is_auction_day"] = 0
        return result
    
    # Merge auction data
    auction_daily = aggregate_daily_auctions(auction_df)
    result = result.merge(
        auction_daily[["date", "auction_price_eur", "auction_volume"]],
        on="date",
        how="left"
    )
    
    # Create auction day indicator
    result["is_auction_day"] = (~result["auction_price_eur"].isna()).astype(int)
    
    # Forward-fill auction price for lag calculations
    # (use most recent auction price)
    result["auction_price_ffill"] = result["auction_price_eur"].ffill()
    
    # Create lagged features
    for lag in lags:
        result[f"auction_price_lag_{lag}"] = result["auction_price_ffill"].shift(lag)
    
    # Rolling statistics (based on filled values)
    result["auction_price_rolling_mean_20"] = (
        result["auction_price_ffill"].rolling(window=20, min_periods=1).mean()
    )
    
    # Drop the intermediate ffill column
    result = result.drop(columns=["auction_price_ffill"])
    
    return result


def standardize_auction_output(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert auction DataFrame to standardized format.
    """
    daily = aggregate_daily_auctions(df)
    
    return pd.DataFrame({
        "date": daily["date"],
        "value": daily["auction_price_eur"],
        "series_name": "AUCTION_PRICE_EUR",
        "currency": "EUR",
        "source": "eex_auction"
    })
