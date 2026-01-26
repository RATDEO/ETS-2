"""
Panel builder: Combine all data sources into a single daily panel.

This module creates the master dataset used for modeling by:
1. Defining a master calendar from the target series
2. Aligning all features to that calendar
3. Creating technical features (returns, rolling stats)
4. Handling missing data appropriately

Supports two target sources:
- EUA_FUTURES: Primary EUA futures prices (preferred)
- ICAP_SECONDARY: ICAP secondary market prices (legacy)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple
import logging
import json

from .load_icap import load_icap_data, get_icap_target_series, create_icap_features
from .load_eua_futures import load_eua_futures_data, get_eua_futures_target_series, create_eua_futures_features
from .load_indices import load_carbon_indices, get_indices_combined
from .load_auctions import load_auction_data, create_auction_features
from .load_energy import load_energy_benchmarks, create_energy_features
from .load_vstoxx import load_vstoxx, create_vstoxx_features
from .fx import load_eurusd_fx, convert_usd_to_eur

logger = logging.getLogger(__name__)


def create_master_calendar(
    target_df: pd.DataFrame,
    date_col: str = "date",
    min_date: Optional[str] = None,
    max_date: Optional[str] = None
) -> pd.DatetimeIndex:
    """
    Create the master calendar from the target series.
    
    Uses dates where the target has valid prices.
    
    Args:
        target_df: Target series DataFrame
        date_col: Name of date column
        min_date: Optional minimum date filter
        max_date: Optional maximum date filter
        
    Returns:
        DatetimeIndex of valid trading days
    """
    dates = target_df[date_col].dropna().sort_values().unique()
    calendar = pd.DatetimeIndex(dates)
    
    if min_date:
        calendar = calendar[calendar >= pd.Timestamp(min_date)]
    if max_date:
        calendar = calendar[calendar <= pd.Timestamp(max_date)]
    
    logger.info(f"Master calendar: {len(calendar)} days from {calendar.min()} to {calendar.max()}")
    
    return calendar


def build_panel(
    data_dir: Union[str, Path],
    config: Optional[Dict] = None,
    save_path: Optional[Path] = None
) -> Tuple[pd.DataFrame, Dict]:
    """
    Build the complete daily panel from all data sources.
    
    Args:
        data_dir: Path to Data/ directory
        config: Optional configuration dictionary
        save_path: Optional path to save the panel
        
    Returns:
        Tuple of (panel DataFrame, schema dictionary)
    """
    data_dir = Path(data_dir)
    config = config or {}
    
    feature_config = config.get("features", {})
    target_config = config.get("target", {})
    
    # Determine target source
    target_source = target_config.get("source", "eua_futures")  # Default to EUA futures
    target_instrument = target_config.get("instrument", "EUA_FUTURES")
    
    logger.info("=" * 60)
    logger.info("Building panel dataset")
    logger.info(f"Target source: {target_source} ({target_instrument})")
    logger.info("=" * 60)
    
    # =========================================================================
    # Step 1: Load target series based on configuration
    # =========================================================================
    if target_source == "eua_futures" or target_instrument == "EUA_FUTURES":
        logger.info("\n[1/7] Loading target series (EUA Futures)...")
        try:
            eua_df = load_eua_futures_data(data_dir)
            price_col = target_config.get("price_column", "close")
            target_df = get_eua_futures_target_series(eua_df, price_col=price_col)
            
            if target_df.empty:
                raise ValueError("No target data available from EUA Futures")
            
            logger.info(f"EUA Futures loaded: {len(target_df)} records")
        except FileNotFoundError as e:
            logger.warning(f"EUA Futures data not found: {e}")
            logger.warning("Falling back to ICAP secondary market...")
            target_source = "icap"
    
    if target_source == "icap" or target_instrument == "ICAP_SECONDARY":
        logger.info("\n[1/7] Loading target series (ICAP Secondary Market)...")
        icap_df = load_icap_data(data_dir)
        target_df = get_icap_target_series(icap_df)
        
        if target_df.empty:
            raise ValueError("No target data available from ICAP")
    
    # Create master calendar from target series
    calendar = create_master_calendar(target_df)
    
    # Start panel with target
    panel = pd.DataFrame({"date": calendar})
    panel = panel.merge(target_df, on="date", how="left")
    panel = panel.rename(columns={"close_eur": "y"})  # Target variable
    
    logger.info(f"Target coverage: {(~panel['y'].isna()).sum()} / {len(panel)} days")
    
    # Add EUA futures OHLC and volume features if using EUA futures
    if target_source == "eua_futures" or target_instrument == "EUA_FUTURES":
        logger.info("Adding EUA futures OHLC and volume features...")
        try:
            eua_features = create_eua_futures_features(eua_df, calendar, compute_returns=True, compute_range=True)
            # Only include volume and range features (target is already added)
            eua_feature_cols = [c for c in eua_features.columns if c != "date" and c not in ["eua_close", "eua_return"]]
            if eua_feature_cols:
                panel = panel.merge(eua_features[["date"] + eua_feature_cols], on="date", how="left")
                logger.info(f"Added EUA features: {eua_feature_cols}")
        except Exception as e:
            logger.warning(f"Failed to add EUA features: {e}")
        
        # Also add ICAP secondary market as exogenous feature
        if feature_config.get("include_icap_secondary", True):
            logger.info("Adding ICAP secondary market as exogenous feature...")
            try:
                icap_df = load_icap_data(data_dir)
                icap_target = get_icap_target_series(icap_df)
                if not icap_target.empty:
                    icap_target = icap_target.rename(columns={"close_eur": "icap_secondary"})
                    panel = panel.merge(icap_target[["date", "icap_secondary"]], on="date", how="left")
                    panel["icap_secondary"] = panel["icap_secondary"].ffill(limit=5)
                    # Compute ICAP return
                    panel["icap_return"] = np.log(panel["icap_secondary"] / panel["icap_secondary"].shift(1))
                    logger.info("Added ICAP secondary market feature")
            except Exception as e:
                logger.warning(f"Failed to add ICAP secondary feature: {e}")

    # =========================================================================
    # Step 2: Load and convert FX rates
    # =========================================================================
    logger.info("\n[2/7] Loading FX rates...")
    fx_file = data_dir / "usd.xml"
    fx_df = load_eurusd_fx(fx_file)
    
    # Add FX to panel
    panel = panel.merge(fx_df[["date", "eurusd"]], on="date", how="left")
    panel["eurusd"] = panel["eurusd"].ffill(limit=5)
    
    # =========================================================================
    # Step 3: Load and align energy benchmarks (convert USD → EUR)
    # =========================================================================
    logger.info("\n[3/7] Loading energy benchmarks...")
    energy_dict = load_energy_benchmarks(data_dir)
    
    for name, df in energy_dict.items():
        # Convert to EUR
        if not df.empty:
            df_converted, audit = convert_usd_to_eur(
                df, fx_df, 
                price_col="price_usd",
                out_col="price_eur"
            )
            energy_dict[name] = df_converted
    
    if feature_config.get("include_brent", True) or feature_config.get("include_coal", True):
        energy_features = create_energy_features(
            energy_dict, 
            calendar,
            compute_returns=feature_config.get("returns", True)
        )
        panel = panel.merge(energy_features, on="date", how="left")
    
    # =========================================================================
    # Step 4: Load and align carbon indices
    # =========================================================================
    logger.info("\n[4/7] Loading carbon indices...")
    if feature_config.get("include_indices", True):
        indices_list = feature_config.get("indices_list", ["KEUA", "KRBN", "GRN"])
        indices_dict = load_carbon_indices(data_dir, indices_list)
        
        # Indices are USD-denominated, convert to EUR
        for ticker, df in indices_dict.items():
            df_converted, _ = convert_usd_to_eur(
                df.rename(columns={"close": "price_usd"}),
                fx_df,
                price_col="price_usd",
                out_col="price_eur"
            )
            
            # Add to panel
            temp = df_converted[["date", "price_eur"]].copy()
            temp = temp.rename(columns={"price_eur": f"idx_{ticker.lower()}"})
            panel = panel.merge(temp, on="date", how="left")
            panel[f"idx_{ticker.lower()}"] = panel[f"idx_{ticker.lower()}"].ffill(limit=5)
    
    # =========================================================================
    # Step 5: Load and align auction data
    # =========================================================================
    logger.info("\n[5/7] Loading auction data...")
    if feature_config.get("include_auctions", True):
        auction_df = load_auction_data(data_dir)
        auction_lags = feature_config.get("auction_lags", [1, 5, 20])
        
        auction_features = create_auction_features(auction_df, calendar, lags=auction_lags)
        panel = panel.merge(auction_features, on="date", how="left")
    
    # =========================================================================
    # Step 6: Load and align VSTOXX
    # =========================================================================
    logger.info("\n[6/7] Loading VSTOXX...")
    if feature_config.get("include_vstoxx", True):
        vstoxx_file = data_dir / "volatility-proxy" / "vstoxx-index.txt"
        if vstoxx_file.exists():
            vstoxx_df = load_vstoxx(vstoxx_file)
            vstoxx_features = create_vstoxx_features(vstoxx_df, calendar)
            panel = panel.merge(vstoxx_features, on="date", how="left")
    
    # =========================================================================
    # Step 7: Create target features (returns, rolling stats)
    # =========================================================================
    logger.info("\n[7/7] Creating target features...")
    
    # Log returns
    panel["y_return"] = np.log(panel["y"] / panel["y"].shift(1))
    
    # Rolling statistics
    rolling_windows = feature_config.get("rolling_mean_windows", [5, 20, 60])
    vol_window = feature_config.get("rolling_vol_window", 20)
    
    for window in rolling_windows:
        panel[f"y_ma_{window}d"] = panel["y"].rolling(window=window, min_periods=1).mean()
    
    panel[f"y_vol_{vol_window}d"] = panel["y_return"].rolling(window=vol_window, min_periods=1).std()
    
    # Momentum features
    panel["y_momentum_5d"] = panel["y"] / panel["y"].shift(5) - 1
    panel["y_momentum_20d"] = panel["y"] / panel["y"].shift(20) - 1
    
    # =========================================================================
    # Create schema
    # =========================================================================
    schema = {
        "target": "y",
        "target_currency": "EUR",
        "date_col": "date",
        "n_rows": len(panel),
        "date_range": {
            "start": str(panel["date"].min()),
            "end": str(panel["date"].max())
        },
        "columns": {col: str(panel[col].dtype) for col in panel.columns},
        "missing_summary": {col: int(panel[col].isna().sum()) for col in panel.columns}
    }
    
    # =========================================================================
    # Save if requested
    # =========================================================================
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        panel.to_parquet(save_path / "panel.parquet", index=False)
        
        with open(save_path / "panel_schema.json", "w") as f:
            json.dump(schema, f, indent=2)
        
        logger.info(f"Saved panel to {save_path}")
    
    logger.info("=" * 60)
    logger.info(f"Panel built: {len(panel)} rows, {len(panel.columns)} columns")
    logger.info(f"Target coverage: {(~panel['y'].isna()).sum() / len(panel) * 100:.1f}%")
    logger.info("=" * 60)
    
    return panel, schema


def get_coverage_report(panel: pd.DataFrame) -> pd.DataFrame:
    """
    Generate a coverage report for the panel.
    
    Args:
        panel: Panel DataFrame
        
    Returns:
        Coverage statistics DataFrame
    """
    report = []
    
    for col in panel.columns:
        if col == "date":
            continue
            
        valid = (~panel[col].isna()).sum()
        total = len(panel)
        coverage = valid / total * 100
        
        # Get date range where data exists
        valid_dates = panel.loc[~panel[col].isna(), "date"]
        
        report.append({
            "column": col,
            "valid_count": valid,
            "missing_count": total - valid,
            "coverage_pct": coverage,
            "first_date": str(valid_dates.min()) if len(valid_dates) > 0 else None,
            "last_date": str(valid_dates.max()) if len(valid_dates) > 0 else None
        })
    
    return pd.DataFrame(report)


def plot_coverage_heatmap(
    panel: pd.DataFrame,
    save_path: Optional[Path] = None
) -> None:
    """
    Create a coverage heatmap visualization.
    
    Args:
        panel: Panel DataFrame
        save_path: Optional path to save the figure
    """
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        logger.warning("matplotlib/seaborn not available for coverage plot")
        return
    
    # Create binary coverage matrix
    cols = [c for c in panel.columns if c != "date"]
    coverage_matrix = (~panel[cols].isna()).astype(int)
    
    # Resample to monthly for visualization
    coverage_matrix["date"] = panel["date"]
    monthly = coverage_matrix.set_index("date").resample("M").mean()
    
    # Plot
    fig, ax = plt.subplots(figsize=(14, 8))
    sns.heatmap(
        monthly.T,
        cmap="YlGn",
        ax=ax,
        cbar_kws={"label": "Coverage %"}
    )
    
    ax.set_title("Data Coverage by Feature (Monthly Average)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Feature")
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved coverage heatmap to {save_path}")
    
    plt.close()
