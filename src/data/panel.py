"""
Panel builder: Combine all data sources into a single daily panel.

This module creates the master dataset used for modeling by:
1. Defining a master calendar from the target series
2. Aligning all features to that calendar
3. Creating technical features (returns, rolling stats)
4. Handling missing data appropriately

Supports two target source families:
- *_FUTURES: Primary carbon futures prices (EUA, UKA, etc.)
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
from .load_weather import load_weather_feature_pack
from .load_vstoxx import load_vstoxx, create_vstoxx_features
from .fx import load_eurusd_fx, convert_usd_to_eur

logger = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[2]
CARBON_FUTURES_SOURCES = {"eua_futures", "uka_futures", "carbon_futures", "futures"}


def _normalize_target_source(value: Optional[str]) -> str:
    return str(value or "eua_futures").strip().lower()


def _normalize_target_instrument(value: Optional[str]) -> str:
    return str(value or "EUA_FUTURES").strip().upper()


def _is_carbon_futures_target(target_source: str, target_instrument: str) -> bool:
    return (
        _normalize_target_source(target_source) in CARBON_FUTURES_SOURCES
        or _normalize_target_instrument(target_instrument).endswith("_FUTURES")
    )


def _target_display_name(target_instrument: str) -> str:
    instrument = _normalize_target_instrument(target_instrument)
    if instrument == "UKA_FUTURES":
        return "UKA Futures"
    if instrument == "EUA_FUTURES":
        return "EUA Futures"
    return instrument.replace("_", " ")


def _add_target_feature_aliases(panel: pd.DataFrame, target_instrument: str) -> pd.DataFrame:
    """Expose neutral target-feature names for non-EUA markets without breaking EU defaults."""
    instrument = _normalize_target_instrument(target_instrument)
    alias_map = {
        "eua_volume": ["target_volume"],
        "eua_return": ["target_return"],
        "eua_range": ["target_range"],
        "eua_range_pct": ["target_range_pct"],
    }
    if instrument == "UKA_FUTURES":
        alias_map["eua_volume"].append("uka_volume")
        alias_map["eua_return"].append("uka_return")
        alias_map["eua_range"].append("uka_range")
        alias_map["eua_range_pct"].append("uka_range_pct")

    for source_col, alias_cols in alias_map.items():
        if source_col not in panel.columns:
            continue
        for alias_col in alias_cols:
            if alias_col not in panel.columns:
                panel[alias_col] = panel[source_col]
    return panel


def _extend_proxy_event_features(panel: pd.DataFrame, days_since_cap: Optional[float] = None) -> pd.DataFrame:
    """Extend UK ICAP proxy event-style features across the full calendar."""
    if "date" not in panel.columns:
        return panel

    date_series = pd.to_datetime(panel["date"], errors="coerce")
    event_specs = [
        ("uk_icap_primary_print_day", "uk_icap_primary_days_since_print"),
        ("uk_icap_secondary_print_day", "uk_icap_secondary_days_since_print"),
    ]

    for flag_col, days_col in event_specs:
        if flag_col not in panel.columns:
            continue
        panel[flag_col] = panel[flag_col].fillna(0.0).astype(float)
        event_dates = date_series.where(panel[flag_col] > 0.0)
        last_event = event_dates.ffill()
        panel[days_col] = (date_series - last_event).dt.days.astype(float)
        if days_since_cap is not None:
            panel[days_col] = panel[days_col].clip(upper=float(days_since_cap))

    if any(col.startswith("uk_icap_") for col in panel.columns):
        dow = date_series.dt.dayofweek.astype(float)
        panel["uk_icap_dow_sin"] = np.sin(2.0 * np.pi * dow / 7.0)
        panel["uk_icap_dow_cos"] = np.cos(2.0 * np.pi * dow / 7.0)

    return panel


def _rolling_zscore(series: pd.Series, window: int) -> pd.Series:
    min_periods = min(window, 5)
    mean = series.rolling(window=window, min_periods=min_periods).mean()
    std = series.rolling(window=window, min_periods=min_periods).std()
    std = std.replace(0.0, np.nan)
    return (series - mean) / std


def _safe_ratio(numer: pd.Series, denom: pd.Series) -> pd.Series:
    denom = denom.replace(0.0, np.nan)
    return numer / denom


def _add_uk_energy_weather_interactions(panel: pd.DataFrame) -> pd.DataFrame:
    """Add UK-specific energy-demand and relative-value features."""
    if "uk_hdd18" in panel.columns and "uk_hdd18_7d_ma" in panel.columns:
        panel["uk_hdd18_surprise"] = panel["uk_hdd18"] - panel["uk_hdd18_7d_ma"]
    if "uk_temp_mean_c" in panel.columns and "uk_temp_mean_7d_ma" in panel.columns:
        panel["uk_temp_mean_anom"] = panel["uk_temp_mean_c"] - panel["uk_temp_mean_7d_ma"]

    interaction_specs = [
        ("uk_gas_return", "uk_hdd18", "uk_gas_hdd18_interaction"),
        ("uk_gas_return", "uk_hdd18_surprise", "uk_gas_hdd18_surprise_interaction"),
        ("uk_power_return", "uk_hdd18", "uk_power_hdd18_interaction"),
        ("uk_power_return", "uk_hdd18_surprise", "uk_power_hdd18_surprise_interaction"),
    ]
    for left_col, right_col, out_col in interaction_specs:
        if left_col in panel.columns and right_col in panel.columns:
            panel[out_col] = panel[left_col] * panel[right_col]

    ratio_specs = [
        ("uk_gas_price", "uka_gas_ratio"),
        ("uk_power_price", "uka_power_ratio"),
        ("brent_price", "uka_brent_ratio"),
        ("coal_price", "uka_coal_ratio"),
    ]
    for price_col, ratio_col in ratio_specs:
        if "y" in panel.columns and price_col in panel.columns:
            panel[ratio_col] = _safe_ratio(panel["y"], panel[price_col])
            panel[f"{ratio_col}_z20"] = _rolling_zscore(panel[ratio_col], 20)

    return_spread_specs = [
        ("uk_gas_return", "uka_gas_return_spread"),
        ("uk_power_return", "uka_power_return_spread"),
        ("brent_return", "uka_brent_return_spread"),
        ("coal_return", "uka_coal_return_spread"),
    ]
    for exog_col, out_col in return_spread_specs:
        if "y_return" in panel.columns and exog_col in panel.columns:
            panel[out_col] = panel["y_return"] - panel[exog_col]

    return panel


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


def load_daily_sentiment_features(
    path: Union[str, Path],
    date_col: str = "seendate",
    score_col: str = "sent_score",
) -> pd.DataFrame:
    """Load a daily sentiment CSV and aggregate numeric columns to one row per date."""
    sentiment_path = Path(path)
    if not sentiment_path.is_absolute():
        sentiment_path = ROOT / sentiment_path
    if not sentiment_path.exists():
        raise FileNotFoundError(f"Sentiment file not found: {sentiment_path}")

    df = pd.read_csv(sentiment_path)
    if date_col not in df.columns:
        fallback = "seendate" if "seendate" in df.columns else ("date" if "date" in df.columns else None)
        if fallback is None:
            raise ValueError(f"Sentiment date column not found in {sentiment_path}")
        date_col = fallback
    if score_col not in df.columns:
        raise ValueError(f"Sentiment score column '{score_col}' not found in {sentiment_path}")

    work = df.copy()
    parsed_dates = pd.to_datetime(work[date_col], errors="coerce")
    if getattr(parsed_dates.dt, "tz", None) is not None:
        parsed_dates = parsed_dates.dt.tz_convert(None)
    work["date"] = parsed_dates.dt.normalize()

    numeric_cols = []
    for col in work.columns:
        if col in {date_col, "date"}:
            continue
        converted = pd.to_numeric(work[col], errors="coerce")
        if converted.notna().any():
            work[col] = converted
            numeric_cols.append(col)
    if score_col not in numeric_cols:
        raise ValueError(f"Sentiment score column '{score_col}' did not parse as numeric in {sentiment_path}")

    agg_spec = {score_col: "mean"}
    for col in numeric_cols:
        if col == score_col:
            continue
        agg_spec[col] = "mean"

    daily = work.groupby("date", as_index=False).agg(agg_spec).sort_values("date").reset_index(drop=True)
    daily = daily.rename(columns={score_col: "sent_score"})
    daily["sent_news_count"] = work.groupby("date").size().reindex(daily["date"]).to_numpy(dtype=float)
    return daily


def add_sentiment_features(
    panel: pd.DataFrame,
    calendar: pd.DatetimeIndex,
    feature_config: Dict,
) -> pd.DataFrame:
    """Merge daily sentiment onto the panel and create derived sentiment features."""
    if not feature_config.get("include_sentiment_features", False):
        return panel

    sentiment_path = feature_config.get("sentiment_path", "data/news/daily_sentiment.csv")
    sentiment_date_col = feature_config.get("sentiment_date_col", "seendate")
    sentiment_score_col = feature_config.get("sentiment_score_col", "sent_score")
    sentiment_fill_limit = int(feature_config.get("sentiment_fill_limit", 5))
    sentiment_windows = [int(w) for w in feature_config.get("sentiment_windows", [3, 7])]
    include_sentiment_change = bool(feature_config.get("include_sentiment_change", True))
    include_sentiment_abs = bool(feature_config.get("include_sentiment_abs", False))
    include_sentiment_volume = bool(feature_config.get("include_sentiment_volume", True))
    include_sentiment_interaction = bool(feature_config.get("include_sentiment_interaction", True))
    include_source_sent_change = bool(feature_config.get("include_source_sent_change", True))
    sentiment_volume_windows = [int(w) for w in feature_config.get("sentiment_volume_windows", [3])]

    logger.info("Adding sentiment features from %s", sentiment_path)
    sentiment_daily = load_daily_sentiment_features(
        path=sentiment_path,
        date_col=sentiment_date_col,
        score_col=sentiment_score_col,
    )

    sent_frame = pd.DataFrame({"date": calendar}).merge(sentiment_daily, on="date", how="left")
    sent_frame["sent_score"] = sent_frame["sent_score"].ffill(limit=sentiment_fill_limit).fillna(0.0)

    added_cols = ["sent_score"]
    for window in sentiment_windows:
        col = f"sent_score_{int(window)}d_ma"
        sent_frame[col] = sent_frame["sent_score"].rolling(window=int(window), min_periods=1).mean()
        added_cols.append(col)

    if include_sentiment_change:
        sent_frame["sent_score_change_1d"] = sent_frame["sent_score"].diff().fillna(0.0)
        added_cols.append("sent_score_change_1d")

    if include_sentiment_abs:
        sent_frame["sent_score_abs"] = sent_frame["sent_score"].abs()
        added_cols.append("sent_score_abs")

    if include_source_sent_change and "sent_change" in sent_frame.columns:
        sent_frame["sent_change"] = pd.to_numeric(sent_frame["sent_change"], errors="coerce").fillna(0.0)
        added_cols.append("sent_change")
        for window in sentiment_windows:
            col = f"sent_change_{int(window)}d_ma"
            sent_frame[col] = sent_frame["sent_change"].rolling(window=int(window), min_periods=1).mean()
            added_cols.append(col)

    if include_sentiment_volume and "news_volume" in sent_frame.columns:
        sent_frame["sent_news_volume"] = pd.to_numeric(sent_frame["news_volume"], errors="coerce").fillna(0.0)
        added_cols.append("sent_news_volume")
        for window in sentiment_volume_windows:
            col = f"sent_news_volume_{int(window)}d_ma"
            sent_frame[col] = sent_frame["sent_news_volume"].rolling(window=int(window), min_periods=1).mean()
            added_cols.append(col)
        if include_sentiment_interaction:
            sent_frame["sent_score_x_volume"] = sent_frame["sent_score"] * np.log1p(sent_frame["sent_news_volume"])
            added_cols.append("sent_score_x_volume")

    if sent_frame["sent_news_count"].nunique(dropna=True) > 1:
        sent_frame["sent_news_count"] = sent_frame["sent_news_count"].fillna(0.0)
        added_cols.append("sent_news_count")
    else:
        sent_frame = sent_frame.drop(columns=["sent_news_count"])

    for col in list(sent_frame.columns):
        if col in {"date", "news_volume"}:
            continue
        if col not in added_cols and col.startswith("sent_"):
            added_cols.append(col)

    panel = panel.merge(sent_frame[["date"] + added_cols], on="date", how="left")
    logger.info("Added sentiment features: %s", added_cols)
    return panel


def load_auction_proxy_feature_pack(
    data_dir: Union[str, Path],
    relative_path: str = "auction-proxy-features/uk_icap_primary_secondary_feature_pack.csv",
) -> pd.DataFrame:
    """Load a precomputed auction proxy feature pack and normalize column types."""
    data_dir = Path(data_dir)
    pack_path = Path(relative_path)
    if not pack_path.is_absolute():
        pack_path = data_dir / pack_path

    if not pack_path.exists():
        raise FileNotFoundError(f"Auction proxy feature pack not found: {pack_path}")

    df = pd.read_csv(pack_path)
    if "date" in df.columns:
        date_col = "date"
    elif "Date" in df.columns:
        date_col = "Date"
    else:
        raise ValueError(f"No date column found in auction proxy pack: {pack_path}")

    df["date"] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date").drop_duplicates(subset=["date"], keep="last")

    for col in df.columns:
        if col in {"date", "Date"}:
            continue
        df[col] = pd.to_numeric(df[col], errors="coerce")

    drop_cols = [c for c in ["Date"] if c in df.columns]
    if drop_cols:
        df = df.drop(columns=drop_cols)
    return df.reset_index(drop=True)


def load_weather_features(
    data_dir: Union[str, Path],
    relative_path: str = "weather-proxy/uk_weather_daily_feature_pack.csv",
) -> pd.DataFrame:
    """Load a precomputed UK weather-demand feature pack."""
    return load_weather_feature_pack(data_dir, relative_path=relative_path)


def select_feature_columns(
    panel: pd.DataFrame,
    target_col: str,
    max_exogenous_features: int = 10,
    preferred_feature_order: Optional[List[str]] = None,
) -> List[str]:
    """Select model feature columns with optional explicit priority ordering."""
    feature_candidates = [c for c in panel.columns if c not in ["date", target_col]]
    ordered: List[str] = []

    def add(name: str) -> None:
        if name in feature_candidates and name not in ordered:
            ordered.append(name)

    add("y")
    for name in preferred_feature_order or []:
        add(name)
    for name in feature_candidates:
        add(name)

    return [target_col] + ordered[: int(max_exogenous_features)]


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
    target_source = _normalize_target_source(target_config.get("source", "eua_futures"))
    target_instrument = _normalize_target_instrument(target_config.get("instrument", "EUA_FUTURES"))
    target_currency = str(target_config.get("currency", "EUR")).strip().upper()
    target_label = _target_display_name(target_instrument)
    
    logger.info("=" * 60)
    logger.info("Building panel dataset")
    logger.info(f"Target source: {target_source} ({target_instrument})")
    logger.info("=" * 60)
    
    # =========================================================================
    # Step 1: Load target series based on configuration
    # =========================================================================
    if _is_carbon_futures_target(target_source, target_instrument):
        logger.info("\n[1/7] Loading target series (%s)...", target_label)
        try:
            eua_df = load_eua_futures_data(data_dir)
            eua_df["instrument"] = target_instrument
            eua_df["source"] = target_source
            if "currency" in eua_df.columns:
                eua_df["currency"] = target_currency
            price_col = target_config.get("price_column", "close")
            target_df = get_eua_futures_target_series(eua_df, price_col=price_col)
            
            if target_df.empty:
                raise ValueError(f"No target data available from {target_label}")
            
            logger.info("%s loaded: %d records", target_label, len(target_df))
        except FileNotFoundError as e:
            logger.warning("%s data not found: %s", target_label, e)
            logger.warning("Falling back to ICAP secondary market...")
            target_source = "icap"
    
    if target_source == "icap" or target_instrument == "ICAP_SECONDARY":
        logger.info("\n[1/7] Loading target series (ICAP Secondary Market)...")
        icap_df = load_icap_data(data_dir)
        target_df = get_icap_target_series(icap_df)
        
        if target_df.empty:
            raise ValueError("No target data available from ICAP")
    
    # Optional date filtering for regime-specific experiments.
    target_min_date = target_config.get("min_date")
    target_max_date = target_config.get("max_date")
    if target_min_date or target_max_date:
        logger.info(
            "Applying target date filter: min_date=%s max_date=%s",
            target_min_date or "<none>",
            target_max_date or "<none>",
        )

    # Create master calendar from target series
    calendar = create_master_calendar(
        target_df,
        min_date=target_min_date,
        max_date=target_max_date,
    )
    
    # Start panel with target
    panel = pd.DataFrame({"date": calendar})
    panel = panel.merge(target_df, on="date", how="left")
    panel = panel.rename(columns={"close_eur": "y"})  # Target variable
    
    logger.info(f"Target coverage: {(~panel['y'].isna()).sum()} / {len(panel)} days")
    
    # Add EUA futures OHLC and volume features if using EUA futures
    if _is_carbon_futures_target(target_source, target_instrument):
        logger.info("Adding %s OHLC and volume features...", target_label)
        try:
            eua_features = create_eua_futures_features(eua_df, calendar, compute_returns=True, compute_range=True)
            # Only include volume and range features (target is already added)
            eua_feature_cols = [c for c in eua_features.columns if c != "date" and c not in ["eua_close", "eua_return"]]
            if eua_feature_cols:
                panel = panel.merge(eua_features[["date"] + eua_feature_cols], on="date", how="left")
                panel = _add_target_feature_aliases(panel, target_instrument)
                logger.info("Added %s features: %s", target_label, eua_feature_cols)
        except Exception as e:
            logger.warning("Failed to add %s features: %s", target_label, e)
        
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

    for name, df in list(energy_dict.items()):
        # Only USD-denominated benchmarks need FX conversion.
        if not df.empty and "price_usd" in df.columns:
            df_converted, audit = convert_usd_to_eur(
                df,
                fx_df,
                price_col="price_usd",
                out_col="price_eur",
            )
            energy_dict[name] = df_converted

    energy_enabled = {
        "brent": bool(feature_config.get("include_brent", True)),
        "coal": bool(feature_config.get("include_coal", True)),
        "uk_gas": bool(feature_config.get("include_uk_gas", False)),
        "uk_power": bool(feature_config.get("include_uk_power", False)),
    }
    selected_energy = {
        name: df
        for name, df in energy_dict.items()
        if energy_enabled.get(name, False)
    }

    if selected_energy:
        energy_features = create_energy_features(
            selected_energy,
            calendar,
            compute_returns=feature_config.get("returns", True),
        )
        panel = panel.merge(energy_features, on="date", how="left")

    # =========================================================================
    # Step 3b: Load UK weather-demand proxy features
    # =========================================================================
    if feature_config.get("include_uk_weather", False):
        weather_pack_path = feature_config.get(
            "weather_feature_pack_path",
            "weather-proxy/uk_weather_daily_feature_pack.csv",
        )
        weather_lag_days = int(feature_config.get("weather_lag_days", 1))
        try:
            weather_df = load_weather_features(data_dir, relative_path=weather_pack_path)
            weather_cols = [c for c in weather_df.columns if c != "date"]
            if weather_lag_days > 0 and weather_cols:
                weather_df = weather_df.sort_values("date").reset_index(drop=True)
                weather_df[weather_cols] = weather_df[weather_cols].shift(weather_lag_days)
            panel = panel.merge(weather_df, on="date", how="left")
            logger.info("Added UK weather feature pack columns: %s", weather_cols)
        except Exception as e:
            logger.warning(f"Failed to add UK weather feature pack: {e}")

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
        requested_auction = {
            str(x).strip().lower()
            for x in feature_config.get("auction_features", ["price", "volume", "is_auction_day"])
        }
        if requested_auction:
            drop_cols: List[str] = []
            if "price" not in requested_auction:
                drop_cols.extend([c for c in auction_features.columns if c.startswith("auction_price")])
            if "volume" not in requested_auction and "auction_volume" in auction_features.columns:
                drop_cols.append("auction_volume")
            if "is_auction_day" not in requested_auction and "is_auction_day" in auction_features.columns:
                drop_cols.append("is_auction_day")
            if drop_cols:
                auction_features = auction_features.drop(columns=sorted(set(drop_cols)))
        panel = panel.merge(auction_features, on="date", how="left")

    if feature_config.get("include_auction_proxy_pack", False):
        proxy_pack_path = feature_config.get(
            "auction_proxy_pack_path",
            "auction-proxy-features/uk_icap_primary_secondary_feature_pack.csv",
        )
        proxy_fill_limit = int(feature_config.get("auction_proxy_fill_limit", 10))
        try:
            proxy_df = load_auction_proxy_feature_pack(data_dir, relative_path=proxy_pack_path)
            proxy_cols = [c for c in proxy_df.columns if c != "date"]
            panel = panel.merge(proxy_df, on="date", how="left")

            ffill_cols = [
                c for c in proxy_cols
                if c.endswith("_ffill")
                or c.endswith("_spread")
                or c.endswith("_spread_pct")
                or c.endswith("_z20")
            ]
            if ffill_cols:
                panel[ffill_cols] = panel[ffill_cols].ffill(limit=proxy_fill_limit)
            panel = _extend_proxy_event_features(
                panel,
                days_since_cap=feature_config.get("auction_proxy_days_since_cap", 30),
            )
            logger.info("Added auction proxy feature pack columns: %s", proxy_cols)
        except Exception as e:
            logger.warning(f"Failed to add auction proxy feature pack: {e}")
    
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

    # Optional daily sentiment features for sentiment-aware TSM variants.
    panel = add_sentiment_features(panel, calendar, feature_config)
    
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

    panel = _add_uk_energy_weather_interactions(panel)
    
    # =========================================================================
    # Create schema
    # =========================================================================
    schema = {
        "target": "y",
        "target_currency": target_currency,
        "target_source": target_source,
        "target_instrument": target_instrument,
        "date_col": "date",
        "n_rows": len(panel),
        "date_range": {
            "start": str(panel["date"].min()),
            "end": str(panel["date"].max())
        },
        "requested_date_range": {
            "min_date": str(target_min_date) if target_min_date is not None else None,
            "max_date": str(target_max_date) if target_max_date is not None else None,
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
