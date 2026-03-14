"""Regime audit utilities for benchmark forecast comparisons."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd

from ..data.load_eua_futures import get_eua_futures_target_series, load_eua_futures_data
from ..data.panel import load_daily_sentiment_features
from ..news.official_event_features import build_official_origin_feature_frame
from .trend_classification import classify_trend, compute_threshold


HORIZONS: Tuple[int, ...] = (1, 5, 20, 30)


@dataclass(frozen=True)
class RegimeThresholds:
    volatility_median: float
    sentiment_abs_median: float
    return_abs_median: float
    trend_threshold_multiplier: float
    volatility_window: int
    trend_window: int
    return_window: int
    sentiment_window: int
    sentiment_fill_limit: int
    official_event_abs_median: float = np.nan
    official_supply_abs_median: float = np.nan


def sample_path_mse(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Per-sample mean squared error across the full forecast path."""
    return np.mean((y_true - y_pred) ** 2, axis=1)


def overall_metrics(
    y_true: np.ndarray,
    predictions: Mapping[str, np.ndarray],
    horizons: Sequence[int] = HORIZONS,
) -> pd.DataFrame:
    """Compute overall path/horizon MSE for each model."""
    rows: List[Dict[str, float]] = []
    for model_name, y_pred in predictions.items():
        row: Dict[str, float] = {
            "model": model_name,
            "n_samples": int(y_true.shape[0]),
            "path_mse": float(sample_path_mse(y_true, y_pred).mean()),
        }
        for h in horizons:
            row[f"h{int(h)}_mse"] = float(np.mean((y_true[:, h - 1] - y_pred[:, h - 1]) ** 2))
        rows.append(row)
    return pd.DataFrame(rows).sort_values("path_mse").reset_index(drop=True)


def _compute_trend_regime(
    prices: pd.Series,
    trend_window: int,
    trend_threshold_multiplier: float,
) -> Tuple[List[str], List[float]]:
    labels: List[str] = []
    thresholds: List[float] = []
    values = prices.to_numpy(dtype=float)
    for idx in range(len(values)):
        start = max(0, idx - trend_window)
        history = values[start : idx + 1]
        if len(history) < 2 or not np.isfinite(history).all():
            labels.append("unknown")
            thresholds.append(np.nan)
            continue
        threshold = compute_threshold(
            history,
            method="volatility",
            multiplier=float(trend_threshold_multiplier),
            window=min(trend_window, max(1, len(history) - 1)),
        )
        trend_code = classify_trend(history[0], history[-1], threshold)
        label = {1: "up", 0: "flat", -1: "down"}[int(trend_code)]
        labels.append(label)
        thresholds.append(float(threshold))
    return labels, thresholds


def build_origin_regime_frame(
    origin_dates: Sequence[pd.Timestamp],
    data_dir: Path | str,
    sentiment_path: Path | str,
    official_event_path: Path | str | None = None,
    *,
    sentiment_date_col: str = "seendate",
    sentiment_score_col: str = "sent_score",
    sentiment_fill_limit: int = 5,
    sentiment_window: int = 3,
    return_window: int = 5,
    volatility_window: int = 20,
    trend_window: int = 20,
    trend_threshold_multiplier: float = 0.25,
) -> Tuple[pd.DataFrame, RegimeThresholds]:
    """Create regime labels for each forecast origin date using only prior information."""
    price_df = get_eua_futures_target_series(load_eua_futures_data(data_dir))
    price_df = price_df[["date", "close_eur"]].dropna().sort_values("date").drop_duplicates("date")
    price_df["date"] = pd.to_datetime(price_df["date"]).dt.normalize()

    sentiment_df = load_daily_sentiment_features(
        path=sentiment_path,
        date_col=sentiment_date_col,
        score_col=sentiment_score_col,
    )
    sentiment_df["date"] = pd.to_datetime(sentiment_df["date"]).dt.normalize()
    sentiment_df = sentiment_df[["date", "sent_score"]].drop_duplicates("date")

    feature_df = price_df.merge(sentiment_df, on="date", how="left")
    feature_df["sent_score"] = feature_df["sent_score"].ffill(limit=sentiment_fill_limit).fillna(0.0)
    feature_df["sent_score_3d_ma"] = (
        feature_df["sent_score"].rolling(window=sentiment_window, min_periods=1).mean()
    )
    feature_df["log_ret_1d"] = np.log(feature_df["close_eur"]).diff()
    feature_df["hist_vol_20d"] = (
        feature_df["log_ret_1d"].rolling(window=volatility_window, min_periods=2).std()
    )
    feature_df["ret_5d"] = feature_df["close_eur"].pct_change(return_window)
    feature_df["ret_20d"] = feature_df["close_eur"].pct_change(trend_window)

    trend_labels, trend_thresholds = _compute_trend_regime(
        feature_df["close_eur"],
        trend_window=trend_window,
        trend_threshold_multiplier=trend_threshold_multiplier,
    )
    feature_df["trend_regime_20d"] = trend_labels
    feature_df["trend_threshold_20d"] = trend_thresholds

    origin_index = pd.Index(pd.to_datetime(origin_dates).normalize(), name="date")
    origin_df = (
        feature_df.set_index("date")
        .reindex(origin_index)
        .reset_index()
        .rename(columns={"index": "date"})
    )
    if origin_df[["close_eur", "sent_score_3d_ma"]].isna().any().any():
        missing = origin_df.loc[
            origin_df[["close_eur", "sent_score_3d_ma"]].isna().any(axis=1), "date"
        ].dt.strftime("%Y-%m-%d")
        raise ValueError(f"Missing regime features for origin dates: {missing.tolist()[:10]}")

    vol_threshold = float(origin_df["hist_vol_20d"].median())
    sent_threshold = float(max(0.05, origin_df["sent_score_3d_ma"].abs().median()))
    ret_threshold = float(max(0.0025, origin_df["ret_5d"].abs().median()))

    origin_df["volatility_regime"] = np.where(
        origin_df["hist_vol_20d"] >= vol_threshold,
        "high_vol",
        "low_vol",
    )

    def _sentiment_alignment(row: pd.Series) -> str:
        sent_value = float(row["sent_score_3d_ma"])
        ret_value = float(row["ret_5d"])
        if abs(sent_value) < sent_threshold or abs(ret_value) < ret_threshold:
            return "neutral"
        if np.sign(sent_value) == np.sign(ret_value):
            return "aligned"
        return "divergent"

    origin_df["sentiment_alignment_5d"] = origin_df.apply(_sentiment_alignment, axis=1)
    origin_df["trend_vol_regime"] = (
        origin_df["trend_regime_20d"].astype(str) + "__" + origin_df["volatility_regime"].astype(str)
    )

    official_event_abs_median = np.nan
    official_supply_abs_median = np.nan
    if official_event_path is not None:
        official_df, official_thresholds = build_official_origin_feature_frame(
            origin_dates,
            official_event_path,
        )
        official_df["date"] = pd.to_datetime(official_df["date"]).dt.normalize()
        origin_df = origin_df.merge(official_df, on="date", how="left")
        origin_df[[c for c in official_df.columns if c != "date"]] = origin_df[
            [c for c in official_df.columns if c != "date"]
        ].fillna(0.0)
        official_event_abs_median = float(official_thresholds.official_event_abs_median)
        official_supply_abs_median = float(official_thresholds.official_supply_abs_median)

        def _official_alignment(row: pd.Series) -> str:
            official_value = float(row["official_event_net_30d"])
            ret_value = float(row["ret_5d"])
            if abs(official_value) < official_event_abs_median or abs(ret_value) < ret_threshold:
                return "neutral"
            if np.sign(official_value) == np.sign(ret_value):
                return "aligned"
            return "divergent"

        origin_df["official_event_alignment_5d"] = origin_df.apply(_official_alignment, axis=1)

    thresholds = RegimeThresholds(
        volatility_median=vol_threshold,
        sentiment_abs_median=sent_threshold,
        return_abs_median=ret_threshold,
        official_event_abs_median=official_event_abs_median,
        official_supply_abs_median=official_supply_abs_median,
        trend_threshold_multiplier=float(trend_threshold_multiplier),
        volatility_window=int(volatility_window),
        trend_window=int(trend_window),
        return_window=int(return_window),
        sentiment_window=int(sentiment_window),
        sentiment_fill_limit=int(sentiment_fill_limit),
    )
    return origin_df, thresholds


def build_sample_error_frame(
    dates: Sequence[pd.Timestamp],
    y_true: np.ndarray,
    predictions: Mapping[str, np.ndarray],
) -> pd.DataFrame:
    """Per-origin path MSE and winner summary."""
    frame = pd.DataFrame({"date": pd.to_datetime(dates)})
    frame["date"] = frame["date"].dt.normalize()
    model_names = list(predictions.keys())
    for model_name, y_pred in predictions.items():
        frame[f"{model_name}_path_mse"] = sample_path_mse(y_true, y_pred)

    mse_matrix = frame[[f"{name}_path_mse" for name in model_names]].to_numpy(dtype=float)
    best_idx = np.argmin(mse_matrix, axis=1)
    sorted_mse = np.sort(mse_matrix, axis=1)
    frame["best_model"] = [model_names[idx] for idx in best_idx]
    frame["winner_margin_to_second"] = sorted_mse[:, 1] - sorted_mse[:, 0]
    return frame


def regime_metric_tables(
    origin_frame: pd.DataFrame,
    y_true: np.ndarray,
    predictions: Mapping[str, np.ndarray],
    *,
    regime_columns: Sequence[str],
    horizons: Sequence[int] = HORIZONS,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Aggregate path and horizon metrics by regime label."""
    path_rows: List[Dict[str, float]] = []
    horizon_rows: List[Dict[str, float]] = []
    winner_rows: List[Dict[str, float]] = []

    sample_errors = build_sample_error_frame(origin_frame["date"], y_true, predictions)
    sample_with_regimes = origin_frame.merge(sample_errors, on="date", how="left")

    for regime_col in regime_columns:
        for regime_value, subset in sample_with_regimes.groupby(regime_col, dropna=False):
            idx = subset.index.to_numpy(dtype=int)
            n_samples = int(len(idx))
            if n_samples == 0:
                continue

            for model_name, y_pred in predictions.items():
                path_rows.append(
                    {
                        "regime_family": regime_col,
                        "regime_value": regime_value,
                        "model": model_name,
                        "n_samples": n_samples,
                        "path_mse": float(sample_path_mse(y_true[idx], y_pred[idx]).mean()),
                    }
                )
                for h in horizons:
                    horizon_rows.append(
                        {
                            "regime_family": regime_col,
                            "regime_value": regime_value,
                            "model": model_name,
                            "horizon": int(h),
                            "n_samples": n_samples,
                            "mse": float(np.mean((y_true[idx, h - 1] - y_pred[idx, h - 1]) ** 2)),
                        }
                    )

            winner_counts = subset["best_model"].value_counts()
            margin_means = subset.groupby("best_model")["winner_margin_to_second"].mean()
            for model_name, count in winner_counts.items():
                winner_rows.append(
                    {
                        "regime_family": regime_col,
                        "regime_value": regime_value,
                        "model": model_name,
                        "winner_count": int(count),
                        "winner_share": float(count / n_samples),
                        "avg_margin_to_second": float(margin_means.get(model_name, np.nan)),
                        "n_samples": n_samples,
                    }
                )

    return (
        pd.DataFrame(path_rows).sort_values(["regime_family", "regime_value", "path_mse"]).reset_index(drop=True),
        pd.DataFrame(horizon_rows).sort_values(
            ["regime_family", "regime_value", "horizon", "mse"]
        ).reset_index(drop=True),
        pd.DataFrame(winner_rows).sort_values(
            ["regime_family", "regime_value", "winner_share"], ascending=[True, True, False]
        ).reset_index(drop=True),
    )


def gate_candidate_summary(
    regime_path_df: pd.DataFrame,
    *,
    min_samples_per_bucket: int = 25,
    min_margin: float = 0.15,
) -> Dict[str, object]:
    """Summarize whether any regime family is a plausible gating candidate."""
    summary: Dict[str, object] = {"families": []}
    for regime_family, family_df in regime_path_df.groupby("regime_family"):
        family_entry: Dict[str, object] = {
            "regime_family": regime_family,
            "values": [],
            "candidate": False,
        }
        winners = set()
        viable_values = 0
        for regime_value, bucket_df in family_df.groupby("regime_value"):
            ordered = bucket_df.sort_values("path_mse").reset_index(drop=True)
            best = ordered.iloc[0]
            runner_up = ordered.iloc[1] if len(ordered) > 1 else None
            margin = float(runner_up["path_mse"] - best["path_mse"]) if runner_up is not None else np.nan
            value_entry = {
                "regime_value": regime_value,
                "best_model": str(best["model"]),
                "best_path_mse": float(best["path_mse"]),
                "n_samples": int(best["n_samples"]),
                "runner_up_model": str(runner_up["model"]) if runner_up is not None else None,
                "margin_to_runner_up": margin,
            }
            family_entry["values"].append(value_entry)
            winners.add(str(best["model"]))
            if int(best["n_samples"]) >= min_samples_per_bucket and np.isfinite(margin) and margin >= min_margin:
                viable_values += 1

        family_entry["distinct_winners"] = sorted(winners)
        family_entry["candidate"] = len(winners) > 1 and viable_values >= 2
        summary["families"].append(family_entry)

    summary["recommended_families"] = [
        entry["regime_family"] for entry in summary["families"] if entry["candidate"]
    ]
    return summary


def oracle_gate_upper_bounds(
    origin_frame: pd.DataFrame,
    y_true: np.ndarray,
    predictions: Mapping[str, np.ndarray],
    *,
    regime_columns: Sequence[str],
) -> pd.DataFrame:
    """Ex post upper bound from selecting the best model per regime bucket on the same sample."""
    sample_errors = build_sample_error_frame(origin_frame["date"], y_true, predictions)
    merged = origin_frame.merge(sample_errors, on="date", how="left")
    rows: List[Dict[str, object]] = []
    model_error_cols = {name: f"{name}_path_mse" for name in predictions.keys()}

    for regime_col in regime_columns:
        chosen_errors: List[float] = []
        bucket_rows: List[Dict[str, object]] = []
        for regime_value, subset in merged.groupby(regime_col, dropna=False):
            bucket_means = {
                model_name: float(subset[col].mean()) for model_name, col in model_error_cols.items()
            }
            best_model = min(bucket_means.items(), key=lambda item: item[1])[0]
            chosen_errors.extend(subset[model_error_cols[best_model]].tolist())
            bucket_rows.append(
                {
                    "regime_family": regime_col,
                    "regime_value": regime_value,
                    "selected_model": best_model,
                    "bucket_path_mse": bucket_means[best_model],
                    "n_samples": int(len(subset)),
                }
            )
        rows.append(
            {
                "regime_family": regime_col,
                "oracle_path_mse": float(np.mean(chosen_errors)),
                "n_buckets": int(len(bucket_rows)),
                "bucket_models": bucket_rows,
            }
        )
    return pd.DataFrame(rows).sort_values("oracle_path_mse").reset_index(drop=True)


def apply_fixed_gate(
    origin_frame: pd.DataFrame,
    predictions: Mapping[str, np.ndarray],
    *,
    regime_column: str,
    model_by_regime: Mapping[str, str],
    fallback_model: str,
    strategy_name: str | None = None,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Select per-origin predictions from a fixed regime-to-model mapping."""
    model_names = set(predictions.keys())
    if fallback_model not in model_names:
        raise ValueError(f"Fallback model '{fallback_model}' not found in predictions.")
    unknown_targets = sorted(set(model_by_regime.values()) - model_names)
    if unknown_targets:
        raise ValueError(f"Gate references unknown models: {unknown_targets}")

    selections: List[str] = []
    chosen_rows: List[np.ndarray] = []
    for idx, regime_value in enumerate(origin_frame[regime_column].tolist()):
        chosen_model = model_by_regime.get(regime_value, fallback_model)
        selections.append(chosen_model)
        chosen_rows.append(np.asarray(predictions[chosen_model][idx], dtype=float))

    selected_pred = np.vstack(chosen_rows)
    selection_df = origin_frame[["date", regime_column]].copy()
    selection_df["selected_model"] = selections
    selection_df["strategy_name"] = strategy_name or f"gate::{regime_column}"
    return selected_pred, selection_df
