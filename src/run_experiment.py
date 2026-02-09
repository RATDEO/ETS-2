"""
Main experiment runner for EU ETS forecasting.

This script orchestrates the entire pipeline:
1. Load and standardize data
2. FX convert USD→EUR
3. Build panel + windows
4. Train baselines + TSM
5. Generate TSM predictions
6. Run LLM refinement methods
7. Evaluate + significance tests
8. Robustness suite
9. Drift/return + long-short evaluation
10. Trend classification
11. Write paper

Usage:
    python -m src.run_experiment --config src/config/default.yaml
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional
import logging
import json
import copy
import re
import numpy as np
import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from config import load_config
from data import build_panel
from data.windows import make_windows, split_windows, StandardScaler, save_datasets, WindowConfig
from data.panel import get_coverage_report, plot_coverage_heatmap
from models.baselines import NaivePersistence, SeasonalNaive, LinearBaseline
from eval.metrics import compute_metrics_by_horizon, compute_path_metrics, get_per_sample_errors
from eval.return_metrics import (
    compute_drift_metrics_by_horizon,
    compute_long_short_portfolio_metrics
)
from eval.trend_classification import compute_paper_trend_accuracy, compute_trend_accuracy
from eval.significance import compare_all_models
from eval.quantile_metrics import compute_quantile_metrics
from models.quantile_lasso import QuantileLasso, QuantileLassoConfig
from paper import PaperWriter
from utils import setup_logging, set_seed

logger = logging.getLogger(__name__)


def returns_to_prices(returns: np.ndarray, last_price: np.ndarray) -> np.ndarray:
    """Convert log returns to price paths."""
    last_price = np.asarray(last_price).reshape(-1, 1)
    return last_price * np.exp(np.cumsum(returns, axis=1))


def build_exogenous_summary(
    window: np.ndarray,
    feature_cols: list,
    target_col: str,
    max_features: int = 6
) -> dict:
    """Summarize recent exogenous signals for LLM prompts."""
    summary = {}
    eps = 1e-8
    feature_idx = {name: i for i, name in enumerate(feature_cols)}
    preferred = [
        "is_auction_day",
        "auction_volume",
        "vstoxx",
        "brent_return",
        "coal_return",
    ]
    preferred += [name for name in feature_cols if name.startswith("idx_")]
    
    ordered = [name for name in preferred if name in feature_idx and name != target_col]
    for name in feature_cols:
        if name != target_col and name not in ordered:
            ordered.append(name)
    
    for name in ordered:
        idx = feature_idx.get(name)
        if idx is None:
            continue
        series = window[:, idx]
        if series.size == 0:
            continue
        last = float(series[-1])
        if series.size >= 5:
            denom = series[-5] if abs(series[-5]) > eps else eps
            change_5d = (series[-1] / denom - 1.0) * 100.0
        else:
            denom = series[0] if abs(series[0]) > eps else eps
            change_5d = (series[-1] / denom - 1.0) * 100.0
        if series.size >= 20:
            denom_20 = series[-20] if abs(series[-20]) > eps else eps
            change_20d = (series[-1] / denom_20 - 1.0) * 100.0
        else:
            denom_20 = series[0] if abs(series[0]) > eps else eps
            change_20d = (series[-1] / denom_20 - 1.0) * 100.0
        vol_20 = float(np.std(series[-20:])) if series.size >= 20 else float(np.std(series))
        
        unique_recent = np.unique(series[-20:]) if series.size >= 20 else np.unique(series)
        if unique_recent.size <= 2 and set(unique_recent.tolist()).issubset({0.0, 1.0}):
            mean_20 = float(np.mean(series[-20:])) if series.size >= 20 else float(np.mean(series))
            summary[name] = f"last={int(round(last))}, mean20={mean_20:.2f}"
        else:
            summary[name] = (
                f"last={last:.3f}, 5d={change_5d:.2f}%, "
                f"20d={change_20d:.2f}%, vol20={vol_20:.3f}"
            )
        
        if len(summary) >= max_features:
            break
    return summary


def compute_history_volatility(history: np.ndarray, window: int = 20) -> float:
    """Compute volatility of log returns for gating LLM refinement."""
    series = history[-window:] if len(history) >= window else history
    if len(series) < 2:
        return 0.0
    eps = 1e-8
    returns = np.diff(np.log(np.clip(series, eps, None)))
    if len(returns) == 0:
        return 0.0
    return float(np.std(returns))


def load_daily_sentiment(
    path: str,
    date_col: str = "date",
    score_col: str = "sent_score"
) -> dict:
    """Load daily sentiment series as a date->score mapping."""
    sentiment_path = Path(path)
    if not sentiment_path.exists():
        logger.warning("Sentiment file not found: %s", sentiment_path)
        return {}

    df = pd.read_csv(sentiment_path)
    if date_col not in df.columns:
        fallback = "seendate" if "seendate" in df.columns else None
        if fallback:
            date_col = fallback
        else:
            logger.warning("Sentiment date column not found in %s", sentiment_path)
            return {}

    if score_col not in df.columns:
        logger.warning("Sentiment score column not found in %s", sentiment_path)
        return {}

    df[date_col] = pd.to_datetime(df[date_col]).dt.normalize()
    grouped = df.groupby(date_col, as_index=False)[score_col].mean()
    return {
        pd.Timestamp(row[date_col]): float(row[score_col])
        for _, row in grouped.iterrows()
    }


def build_history_dates(
    panel_dates: np.ndarray,
    date_to_idx: dict,
    pred_dates: np.ndarray,
    history_points: int
) -> list:
    """Build aligned history date arrays for each prediction date."""
    date_arrays = []
    for pred_date in pred_dates:
        pred_date = pd.Timestamp(pred_date)
        idx = date_to_idx.get(pred_date)
        if idx is None:
            history_dates = panel_dates[:history_points]
        else:
            start_idx = max(0, idx - history_points)
            history_dates = panel_dates[start_idx:idx]
        date_arrays.append([str(d) for d in history_dates])
    return date_arrays


def build_sentiment_histories(
    date_arrays: list,
    sentiment_map: dict,
    sentiment_points: int
) -> list:
    """Build sentiment history arrays aligned to history dates."""
    histories = []
    for history_dates in date_arrays:
        values = [
            float(sentiment_map.get(pd.Timestamp(d).normalize(), 0.0))
            for d in history_dates
        ]
        if sentiment_points > 0:
            values = values[-sentiment_points:]
            if len(values) < sentiment_points:
                values = [0.0] * (sentiment_points - len(values)) + values
        histories.append(np.array(values, dtype=float))
    return histories


def fit_sentiment_calibration(
    histories: np.ndarray,
    future: np.ndarray,
    sentiment_histories: list,
    sentiment_window: int
) -> dict:
    """Fit a linear mapping from sentiment score to next-day log return."""
    scores = []
    returns = []
    for idx in range(len(histories)):
        hist = histories[idx]
        fut = future[idx]
        if len(hist) == 0 or len(fut) == 0:
            continue
        base = float(hist[-1])
        target = float(fut[0])
        if base <= 0 or target <= 0:
            continue
        sent_hist = sentiment_histories[idx]
        if sentiment_window > 0:
            score = float(np.mean(sent_hist[-sentiment_window:]))
        else:
            score = float(np.mean(sent_hist)) if len(sent_hist) else 0.0
        scores.append(score)
        returns.append(float(np.log(target / base)))

    if len(scores) < 2:
        return {
            "alpha": 0.0,
            "beta": 0.0,
            "n": len(scores),
            "score_mean": float(np.mean(scores)) if scores else 0.0,
            "score_std": float(np.std(scores)) if scores else 0.0,
            "ret_std": float(np.std(returns)) if returns else 0.0,
        }

    s = np.array(scores, dtype=float)
    r = np.array(returns, dtype=float)
    var = float(np.var(s))
    if var < 1e-12:
        beta = 0.0
        alpha = float(np.mean(r))
    else:
        cov = float(np.mean((s - np.mean(s)) * (r - np.mean(r))))
        beta = cov / var
        alpha = float(np.mean(r) - beta * np.mean(s))

    return {
        "alpha": alpha,
        "beta": beta,
        "n": len(scores),
        "score_mean": float(np.mean(s)),
        "score_std": float(np.std(s)),
        "ret_std": float(np.std(r)),
    }


def fit_blend_weights(
    y_true: np.ndarray,
    base_pred: np.ndarray,
    llm_pred: np.ndarray,
    horizons: list,
    min_weight: float = 0.0,
    max_weight: float = 1.0
) -> dict:
    """Fit per-horizon blend weights to minimize MSE on validation data."""
    weights = {}
    for h in horizons:
        idx = h - 1
        y = y_true[:, idx]
        base = base_pred[:, idx]
        llm = llm_pred[:, idx]
        diff = llm - base
        denom = float(np.sum(diff * diff))
        if denom < 1e-12:
            weight = 0.0
        else:
            weight = float(np.sum(diff * (y - base)) / denom)
        weight = float(np.clip(weight, min_weight, max_weight))
        weights[h] = weight
    return weights


def apply_blend_weights(
    base_pred: np.ndarray,
    llm_pred: np.ndarray,
    weights: dict,
    pred_len: int
) -> np.ndarray:
    """Apply per-horizon blend weights to LLM forecasts."""
    blended = base_pred.copy()
    for h, weight in weights.items():
        idx = h - 1
        if 0 <= idx < pred_len:
            blended[:, idx] = base_pred[:, idx] + weight * (llm_pred[:, idx] - base_pred[:, idx])
    return blended


def _parse_float_list(value, default=None) -> list:
    """Parse a YAML value that may be a list[float] or a comma-separated string."""
    if value is None:
        return list(default) if default is not None else []
    if isinstance(value, (list, tuple)):
        return [float(x) for x in value]
    if isinstance(value, str):
        parts = [p.strip() for p in value.split(",") if p.strip()]
        return [float(p) for p in parts]
    return [float(value)]


def blend_forecasts(
    base_pred: np.ndarray,
    llm_pred: np.ndarray,
    strength: float,
    schedule: str,
    pred_len: int,
    min_weight: float = 0.0,
    power: float = 1.0,
) -> np.ndarray:
    """Blend base and LLM forecasts with either a uniform weight or a horizon ramp."""
    strength = float(strength)
    schedule = str(schedule or "uniform").lower()
    if schedule == "uniform":
        return base_pred + strength * (llm_pred - base_pred)
    if schedule != "ramp":
        raise ValueError(f"Unknown blend schedule: {schedule}")
    w = np.linspace(float(min_weight), strength, int(pred_len))
    if float(power) != 1.0:
        w = np.power(w, float(power))
    return base_pred * (1.0 - w) + llm_pred * w


def mse_by_horizon(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Compute MSE at each forecast step (vector length pred_len)."""
    return np.mean((y_true - y_pred) ** 2, axis=0)


def select_eval_indices(
    eligible_idx: np.ndarray,
    max_samples: int,
    strategy: str = "first",
    seed: int = 42,
) -> np.ndarray:
    """Select evaluation indices from eligible indices."""
    eligible_idx = np.asarray(eligible_idx, dtype=int)
    if eligible_idx.size == 0 or max_samples <= 0:
        return np.array([], dtype=int)

    max_samples = int(min(max_samples, eligible_idx.size))
    strategy = str(strategy or "first").lower()

    if strategy == "first":
        chosen = eligible_idx[:max_samples]
        return np.asarray(chosen, dtype=int)

    if strategy == "random":
        rng = np.random.default_rng(int(seed))
        chosen = rng.choice(eligible_idx, size=max_samples, replace=False)
        return np.sort(np.asarray(chosen, dtype=int))

    if strategy == "spaced":
        if max_samples == 1:
            return np.array([int(eligible_idx[-1])], dtype=int)
        pos = np.linspace(0, eligible_idx.size - 1, max_samples)
        chosen = eligible_idx[np.round(pos).astype(int)]
        return np.asarray(chosen, dtype=int)

    raise ValueError(f"Unknown subset strategy: {strategy}")


def evaluate_blend_grid(
    y_true: np.ndarray,
    base_pred: np.ndarray,
    llm_pred: np.ndarray,
    weights: list,
    schedule: str,
    key_horizons: list,
    min_weight: float = 0.0,
    power: float = 1.0,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Evaluate blend strengths over a grid, returning:
      - summary_df: one row per strength with MSE_path + selected horizon MSEs
      - mse_by_h_df: long form (w, horizon, mse) for all horizons 1..pred_len
      - best_by_h_df: best w per horizon (min mse) for each horizon 1..pred_len
    """
    pred_len = int(base_pred.shape[1])
    rows = []
    mse_rows = []
    for w in weights:
        blended = blend_forecasts(
            base_pred=base_pred,
            llm_pred=llm_pred,
            strength=float(w),
            schedule=schedule,
            pred_len=pred_len,
            min_weight=min_weight,
            power=power,
        )
        mse_h = mse_by_horizon(y_true, blended)
        row = {"w": float(w), "mse_path": float(np.mean(mse_h))}
        for h in key_horizons:
            idx = int(h) - 1
            if 0 <= idx < pred_len:
                row[f"h{int(h)}_mse"] = float(mse_h[idx])
        rows.append(row)
        for h in range(1, pred_len + 1):
            mse_rows.append({"w": float(w), "horizon": int(h), "mse": float(mse_h[h - 1])})

    summary_df = pd.DataFrame(rows).sort_values("w").reset_index(drop=True)
    mse_by_h_df = pd.DataFrame(mse_rows).sort_values(["horizon", "w"]).reset_index(drop=True)
    best_by_h_df = (
        mse_by_h_df.loc[mse_by_h_df.groupby("horizon")["mse"].idxmin()]
        .sort_values("horizon")
        .reset_index(drop=True)
    )
    return summary_df, mse_by_h_df, best_by_h_df


def save_blend_grid_artifacts(
    summary_df: pd.DataFrame,
    mse_by_h_df: pd.DataFrame,
    best_by_h_df: pd.DataFrame,
    out_dir: Path,
) -> None:
    """Save blend grid CSVs and plots to out_dir."""
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(out_dir / "blend_grid_summary.csv", index=False)
    mse_by_h_df.to_csv(out_dir / "blend_grid_mse_by_horizon.csv", index=False)
    best_by_h_df.to_csv(out_dir / "blend_grid_best_w_by_horizon.csv", index=False)

    try:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(12, 6))
        for w, grp in mse_by_h_df.groupby("w", sort=True):
            ax.plot(grp["horizon"], grp["mse"], label=f"w={w:g}", linewidth=2)
        ax.set_title("Blend Grid — MSE by Horizon (TSM + w·(LLM−TSM))")
        ax.set_xlabel("Horizon (days ahead)")
        ax.set_ylabel("MSE (EUR^2)")
        ax.grid(True, alpha=0.3)
        ax.legend(ncol=3, fontsize=9)
        fig.tight_layout()
        fig.savefig(out_dir / "blend_grid_mse_by_horizon.png", dpi=200)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(12, 3.5))
        ax.step(best_by_h_df["horizon"], best_by_h_df["w"], where="mid", linewidth=2)
        ax.set_title("Blend Grid — Best w by Horizon (min MSE at each horizon)")
        ax.set_xlabel("Horizon (days ahead)")
        ax.set_ylabel("Best w")
        ax.set_yticks(sorted(best_by_h_df["w"].unique()))
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_dir / "blend_grid_best_w_by_horizon.png", dpi=200)
        plt.close(fig)
    except Exception as e:
        logger.warning("Failed to save blend grid plots: %s", e)


def select_features_via_lasso(
    X_enc_train: np.ndarray,
    y_train_future: np.ndarray,
    feature_cols: list,
    target_col: str,
    config: dict,
) -> list:
    """
    Select features using LASSO on summary stats to predict aggregated horizons.
    """
    lasso_cfg = config.get("econometric", {}).get("lasso_feature_selection", {})
    if not lasso_cfg.get("enabled", False):
        return feature_cols

    try:
        from sklearn.linear_model import Lasso, ElasticNet
        from sklearn.preprocessing import StandardScaler as SklearnScaler
        from sklearn.feature_selection import mutual_info_regression
    except ImportError as e:
        logger.warning(f"LASSO feature selection skipped (sklearn missing): {e}")
        return feature_cols

    if X_enc_train.ndim != 3:
        logger.warning("LASSO feature selection skipped (unexpected X_enc shape)")
        return feature_cols

    stats = lasso_cfg.get("feature_stats", ["last", "change_5", "change_20", "std_20"])
    target_horizons = lasso_cfg.get("target_horizons", [1, 5, 20, 30])
    target_aggregation = lasso_cfg.get("target_aggregation", "mean")
    mode = lasso_cfg.get("mode", "aggregate")
    selection_method = lasso_cfg.get("selection_method", "lasso")

    X_summary = []
    feature_map = []
    for idx, name in enumerate(feature_cols):
        series = X_enc_train[:, :, idx]
        if "last" in stats:
            X_summary.append(series[:, -1])
            feature_map.append(name)
        if "change_5" in stats and series.shape[1] >= 5:
            denom = np.where(np.abs(series[:, -5]) > 1e-8, series[:, -5], 1e-8)
            X_summary.append(series[:, -1] / denom - 1.0)
            feature_map.append(name)
        if "change_20" in stats and series.shape[1] >= 20:
            denom = np.where(np.abs(series[:, -20]) > 1e-8, series[:, -20], 1e-8)
            X_summary.append(series[:, -1] / denom - 1.0)
            feature_map.append(name)
        if "std_20" in stats:
            window = min(20, series.shape[1])
            X_summary.append(np.std(series[:, -window:], axis=1))
            feature_map.append(name)

    if not X_summary:
        logger.warning("LASSO feature selection skipped (no summary features)")
        return feature_cols

    X_mat = np.column_stack(X_summary).astype(np.float32)

    scaler = SklearnScaler()
    X_scaled = scaler.fit_transform(X_mat)

    alpha = float(lasso_cfg.get("alpha", 0.01))
    max_iter = int(lasso_cfg.get("max_iter", 10000))
    coef_threshold = float(lasso_cfg.get("coef_threshold", 1.0e-6))
    min_features = int(lasso_cfg.get("min_features", 4))
    max_features = int(lasso_cfg.get("max_features", len(feature_cols)))

    horizon_indices = [h - 1 for h in target_horizons if 1 <= h <= y_train_future.shape[1]]
    if not horizon_indices:
        horizon_indices = [0]

    def score_features(y_target: np.ndarray, alpha_value: float) -> np.ndarray:
        if selection_method == "stability":
            n_boot = int(lasso_cfg.get("stability_bootstraps", 100))
            sample_frac = float(lasso_cfg.get("stability_sample_frac", 0.8))
            coef_eps = float(lasso_cfg.get("stability_coef_threshold", coef_threshold))
            random_state = lasso_cfg.get("stability_random_state", None)
            rng = np.random.default_rng(random_state)
            n_samples = X_scaled.shape[0]
            draw_size = max(2, int(round(sample_frac * n_samples)))
            counts = np.zeros(X_scaled.shape[1], dtype=np.float32)
            for _ in range(n_boot):
                idx = rng.choice(n_samples, size=draw_size, replace=True)
                model = Lasso(alpha=alpha_value, max_iter=max_iter)
                model.fit(X_scaled[idx], y_target[idx])
                counts += np.abs(model.coef_) > coef_eps
            return counts / max(n_boot, 1)
        if selection_method == "mutual_info":
            n_neighbors = int(lasso_cfg.get("mi_neighbors", 3))
            random_state = lasso_cfg.get("mi_random_state", None)
            return mutual_info_regression(
                X_scaled, y_target, n_neighbors=n_neighbors, random_state=random_state
            )
        if selection_method == "elastic_net":
            l1_ratio = float(lasso_cfg.get("elastic_net_l1_ratio", 0.7))
            model = ElasticNet(alpha=alpha_value, l1_ratio=l1_ratio, max_iter=max_iter)
        else:
            model = Lasso(alpha=alpha_value, max_iter=max_iter)
        model.fit(X_scaled, y_target)
        return np.abs(model.coef_)

    scores = {}
    if mode == "per_horizon_union":
        for h_idx in horizon_indices:
            if len(target_horizons) == 1:
                alpha = float(
                    lasso_cfg.get("alpha_by_horizon", {}).get(str(target_horizons[0]), alpha)
                )
            else:
                alpha = float(
                    lasso_cfg.get("alpha_by_horizon", {}).get(str(h_idx + 1), alpha)
                )
            y_target = y_train_future[:, h_idx]
            coef_scores = score_features(y_target, alpha)
            for base_name, c in zip(feature_map, coef_scores):
                scores[base_name] = scores.get(base_name, 0.0) + abs(c)
    else:
        y_subset = y_train_future[:, horizon_indices]
        if mode == "weighted":
            weights = lasso_cfg.get("target_weights", [1.0] * len(horizon_indices))
            if len(weights) != len(horizon_indices):
                weights = [1.0] * len(horizon_indices)
            weights = np.asarray(weights, dtype=np.float32)
            y_target = np.dot(y_subset, weights) / max(np.sum(weights), 1e-8)
        else:
            if target_aggregation == "median":
                y_target = np.median(y_subset, axis=1)
            else:
                y_target = np.mean(y_subset, axis=1)
        if len(target_horizons) == 1:
            alpha = float(
                lasso_cfg.get("alpha_by_horizon", {}).get(str(target_horizons[0]), alpha)
            )
        coef_scores = score_features(y_target, alpha)
        for base_name, c in zip(feature_map, coef_scores):
            scores[base_name] = scores.get(base_name, 0.0) + abs(c)

    selected = [name for name, score in scores.items() if score > coef_threshold]
    if target_col not in selected:
        selected.insert(0, target_col)

    if len(selected) < min_features:
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        for name, _ in ranked:
            if name not in selected:
                selected.append(name)
            if len(selected) >= min_features:
                break

    drop_weakest = int(lasso_cfg.get("drop_weakest", 0))
    if drop_weakest > 0 and len(scores) > drop_weakest:
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        kept = [name for name, _ in ranked[:-drop_weakest]]
        selected = [name for name in selected if name in kept]
        if target_col not in selected:
            selected.insert(0, target_col)

    if len(selected) > max_features:
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        selected = []
        for name, _ in ranked:
            if name == target_col or len(selected) < max_features:
                if name not in selected:
                    selected.append(name)
            if len(selected) >= max_features:
                break
        if target_col not in selected:
            selected.insert(0, target_col)

    # Ensure target is first
    if target_col in selected:
        selected = [target_col] + [name for name in selected if name != target_col]

    return selected


def run_experiment(config_path: str = None, overrides: dict = None):
    """
    Run the full experiment pipeline.
    
    Args:
        config_path: Path to config YAML file
        overrides: Config overrides
    """
    # =========================================================================
    # Step 1: Load configuration
    # =========================================================================
    logger.info("=" * 70)
    logger.info("EU ETS FUTURES FORECASTING EXPERIMENT")
    logger.info("=" * 70)
    
    config = load_config(config_path, overrides)
    run_dir = config.setup_run_dir()
    
    # Setup logging to run directory
    setup_logging(run_dir)
    
    logger.info(f"Run ID: {config.run_id}")
    logger.info(f"Run directory: {run_dir}")
    
    # Set random seed
    set_seed(config.seed)
    
    # Save config
    config.save()
    config.save_reproducibility_info()
    
    # =========================================================================
    # Step 2: Build data panel
    # =========================================================================
    logger.info("\n" + "=" * 70)
    logger.info("STEP 2: Building data panel")
    logger.info("=" * 70)
    
    data_dir = Path(__file__).parent.parent / "Data"
    
    panel, schema = build_panel(
        data_dir=data_dir,
        config=config.raw,
        save_path=run_dir / "data"
    )
    
    # Generate coverage report
    coverage = get_coverage_report(panel)
    coverage.to_csv(run_dir / "data" / "coverage_report.csv", index=False)
    
    try:
        plot_coverage_heatmap(panel, run_dir / "paper_snapshot" / "fig_data_coverage.png")
    except Exception as e:
        logger.warning(f"Failed to create coverage plot: {e}")
    
    # =========================================================================
    # Step 3: Create windows and split data
    # =========================================================================
    logger.info("\n" + "=" * 70)
    logger.info("STEP 3: Creating windows and splitting data")
    logger.info("=" * 70)
    
    target_mode = config.target.get("mode", "price")
    target_col = "y_return" if target_mode == "returns" else "y"
    
    feature_candidates = [c for c in panel.columns if c not in ["date", target_col]]
    if "y" in feature_candidates:
        feature_candidates.remove("y")
        feature_candidates = ["y"] + feature_candidates
    
    feature_cols = [target_col] + feature_candidates[:10]
    
    window_config = WindowConfig(
        seq_len=config.seq_len,
        label_len=config.label_len,
        pred_len=config.pred_len,
        target_col=target_col,
        feature_cols=feature_cols
    )
    
    price_idx = feature_cols.index("y") if "y" in feature_cols else None
    if target_mode == "returns" and price_idx is None:
        raise ValueError("Returns target requires price feature 'y' in window features.")
    
    X_enc, X_dec, y, dates = make_windows(panel, window_config, mode='MS')
    
    # Split data
    splits = split_windows(
        X_enc, X_dec, y, dates,
        train_end=config.split.get('train_end', '2022-12-31'),
        val_end=config.split.get('val_end', '2023-12-31')
    )
    
    # Fit scaler on training data
    scaler = StandardScaler()
    scaler.fit(splits['train']['X_enc'])
    
    # Save datasets
    save_datasets(splits, scaler, run_dir / "data" / "datasets")
    
    # =========================================================================
    # Step 4: Train and evaluate baselines
    # =========================================================================
    logger.info("\n" + "=" * 70)
    logger.info("STEP 4: Training and evaluating baselines")
    logger.info("=" * 70)
    
    target_is_returns = target_mode == "returns"
    
    if target_is_returns:
        y_train_hist = splits['train']['X_enc'][:, :, price_idx]
        y_val_hist = splits['val']['X_enc'][:, :, price_idx]
        y_test_hist = splits['test']['X_enc'][:, :, price_idx]
        y_train_base = y_train_hist[:, -1]
        y_val_base = y_val_hist[:, -1]
        y_test_base = y_test_hist[:, -1]
        y_train_future_returns = splits['train']['y']
        y_val_future_returns = splits['val']['y']
        y_test_future_returns = splits['test']['y']
        y_train_future = returns_to_prices(y_train_future_returns, y_train_base)
        y_val_future = returns_to_prices(y_val_future_returns, y_val_base)
        y_test_future = returns_to_prices(y_test_future_returns, y_test_base)
    else:
        y_train_hist = splits['train']['X_enc'][:, :, 0]  # Target is first feature
        y_val_hist = splits['val']['X_enc'][:, :, 0]
        y_test_hist = splits['test']['X_enc'][:, :, 0]
        y_train_future = splits['train']['y']
        y_val_future = splits['val']['y']
        y_test_future = splits['test']['y']
        # Get base price (last known price for each sample)
        y_test_base = y_test_hist[:, -1]
    
    llm_histories = y_test_hist
    horizons = config.horizons
    
    # LASSO feature selection (optional, for TSM+LASSO)
    lasso_cfg = config.raw.get("econometric", {}).get("lasso_feature_selection", {})
    lasso_mode = lasso_cfg.get("mode", "aggregate")
    selected_feature_cols = feature_cols
    selected_feature_cols_by_horizon = {}

    if lasso_cfg.get("enabled", False):
        if lasso_mode == "per_horizon_models":
            for h in horizons:
                override = copy.deepcopy(config.raw)
                override["econometric"]["lasso_feature_selection"]["target_horizons"] = [h]
                override["econometric"]["lasso_feature_selection"]["mode"] = "aggregate"
                alpha_by_h = override["econometric"]["lasso_feature_selection"].get("alpha_by_horizon", {})
                if str(h) in alpha_by_h:
                    override["econometric"]["lasso_feature_selection"]["alpha"] = alpha_by_h[str(h)]
                selected_feature_cols_by_horizon[h] = select_features_via_lasso(
                    splits["train"]["X_enc"],
                    y_train_future,
                    feature_cols,
                    target_col,
                    override,
                )
        else:
            selected_feature_cols = select_features_via_lasso(
                splits["train"]["X_enc"],
                y_train_future,
                feature_cols,
                target_col,
                config.raw,
            )

        lasso_dir = run_dir / "results" / "lasso_feature_selection"
        lasso_dir.mkdir(parents=True, exist_ok=True)
        with open(lasso_dir / "selected_features.json", "w") as f:
            payload = {
                "selected_features": selected_feature_cols,
                "selected_features_by_horizon": selected_feature_cols_by_horizon,
                "total_features": feature_cols,
                "config": lasso_cfg,
            }
            json.dump(payload, f, indent=2)

        if lasso_mode == "per_horizon_models":
            logger.info(
                "LASSO selected horizon-specific features for %d horizons",
                len(selected_feature_cols_by_horizon)
            )
        else:
            logger.info(
                "LASSO selected %d/%d features for TSM+LASSO",
                len(selected_feature_cols),
                len(feature_cols)
            )

    # Evaluate baselines
    baseline_results = {}
    
    # Naive persistence
    naive = NaivePersistence(config.pred_len)
    naive.fit(y_train_hist.flatten())
    naive_pred = naive.predict(y_test_hist)
    
    baseline_results['naive_persistence'] = {
        'predictions': naive_pred,
        'metrics': compute_metrics_by_horizon(y_test_future, naive_pred, horizons),
        'path_metrics': compute_path_metrics(y_test_future, naive_pred),
        'errors': get_per_sample_errors(y_test_future, naive_pred, horizons)
    }
    
    # Seasonal naive
    seasonal = SeasonalNaive(config.pred_len, season_period=5)
    seasonal.fit(y_train_hist.flatten())
    seasonal_pred = seasonal.predict(y_test_hist)
    
    baseline_results['seasonal_naive'] = {
        'predictions': seasonal_pred,
        'metrics': compute_metrics_by_horizon(y_test_future, seasonal_pred, horizons),
        'path_metrics': compute_path_metrics(y_test_future, seasonal_pred),
        'errors': get_per_sample_errors(y_test_future, seasonal_pred, horizons)
    }

    # Linear baselines (ridge + lasso)
    try:
        linear_ridge = LinearBaseline(config.pred_len, model_type="ridge")
        linear_ridge.fit(y_train_hist, y_train_future)
        linear_ridge_pred = linear_ridge.predict(y_test_hist)
        baseline_results['linear_ridge'] = {
            'predictions': linear_ridge_pred,
            'metrics': compute_metrics_by_horizon(y_test_future, linear_ridge_pred, horizons),
            'path_metrics': compute_path_metrics(y_test_future, linear_ridge_pred),
            'errors': get_per_sample_errors(y_test_future, linear_ridge_pred, horizons)
        }
    except Exception as e:
        logger.warning(f"Linear ridge baseline failed: {e}")

    try:
        linear_lasso = LinearBaseline(config.pred_len, model_type="lasso")
        linear_lasso.fit(y_train_hist, y_train_future)
        linear_lasso_pred = linear_lasso.predict(y_test_hist)
        baseline_results['linear_lasso'] = {
            'predictions': linear_lasso_pred,
            'metrics': compute_metrics_by_horizon(y_test_future, linear_lasso_pred, horizons),
            'path_metrics': compute_path_metrics(y_test_future, linear_lasso_pred),
            'errors': get_per_sample_errors(y_test_future, linear_lasso_pred, horizons)
        }
    except Exception as e:
        logger.warning(f"Linear lasso baseline failed: {e}")
    
    # Print baseline results
    logger.info("\nBaseline Results (MSE by horizon):")
    for name, result in baseline_results.items():
        logger.info(f"\n{name}:")
        logger.info(result['metrics'])
        path_mse = (result.get("path_metrics") or {}).get("mse_path")
        if path_mse is not None:
            logger.info("Path MSE (avg over %dd): %.6f", config.pred_len, float(path_mse))
    
    # Save baseline results
    for name, result in baseline_results.items():
        result['metrics'].to_csv(run_dir / "results" / f"{name}_metrics.csv")
    
    # =========================================================================
    # Step 4b: Quantile LASSO (optional econometric track)
    # =========================================================================
    econometric_config = config.raw.get("econometric", {}).get("quantile_lasso", {})
    if econometric_config.get("enabled", False):
        logger.info("\n" + "=" * 70)
        logger.info("STEP 4B: Quantile LASSO (feature selection + quantiles)")
        logger.info("=" * 70)
        try:
            q_config = QuantileLassoConfig(
                quantiles=econometric_config.get("quantiles", [0.1, 0.5, 0.9]),
                alpha=econometric_config.get("alpha", 0.01),
                lags=econometric_config.get("lags", [1, 5, 10, 20]),
            )
            qlasso = QuantileLasso(q_config)
            qlasso.fit(
                splits["train"]["X_enc"],
                feature_cols,
                target_idx=feature_cols.index(target_col),
                y_future=y_train_future,
            )
            q_pred = qlasso.predict(splits["test"]["X_enc"])
            quantile_metrics = compute_quantile_metrics(
                y_test_future,
                q_pred,
                q_config.quantiles,
                horizons,
            )

            quantile_dir = run_dir / "results" / "quantile_lasso"
            quantile_dir.mkdir(parents=True, exist_ok=True)
            quantile_metrics.to_csv(quantile_dir / "quantile_metrics.csv", index=False)

            coef_map = qlasso.coefficients()
            coef_rows = []
            for q, coef_matrix in coef_map.items():
                for h_idx, coef in enumerate(coef_matrix):
                    for name, value in zip(qlasso.feature_names_, coef):
                        coef_rows.append(
                            {
                                "quantile": q,
                                "horizon": h_idx + 1,
                                "feature": name,
                                "coefficient": float(value),
                            }
                        )
            if coef_rows:
                pd.DataFrame(coef_rows).to_csv(
                    quantile_dir / "quantile_coefficients.csv", index=False
                )

            selected = qlasso.selected_features()
            with open(quantile_dir / "selected_features.json", "w") as f:
                json.dump(selected, f, indent=2)
        except Exception as e:
            logger.warning(f"Quantile LASSO failed: {e}")

    # =========================================================================
    # Step 5: TSM training (if PyTorch available)
    # =========================================================================
    logger.info("\n" + "=" * 70)
    logger.info("STEP 5: TSM model training")
    logger.info("=" * 70)
    
    tsm_pred_eval = None
    tsm_pred_returns = None
    
    try:
        import torch
        from torch.utils.data import DataLoader
        from models.tsm import TSMForecaster
        from data.windows import TimeSeriesDataset

        def _train_tsm(
            X_enc_train,
            X_dec_train,
            y_train,
            X_enc_val,
            X_dec_val,
            y_val,
            X_enc_test,
            X_dec_test,
            y_test,
            tsm_label: str,
            config_override: dict,
            return_artifacts: bool = False,
        ):
            local_scaler = StandardScaler()
            local_scaler.fit(X_enc_train)

            train_dataset = TimeSeriesDataset(
                local_scaler.transform(X_enc_train),
                local_scaler.transform(X_dec_train),
                (y_train - local_scaler.mean_[0]) / local_scaler.std_[0]
            )

            val_dataset = TimeSeriesDataset(
                local_scaler.transform(X_enc_val),
                local_scaler.transform(X_dec_val),
                (y_val - local_scaler.mean_[0]) / local_scaler.std_[0]
            )

            test_dataset = TimeSeriesDataset(
                local_scaler.transform(X_enc_test),
                local_scaler.transform(X_dec_test),
                (y_test - local_scaler.mean_[0]) / local_scaler.std_[0]
            )

            train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=32)
            test_loader = DataLoader(test_dataset, batch_size=32)

            tsm_model = TSMForecaster(config_override)
            tsm_model.fit(
                train_loader,
                val_loader,
                epochs=config.model.get('max_epochs', 100),
                patience=config.model.get('early_stopping_patience', 10),
                save_path=run_dir / "models" / f"{tsm_label}_checkpoint.pt"
            )

            tsm_pred_scaled, _ = tsm_model.predict(test_loader)
            tsm_pred = local_scaler.inverse_transform_target(tsm_pred_scaled)
            if return_artifacts:
                return {
                    "pred": tsm_pred,
                    "model": tsm_model,
                    "scaler": local_scaler,
                    "datasets": {
                        "train": train_dataset,
                        "val": val_dataset,
                        "test": test_dataset,
                    },
                }
            return tsm_pred
        
        # Update config with actual feature count
        tsm_config = copy.deepcopy(config.raw)
        tsm_config['model']['enc_in'] = splits['train']['X_enc'].shape[-1]
        tsm_config['model']['dec_in'] = splits['train']['X_dec'].shape[-1]

        tsm_artifacts = _train_tsm(
            splits['train']['X_enc'],
            splits['train']['X_dec'],
            splits['train']['y'],
            splits['val']['X_enc'],
            splits['val']['X_dec'],
            splits['val']['y'],
            splits['test']['X_enc'],
            splits['test']['X_dec'],
            splits['test']['y'],
            "tsm",
            tsm_config,
            return_artifacts=True,
        )

        tsm_pred = tsm_artifacts["pred"]
        tsm_model = tsm_artifacts["model"]
        tsm_scaler = tsm_artifacts["scaler"]
        tsm_datasets = tsm_artifacts["datasets"]

        train_loader_eval = DataLoader(tsm_datasets["train"], batch_size=32)
        val_loader_eval = DataLoader(tsm_datasets["val"], batch_size=32)

        tsm_pred_train_scaled, _ = tsm_model.predict(train_loader_eval)
        tsm_pred_val_scaled, _ = tsm_model.predict(val_loader_eval)

        tsm_pred_train = tsm_scaler.inverse_transform_target(tsm_pred_train_scaled)
        tsm_pred_val = tsm_scaler.inverse_transform_target(tsm_pred_val_scaled)
        
        if target_is_returns:
            tsm_pred_returns = tsm_pred
            tsm_pred_eval = returns_to_prices(tsm_pred, y_test_base)
            tsm_pred_train_eval = returns_to_prices(tsm_pred_train, y_train_base)
            tsm_pred_val_eval = returns_to_prices(tsm_pred_val, y_val_base)
        else:
            tsm_pred_eval = tsm_pred
            tsm_pred_train_eval = tsm_pred_train
            tsm_pred_val_eval = tsm_pred_val
        
        # Evaluate
        baseline_results['tsm'] = {
            'predictions': tsm_pred_eval,
            'metrics': compute_metrics_by_horizon(y_test_future, tsm_pred_eval, horizons),
            'path_metrics': compute_path_metrics(y_test_future, tsm_pred_eval),
            'errors': get_per_sample_errors(y_test_future, tsm_pred_eval, horizons)
        }
        
        logger.info("\nTSM Results:")
        logger.info(baseline_results['tsm']['metrics'])
        tsm_path_mse = (baseline_results["tsm"].get("path_metrics") or {}).get("mse_path")
        if tsm_path_mse is not None:
            logger.info("TSM Path MSE (avg over %dd): %.6f", config.pred_len, float(tsm_path_mse))
        
        # Save predictions
        pred_df = pd.DataFrame({
            'date': splits['test']['dates']
        })
        for h in range(config.pred_len):
            pred_df[f'y_true_t_plus_{h+1}'] = y_test_future[:, h]
            pred_df[f'yhat_t_plus_{h+1}'] = tsm_pred_eval[:, h]
        
        pred_df.to_parquet(run_dir / "predictions" / "tsm_pred_test.parquet")

        # TSM + LASSO feature selection
        if lasso_cfg.get("enabled", False) and lasso_mode == "per_horizon_models":
            for h in horizons:
                cols = selected_feature_cols_by_horizon.get(h, feature_cols)
                selected_indices = [feature_cols.index(name) for name in cols]
                tsm_lasso_config = copy.deepcopy(config.raw)
                tsm_lasso_config['model']['enc_in'] = len(selected_indices)
                tsm_lasso_config['model']['dec_in'] = len(selected_indices)

                tsm_lasso_pred = _train_tsm(
                    splits['train']['X_enc'][:, :, selected_indices],
                    splits['train']['X_dec'][:, :, selected_indices],
                    splits['train']['y'],
                    splits['val']['X_enc'][:, :, selected_indices],
                    splits['val']['X_dec'][:, :, selected_indices],
                    splits['val']['y'],
                    splits['test']['X_enc'][:, :, selected_indices],
                    splits['test']['X_dec'][:, :, selected_indices],
                    splits['test']['y'],
                    f"tsm_lasso_h{h}",
                    tsm_lasso_config,
                )

                if target_is_returns:
                    tsm_lasso_pred_eval = returns_to_prices(tsm_lasso_pred, y_test_base)
                else:
                    tsm_lasso_pred_eval = tsm_lasso_pred

                model_name = f"tsm_lasso_h{h}"
                baseline_results[model_name] = {
                    'predictions': tsm_lasso_pred_eval,
                    'metrics': compute_metrics_by_horizon(
                        y_test_future, tsm_lasso_pred_eval, [h]
                    ),
                    'path_metrics': compute_path_metrics(y_test_future, tsm_lasso_pred_eval),
                    'errors': get_per_sample_errors(y_test_future, tsm_lasso_pred_eval, [h])
                }

                pred_df_lasso = pd.DataFrame({
                    'date': splits['test']['dates']
                })
                for step in range(config.pred_len):
                    pred_df_lasso[f'y_true_t_plus_{step+1}'] = y_test_future[:, step]
                    pred_df_lasso[f'yhat_t_plus_{step+1}'] = tsm_lasso_pred_eval[:, step]
                pred_df_lasso.to_parquet(
                    run_dir / "predictions" / f"tsm_lasso_h{h}_pred_test.parquet"
                )

        elif selected_feature_cols != feature_cols:
            selected_indices = [feature_cols.index(name) for name in selected_feature_cols]
            tsm_lasso_config = copy.deepcopy(config.raw)
            tsm_lasso_config['model']['enc_in'] = len(selected_indices)
            tsm_lasso_config['model']['dec_in'] = len(selected_indices)

            tsm_lasso_pred = _train_tsm(
                splits['train']['X_enc'][:, :, selected_indices],
                splits['train']['X_dec'][:, :, selected_indices],
                splits['train']['y'],
                splits['val']['X_enc'][:, :, selected_indices],
                splits['val']['X_dec'][:, :, selected_indices],
                splits['val']['y'],
                splits['test']['X_enc'][:, :, selected_indices],
                splits['test']['X_dec'][:, :, selected_indices],
                splits['test']['y'],
                "tsm_lasso",
                tsm_lasso_config,
            )

            if target_is_returns:
                tsm_lasso_pred_eval = returns_to_prices(tsm_lasso_pred, y_test_base)
            else:
                tsm_lasso_pred_eval = tsm_lasso_pred

            baseline_results['tsm_lasso'] = {
                'predictions': tsm_lasso_pred_eval,
                'metrics': compute_metrics_by_horizon(y_test_future, tsm_lasso_pred_eval, horizons),
                'path_metrics': compute_path_metrics(y_test_future, tsm_lasso_pred_eval),
                'errors': get_per_sample_errors(y_test_future, tsm_lasso_pred_eval, horizons)
            }

            pred_df_lasso = pd.DataFrame({
                'date': splits['test']['dates']
            })
            for step in range(config.pred_len):
                pred_df_lasso[f'y_true_t_plus_{step+1}'] = y_test_future[:, step]
                pred_df_lasso[f'yhat_t_plus_{step+1}'] = tsm_lasso_pred_eval[:, step]
            pred_df_lasso.to_parquet(run_dir / "predictions" / "tsm_lasso_pred_test.parquet")
        
    except ImportError:
        logger.warning("PyTorch not available - skipping TSM training")
    except Exception as e:
        logger.error(f"TSM training failed: {e}")
        import traceback
        traceback.print_exc()
    
    # =========================================================================
    # Step 6: LLM refinement (if API key available)
    # =========================================================================
    logger.info("\n" + "=" * 70)
    logger.info("STEP 6: LLM refinement")
    logger.info("=" * 70)
    
    llm_results = {}
    llm_eval_indices = None
    baseline_subset_results = {}
    
    try:
        import os
        has_api_key = bool(config.llm.get("api_key") or os.environ.get("OPENAI_API_KEY"))
        if not has_api_key:
            logger.warning(
                "No API key available (set llm.api_key in config or OPENAI_API_KEY) - skipping LLM refinement"
            )
        else:
            from llm import LLMRefiner

            history_points = min(
                int(config.llm.get("history_points", config.seq_len)),
                config.seq_len
            )
            llm_histories = llm_histories[:, -history_points:]
            if target_is_returns:
                eps = 1e-8
                llm_return_histories = np.log(
                    np.clip(llm_histories[:, 1:], eps, None)
                    / np.clip(llm_histories[:, :-1], eps, None)
                )
            else:
                llm_return_histories = None
            
            max_exo = int(config.llm.get("max_exogenous_features", 6))
            exogenous_summaries = [
                build_exogenous_summary(
                    splits["test"]["X_enc"][i, -history_points:, :],
                    feature_cols,
                    target_col,
                    max_features=max_exo
                )
                for i in range(len(llm_histories))
            ]
            
            gating_config = config.llm.get("gating", {})
            gating_enabled = bool(gating_config.get("enabled", False))
            if gating_enabled:
                vol_window = int(gating_config.get("volatility_window", 20))
                vols = np.array([
                    compute_history_volatility(history, vol_window)
                    for history in llm_histories
                ])
                threshold = gating_config.get("volatility_threshold")
                if threshold is None:
                    percentile = float(gating_config.get("volatility_percentile", 0.6))
                    threshold = float(np.quantile(vols, percentile))
                eligible_idx = np.where(vols >= threshold)[0]
                logger.info(
                    "LLM gating enabled: %d/%d samples above volatility threshold %.6f",
                    len(eligible_idx),
                    len(llm_histories),
                    threshold
                )
            else:
                eligible_idx = np.arange(len(llm_histories))
            
            # Prepare date arrays aligned to history windows
            panel_dates = pd.to_datetime(panel["date"]).to_numpy()
            date_to_idx = {pd.Timestamp(d): i for i, d in enumerate(panel_dates)}
            test_dates = build_history_dates(
                panel_dates,
                date_to_idx,
                splits["test"]["dates"],
                history_points
            )

            sentiment_cfg = config.llm.get("sentiment", {})
            sentiment_map = {}
            sentiment_points = int(sentiment_cfg.get("history_points", 18))
            sentiment_histories = None
            if sentiment_cfg.get("enabled", False):
                sentiment_map = load_daily_sentiment(
                    sentiment_cfg.get("path", "data/news/daily_sentiment.csv"),
                    date_col=sentiment_cfg.get("date_col", "date"),
                    score_col=sentiment_cfg.get("score_col", "sent_score")
                )
                if sentiment_map:
                    sentiment_histories = build_sentiment_histories(
                        test_dates,
                        sentiment_map,
                        sentiment_points
                    )
                else:
                    logger.warning("Sentiment enabled but no sentiment data loaded.")

            drift_cfg = config.llm.get("news_drift", {})
            calib_cfg = drift_cfg.get("calibration", {})
            if calib_cfg.get("enabled", False) and sentiment_map:
                calib_split = calib_cfg.get("split", "val")
                if calib_split == "train":
                    calib_histories = y_train_hist[:, -history_points:]
                    calib_future = y_train_future
                    calib_dates = build_history_dates(
                        panel_dates,
                        date_to_idx,
                        splits["train"]["dates"],
                        history_points
                    )
                else:
                    calib_histories = y_val_hist[:, -history_points:]
                    calib_future = y_val_future
                    calib_dates = build_history_dates(
                        panel_dates,
                        date_to_idx,
                        splits["val"]["dates"],
                        history_points
                    )
                calib_sentiment = build_sentiment_histories(
                    calib_dates,
                    sentiment_map,
                    sentiment_points
                )
                sentiment_window = int(drift_cfg.get("sentiment_window", 1))
                calib = fit_sentiment_calibration(
                    calib_histories,
                    calib_future,
                    calib_sentiment,
                    sentiment_window
                )
                drift_cfg["calibration"] = {
                    **calib,
                    "enabled": True,
                    "split": calib_split,
                    "sentiment_window": sentiment_window,
                }
                config.llm["news_drift"] = drift_cfg
                logger.info(
                    "Sentiment calibration: alpha=%.6f beta=%.6f n=%d",
                    calib.get("alpha", 0.0),
                    calib.get("beta", 0.0),
                    calib.get("n", 0),
                )

            refiner = LLMRefiner(
                config.llm,
                cache_dir=run_dir / "llm" / "cache",
                log_dir=run_dir / "llm" / "logs"
            )

            calibrate_cfg = config.llm.get("calibrate_blend", {})
            calibrate_enabled = bool(calibrate_cfg.get("enabled", False))
            calibrate_methods = calibrate_cfg.get("methods")
            if calibrate_methods is None:
                calibrate_methods = config.llm.get("methods", ["TSM+LLM"])
            min_weight = float(calibrate_cfg.get("min_weight", 0.0))
            max_weight = float(calibrate_cfg.get("max_weight", 1.0))

            subset_cfg = config.llm.get("subset", {}) or {}
            subset_strategy = str(subset_cfg.get("strategy", "first")).lower()
            subset_seed = int(subset_cfg.get("seed", config.seed))
            val_subset_strategy = str(subset_cfg.get("val_strategy", subset_strategy)).lower()
            val_subset_seed = int(subset_cfg.get("val_seed", subset_seed))

            blend_grid_cfg = config.llm.get("blend_grid", {}) or {}
            blend_grid_enabled = bool(blend_grid_cfg.get("enabled", False))
            blend_grid_methods = blend_grid_cfg.get("methods")
            if blend_grid_methods is None:
                blend_grid_methods = config.llm.get("methods", ["TSM+LLM"])
            blend_grid_schedule = str(blend_grid_cfg.get("schedule", "ramp")).lower()
            blend_grid_weights = _parse_float_list(
                blend_grid_cfg.get("weights"),
                default=[0.0, 0.25, 0.5, 0.75, 1.0],
            )
            blend_grid_key_horizons = [
                int(x)
                for x in _parse_float_list(
                    blend_grid_cfg.get("key_horizons"),
                    default=[1, 5, 10, 20, 30],
                )
            ]
            blend_grid_min_weight = float(blend_grid_cfg.get("min_weight", 0.0))
            blend_grid_power = float(blend_grid_cfg.get("power", 1.0))
            blend_grid_tune_split = str(blend_grid_cfg.get("tune_split", "val")).lower()
            blend_grid_metric = str(blend_grid_cfg.get("metric", "mse_path")).lower()

            val_histories = None
            val_dates = None
            val_exogenous_summaries = None
            val_sentiment_histories = None
            val_eval_indices = None
            val_eval_indices_grid = None
            if calibrate_enabled or blend_grid_enabled:
                val_histories = y_val_hist[:, -history_points:]
                val_dates = build_history_dates(
                    panel_dates,
                    date_to_idx,
                    splits["val"]["dates"],
                    history_points
                )
                val_exogenous_summaries = [
                    build_exogenous_summary(
                        splits["val"]["X_enc"][i, -history_points:, :],
                        feature_cols,
                        target_col,
                        max_features=max_exo
                    )
                    for i in range(len(val_histories))
                ]
                if sentiment_cfg.get("enabled", False) and sentiment_map:
                    val_sentiment_histories = build_sentiment_histories(
                        val_dates,
                        sentiment_map,
                        sentiment_points
                    )

                if gating_enabled:
                    vol_window = int(gating_config.get("volatility_window", 20))
                    val_vols = np.array([
                        compute_history_volatility(history, vol_window)
                        for history in val_histories
                    ])
                    threshold = gating_config.get("volatility_threshold")
                    if threshold is None:
                        percentile = float(gating_config.get("volatility_percentile", 0.6))
                        threshold = float(np.quantile(val_vols, percentile))
                    eligible_val_idx = np.where(val_vols >= threshold)[0]
                else:
                    eligible_val_idx = np.arange(len(val_histories))

                if calibrate_enabled:
                    cal_max = int(
                        calibrate_cfg.get("max_samples", config.llm.get("max_samples", 50))
                    )
                    val_eval_indices = select_eval_indices(
                        eligible_val_idx,
                        cal_max,
                        strategy=val_subset_strategy,
                        seed=val_subset_seed,
                    )

                if blend_grid_enabled and blend_grid_tune_split == "val":
                    grid_max = int(
                        blend_grid_cfg.get("max_samples", config.llm.get("max_samples", 50))
                    )
                    val_eval_indices_grid = select_eval_indices(
                        eligible_val_idx,
                        grid_max,
                        strategy=val_subset_strategy,
                        seed=val_subset_seed,
                    )
            
            # Run each method (limit samples for cost control)
            max_samples = min(config.llm.get("max_samples", 50), len(eligible_idx))
            llm_eval_indices = select_eval_indices(
                eligible_idx,
                max_samples,
                strategy=subset_strategy,
                seed=subset_seed,
            )
            histories_subset = llm_histories[llm_eval_indices]
            dates_subset = [test_dates[i] for i in llm_eval_indices]
            exogenous_subset = [exogenous_summaries[i] for i in llm_eval_indices]
            sentiment_subset = (
                [sentiment_histories[i] for i in llm_eval_indices]
                if sentiment_histories is not None
                else None
            )
            
            for method in config.llm.get('methods', ['TSM+LLM']):
                logger.info(f"Running method: {method}")
                val_teaching_examples_subset = None

                if method in ("TSM+LLM-COT-RF", "TSM+LLM-COT-SENT-RF", "TSM+LLM-COT-SENT-RF-DELTA"):
                    cot_cfg = config.llm.get("cot_rf", {})
                    k_examples = int(cot_cfg.get("k_examples", 5))
                    selection_mode = cot_cfg.get("example_selection", "recent")
                    feature_window = int(cot_cfg.get("feature_window", 20))
                    lookback_days = cot_cfg.get("lookback_days")
                    lookback_days = int(lookback_days) if lookback_days is not None else None

                    all_dates = np.concatenate(
                        [
                            splits["train"]["dates"],
                            splits["val"]["dates"],
                            splits["test"]["dates"],
                        ]
                    )
                    all_dates = pd.to_datetime(all_dates).to_numpy()

                    all_histories = np.concatenate(
                        [y_train_hist, y_val_hist, y_test_hist]
                    )
                    all_forecasts = np.concatenate(
                        [tsm_pred_train_eval, tsm_pred_val_eval, tsm_pred_eval]
                    )
                    all_truth = np.concatenate(
                        [y_train_future, y_val_future, y_test_future]
                    )

                    # Pre-compute per-window TSM error so we can pick informative
                    # "teaching examples" (if requested). If the model is already
                    # accurate in the teaching set, the LLM learns that the best
                    # correction is often “do nothing”.
                    all_path_mse = np.mean((all_forecasts - all_truth) ** 2, axis=1)

                    sorted_idx = np.argsort(all_dates)
                    dates_sorted = all_dates[sorted_idx]
                    if selection_mode == "similarity":
                        def _history_features(hist: np.ndarray) -> np.ndarray:
                            window = hist[-feature_window:] if len(hist) >= feature_window else hist
                            mean = float(np.mean(window)) if len(window) else 0.0
                            std = float(np.std(window)) if len(window) else 0.0
                            if len(window) > 1:
                                slope = float(np.polyfit(np.arange(len(window)), window, 1)[0])
                            else:
                                slope = 0.0
                            return np.array([mean, std, slope], dtype=float)

                        all_history_features = np.vstack([
                            _history_features(hist) for hist in all_histories
                        ])

                    teaching_examples_subset = []
                    val_teaching_examples_subset = []
                    for sample_idx in llm_eval_indices:
                        test_date = pd.Timestamp(splits["test"]["dates"][sample_idx]).to_datetime64()
                        pos = np.searchsorted(dates_sorted, test_date, side="left")
                        if selection_mode in ("high_error", "recent_high_error") and pos > 0:
                            candidate_indices = sorted_idx[:pos]
                            if selection_mode == "recent_high_error" and lookback_days is not None:
                                cutoff = np.datetime64(
                                    pd.Timestamp(test_date) - pd.Timedelta(days=lookback_days)
                                )
                                candidate_indices = candidate_indices[all_dates[candidate_indices] >= cutoff]
                                if candidate_indices.size == 0:
                                    candidate_indices = sorted_idx[:pos]
                            cand_errors = all_path_mse[candidate_indices]
                            if len(candidate_indices) > k_examples:
                                order = np.argsort(cand_errors)[-k_examples:]
                                example_indices = candidate_indices[order]
                            else:
                                example_indices = candidate_indices
                            # Keep examples in chronological order for readability
                            example_indices = example_indices[np.argsort(all_dates[example_indices])]
                        elif selection_mode == "similarity" and pos > 0:
                            candidate_indices = sorted_idx[:pos]
                            if lookback_days is not None:
                                cutoff = np.datetime64(
                                    pd.Timestamp(test_date) - pd.Timedelta(days=lookback_days)
                                )
                                candidate_indices = candidate_indices[all_dates[candidate_indices] >= cutoff]
                                if candidate_indices.size == 0:
                                    candidate_indices = sorted_idx[:pos]
                            sample_feat = _history_features(y_test_hist[sample_idx])
                            cand_feats = all_history_features[candidate_indices]
                            distances = np.linalg.norm(cand_feats - sample_feat, axis=1)
                            order = np.argsort(distances)[:k_examples]
                            example_indices = candidate_indices[order]
                        else:
                            start = max(0, pos - k_examples)
                            example_indices = sorted_idx[start:pos]

                        examples = []
                        for idx in example_indices:
                            sent_hist = None
                            if sentiment_map:
                                ex_date = pd.Timestamp(all_dates[idx])
                                ex_dates = build_history_dates(
                                    panel_dates,
                                    date_to_idx,
                                    np.array([ex_date]),
                                    history_points
                                )[0]
                                sent_hist = build_sentiment_histories(
                                    [ex_dates],
                                    sentiment_map,
                                    sentiment_points
                                )[0]
                            examples.append(
                                {
                                    "history": all_histories[idx],
                                    "forecast": all_forecasts[idx],
                                    "truth": all_truth[idx],
                                    "date": str(pd.Timestamp(all_dates[idx])),
                                    "sentiment_history": sent_hist if sent_hist is not None else [],
                                }
                            )
                        teaching_examples_subset.append(examples)

                    if val_eval_indices_grid is not None and len(val_eval_indices_grid) > 0:
                        for sample_idx in val_eval_indices_grid:
                            val_date = pd.Timestamp(splits["val"]["dates"][sample_idx]).to_datetime64()
                            pos = np.searchsorted(dates_sorted, val_date, side="left")
                            if selection_mode in ("high_error", "recent_high_error") and pos > 0:
                                candidate_indices = sorted_idx[:pos]
                                if selection_mode == "recent_high_error" and lookback_days is not None:
                                    cutoff = np.datetime64(
                                        pd.Timestamp(val_date) - pd.Timedelta(days=lookback_days)
                                    )
                                    candidate_indices = candidate_indices[all_dates[candidate_indices] >= cutoff]
                                    if candidate_indices.size == 0:
                                        candidate_indices = sorted_idx[:pos]
                                cand_errors = all_path_mse[candidate_indices]
                                if len(candidate_indices) > k_examples:
                                    order = np.argsort(cand_errors)[-k_examples:]
                                    example_indices = candidate_indices[order]
                                else:
                                    example_indices = candidate_indices
                                example_indices = example_indices[np.argsort(all_dates[example_indices])]
                            elif selection_mode == "similarity" and pos > 0:
                                candidate_indices = sorted_idx[:pos]
                                if lookback_days is not None:
                                    cutoff = np.datetime64(
                                        pd.Timestamp(val_date) - pd.Timedelta(days=lookback_days)
                                    )
                                    candidate_indices = candidate_indices[all_dates[candidate_indices] >= cutoff]
                                    if candidate_indices.size == 0:
                                        candidate_indices = sorted_idx[:pos]
                                sample_feat = _history_features(y_val_hist[sample_idx])
                                cand_feats = all_history_features[candidate_indices]
                                distances = np.linalg.norm(cand_feats - sample_feat, axis=1)
                                order = np.argsort(distances)[:k_examples]
                                example_indices = candidate_indices[order]
                            else:
                                start = max(0, pos - k_examples)
                                example_indices = sorted_idx[start:pos]

                            examples = []
                            for idx in example_indices:
                                sent_hist = None
                                if sentiment_map:
                                    ex_date = pd.Timestamp(all_dates[idx])
                                    ex_dates = build_history_dates(
                                        panel_dates,
                                        date_to_idx,
                                        np.array([ex_date]),
                                        history_points
                                    )[0]
                                    sent_hist = build_sentiment_histories(
                                        [ex_dates],
                                        sentiment_map,
                                        sentiment_points
                                    )[0]
                                examples.append(
                                    {
                                        "history": all_histories[idx],
                                        "forecast": all_forecasts[idx],
                                        "truth": all_truth[idx],
                                        "date": str(pd.Timestamp(all_dates[idx])),
                                        "sentiment_history": sent_hist if sent_hist is not None else [],
                                    }
                                )
                            val_teaching_examples_subset.append(examples)
                    exogenous_subset = [None for _ in llm_eval_indices]
                
                tsm_forecast = (
                    tsm_pred_eval[llm_eval_indices]
                    if tsm_pred_eval is not None
                    else None
                )
                price_bases = (
                    y_test_base[llm_eval_indices]
                    if y_test_base is not None
                    else None
                )
                sentiment_subset = (
                    [sentiment_histories[i] for i in llm_eval_indices]
                    if sentiment_histories is not None
                    else None
                )

                try:
                    method_tsm_forecast = tsm_forecast
                    method_exogenous = exogenous_subset
                    method_price_bases = price_bases
                    method_sentiment = sentiment_subset
                    if method == "NEWS-SENTIMENT-ONLY":
                        method_tsm_forecast = None

                    predictions, metadata = refiner.refine_batch(
                        method=method,
                        histories=histories_subset,
                        date_arrays=dates_subset,
                        tsm_forecasts=method_tsm_forecast,
                        pred_len=config.pred_len,
                        exogenous_summaries=method_exogenous,
                        price_bases=method_price_bases,
                        teaching_examples=(
                            teaching_examples_subset
                            if method in ("TSM+LLM-COT-RF", "TSM+LLM-COT-SENT-RF", "TSM+LLM-COT-SENT-RF-DELTA")
                            else None
                        ),
                        sentiment_histories=method_sentiment
                    )
                    
                    llm_results[method] = {
                        'predictions': predictions,
                        'metrics': compute_metrics_by_horizon(
                            y_test_future[llm_eval_indices], predictions, horizons
                        ),
                        'path_metrics': compute_path_metrics(
                            y_test_future[llm_eval_indices], predictions
                        ),
                        'errors': get_per_sample_errors(
                            y_test_future[llm_eval_indices], predictions, horizons
                        ),
                        'eval_indices': llm_eval_indices
                    }

                    # Persist subset predictions so we can debug deltas vs TSM and
                    # produce paper-style plots without re-running the LLM.
                    try:
                        out_path = run_dir / "predictions" / f"{method}_pred_test_subset.npz"
                        np.savez_compressed(
                            out_path,
                            method=np.array([method]),
                            eval_indices=np.asarray(llm_eval_indices, dtype=int),
                            dates=np.asarray(
                                [str(splits["test"]["dates"][i]) for i in llm_eval_indices],
                                dtype="U",
                            ),
                            y_true=np.asarray(y_test_future[llm_eval_indices], dtype=float),
                            tsm_pred=np.asarray(tsm_pred_eval[llm_eval_indices], dtype=float)
                            if tsm_pred_eval is not None
                            else np.empty((0, 0), dtype=float),
                            yhat=np.asarray(predictions, dtype=float),
                        )
                    except Exception as e:
                        logger.warning("Failed to save LLM subset predictions: %s", e)
                    
                    logger.info(f"\n{method} Results:")
                    logger.info(llm_results[method]['metrics'])
                    method_path_mse = (llm_results[method].get("path_metrics") or {}).get("mse_path")
                    if method_path_mse is not None:
                        logger.info(
                            "%s Path MSE (avg over %dd): %.6f",
                            method,
                            config.pred_len,
                            float(method_path_mse),
                        )

                    if calibrate_enabled and method in calibrate_methods:
                        if val_eval_indices is None or len(val_eval_indices) == 0:
                            logger.warning("Blend calibration skipped (no validation samples).")
                        elif method == "NEWS-SENTIMENT-ONLY":
                            logger.warning("Blend calibration skipped for NEWS-SENTIMENT-ONLY.")
                        elif method == "TSM+LLM-COT-RF":
                            logger.warning("Blend calibration skipped for CoT-RF.")
                        else:
                            val_predictions, _ = refiner.refine_batch(
                                method=method,
                                histories=val_histories[val_eval_indices],
                                date_arrays=[val_dates[i] for i in val_eval_indices],
                                tsm_forecasts=tsm_pred_val_eval[val_eval_indices],
                                pred_len=config.pred_len,
                                exogenous_summaries=(
                                    [val_exogenous_summaries[i] for i in val_eval_indices]
                                    if val_exogenous_summaries is not None
                                    else None
                                ),
                                price_bases=None,
                                teaching_examples=None,
                                sentiment_histories=(
                                    [val_sentiment_histories[i] for i in val_eval_indices]
                                    if val_sentiment_histories is not None
                                    else None
                                )
                            )

                            weights = fit_blend_weights(
                                y_val_future[val_eval_indices],
                                tsm_pred_val_eval[val_eval_indices],
                                val_predictions,
                                horizons,
                                min_weight=min_weight,
                                max_weight=max_weight
                            )
                            blended = apply_blend_weights(
                                tsm_pred_eval[llm_eval_indices],
                                predictions,
                                weights,
                                config.pred_len
                            )
                            blend_name = f"{method}_calibrated"
                            llm_results[blend_name] = {
                                'predictions': blended,
                                'metrics': compute_metrics_by_horizon(
                                    y_test_future[llm_eval_indices], blended, horizons
                                ),
                                'path_metrics': compute_path_metrics(
                                    y_test_future[llm_eval_indices], blended
                                ),
                                'errors': get_per_sample_errors(
                                    y_test_future[llm_eval_indices], blended, horizons
                                ),
                                'eval_indices': llm_eval_indices,
                                'weights': weights
                            }
                            logger.info(f"\n{blend_name} Results:")
                            logger.info(llm_results[blend_name]['metrics'])
                            blend_path_mse = (llm_results[blend_name].get("path_metrics") or {}).get("mse_path")
                            if blend_path_mse is not None:
                                logger.info(
                                    "%s Path MSE (avg over %dd): %.6f",
                                    blend_name,
                                    config.pred_len,
                                    float(blend_path_mse),
                                )
                            weights_path = run_dir / "llm" / "blend_weights.json"
                            with weights_path.open("w") as f:
                                json.dump({blend_name: weights}, f, indent=2)

                    # Blend-grid tuning: select a blend strength on validation, then
                    # evaluate the same strength on the test subset. This avoids
                    # leaking test information when we "dial up" LLM influence.
                    if blend_grid_enabled and method in blend_grid_methods:
                        if method == "NEWS-SENTIMENT-ONLY":
                            logger.warning("Blend grid skipped for NEWS-SENTIMENT-ONLY.")
                        elif tsm_pred_eval is None or tsm_pred_val_eval is None:
                            logger.warning("Blend grid skipped (missing TSM predictions).")
                        elif blend_grid_tune_split != "val":
                            logger.warning(
                                "Blend grid skipped (unsupported tune_split=%s).",
                                blend_grid_tune_split,
                            )
                        elif val_eval_indices_grid is None or len(val_eval_indices_grid) == 0:
                            logger.warning("Blend grid skipped (no validation samples).")
                        else:
                            method_slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", method)
                            out_root = run_dir / "results" / f"blend_grid_{blend_grid_schedule}" / method_slug

                            # 1) Get LLM predictions on the validation subset
                            val_method_exogenous = (
                                [val_exogenous_summaries[i] for i in val_eval_indices_grid]
                                if val_exogenous_summaries is not None
                                else None
                            )
                            if method in ("TSM+LLM-COT-RF", "TSM+LLM-COT-SENT-RF", "TSM+LLM-COT-SENT-RF-DELTA"):
                                # These methods don't consume exogenous summaries in prompts.
                                val_method_exogenous = [None for _ in val_eval_indices_grid]

                            val_method_sentiment = (
                                [val_sentiment_histories[i] for i in val_eval_indices_grid]
                                if val_sentiment_histories is not None
                                else None
                            )

                            val_teaching = (
                                val_teaching_examples_subset
                                if method in ("TSM+LLM-COT-RF", "TSM+LLM-COT-SENT-RF", "TSM+LLM-COT-SENT-RF-DELTA")
                                else None
                            )

                            val_llm_pred, _ = refiner.refine_batch(
                                method=method,
                                histories=val_histories[val_eval_indices_grid],
                                date_arrays=[val_dates[i] for i in val_eval_indices_grid],
                                tsm_forecasts=tsm_pred_val_eval[val_eval_indices_grid],
                                pred_len=config.pred_len,
                                exogenous_summaries=val_method_exogenous,
                                price_bases=None,
                                teaching_examples=val_teaching,
                                sentiment_histories=val_method_sentiment,
                            )

                            # 2) Evaluate blend grid on validation (selection split)
                            val_summary, val_mse_by_h, val_best_by_h = evaluate_blend_grid(
                                y_true=y_val_future[val_eval_indices_grid],
                                base_pred=tsm_pred_val_eval[val_eval_indices_grid],
                                llm_pred=val_llm_pred,
                                weights=blend_grid_weights,
                                schedule=blend_grid_schedule,
                                key_horizons=blend_grid_key_horizons,
                                min_weight=blend_grid_min_weight,
                                power=blend_grid_power,
                            )
                            save_blend_grid_artifacts(
                                summary_df=val_summary,
                                mse_by_h_df=val_mse_by_h,
                                best_by_h_df=val_best_by_h,
                                out_dir=out_root / "val",
                            )

                            # 3) Evaluate blend grid on test (reporting split)
                            test_summary, test_mse_by_h, test_best_by_h = evaluate_blend_grid(
                                y_true=y_test_future[llm_eval_indices],
                                base_pred=tsm_pred_eval[llm_eval_indices],
                                llm_pred=predictions,
                                weights=blend_grid_weights,
                                schedule=blend_grid_schedule,
                                key_horizons=blend_grid_key_horizons,
                                min_weight=blend_grid_min_weight,
                                power=blend_grid_power,
                            )
                            save_blend_grid_artifacts(
                                summary_df=test_summary,
                                mse_by_h_df=test_mse_by_h,
                                best_by_h_df=test_best_by_h,
                                out_dir=out_root / "test",
                            )

                            # 4) Pick the best blend strength on validation
                            metric_col = "mse_path"
                            if blend_grid_metric != "mse_path":
                                candidate = blend_grid_metric
                                if candidate in val_summary.columns:
                                    metric_col = candidate
                            best_w_path = float(val_summary.loc[val_summary[metric_col].idxmin(), "w"])

                            best_w_by_horizon = {}
                            for h in horizons:
                                col = f"h{int(h)}_mse"
                                if col in val_summary.columns:
                                    best_w_by_horizon[int(h)] = float(
                                        val_summary.loc[val_summary[col].idxmin(), "w"]
                                    )

                            selection_payload = {
                                "method": method,
                                "schedule": blend_grid_schedule,
                                "weights": blend_grid_weights,
                                "tune_split": blend_grid_tune_split,
                                "metric": metric_col,
                                "min_weight": blend_grid_min_weight,
                                "power": blend_grid_power,
                                "best_w_path": best_w_path,
                                "best_w_by_horizon": best_w_by_horizon,
                                "n_val": int(len(val_eval_indices_grid)),
                                "n_test": int(len(llm_eval_indices)),
                            }
                            with (run_dir / "llm" / f"blend_grid_selection_{method_slug}.json").open("w") as f:
                                json.dump(selection_payload, f, indent=2)

                            logger.info(
                                "Blend grid (val): best %s at w=%.2f",
                                metric_col,
                                best_w_path,
                            )

                            # 5) Register best-on-val blended variants as first-class models
                            def _register_blend(name_suffix: str, w: float) -> None:
                                blended_pred = blend_forecasts(
                                    base_pred=tsm_pred_eval[llm_eval_indices],
                                    llm_pred=predictions,
                                    strength=float(w),
                                    schedule=blend_grid_schedule,
                                    pred_len=config.pred_len,
                                    min_weight=blend_grid_min_weight,
                                    power=blend_grid_power,
                                )
                                model_name = f"{method}_blend_{blend_grid_schedule}_{name_suffix}"
                                llm_results[model_name] = {
                                    "predictions": blended_pred,
                                    "metrics": compute_metrics_by_horizon(
                                        y_test_future[llm_eval_indices], blended_pred, horizons
                                    ),
                                    "path_metrics": compute_path_metrics(
                                        y_test_future[llm_eval_indices], blended_pred
                                    ),
                                    "errors": get_per_sample_errors(
                                        y_test_future[llm_eval_indices], blended_pred, horizons
                                    ),
                                    "eval_indices": llm_eval_indices,
                                    "blend": {
                                        "schedule": blend_grid_schedule,
                                        "w": float(w),
                                        "min_weight": blend_grid_min_weight,
                                        "power": blend_grid_power,
                                        "selected_on": "val",
                                        "metric": metric_col,
                                    },
                                }
                                logger.info("Added blended model: %s (w=%.2f)", model_name, float(w))

                            _register_blend("bestval_path", best_w_path)
                            for h, w in best_w_by_horizon.items():
                                _register_blend(f"bestval_h{int(h)}", float(w))

                    if method == "TSM+LLM-COT-RF":
                        rules_path = run_dir / "llm" / "cot_rf_rules.jsonl"
                        with rules_path.open("w") as f:
                            for idx, meta in zip(llm_eval_indices, metadata):
                                record = {
                                    "sample_index": int(idx),
                                    "date": str(splits["test"]["dates"][idx]),
                                    "rules_text": meta.get("rules_text"),
                                    "teaching_dates": meta.get("teaching_dates"),
                                }
                                f.write(json.dumps(record) + "\n")
                
                except Exception as e:
                    logger.error(f"LLM method {method} failed: {e}")
            
            if llm_eval_indices is not None and len(llm_eval_indices) > 0:
                for name, result in baseline_results.items():
                    preds = result.get("predictions")
                    if preds is None:
                        continue
                    subset_metrics = compute_metrics_by_horizon(
                        y_test_future[llm_eval_indices],
                        preds[llm_eval_indices],
                        horizons
                    )
                    subset_errors = get_per_sample_errors(
                        y_test_future[llm_eval_indices],
                        preds[llm_eval_indices],
                        horizons
                    )
                    baseline_subset_results[f"{name}_llm_subset"] = {
                        "predictions": preds[llm_eval_indices],
                        "metrics": subset_metrics,
                        "path_metrics": compute_path_metrics(
                            y_test_future[llm_eval_indices], preds[llm_eval_indices]
                        ),
                        "errors": subset_errors,
                        "eval_indices": llm_eval_indices,
                    }
    
    except Exception as e:
        logger.error(f"LLM refinement failed: {e}")

    # Save paper-style path metrics (MSE over full forecast path) and a combined
    # metrics-by-horizon table so we can benchmark methods without enabling
    # full paper generation.
    try:
        path_rows = []
        metrics_rows = []

        def _append_rows(model_name: str, result: dict, subset: str) -> None:
            preds = result.get("predictions")
            n_samples = int(len(preds)) if preds is not None else 0

            pm = result.get("path_metrics")
            if pm:
                row = {"model": model_name, "subset": subset, "n_samples": n_samples}
                row.update({k: float(v) for k, v in pm.items()})
                path_rows.append(row)

            mdf = result.get("metrics")
            if mdf is not None:
                tmp = mdf.copy()
                tmp["model"] = model_name
                tmp["subset"] = subset
                tmp["n_samples"] = n_samples
                metrics_rows.append(tmp.reset_index())

        for name, result in baseline_results.items():
            _append_rows(name, result, subset="full")
        for name, result in llm_results.items():
            subset = "llm_subset" if result.get("eval_indices") is not None else "full"
            _append_rows(name, result, subset=subset)
        for name, result in baseline_subset_results.items():
            _append_rows(name, result, subset="llm_subset")

        if path_rows:
            pd.DataFrame(path_rows).to_csv(run_dir / "results" / "path_metrics.csv", index=False)
        if metrics_rows:
            pd.concat(metrics_rows, ignore_index=True).to_csv(
                run_dir / "results" / "metrics_by_horizon.csv", index=False
            )
    except Exception as e:
        logger.warning("Failed to save combined metrics tables: %s", e)
    
    # =========================================================================
    # Step 7: Statistical significance tests
    # =========================================================================
    logger.info("\n" + "=" * 70)
    logger.info("STEP 7: Statistical significance tests")
    logger.info("=" * 70)
    
    # Combine all results
    if baseline_subset_results:
        all_errors = {name: result['errors'] for name, result in baseline_subset_results.items()}
    else:
        all_errors = {name: result['errors'] for name, result in baseline_results.items()}
    all_errors.update({name: result['errors'] for name, result in llm_results.items()})
    
    if len(all_errors) > 1:
        baseline_name = "naive_persistence"
        if baseline_subset_results:
            subset_name = "naive_persistence_llm_subset"
            if subset_name in all_errors:
                baseline_name = subset_name
        significance_df = compare_all_models(
            all_errors,
            baseline_name=baseline_name,
            horizons=horizons
        )
        
        significance_df.to_csv(run_dir / "results" / "significance_tests.csv", index=False)
        logger.info("\nSignificance test results saved")
    
    # =========================================================================
    # Step 8: Robustness suite
    # =========================================================================
    logger.info("\n" + "=" * 70)
    logger.info("STEP 8: Robustness suite")
    logger.info("=" * 70)

    noise_results = None

    try:
        noise_levels = config.robustness.get("noise_levels", [])
        if noise_levels:
            from llm.refine import add_noise_to_forecast

            noise_rows = []
            candidate_models = {}

            if baseline_results.get("tsm") and baseline_results["tsm"].get("predictions") is not None:
                candidate_models["tsm"] = baseline_results["tsm"]["predictions"]

            for name, result in llm_results.items():
                if result.get("predictions") is not None:
                    candidate_models[name] = result["predictions"]

            for name, preds in candidate_models.items():
                y_true = y_test_future[:len(preds)]
                for level in noise_levels:
                    noisy_pred = add_noise_to_forecast(preds, level, seed=config.seed)
                    metrics = compute_metrics_by_horizon(y_true, noisy_pred, horizons)
                    metrics = metrics.reset_index()
                    metrics["noise_level"] = level
                    metrics["model"] = name
                    noise_rows.append(metrics)

            if noise_rows:
                noise_results = pd.concat(noise_rows, ignore_index=True)
                noise_results.to_csv(
                    run_dir / "results" / "robustness" / "noise_injection.csv",
                    index=False
                )
                logger.info("Robustness noise injection results saved")
        else:
            logger.info("No noise levels configured; skipping robustness suite")
    except Exception as e:
        logger.warning(f"Robustness suite failed: {e}")

    # =========================================================================
    # Step 9: Drift/return and long-short portfolio evaluation
    # =========================================================================
    logger.info("\n" + "=" * 70)
    logger.info("STEP 9: Drift/return and long-short portfolio evaluation")
    logger.info("=" * 70)

    return_cfg = config.evaluation.get("return_metrics", {})
    return_metrics_list = return_cfg.get("metrics", ["mse", "rmse", "mae", "mape"])
    signal_threshold = float(return_cfg.get("signal_threshold", 0.0))

    drift_rows = []
    portfolio_rows = []

    def _resolve_eval_indices(result: dict) -> Optional[np.ndarray]:
        indices = result.get("eval_indices")
        if indices is not None:
            return indices
        preds = result.get("predictions")
        if preds is None:
            return None
        if llm_eval_indices is not None and len(preds) == len(llm_eval_indices):
            return llm_eval_indices
        return None

    def _append_return_metrics(name: str, preds: np.ndarray, indices: Optional[np.ndarray]) -> None:
        if preds is None:
            return
        if indices is None:
            y_true_prices = y_test_future
            base_prices = y_test_base
        else:
            y_true_prices = y_test_future[indices]
            base_prices = y_test_base[indices]

        if preds.shape != y_true_prices.shape:
            logger.warning(
                "Skipping return metrics for %s due to shape mismatch: %s vs %s",
                name,
                preds.shape,
                y_true_prices.shape
            )
            return

        try:
            drift_df = compute_drift_metrics_by_horizon(
                y_true_prices,
                preds,
                base_prices,
                horizons,
                return_metrics_list
            )
            drift_df = drift_df.copy()
            drift_df["model"] = name
            drift_rows.append(drift_df.reset_index())

            portfolio_df = compute_long_short_portfolio_metrics(
                y_true_prices,
                preds,
                base_prices,
                horizons,
                signal_threshold=signal_threshold
            )
            portfolio_df = portfolio_df.copy()
            portfolio_df["model"] = name
            portfolio_rows.append(portfolio_df.reset_index())
        except Exception as e:
            logger.warning("Return metrics failed for %s: %s", name, e)

    for name, result in {**baseline_results, **llm_results}.items():
        preds = result.get("predictions")
        eval_indices = _resolve_eval_indices(result)
        _append_return_metrics(name, preds, eval_indices)

    if baseline_subset_results and llm_eval_indices is not None:
        for name, result in baseline_subset_results.items():
            preds = result.get("predictions")
            _append_return_metrics(name, preds, llm_eval_indices)

    drift_df = None
    if drift_rows:
        drift_df = pd.concat(drift_rows, ignore_index=True)
        drift_df.to_csv(run_dir / "results" / "drift_metrics.csv", index=False)
        logger.info("Drift metrics saved")

    portfolio_df = None
    if portfolio_rows:
        portfolio_df = pd.concat(portfolio_rows, ignore_index=True)
        portfolio_df.to_csv(run_dir / "results" / "long_short_metrics.csv", index=False)
        logger.info("Long-short portfolio metrics saved")

    # =========================================================================
    # Step 10: Trend classification
    # =========================================================================
    logger.info("\n" + "=" * 70)
    logger.info("STEP 10: Trend classification accuracy")
    logger.info("=" * 70)
    
    trend_results = {}
    paper_trend_results = {}
    trend_meta = {}

    paper_trend_cfg = config.evaluation.get("paper_trend", {})
    paper_alpha = float(paper_trend_cfg.get("alpha", 0.02))
    paper_history_window = int(paper_trend_cfg.get("history_window", 18))
    paper_horizons = paper_trend_cfg.get("horizons", [10, 20, 30])
    
    for name, result in {**baseline_results, **llm_results, **baseline_subset_results}.items():
        pred = result.get("predictions")
        if pred is None:
            continue

        indices = result.get("eval_indices")
        subset_label = "llm_subset" if indices is not None else "full"
        trend_meta[name] = {"subset": subset_label, "n_samples": int(len(pred))}
        if indices is None:
            y_true_prices = y_test_future[: len(pred)]
            base_prices = y_test_base[: len(pred)]
            hist_prices = y_test_hist[: len(pred)]
        else:
            y_true_prices = y_test_future[indices]
            base_prices = y_test_base[indices]
            hist_prices = y_test_hist[indices]

        if pred.shape != y_true_prices.shape:
            logger.warning(
                "Skipping trend metrics for %s due to shape mismatch: %s vs %s",
                name,
                pred.shape,
                y_true_prices.shape,
            )
            continue

        trend_acc = compute_trend_accuracy(
            y_true_prices,
            pred,
            base_prices,
            threshold=0.5,  # EUR (legacy, non-paper)
            horizons=horizons,
        )
        trend_results[name] = trend_acc

        paper_trend_acc = compute_paper_trend_accuracy(
            hist_prices,
            y_true_prices,
            pred,
            alpha=paper_alpha,
            history_window=paper_history_window,
            horizons=paper_horizons,
        )
        paper_trend_results[name] = paper_trend_acc

        logger.info(f"\n{name} Trend Accuracy (legacy, fixed threshold):")
        logger.info(trend_acc[["accuracy"]])
        logger.info(
            "\n%s Trend Accuracy (paper-style: Th=%d, α=%.3f):",
            name,
            paper_history_window,
            paper_alpha,
        )
        logger.info(paper_trend_acc[["accuracy"]])

    if trend_results:
        trend_dfs = []
        for name, df in trend_results.items():
            df = df.copy()
            df["model"] = name
            df["subset"] = trend_meta.get(name, {}).get("subset", "unknown")
            df["n_samples"] = trend_meta.get(name, {}).get("n_samples", 0)
            trend_dfs.append(df.reset_index())
        pd.concat(trend_dfs, ignore_index=True).to_csv(
            run_dir / "results" / "trend_accuracy.csv", index=False
        )

    if paper_trend_results:
        paper_trend_dfs = []
        for name, df in paper_trend_results.items():
            df = df.copy()
            df["model"] = name
            df["subset"] = trend_meta.get(name, {}).get("subset", "unknown")
            df["n_samples"] = trend_meta.get(name, {}).get("n_samples", 0)
            paper_trend_dfs.append(df.reset_index())
        pd.concat(paper_trend_dfs, ignore_index=True).to_csv(
            run_dir / "results" / "trend_accuracy_paper.csv", index=False
        )
    
    # =========================================================================
    # Step 11: Generate paper
    # =========================================================================
    if config.output.get("generate_paper", True):
        logger.info("\n" + "=" * 70)
        logger.info("STEP 10: Generating paper")
        logger.info("=" * 70)
    else:
        logger.info("Paper generation disabled by config")
        logger.info("\n" + "=" * 70)
        logger.info("EXPERIMENT COMPLETE")
        logger.info("=" * 70)
        logger.info(f"Results saved to: {run_dir}")
        return run_dir
    
    # Collect metrics for paper
    all_metrics = []
    for name, result in {**baseline_results, **llm_results}.items():
        metrics = result['metrics'].copy()
        metrics['model'] = name
        all_metrics.append(metrics.reset_index())
    
    if baseline_subset_results:
        for name, result in baseline_subset_results.items():
            metrics = result['metrics'].copy()
            metrics['model'] = name
            all_metrics.append(metrics.reset_index())
    
    if all_metrics:
        metrics_df = pd.concat(all_metrics, ignore_index=True)
    else:
        metrics_df = None
    
    trend_df = None
    if trend_results:
        trend_dfs = []
        for name, df in trend_results.items():
            df = df.copy()
            df['model'] = name
            trend_dfs.append(df.reset_index())
        trend_df = pd.concat(trend_dfs, ignore_index=True)
    
    def _build_paper(output_dir: Path) -> Path:
        writer = PaperWriter(output_dir)
        writer.write_all_sections(
            panel_schema=schema,
            config=config.raw,
            metrics_by_horizon=metrics_df,
            trend_accuracy=trend_df,
            significance_tests=significance_df if 'significance_df' in dir() else None,
            noise_results=noise_results
        )
        return writer.save()

    paper_snapshot_path = _build_paper(run_dir / "paper_snapshot")
    logger.info(f"Paper snapshot saved to: {paper_snapshot_path}")

    paper_root = Path(__file__).parent.parent / config.output.get("paper_dir", "paper")
    paper_root_path = _build_paper(paper_root)
    logger.info(f"Paper saved to: {paper_root_path}")
    
    # =========================================================================
    # Complete
    # =========================================================================
    logger.info("\n" + "=" * 70)
    logger.info("EXPERIMENT COMPLETE")
    logger.info("=" * 70)
    logger.info(f"Results saved to: {run_dir}")
    
    return run_dir


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="EU ETS Futures Forecasting Experiment"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config YAML file"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed override"
    )
    
    args = parser.parse_args()
    
    overrides = {}
    if args.seed:
        overrides["reproducibility"] = {"seed": args.seed}
    
    run_experiment(args.config, overrides)


if __name__ == "__main__":
    main()
