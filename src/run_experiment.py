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
from typing import Optional, Sequence
import logging
import json
import copy
import re
import numpy as np
import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from config import load_config
from data import build_panel, select_feature_columns
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

RULE_GATE_DEFAULT_CANDIDATES = (
    "none",
    "h5_pos_h20_neg",
    "h5_pos_h30_neg",
    "h20_neg_h30_zero",
    "mixed_long_signs",
    "conflict_any",
    "conflict_any_nonhigh_conf",
    "conflict_any_low_match",
    "conflict_any_low_match_or_nonhigh_conf",
    "h30_negative",
    "abs_h30_ge_0_25",
    "abs_h30_ge_0_40",
    "conflict_any_or_abs_h30_ge_0_25",
    "conflict_any_or_abs_h30_ge_0_40",
)

DEFAULT_EXOGENOUS_FEATURE_PRIORITY = (
    "target_range_pct",
    "target_volume",
    "is_auction_day",
    "uk_icap_primary_print_day",
    "uk_icap_secondary_print_day",
    "uk_icap_primary_secondary_spread_pct",
    "uk_icap_spread_z20",
    "uk_icap_primary_secondary_spread",
    "uk_icap_primary",
    "uk_icap_secondary",
    "y_vol_20d",
    "y_ma_5d",
    "y_momentum_5d",
    "y_momentum_20d",
    "auction_volume",
    "vstoxx",
    "brent_return",
    "coal_return",
)


def returns_to_prices(returns: np.ndarray, last_price: np.ndarray) -> np.ndarray:
    """Convert log returns to price paths."""
    last_price = np.asarray(last_price).reshape(-1, 1)
    return last_price * np.exp(np.cumsum(returns, axis=1))


def parse_name_list(value: object) -> list[str]:
    """Normalize a config value into a flat list of non-empty strings."""
    if value is None:
        return []
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        items: list[str] = []
        for item in value:
            if item is None:
                continue
            text = str(item).strip()
            if text:
                items.append(text)
        return items
    text = str(value).strip()
    return [text] if text else []


def resolve_exogenous_feature_names(
    feature_cols: list[str],
    target_col: str,
    preferred_features: Optional[Sequence[str]] = None,
    include_features: Optional[Sequence[str]] = None,
    max_features: Optional[int] = None,
) -> list[str]:
    """Resolve a stable, UK-aware feature order for prompt context and retrieval."""
    feature_idx = {name: i for i, name in enumerate(feature_cols)}
    requested = parse_name_list(include_features)
    preferred = parse_name_list(preferred_features) or list(DEFAULT_EXOGENOUS_FEATURE_PRIORITY)

    ordered: list[str] = []
    seed_names = requested if requested else preferred + [
        name for name in feature_cols if name.startswith("idx_")
    ]
    for name in seed_names:
        if name == target_col or name not in feature_idx or name in ordered:
            continue
        ordered.append(name)

    if not requested:
        for name in feature_cols:
            if name != target_col and name not in ordered:
                ordered.append(name)

    if max_features is not None:
        ordered = ordered[: int(max_features)]
    return ordered


def build_exogenous_summary(
    window: np.ndarray,
    feature_cols: list,
    target_col: str,
    max_features: int = 6,
    preferred_features: Optional[Sequence[str]] = None,
    include_features: Optional[Sequence[str]] = None,
) -> dict:
    """Summarize recent exogenous signals for LLM prompts."""
    summary = {}
    eps = 1e-8
    feature_idx = {name: i for i, name in enumerate(feature_cols)}
    ordered = resolve_exogenous_feature_names(
        feature_cols,
        target_col,
        preferred_features=preferred_features,
        include_features=include_features,
        max_features=max_features,
    )
    
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


def build_retrieval_feature_vector(
    window: np.ndarray,
    feature_cols: list[str],
    target_col: str,
    feature_window: int = 20,
    preferred_features: Optional[Sequence[str]] = None,
    include_features: Optional[Sequence[str]] = None,
    max_features: int = 6,
) -> np.ndarray:
    """Encode recent exogenous state into a compact numeric vector for retrieval."""
    selected = resolve_exogenous_feature_names(
        feature_cols,
        target_col,
        preferred_features=preferred_features,
        include_features=include_features,
        max_features=max_features,
    )
    if not selected:
        return np.zeros(0, dtype=float)

    feature_idx = {name: i for i, name in enumerate(feature_cols)}
    recent = window[-feature_window:] if len(window) >= feature_window else window
    eps = 1e-8
    values: list[float] = []
    for name in selected:
        idx = feature_idx.get(name)
        if idx is None:
            values.extend((0.0, 0.0))
            continue
        series = np.asarray(recent[:, idx], dtype=float)
        if series.size == 0:
            values.extend((0.0, 0.0))
            continue
        last = float(series[-1])
        unique_recent = np.unique(series)
        rounded = set(np.round(unique_recent, 6).tolist())
        if unique_recent.size <= 2 and rounded.issubset({0.0, 1.0}):
            values.extend((last, float(np.mean(series))))
            continue
        base = float(series[0])
        if abs(base) > eps:
            change = float((series[-1] / base - 1.0) * 100.0)
        else:
            change = 0.0
        values.extend((last, change))
    return np.asarray(values, dtype=float)


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


def fit_delta_calibration_scales(
    y_true: np.ndarray,
    base_pred: np.ndarray,
    llm_pred: np.ndarray,
    horizons: Sequence[int],
    target_horizons: Optional[Sequence[int]] = None,
    min_scale: float = 0.0,
    max_scale: float = 1.0,
    shared: bool = False,
) -> dict[int, float]:
    """Fit validation-trained delta scales for selected horizons."""
    all_horizons = [int(h) for h in horizons]
    selected = [int(h) for h in (target_horizons or all_horizons) if int(h) in all_horizons]
    scales = {int(h): 1.0 for h in all_horizons}
    if not selected:
        return scales

    if not shared:
        fitted = fit_blend_weights(
            y_true,
            base_pred,
            llm_pred,
            selected,
            min_weight=min_scale,
            max_weight=max_scale,
        )
        scales.update({int(h): float(v) for h, v in fitted.items()})
        return scales

    idx = [int(h) - 1 for h in selected]
    y = np.asarray(y_true[:, idx], dtype=float).reshape(-1)
    base = np.asarray(base_pred[:, idx], dtype=float).reshape(-1)
    llm = np.asarray(llm_pred[:, idx], dtype=float).reshape(-1)
    diff = llm - base
    denom = float(np.sum(diff * diff))
    if denom < 1e-12:
        scale = 0.0
    else:
        scale = float(np.sum(diff * (y - base)) / denom)
    scale = float(np.clip(scale, min_scale, max_scale))
    for h in selected:
        scales[int(h)] = scale
    return scales


def apply_delta_calibration(
    base_pred: np.ndarray,
    llm_pred: np.ndarray,
    scales: dict[int, float],
    pred_len: int,
    default_scale: float = 1.0,
) -> np.ndarray:
    """Apply horizon-specific scales to the LLM delta relative to the base forecast."""
    adjusted = llm_pred.copy()
    for h in range(1, int(pred_len) + 1):
        idx = h - 1
        scale = float(scales.get(int(h), default_scale))
        adjusted[:, idx] = base_pred[:, idx] + scale * (llm_pred[:, idx] - base_pred[:, idx])
    return adjusted


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


def restrict_to_recent_tail(
    eligible_idx: np.ndarray,
    tail_fraction: float = 1.0,
    min_samples: int = 0,
) -> np.ndarray:
    """Keep only the most recent tail of an already time-ordered index array."""
    eligible_idx = np.asarray(eligible_idx, dtype=int)
    if eligible_idx.size == 0:
        return np.array([], dtype=int)

    tail_fraction = float(tail_fraction)
    if not 0.0 < tail_fraction <= 1.0:
        raise ValueError(f"tail_fraction must be in (0, 1], got {tail_fraction}")

    min_samples = int(max(0, min_samples))
    tail_n = int(np.ceil(eligible_idx.size * tail_fraction))
    tail_n = max(tail_n, min_samples, 1)
    tail_n = min(tail_n, eligible_idx.size)
    return np.asarray(eligible_idx[-tail_n:], dtype=int)


def _error_profile(forecast: np.ndarray, truth: np.ndarray) -> np.ndarray:
    """Signed error profile at key horizons for example stratification."""
    horizon_idx = [0, 4, 19, 29]
    horizon_idx = [idx for idx in horizon_idx if idx < len(forecast) and idx < len(truth)]
    errs = np.asarray(forecast, dtype=float)[horizon_idx] - np.asarray(truth, dtype=float)[horizon_idx]
    if len(errs) < 4:
        errs = np.pad(errs, (0, 4 - len(errs)))
    return errs.astype(float)


def select_error_stratified_indices(
    candidate_indices: np.ndarray,
    pool_dates: np.ndarray,
    pool_error_profiles: np.ndarray,
    k_examples: int,
) -> np.ndarray:
    """Pick a small but diverse set of teaching examples across error modes."""
    candidate_indices = np.asarray(candidate_indices, dtype=int)
    if candidate_indices.size <= int(k_examples):
        return candidate_indices[np.argsort(pool_dates[candidate_indices])]

    cand_profiles = np.asarray(pool_error_profiles[candidate_indices], dtype=float)
    center = np.mean(cand_profiles, axis=0, keepdims=True)
    scale = np.std(cand_profiles, axis=0, keepdims=True)
    scale[scale < 1e-6] = 1.0
    norm_profiles = (cand_profiles - center) / scale
    norms = np.linalg.norm(norm_profiles, axis=1)

    selected_pos = [int(np.argmin(norms))]
    while len(selected_pos) < int(k_examples):
        remaining = [i for i in range(len(candidate_indices)) if i not in selected_pos]
        if not remaining:
            break
        scored = []
        for pos in remaining:
            min_dist = min(
                float(np.linalg.norm(norm_profiles[pos] - norm_profiles[sel]))
                for sel in selected_pos
            )
            recency = float(candidate_indices[pos])
            scored.append((min_dist, recency, pos))
        scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
        selected_pos.append(int(scored[0][2]))

    selected = candidate_indices[np.asarray(selected_pos, dtype=int)]
    return selected[np.argsort(pool_dates[selected])]


def _case_profile_features(
    history: np.ndarray,
    forecast: np.ndarray | None = None,
    feature_window: int = 20,
) -> np.ndarray:
    """Compact regime features for example matching."""
    hist = np.asarray(history, dtype=float)
    fc = None if forecast is None else np.asarray(forecast, dtype=float)
    if hist.size == 0:
        return np.zeros(7, dtype=float)

    window = hist[-feature_window:] if hist.size >= feature_window else hist
    last_price = float(window[-1])
    change_5 = 0.0
    change_20 = 0.0
    if window.size >= 5 and abs(window[-5]) > 1e-8:
        change_5 = float((window[-1] / window[-5] - 1.0) * 100.0)
    if window.size >= 20 and abs(window[-20]) > 1e-8:
        change_20 = float((window[-1] / window[-20] - 1.0) * 100.0)
    vol_20 = float(np.std(window[-20:])) if window.size else 0.0

    fc_h5 = 0.0
    fc_h20 = 0.0
    fc_h30 = 0.0
    if fc is not None and fc.size:
        if fc.size >= 5 and abs(last_price) > 1e-8:
            fc_h5 = float((fc[4] / last_price - 1.0) * 100.0)
        if fc.size >= 20 and abs(last_price) > 1e-8:
            fc_h20 = float((fc[19] / last_price - 1.0) * 100.0)
        if fc.size >= 30 and abs(last_price) > 1e-8:
            fc_h30 = float((fc[29] / last_price - 1.0) * 100.0)

    return np.array(
        [last_price, change_5, change_20, vol_20, fc_h5, fc_h20, fc_h30],
        dtype=float,
    )


def select_similarity_error_hybrid_indices(
    candidate_indices: np.ndarray,
    pool_dates: np.ndarray,
    pool_case_profiles: np.ndarray,
    pool_error_profiles: np.ndarray,
    reference_profile: np.ndarray,
    k_examples: int,
    n_similarity: int = 3,
) -> np.ndarray:
    """Mix nearest-regime examples with diverse error modes."""
    candidate_indices = np.asarray(candidate_indices, dtype=int)
    if candidate_indices.size <= int(k_examples):
        return candidate_indices[np.argsort(pool_dates[candidate_indices])]

    k_examples = int(min(k_examples, candidate_indices.size))
    n_similarity = int(max(1, min(n_similarity, k_examples)))

    cand_profiles = np.asarray(pool_case_profiles[candidate_indices], dtype=float)
    ref_profile = np.asarray(reference_profile, dtype=float).reshape(1, -1)
    center = np.mean(cand_profiles, axis=0, keepdims=True)
    scale = np.std(cand_profiles, axis=0, keepdims=True)
    scale[scale < 1e-6] = 1.0
    distances = np.linalg.norm((cand_profiles - ref_profile) / scale, axis=1)
    sim_order = np.argsort(distances)
    selected = list(candidate_indices[sim_order[:n_similarity]])

    remaining = np.asarray([idx for idx in candidate_indices if idx not in set(selected)], dtype=int)
    n_remaining = k_examples - len(selected)
    if n_remaining > 0 and remaining.size > 0:
        diverse = select_error_stratified_indices(
            candidate_indices=remaining,
            pool_dates=pool_dates,
            pool_error_profiles=pool_error_profiles,
            k_examples=n_remaining,
        )
        selected.extend(int(idx) for idx in diverse.tolist())

    selected_arr = np.asarray(sorted(set(selected)), dtype=int)
    if selected_arr.size > k_examples:
        selected_arr = selected_arr[:k_examples]
    return selected_arr[np.argsort(pool_dates[selected_arr])]


def _minmax_scale(values: np.ndarray) -> np.ndarray:
    """Scale a vector into [0, 1] with a safe constant fallback."""
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return arr
    lo = float(np.min(arr))
    hi = float(np.max(arr))
    if hi - lo < 1e-8:
        return np.full(arr.shape, 0.5, dtype=float)
    return (arr - lo) / (hi - lo)


def _sign_bucket(value: float, tol: float) -> str:
    """Map a signed scalar into a coarse bucket."""
    if value >= tol:
        return "pos"
    if value <= -tol:
        return "neg"
    return "flat"


def _profile_regime_tag(profile: np.ndarray) -> str:
    """Compact regime tag used for retrieval filtering and prompt attribute marking."""
    profile = np.asarray(profile, dtype=float)
    if profile.size < 7:
        return "trend5=flat|trend20=flat|drift20=flat|drift30=flat|vol=normal"
    last_price = max(abs(float(profile[0])), 1e-8)
    vol_pct = float(abs(profile[3]) / last_price * 100.0)
    if vol_pct < 1.0:
        vol_bucket = "calm"
    elif vol_pct < 2.5:
        vol_bucket = "normal"
    else:
        vol_bucket = "elevated"
    return (
        f"trend5={_sign_bucket(float(profile[1]), 0.75)}|"
        f"trend20={_sign_bucket(float(profile[2]), 1.50)}|"
        f"drift20={_sign_bucket(float(profile[5]), 0.75)}|"
        f"drift30={_sign_bucket(float(profile[6]), 1.00)}|"
        f"vol={vol_bucket}"
    )


def _regime_match_score(candidate_profile: np.ndarray, reference_profile: np.ndarray) -> float:
    """Score coarse regime compatibility between a candidate and the current case."""
    cand = np.asarray(candidate_profile, dtype=float)
    ref = np.asarray(reference_profile, dtype=float)
    if cand.size < 7 or ref.size < 7:
        return 0.0
    cand_vol = _profile_regime_tag(cand).split("|")[-1]
    ref_vol = _profile_regime_tag(ref).split("|")[-1]
    score = 0.0
    if _sign_bucket(float(cand[1]), 0.75) == _sign_bucket(float(ref[1]), 0.75):
        score += 1.0
    if _sign_bucket(float(cand[2]), 1.50) == _sign_bucket(float(ref[2]), 1.50):
        score += 1.0
    if _sign_bucket(float(cand[5]), 0.75) == _sign_bucket(float(ref[5]), 0.75):
        score += 1.5
    if _sign_bucket(float(cand[6]), 1.00) == _sign_bucket(float(ref[6]), 1.00):
        score += 2.0
    if cand_vol == ref_vol:
        score += 0.5
    return score


def select_utility_score_indices(
    candidate_indices: np.ndarray,
    pool_dates: np.ndarray,
    pool_case_profiles: np.ndarray,
    pool_path_mse: np.ndarray,
    reference_profile: np.ndarray,
    k_examples: int,
    similarity_weight: float = 0.45,
    error_weight: float = 0.35,
    recency_weight: float = 0.20,
    regime_scores: np.ndarray | None = None,
    regime_weight: float = 0.0,
) -> np.ndarray:
    """Rank examples by a utility score combining similarity, correction value, and recency."""
    candidate_indices = np.asarray(candidate_indices, dtype=int)
    if candidate_indices.size <= int(k_examples):
        return candidate_indices[np.argsort(pool_dates[candidate_indices])]

    cand_profiles = np.asarray(pool_case_profiles[candidate_indices], dtype=float)
    ref_profile = np.asarray(reference_profile, dtype=float).reshape(1, -1)
    center = np.mean(cand_profiles, axis=0, keepdims=True)
    scale = np.std(cand_profiles, axis=0, keepdims=True)
    scale[scale < 1e-6] = 1.0
    distances = np.linalg.norm((cand_profiles - ref_profile) / scale, axis=1)
    sim_score = 1.0 - _minmax_scale(distances)
    err_score = _minmax_scale(pool_path_mse[candidate_indices])
    recency_score = _minmax_scale(candidate_indices.astype(float))
    total = (
        float(similarity_weight) * sim_score
        + float(error_weight) * err_score
        + float(recency_weight) * recency_score
    )
    if regime_scores is not None:
        total = total + float(regime_weight) * _minmax_scale(np.asarray(regime_scores, dtype=float))

    order = np.argsort(total)[-int(k_examples):]
    selected = candidate_indices[np.asarray(order, dtype=int)]
    return selected[np.argsort(pool_dates[selected])]


def select_utility_mmr_indices(
    candidate_indices: np.ndarray,
    pool_dates: np.ndarray,
    pool_case_profiles: np.ndarray,
    pool_path_mse: np.ndarray,
    reference_profile: np.ndarray,
    k_examples: int,
    similarity_weight: float = 0.45,
    error_weight: float = 0.35,
    recency_weight: float = 0.20,
    mmr_lambda: float = 0.75,
    regime_scores: np.ndarray | None = None,
    regime_weight: float = 0.0,
) -> np.ndarray:
    """Greedy MMR selection over utility-scored candidates to reduce redundancy."""
    candidate_indices = np.asarray(candidate_indices, dtype=int)
    if candidate_indices.size <= int(k_examples):
        return candidate_indices[np.argsort(pool_dates[candidate_indices])]

    cand_profiles = np.asarray(pool_case_profiles[candidate_indices], dtype=float)
    ref_profile = np.asarray(reference_profile, dtype=float).reshape(1, -1)
    center = np.mean(cand_profiles, axis=0, keepdims=True)
    scale = np.std(cand_profiles, axis=0, keepdims=True)
    scale[scale < 1e-6] = 1.0
    norm_profiles = (cand_profiles - center) / scale
    profile_norms = np.linalg.norm(norm_profiles, axis=1, keepdims=True)
    profile_norms[profile_norms < 1e-6] = 1.0
    unit_profiles = norm_profiles / profile_norms

    distances = np.linalg.norm((cand_profiles - ref_profile) / scale, axis=1)
    sim_score = 1.0 - _minmax_scale(distances)
    err_score = _minmax_scale(pool_path_mse[candidate_indices])
    recency_score = _minmax_scale(candidate_indices.astype(float))
    utility = (
        float(similarity_weight) * sim_score
        + float(error_weight) * err_score
        + float(recency_weight) * recency_score
    )
    if regime_scores is not None:
        utility = utility + float(regime_weight) * _minmax_scale(np.asarray(regime_scores, dtype=float))

    selected_pos: list[int] = [int(np.argmax(utility))]
    while len(selected_pos) < int(min(k_examples, candidate_indices.size)):
        remaining = [pos for pos in range(candidate_indices.size) if pos not in selected_pos]
        if not remaining:
            break
        scored = []
        for pos in remaining:
            redundancy = max(
                float(np.dot(unit_profiles[pos], unit_profiles[sel]))
                for sel in selected_pos
            )
            score = float(mmr_lambda) * float(utility[pos]) - (1.0 - float(mmr_lambda)) * redundancy
            scored.append((score, float(utility[pos]), float(candidate_indices[pos]), pos))
        scored.sort(key=lambda item: (item[0], item[1], item[2]), reverse=True)
        selected_pos.append(int(scored[0][3]))

    selected = candidate_indices[np.asarray(selected_pos, dtype=int)]
    return selected[np.argsort(pool_dates[selected])]


def select_top_score_indices(
    candidate_indices: np.ndarray,
    pool_dates: np.ndarray,
    scores: np.ndarray,
    k_examples: int,
) -> np.ndarray:
    """Select the highest-scoring examples, breaking ties by recency."""
    candidate_indices = np.asarray(candidate_indices, dtype=int)
    if candidate_indices.size <= int(k_examples):
        return candidate_indices[np.argsort(pool_dates[candidate_indices])]
    scored = [(float(scores[idx]), float(idx), int(idx)) for idx in candidate_indices]
    scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
    selected = np.asarray([idx for _, _, idx in scored[: int(k_examples)]], dtype=int)
    return selected[np.argsort(pool_dates[selected])]


def select_balanced_long_horizon_indices(
    candidate_indices: np.ndarray,
    pool_dates: np.ndarray,
    h20_scores: np.ndarray,
    h30_scores: np.ndarray,
    k_examples: int,
) -> np.ndarray:
    """Balance teaching examples across h20 and h30 error modes."""
    candidate_indices = np.asarray(candidate_indices, dtype=int)
    if candidate_indices.size <= int(k_examples):
        return candidate_indices[np.argsort(pool_dates[candidate_indices])]

    target_k = int(min(k_examples, candidate_indices.size))
    h20_k = max(1, int(np.ceil(target_k / 2.0)))
    h30_k = max(1, target_k - h20_k)
    selected: list[int] = []

    for idx in select_top_score_indices(candidate_indices, pool_dates, h20_scores, h20_k):
        if int(idx) not in selected:
            selected.append(int(idx))

    remaining = np.asarray([idx for idx in candidate_indices if int(idx) not in selected], dtype=int)
    if remaining.size > 0:
        for idx in select_top_score_indices(remaining, pool_dates, h30_scores, h30_k):
            if int(idx) not in selected:
                selected.append(int(idx))

    if len(selected) < target_k:
        remaining = np.asarray([idx for idx in candidate_indices if int(idx) not in selected], dtype=int)
        combined = np.asarray(h20_scores, dtype=float) + np.asarray(h30_scores, dtype=float)
        for idx in select_top_score_indices(remaining, pool_dates, combined, target_k - len(selected)):
            if int(idx) not in selected:
                selected.append(int(idx))

    selected_arr = np.asarray(selected[:target_k], dtype=int)
    return selected_arr[np.argsort(pool_dates[selected_arr])]


def select_prototype_indices(
    candidate_indices: np.ndarray,
    pool_dates: np.ndarray,
    pool_case_profiles: np.ndarray,
    scores: np.ndarray,
    k_examples: int,
    top_pool_size: int,
) -> np.ndarray:
    """Pick diverse prototypes from the top-scoring candidate pool."""
    candidate_indices = np.asarray(candidate_indices, dtype=int)
    if candidate_indices.size <= int(k_examples):
        return candidate_indices[np.argsort(pool_dates[candidate_indices])]

    top_pool = select_top_score_indices(
        candidate_indices,
        pool_dates,
        scores,
        min(int(top_pool_size), candidate_indices.size),
    )
    if top_pool.size <= int(k_examples):
        return top_pool[np.argsort(pool_dates[top_pool])]

    profiles = np.asarray(pool_case_profiles[top_pool], dtype=float)
    center = np.mean(profiles, axis=0, keepdims=True)
    scale = np.std(profiles, axis=0, keepdims=True)
    scale[scale < 1e-6] = 1.0
    norm_profiles = (profiles - center) / scale

    selected_pos = [0]
    while len(selected_pos) < int(k_examples):
        remaining = [i for i in range(len(top_pool)) if i not in selected_pos]
        if not remaining:
            break
        best_pos = remaining[0]
        best_tuple = None
        for pos in remaining:
            min_dist = min(
                float(np.linalg.norm(norm_profiles[pos] - norm_profiles[sel]))
                for sel in selected_pos
            )
            score = float(scores[int(top_pool[pos])])
            recency = float(top_pool[pos])
            candidate_tuple = (min_dist, score, recency)
            if best_tuple is None or candidate_tuple > best_tuple:
                best_tuple = candidate_tuple
                best_pos = pos
        selected_pos.append(int(best_pos))

    selected = top_pool[np.asarray(selected_pos, dtype=int)]
    return selected[np.argsort(pool_dates[selected])]


def _tag_component_overlap(tag_a: str, tag_b: str) -> int:
    """Count exact overlaps between two compact regime tags."""
    left = {part.strip() for part in str(tag_a or "").split("|") if part.strip()}
    right = {part.strip() for part in str(tag_b or "").split("|") if part.strip()}
    return int(len(left.intersection(right)))


def select_skill_tag_recent_high_error_indices(
    candidate_indices: np.ndarray,
    pool_dates: np.ndarray,
    pool_tags: np.ndarray,
    pool_error_scores: np.ndarray,
    reference_tag: str,
    k_examples: int,
    min_overlap: int = 3,
) -> np.ndarray:
    """Prioritize recent hard cases that share the same coarse regime tag."""
    candidate_indices = np.asarray(candidate_indices, dtype=int)
    if candidate_indices.size <= int(k_examples):
        return candidate_indices[np.argsort(pool_dates[candidate_indices])]

    overlaps = np.asarray(
        [_tag_component_overlap(pool_tags[idx], reference_tag) for idx in candidate_indices],
        dtype=int,
    )
    keep_mask = overlaps >= int(min_overlap)
    filtered = candidate_indices[keep_mask]
    if filtered.size == 0:
        best_overlap = int(np.max(overlaps)) if overlaps.size else 0
        filtered = candidate_indices[overlaps == best_overlap]
    return select_top_score_indices(filtered, pool_dates, pool_error_scores, k_examples)


def select_counterexample_index(
    candidate_indices: np.ndarray,
    pool_case_profiles: np.ndarray,
    reference_profile: np.ndarray,
    pool_metric: np.ndarray,
    exclude_indices: Sequence[int],
    quantile: float = 0.35,
) -> Optional[int]:
    """Pick a similar low-error or low-gain counterexample for contrastive prompting."""
    candidate_indices = np.asarray(candidate_indices, dtype=int)
    exclude = {int(x) for x in exclude_indices}
    remaining = np.asarray([idx for idx in candidate_indices if int(idx) not in exclude], dtype=int)
    if remaining.size == 0:
        return None

    metric_values = np.asarray(pool_metric[remaining], dtype=float)
    cutoff = float(np.quantile(metric_values, float(quantile)))
    filtered = remaining[metric_values <= cutoff]
    if filtered.size == 0:
        order = np.argsort(metric_values)
        filtered = remaining[order[: max(1, min(3, remaining.size))]]

    profiles = np.asarray(pool_case_profiles[filtered], dtype=float)
    ref = np.asarray(reference_profile, dtype=float).reshape(1, -1)
    center = np.mean(profiles, axis=0, keepdims=True)
    scale = np.std(profiles, axis=0, keepdims=True)
    scale[scale < 1e-6] = 1.0
    distances = np.linalg.norm((profiles - ref) / scale, axis=1)
    return int(filtered[int(np.argmin(distances))])


def summarize_hindsight_feedback(
    anchor_error_pct: dict[int, float],
    mode: str = "standard",
) -> str:
    """Compress historical base-model mistakes into one short feedback sentence."""
    if not anchor_error_pct:
        return ""

    mode_key = str(mode or "standard").strip().lower()

    if mode_key == "compact_tags":
        tags: list[str] = []
        err5 = float(anchor_error_pct.get(5, 0.0))
        err20 = float(anchor_error_pct.get(20, 0.0))
        err30 = float(anchor_error_pct.get(30, 0.0))
        if err20 > 0.0 and err30 > 0.0:
            tags.append("long_tail=overshoot")
        elif err20 < 0.0 and err30 < 0.0:
            tags.append("long_tail=undershoot")
        elif 20 in anchor_error_pct or 30 in anchor_error_pct:
            tags.append("long_tail=mixed")
        if abs(err5) > 0.0:
            tags.append("h5=" + ("overshoot" if err5 > 0.0 else "undershoot"))
        dominant_h, dominant_err = max(
            ((int(h), float(v)) for h, v in anchor_error_pct.items()),
            key=lambda item: abs(item[1]),
        )
        tags.append(f"dominant=h{dominant_h}")
        if dominant_err == 0.0:
            tags.append("action=freeze")
        else:
            tags.append("action=" + ("down" if dominant_err > 0.0 else "up"))
        return "; ".join(tags)

    if mode_key == "long_horizon_only":
        long_only = {int(h): float(v) for h, v in anchor_error_pct.items() if int(h) >= 20}
        if long_only:
            anchor_error_pct = long_only

    if 20 in anchor_error_pct and 30 in anchor_error_pct:
        err20 = float(anchor_error_pct[20])
        err30 = float(anchor_error_pct[30])
        if err20 > 0.0 and err30 > 0.0:
            return "Base overshot the long horizon; downward h20/h30 correction would have helped."
        if err20 < 0.0 and err30 < 0.0:
            return "Base undershot the long horizon; upward h20/h30 correction would have helped."
        return "Long-horizon error signs split; avoid forcing one tail direction across h20 and h30."

    horizon, err = max(
        ((int(h), float(v)) for h, v in anchor_error_pct.items()),
        key=lambda item: abs(item[1]),
    )
    if err > 0.0:
        return f"Base overshot most at h{horizon}; a bounded downward correction was warranted."
    if err < 0.0:
        return f"Base undershot most at h{horizon}; a bounded upward correction was warranted."
    return f"Base was already aligned at h{horizon}; freeze was acceptable."


def build_aux_teacher_case_summary(
    base_forecast: np.ndarray,
    aux_forecast: np.ndarray,
    teacher_name: str = "linear_ridge",
    truth: Optional[np.ndarray] = None,
    horizons: Sequence[int] = (5, 20, 30),
) -> dict[str, str]:
    """Summarize what an auxiliary teacher implies relative to the base forecast."""
    base_arr = np.asarray(base_forecast, dtype=float)
    aux_arr = np.asarray(aux_forecast, dtype=float)
    if base_arr.size == 0 or aux_arr.size == 0:
        return {}

    label = str(teacher_name).replace("_", " ")
    delta_parts: list[str] = []
    for horizon in horizons:
        idx = int(horizon) - 1
        if idx < 0 or idx >= base_arr.size or idx >= aux_arr.size:
            continue
        base_val = float(base_arr[idx])
        aux_val = float(aux_arr[idx])
        delta_pct = 0.0 if abs(base_val) < 1e-8 else float((aux_val - base_val) / base_val * 100.0)
        delta_parts.append(f"h{int(horizon)}={delta_pct:+.2f}%")

    summary: dict[str, str] = {}
    if delta_parts:
        summary[f"{label}_delta_pct"] = "; ".join(delta_parts)

    if truth is not None:
        truth_arr = np.asarray(truth, dtype=float)
        usable = [int(h) for h in horizons if 0 <= int(h) - 1 < truth_arr.size]
        if usable:
            idx = [int(h) - 1 for h in usable]
            base_mse = float(np.mean((base_arr[idx] - truth_arr[idx]) ** 2))
            aux_mse = float(np.mean((aux_arr[idx] - truth_arr[idx]) ** 2))
            gain = base_mse - aux_mse
            summary[f"{label}_gain_long"] = f"{gain:+.3f} MSE vs base on h{usable[0]}-h{usable[-1]}"
    return summary


def augment_exogenous_summary_with_teacher(
    exogenous_summary: Optional[dict],
    base_forecast: np.ndarray,
    aux_forecast: np.ndarray,
    teacher_name: str = "linear_ridge",
    horizons: Sequence[int] = (5, 20, 30),
) -> dict:
    """Append auxiliary-teacher hints to the current-case context block."""
    merged = dict(exogenous_summary or {})
    merged.update(
        build_aux_teacher_case_summary(
            base_forecast=base_forecast,
            aux_forecast=aux_forecast,
            teacher_name=teacher_name,
            truth=None,
            horizons=horizons,
        )
    )
    return merged


def normalize_selection_bundle(selection: object) -> dict[str, object]:
    """Normalize a selection result into indices plus optional per-example roles."""
    if isinstance(selection, dict):
        indices = np.asarray(selection.get("indices", []), dtype=int)
        roles = {
            int(k): str(v)
            for k, v in (selection.get("roles") or {}).items()
            if str(v).strip()
        }
        return {"indices": indices, "roles": roles}
    return {"indices": np.asarray(selection, dtype=int), "roles": {}}


LLM_BASE_MODEL_ALIASES = {
    "tsm": "tsm",
    "ridge": "linear_ridge",
    "linear": "linear_ridge",
    "linear_ridge": "linear_ridge",
    "linear-ridge": "linear_ridge",
    "lasso": "linear_lasso",
    "linear_lasso": "linear_lasso",
    "linear-lasso": "linear_lasso",
    "naive": "naive_persistence",
    "naive_persistence": "naive_persistence",
    "seasonal": "seasonal_naive",
    "seasonal_naive": "seasonal_naive",
    "seasonal-naive": "seasonal_naive",
}


def normalize_llm_base_model_name(base_model: str | None) -> str:
    """Normalize the configured LLM base model name to an internal model key."""
    raw = str(base_model or "tsm").strip().lower().replace("-", "_")
    return LLM_BASE_MODEL_ALIASES.get(raw, raw)


def llm_method_requires_base_forecast(method: str | None) -> bool:
    """Whether an LLM method requires an existing quantitative forecast path."""
    return str(method or "").upper().startswith("TSM+")


def llm_result_name(method: str, base_model: str | None = None) -> str:
    """Map an internal LLM method name to the reported model name."""
    method = str(method)
    model_name = normalize_llm_base_model_name(base_model)
    if model_name == "tsm" or not llm_method_requires_base_forecast(method):
        return method
    if method.startswith("TSM"):
        return f"{model_name}{method[len('TSM'):]}"
    return f"{model_name}_{method}"


def infer_llm_market_name(target_cfg: dict) -> str:
    """Infer a human-readable market label for prompt context."""
    instrument = str((target_cfg or {}).get("instrument", "")).upper()
    if "UKA" in instrument:
        return "UK ETS carbon allowance market"
    if "EUA" in instrument:
        return "EU ETS carbon allowance market"
    if instrument:
        return f"{instrument} carbon allowance market"
    return "carbon allowance market"


def build_llm_refiner_config(config) -> dict:
    """Merge llm config with top-level method-specific subconfigs for the refiner."""
    llm_refiner_config = copy.deepcopy(config.llm)
    llm_refiner_config.setdefault("currency", config.target.get("currency", "EUR"))
    llm_refiner_config.setdefault("market_name", infer_llm_market_name(config.target))
    for key in ("delta", "hdelta", "hprice", "norm_delta", "news_drift"):
        section = None
        raw_cfg = getattr(config, "raw", None)
        if isinstance(raw_cfg, dict):
            section = raw_cfg.get(key)
        if section is None:
            section = getattr(config, key, None)
        if section is not None:
            llm_refiner_config[key] = copy.deepcopy(section)
    return llm_refiner_config


def _rule_gate_confidence_rank(value: object) -> int:
    """Map structured-guidance confidence labels to an ordinal rank."""
    return {"low": 0, "medium": 1, "high": 2}.get(str(value or "").strip().lower(), -1)


def _rule_gate_adjustment_pct(base_value: float, llm_value: float) -> float:
    """Compute percentage adjustment from base to LLM forecast at a horizon."""
    base = float(base_value)
    llm = float(llm_value)
    if not np.isfinite(base) or abs(base) < 1e-8:
        return 0.0
    return float((llm / base - 1.0) * 100.0)


def _rule_gate_sign(value: float, eps: float = 1e-12) -> int:
    """Return the signed direction of an adjustment."""
    if value > eps:
        return 1
    if value < -eps:
        return -1
    return 0


def build_rule_gate_feature_frame(
    base_pred: np.ndarray,
    llm_pred: np.ndarray,
    metadata: Sequence[dict],
    key_horizons: Sequence[int] = (1, 5, 20, 30),
) -> pd.DataFrame:
    """Extract auditable per-sample gate features from structured HDELTA outputs."""
    base_arr = np.asarray(base_pred, dtype=float)
    llm_arr = np.asarray(llm_pred, dtype=float)
    if base_arr.shape != llm_arr.shape:
        raise ValueError(f"Shape mismatch: base_pred={base_arr.shape}, llm_pred={llm_arr.shape}")
    if base_arr.shape[0] != len(metadata):
        raise ValueError(
            f"Metadata length mismatch: {len(metadata)} rows for {base_arr.shape[0]} predictions"
        )

    rows: list[dict] = []
    for row_idx, meta in enumerate(metadata):
        payload = meta or {}
        guidance = payload.get("structured_horizon_guidance") or {}
        matched_dates = payload.get("matched_teaching_dates") or []
        dynamic_frozen = payload.get("dynamic_frozen_horizons") or []
        row: dict[str, object] = {
            "row_idx": int(row_idx),
            "success": bool(payload.get("success", False)),
            "fallback": bool(payload.get("fallback", False)),
            "matched_teaching_count": int(len(matched_dates)),
            "dynamic_frozen_count": int(len(dynamic_frozen)),
        }
        for raw_h in key_horizons:
            horizon = int(raw_h)
            col_idx = horizon - 1
            if col_idx < 0 or col_idx >= base_arr.shape[1]:
                continue
            delta_pct = _rule_gate_adjustment_pct(base_arr[row_idx, col_idx], llm_arr[row_idx, col_idx])
            guidance_payload = guidance.get(horizon) or {}
            confidence = str(guidance_payload.get("confidence", "")).strip().lower()
            row[f"h{horizon}_delta_pct"] = float(delta_pct)
            row[f"h{horizon}_sign"] = int(_rule_gate_sign(delta_pct))
            row[f"h{horizon}_mode"] = str(guidance_payload.get("mode", "")).strip().lower()
            row[f"h{horizon}_confidence"] = confidence
            row[f"h{horizon}_confidence_rank"] = int(_rule_gate_confidence_rank(confidence))
            row[f"h{horizon}_magnitude"] = str(guidance_payload.get("magnitude", "")).strip().lower()
        rows.append(row)
    return pd.DataFrame(rows)


def build_rule_gate_candidate_masks(
    feature_df: pd.DataFrame,
    candidates: Optional[Sequence[str]] = None,
    low_match_threshold: int = 2,
    abs_h30_thresholds_pct: Sequence[float] = (0.25, 0.40),
) -> dict[str, np.ndarray]:
    """Build a family of sample-level reject masks for validation-based gate selection."""
    if feature_df.empty:
        return {"none": np.zeros(0, dtype=bool)}

    h5_sign = feature_df.get("h5_sign", pd.Series(0, index=feature_df.index)).to_numpy(dtype=int)
    h20_sign = feature_df.get("h20_sign", pd.Series(0, index=feature_df.index)).to_numpy(dtype=int)
    h30_sign = feature_df.get("h30_sign", pd.Series(0, index=feature_df.index)).to_numpy(dtype=int)
    h30_abs = np.abs(
        feature_df.get("h30_delta_pct", pd.Series(0.0, index=feature_df.index)).to_numpy(dtype=float)
    )
    matched_count = feature_df.get(
        "matched_teaching_count",
        pd.Series(0, index=feature_df.index),
    ).to_numpy(dtype=int)
    h20_conf = feature_df.get(
        "h20_confidence_rank",
        pd.Series(-1, index=feature_df.index),
    ).to_numpy(dtype=int)
    h30_conf = feature_df.get(
        "h30_confidence_rank",
        pd.Series(-1, index=feature_df.index),
    ).to_numpy(dtype=int)

    h5_pos = h5_sign > 0
    h20_neg = h20_sign < 0
    h30_neg = h30_sign < 0
    h30_zero = h30_sign == 0
    mixed_long_signs = (h20_sign != 0) & (h30_sign != 0) & (h20_sign != h30_sign)
    conflict_any = (h5_pos & (h20_neg | h30_neg)) | (h20_neg & h30_zero) | mixed_long_signs
    nonhigh_long_conf = (h20_conf < 2) | (h30_conf < 2)
    low_match = matched_count <= int(low_match_threshold)

    masks: dict[str, np.ndarray] = {
        "none": np.zeros(len(feature_df), dtype=bool),
        "h5_pos_h20_neg": h5_pos & h20_neg,
        "h5_pos_h30_neg": h5_pos & h30_neg,
        "h20_neg_h30_zero": h20_neg & h30_zero,
        "mixed_long_signs": mixed_long_signs,
        "conflict_any": conflict_any,
        "conflict_any_nonhigh_conf": conflict_any & nonhigh_long_conf,
        "conflict_any_low_match": conflict_any & low_match,
        "conflict_any_low_match_or_nonhigh_conf": conflict_any & (low_match | nonhigh_long_conf),
        "h30_negative": h30_neg,
    }
    for raw_threshold in abs_h30_thresholds_pct:
        threshold = float(raw_threshold)
        name = f"abs_h30_ge_{threshold:.2f}".replace(".", "_")
        conflict_name = f"conflict_any_or_abs_h30_ge_{threshold:.2f}".replace(".", "_")
        over_threshold = h30_abs >= threshold
        masks[name] = over_threshold
        masks[conflict_name] = conflict_any | over_threshold

    if candidates is None:
        ordered_names = list(RULE_GATE_DEFAULT_CANDIDATES)
    else:
        ordered_names = [str(name) for name in candidates]
    selected = {
        name: np.asarray(masks[name], dtype=bool)
        for name in ordered_names
        if name in masks
    }
    if not selected:
        selected["none"] = np.zeros(len(feature_df), dtype=bool)
    return selected


def apply_rule_gate(
    base_pred: np.ndarray,
    llm_pred: np.ndarray,
    reject_mask: np.ndarray,
) -> np.ndarray:
    """Fallback flagged samples to the base forecast while keeping accepted LLM outputs."""
    base_arr = np.asarray(base_pred, dtype=float)
    llm_arr = np.asarray(llm_pred, dtype=float)
    mask = np.asarray(reject_mask, dtype=bool).reshape(-1)
    if base_arr.shape != llm_arr.shape:
        raise ValueError(f"Shape mismatch: base_pred={base_arr.shape}, llm_pred={llm_arr.shape}")
    if base_arr.shape[0] != mask.shape[0]:
        raise ValueError(
            f"Reject-mask length mismatch: {mask.shape[0]} rows for {base_arr.shape[0]} predictions"
        )
    gated = np.array(llm_arr, copy=True)
    gated[mask] = base_arr[mask]
    return gated


def evaluate_rule_gate_candidates(
    y_true: np.ndarray,
    base_pred: np.ndarray,
    llm_pred: np.ndarray,
    feature_df: pd.DataFrame,
    candidates: Optional[Sequence[str]] = None,
    low_match_threshold: int = 2,
    abs_h30_thresholds_pct: Sequence[float] = (0.25, 0.40),
    horizons: Sequence[int] = (1, 5, 20, 30),
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    """Evaluate rule-gate candidates by replacing rejected samples with the base forecast."""
    candidate_masks = build_rule_gate_candidate_masks(
        feature_df,
        candidates=candidates,
        low_match_threshold=low_match_threshold,
        abs_h30_thresholds_pct=abs_h30_thresholds_pct,
    )

    rows: list[dict] = []
    gated_predictions: dict[str, np.ndarray] = {}
    for name, reject_mask in candidate_masks.items():
        gated = apply_rule_gate(base_pred, llm_pred, reject_mask)
        gated_predictions[name] = gated
        path_metrics = compute_path_metrics(y_true, gated)
        row = {
            "candidate": name,
            "flagged_count": int(np.sum(reject_mask)),
            "flagged_share": float(np.mean(reject_mask)) if len(reject_mask) else 0.0,
            **path_metrics,
        }
        by_horizon = compute_metrics_by_horizon(y_true, gated, list(horizons))
        for horizon in horizons:
            if int(horizon) in by_horizon.index:
                row[f"h{int(horizon)}_mse"] = float(by_horizon.loc[int(horizon), "mse"])
        rows.append(row)

    summary_df = pd.DataFrame(rows)
    if not summary_df.empty and "mse_path" in summary_df.columns:
        summary_df = summary_df.sort_values(
            by=["mse_path", "flagged_count", "candidate"],
            ascending=[True, True, True],
        ).reset_index(drop=True)
    return summary_df, gated_predictions


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


def run_experiment(
    config_path: str = None,
    overrides: dict = None,
    data_dir: Optional[str] = None,
):
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
    
    if data_dir:
        resolved_data_dir = Path(data_dir).expanduser()
    else:
        resolved_data_dir = Path(__file__).parent.parent / "Data"

    logger.info(f"Using data directory: {resolved_data_dir.resolve()}")
    
    panel, schema = build_panel(
        data_dir=resolved_data_dir,
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
    
    feature_cols = select_feature_columns(
        panel,
        target_col=target_col,
        max_exogenous_features=(config.features or {}).get("max_exogenous_features_model", 10),
        preferred_feature_order=(config.features or {}).get("preferred_feature_order"),
    )
    
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
    
    X_enc, X_dec, y, dates, window_meta = make_windows(
        panel,
        window_config,
        mode='MS',
        return_metadata=True,
    )
    
    # Split data
    splits = split_windows(
        X_enc, X_dec, y, dates,
        train_end=config.split.get('train_end', '2022-12-31'),
        val_end=config.split.get('val_end', '2023-12-31'),
        window_meta=window_meta,
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
    llm_base_predictions = {}
    
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
    llm_base_predictions['naive_persistence'] = {
        'train': naive.predict(y_train_hist),
        'val': naive.predict(y_val_hist),
        'test': naive_pred,
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
    llm_base_predictions['seasonal_naive'] = {
        'train': seasonal.predict(y_train_hist),
        'val': seasonal.predict(y_val_hist),
        'test': seasonal_pred,
    }

    # Linear baselines (ridge + lasso)
    try:
        linear_ridge = LinearBaseline(config.pred_len, model_type="ridge")
        linear_ridge.fit(y_train_hist, y_train_future)
        linear_ridge_train_pred = linear_ridge.predict(y_train_hist)
        linear_ridge_val_pred = linear_ridge.predict(y_val_hist)
        linear_ridge_pred = linear_ridge.predict(y_test_hist)
        baseline_results['linear_ridge'] = {
            'predictions': linear_ridge_pred,
            'metrics': compute_metrics_by_horizon(y_test_future, linear_ridge_pred, horizons),
            'path_metrics': compute_path_metrics(y_test_future, linear_ridge_pred),
            'errors': get_per_sample_errors(y_test_future, linear_ridge_pred, horizons)
        }
        llm_base_predictions['linear_ridge'] = {
            'train': linear_ridge_train_pred,
            'val': linear_ridge_val_pred,
            'test': linear_ridge_pred,
        }
    except Exception as e:
        logger.warning(f"Linear ridge baseline failed: {e}")

    try:
        linear_lasso = LinearBaseline(config.pred_len, model_type="lasso")
        linear_lasso.fit(y_train_hist, y_train_future)
        linear_lasso_train_pred = linear_lasso.predict(y_train_hist)
        linear_lasso_val_pred = linear_lasso.predict(y_val_hist)
        linear_lasso_pred = linear_lasso.predict(y_test_hist)
        baseline_results['linear_lasso'] = {
            'predictions': linear_lasso_pred,
            'metrics': compute_metrics_by_horizon(y_test_future, linear_lasso_pred, horizons),
            'path_metrics': compute_path_metrics(y_test_future, linear_lasso_pred),
            'errors': get_per_sample_errors(y_test_future, linear_lasso_pred, horizons)
        }
        llm_base_predictions['linear_lasso'] = {
            'train': linear_lasso_train_pred,
            'val': linear_lasso_val_pred,
            'test': linear_lasso_pred,
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
    tsm_pred_train_eval = None
    tsm_pred_val_eval = None
    
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
            batch_size = int(config_override.get("model", {}).get("batch_size", 32))
            compute_cfg = config_override.get("compute", {}) or {}
            num_workers = int(compute_cfg.get("num_workers", 0))
            pin_memory = bool(compute_cfg.get("pin_memory", False))

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

            train_loader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers,
                pin_memory=pin_memory,
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                num_workers=num_workers,
                pin_memory=pin_memory,
            )
            test_loader = DataLoader(
                test_dataset,
                batch_size=batch_size,
                num_workers=num_workers,
                pin_memory=pin_memory,
            )

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

        eval_batch_size = int(tsm_config.get("model", {}).get("batch_size", 32))
        eval_workers = int(tsm_config.get("compute", {}).get("num_workers", 0))
        eval_pin_memory = bool(tsm_config.get("compute", {}).get("pin_memory", False))
        train_loader_eval = DataLoader(
            tsm_datasets["train"],
            batch_size=eval_batch_size,
            num_workers=eval_workers,
            pin_memory=eval_pin_memory,
        )
        val_loader_eval = DataLoader(
            tsm_datasets["val"],
            batch_size=eval_batch_size,
            num_workers=eval_workers,
            pin_memory=eval_pin_memory,
        )

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
        llm_base_predictions['tsm'] = {
            'train': tsm_pred_train_eval,
            'val': tsm_pred_val_eval,
            'test': tsm_pred_eval,
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

        pred_val_df = pd.DataFrame({
            'date': splits['val']['dates']
        })
        for h in range(config.pred_len):
            pred_val_df[f'y_true_t_plus_{h+1}'] = y_val_future[:, h]
            pred_val_df[f'yhat_t_plus_{h+1}'] = tsm_pred_val_eval[:, h]
        pred_val_df.to_parquet(run_dir / "predictions" / "tsm_pred_val.parquet")

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
            from llm import (
                LLMRefiner,
                blend_mode_from_config,
                method_supports_internal_blend,
                method_uses_internal_blend,
            )

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
            llm_cot_cfg = config.llm.get("cot_rf", {})
            llm_exogenous_feature_priority = parse_name_list(
                llm_cot_cfg.get("exogenous_feature_priority")
            )
            if not llm_exogenous_feature_priority:
                llm_exogenous_feature_priority = parse_name_list(
                    config.features.get("preferred_feature_order", [])
                )
            llm_retrieval_feature_columns = parse_name_list(
                llm_cot_cfg.get("retrieval_feature_columns")
            )
            exogenous_summaries = [
                build_exogenous_summary(
                    splits["test"]["X_enc"][i, -history_points:, :],
                    feature_cols,
                    target_col,
                    max_features=max_exo,
                    preferred_features=llm_exogenous_feature_priority,
                    include_features=llm_retrieval_feature_columns,
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

            llm_refiner_config = build_llm_refiner_config(config)

            refiner = LLMRefiner(
                llm_refiner_config,
                cache_dir=run_dir / "llm" / "cache",
                log_dir=run_dir / "llm" / "logs"
            )

            llm_base_model_name = normalize_llm_base_model_name(
                config.llm.get("base_model", "tsm")
            )
            available_llm_base_models = sorted(
                name
                for name, preds in llm_base_predictions.items()
                if preds.get("train") is not None
                and preds.get("val") is not None
                and preds.get("test") is not None
            )
            llm_base_bundle = llm_base_predictions.get(llm_base_model_name)
            if llm_base_bundle is None or any(
                llm_base_bundle.get(split) is None for split in ("train", "val", "test")
            ):
                logger.warning(
                    "Configured llm.base_model=%s is unavailable. Available models: %s",
                    llm_base_model_name,
                    ", ".join(available_llm_base_models) if available_llm_base_models else "<none>",
                )
                llm_base_train_eval = None
                llm_base_val_eval = None
                llm_base_test_eval = None
            else:
                llm_base_train_eval = llm_base_bundle["train"]
                llm_base_val_eval = llm_base_bundle["val"]
                llm_base_test_eval = llm_base_bundle["test"]
                logger.info("LLM base forecast model: %s", llm_base_model_name)

            calibrate_cfg = config.llm.get("calibrate_blend", {})
            calibrate_enabled = bool(calibrate_cfg.get("enabled", False))
            calibrate_methods = calibrate_cfg.get("methods")
            if calibrate_methods is None:
                calibrate_methods = config.llm.get("methods", ["TSM+LLM"])
            min_weight = float(calibrate_cfg.get("min_weight", 0.0))
            max_weight = float(calibrate_cfg.get("max_weight", 1.0))

            delta_calib_cfg = config.llm.get("delta_calibration", {}) or {}
            delta_calib_enabled = bool(delta_calib_cfg.get("enabled", False))
            delta_calib_methods = delta_calib_cfg.get("methods")
            if delta_calib_methods is None:
                delta_calib_methods = config.llm.get("methods", ["TSM+LLM"])
            delta_calib_target_horizons = [
                int(x)
                for x in _parse_float_list(
                    delta_calib_cfg.get("target_horizons"),
                    default=[20, 30],
                )
            ]
            delta_calib_min_scale = float(delta_calib_cfg.get("min_scale", 0.0))
            delta_calib_max_scale = float(delta_calib_cfg.get("max_scale", 1.0))
            delta_calib_shared = bool(delta_calib_cfg.get("shared", False))
            delta_calib_tune_scope = str(delta_calib_cfg.get("tune_scope", "full")).lower()
            delta_calib_recent_tail_fraction = float(
                delta_calib_cfg.get("recent_tail_fraction", 1.0)
            )
            delta_calib_recent_tail_min_samples = int(
                delta_calib_cfg.get("recent_tail_min_samples", 0)
            )

            subset_cfg = config.llm.get("subset", {}) or {}
            subset_strategy = str(subset_cfg.get("strategy", "first")).lower()
            subset_seed = int(subset_cfg.get("seed", config.seed))
            subset_scope = str(subset_cfg.get("scope", "full")).lower()
            subset_recent_tail_fraction = float(
                subset_cfg.get("recent_tail_fraction", 1.0)
            )
            subset_recent_tail_min_samples = int(
                subset_cfg.get("recent_tail_min_samples", 0)
            )
            val_subset_strategy = str(subset_cfg.get("val_strategy", subset_strategy)).lower()
            val_subset_seed = int(subset_cfg.get("val_seed", subset_seed))
            export_val_predictions = bool(config.llm.get("export_val_predictions", False))

            blend_grid_cfg = config.llm.get("blend_grid", {}) or {}
            blend_grid_enabled = bool(blend_grid_cfg.get("enabled", False))
            blend_grid_methods = blend_grid_cfg.get("methods")
            if blend_grid_methods is None:
                blend_grid_methods = config.llm.get("methods", ["TSM+LLM"])
            llm_blend_mode = blend_mode_from_config(config.llm)
            llm_blend_cfg = config.llm.get("blend", {}) or {}
            llm_blend_configured = bool(llm_blend_cfg)
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
            blend_grid_tune_scope = str(blend_grid_cfg.get("tune_scope", "full")).lower()
            blend_grid_recent_tail_fraction = float(
                blend_grid_cfg.get("recent_tail_fraction", 1.0)
            )
            blend_grid_recent_tail_min_samples = int(
                blend_grid_cfg.get("recent_tail_min_samples", 0)
            )

            rule_gate_cfg = config.llm.get("rule_gate", {}) or {}
            rule_gate_enabled = bool(rule_gate_cfg.get("enabled", False))
            rule_gate_methods = rule_gate_cfg.get("methods")
            if rule_gate_methods is None:
                rule_gate_methods = config.llm.get("methods", ["TSM+LLM"])
            rule_gate_tune_split = str(rule_gate_cfg.get("tune_split", "val")).lower()
            rule_gate_metric = str(rule_gate_cfg.get("metric", "mse_path")).lower()
            rule_gate_key_horizons = [
                int(x)
                for x in _parse_float_list(
                    rule_gate_cfg.get("key_horizons"),
                    default=[1, 5, 20, 30],
                )
            ]
            rule_gate_low_match_threshold = int(rule_gate_cfg.get("low_match_threshold", 2))
            rule_gate_abs_h30_thresholds = _parse_float_list(
                rule_gate_cfg.get("abs_h30_thresholds_pct"),
                default=[0.25, 0.40],
            )
            rule_gate_candidates = rule_gate_cfg.get("candidates")

            val_histories = None
            val_dates = None
            val_exogenous_summaries = None
            val_sentiment_histories = None
            val_eval_indices = None
            val_eval_indices_grid = None
            val_eval_indices_rule_gate = None
            val_eval_indices_delta = None
            if (
                calibrate_enabled
                or blend_grid_enabled
                or rule_gate_enabled
                or delta_calib_enabled
                or export_val_predictions
            ):
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
                        max_features=max_exo,
                        preferred_features=llm_exogenous_feature_priority,
                        include_features=llm_retrieval_feature_columns,
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
                    blend_grid_candidates = eligible_val_idx
                    if blend_grid_tune_scope in {"recent", "recent_tail", "tail"}:
                        blend_grid_candidates = restrict_to_recent_tail(
                            eligible_val_idx,
                            tail_fraction=blend_grid_recent_tail_fraction,
                            min_samples=blend_grid_recent_tail_min_samples,
                        )
                    val_eval_indices_grid = select_eval_indices(
                        blend_grid_candidates,
                        grid_max,
                        strategy=val_subset_strategy,
                        seed=val_subset_seed,
                    )

                if rule_gate_enabled and rule_gate_tune_split == "val":
                    gate_max = int(
                        rule_gate_cfg.get("max_samples", config.llm.get("max_samples", 50))
                    )
                    val_eval_indices_rule_gate = select_eval_indices(
                        eligible_val_idx,
                        gate_max,
                        strategy=val_subset_strategy,
                        seed=val_subset_seed,
                    )

                if delta_calib_enabled:
                    delta_max = int(
                        delta_calib_cfg.get("max_samples", config.llm.get("max_samples", 50))
                    )
                    delta_candidates = eligible_val_idx
                    if delta_calib_tune_scope in {"recent", "recent_tail", "tail"}:
                        delta_candidates = restrict_to_recent_tail(
                            eligible_val_idx,
                            tail_fraction=delta_calib_recent_tail_fraction,
                            min_samples=delta_calib_recent_tail_min_samples,
                        )
                    val_eval_indices_delta = select_eval_indices(
                        delta_candidates,
                        delta_max,
                        strategy=val_subset_strategy,
                        seed=val_subset_seed,
                    )
            
            # Run each method (limit samples for cost control)
            llm_candidates = eligible_idx
            if subset_scope in {"recent", "recent_tail", "tail"}:
                llm_candidates = restrict_to_recent_tail(
                    eligible_idx,
                    tail_fraction=subset_recent_tail_fraction,
                    min_samples=subset_recent_tail_min_samples,
                )
            max_samples = min(config.llm.get("max_samples", 50), len(llm_candidates))
            llm_eval_indices = select_eval_indices(
                llm_candidates,
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
                result_name = llm_result_name(method, llm_base_model_name)
                result_slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", result_name)
                checkpoint_dir = run_dir / "llm" / "checkpoints" / result_slug
                resume_from_run_dir = config.llm.get("resume_from_run_dir")
                resume_checkpoint_dir = None
                if resume_from_run_dir:
                    resume_checkpoint_dir = (
                        Path(str(resume_from_run_dir)).expanduser().resolve()
                        / "llm"
                        / "checkpoints"
                        / result_slug
                    )
                logger.info("Running method: %s", result_name)
                val_teaching_examples_subset = None
                val_teaching_examples_rule_gate = None
                val_teaching_examples_export = None
                teaching_examples_subset = None
                method_supports_blend = method_supports_internal_blend(method)
                method_uses_blend = method_uses_internal_blend(method, config.llm)
                method_requires_base = llm_method_requires_base_forecast(method)

                if method_requires_base and llm_base_test_eval is None:
                    logger.warning(
                        "Skipping %s because llm.base_model=%s has no test forecasts.",
                        result_name,
                        llm_base_model_name,
                    )
                    continue

                if llm_blend_configured and not method_supports_blend and llm_blend_mode != "none":
                    supported_blend_methods = "TSM+LLM-COT-SENT-RF-DELTA, TSM+LLM-NORM-DELTA"
                    logger.info(
                        "Method %s ignores llm.blend mode=%s; the blend config only applies to %s.",
                        result_name,
                        llm_blend_mode,
                        supported_blend_methods,
                    )
                elif method_uses_blend:
                    logger.info(
                        "Method %s will use internal llm.blend mode=%s.",
                        result_name,
                        llm_blend_mode,
                    )

                if method in ("TSM+LLM-COT-RF", "TSM+LLM-COT-RF-HDELTA", "TSM+LLM-COT-SENT-RF", "TSM+LLM-COT-SENT-RF-DELTA", "TSM+LLM-COT-SENT-RF-HDELTA", "TSM+LLM-COT-SENT-RF-HPRICE"):
                    if llm_base_train_eval is None or llm_base_val_eval is None:
                        logger.warning(
                            "Skipping %s because llm.base_model=%s has no train/val teaching pool forecasts.",
                            result_name,
                            llm_base_model_name,
                        )
                        continue
                    cot_cfg = config.llm.get("cot_rf", {})
                    k_examples = int(cot_cfg.get("k_examples", 5))
                    selection_mode = cot_cfg.get("example_selection", "recent")
                    feature_window = int(cot_cfg.get("feature_window", 20))
                    lookback_days = cot_cfg.get("lookback_days")
                    lookback_days = int(lookback_days) if lookback_days is not None else None
                    retrieval_feature_columns = parse_name_list(cot_cfg.get("retrieval_feature_columns"))
                    retrieval_max_features = int(
                        cot_cfg.get("retrieval_max_features", max(len(retrieval_feature_columns), max_exo))
                    )
                    exogenous_feature_priority = parse_name_list(cot_cfg.get("exogenous_feature_priority"))
                    if not exogenous_feature_priority:
                        exogenous_feature_priority = parse_name_list(
                            config.features.get("preferred_feature_order", [])
                        )
                    include_example_exogenous_summary = bool(
                        cot_cfg.get("include_example_exogenous_summary", False)
                    )
                    augment_case_profiles_with_retrieval = bool(
                        cot_cfg.get("augment_case_profiles_with_retrieval", False)
                    )
                    example_exogenous_max_features = int(
                        cot_cfg.get("example_exogenous_max_features", max_exo)
                    )
                    include_hindsight_feedback = bool(
                        cot_cfg.get("include_hindsight_feedback", False)
                    )
                    include_counterexample_freeze = bool(
                        cot_cfg.get("include_counterexample_freeze", False)
                    )
                    counterexample_mode = str(
                        cot_cfg.get("counterexample_mode", "low_error")
                    ).strip().lower()
                    counterexample_quantile = float(
                        cot_cfg.get("counterexample_quantile", 0.35)
                    )
                    prototype_pool_multiplier = int(
                        cot_cfg.get("prototype_pool_multiplier", 3)
                    )
                    skill_tag_min_overlap = int(
                        cot_cfg.get("skill_tag_min_overlap", 3)
                    )
                    include_aux_teacher_summary = bool(
                        cot_cfg.get("include_aux_teacher_summary", False)
                    )
                    include_current_aux_teacher_summary = bool(
                        cot_cfg.get(
                            "include_current_aux_teacher_summary",
                            include_aux_teacher_summary,
                        )
                    )
                    aux_teacher_model_name = normalize_llm_base_model_name(
                        cot_cfg.get("aux_teacher_model", "linear_ridge")
                    )
                    aux_teacher_horizons = [
                        int(x)
                        for x in _parse_float_list(
                            cot_cfg.get("aux_teacher_horizons"),
                            default=[5, 20, 30],
                        )
                    ]
                    aux_teacher_bundle = (
                        llm_base_predictions.get(aux_teacher_model_name, {})
                        if aux_teacher_model_name
                        else {}
                    )
                    aux_teacher_train_eval = aux_teacher_bundle.get("train")
                    aux_teacher_val_eval = aux_teacher_bundle.get("val")
                    aux_teacher_test_eval = aux_teacher_bundle.get("test")
                    aux_teacher_available = all(
                        arr is not None
                        for arr in (
                            aux_teacher_train_eval,
                            aux_teacher_val_eval,
                            aux_teacher_test_eval,
                        )
                    )

                    def _history_features(hist: np.ndarray) -> np.ndarray:
                        window = hist[-feature_window:] if len(hist) >= feature_window else hist
                        mean = float(np.mean(window)) if len(window) else 0.0
                        std = float(np.std(window)) if len(window) else 0.0
                        if len(window) > 1:
                            slope = float(np.polyfit(np.arange(len(window)), window, 1)[0])
                        else:
                            slope = 0.0
                        return np.array([mean, std, slope], dtype=float)

                    def _prepare_example_pool(
                        pool_dates: np.ndarray,
                        pool_histories: np.ndarray,
                        pool_forecasts: np.ndarray,
                        pool_truth: np.ndarray,
                        pool_windows: np.ndarray | None = None,
                        pool_aux_forecasts: Optional[dict[str, np.ndarray]] = None,
                    ) -> dict:
                        dates_arr = pd.to_datetime(pool_dates).to_numpy()
                        order = np.argsort(dates_arr)
                        prepared = {
                            "dates": dates_arr[order],
                            "histories": pool_histories[order],
                            "forecasts": pool_forecasts[order],
                            "truth": pool_truth[order],
                            "windows": None if pool_windows is None else pool_windows[order],
                            "aux_forecasts": {},
                        }
                        for name, values in (pool_aux_forecasts or {}).items():
                            if values is not None:
                                prepared["aux_forecasts"][str(name)] = values[order]
                        prepared["path_mse"] = np.mean(
                            (prepared["forecasts"] - prepared["truth"]) ** 2,
                            axis=1,
                        )
                        prepared["error_profiles"] = np.vstack(
                            [
                                _error_profile(fc, truth)
                                for fc, truth in zip(prepared["forecasts"], prepared["truth"], strict=False)
                            ]
                        )

                        def _anchor_sqerr(arr_idx: int, preds: np.ndarray) -> np.ndarray:
                            if arr_idx >= preds.shape[1] or arr_idx >= prepared["truth"].shape[1]:
                                return np.zeros(len(prepared["dates"]), dtype=float)
                            return (preds[:, arr_idx] - prepared["truth"][:, arr_idx]) ** 2

                        def _anchor_err_pct(arr_idx: int, preds: np.ndarray) -> np.ndarray:
                            if arr_idx >= preds.shape[1] or arr_idx >= prepared["truth"].shape[1]:
                                return np.zeros(len(prepared["dates"]), dtype=float)
                            denom = np.where(
                                np.abs(preds[:, arr_idx]) > 1e-8,
                                preds[:, arr_idx],
                                prepared["truth"][:, arr_idx],
                            )
                            denom = np.where(np.abs(denom) > 1e-8, denom, 1.0)
                            return ((preds[:, arr_idx] - prepared["truth"][:, arr_idx]) / denom) * 100.0

                        prepared["h5_mse"] = _anchor_sqerr(4, prepared["forecasts"])
                        prepared["h20_mse"] = _anchor_sqerr(19, prepared["forecasts"])
                        prepared["h30_mse"] = _anchor_sqerr(29, prepared["forecasts"])
                        prepared["err_pct_h20"] = _anchor_err_pct(19, prepared["forecasts"])
                        prepared["err_pct_h30"] = _anchor_err_pct(29, prepared["forecasts"])
                        prepared["long_error_score"] = 0.5 * (
                            prepared["h20_mse"] + prepared["h30_mse"]
                        )
                        prepared["long_error_consistent"] = (
                            np.sign(prepared["err_pct_h20"]) == np.sign(prepared["err_pct_h30"])
                        ) & (
                            np.abs(prepared["err_pct_h20"]) >= 0.10
                        ) & (
                            np.abs(prepared["err_pct_h30"]) >= 0.10
                        )
                        if selection_mode == "similarity":
                            prepared["history_features"] = np.vstack(
                                [_history_features(hist) for hist in prepared["histories"]]
                            )
                        if selection_mode == "event_similarity":
                            history_features = np.vstack(
                                [_history_features(hist) for hist in prepared["histories"]]
                            )
                            prepared["history_features"] = history_features
                            retrieval_names = resolve_exogenous_feature_names(
                                feature_cols,
                                target_col,
                                preferred_features=exogenous_feature_priority,
                                include_features=retrieval_feature_columns,
                                max_features=retrieval_max_features,
                            )
                            prepared["retrieval_feature_names"] = np.asarray(retrieval_names, dtype=object)
                            retrieval_width = len(retrieval_names) * 2
                            prepared["retrieval_feature_width"] = retrieval_width
                            if pool_windows is not None and retrieval_width > 0:
                                retrieval_features = np.vstack(
                                    [
                                        build_retrieval_feature_vector(
                                            window,
                                            feature_cols,
                                            target_col,
                                            feature_window=feature_window,
                                            preferred_features=exogenous_feature_priority,
                                            include_features=retrieval_names,
                                            max_features=len(retrieval_names),
                                        )
                                        for window in prepared["windows"]
                                    ]
                                )
                            else:
                                retrieval_features = np.zeros((len(prepared["dates"]), retrieval_width), dtype=float)
                            combined = np.hstack([history_features, retrieval_features])
                            means = combined.mean(axis=0)
                            stds = combined.std(axis=0)
                            stds = np.where(stds > 1e-8, stds, 1.0)
                            prepared["event_similarity_features"] = (combined - means) / stds
                            prepared["event_similarity_means"] = means
                            prepared["event_similarity_stds"] = stds
                        case_profiles = np.vstack(
                            [
                                _case_profile_features(hist, fc, feature_window=feature_window)
                                for hist, fc in zip(prepared["histories"], prepared["forecasts"], strict=False)
                            ]
                        )
                        if augment_case_profiles_with_retrieval and pool_windows is not None:
                            retrieval_vectors = np.vstack(
                                [
                                    build_retrieval_feature_vector(
                                        window,
                                        feature_cols,
                                        target_col,
                                        feature_window=feature_window,
                                        preferred_features=exogenous_feature_priority,
                                        include_features=retrieval_feature_columns,
                                        max_features=retrieval_max_features,
                                    )
                                    for window in prepared["windows"]
                                ]
                            )
                            if retrieval_vectors.shape[1] > 0:
                                case_profiles = np.hstack([case_profiles, retrieval_vectors])
                        prepared["case_profiles"] = case_profiles
                        prepared["retrieval_tags"] = np.array(
                            [
                                _profile_regime_tag(profile)
                                for profile in prepared["case_profiles"]
                            ],
                            dtype=object,
                        )
                        if aux_teacher_available:
                            teacher_forecasts = prepared["aux_forecasts"].get(aux_teacher_model_name)
                            if teacher_forecasts is not None:
                                prepared["aux_teacher_gain_long"] = prepared["long_error_score"] - 0.5 * (
                                    _anchor_sqerr(19, teacher_forecasts) + _anchor_sqerr(29, teacher_forecasts)
                                )
                            else:
                                prepared["aux_teacher_gain_long"] = np.zeros(
                                    len(prepared["dates"]), dtype=float
                                )
                        else:
                            prepared["aux_teacher_gain_long"] = np.zeros(
                                len(prepared["dates"]), dtype=float
                            )
                        return prepared

                    def _reference_case_profile(
                        reference_history: np.ndarray,
                        reference_forecast: np.ndarray | None = None,
                        reference_window: np.ndarray | None = None,
                    ) -> np.ndarray:
                        reference_profile = _case_profile_features(
                            reference_history,
                            reference_forecast,
                            feature_window=feature_window,
                        )
                        if augment_case_profiles_with_retrieval and reference_window is not None:
                            retrieval_vector = build_retrieval_feature_vector(
                                reference_window,
                                feature_cols,
                                target_col,
                                feature_window=feature_window,
                                preferred_features=exogenous_feature_priority,
                                include_features=retrieval_feature_columns,
                                max_features=retrieval_max_features,
                            )
                            if retrieval_vector.size:
                                reference_profile = np.concatenate([reference_profile, retrieval_vector])
                        return reference_profile

                    def _select_example_indices(
                        pool: dict,
                        reference_date,
                        reference_history: np.ndarray,
                        reference_forecast: np.ndarray | None = None,
                        reference_window: np.ndarray | None = None,
                    ) -> dict[str, object]:
                        reference_date = pd.Timestamp(reference_date).to_datetime64()
                        pos = np.searchsorted(pool["dates"], reference_date, side="left")
                        if pos <= 0:
                            return {"indices": np.array([], dtype=int), "roles": {}}

                        candidate_indices = np.arange(pos, dtype=int)
                        if selection_mode in (
                            "recent_high_error",
                            "recent_long_error",
                            "recent_consistent_long_error",
                            "balanced_long_horizon",
                            "prototype_recent_long_error",
                            "skill_tag_recent_high_error",
                            "ridge_gain",
                            "similarity",
                            "error_stratified",
                            "similarity_error_hybrid",
                            "utility_score",
                            "utility_mmr",
                            "regime_mmr",
                        ) and lookback_days is not None:
                            cutoff = np.datetime64(
                                pd.Timestamp(reference_date) - pd.Timedelta(days=lookback_days)
                            )
                            candidate_indices = candidate_indices[pool["dates"][candidate_indices] >= cutoff]
                            if candidate_indices.size == 0:
                                candidate_indices = np.arange(pos, dtype=int)

                        primary_k = int(k_examples)
                        if include_counterexample_freeze and candidate_indices.size > 1:
                            primary_k = max(1, primary_k - 1)
                        roles: dict[int, str] = {}

                        if selection_mode in ("high_error", "recent_high_error"):
                            example_indices = select_top_score_indices(
                                candidate_indices,
                                pool["dates"],
                                pool["path_mse"],
                                primary_k,
                            )
                        elif selection_mode == "recent_long_error":
                            example_indices = select_top_score_indices(
                                candidate_indices,
                                pool["dates"],
                                pool["long_error_score"],
                                primary_k,
                            )
                        elif selection_mode == "recent_consistent_long_error":
                            consistent = candidate_indices[pool["long_error_consistent"][candidate_indices]]
                            working = consistent if consistent.size >= max(2, primary_k) else candidate_indices
                            example_indices = select_top_score_indices(
                                working,
                                pool["dates"],
                                pool["long_error_score"],
                                primary_k,
                            )
                        elif selection_mode == "balanced_long_horizon":
                            example_indices = select_balanced_long_horizon_indices(
                                candidate_indices,
                                pool["dates"],
                                pool["h20_mse"],
                                pool["h30_mse"],
                                primary_k,
                            )
                        elif selection_mode == "prototype_recent_long_error":
                            example_indices = select_prototype_indices(
                                candidate_indices,
                                pool["dates"],
                                pool["case_profiles"],
                                pool["long_error_score"],
                                primary_k,
                                top_pool_size=max(primary_k * prototype_pool_multiplier, primary_k),
                            )
                        elif selection_mode == "skill_tag_recent_high_error":
                            reference_tag = _profile_regime_tag(
                                _reference_case_profile(
                                    reference_history,
                                    reference_forecast,
                                    reference_window,
                                )
                            )
                            example_indices = select_skill_tag_recent_high_error_indices(
                                candidate_indices,
                                pool["dates"],
                                pool["retrieval_tags"],
                                pool["path_mse"],
                                reference_tag,
                                primary_k,
                                min_overlap=skill_tag_min_overlap,
                            )
                        elif selection_mode == "ridge_gain":
                            gain_scores = pool.get("aux_teacher_gain_long")
                            if gain_scores is None or not np.any(np.abs(gain_scores[candidate_indices]) > 1e-12):
                                example_indices = select_top_score_indices(
                                    candidate_indices,
                                    pool["dates"],
                                    pool["path_mse"],
                                    primary_k,
                                )
                            else:
                                example_indices = select_top_score_indices(
                                    candidate_indices,
                                    pool["dates"],
                                    gain_scores,
                                    primary_k,
                                )
                        elif selection_mode == "similarity":
                            sample_feat = _history_features(reference_history)
                            cand_feats = pool["history_features"][candidate_indices]
                            distances = np.linalg.norm(cand_feats - sample_feat, axis=1)
                            order = np.argsort(distances)[:primary_k]
                            example_indices = candidate_indices[order]
                            example_indices = example_indices[np.argsort(pool["dates"][example_indices])]
                        elif selection_mode == "event_similarity":
                            retrieval_names = pool.get("retrieval_feature_names", np.array([], dtype=object))
                            sample_retrieval = build_retrieval_feature_vector(
                                reference_window if reference_window is not None else np.zeros((0, len(feature_cols))),
                                feature_cols,
                                target_col,
                                feature_window=feature_window,
                                preferred_features=exogenous_feature_priority,
                                include_features=retrieval_names.tolist(),
                                max_features=len(retrieval_names),
                            )
                            sample_feat = np.concatenate(
                                [
                                    _history_features(reference_history),
                                    sample_retrieval,
                                ]
                            )
                            sample_feat = (
                                sample_feat - pool["event_similarity_means"]
                            ) / pool["event_similarity_stds"]
                            cand_feats = pool["event_similarity_features"][candidate_indices]
                            distances = np.linalg.norm(cand_feats - sample_feat, axis=1)
                            order = np.argsort(distances)[:primary_k]
                            example_indices = candidate_indices[order]
                            example_indices = example_indices[np.argsort(pool["dates"][example_indices])]
                        elif selection_mode == "error_stratified":
                            example_indices = select_error_stratified_indices(
                                candidate_indices=candidate_indices,
                                pool_dates=pool["dates"],
                                pool_error_profiles=pool["error_profiles"],
                                k_examples=primary_k,
                            )
                        elif selection_mode == "similarity_error_hybrid":
                            reference_profile = _reference_case_profile(
                                reference_history,
                                reference_forecast,
                                reference_window,
                            )
                            example_indices = select_similarity_error_hybrid_indices(
                                candidate_indices=candidate_indices,
                                pool_dates=pool["dates"],
                                pool_case_profiles=pool["case_profiles"],
                                pool_error_profiles=pool["error_profiles"],
                                reference_profile=reference_profile,
                                k_examples=primary_k,
                                n_similarity=int(cot_cfg.get("n_similarity_examples", min(3, primary_k))),
                            )
                        elif selection_mode in ("utility_score", "utility_mmr", "regime_mmr"):
                            reference_profile = _reference_case_profile(
                                reference_history,
                                reference_forecast,
                                reference_window,
                            )
                            regime_scores = None
                            working_indices = candidate_indices
                            if selection_mode == "regime_mmr":
                                all_regime_scores = np.asarray(
                                    [
                                        _regime_match_score(
                                            pool["case_profiles"][idx],
                                            reference_profile,
                                        )
                                        for idx in candidate_indices
                                    ],
                                    dtype=float,
                                )
                                min_candidates = int(
                                    cot_cfg.get(
                                        "regime_filter_min_candidates",
                                        max(primary_k, min(len(candidate_indices), primary_k * 2)),
                                    )
                                )
                                if all_regime_scores.size:
                                    best_regime = float(np.max(all_regime_scores))
                                    keep_mask = all_regime_scores >= (best_regime - 0.5)
                                    filtered = candidate_indices[keep_mask]
                                    filtered_scores = all_regime_scores[keep_mask]
                                    if filtered.size < min_candidates:
                                        order = np.argsort(all_regime_scores)[-min_candidates:]
                                        filtered = candidate_indices[order]
                                        filtered_scores = all_regime_scores[order]
                                    working_indices = filtered
                                    regime_scores = filtered_scores
                            if selection_mode == "utility_score":
                                example_indices = select_utility_score_indices(
                                    candidate_indices=working_indices,
                                    pool_dates=pool["dates"],
                                    pool_case_profiles=pool["case_profiles"],
                                    pool_path_mse=pool["path_mse"],
                                    reference_profile=reference_profile,
                                    k_examples=primary_k,
                                    similarity_weight=float(cot_cfg.get("utility_similarity_weight", 0.45)),
                                    error_weight=float(cot_cfg.get("utility_error_weight", 0.35)),
                                    recency_weight=float(cot_cfg.get("utility_recency_weight", 0.20)),
                                    regime_scores=regime_scores,
                                    regime_weight=float(cot_cfg.get("utility_regime_weight", 0.0)),
                                )
                            else:
                                example_indices = select_utility_mmr_indices(
                                    candidate_indices=working_indices,
                                    pool_dates=pool["dates"],
                                    pool_case_profiles=pool["case_profiles"],
                                    pool_path_mse=pool["path_mse"],
                                    reference_profile=reference_profile,
                                    k_examples=primary_k,
                                    similarity_weight=float(cot_cfg.get("utility_similarity_weight", 0.45)),
                                    error_weight=float(cot_cfg.get("utility_error_weight", 0.35)),
                                    recency_weight=float(cot_cfg.get("utility_recency_weight", 0.20)),
                                    mmr_lambda=float(cot_cfg.get("mmr_lambda", 0.75)),
                                    regime_scores=regime_scores,
                                    regime_weight=float(cot_cfg.get("utility_regime_weight", 0.15)),
                                )
                        else:
                            start = max(0, pos - primary_k)
                            example_indices = np.arange(start, pos, dtype=int)

                        for idx in np.asarray(example_indices, dtype=int):
                            roles[int(idx)] = "support"

                        if include_counterexample_freeze and candidate_indices.size > len(example_indices):
                            reference_profile = _reference_case_profile(
                                reference_history,
                                reference_forecast,
                                reference_window,
                            )
                            counter_metric = (
                                pool.get("aux_teacher_gain_long", pool["path_mse"])
                                if counterexample_mode == "low_gain"
                                else pool["path_mse"]
                            )
                            counter_idx = select_counterexample_index(
                                candidate_indices=candidate_indices,
                                pool_case_profiles=pool["case_profiles"],
                                reference_profile=reference_profile,
                                pool_metric=counter_metric,
                                exclude_indices=example_indices,
                                quantile=counterexample_quantile,
                            )
                            if counter_idx is not None:
                                example_indices = np.asarray(
                                    list(np.asarray(example_indices, dtype=int)) + [int(counter_idx)],
                                    dtype=int,
                                )
                                roles[int(counter_idx)] = "freeze_counterexample"

                        return {
                            "indices": np.asarray(example_indices, dtype=int),
                            "roles": roles,
                        }

                    def _build_examples(pool: dict, selection_bundle: object) -> list[dict]:
                        bundle = normalize_selection_bundle(selection_bundle)
                        example_indices = bundle["indices"]
                        example_roles = bundle["roles"]
                        examples = []
                        for idx in example_indices:
                            sent_hist = None
                            if sentiment_map:
                                ex_date = pd.Timestamp(pool["dates"][idx])
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
                            exogenous_summary = None
                            if include_example_exogenous_summary and pool.get("windows") is not None:
                                exogenous_summary = build_exogenous_summary(
                                    pool["windows"][idx],
                                    feature_cols,
                                    target_col,
                                    max_features=example_exogenous_max_features,
                                    preferred_features=exogenous_feature_priority,
                                    include_features=retrieval_feature_columns,
                                )
                            example_case_summary: dict[str, str] = {}
                            role = example_roles.get(int(idx))
                            if role:
                                example_case_summary["selection_role"] = role

                            anchor_error_pct = {}
                            for horizon in (1, 5, 20, 30):
                                arr_idx = int(horizon) - 1
                                if arr_idx >= pool["forecasts"].shape[1] or arr_idx >= pool["truth"].shape[1]:
                                    continue
                                base_val = float(pool["forecasts"][idx][arr_idx])
                                truth_val = float(pool["truth"][idx][arr_idx])
                                denom = base_val if abs(base_val) > 1e-8 else truth_val
                                anchor_error_pct[int(horizon)] = (
                                    0.0
                                    if abs(float(denom)) < 1e-8
                                    else float((base_val - truth_val) / denom * 100.0)
                                )

                            if include_hindsight_feedback:
                                hindsight = summarize_hindsight_feedback(
                                    anchor_error_pct,
                                    mode=str(cot_cfg.get("feedback_mode", "standard")),
                                )
                                if hindsight:
                                    example_case_summary["hindsight_feedback"] = hindsight

                            if include_aux_teacher_summary and aux_teacher_available:
                                teacher_forecasts = pool["aux_forecasts"].get(aux_teacher_model_name)
                                if teacher_forecasts is not None:
                                    example_case_summary.update(
                                        build_aux_teacher_case_summary(
                                            base_forecast=pool["forecasts"][idx],
                                            aux_forecast=teacher_forecasts[idx],
                                            teacher_name=aux_teacher_model_name,
                                            truth=pool["truth"][idx],
                                            horizons=aux_teacher_horizons,
                                        )
                                    )
                            examples.append(
                                {
                                    "history": pool["histories"][idx],
                                    "forecast": pool["forecasts"][idx],
                                    "truth": pool["truth"][idx],
                                    "date": str(pd.Timestamp(pool["dates"][idx])),
                                    "sentiment_history": sent_hist if sent_hist is not None else [],
                                    "exogenous_summary": exogenous_summary,
                                    "case_summary": example_case_summary,
                                    "retrieval_tag": (
                                        str(pool["retrieval_tags"][idx])
                                        if "retrieval_tags" in pool
                                        else None
                                    ),
                                }
                            )
                        return examples

                    # Strict non-leaking pools:
                    # - validation examples may only use training truths
                    # - test examples may only use training + validation truths
                    val_aux_forecasts = (
                        {aux_teacher_model_name: aux_teacher_train_eval}
                        if aux_teacher_available
                        else None
                    )
                    val_pool = _prepare_example_pool(
                        splits["train"]["dates"],
                        y_train_hist,
                        llm_base_train_eval,
                        y_train_future,
                        splits["train"]["X_enc"][:, -history_points:, :],
                        val_aux_forecasts,
                    )
                    test_aux_forecasts = (
                        {
                            aux_teacher_model_name: np.concatenate(
                                [aux_teacher_train_eval, aux_teacher_val_eval]
                            )
                        }
                        if aux_teacher_available
                        else None
                    )
                    test_pool = _prepare_example_pool(
                        np.concatenate([splits["train"]["dates"], splits["val"]["dates"]]),
                        np.concatenate([y_train_hist, y_val_hist]),
                        np.concatenate([llm_base_train_eval, llm_base_val_eval]),
                        np.concatenate([y_train_future, y_val_future]),
                        np.concatenate(
                            [
                                splits["train"]["X_enc"][:, -history_points:, :],
                                splits["val"]["X_enc"][:, -history_points:, :],
                            ]
                        ),
                        test_aux_forecasts,
                    )

                    teaching_examples_subset = []
                    val_teaching_examples_subset = []
                    val_teaching_examples_rule_gate = []
                    val_teaching_examples_delta = []
                    for sample_idx in llm_eval_indices:
                        example_indices = _select_example_indices(
                            test_pool,
                            splits["test"]["dates"][sample_idx],
                            y_test_hist[sample_idx],
                            llm_base_test_eval[sample_idx],
                            splits["test"]["X_enc"][sample_idx, -history_points:, :],
                        )
                        teaching_examples_subset.append(
                            _build_examples(test_pool, example_indices)
                        )

                    val_needed = []
                    if val_eval_indices_grid is not None and len(val_eval_indices_grid) > 0:
                        val_needed.extend(int(x) for x in np.asarray(val_eval_indices_grid, dtype=int))
                    if val_eval_indices_rule_gate is not None and len(val_eval_indices_rule_gate) > 0:
                        val_needed.extend(int(x) for x in np.asarray(val_eval_indices_rule_gate, dtype=int))
                    if val_eval_indices_delta is not None and len(val_eval_indices_delta) > 0:
                        val_needed.extend(int(x) for x in np.asarray(val_eval_indices_delta, dtype=int))
                    if export_val_predictions:
                        val_needed.extend(range(len(y_val_hist)))
                    val_example_map: dict[int, list[dict]] = {}
                    for sample_idx in sorted(set(val_needed)):
                        example_indices = _select_example_indices(
                            val_pool,
                            splits["val"]["dates"][sample_idx],
                            y_val_hist[sample_idx],
                            llm_base_val_eval[sample_idx],
                            splits["val"]["X_enc"][sample_idx, -history_points:, :],
                        )
                        val_example_map[int(sample_idx)] = _build_examples(val_pool, example_indices)

                    if val_eval_indices_grid is not None and len(val_eval_indices_grid) > 0:
                        for sample_idx in val_eval_indices_grid:
                            val_teaching_examples_subset.append(
                                val_example_map[int(sample_idx)]
                            )

                    if val_eval_indices_rule_gate is not None and len(val_eval_indices_rule_gate) > 0:
                        for sample_idx in val_eval_indices_rule_gate:
                            val_teaching_examples_rule_gate.append(
                                val_example_map[int(sample_idx)]
                            )
                    if val_eval_indices_delta is not None and len(val_eval_indices_delta) > 0:
                        for sample_idx in val_eval_indices_delta:
                            val_teaching_examples_delta.append(
                                val_example_map[int(sample_idx)]
                            )
                    if export_val_predictions:
                        val_teaching_examples_export = [
                            val_example_map[int(sample_idx)]
                            for sample_idx in range(len(y_val_hist))
                        ]
                    exogenous_subset = [None for _ in llm_eval_indices]
                
                base_forecast = (
                    llm_base_test_eval[llm_eval_indices]
                    if llm_base_test_eval is not None
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
                    method_tsm_forecast = base_forecast
                    method_exogenous = exogenous_subset
                    method_price_bases = price_bases
                    method_sentiment = sentiment_subset
                    if (
                        include_current_aux_teacher_summary
                        and aux_teacher_available
                        and base_forecast is not None
                    ):
                        method_exogenous = [
                            augment_exogenous_summary_with_teacher(
                                exogenous_summary=exogenous_summaries[i],
                                base_forecast=base_forecast[row_pos],
                                aux_forecast=aux_teacher_test_eval[i],
                                teacher_name=aux_teacher_model_name,
                                horizons=aux_teacher_horizons,
                            )
                            for row_pos, i in enumerate(llm_eval_indices)
                        ]
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
                            if method in ("TSM+LLM-COT-RF", "TSM+LLM-COT-RF-HDELTA", "TSM+LLM-COT-SENT-RF", "TSM+LLM-COT-SENT-RF-DELTA", "TSM+LLM-COT-SENT-RF-HDELTA", "TSM+LLM-COT-SENT-RF-HPRICE")
                            else None
                        ),
                        sentiment_histories=method_sentiment,
                        checkpoint_dir=checkpoint_dir,
                        resume_checkpoint_dir=resume_checkpoint_dir,
                        sample_keys=[str(int(i)) for i in llm_eval_indices],
                    )
                    
                    llm_results[result_name] = {
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
                        'eval_indices': llm_eval_indices,
                        'base_model': llm_base_model_name if method_requires_base else None,
                        'method_template': method,
                        'blend_semantics': {
                            'llm_blend_mode': llm_blend_mode,
                            'llm_blend_configured': llm_blend_configured,
                            'method_supports_internal_blend': method_supports_blend,
                            'method_uses_internal_blend': method_uses_blend,
                        },
                    }

                    # Persist subset predictions so we can debug deltas vs the
                    # configured base forecast and reproduce paper-style plots
                    # without re-running the LLM.
                    try:
                        out_path = run_dir / "predictions" / f"{result_name}_pred_test_subset.npz"
                        np.savez_compressed(
                            out_path,
                            method=np.array([method]),
                            result_name=np.array([result_name]),
                            base_model=np.array([llm_base_model_name]),
                            eval_indices=np.asarray(llm_eval_indices, dtype=int),
                            dates=np.asarray(
                                [str(splits["test"]["dates"][i]) for i in llm_eval_indices],
                                dtype="U",
                            ),
                            y_true=np.asarray(y_test_future[llm_eval_indices], dtype=float),
                            base_pred=np.asarray(base_forecast, dtype=float)
                            if base_forecast is not None
                            else np.empty((0, 0), dtype=float),
                            tsm_pred=np.asarray(base_forecast, dtype=float)
                            if llm_base_model_name == "tsm" and base_forecast is not None
                            else np.empty((0, 0), dtype=float),
                            yhat=np.asarray(predictions, dtype=float),
                        )
                    except Exception as e:
                        logger.warning("Failed to save LLM subset predictions: %s", e)
                    
                    logger.info("\n%s Results:", result_name)
                    logger.info(llm_results[result_name]['metrics'])
                    method_path_mse = (llm_results[result_name].get("path_metrics") or {}).get("mse_path")
                    if method_path_mse is not None:
                        logger.info(
                            "%s Path MSE (avg over %dd): %.6f",
                            result_name,
                            config.pred_len,
                            float(method_path_mse),
                        )

                    val_llm_cache: dict[tuple[int, ...], tuple[np.ndarray, list[dict]]] = {}

                    def _run_val_llm_for_indices(
                        eval_indices: np.ndarray,
                        val_teaching: Optional[list[list[dict]]],
                    ) -> tuple[np.ndarray, list[dict]]:
                        cache_key = tuple(int(x) for x in np.asarray(eval_indices, dtype=int))
                        cached = val_llm_cache.get(cache_key)
                        if cached is not None:
                            return cached
                        val_method_exogenous = (
                            [val_exogenous_summaries[i] for i in eval_indices]
                            if val_exogenous_summaries is not None
                            else None
                        )
                        if (
                            include_current_aux_teacher_summary
                            and aux_teacher_available
                            and val_method_exogenous is not None
                        ):
                            val_method_exogenous = [
                                augment_exogenous_summary_with_teacher(
                                    exogenous_summary=val_exogenous_summaries[i],
                                    base_forecast=llm_base_val_eval[i],
                                    aux_forecast=aux_teacher_val_eval[i],
                                    teacher_name=aux_teacher_model_name,
                                    horizons=aux_teacher_horizons,
                                )
                                for i in eval_indices
                            ]
                        if method in (
                            "TSM+LLM-COT-RF",
                            "TSM+LLM-COT-RF-HDELTA",
                            "TSM+LLM-COT-SENT-RF",
                            "TSM+LLM-COT-SENT-RF-DELTA",
                            "TSM+LLM-COT-SENT-RF-HDELTA",
                            "TSM+LLM-COT-SENT-RF-HPRICE",
                        ):
                            val_method_exogenous = [None for _ in eval_indices]

                        val_method_sentiment = (
                            [val_sentiment_histories[i] for i in eval_indices]
                            if val_sentiment_histories is not None
                            else None
                        )
                        val_pred, val_meta = refiner.refine_batch(
                            method=method,
                            histories=val_histories[eval_indices],
                            date_arrays=[val_dates[i] for i in eval_indices],
                            tsm_forecasts=llm_base_val_eval[eval_indices],
                            pred_len=config.pred_len,
                            exogenous_summaries=val_method_exogenous,
                            price_bases=None,
                            teaching_examples=val_teaching,
                            sentiment_histories=val_method_sentiment,
                        )
                        val_llm_cache[cache_key] = (val_pred, val_meta)
                        return val_pred, val_meta

                    if export_val_predictions and val_histories is not None and val_dates is not None:
                        full_val_indices = np.arange(len(val_histories), dtype=int)
                        full_val_pred, full_val_meta = _run_val_llm_for_indices(
                            full_val_indices,
                            val_teaching_examples_export,
                        )
                        try:
                            out_path = run_dir / "predictions" / f"{result_name}_pred_val_full.npz"
                            np.savez_compressed(
                                out_path,
                                method=np.array([method]),
                                result_name=np.array([result_name]),
                                base_model=np.array([llm_base_model_name]),
                                eval_indices=full_val_indices,
                                dates=np.asarray(
                                    [str(splits["val"]["dates"][i]) for i in full_val_indices],
                                    dtype="U",
                                ),
                                y_true=np.asarray(y_val_future[full_val_indices], dtype=float),
                                base_pred=np.asarray(llm_base_val_eval[full_val_indices], dtype=float)
                                if llm_base_val_eval is not None
                                else np.empty((0, 0), dtype=float),
                                yhat=np.asarray(full_val_pred, dtype=float),
                            )
                            meta_path = run_dir / "llm" / f"{result_slug}_val_metadata.json"
                            with meta_path.open("w", encoding="utf-8") as f:
                                json.dump(
                                    {
                                        "n_samples": int(len(full_val_indices)),
                                        "export_val_predictions": True,
                                        "method": method,
                                        "result_name": result_name,
                                    },
                                    f,
                                    indent=2,
                                )
                        except Exception as e:
                            logger.warning("Failed to save validation LLM predictions: %s", e)

                    if rule_gate_enabled and method in rule_gate_methods:
                        if method == "NEWS-SENTIMENT-ONLY":
                            logger.warning("Rule gate skipped for NEWS-SENTIMENT-ONLY.")
                        elif llm_base_test_eval is None or llm_base_val_eval is None:
                            logger.warning(
                                "Rule gate skipped for %s (missing %s predictions).",
                                result_name,
                                llm_base_model_name,
                            )
                        elif rule_gate_tune_split != "val":
                            logger.warning(
                                "Rule gate skipped (unsupported tune_split=%s).",
                                rule_gate_tune_split,
                            )
                        elif val_eval_indices_rule_gate is None or len(val_eval_indices_rule_gate) == 0:
                            logger.warning("Rule gate skipped (no validation samples).")
                        else:
                            val_llm_pred, val_llm_meta = _run_val_llm_for_indices(
                                val_eval_indices_rule_gate,
                                val_teaching_examples_rule_gate,
                            )
                            val_feature_df = build_rule_gate_feature_frame(
                                llm_base_val_eval[val_eval_indices_rule_gate],
                                val_llm_pred,
                                val_llm_meta,
                                key_horizons=rule_gate_key_horizons,
                            )
                            test_feature_df = build_rule_gate_feature_frame(
                                llm_base_test_eval[llm_eval_indices],
                                predictions,
                                metadata,
                                key_horizons=rule_gate_key_horizons,
                            )
                            val_summary, _ = evaluate_rule_gate_candidates(
                                y_true=y_val_future[val_eval_indices_rule_gate],
                                base_pred=llm_base_val_eval[val_eval_indices_rule_gate],
                                llm_pred=val_llm_pred,
                                feature_df=val_feature_df,
                                candidates=rule_gate_candidates,
                                low_match_threshold=rule_gate_low_match_threshold,
                                abs_h30_thresholds_pct=rule_gate_abs_h30_thresholds,
                                horizons=horizons,
                            )
                            test_summary, test_gated_predictions = evaluate_rule_gate_candidates(
                                y_true=y_test_future[llm_eval_indices],
                                base_pred=llm_base_test_eval[llm_eval_indices],
                                llm_pred=predictions,
                                feature_df=test_feature_df,
                                candidates=rule_gate_candidates,
                                low_match_threshold=rule_gate_low_match_threshold,
                                abs_h30_thresholds_pct=rule_gate_abs_h30_thresholds,
                                horizons=horizons,
                            )

                            out_root = run_dir / "results" / "rule_gate" / result_slug
                            (out_root / "val").mkdir(parents=True, exist_ok=True)
                            (out_root / "test").mkdir(parents=True, exist_ok=True)
                            val_summary.to_csv(out_root / "val" / "rule_gate_summary.csv", index=False)
                            test_summary.to_csv(out_root / "test" / "rule_gate_summary.csv", index=False)
                            val_feature_df.to_csv(out_root / "val" / "rule_gate_features.csv", index=False)
                            test_feature_df.to_csv(out_root / "test" / "rule_gate_features.csv", index=False)

                            metric_col = "mse_path"
                            if rule_gate_metric != "mse_path" and rule_gate_metric in val_summary.columns:
                                metric_col = rule_gate_metric
                            best_candidate = str(
                                val_summary.loc[val_summary[metric_col].idxmin(), "candidate"]
                            )
                            gated_predictions = test_gated_predictions[best_candidate]
                            selection_payload = {
                                "method": result_name,
                                "method_template": method,
                                "base_model": llm_base_model_name,
                                "selected_candidate": best_candidate,
                                "tune_split": rule_gate_tune_split,
                                "metric": metric_col,
                                "key_horizons": rule_gate_key_horizons,
                                "low_match_threshold": rule_gate_low_match_threshold,
                                "abs_h30_thresholds_pct": rule_gate_abs_h30_thresholds,
                                "n_val": int(len(val_eval_indices_rule_gate)),
                                "n_test": int(len(llm_eval_indices)),
                                "val_flagged_count": int(
                                    val_summary.loc[
                                        val_summary["candidate"] == best_candidate,
                                        "flagged_count",
                                    ].iloc[0]
                                ),
                                "test_flagged_count": int(
                                    test_summary.loc[
                                        test_summary["candidate"] == best_candidate,
                                        "flagged_count",
                                    ].iloc[0]
                                ),
                            }
                            with (run_dir / "llm" / f"rule_gate_selection_{result_slug}.json").open("w") as f:
                                json.dump(selection_payload, f, indent=2)

                            gated_name = f"{result_name}_rulegate_bestval"
                            llm_results[gated_name] = {
                                "predictions": gated_predictions,
                                "metrics": compute_metrics_by_horizon(
                                    y_test_future[llm_eval_indices], gated_predictions, horizons
                                ),
                                "path_metrics": compute_path_metrics(
                                    y_test_future[llm_eval_indices], gated_predictions
                                ),
                                "errors": get_per_sample_errors(
                                    y_test_future[llm_eval_indices], gated_predictions, horizons
                                ),
                                "eval_indices": llm_eval_indices,
                                "rule_gate": selection_payload,
                            }
                            logger.info(
                                "Rule gate (val): best %s candidate=%s",
                                metric_col,
                                best_candidate,
                            )
                            logger.info("\n%s Results:", gated_name)
                            logger.info(llm_results[gated_name]["metrics"])
                            gated_path_mse = (
                                llm_results[gated_name].get("path_metrics") or {}
                            ).get("mse_path")
                            if gated_path_mse is not None:
                                logger.info(
                                    "%s Path MSE (avg over %dd): %.6f",
                                    gated_name,
                                    config.pred_len,
                                    float(gated_path_mse),
                                )

                    if delta_calib_enabled and method in delta_calib_methods:
                        if val_eval_indices_delta is None or len(val_eval_indices_delta) == 0:
                            logger.warning("Delta calibration skipped (no validation samples).")
                        elif llm_base_val_eval is None or llm_base_test_eval is None:
                            logger.warning(
                                "Delta calibration skipped for %s (missing %s validation/test forecasts).",
                                result_name,
                                llm_base_model_name,
                            )
                        else:
                            val_llm_pred, _ = _run_val_llm_for_indices(
                                val_eval_indices_delta,
                                val_teaching_examples_delta,
                            )
                            scales = fit_delta_calibration_scales(
                                y_true=y_val_future[val_eval_indices_delta],
                                base_pred=llm_base_val_eval[val_eval_indices_delta],
                                llm_pred=val_llm_pred,
                                horizons=horizons,
                                target_horizons=delta_calib_target_horizons,
                                min_scale=delta_calib_min_scale,
                                max_scale=delta_calib_max_scale,
                                shared=delta_calib_shared,
                            )
                            calibrated = apply_delta_calibration(
                                base_pred=llm_base_test_eval[llm_eval_indices],
                                llm_pred=predictions,
                                scales=scales,
                                pred_len=config.pred_len,
                            )
                            calib_name = f"{result_name}_delta_calibrated"
                            llm_results[calib_name] = {
                                "predictions": calibrated,
                                "metrics": compute_metrics_by_horizon(
                                    y_test_future[llm_eval_indices], calibrated, horizons
                                ),
                                "path_metrics": compute_path_metrics(
                                    y_test_future[llm_eval_indices], calibrated
                                ),
                                "errors": get_per_sample_errors(
                                    y_test_future[llm_eval_indices], calibrated, horizons
                                ),
                                "eval_indices": llm_eval_indices,
                                "delta_calibration": {
                                    "target_horizons": delta_calib_target_horizons,
                                    "min_scale": delta_calib_min_scale,
                                    "max_scale": delta_calib_max_scale,
                                    "shared": delta_calib_shared,
                                    "tune_scope": delta_calib_tune_scope,
                                    "recent_tail_fraction": delta_calib_recent_tail_fraction,
                                    "recent_tail_min_samples": delta_calib_recent_tail_min_samples,
                                    "scales": {str(k): float(v) for k, v in scales.items()},
                                },
                            }
                            logger.info("\n%s Results:", calib_name)
                            logger.info(llm_results[calib_name]["metrics"])
                            calib_path_mse = (
                                llm_results[calib_name].get("path_metrics") or {}
                            ).get("mse_path")
                            if calib_path_mse is not None:
                                logger.info(
                                    "%s Path MSE (avg over %dd): %.6f",
                                    calib_name,
                                    config.pred_len,
                                    float(calib_path_mse),
                                )
                            with (run_dir / "llm" / f"delta_calibration_{result_slug}.json").open("w") as f:
                                json.dump(llm_results[calib_name]["delta_calibration"], f, indent=2)

                    if calibrate_enabled and method in calibrate_methods:
                        if val_eval_indices is None or len(val_eval_indices) == 0:
                            logger.warning("Blend calibration skipped (no validation samples).")
                        elif method == "NEWS-SENTIMENT-ONLY":
                            logger.warning("Blend calibration skipped for NEWS-SENTIMENT-ONLY.")
                        elif method in (
                            "TSM+LLM-COT-RF",
                            "TSM+LLM-COT-RF-HDELTA",
                            "TSM+LLM-COT-SENT-RF",
                            "TSM+LLM-COT-SENT-RF-DELTA",
                            "TSM+LLM-COT-SENT-RF-HDELTA",
                            "TSM+LLM-COT-SENT-RF-HPRICE",
                        ):
                            logger.warning("Blend calibration skipped for %s.", result_name)
                        elif llm_base_val_eval is None or llm_base_test_eval is None:
                            logger.warning(
                                "Blend calibration skipped for %s (missing %s validation/test forecasts).",
                                result_name,
                                llm_base_model_name,
                            )
                        else:
                            val_predictions, _ = refiner.refine_batch(
                                method=method,
                                histories=val_histories[val_eval_indices],
                                date_arrays=[val_dates[i] for i in val_eval_indices],
                                tsm_forecasts=llm_base_val_eval[val_eval_indices],
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
                                llm_base_val_eval[val_eval_indices],
                                val_predictions,
                                horizons,
                                min_weight=min_weight,
                                max_weight=max_weight
                            )
                            blended = apply_blend_weights(
                                llm_base_test_eval[llm_eval_indices],
                                predictions,
                                weights,
                                config.pred_len
                            )
                            blend_name = f"{result_name}_calibrated"
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
                        elif method_uses_blend:
                            logger.warning(
                                "Blend grid skipped for %s because it already uses internal llm.blend mode=%s.",
                                result_name,
                                llm_blend_mode,
                            )
                        elif llm_base_test_eval is None or llm_base_val_eval is None:
                            logger.warning(
                                "Blend grid skipped for %s (missing %s predictions).",
                                result_name,
                                llm_base_model_name,
                            )
                        elif blend_grid_tune_split != "val":
                            logger.warning(
                                "Blend grid skipped (unsupported tune_split=%s).",
                                blend_grid_tune_split,
                            )
                        elif val_eval_indices_grid is None or len(val_eval_indices_grid) == 0:
                            logger.warning("Blend grid skipped (no validation samples).")
                        else:
                            out_root = run_dir / "results" / f"blend_grid_{blend_grid_schedule}" / result_slug
                            val_llm_pred, _ = _run_val_llm_for_indices(
                                val_eval_indices_grid,
                                val_teaching_examples_subset,
                            )

                            # 2) Evaluate blend grid on validation (selection split)
                            val_summary, val_mse_by_h, val_best_by_h = evaluate_blend_grid(
                                y_true=y_val_future[val_eval_indices_grid],
                                base_pred=llm_base_val_eval[val_eval_indices_grid],
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
                                base_pred=llm_base_test_eval[llm_eval_indices],
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
                                "method": result_name,
                                "method_template": method,
                                "base_model": llm_base_model_name,
                                "schedule": blend_grid_schedule,
                                "weights": blend_grid_weights,
                                "tune_split": blend_grid_tune_split,
                                "metric": metric_col,
                                "min_weight": blend_grid_min_weight,
                                "power": blend_grid_power,
                                "tune_scope": blend_grid_tune_scope,
                                "recent_tail_fraction": blend_grid_recent_tail_fraction,
                                "recent_tail_min_samples": blend_grid_recent_tail_min_samples,
                                "best_w_path": best_w_path,
                                "best_w_by_horizon": best_w_by_horizon,
                                "n_val": int(len(val_eval_indices_grid)),
                                "n_test": int(len(llm_eval_indices)),
                            }
                            with (run_dir / "llm" / f"blend_grid_selection_{result_slug}.json").open("w") as f:
                                json.dump(selection_payload, f, indent=2)

                            logger.info(
                                "Blend grid (%s): best %s at w=%.2f",
                                blend_grid_tune_scope,
                                metric_col,
                                best_w_path,
                            )

                            # 5) Register best-on-val blended variants as first-class models
                            base_pred_subset = llm_base_test_eval[llm_eval_indices]
                            y_test_subset = y_test_future[llm_eval_indices]

                            def _register_blend(
                                name_suffix: str,
                                w: float,
                                h1_base_override: bool = False,
                            ) -> None:
                                blended_pred = blend_forecasts(
                                    base_pred=base_pred_subset,
                                    llm_pred=predictions,
                                    strength=float(w),
                                    schedule=blend_grid_schedule,
                                    pred_len=config.pred_len,
                                    min_weight=blend_grid_min_weight,
                                    power=blend_grid_power,
                                )
                                model_name = f"{result_name}_blend_{blend_grid_schedule}_{name_suffix}"
                                blend_meta = {
                                    "schedule": blend_grid_schedule,
                                    "w": float(w),
                                    "min_weight": blend_grid_min_weight,
                                    "power": blend_grid_power,
                                    "selected_on": "val",
                                    "metric": metric_col,
                                    "base_model": llm_base_model_name,
                                    "method_template": method,
                                }
                                if h1_base_override and blended_pred.shape[1] > 0:
                                    blended_pred = blended_pred.copy()
                                    blended_pred[:, 0] = base_pred_subset[:, 0]
                                    model_name = f"{model_name}_h1base"
                                    blend_meta["h1_override_model"] = llm_base_model_name

                                llm_results[model_name] = {
                                    "predictions": blended_pred,
                                    "metrics": compute_metrics_by_horizon(
                                        y_test_subset, blended_pred, horizons
                                    ),
                                    "path_metrics": compute_path_metrics(
                                        y_test_subset, blended_pred
                                    ),
                                    "errors": get_per_sample_errors(
                                        y_test_subset, blended_pred, horizons
                                    ),
                                    "eval_indices": llm_eval_indices,
                                    "blend": blend_meta,
                                }
                                logger.info("Added blended model: %s (w=%.2f)", model_name, float(w))

                            _register_blend("bestval_path", best_w_path)
                            _register_blend("bestval_path", best_w_path, h1_base_override=True)
                            for h, w in best_w_by_horizon.items():
                                _register_blend(f"bestval_h{int(h)}", float(w))

                    if method in (
                        "TSM+LLM-COT-RF",
                        "TSM+LLM-COT-RF-HDELTA",
                        "TSM+LLM-COT-SENT-RF",
                        "TSM+LLM-COT-SENT-RF-DELTA",
                        "TSM+LLM-COT-SENT-RF-HDELTA",
                        "TSM+LLM-COT-SENT-RF-HPRICE",
                    ):
                        rules_path = run_dir / "llm" / f"{result_slug}_rules.jsonl"
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
                    logger.error("LLM method %s failed: %s", result_name, e)
            
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
    average_non_overlap_offsets = bool(return_cfg.get("average_non_overlap_offsets", True))
    transaction_cost_bps = float(return_cfg.get("transaction_cost_bps", 0.0))

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
                signal_threshold=signal_threshold,
                average_non_overlap_offsets=average_non_overlap_offsets,
                transaction_cost_bps=transaction_cost_bps,
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

    if config.output.get("write_project_paper", True):
        paper_root = Path(__file__).parent.parent / config.output.get("paper_dir", "paper")
        paper_root_path = _build_paper(paper_root)
        logger.info(f"Paper saved to: {paper_root_path}")
    else:
        logger.info("Skipping shared project paper write (output.write_project_paper=false)")
    
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
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Path to data directory (default: ./Data)",
    )
    
    args = parser.parse_args()
    
    overrides = {}
    if args.seed:
        overrides["reproducibility"] = {"seed": args.seed}
    
    run_experiment(args.config, overrides, data_dir=args.data_dir)


if __name__ == "__main__":
    main()
