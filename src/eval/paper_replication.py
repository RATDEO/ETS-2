from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from .metrics import compute_metrics_by_horizon, compute_path_metrics
from .trend_classification import compute_paper_trend_accuracy


@dataclass(frozen=True)
class PaperReplicationCandidate:
    name: str
    method: str
    sentiment_path: str
    base_url: str | None = None
    model: str | None = None
    history_points: int = 18
    prompt_history_points: int = 18
    prompt_sentiment_points: int = 18
    sentiment_points: int = 18
    k_examples: int = 5
    example_selection: str = "similarity"
    feature_window: int = 18
    lookback_days: int | None = 365
    retain_context: bool = True
    strict_json_prompt: bool = False
    strict_json_response_format: bool = True
    hdelta_max_adjustment_pct: float | None = None
    hdelta_key_horizons: tuple[int, ...] = (1, 5, 20, 30)
    hdelta_freeze_horizons: tuple[int, ...] = ()
    hdelta_sentiment_secondary: bool = False
    hprice_max_adjustment_pct: float | None = None
    hprice_key_horizons: tuple[int, ...] = (1, 10, 20, 30)
    hprice_freeze_horizons: tuple[int, ...] = ()
    hprice_sentiment_secondary: bool = False
    official_event_path: str | None = None
    official_event_lookback_days: int = 30
    official_event_max_titles: int = 2
    retrieval_official_event_path: str | None = None


def build_outer_folds(
    n_samples: int,
    n_folds: int = 3,
    max_samples_per_fold: int | None = None,
) -> list[np.ndarray]:
    if n_samples <= 0:
        return []
    if n_folds <= 0:
        raise ValueError("n_folds must be positive")

    raw_folds = [np.asarray(chunk, dtype=int) for chunk in np.array_split(np.arange(n_samples), n_folds)]
    folds: list[np.ndarray] = []
    for chunk in raw_folds:
        if len(chunk) == 0:
            continue
        if max_samples_per_fold is not None and len(chunk) > max_samples_per_fold:
            positions = np.linspace(0, len(chunk) - 1, num=max_samples_per_fold, dtype=int)
            chunk = chunk[positions]
        folds.append(chunk)
    return folds


def score_paper_candidate(
    name: str,
    histories: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    horizons: Iterable[int] = (1, 5, 20, 30),
    paper_alpha: float = 0.02,
    paper_history_window: int = 18,
    paper_horizons: Iterable[int] = (10, 20, 30),
) -> dict[str, float | str]:
    row: dict[str, float | str] = {
        "candidate": name,
        "n_samples": int(y_true.shape[0]),
        "mse_path": float(compute_path_metrics(y_true, y_pred)["mse_path"]),
    }
    by_h = compute_metrics_by_horizon(y_true, y_pred, list(horizons)).reset_index()
    for _, metric_row in by_h.iterrows():
        row[f"h{int(metric_row['horizon'])}_mse"] = float(metric_row["mse"])

    paper_trend = compute_paper_trend_accuracy(
        histories,
        y_true,
        y_pred,
        alpha=paper_alpha,
        history_window=paper_history_window,
        horizons=list(paper_horizons),
    ).reset_index()
    for _, trend_row in paper_trend.iterrows():
        row[f"paper_d{int(trend_row['horizon'])}_acc"] = float(trend_row["accuracy"])
    return row


def aggregate_fold_rows(rows: pd.DataFrame) -> pd.DataFrame:
    if rows.empty:
        return pd.DataFrame()

    metric_cols = [col for col in rows.columns if col not in {"candidate", "fold"}]
    grouped = rows.groupby("candidate", sort=False)
    mean_df = grouped[metric_cols].mean().add_suffix("_mean")
    std_df = grouped[metric_cols].std(ddof=0).fillna(0.0).add_suffix("_std")
    count_df = grouped.size().to_frame("n_folds")
    out = pd.concat([count_df, mean_df, std_df], axis=1).reset_index()
    return out.sort_values("mse_path_mean", kind="stable").reset_index(drop=True)
