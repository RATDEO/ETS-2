"""Residual correction models over a frozen base TSM forecast."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler as SklearnStandardScaler

from .metrics import compute_metrics_by_horizon, compute_path_metrics

DEFAULT_MARKET_COLUMNS = (
    "y",
    "y_return",
    "y_ma_5d",
    "y_ma_20d",
    "y_vol_20d",
    "y_momentum_5d",
    "y_momentum_20d",
    "eua_volume",
    "eua_range_pct",
    "brent_return",
    "coal_return",
    "vstoxx",
    "vstoxx_high_vol",
)


@dataclass(frozen=True)
class ResidualCandidate:
    name: str
    estimator: str
    feature_blocks: tuple[str, ...]
    target_kind: str = "absolute"
    alpha: float = 1.0
    learning_rate: float = 0.05
    max_depth: int = 3
    max_iter: int = 250
    min_samples_leaf: int = 10
    l2_regularization: float = 0.0
    max_adjustment_pct: float | None = None


def build_origin_feature_frame(
    panel: pd.DataFrame,
    event_panel: pd.DataFrame,
    origin_dates: Iterable[pd.Timestamp],
    base_pred: np.ndarray,
    market_cols: Iterable[str] = DEFAULT_MARKET_COLUMNS,
    anchor_horizons: Iterable[int] = (5, 10, 20, 30),
) -> pd.DataFrame:
    """Build per-origin supervised features using only information known at the origin."""
    origin_index = pd.DatetimeIndex(pd.to_datetime(list(origin_dates))).normalize()
    panel_idx = panel.copy()
    panel_idx["date"] = pd.to_datetime(panel_idx["date"]).dt.normalize()
    panel_idx = panel_idx.drop_duplicates(subset="date", keep="last").set_index("date")

    event_idx = event_panel.copy()
    event_idx["date"] = pd.to_datetime(event_idx["date"]).dt.normalize()
    event_idx = event_idx.drop_duplicates(subset="date", keep="last").set_index("date")

    frame = pd.DataFrame({"origin_date": origin_index})
    frame["origin_date"] = pd.to_datetime(frame["origin_date"]).dt.normalize()

    market_cols = [col for col in market_cols if col in panel_idx.columns]
    market = panel_idx.reindex(origin_index)[market_cols].reset_index(drop=True).fillna(0.0)
    market = market.add_prefix("mkt_")
    frame = pd.concat([frame.reset_index(drop=True), market], axis=1)

    event = event_idx.reindex(origin_index).reset_index(drop=True)
    if "date" in event.columns:
        event = event.drop(columns=["date"])
    event = event.fillna(0.0)
    frame = pd.concat([frame, event], axis=1)

    base_pred = np.asarray(base_pred, dtype=float)
    anchor_horizons = tuple(int(h) for h in anchor_horizons)
    frame["base_yhat_h1"] = base_pred[:, 0]
    for horizon in anchor_horizons:
        frame[f"base_yhat_h{horizon}"] = base_pred[:, horizon - 1]

    frame["base_spread_30_1"] = base_pred[:, 29] - base_pred[:, 0]
    frame["base_slope_1_5"] = (base_pred[:, 4] - base_pred[:, 0]) / 4.0
    frame["base_slope_5_10"] = (base_pred[:, 9] - base_pred[:, 4]) / 5.0
    frame["base_slope_10_20"] = (base_pred[:, 19] - base_pred[:, 9]) / 10.0
    frame["base_slope_20_30"] = (base_pred[:, 29] - base_pred[:, 19]) / 10.0
    return frame


def block_columns(feature_frame: pd.DataFrame) -> dict[str, list[str]]:
    """Expose stable feature blocks for candidate definitions."""
    base_cols = sorted(col for col in feature_frame.columns if col.startswith("base_"))
    market_cols = sorted(col for col in feature_frame.columns if col.startswith("mkt_"))
    event_cols = sorted(
        col
        for col in feature_frame.columns
        if col not in {"origin_date"} and not col.startswith(("base_", "mkt_"))
    )
    event_core_names = {
        "evt_news_count",
        "evt_news_count_3d_sum",
        "evt_news_count_7d_sum",
        "evt_weighted_sent_sum",
        "evt_weighted_sent_sum_3d_sum",
        "evt_weighted_sent_sum_7d_sum",
        "evt_importance_sum",
        "evt_importance_sum_3d_sum",
        "evt_importance_sum_7d_sum",
        "evt_eu_ets_strong_count",
        "evt_eu_ets_strong_count_7d_sum",
        "evt_energy_driver_count",
        "evt_energy_driver_count_7d_sum",
        "evt_carbon_terms_sum",
        "evt_carbon_terms_sum_7d_sum",
        "proxy_sent_score",
        "proxy_sent_change",
        "proxy_news_volume",
        "proxy_news_volume_3d_sum",
        "topical_sent_score",
        "ds_p_yes",
        "ds_p_no",
        "ds_sent_score_ds",
        "evt2_policy_count",
        "evt2_policy_count_7d_sum",
        "evt2_policy_weighted_sent_sum",
        "evt2_policy_weighted_sent_sum_7d_sum",
        "evt2_auction_count",
        "evt2_auction_count_7d_sum",
        "evt2_energy_count",
        "evt2_energy_count_7d_sum",
        "evt2_energy_weighted_sent_sum",
        "evt2_energy_weighted_sent_sum_7d_sum",
        "evt2_shipping_count",
        "evt2_shipping_count_7d_sum",
        "evt2_shipping_weighted_sent_sum",
        "evt2_shipping_weighted_sent_sum_7d_sum",
        "evt2_cbam_count",
        "evt2_cbam_count_7d_sum",
        "evt2_geopolitics_count",
        "evt2_geopolitics_count_7d_sum",
        "evt_official_type_auction_supply_count",
        "evt_official_type_auction_supply_count_7d_sum",
        "evt_official_type_auction_schedule_count",
        "evt_official_type_auction_schedule_count_7d_sum",
        "evt_official_type_policy_rule_change_count",
        "evt_official_type_policy_rule_change_count_7d_sum",
        "evt_official_type_verified_emissions_count",
        "evt_official_type_verified_emissions_count_7d_sum",
        "evt_official_type_maritime_ets_count",
        "evt_official_type_maritime_ets_count_7d_sum",
        "evt_official_type_aviation_ets_count",
        "evt_official_type_aviation_ets_count_7d_sum",
        "evt_official_type_cbam_count",
        "evt_official_type_cbam_count_7d_sum",
        "evt_official_type_auction_supply_weighted_sum",
        "evt_official_type_auction_supply_weighted_sum_7d_sum",
        "evt_official_type_policy_rule_change_weighted_sum",
        "evt_official_type_policy_rule_change_weighted_sum_7d_sum",
        "evt_official_type_verified_emissions_weighted_sum",
        "evt_official_type_verified_emissions_weighted_sum_7d_sum",
        "evt_official_channel_allowance_supply_count",
        "evt_official_channel_allowance_supply_count_7d_sum",
        "evt_official_channel_compliance_demand_count",
        "evt_official_channel_compliance_demand_count_7d_sum",
        "evt_official_horizon_1_5d_count",
        "evt_official_horizon_1_5d_count_7d_sum",
        "evt_official_horizon_5_20d_count",
        "evt_official_horizon_5_20d_count_7d_sum",
        "evt_official_dir_bullish_count",
        "evt_official_dir_bullish_count_7d_sum",
        "evt_official_dir_bearish_count",
        "evt_official_dir_bearish_count_7d_sum",
        "evt_official_stage_implementation_count",
        "evt_official_stage_implementation_count_7d_sum",
        "evt_official_novelty_scheduled_count",
        "evt_official_novelty_scheduled_count_7d_sum",
    }
    event_core_cols = [col for col in event_cols if col in event_core_names]
    return {
        "base": base_cols,
        "market": market_cols,
        "event": event_cols,
        "event_core": event_core_cols,
    }


def extract_anchor_residual_targets(
    y_true: np.ndarray,
    base_pred: np.ndarray,
    anchor_horizons: Iterable[int] = (5, 10, 20, 30),
    target_kind: str = "absolute",
) -> np.ndarray:
    anchor_horizons = tuple(int(h) for h in anchor_horizons)
    targets = np.column_stack([y_true[:, h - 1] - base_pred[:, h - 1] for h in anchor_horizons]).astype(float)
    if target_kind == "absolute":
        return targets
    if target_kind == "pct":
        denom = np.maximum(np.abs(np.column_stack([base_pred[:, h - 1] for h in anchor_horizons]).astype(float)), 1e-6)
        return targets / denom
    raise ValueError(f"Unknown target_kind: {target_kind}")


def interpolate_anchor_residuals(
    anchor_residuals: np.ndarray,
    pred_len: int = 30,
    anchor_horizons: Iterable[int] = (5, 10, 20, 30),
    freeze_h1: bool = True,
) -> np.ndarray:
    """Linearly interpolate anchor residuals into a full residual path."""
    anchor_residuals = np.asarray(anchor_residuals, dtype=float)
    anchor_horizons = tuple(int(h) for h in anchor_horizons)
    if anchor_residuals.ndim != 2 or anchor_residuals.shape[1] != len(anchor_horizons):
        raise ValueError("anchor_residuals must have shape (n_samples, len(anchor_horizons)).")

    target_x = np.arange(1, pred_len + 1, dtype=float)
    base_x = list(anchor_horizons)
    residual_paths = np.zeros((anchor_residuals.shape[0], pred_len), dtype=float)

    for idx in range(anchor_residuals.shape[0]):
        x_points = base_x
        y_points = anchor_residuals[idx]
        if freeze_h1:
            x_points = [1] + x_points
            y_points = np.concatenate([[0.0], y_points])
        residual_paths[idx] = np.interp(target_x, np.asarray(x_points, dtype=float), np.asarray(y_points, dtype=float))
    return residual_paths


def summarize_prediction(name: str, y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float | str]:
    metrics = compute_metrics_by_horizon(y_true, y_pred, horizons=[1, 5, 10, 20, 30]).reset_index()
    row: dict[str, float | str] = {"variant": name, "path_mse": float(compute_path_metrics(y_true, y_pred)["mse_path"])}
    for _, metric_row in metrics.iterrows():
        row[f"h{int(metric_row['horizon'])}_mse"] = float(metric_row["mse"])
    return row


def make_candidate_predictions(
    candidate: ResidualCandidate,
    X_fit: np.ndarray,
    y_fit_anchor: np.ndarray,
    X_apply: np.ndarray,
    base_apply_pred: np.ndarray,
    pred_len: int = 30,
    anchor_horizons: Iterable[int] = (5, 10, 20, 30),
) -> np.ndarray:
    """Fit a candidate on anchor residuals and return full corrected price paths."""
    if candidate.estimator == "identity":
        return np.asarray(base_apply_pred, dtype=float)

    anchor_horizons = tuple(int(h) for h in anchor_horizons)
    anchor_predictions = np.zeros((X_apply.shape[0], len(anchor_horizons)), dtype=float)

    for anchor_idx in range(len(anchor_horizons)):
        if candidate.estimator == "ridge":
            model = make_pipeline(
                SklearnStandardScaler(),
                Ridge(alpha=float(candidate.alpha)),
            )
        elif candidate.estimator == "hgb":
            model = HistGradientBoostingRegressor(
                learning_rate=float(candidate.learning_rate),
                max_depth=int(candidate.max_depth),
                max_iter=int(candidate.max_iter),
                min_samples_leaf=int(candidate.min_samples_leaf),
                l2_regularization=float(candidate.l2_regularization),
                random_state=0,
            )
        else:
            raise ValueError(f"Unknown estimator: {candidate.estimator}")
        model.fit(X_fit, y_fit_anchor[:, anchor_idx])
        anchor_predictions[:, anchor_idx] = model.predict(X_apply)

    if candidate.target_kind == "absolute":
        residual_path = interpolate_anchor_residuals(
            anchor_predictions,
            pred_len=pred_len,
            anchor_horizons=anchor_horizons,
            freeze_h1=True,
        )
        return np.asarray(base_apply_pred, dtype=float) + residual_path
    if candidate.target_kind == "pct":
        if candidate.max_adjustment_pct is not None:
            clip_value = float(candidate.max_adjustment_pct) / 100.0
            anchor_predictions = np.clip(anchor_predictions, -clip_value, clip_value)
        delta_path = interpolate_anchor_residuals(
            anchor_predictions,
            pred_len=pred_len,
            anchor_horizons=anchor_horizons,
            freeze_h1=True,
        )
        return np.asarray(base_apply_pred, dtype=float) * (1.0 + delta_path)
    raise ValueError(f"Unknown target_kind: {candidate.target_kind}")
