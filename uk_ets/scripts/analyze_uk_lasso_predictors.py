#!/usr/bin/env python3
"""Run the repo's existing lasso feature-selection methodology on the UK panel."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]

import sys

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sklearn.feature_selection import mutual_info_regression
from sklearn.linear_model import ElasticNet, Lasso
from sklearn.preprocessing import StandardScaler as SklearnScaler

from src.data.panel import build_panel, get_coverage_report
from src.data.windows import WindowConfig, make_windows, split_windows
from src.run_experiment import returns_to_prices, select_features_via_lasso


def _resolve(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return (PROJECT_ROOT / path).resolve()


def _feature_family(name: str) -> str:
    lname = name.lower()
    if lname in {"y", "y_return", "target_range_pct", "target_volume"} or lname.startswith("y_") or lname.startswith("target_"):
        return "target_derived"
    if lname.startswith("uk_icap") or lname.startswith("icap_"):
        return "uk_icap"
    if lname.startswith("auction_") or lname == "is_auction_day":
        return "uk_auction"
    if lname.startswith("uk_gas") or lname.startswith("uk_power") or lname.startswith("brent") or lname.startswith("coal"):
        return "energy"
    if lname.startswith("uk_temp") or lname.startswith("uk_hdd"):
        return "weather"
    if lname.startswith("idx_"):
        return "carbon_index"
    if "vstoxx" in lname or "vix" in lname or lname.startswith("uk_vol"):
        return "volatility"
    if lname.startswith("eurusd"):
        return "fx"
    return "other"


def _build_lasso_scores(
    X_enc_train: np.ndarray,
    y_train_future: np.ndarray,
    feature_cols: list[str],
    config: dict[str, Any],
) -> pd.DataFrame:
    """Mirror the current select_features_via_lasso summary-stat logic, but keep scores."""
    lasso_cfg = (config.get("econometric", {}) or {}).get("lasso_feature_selection", {}) or {}
    stats = lasso_cfg.get("feature_stats", ["last", "change_5", "change_20", "std_20"])
    target_horizons = lasso_cfg.get("target_horizons", [1, 5, 20, 30])
    target_aggregation = lasso_cfg.get("target_aggregation", "mean")
    mode = lasso_cfg.get("mode", "aggregate")
    selection_method = lasso_cfg.get("selection_method", "lasso")
    alpha = float(lasso_cfg.get("alpha", 0.01))
    max_iter = int(lasso_cfg.get("max_iter", 10000))
    coef_threshold = float(lasso_cfg.get("coef_threshold", 1.0e-6))

    X_summary = []
    feature_map = []
    stat_map = []
    for idx, name in enumerate(feature_cols):
        series = X_enc_train[:, :, idx]
        if "last" in stats:
            X_summary.append(series[:, -1])
            feature_map.append(name)
            stat_map.append("last")
        if "change_5" in stats and series.shape[1] >= 5:
            denom = np.where(np.abs(series[:, -5]) > 1e-8, series[:, -5], 1e-8)
            X_summary.append(series[:, -1] / denom - 1.0)
            feature_map.append(name)
            stat_map.append("change_5")
        if "change_20" in stats and series.shape[1] >= 20:
            denom = np.where(np.abs(series[:, -20]) > 1e-8, series[:, -20], 1e-8)
            X_summary.append(series[:, -1] / denom - 1.0)
            feature_map.append(name)
            stat_map.append("change_20")
        if "std_20" in stats:
            window = min(20, series.shape[1])
            X_summary.append(np.std(series[:, -window:], axis=1))
            feature_map.append(name)
            stat_map.append("std_20")

    X_mat = np.column_stack(X_summary).astype(np.float32)
    scaler = SklearnScaler()
    X_scaled = scaler.fit_transform(X_mat)

    horizon_indices = [h - 1 for h in target_horizons if 1 <= h <= y_train_future.shape[1]]
    if not horizon_indices:
        horizon_indices = [0]

    def score_one_target(y_target: np.ndarray, alpha_value: float) -> np.ndarray:
        if selection_method == "mutual_info":
            return mutual_info_regression(X_scaled, y_target, n_neighbors=3, random_state=42)
        if selection_method == "elastic_net":
            model = ElasticNet(
                alpha=alpha_value,
                l1_ratio=float(lasso_cfg.get("elastic_net_l1_ratio", 0.7)),
                max_iter=max_iter,
            )
        else:
            model = Lasso(alpha=alpha_value, max_iter=max_iter)
        model.fit(X_scaled, y_target)
        return np.abs(model.coef_)

    per_stat_scores: dict[tuple[str, str], float] = {}
    if mode == "per_horizon_union":
        for h_idx in horizon_indices:
            alpha_h = float(lasso_cfg.get("alpha_by_horizon", {}).get(str(h_idx + 1), alpha))
            coef_scores = score_one_target(y_train_future[:, h_idx], alpha_h)
            for base_name, stat_name, score in zip(feature_map, stat_map, coef_scores):
                per_stat_scores[(base_name, stat_name)] = per_stat_scores.get((base_name, stat_name), 0.0) + float(abs(score))
    else:
        y_subset = y_train_future[:, horizon_indices]
        if mode == "weighted":
            weights = np.asarray(lasso_cfg.get("target_weights", [1.0] * len(horizon_indices)), dtype=np.float32)
            if len(weights) != len(horizon_indices):
                weights = np.ones(len(horizon_indices), dtype=np.float32)
            y_target = np.dot(y_subset, weights) / max(float(np.sum(weights)), 1e-8)
        else:
            y_target = np.median(y_subset, axis=1) if target_aggregation == "median" else np.mean(y_subset, axis=1)
        if len(target_horizons) == 1:
            alpha = float(lasso_cfg.get("alpha_by_horizon", {}).get(str(target_horizons[0]), alpha))
        coef_scores = score_one_target(y_target, alpha)
        for base_name, stat_name, score in zip(feature_map, stat_map, coef_scores):
            per_stat_scores[(base_name, stat_name)] = per_stat_scores.get((base_name, stat_name), 0.0) + float(abs(score))

    rows = []
    feature_total: dict[str, float] = {}
    for (feature_name, stat_name), score in per_stat_scores.items():
        feature_total[feature_name] = feature_total.get(feature_name, 0.0) + score
        rows.append(
            {
                "feature": feature_name,
                "family": _feature_family(feature_name),
                "summary_stat": stat_name,
                "score": score,
                "selected_above_threshold": bool(score > coef_threshold),
            }
        )
    score_df = pd.DataFrame(rows).sort_values(["score", "feature"], ascending=[False, True]).reset_index(drop=True)

    total_df = (
        pd.DataFrame(
            [{"feature": k, "family": _feature_family(k), "total_score": v} for k, v in feature_total.items()]
        )
        .sort_values(["total_score", "feature"], ascending=[False, True])
        .reset_index(drop=True)
    )
    return score_df, total_df


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze UK predictors with the repo's lasso methodology.")
    parser.add_argument("--config", default="uk_ets/config/uk_ets_llm_4b_current_default.yaml")
    parser.add_argument("--data-dir", default="uk_ets/Data_auto_uk")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--max-candidate-features", type=int, default=999)
    args = parser.parse_args()

    config_path = _resolve(args.config)
    data_dir = _resolve(args.data_dir)
    cfg = yaml.safe_load(config_path.read_text())

    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.output_dir:
        out_dir = _resolve(args.output_dir)
    else:
        out_dir = PROJECT_ROOT / "reports" / "uk_ets_lasso_energy_weather" / now
    out_dir.mkdir(parents=True, exist_ok=True)

    panel, _ = build_panel(data_dir, cfg)
    coverage = get_coverage_report(panel)
    coverage["family"] = coverage["column"].map(_feature_family)
    coverage.to_csv(out_dir / "panel_coverage.csv", index=False)

    target_mode = cfg.get("target", {}).get("mode", "price")
    target_col = "y_return" if target_mode == "returns" else "y"
    candidate_exog = [c for c in panel.select_dtypes(include=[np.number]).columns if c not in {"date", target_col}]
    feature_cols = [target_col] + candidate_exog[: int(args.max_candidate_features)]

    window_config = WindowConfig(
        seq_len=int(cfg["time_series"]["seq_len"]),
        label_len=int(cfg["time_series"]["label_len"]),
        pred_len=int(cfg["time_series"]["pred_len"]),
        target_col=target_col,
        feature_cols=feature_cols,
    )
    X_enc, X_dec, y, dates, window_meta = make_windows(panel, window_config, mode="MS", return_metadata=True)
    splits = split_windows(
        X_enc,
        X_dec,
        y,
        dates,
        train_end=cfg["split"]["train_end"],
        val_end=cfg["split"]["val_end"],
        window_meta=window_meta,
    )

    if target_mode == "returns":
        price_idx = feature_cols.index("y")
        y_train_base = splits["train"]["X_enc"][:, :, price_idx][:, -1]
        y_train_future = returns_to_prices(splits["train"]["y"], y_train_base)
    else:
        y_train_future = splits["train"]["y"]

    analysis_cfg = json.loads(json.dumps(cfg))
    analysis_cfg.setdefault("econometric", {}).setdefault("lasso_feature_selection", {})
    analysis_cfg["econometric"]["lasso_feature_selection"]["enabled"] = True
    selected = select_features_via_lasso(
        splits["train"]["X_enc"],
        y_train_future,
        feature_cols,
        target_col,
        analysis_cfg,
    )

    stat_score_df, feature_score_df = _build_lasso_scores(
        splits["train"]["X_enc"],
        y_train_future,
        feature_cols,
        analysis_cfg,
    )
    stat_score_df.to_csv(out_dir / "lasso_summary_stat_scores.csv", index=False)
    feature_score_df.to_csv(out_dir / "lasso_feature_scores.csv", index=False)

    selected_payload = {
        "selected_features": selected,
        "config_path": str(config_path),
        "data_dir": str(data_dir),
        "train_windows": int(splits["train"]["X_enc"].shape[0]),
        "candidate_feature_count": int(len(feature_cols)),
        "analysis_generated_at": datetime.now().isoformat(),
    }
    (out_dir / "selected_features.json").write_text(json.dumps(selected_payload, indent=2), encoding="utf-8")

    focus_features = feature_score_df[
        feature_score_df["family"].isin(["energy", "weather"])
    ].copy()
    focus_features.to_csv(out_dir / "energy_weather_feature_scores.csv", index=False)

    summary_lines = [
        "# UK ETS LASSO Predictor Screen",
        "",
        f"Generated: {datetime.now().isoformat()}",
        f"Config: `{config_path}`",
        f"Data dir: `{data_dir}`",
        "",
        "Method:",
        "- Exact repo lasso-selection path from `src/run_experiment.py::select_features_via_lasso`.",
        "- Training split only, using the configured UK train boundary.",
        "- Current settings: summary stats = `last, change_5, change_20, std_20`; mode = weighted aggregate over horizons `[1,5,20,30]` with weights `[0.1,0.2,0.3,0.4]`.",
        f"- Candidate numeric feature count: `{len(feature_cols)}` including target and all available exogenous columns.",
        f"- Train windows used: `{splits['train']['X_enc'].shape[0]}`.",
        "",
        "Selected features:",
    ]
    for name in selected:
        summary_lines.append(f"- `{name}` ({_feature_family(name)})")

    summary_lines.extend(["", "Top energy/weather features by aggregate lasso score:"])
    for row in focus_features.head(12).itertuples(index=False):
        summary_lines.append(f"- `{row.feature}`: `{row.total_score:.6f}` ({row.family})")

    (out_dir / "summary.md").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    print(out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
