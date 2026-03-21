#!/usr/bin/env python3
"""Benchmark a validation-selected ridge + DLinear hybrid base on W1-W4."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]

import sys

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.windows import StandardScaler
from src.eval.metrics import compute_metrics_by_horizon, compute_path_metrics
from src.models.baselines import LinearBaseline
from src.run_experiment import blend_forecasts, returns_to_prices, run_experiment


DATA_DIR = "uk_ets/Data_auto_uk"
CONFIG_PATH = "uk_ets/config/uk_ets_llm_4b_tsm_live_gbdt_top20.yaml"
BLEND_WEIGHTS = [0.0, 0.25, 0.5, 0.75, 1.0]
SCHEDULES = ["uniform", "ramp"]


@dataclass(frozen=True)
class WindowSpec:
    name: str
    train_end: str
    val_end: str
    test_end: str


WINDOWS = [
    WindowSpec("W1", "2023-10-27", "2024-10-26", "2025-06-30"),
    WindowSpec("W2", "2023-02-22", "2024-02-22", "2024-10-26"),
    WindowSpec("W3", "2022-06-20", "2023-06-20", "2024-02-22"),
    WindowSpec("W4", "2021-10-16", "2022-10-16", "2023-06-20"),
]


BASE_MODEL = {
    "learning_rate": 0.001,
    "max_epochs": 40,
    "early_stopping_patience": 8,
    "dropout": 0.5,
    "weight_decay": 0.02,
    "dlinear_individual": False,
    "loss_type": "huber",
    "loss_huber_beta": 0.5,
}

ENERGY_INTERACTIONS = [
    "target_range_pct",
    "target_volume",
    "y_vol_20d",
    "y_ma_5d",
    "y_momentum_20d",
    "is_auction_day",
    "uk_icap_secondary_print_day",
    "coal_brent_ratio",
    "coal_brent_ratio_z20",
    "uk_power_gas_vol_ratio_20d",
    "uk_gas_hdd18_surprise_interaction",
    "uk_hdd18_7d_ma",
    "uka_brent_ratio",
    "uka_brent_ratio_z20",
    "uk_gas_vol_20d",
    "uk_temp_mean_c",
]


def _merge(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    out = dict(a)
    for k, v in b.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


def _base_overrides(window: WindowSpec) -> dict[str, Any]:
    return {
        "llm": {"methods": []},
        "time_series": {"seq_len": 20, "label_len": 10},
        "model": BASE_MODEL,
        "features": {
            "max_exogenous_features_model": len(ENERGY_INTERACTIONS),
            "preferred_feature_order": ENERGY_INTERACTIONS,
        },
        "split": {
            "train_end": window.train_end,
            "val_end": window.val_end,
            "test_end": window.test_end,
        },
    }


def _run_window(window: WindowSpec) -> Path:
    return Path(
        run_experiment(
            config_path=CONFIG_PATH,
            overrides=_base_overrides(window),
            data_dir=DATA_DIR,
        )
    ).resolve()


def _load_tsm_predictions(run_dir: Path, split_name: str) -> tuple[np.ndarray, np.ndarray]:
    pred_path = run_dir / "predictions" / f"tsm_pred_{split_name}.parquet"
    df = pd.read_parquet(pred_path)
    y_true_cols = [c for c in df.columns if c.startswith("y_true_t_plus_")]
    y_hat_cols = [c for c in df.columns if c.startswith("yhat_t_plus_")]
    y_true = df[y_true_cols].to_numpy(dtype=np.float32)
    y_pred = df[y_hat_cols].to_numpy(dtype=np.float32)
    return y_true, y_pred


def _load_raw_split(run_dir: Path, split_name: str) -> tuple[np.ndarray, np.ndarray]:
    data = np.load(run_dir / "data" / "datasets" / f"{split_name}.npz")
    scaler = StandardScaler.load(run_dir / "data" / "datasets" / "scaler.pkl")
    X_enc_scaled = data["X_enc"]
    y_scaled = data["y"]
    X_enc_raw = scaler.inverse_transform(X_enc_scaled)
    y_returns = scaler.inverse_transform_target(y_scaled)
    y_base = X_enc_raw[:, -1, 0]
    y_future = returns_to_prices(y_returns, y_base)
    y_hist = X_enc_raw[:, :, 0]
    return y_hist, y_future


def _fit_ridge_predictions(run_dir: Path) -> dict[str, np.ndarray]:
    y_train_hist, y_train_future = _load_raw_split(run_dir, "train")
    y_val_hist, _ = _load_raw_split(run_dir, "val")
    y_test_hist, _ = _load_raw_split(run_dir, "test")
    ridge = LinearBaseline(pred_len=30, model_type="ridge")
    ridge.fit(y_train_hist, y_train_future)
    return {
        "val": ridge.predict(y_val_hist),
        "test": ridge.predict(y_test_hist),
    }


def _select_blend(y_val_true: np.ndarray, ridge_val: np.ndarray, tsm_val: np.ndarray) -> dict[str, Any]:
    best: dict[str, Any] | None = None
    pred_len = int(y_val_true.shape[1])
    for schedule in SCHEDULES:
        for weight in BLEND_WEIGHTS:
            blended = blend_forecasts(
                ridge_val,
                tsm_val,
                strength=float(weight),
                schedule=schedule,
                pred_len=pred_len,
            )
            mse_path = float(compute_path_metrics(y_val_true, blended)["mse_path"])
            candidate = {
                "schedule": schedule,
                "weight": float(weight),
                "val_mse_path": mse_path,
            }
            if best is None or candidate["val_mse_path"] < best["val_mse_path"]:
                best = candidate
    assert best is not None
    return best


def _metrics_row(model_name: str, window: WindowSpec, run_dir: Path, y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    path = compute_path_metrics(y_true, y_pred)
    by_h = compute_metrics_by_horizon(y_true, y_pred, horizons=[1, 5, 20, 30])
    if "horizon" in by_h.columns:
        by_h = by_h.set_index("horizon")
    return {
        "model": model_name,
        "window": window.name,
        "train_end": window.train_end,
        "val_end": window.val_end,
        "test_end": window.test_end,
        "run_dir": str(run_dir),
        "path_mse": float(path["mse_path"]),
        "h1": float(by_h.loc[1, "mse"]),
        "h5": float(by_h.loc[5, "mse"]),
        "h20": float(by_h.loc[20, "mse"]),
        "h30": float(by_h.loc[30, "mse"]),
    }


def _write_summary(out_dir: Path, df: pd.DataFrame, selections: pd.DataFrame) -> None:
    mean_df = (
        df.groupby("model")[["path_mse", "h1", "h5", "h20", "h30"]]
        .mean()
        .sort_values("path_mse")
        .reset_index()
    )
    baseline = mean_df[mean_df["model"] == "dlinear_huber_energy16"].iloc[0]
    best = mean_df.iloc[0]
    gain = 1.0 - (float(best["path_mse"]) / float(baseline["path_mse"]))
    best_rows = df.loc[df.groupby("window")["path_mse"].idxmin()].sort_values("window")
    lines = [
        "# W1-W4 Hybrid Base Benchmark",
        "",
        f"Generated: {datetime.now().isoformat()}",
        "",
        "## Setup",
        "",
        "- Base-only benchmark on the current improved `Huber` `DLinear` anchor with the energy-interactions feature slate.",
        "- Fit `linear_ridge` on the exact same train split reconstructed from saved datasets.",
        "- Select the ridge/DLinear blend on validation only across weights `0, 0.25, 0.5, 0.75, 1.0` and schedules `uniform, ramp`.",
        "",
        "## Test Results",
        "",
        df.to_markdown(index=False),
        "",
        "## Mean By Model",
        "",
        mean_df.to_markdown(index=False),
        "",
        "## Blend Selections",
        "",
        selections.to_markdown(index=False),
        "",
        "## Best Model Per Window",
        "",
        best_rows.to_markdown(index=False),
        "",
        "## Headline",
        "",
        f"- Best mean model: `{best['model']}` with path MSE `{best['path_mse']:.6f}`.",
        f"- Baseline `dlinear_huber_energy16` path MSE: `{baseline['path_mse']:.6f}`.",
        f"- Improvement vs baseline: `{gain:.2%}`.",
    ]
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    out_dir = (
        PROJECT_ROOT
        / "reports"
        / "uk_ets_tsm_w1_w4_hybrid_base_benchmark"
        / datetime.now().strftime("%Y%m%d_%H%M%S")
    ).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    selections: list[dict[str, Any]] = []

    existing_results_path = out_dir / "results.csv"
    existing_selections_path = out_dir / "blend_selections.csv"
    completed_windows: set[str] = set()
    if existing_results_path.exists():
        existing_df = pd.read_csv(existing_results_path)
        rows.extend(existing_df.to_dict(orient="records"))
        completed_windows = set(existing_df["window"].unique())
    if existing_selections_path.exists():
        existing_sel = pd.read_csv(existing_selections_path)
        selections.extend(existing_sel.to_dict(orient="records"))

    for window in WINDOWS:
        if window.name in completed_windows:
            continue
        run_dir = _run_window(window)
        y_val_true, tsm_val = _load_tsm_predictions(run_dir, "val")
        y_test_true, tsm_test = _load_tsm_predictions(run_dir, "test")
        ridge_preds = _fit_ridge_predictions(run_dir)
        selection = _select_blend(y_val_true, ridge_preds["val"], tsm_val)
        blended_test = blend_forecasts(
            ridge_preds["test"],
            tsm_test,
            strength=float(selection["weight"]),
            schedule=str(selection["schedule"]),
            pred_len=int(y_test_true.shape[1]),
        )

        selections.append(
            {
                "window": window.name,
                "run_dir": str(run_dir),
                **selection,
            }
        )
        rows.append(_metrics_row("dlinear_huber_energy16", window, run_dir, y_test_true, tsm_test))
        rows.append(_metrics_row("linear_ridge", window, run_dir, y_test_true, ridge_preds["test"]))
        rows.append(_metrics_row("ridge_dlinear_blend", window, run_dir, y_test_true, blended_test))

        pd.DataFrame(rows).to_csv(out_dir / "results.csv", index=False)
        pd.DataFrame(selections).to_csv(out_dir / "blend_selections.csv", index=False)

    df = pd.DataFrame(rows)
    sel_df = pd.DataFrame(selections)
    df.to_csv(out_dir / "results.csv", index=False)
    sel_df.to_csv(out_dir / "blend_selections.csv", index=False)
    _write_summary(out_dir, df, sel_df)
    print(out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
