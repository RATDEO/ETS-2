#!/usr/bin/env python3
"""
Plot actual vs model predictions by horizon for a given run directory.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import yaml
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.windows import make_windows, split_windows, WindowConfig
from src.models.baselines import NaivePersistence, SeasonalNaive, LinearBaseline
from src.models.quantile_lasso import QuantileLasso, QuantileLassoConfig


def returns_to_prices(returns: np.ndarray, last_price: np.ndarray) -> np.ndarray:
    last_price = np.asarray(last_price).reshape(-1, 1)
    return last_price * np.exp(np.cumsum(returns, axis=1))


def load_config(run_dir: Path) -> dict:
    config_path = run_dir / "config_resolved.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing config_resolved.yaml in {run_dir}")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def build_feature_cols(panel: pd.DataFrame, target_col: str) -> List[str]:
    feature_candidates = [c for c in panel.columns if c not in ["date", target_col]]
    if "y" in feature_candidates:
        feature_candidates.remove("y")
        feature_candidates = ["y"] + feature_candidates
    return [target_col] + feature_candidates[:10]


def compute_baselines(
    y_train_hist: np.ndarray,
    y_test_hist: np.ndarray,
    y_train_future: np.ndarray,
    pred_len: int,
) -> Dict[str, np.ndarray]:
    preds: Dict[str, np.ndarray] = {}

    naive = NaivePersistence(pred_len)
    naive.fit(y_train_hist.flatten())
    preds["naive_persistence"] = naive.predict(y_test_hist)

    seasonal = SeasonalNaive(pred_len, season_period=5)
    seasonal.fit(y_train_hist.flatten())
    preds["seasonal_naive"] = seasonal.predict(y_test_hist)

    linear_ridge = LinearBaseline(pred_len, model_type="ridge")
    linear_ridge.fit(y_train_hist, y_train_future)
    preds["linear_ridge"] = linear_ridge.predict(y_test_hist)

    linear_lasso = LinearBaseline(pred_len, model_type="lasso")
    linear_lasso.fit(y_train_hist, y_train_future)
    preds["linear_lasso"] = linear_lasso.predict(y_test_hist)

    return preds


def load_tsm_predictions(run_dir: Path, label: str = "tsm") -> Dict[str, np.ndarray]:
    pred_path = run_dir / "predictions" / "tsm_pred_test.parquet"
    if not pred_path.exists():
        return {}
    df = pd.read_parquet(pred_path)
    horizons = sorted(
        int(col.replace("yhat_t_plus_", ""))
        for col in df.columns
        if col.startswith("yhat_t_plus_")
    )
    preds = np.column_stack([df[f"yhat_t_plus_{h}"].to_numpy() for h in horizons])
    return {label: preds}


def load_tsm_lasso_predictions(run_dir: Path, label: str = "tsm_lasso") -> Dict[str, np.ndarray]:
    pred_path = run_dir / "predictions" / "tsm_lasso_pred_test.parquet"
    if not pred_path.exists():
        return {}
    df = pd.read_parquet(pred_path)
    horizons = sorted(
        int(col.replace("yhat_t_plus_", ""))
        for col in df.columns
        if col.startswith("yhat_t_plus_")
    )
    preds = np.column_stack([df[f"yhat_t_plus_{h}"].to_numpy() for h in horizons])
    return {label: preds}


def maybe_add_quantile_lasso(
    preds: Dict[str, np.ndarray],
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train_future: np.ndarray,
    feature_cols: List[str],
    target_col: str,
    config: dict,
    quantile: float,
    enabled_override: Optional[bool] = None,
) -> None:
    econ_cfg = config.get("econometric", {}).get("quantile_lasso", {})
    enabled = econ_cfg.get("enabled", False) if enabled_override is None else enabled_override
    if not enabled:
        return
    q_config = QuantileLassoConfig(
        quantiles=econ_cfg.get("quantiles", [0.1, 0.5, 0.9]),
        alpha=econ_cfg.get("alpha", 0.01),
        lags=econ_cfg.get("lags", [1, 5, 10, 20]),
    )
    if quantile not in q_config.quantiles:
        raise ValueError(f"Quantile {quantile} not in config quantiles {q_config.quantiles}")
    qlasso = QuantileLasso(q_config)
    qlasso.fit(
        X_train,
        feature_cols,
        target_idx=feature_cols.index(target_col),
        y_future=y_train_future,
    )
    q_pred = qlasso.predict(X_test)
    q_idx = q_config.quantiles.index(quantile)
    preds[f"quantile_lasso_q{quantile}"] = q_pred[:, :, q_idx]


def plot_horizons(
    dates: np.ndarray,
    y_true: np.ndarray,
    preds: Dict[str, np.ndarray],
    horizons: List[int],
    out_path: Path,
):
    n = len(horizons)
    ncols = 2
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 6 + 3 * nrows), sharex=False)
    axes = np.atleast_1d(axes).reshape(nrows, ncols)

    for idx, h in enumerate(horizons):
        r = idx // ncols
        c = idx % ncols
        ax = axes[r, c]
        h_idx = h - 1
        ax.plot(dates, y_true[:, h_idx], label="actual", color="black", linewidth=2)
        for name, pred in preds.items():
            ax.plot(dates, pred[:, h_idx], label=name, alpha=0.75)
        ax.set_title(f"Horizon t+{h}")
        ax.set_xlabel("date")
        ax.set_ylabel("price")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, ncol=2)

    for idx in range(n, nrows * ncols):
        r = idx // ncols
        c = idx % ncols
        axes[r, c].axis("off")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot model predictions vs actuals.")
    parser.add_argument("--run-dir", required=True, help="Run directory path.")
    parser.add_argument(
        "--out",
        default=None,
        help="Output path for PNG (default: run_dir/paper_snapshot/fig_predictions_horizons.png).",
    )
    parser.add_argument(
        "--horizons",
        default="1,5,20,30",
        help="Comma-separated horizons to plot.",
    )
    parser.add_argument(
        "--compare-tsm-run",
        default=None,
        help="Optional run dir for previous TSM predictions to compare.",
    )
    parser.add_argument(
        "--quantile",
        default="0.5",
        help="Quantile to plot for quantile-lasso (default: 0.5).",
    )
    parser.add_argument(
        "--only-models",
        default=None,
        help="Comma-separated model keys to plot (actual is always shown).",
    )
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    config = load_config(run_dir)
    panel_path = run_dir / "data" / "panel.parquet"
    panel = pd.read_parquet(panel_path)

    target_mode = config.get("target", {}).get("mode", "price")
    target_col = "y_return" if target_mode == "returns" else "y"
    feature_cols = build_feature_cols(panel, target_col)

    window_config = WindowConfig(
        seq_len=config.get("time_series", {}).get("seq_len", 120),
        label_len=config.get("time_series", {}).get("label_len", 30),
        pred_len=config.get("time_series", {}).get("pred_len", 30),
        target_col=target_col,
        feature_cols=feature_cols,
    )

    X_enc, X_dec, y, dates, window_meta = make_windows(
        panel,
        window_config,
        mode="MS",
        return_metadata=True,
    )
    splits = split_windows(
        X_enc,
        X_dec,
        y,
        dates,
        train_end=config.get("split", {}).get("train_end", "2022-12-31"),
        val_end=config.get("split", {}).get("val_end", "2023-12-31"),
        window_meta=window_meta,
    )

    target_is_returns = target_mode == "returns"
    price_idx = feature_cols.index("y") if "y" in feature_cols else None
    if target_is_returns and price_idx is None:
        raise ValueError("Returns target requires price feature 'y' in window features.")

    if target_is_returns:
        y_train_hist = splits["train"]["X_enc"][:, :, price_idx]
        y_test_hist = splits["test"]["X_enc"][:, :, price_idx]
        y_train_base = y_train_hist[:, -1]
        y_test_base = y_test_hist[:, -1]
        y_train_future = returns_to_prices(splits["train"]["y"], y_train_base)
        y_test_future = returns_to_prices(splits["test"]["y"], y_test_base)
    else:
        y_train_hist = splits["train"]["X_enc"][:, :, 0]
        y_test_hist = splits["test"]["X_enc"][:, :, 0]
        y_train_future = splits["train"]["y"]
        y_test_future = splits["test"]["y"]

    preds = compute_baselines(
        y_train_hist=y_train_hist,
        y_test_hist=y_test_hist,
        y_train_future=y_train_future,
        pred_len=window_config.pred_len,
    )
    preds.update(load_tsm_predictions(run_dir, label="tsm_current"))
    preds.update(load_tsm_lasso_predictions(run_dir, label="tsm_lasso"))
    if args.compare_tsm_run:
        preds.update(load_tsm_predictions(Path(args.compare_tsm_run), label="tsm_previous"))

    quantile_value = float(args.quantile)
    maybe_add_quantile_lasso(
        preds,
        X_train=splits["train"]["X_enc"],
        X_test=splits["test"]["X_enc"],
        y_train_future=y_train_future,
        feature_cols=feature_cols,
        target_col=target_col,
        config=config,
        quantile=quantile_value,
    )

    if args.only_models:
        allowed = {name.strip() for name in args.only_models.split(",") if name.strip()}
        preds = {name: pred for name, pred in preds.items() if name in allowed}

    horizons = [int(h.strip()) for h in args.horizons.split(",") if h.strip()]
    out_path = (
        Path(args.out)
        if args.out
        else run_dir / "paper_snapshot" / "fig_predictions_horizons.png"
    )

    plot_horizons(
        dates=splits["test"]["dates"],
        y_true=y_test_future,
        preds=preds,
        horizons=horizons,
        out_path=out_path,
    )

    print(f"Saved plot: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
