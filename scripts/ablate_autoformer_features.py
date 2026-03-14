#!/usr/bin/env python3
"""Run Autoformer feature-group ablations on the EU ETS pipeline."""

from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List
import sys

import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config
from src.data.panel import build_panel
from src.data.windows import WindowConfig, make_windows, split_windows, StandardScaler, TimeSeriesDataset
from src.eval.metrics import compute_metrics_by_horizon, compute_path_metrics
from src.models.tsm import TSMForecaster
from src.utils import set_seed


@dataclass
class Variant:
    name: str
    feature_overrides: Dict[str, bool]
    drop_prefixes: List[str] = field(default_factory=list)
    drop_columns: List[str] = field(default_factory=list)


def returns_to_prices(returns: np.ndarray, last_price: np.ndarray) -> np.ndarray:
    last_price = np.asarray(last_price).reshape(-1, 1)
    return last_price * np.exp(np.cumsum(returns, axis=1))


def build_feature_cols(panel: pd.DataFrame, target_col: str, max_features: int) -> List[str]:
    feature_candidates = [c for c in panel.columns if c not in ["date", target_col]]
    if "y" in feature_candidates:
        feature_candidates.remove("y")
        feature_candidates = ["y"] + feature_candidates
    return [target_col] + feature_candidates[:max_features]


def train_and_eval_variant(
    cfg_raw: dict,
    data_dir: Path,
    variant: Variant,
    seed: int,
    epochs: int,
    patience: int,
    batch_size: int,
    max_features: int,
) -> dict:
    set_seed(seed)

    run_cfg = copy.deepcopy(cfg_raw)
    run_cfg.setdefault("features", {})
    run_cfg["features"].update(variant.feature_overrides)

    panel, _ = build_panel(data_dir=data_dir, config=run_cfg)
    if variant.drop_prefixes or variant.drop_columns:
        drop_cols = []
        for col in panel.columns:
            if col in variant.drop_columns:
                drop_cols.append(col)
                continue
            if any(col.startswith(prefix) for prefix in variant.drop_prefixes):
                drop_cols.append(col)
        if drop_cols:
            panel = panel.drop(columns=drop_cols)

    target_mode = run_cfg.get("target", {}).get("mode", "price")
    target_col = "y_return" if target_mode == "returns" else "y"
    feature_cols = build_feature_cols(panel, target_col, max_features=max_features)

    window_cfg = WindowConfig(
        seq_len=run_cfg.get("time_series", {}).get("seq_len", 120),
        label_len=run_cfg.get("time_series", {}).get("label_len", 30),
        pred_len=run_cfg.get("time_series", {}).get("pred_len", 30),
        target_col=target_col,
        feature_cols=feature_cols,
    )

    X_enc, X_dec, y, dates, window_meta = make_windows(
        panel,
        window_cfg,
        mode="MS",
        return_metadata=True,
    )
    splits = split_windows(
        X_enc,
        X_dec,
        y,
        dates,
        train_end=run_cfg.get("split", {}).get("train_end", "2023-06-30"),
        val_end=run_cfg.get("split", {}).get("val_end", "2024-06-30"),
        window_meta=window_meta,
    )

    target_is_returns = target_mode == "returns"
    if target_is_returns:
        if "y" not in feature_cols:
            raise ValueError("Returns mode requires 'y' in selected features.")
        price_idx = feature_cols.index("y")
        y_train_hist = splits["train"]["X_enc"][:, :, price_idx]
        y_val_hist = splits["val"]["X_enc"][:, :, price_idx]
        y_test_hist = splits["test"]["X_enc"][:, :, price_idx]
        y_train_base = y_train_hist[:, -1]
        y_val_base = y_val_hist[:, -1]
        y_test_base = y_test_hist[:, -1]
        y_train_future = returns_to_prices(splits["train"]["y"], y_train_base)
        y_val_future = returns_to_prices(splits["val"]["y"], y_val_base)
        y_test_future = returns_to_prices(splits["test"]["y"], y_test_base)
    else:
        y_train_future = splits["train"]["y"]
        y_val_future = splits["val"]["y"]
        y_test_future = splits["test"]["y"]

    scaler = StandardScaler()
    scaler.fit(splits["train"]["X_enc"])

    train_ds = TimeSeriesDataset(
        scaler.transform(splits["train"]["X_enc"]),
        scaler.transform(splits["train"]["X_dec"]),
        (splits["train"]["y"] - scaler.mean_[0]) / scaler.std_[0],
    )
    val_ds = TimeSeriesDataset(
        scaler.transform(splits["val"]["X_enc"]),
        scaler.transform(splits["val"]["X_dec"]),
        (splits["val"]["y"] - scaler.mean_[0]) / scaler.std_[0],
    )
    test_ds = TimeSeriesDataset(
        scaler.transform(splits["test"]["X_enc"]),
        scaler.transform(splits["test"]["X_dec"]),
        (splits["test"]["y"] - scaler.mean_[0]) / scaler.std_[0],
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)
    test_loader = DataLoader(test_ds, batch_size=batch_size)

    tsm_cfg = copy.deepcopy(run_cfg)
    tsm_cfg.setdefault("model", {})
    tsm_cfg["model"]["tsm_type"] = "autoformer"
    tsm_cfg["model"]["enc_in"] = splits["train"]["X_enc"].shape[-1]
    tsm_cfg["model"]["dec_in"] = splits["train"]["X_dec"].shape[-1]

    model = TSMForecaster(tsm_cfg)
    history = model.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=epochs,
        patience=patience,
        save_path=None,
    )

    pred_scaled, _ = model.predict(test_loader)
    pred = scaler.inverse_transform_target(pred_scaled)
    if target_is_returns:
        pred_eval = returns_to_prices(pred, y_test_base)
    else:
        pred_eval = pred

    horizons = run_cfg.get("time_series", {}).get("horizons", [1, 5, 20, 30])
    metrics_h = compute_metrics_by_horizon(y_test_future, pred_eval, horizons=horizons)
    path_metrics = compute_path_metrics(y_test_future, pred_eval)

    rows = {
        "variant": variant.name,
        "seed": seed,
        "n_features": len(feature_cols),
        "feature_cols": ",".join(feature_cols),
        "dropped_cols": ",".join(sorted(set(variant.drop_columns))),
        "dropped_prefixes": ",".join(sorted(set(variant.drop_prefixes))),
        "contains_icap": int(any(c.startswith("icap_") for c in feature_cols)),
        "contains_coal": int(any(c.startswith("coal_") for c in feature_cols)),
        "contains_brent": int(any(c.startswith("brent_") for c in feature_cols)),
        "contains_indices": int(any(c.startswith("idx_") for c in feature_cols)),
        "contains_auctions": int(any("auction" in c for c in feature_cols)),
        "contains_vstoxx": int(any(c.startswith("vstoxx") for c in feature_cols)),
        "best_val_loss": float(model.best_val_loss),
        "epochs_ran": len(history.get("val_loss", [])),
        "mse_path": float(path_metrics.get("mse_path", np.nan)),
        "rmse_path": float(path_metrics.get("rmse_path", np.nan)),
    }
    for h in horizons:
        if h in metrics_h.index:
            rows[f"h{h}_mse"] = float(metrics_h.loc[h, "mse"])
    return rows


def build_variants(compare_all_groups: bool) -> List[Variant]:
    variants = [
        Variant("full", {}),
        Variant(
            "no_icap",
            {"include_icap_secondary": False},
            drop_prefixes=["icap_"],
        ),
        Variant(
            "no_coal",
            {"include_coal": False},
            drop_prefixes=["coal_"],
            drop_columns=["coal_brent_ratio"],
        ),
        Variant(
            "no_icap_no_coal",
            {"include_icap_secondary": False, "include_coal": False},
            drop_prefixes=["icap_", "coal_"],
            drop_columns=["coal_brent_ratio"],
        ),
    ]
    if compare_all_groups:
        variants.extend(
            [
                Variant("no_brent", {"include_brent": False}),
                Variant("no_indices", {"include_indices": False}),
                Variant("no_auctions", {"include_auctions": False}),
                Variant("no_vstoxx", {"include_vstoxx": False}),
            ]
        )
    return variants


def main() -> int:
    parser = argparse.ArgumentParser(description="Autoformer feature-group ablations.")
    parser.add_argument("--config", default="src/config/default.yaml")
    parser.add_argument("--data-dir", default="Data_auto")
    parser.add_argument("--output-dir", default="reports/feature_ablation")
    parser.add_argument("--seeds", default="42", help="Comma-separated seeds")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--max-features",
        type=int,
        default=10,
        help="Max non-date features to keep (matches run_experiment default=10).",
    )
    parser.add_argument(
        "--compare-all-groups",
        action="store_true",
        help="Also run no_brent/no_indices/no_auctions/no_vstoxx variants.",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    config = load_config(args.config)
    seeds = [int(x.strip()) for x in str(args.seeds).split(",") if x.strip()]
    variants = build_variants(compare_all_groups=args.compare_all_groups)

    all_rows: List[dict] = []
    for seed in seeds:
        for variant in variants:
            print(f"[ablation] seed={seed} variant={variant.name}")
            row = train_and_eval_variant(
                cfg_raw=config.raw,
                data_dir=data_dir,
                variant=variant,
                seed=seed,
                epochs=args.epochs,
                patience=args.patience,
                batch_size=args.batch_size,
                max_features=args.max_features,
            )
            all_rows.append(row)

    runs_df = pd.DataFrame(all_rows)
    runs_path = output_dir / "autoformer_feature_ablation_runs.csv"
    runs_df.to_csv(runs_path, index=False)

    agg = (
        runs_df.groupby("variant", as_index=False)
        .agg(
            n_runs=("seed", "count"),
            n_features=("n_features", "mean"),
            mse_path_mean=("mse_path", "mean"),
            mse_path_std=("mse_path", "std"),
            h1_mse_mean=("h1_mse", "mean"),
            h5_mse_mean=("h5_mse", "mean"),
            h20_mse_mean=("h20_mse", "mean"),
            h30_mse_mean=("h30_mse", "mean"),
            contains_icap=("contains_icap", "max"),
            contains_coal=("contains_coal", "max"),
        )
    )

    if "full" in set(agg["variant"]):
        base = float(agg.loc[agg["variant"] == "full", "mse_path_mean"].iloc[0])
        agg["delta_mse_path_vs_full"] = agg["mse_path_mean"] - base
        agg["pct_change_vs_full"] = np.where(
            base != 0,
            (agg["delta_mse_path_vs_full"] / base) * 100.0,
            np.nan,
        )

    summary_path = output_dir / "autoformer_feature_ablation_summary.csv"
    agg.to_csv(summary_path, index=False)

    print(f"Saved run-level results: {runs_path}")
    print(f"Saved summary results: {summary_path}")
    print(agg.sort_values("mse_path_mean").to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
