#!/usr/bin/env python3
"""Targeted post-2023 TSM search for the persistence-dominated regime."""

from __future__ import annotations

import argparse
import copy
import json
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from config import load_config
from data import build_panel, select_feature_columns
from data.windows import StandardScaler, TimeSeriesDataset, WindowConfig, make_windows, split_windows
from eval.metrics import compute_metrics_by_horizon, compute_path_metrics
from models.baselines import NaivePersistence, SeasonalNaive, LinearBaseline
from models.tsm import TSMForecaster
from run_experiment import returns_to_prices
from utils import set_seed

try:
    from torch.utils.data import DataLoader
except ImportError as exc:  # pragma: no cover
    raise SystemExit("PyTorch is required for tuning") from exc


@dataclass(frozen=True)
class Candidate:
    name: str
    tsm_type: str
    target_mode: str
    seq_len: int
    label_len: int
    batch_size: int
    learning_rate: float
    max_epochs: int
    patience: int
    d_model: int = 32
    n_heads: int = 2
    e_layers: int = 1
    d_ff: int = 64
    dropout: float = 0.30
    weight_decay: float = 1e-2
    grad_clip: float = 0.5
    dlinear_individual: bool = True
    kernel_size: int = 25
    dlinear_channel_mixer: str = "target_only"
    max_exogenous_features_model: int = 12
    use_residual_wrapper: bool = False
    residual_alpha: float = 0.1


CANDIDATES: list[Candidate] = [
    Candidate(
        name="dlinear_ret_s10_l5",
        tsm_type="dlinear",
        target_mode="returns",
        seq_len=10,
        label_len=5,
        batch_size=64,
        learning_rate=1e-3,
        max_epochs=40,
        patience=8,
    ),
    Candidate(
        name="dlinear_ret_s20_l10",
        tsm_type="dlinear",
        target_mode="returns",
        seq_len=20,
        label_len=10,
        batch_size=64,
        learning_rate=1e-3,
        max_epochs=40,
        patience=8,
    ),
    Candidate(
        name="dlinear_ret_s40_l15",
        tsm_type="dlinear",
        target_mode="returns",
        seq_len=40,
        label_len=15,
        batch_size=64,
        learning_rate=1e-3,
        max_epochs=40,
        patience=8,
    ),
    Candidate(
        name="dlinear_ret_s60_l15",
        tsm_type="dlinear",
        target_mode="returns",
        seq_len=60,
        label_len=15,
        batch_size=64,
        learning_rate=1e-3,
        max_epochs=40,
        patience=8,
    ),
    Candidate(
        name="dlinear_price_s10_l5",
        tsm_type="dlinear",
        target_mode="price",
        seq_len=10,
        label_len=5,
        batch_size=64,
        learning_rate=1e-3,
        max_epochs=40,
        patience=8,
    ),
    Candidate(
        name="dlinear_price_s20_l10",
        tsm_type="dlinear",
        target_mode="price",
        seq_len=20,
        label_len=10,
        batch_size=64,
        learning_rate=1e-3,
        max_epochs=40,
        patience=8,
    ),
    Candidate(
        name="af_ret_s15_l5_m16",
        tsm_type="autoformer",
        target_mode="returns",
        seq_len=15,
        label_len=5,
        batch_size=32,
        learning_rate=1e-3,
        max_epochs=40,
        patience=10,
        d_model=16,
        d_ff=32,
    ),
    Candidate(
        name="af_ret_s20_l10_m32",
        tsm_type="autoformer",
        target_mode="returns",
        seq_len=20,
        label_len=10,
        batch_size=32,
        learning_rate=1e-3,
        max_epochs=40,
        patience=10,
    ),
    Candidate(
        name="af_ret_s30_l10_m32",
        tsm_type="autoformer",
        target_mode="returns",
        seq_len=30,
        label_len=10,
        batch_size=32,
        learning_rate=1e-3,
        max_epochs=40,
        patience=10,
    ),
    Candidate(
        name="af_ret_s40_l15_m32",
        tsm_type="autoformer",
        target_mode="returns",
        seq_len=40,
        label_len=15,
        batch_size=32,
        learning_rate=1e-3,
        max_epochs=40,
        patience=10,
    ),
    Candidate(
        name="af_ret_s30_l10_m32_resid",
        tsm_type="autoformer",
        target_mode="returns",
        seq_len=30,
        label_len=10,
        batch_size=32,
        learning_rate=1e-3,
        max_epochs=40,
        patience=10,
        use_residual_wrapper=True,
        residual_alpha=0.5,
    ),
    Candidate(
        name="af_price_s15_l5_m16",
        tsm_type="autoformer",
        target_mode="price",
        seq_len=15,
        label_len=5,
        batch_size=32,
        learning_rate=1e-3,
        max_epochs=40,
        patience=10,
        d_model=16,
        d_ff=32,
    ),
    Candidate(
        name="af_price_s20_l10_m32",
        tsm_type="autoformer",
        target_mode="price",
        seq_len=20,
        label_len=10,
        batch_size=32,
        learning_rate=1e-3,
        max_epochs=40,
        patience=10,
    ),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Targeted post-2023 TSM search.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("src/config/post_2023_paper_event90_proxy.yaml"),
        help="Base config for the post-2023 regime.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("Data_auto"),
        help="Data directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory. Defaults under reports/post_2023_tsm_tune/<timestamp>.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )
    return parser.parse_args()


def _build_loaders(
    cfg_raw: dict[str, Any],
    data_dir: Path,
    candidate: Candidate,
) -> dict[str, Any]:
    raw = copy.deepcopy(cfg_raw)
    raw["target"]["mode"] = str(candidate.target_mode)
    raw["time_series"]["seq_len"] = int(candidate.seq_len)
    raw["time_series"]["label_len"] = int(candidate.label_len)
    raw["model"]["tsm_type"] = str(candidate.tsm_type)
    raw["model"]["batch_size"] = int(candidate.batch_size)
    raw["model"]["learning_rate"] = float(candidate.learning_rate)
    raw["model"]["max_epochs"] = int(candidate.max_epochs)
    raw["model"]["early_stopping_patience"] = int(candidate.patience)
    raw["model"]["d_model"] = int(candidate.d_model)
    raw["model"]["n_heads"] = int(candidate.n_heads)
    raw["model"]["e_layers"] = int(candidate.e_layers)
    raw["model"]["d_ff"] = int(candidate.d_ff)
    raw["model"]["dropout"] = float(candidate.dropout)
    raw["model"]["weight_decay"] = float(candidate.weight_decay)
    raw["model"]["grad_clip"] = float(candidate.grad_clip)
    raw["model"]["dlinear_individual"] = bool(candidate.dlinear_individual)
    raw["model"]["kernel_size"] = int(candidate.kernel_size)
    raw["model"]["dlinear_channel_mixer"] = str(candidate.dlinear_channel_mixer)
    raw.setdefault("features", {})["max_exogenous_features_model"] = int(candidate.max_exogenous_features_model)
    raw["model"]["use_residual_wrapper"] = bool(candidate.use_residual_wrapper)
    raw["model"]["residual_alpha"] = float(candidate.residual_alpha)

    panel, schema = build_panel(data_dir=data_dir, config=raw)
    target_mode = str(raw.get("target", {}).get("mode", "price"))
    target_col = "y_return" if target_mode == "returns" else "y"
    feature_cols = select_feature_columns(
        panel,
        target_col=target_col,
        max_exogenous_features=(raw.get("features", {}) or {}).get("max_exogenous_features_model", 10),
        preferred_feature_order=(raw.get("features", {}) or {}).get("preferred_feature_order"),
    )
    wc = WindowConfig(
        seq_len=int(raw["time_series"]["seq_len"]),
        label_len=int(raw["time_series"]["label_len"]),
        pred_len=int(raw["time_series"]["pred_len"]),
        target_col=target_col,
        feature_cols=feature_cols,
    )
    X_enc, X_dec, y, dates, window_meta = make_windows(panel, wc, mode="MS", return_metadata=True)
    splits = split_windows(
        X_enc,
        X_dec,
        y,
        dates,
        train_end=str(raw["split"]["train_end"]),
        val_end=str(raw["split"]["val_end"]),
        window_meta=window_meta,
    )

    target_is_returns = target_mode == "returns"
    price_idx = feature_cols.index("y") if target_is_returns else 0
    if target_is_returns:
        y_train_hist = splits["train"]["X_enc"][:, :, price_idx]
        y_val_hist = splits["val"]["X_enc"][:, :, price_idx]
        y_test_hist = splits["test"]["X_enc"][:, :, price_idx]
        y_train_base = y_train_hist[:, -1]
        y_val_base = y_val_hist[:, -1]
        y_test_base = y_test_hist[:, -1]
        y_train_eval = returns_to_prices(splits["train"]["y"], y_train_base)
        y_val_eval = returns_to_prices(splits["val"]["y"], y_val_base)
        y_test_eval = returns_to_prices(splits["test"]["y"], y_test_base)
    else:
        y_train_base = y_val_base = y_test_base = None
        y_train_eval = splits["train"]["y"]
        y_val_eval = splits["val"]["y"]
        y_test_eval = splits["test"]["y"]

    scaler = StandardScaler()
    scaler.fit(splits["train"]["X_enc"])

    def _make_ds(split_name: str) -> TimeSeriesDataset:
        split = splits[split_name]
        return TimeSeriesDataset(
            scaler.transform(split["X_enc"]),
            scaler.transform(split["X_dec"]),
            (split["y"] - scaler.mean_[0]) / scaler.std_[0],
        )

    batch_size = int(candidate.batch_size)
    bundle = {
        "raw": raw,
        "schema": schema,
        "panel": panel,
        "feature_cols": feature_cols,
        "splits": splits,
        "scaler": scaler,
        "train_loader": DataLoader(_make_ds("train"), batch_size=batch_size, shuffle=True),
        "val_loader": DataLoader(_make_ds("val"), batch_size=batch_size),
        "test_loader": DataLoader(_make_ds("test"), batch_size=batch_size),
        "target_is_returns": target_is_returns,
        "y_train_eval": y_train_eval,
        "y_val_eval": y_val_eval,
        "y_val_base": y_val_base,
        "y_test_eval": y_test_eval,
        "y_test_base": y_test_base,
    }
    return bundle


def _fit_and_score(candidate: Candidate, bundle: dict[str, Any]) -> dict[str, Any]:
    raw = copy.deepcopy(bundle["raw"])
    splits = bundle["splits"]
    raw["model"]["enc_in"] = int(splits["train"]["X_enc"].shape[-1])
    raw["model"]["dec_in"] = int(splits["train"]["X_dec"].shape[-1])
    model = TSMForecaster(raw, device="auto")
    history = model.fit(
        train_loader=bundle["train_loader"],
        val_loader=bundle["val_loader"],
        epochs=int(candidate.max_epochs),
        patience=int(candidate.patience),
        save_path=None,
    )

    def _predict(loader: DataLoader, base: np.ndarray | None) -> np.ndarray:
        pred_scaled, _ = model.predict(loader)
        pred = bundle["scaler"].inverse_transform_target(pred_scaled)
        if bundle["target_is_returns"]:
            assert base is not None
            return returns_to_prices(pred, base)
        return pred

    y_val_pred = _predict(bundle["val_loader"], bundle["y_val_base"] if bundle["target_is_returns"] else None)
    y_test_pred = _predict(bundle["test_loader"], bundle["y_test_base"] if bundle["target_is_returns"] else None)
    val_metrics = compute_path_metrics(bundle["y_val_eval"], y_val_pred)
    test_metrics = compute_path_metrics(bundle["y_test_eval"], y_test_pred)
    test_by_h = compute_metrics_by_horizon(
        bundle["y_test_eval"],
        y_test_pred,
        horizons=raw["time_series"].get("horizons", [1, 5, 20, 30]),
    )
    row = {
        **asdict(candidate),
        "n_features": int(len(bundle["feature_cols"])),
        "best_val_loss": float(model.best_val_loss),
        "epochs_ran": int(len(history.get("val_loss", []))),
        "val_path_mse": float(val_metrics["mse_path"]),
        "test_path_mse": float(test_metrics["mse_path"]),
    }
    for horizon, metric_row in test_by_h.iterrows():
        row[f"h{int(horizon)}_mse"] = float(metric_row["mse"])
    return row


def _baseline_rows(bundle: dict[str, Any]) -> list[dict[str, Any]]:
    y_train = bundle["y_train_eval"]
    y_test = bundle["y_test_eval"]
    y_train_hist = bundle["splits"]["train"]["X_enc"][:, :, bundle["feature_cols"].index("y")]
    y_test_hist = bundle["splits"]["test"]["X_enc"][:, :, bundle["feature_cols"].index("y")]
    ridge = LinearBaseline(model_type="ridge")
    ridge.fit(y_train_hist, y_train)
    lasso = LinearBaseline(model_type="lasso")
    lasso.fit(y_train_hist, y_train)
    preds = {
        "naive_persistence": NaivePersistence().predict(y_test_hist),
        "seasonal_naive": SeasonalNaive().predict(y_test_hist),
        "linear_ridge": ridge.predict(y_test_hist),
        "linear_lasso": lasso.predict(y_test_hist),
    }
    rows = []
    for name, pred in preds.items():
        row = {
            "name": name,
            "test_path_mse": float(compute_path_metrics(y_test, pred)["mse_path"]),
        }
        rows.append(row)
    return rows


def main() -> int:
    args = parse_args()
    set_seed(int(args.seed))
    config = load_config(args.config)
    output_dir = args.output_dir
    if output_dir is None:
        output_dir = ROOT / "reports" / "post_2023_tsm_tune" / datetime.now().strftime("%Y%m%d_%H%M%S")
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    baseline_bundle = _build_loaders(config.raw, args.data_dir, CANDIDATES[0])
    with (output_dir / "metadata.json").open("w") as fh:
        json.dump(
            {
                "config": str(args.config),
                "data_dir": str(args.data_dir),
                "seed": int(args.seed),
                "n_candidates": len(CANDIDATES),
                "panel_rows": int(len(baseline_bundle["panel"])),
                "split_sizes": {
                    split: int(len(baseline_bundle["splits"][split]["dates"]))
                    for split in ("train", "val", "test")
                },
            },
            fh,
            indent=2,
        )
    pd.DataFrame(_baseline_rows(baseline_bundle)).to_csv(output_dir / "baseline_reference.csv", index=False)

    rows: list[dict[str, Any]] = []
    for candidate in CANDIDATES:
        set_seed(int(args.seed))
        bundle = _build_loaders(config.raw, args.data_dir, candidate)
        row = _fit_and_score(candidate, bundle)
        rows.append(row)
        pd.DataFrame(rows).sort_values(["val_path_mse", "test_path_mse"]).to_csv(
            output_dir / "tsm_search_results.partial.csv",
            index=False,
        )

    out = pd.DataFrame(rows).sort_values(["val_path_mse", "test_path_mse"]).reset_index(drop=True)
    out.to_csv(output_dir / "tsm_search_results.csv", index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
