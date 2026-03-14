#!/usr/bin/env python3
"""Structured news event panel + supervised residual correction over base TSM."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from data.windows import StandardScaler, TimeSeriesDataset, WindowConfig, make_windows, split_windows
from eval.residual_event_model import (
    ResidualCandidate,
    block_columns,
    build_origin_feature_frame,
    extract_anchor_residual_targets,
    make_candidate_predictions,
    summarize_prediction,
)
from models.tsm import TSMForecaster
from news.event_panel import build_event_panel
from run_experiment import returns_to_prices

try:
    import torch  # noqa: F401
    from torch.utils.data import DataLoader
except ImportError as exc:  # pragma: no cover
    raise SystemExit("PyTorch is required for residual_event_model_v1") from exc


VAL_SELECTION_CANDIDATES: list[ResidualCandidate] = [
    ResidualCandidate(name="identity", estimator="identity", feature_blocks=("base",)),
    ResidualCandidate(
        name="ridge_base_pct_a1",
        estimator="ridge",
        feature_blocks=("base",),
        target_kind="pct",
        alpha=1.0,
        max_adjustment_pct=8.0,
    ),
    ResidualCandidate(
        name="ridge_base_pct_a10",
        estimator="ridge",
        feature_blocks=("base",),
        target_kind="pct",
        alpha=10.0,
        max_adjustment_pct=8.0,
    ),
    ResidualCandidate(
        name="ridge_market_pct_a10",
        estimator="ridge",
        feature_blocks=("base", "market"),
        target_kind="pct",
        alpha=10.0,
        max_adjustment_pct=8.0,
    ),
    ResidualCandidate(
        name="ridge_event_core_pct_a10",
        estimator="ridge",
        feature_blocks=("base", "event_core"),
        target_kind="pct",
        alpha=10.0,
        max_adjustment_pct=8.0,
    ),
    ResidualCandidate(
        name="ridge_market_event_core_pct_a10",
        estimator="ridge",
        feature_blocks=("base", "market", "event_core"),
        target_kind="pct",
        alpha=10.0,
        max_adjustment_pct=8.0,
    ),
    ResidualCandidate(
        name="ridge_market_event_core_pct_a100",
        estimator="ridge",
        feature_blocks=("base", "market", "event_core"),
        target_kind="pct",
        alpha=100.0,
        max_adjustment_pct=8.0,
    ),
    ResidualCandidate(
        name="hgb_market_pct",
        estimator="hgb",
        feature_blocks=("base", "market"),
        target_kind="pct",
        learning_rate=0.05,
        max_depth=3,
        max_iter=250,
        min_samples_leaf=12,
        l2_regularization=1.0,
        max_adjustment_pct=8.0,
    ),
    ResidualCandidate(
        name="hgb_market_event_core_pct",
        estimator="hgb",
        feature_blocks=("base", "market", "event_core"),
        target_kind="pct",
        learning_rate=0.05,
        max_depth=3,
        max_iter=250,
        min_samples_leaf=12,
        l2_regularization=1.0,
        max_adjustment_pct=8.0,
    ),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run structured event-panel residual correction over base TSM.")
    parser.add_argument(
        "--base-run",
        type=Path,
        default=Path("runs/20260228_145058_40d91f"),
        help="Clean base run containing the frozen TSM checkpoint and test predictions.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory. Defaults under reports/event_panel_residual_v1/<timestamp>.",
    )
    parser.add_argument(
        "--events-path",
        "--headlines-path",
        dest="events_path",
        type=Path,
        default=Path("data/news/official/official_event_records_v1.csv"),
        help="Event source file. Defaults to the automated official event store; headline-label CSVs still work.",
    )
    parser.add_argument(
        "--proxy-daily-path",
        type=Path,
        default=None,
        help="Auxiliary daily proxy sentiment file.",
    )
    parser.add_argument(
        "--novelty-daily-path",
        type=Path,
        default=None,
        help="Auxiliary daily novelty/topical sentiment file.",
    )
    parser.add_argument(
        "--ds-daily-path",
        type=Path,
        default=None,
        help="Auxiliary daily Dawid-Skene posterior sentiment file.",
    )
    parser.add_argument(
        "--event-version",
        choices=["v1", "v2", "official_v1"],
        default="official_v1",
        help="Structured event feature version.",
    )
    parser.add_argument(
        "--fit-fraction",
        type=float,
        default=0.7,
        help="Chronological fraction of validation windows used to fit candidates before late-val selection.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device for checkpoint inference. Default cpu for deterministic replay.",
    )
    parser.add_argument(
        "--hprice-summary",
        type=Path,
        default=Path("reports/paper_replication_v2/20260303_012030_hprice_4b_full_uncapped/selected_test_summary.json"),
        help="Summary JSON for the current 4B HPRICE benchmark.",
    )
    return parser.parse_args()


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _predict_returns(
    tsm: TSMForecaster,
    scaler: StandardScaler,
    X_enc: np.ndarray,
    X_dec: np.ndarray,
    y: np.ndarray,
    batch_size: int = 32,
) -> np.ndarray:
    dataset = TimeSeriesDataset(
        scaler.transform(X_enc),
        scaler.transform(X_dec),
        (y - scaler.mean_[0]) / scaler.std_[0],
    )
    loader = DataLoader(dataset, batch_size=batch_size)
    pred_scaled, _ = tsm.predict(loader)
    return scaler.inverse_transform_target(pred_scaled)


def _load_tsm_test_table(path: Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    table = pd.read_parquet(path)
    y_true = np.column_stack([table[f"y_true_t_plus_{h}"].to_numpy(dtype=float) for h in range(1, 31)])
    yhat = np.column_stack([table[f"yhat_t_plus_{h}"].to_numpy(dtype=float) for h in range(1, 31)])
    dates = [str(pd.Timestamp(x).date()) for x in pd.to_datetime(table["date"]).tolist()]
    return y_true, yhat, dates


def _feature_columns_from_panel(panel: pd.DataFrame, target_col: str, limit: int = 10) -> list[str]:
    feature_candidates = [c for c in panel.columns if c not in ["date", target_col]]
    if "y" in feature_candidates:
        feature_candidates.remove("y")
        feature_candidates = ["y"] + feature_candidates
    feature_cols = [target_col] + feature_candidates[:limit]
    if "y" not in feature_cols:
        raise ValueError("Expected price feature 'y' in feature columns.")
    return feature_cols


def _candidate_feature_matrix(
    feature_frame: pd.DataFrame,
    candidate: ResidualCandidate,
    blocks: dict[str, list[str]],
) -> np.ndarray:
    cols: list[str] = []
    for block in candidate.feature_blocks:
        cols.extend(blocks[block])
    if not cols:
        return np.zeros((len(feature_frame), 0), dtype=float)
    return feature_frame[cols].to_numpy(dtype=float)


def _load_base_paper_summary(base_run: Path) -> dict[str, float | str]:
    npz = np.load(base_run / "predictions" / "TSM+LLM-COT-SENT-RF_pred_test_subset.npz", allow_pickle=True)
    return summarize_prediction(
        "old_clean_paper_fullpath",
        np.asarray(npz["y_true"], dtype=float),
        np.asarray(npz["yhat"], dtype=float),
    )


def _load_hprice_summary(path: Path) -> dict[str, float | str] | None:
    if not path.exists():
        return None
    payload = json.loads(path.read_text())
    return {
        "variant": "4b_hprice",
        "path_mse": float(payload["mse_path"]),
        "h1_mse": float(payload["h1_mse"]),
        "h5_mse": float(payload["h5_mse"]),
        "h20_mse": float(payload["h20_mse"]),
        "h30_mse": float(payload["h30_mse"]),
    }


def main() -> int:
    args = parse_args()
    base_run = args.base_run if args.base_run.is_absolute() else (ROOT / args.base_run)
    if not base_run.exists():
        raise FileNotFoundError(f"Base run not found: {base_run}")

    if args.output_dir is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = ROOT / "reports" / "event_panel_residual_v1" / stamp
    else:
        output_dir = args.output_dir if args.output_dir.is_absolute() else (ROOT / args.output_dir)
    _ensure_dir(output_dir)

    cfg = _load_yaml(base_run / "config_resolved.yaml")
    panel = pd.read_parquet(base_run / "data" / "panel.parquet")

    target_mode = str((cfg.get("target", {}) or {}).get("mode", "price"))
    if target_mode != "returns":
        raise ValueError("This experiment expects target.mode=returns.")

    target_col = "y_return"
    feature_cols = _feature_columns_from_panel(panel, target_col=target_col)
    price_idx = feature_cols.index("y")
    time_cfg = cfg.get("time_series", {}) or {}
    window_config = WindowConfig(
        seq_len=int(time_cfg.get("seq_len", 60)),
        label_len=int(time_cfg.get("label_len", 15)),
        pred_len=int(time_cfg.get("pred_len", 30)),
        target_col=target_col,
        feature_cols=feature_cols,
    )
    X_enc, X_dec, y, dates, meta = make_windows(panel, window_config, mode="MS", return_metadata=True)
    split_cfg = cfg.get("split", {}) or {}
    splits = split_windows(
        X_enc,
        X_dec,
        y,
        dates,
        train_end=split_cfg.get("train_end", "2023-06-30"),
        val_end=split_cfg.get("val_end", "2024-06-30"),
        window_meta=meta,
    )

    X_train_enc = splits["train"]["X_enc"]
    X_train_dec = splits["train"]["X_dec"]
    y_train = splits["train"]["y"]
    X_val_enc = splits["val"]["X_enc"]
    X_val_dec = splits["val"]["X_dec"]
    y_val = splits["val"]["y"]
    X_test_enc = splits["test"]["X_enc"]
    X_test_dec = splits["test"]["X_dec"]
    y_test = splits["test"]["y"]

    val_origin_dates = pd.to_datetime(splits["val"]["last_input_dates"])
    test_origin_dates = pd.to_datetime(splits["test"]["last_input_dates"])

    y_train_base = X_train_enc[:, -1, price_idx]
    y_val_base = X_val_enc[:, -1, price_idx]
    y_test_base = X_test_enc[:, -1, price_idx]
    y_val_true = returns_to_prices(y_val, y_val_base)
    y_test_true = returns_to_prices(y_test, y_test_base)

    scaler = StandardScaler().fit(X_train_enc)
    tsm_cfg = {
        "time_series": dict(cfg.get("time_series", {}) or {}),
        "model": dict(cfg.get("model", {}) or {}),
    }
    tsm_cfg["model"]["enc_in"] = int(X_train_enc.shape[-1])
    tsm_cfg["model"]["dec_in"] = int(X_train_dec.shape[-1])
    tsm = TSMForecaster(tsm_cfg, device=args.device)
    tsm.load(base_run / "models" / "tsm_checkpoint.pt")

    print("[TSM] Recomputing validation forecasts from saved checkpoint", flush=True)
    val_pred_returns = _predict_returns(tsm, scaler, X_val_enc, X_val_dec, y_val)
    val_pred = returns_to_prices(val_pred_returns, y_val_base)

    print("[TSM] Loading persisted test forecasts from base run", flush=True)
    y_test_true_saved, test_pred_saved, test_dates_saved = _load_tsm_test_table(base_run / "predictions" / "tsm_pred_test.parquet")
    split_test_dates = [str(pd.Timestamp(x).date()) for x in pd.to_datetime(splits["test"]["dates"]).tolist()]
    if split_test_dates != test_dates_saved:
        raise ValueError("Persisted test TSM table does not align with reconstructed split dates.")
    if float(np.mean((y_test_true - y_test_true_saved) ** 2)) > 1e-8:
        raise ValueError("Persisted test y_true does not match reconstructed test truth.")
    test_pred = test_pred_saved

    event_panel = build_event_panel(
        headlines_path=args.events_path,
        calendar=pd.to_datetime(panel["date"]),
        proxy_daily_path=args.proxy_daily_path,
        novelty_daily_path=args.novelty_daily_path,
        ds_daily_path=args.ds_daily_path,
        taxonomy_version=args.event_version,
    )

    val_feature_frame = build_origin_feature_frame(panel, event_panel, val_origin_dates, val_pred)
    test_feature_frame = build_origin_feature_frame(panel, event_panel, test_origin_dates, test_pred)
    blocks = block_columns(val_feature_frame)

    split_at = int(round(len(val_feature_frame) * float(args.fit_fraction)))
    split_at = min(max(split_at, 32), len(val_feature_frame) - 32)
    fit_slice = slice(0, split_at)
    select_slice = slice(split_at, len(val_feature_frame))
    print(f"[Residual] Validation fit/select split: {split_at} / {len(val_feature_frame) - split_at}", flush=True)

    y_val_anchor_by_kind = {
        "absolute": extract_anchor_residual_targets(y_val_true, val_pred, target_kind="absolute"),
        "pct": extract_anchor_residual_targets(y_val_true, val_pred, target_kind="pct"),
    }
    candidate_rows: list[dict[str, float | str | int]] = []
    candidate_forecasts: dict[str, np.ndarray] = {}

    for candidate in VAL_SELECTION_CANDIDATES:
        X_val_all = _candidate_feature_matrix(val_feature_frame, candidate, blocks)
        X_fit = X_val_all[fit_slice]
        X_select = X_val_all[select_slice]
        y_fit_anchor = y_val_anchor_by_kind[candidate.target_kind][fit_slice]
        y_select_true = y_val_true[select_slice]
        base_select_pred = val_pred[select_slice]
        y_select_pred = make_candidate_predictions(
            candidate,
            X_fit=X_fit,
            y_fit_anchor=y_fit_anchor,
            X_apply=X_select,
            base_apply_pred=base_select_pred,
        )
        candidate_forecasts[candidate.name] = y_select_pred
        row = summarize_prediction(candidate.name, y_select_true, y_select_pred)
        row["feature_blocks"] = "+".join(candidate.feature_blocks)
        row["target_kind"] = candidate.target_kind
        row["n_features"] = int(X_val_all.shape[1])
        candidate_rows.append(row)

    val_candidate_df = pd.DataFrame(candidate_rows).sort_values("path_mse", kind="stable").reset_index(drop=True)
    val_candidate_df.to_csv(output_dir / "val_candidate_results.csv", index=False)

    best_name = str(val_candidate_df.iloc[0]["variant"])
    selected = next(candidate for candidate in VAL_SELECTION_CANDIDATES if candidate.name == best_name)
    selected_payload = {
        "candidate": selected.name,
        "estimator": selected.estimator,
        "feature_blocks": list(selected.feature_blocks),
        "target_kind": selected.target_kind,
        "fit_fraction": float(args.fit_fraction),
        "fit_samples": int(split_at),
        "selection_samples": int(len(val_feature_frame) - split_at),
        "val_selection_path_mse": float(val_candidate_df.iloc[0]["path_mse"]),
    }
    (output_dir / "selected_candidate.json").write_text(json.dumps(selected_payload, indent=2))

    X_val_full = _candidate_feature_matrix(val_feature_frame, selected, blocks)
    X_test = _candidate_feature_matrix(test_feature_frame, selected, blocks)
    y_test_pred = make_candidate_predictions(
        selected,
        X_fit=X_val_full,
        y_fit_anchor=y_val_anchor_by_kind[selected.target_kind],
        X_apply=X_test,
        base_apply_pred=test_pred,
    )

    test_summary = summarize_prediction("residual_event_v1", y_test_true_saved, y_test_pred)
    (output_dir / "test_summary.json").write_text(json.dumps(test_summary, indent=2))

    comparison_rows = [
        summarize_prediction("tsm", y_test_true_saved, test_pred),
        _load_base_paper_summary(base_run),
        test_summary,
    ]
    hprice_summary = _load_hprice_summary(args.hprice_summary if args.hprice_summary.is_absolute() else (ROOT / args.hprice_summary))
    if hprice_summary is not None:
        comparison_rows.append(hprice_summary)
    comparison_df = pd.DataFrame(comparison_rows).sort_values("path_mse", kind="stable").reset_index(drop=True)
    comparison_df.to_csv(output_dir / "test_comparison.csv", index=False)

    event_cols = blocks["event"]
    feature_manifest = {
        "events_path": str(args.events_path),
        "event_version": args.event_version,
        "base_feature_columns": blocks["base"],
        "market_feature_columns": blocks["market"],
        "event_feature_columns": event_cols,
        "event_feature_count": len(event_cols),
        "event_core_feature_columns": blocks.get("event_core", []),
        "event_core_feature_count": len(blocks.get("event_core", [])),
        "candidate_names": [candidate.name for candidate in VAL_SELECTION_CANDIDATES],
    }
    (output_dir / "feature_manifest.json").write_text(json.dumps(feature_manifest, indent=2))

    print(json.dumps({"output_dir": str(output_dir), "selected_candidate": selected.name, "test_path_mse": test_summary["path_mse"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
