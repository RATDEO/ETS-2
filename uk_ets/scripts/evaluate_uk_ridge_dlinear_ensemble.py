#!/usr/bin/env python3
"""Evaluate ridge + DLinear ensembles for the UK ETS pivot experiments."""

from __future__ import annotations

import argparse
import copy
import json
from datetime import datetime
from pathlib import Path
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import tune_post_2023_tsm as base

from src.config import load_config
from src.eval.metrics import compute_metrics_by_horizon, compute_path_metrics
from src.models.baselines import LinearBaseline
from src.models.tsm import TSMForecaster
from src.run_experiment import blend_forecasts, evaluate_blend_grid, save_blend_grid_artifacts, returns_to_prices
from src.utils import set_seed


def _parse_csv_list(text: str, cast):
    return [cast(part.strip()) for part in str(text or "").split(",") if part.strip()]


def _as_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _latest_search_results() -> Path:
    matches = sorted(
        (ROOT / "reports" / "uk_ets_dlinear_tune").glob("*/tsm_search_results.csv"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not matches:
        raise FileNotFoundError("No UK DLinear search results found under reports/uk_ets_dlinear_tune/")
    return matches[0]


def _candidate_from_row(row: pd.Series) -> base.Candidate:
    return base.Candidate(
        name=str(row["name"]),
        tsm_type=str(row["tsm_type"]),
        target_mode=str(row["target_mode"]),
        seq_len=int(row["seq_len"]),
        label_len=int(row["label_len"]),
        batch_size=int(row["batch_size"]),
        learning_rate=float(row["learning_rate"]),
        max_epochs=int(row["max_epochs"]),
        patience=int(row["patience"]),
        d_model=int(row["d_model"]),
        n_heads=int(row["n_heads"]),
        e_layers=int(row["e_layers"]),
        d_ff=int(row["d_ff"]),
        dropout=float(row["dropout"]),
        weight_decay=float(row["weight_decay"]),
        grad_clip=float(row["grad_clip"]),
        dlinear_individual=_as_bool(row["dlinear_individual"]),
        kernel_size=int(row["kernel_size"]),
        dlinear_channel_mixer=str(row.get("dlinear_channel_mixer", "target_only")),
        max_exogenous_features_model=int(row.get("max_exogenous_features_model", 12)),
        use_residual_wrapper=_as_bool(row["use_residual_wrapper"]),
        residual_alpha=float(row["residual_alpha"]),
    )


def _select_candidate(search_results: Path, candidate_name: str | None) -> tuple[base.Candidate, pd.Series]:
    results = pd.read_csv(search_results)
    if candidate_name:
        match = results.loc[results["name"] == str(candidate_name)]
        if match.empty:
            raise ValueError(f"Candidate '{candidate_name}' not found in {search_results}")
        row = match.sort_values(["val_path_mse", "test_path_mse"]).iloc[0]
    else:
        row = results.sort_values(["val_path_mse", "test_path_mse"]).iloc[0]
    return _candidate_from_row(row), row


def _fit_tsm_predictions(bundle: dict, candidate: base.Candidate) -> tuple[dict, pd.DataFrame]:
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

    def _predict(loader, base_prices):
        pred_scaled, _ = model.predict(loader)
        pred = bundle["scaler"].inverse_transform_target(pred_scaled)
        if bundle["target_is_returns"]:
            return returns_to_prices(pred, base_prices)
        return pred

    preds = {
        "val": _predict(bundle["val_loader"], bundle["y_val_base"]),
        "test": _predict(bundle["test_loader"], bundle["y_test_base"]),
    }
    history_df = pd.DataFrame(
        {
            "epoch": list(range(1, len(history.get("train_loss", [])) + 1)),
            "train_loss": history.get("train_loss", []),
            "val_loss": history.get("val_loss", []),
        }
    )
    return preds, history_df


def _fit_ridge_predictions(bundle: dict) -> dict:
    price_idx = int(bundle["feature_cols"].index("y"))
    y_train_hist = bundle["splits"]["train"]["X_enc"][:, :, price_idx]
    y_val_hist = bundle["splits"]["val"]["X_enc"][:, :, price_idx]
    y_test_hist = bundle["splits"]["test"]["X_enc"][:, :, price_idx]

    ridge = LinearBaseline(model_type="ridge")
    ridge.fit(y_train_hist, bundle["y_train_eval"])
    return {
        "val": ridge.predict(y_val_hist),
        "test": ridge.predict(y_test_hist),
    }


def _metrics_row(model_name: str, subset: str, y_true, y_pred, horizons: list[int]) -> dict:
    row = {
        "model": str(model_name),
        "subset": str(subset),
        **compute_path_metrics(y_true, y_pred),
    }
    by_h = compute_metrics_by_horizon(y_true, y_pred, horizons=horizons)
    for horizon, metric_row in by_h.iterrows():
        row[f"h{int(horizon)}_mse"] = float(metric_row["mse"])
    return row


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate ridge + DLinear ensembles on UK ETS.")
    parser.add_argument("--config", type=Path, default=Path("uk_ets/config/uk_ets_tuned.yaml"))
    parser.add_argument("--data-dir", type=Path, default=Path("uk_ets/Data_auto_uk"))
    parser.add_argument("--search-results", type=Path, default=None)
    parser.add_argument("--candidate-name", type=str, default=None)
    parser.add_argument("--weights", type=str, default="0,0.25,0.5,0.75,1.0")
    parser.add_argument("--schedules", type=str, default="uniform,ramp")
    parser.add_argument("--key-horizons", type=str, default="1,5,20,30")
    parser.add_argument("--min-weight", type=float, default=0.0)
    parser.add_argument("--power", type=float, default=1.0)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    set_seed(int(args.seed))

    search_results = Path(args.search_results) if args.search_results else _latest_search_results()
    config = load_config(args.config)
    candidate, candidate_row = _select_candidate(search_results, args.candidate_name)

    output_dir = args.output_dir
    if output_dir is None:
        output_dir = ROOT / "reports" / "uk_ets_ridge_dlinear_ensemble" / datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir.mkdir(parents=True, exist_ok=True)

    bundle = base._build_loaders(config.raw, args.data_dir, candidate)
    ridge_preds = _fit_ridge_predictions(bundle)
    set_seed(int(args.seed))
    tsm_preds, history_df = _fit_tsm_predictions(bundle, candidate)
    history_df.to_csv(output_dir / "tsm_training_history.csv", index=False)

    weights = _parse_csv_list(args.weights, float)
    schedules = _parse_csv_list(args.schedules, str)
    key_horizons = _parse_csv_list(args.key_horizons, int)
    pred_len = int(bundle["y_test_eval"].shape[1])

    with (output_dir / "metadata.json").open("w") as fh:
        json.dump(
            {
                "config": str(args.config),
                "data_dir": str(args.data_dir),
                "search_results": str(search_results),
                "candidate": candidate.name,
                "candidate_row": candidate_row.to_dict(),
                "weights": weights,
                "schedules": schedules,
                "key_horizons": key_horizons,
                "min_weight": float(args.min_weight),
                "power": float(args.power),
                "seed": int(args.seed),
            },
            fh,
            indent=2,
            default=str,
        )

    comparison_rows = [
        _metrics_row("linear_ridge", "val", bundle["y_val_eval"], ridge_preds["val"], key_horizons),
        _metrics_row("tsm", "val", bundle["y_val_eval"], tsm_preds["val"], key_horizons),
        _metrics_row("linear_ridge", "test", bundle["y_test_eval"], ridge_preds["test"], key_horizons),
        _metrics_row("tsm", "test", bundle["y_test_eval"], tsm_preds["test"], key_horizons),
    ]
    selection_rows = []

    for schedule in schedules:
        val_out = output_dir / schedule / "val"
        test_out = output_dir / schedule / "test"

        val_summary, val_mse_by_h, val_best_by_h = evaluate_blend_grid(
            bundle["y_val_eval"],
            ridge_preds["val"],
            tsm_preds["val"],
            weights=weights,
            schedule=schedule,
            key_horizons=key_horizons,
            min_weight=float(args.min_weight),
            power=float(args.power),
        )
        save_blend_grid_artifacts(val_summary, val_mse_by_h, val_best_by_h, val_out)

        test_summary, test_mse_by_h, test_best_by_h = evaluate_blend_grid(
            bundle["y_test_eval"],
            ridge_preds["test"],
            tsm_preds["test"],
            weights=weights,
            schedule=schedule,
            key_horizons=key_horizons,
            min_weight=float(args.min_weight),
            power=float(args.power),
        )
        save_blend_grid_artifacts(test_summary, test_mse_by_h, test_best_by_h, test_out)

        val_best = val_summary.sort_values(["mse_path", "w"]).iloc[0]
        selected_w = float(val_best["w"])
        blended_test = blend_forecasts(
            ridge_preds["test"],
            tsm_preds["test"],
            strength=selected_w,
            schedule=schedule,
            pred_len=pred_len,
            min_weight=float(args.min_weight),
            power=float(args.power),
        )
        blended_val = blend_forecasts(
            ridge_preds["val"],
            tsm_preds["val"],
            strength=selected_w,
            schedule=schedule,
            pred_len=pred_len,
            min_weight=float(args.min_weight),
            power=float(args.power),
        )
        comparison_rows.append(_metrics_row(f"ridge_tsm_blend_{schedule}", "val", bundle["y_val_eval"], blended_val, key_horizons))
        comparison_rows.append(_metrics_row(f"ridge_tsm_blend_{schedule}", "test", bundle["y_test_eval"], blended_test, key_horizons))

        test_selected = test_summary.loc[test_summary["w"] == selected_w].iloc[0].to_dict()
        selection = {
            "schedule": schedule,
            "selected_w": selected_w,
            "val_mse_path": float(val_best["mse_path"]),
            "test_mse_path": float(test_selected["mse_path"]),
            "ridge_test_mse_path": float(compute_path_metrics(bundle["y_test_eval"], ridge_preds["test"])["mse_path"]),
            "tsm_test_mse_path": float(compute_path_metrics(bundle["y_test_eval"], tsm_preds["test"])["mse_path"]),
        }
        selection_rows.append(selection)
        with (output_dir / schedule / "selection.json").open("w") as fh:
            json.dump(selection, fh, indent=2)

    pd.DataFrame(comparison_rows).to_csv(output_dir / "comparison_metrics.csv", index=False)
    pd.DataFrame(selection_rows).sort_values("test_mse_path").to_csv(output_dir / "selected_blends.csv", index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
