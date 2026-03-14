#!/usr/bin/env python3
"""
Validation-only tuning for the leakage-free paper-style CoT-SENT-RF run.

Workflow:
1. Load an existing clean run's config, panel, checkpoint, and saved raw test artifact.
2. Recompute TSM forecasts from the saved checkpoint for train/val/test.
3. Run TSM+LLM-COT-SENT-RF on the validation split only using a strict train-only
   teaching-example pool.
4. Select the best outer blend variant on validation.
5. Apply that validation-selected outer variant to the saved raw test artifact from
   the same run and report the held-out result.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from data.windows import StandardScaler, TimeSeriesDataset, WindowConfig, make_windows, split_windows
from eval.metrics import compute_metrics_by_horizon, compute_path_metrics
from llm.refine import LLMRefiner
from models.tsm import TSMForecaster
from run_experiment import (
    blend_forecasts,
    build_history_dates,
    build_sentiment_histories,
    evaluate_blend_grid,
    load_daily_sentiment,
    returns_to_prices,
)

try:
    from torch.utils.data import DataLoader
except ImportError as exc:  # pragma: no cover
    raise SystemExit("PyTorch is required for validation tuning.") from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tune the paper CoT-SENT-RF outer blend on validation only using a saved clean run."
    )
    parser.add_argument(
        "--base-run",
        type=Path,
        default=Path("runs/20260228_145058_40d91f"),
        help="Existing clean paper-style run directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Where to write tuning artifacts. Defaults under reports/paper_sent_validation_tune/.",
    )
    parser.add_argument(
        "--llm-api-key",
        type=str,
        default=None,
        help="LLM API key. Falls back to OPENAI_API_KEY then config value then deo.",
    )
    parser.add_argument(
        "--llm-base-url",
        type=str,
        default=None,
        help="LLM base URL. Falls back to OPENAI_BASE_URL then config value.",
    )
    return parser.parse_args()


def _history_features(history: np.ndarray, feature_window: int) -> np.ndarray:
    window = history[-feature_window:] if len(history) >= feature_window else history
    mean = float(np.mean(window)) if len(window) else 0.0
    std = float(np.std(window)) if len(window) else 0.0
    if len(window) > 1:
        slope = float(np.polyfit(np.arange(len(window)), window, 1)[0])
    else:
        slope = 0.0
    return np.array([mean, std, slope], dtype=float)


def _metric_row(name: str, y_true: np.ndarray, yhat: np.ndarray) -> dict[str, float | str]:
    metrics = compute_metrics_by_horizon(y_true, yhat, [1, 5, 20, 30]).reset_index()
    row: dict[str, float | str] = {"variant": name, "path_mse": float(compute_path_metrics(y_true, yhat)["mse_path"])}
    for _, metric_row in metrics.iterrows():
        row[f"h{int(metric_row['horizon'])}_mse"] = float(metric_row["mse"])
    return row


def _predict_returns(tsm: TSMForecaster, scaler: StandardScaler, x_enc: np.ndarray, x_dec: np.ndarray, y: np.ndarray) -> np.ndarray:
    dataset = TimeSeriesDataset(
        scaler.transform(x_enc),
        scaler.transform(x_dec),
        (y - scaler.mean_[0]) / scaler.std_[0],
    )
    loader = DataLoader(dataset, batch_size=32)
    pred_scaled, _ = tsm.predict(loader)
    return scaler.inverse_transform_target(pred_scaled)


def main() -> int:
    args = parse_args()
    base_run = args.base_run if args.base_run.is_absolute() else (ROOT / args.base_run)
    if not base_run.exists():
        raise FileNotFoundError(f"Run not found: {base_run}")

    if args.output_dir is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = ROOT / "reports" / "paper_sent_validation_tune" / stamp
    else:
        output_dir = args.output_dir if args.output_dir.is_absolute() else (ROOT / args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with (base_run / "config_resolved.yaml").open("r") as handle:
        cfg = yaml.safe_load(handle)

    llm_cfg = dict(cfg.get("llm", {}) or {})
    llm_cfg["api_key"] = args.llm_api_key or llm_cfg.get("api_key") or "deo"
    llm_cfg["base_url"] = args.llm_base_url or llm_cfg.get("base_url")

    target_mode = str((cfg.get("target", {}) or {}).get("mode", "price"))
    if target_mode != "returns":
        raise ValueError("This tuning script currently expects target.mode=returns.")

    target_col = "y_return"
    pred_len = int((cfg.get("time_series", {}) or {}).get("pred_len", 30))
    ts_cfg = cfg.get("time_series", {}) or {}
    seq_len = int(ts_cfg.get("seq_len", 60))
    label_len = int(ts_cfg.get("label_len", 15))

    panel = pd.read_parquet(base_run / "data" / "panel.parquet")
    feature_candidates = [c for c in panel.columns if c not in ["date", target_col]]
    if "y" in feature_candidates:
        feature_candidates.remove("y")
        feature_candidates = ["y"] + feature_candidates
    feature_cols = [target_col] + feature_candidates[:10]
    if "y" not in feature_cols:
        raise ValueError("Expected price feature 'y' in feature columns for returns mode.")
    price_idx = feature_cols.index("y")

    window_config = WindowConfig(
        seq_len=seq_len,
        label_len=label_len,
        pred_len=pred_len,
        target_col=target_col,
        feature_cols=feature_cols,
    )
    X_enc, X_dec, y, dates, window_meta = make_windows(
        panel,
        window_config,
        mode="MS",
        return_metadata=True,
    )
    split_cfg = cfg.get("split", {}) or {}
    splits = split_windows(
        X_enc,
        X_dec,
        y,
        dates,
        train_end=split_cfg.get("train_end", "2023-06-30"),
        val_end=split_cfg.get("val_end", "2024-06-30"),
        window_meta=window_meta,
    )

    x_train_enc, x_train_dec, y_train, dates_train = (
        splits["train"]["X_enc"],
        splits["train"]["X_dec"],
        splits["train"]["y"],
        splits["train"]["dates"],
    )
    x_val_enc, x_val_dec, y_val, dates_val = (
        splits["val"]["X_enc"],
        splits["val"]["X_dec"],
        splits["val"]["y"],
        splits["val"]["dates"],
    )
    x_test_enc, x_test_dec, y_test, dates_test = (
        splits["test"]["X_enc"],
        splits["test"]["X_dec"],
        splits["test"]["y"],
        splits["test"]["dates"],
    )

    y_train_hist = x_train_enc[:, :, price_idx]
    y_val_hist = x_val_enc[:, :, price_idx]
    y_test_hist = x_test_enc[:, :, price_idx]
    y_train_base = y_train_hist[:, -1]
    y_val_base = y_val_hist[:, -1]
    y_test_base = y_test_hist[:, -1]
    y_train_true = returns_to_prices(y_train, y_train_base)
    y_val_true = returns_to_prices(y_val, y_val_base)
    y_test_true = returns_to_prices(y_test, y_test_base)

    scaler = StandardScaler().fit(x_train_enc)
    tsm_config = {
        "time_series": dict(cfg.get("time_series", {}) or {}),
        "model": dict(cfg.get("model", {}) or {}),
    }
    tsm_config["model"]["enc_in"] = int(x_train_enc.shape[-1])
    tsm_config["model"]["dec_in"] = int(x_train_dec.shape[-1])
    tsm = TSMForecaster(tsm_config, device="auto")
    tsm.load(base_run / "models" / "tsm_checkpoint.pt")

    print("[TSM] Recomputing train/val/test forecasts from saved checkpoint", flush=True)
    tsm_pred_train = returns_to_prices(_predict_returns(tsm, scaler, x_train_enc, x_train_dec, y_train), y_train_base)
    tsm_pred_val = returns_to_prices(_predict_returns(tsm, scaler, x_val_enc, x_val_dec, y_val), y_val_base)
    tsm_pred_test = returns_to_prices(_predict_returns(tsm, scaler, x_test_enc, x_test_dec, y_test), y_test_base)

    raw_test_npz = base_run / "predictions" / "TSM+LLM-COT-SENT-RF_pred_test_subset.npz"
    if not raw_test_npz.exists():
        raise FileNotFoundError(f"Missing raw test artifact: {raw_test_npz}")
    z_test = np.load(raw_test_npz, allow_pickle=True)
    test_y_true_saved = np.asarray(z_test["y_true"], dtype=float)
    test_tsm_saved = np.asarray(z_test["tsm_pred"], dtype=float)
    test_yhat_raw_saved = np.asarray(z_test["yhat"], dtype=float)
    saved_dates = [str(x) for x in np.asarray(z_test["dates"]).tolist()]
    rebuilt_dates = [str(x) for x in dates_test.tolist()]
    if saved_dates != rebuilt_dates:
        raise ValueError("Saved raw test artifact dates do not align with rebuilt test dates.")
    if float(np.mean((test_y_true_saved - y_test_true) ** 2)) > 1e-8:
        raise ValueError("Saved raw test y_true does not match rebuilt test truth.")
    if tsm_pred_test.shape != test_tsm_saved.shape:
        raise ValueError("Rebuilt TSM test prediction shape mismatch.")

    panel_dates = pd.to_datetime(panel["date"]).to_numpy()
    date_to_idx = {pd.Timestamp(d): i for i, d in enumerate(panel_dates)}

    sentiment_cfg = llm_cfg.get("sentiment", {}) or {}
    sentiment_path = Path(sentiment_cfg.get("path", "data/news/daily_sentiment.csv"))
    if not sentiment_path.is_absolute():
        sentiment_path = ROOT / sentiment_path
    sentiment_points = int(sentiment_cfg.get("history_points", 18))
    prompt_history_points = int(llm_cfg.get("history_points", 18))
    cot_rf_cfg = llm_cfg.get("cot_rf", {}) or {}
    k_examples = int(cot_rf_cfg.get("k_examples", 5))
    selection_mode = str(cot_rf_cfg.get("example_selection", "similarity")).lower()
    feature_window = int(cot_rf_cfg.get("feature_window", 18))
    lookback_days = cot_rf_cfg.get("lookback_days")
    lookback_days = int(lookback_days) if lookback_days is not None else None
    if selection_mode != "similarity":
        raise ValueError(f"This tuning script currently expects similarity selection, got {selection_mode}.")

    sentiment_map = load_daily_sentiment(
        str(sentiment_path),
        date_col=sentiment_cfg.get("date_col", "seendate"),
        score_col=sentiment_cfg.get("score_col", "sent_score"),
    )
    val_date_arrays = build_history_dates(panel_dates, date_to_idx, dates_val, prompt_history_points)
    val_sentiment_histories = build_sentiment_histories(
        val_date_arrays,
        sentiment_map,
        sentiment_points,
    )

    print("[LLM] Building strict train-only teaching examples for validation", flush=True)
    train_dates = pd.to_datetime(dates_train).to_numpy()
    train_order = np.argsort(train_dates)
    train_dates = train_dates[train_order]
    train_histories = y_train_hist[train_order]
    train_forecasts = tsm_pred_train[train_order]
    train_truth = y_train_true[train_order]
    train_history_features = np.vstack([
        _history_features(hist, feature_window) for hist in train_histories
    ])
    train_sentiment_cache: dict[str, np.ndarray] = {}

    def build_examples_for_val(idx: int) -> list[dict]:
        ref_date = pd.Timestamp(dates_val[idx]).to_datetime64()
        pos = int(np.searchsorted(train_dates, ref_date, side="left"))
        if pos <= 0:
            return []
        candidate_indices = np.arange(pos, dtype=int)
        if lookback_days is not None:
            cutoff = np.datetime64(pd.Timestamp(ref_date) - pd.Timedelta(days=lookback_days))
            candidate_indices = candidate_indices[train_dates[candidate_indices] >= cutoff]
            if candidate_indices.size == 0:
                candidate_indices = np.arange(pos, dtype=int)

        sample_feat = _history_features(y_val_hist[idx], feature_window)
        cand_feats = train_history_features[candidate_indices]
        distances = np.linalg.norm(cand_feats - sample_feat, axis=1)
        chosen = candidate_indices[np.argsort(distances)[:k_examples]]
        chosen = chosen[np.argsort(train_dates[chosen])]

        examples: list[dict] = []
        for example_idx in chosen:
            example_date = pd.Timestamp(train_dates[example_idx]).normalize()
            key = str(example_date)
            if key not in train_sentiment_cache:
                ex_dates = build_history_dates(panel_dates, date_to_idx, np.array([example_date]), prompt_history_points)[0]
                train_sentiment_cache[key] = build_sentiment_histories(
                    [ex_dates],
                    sentiment_map,
                    sentiment_points,
                )[0]
            examples.append(
                {
                    "history": train_histories[example_idx],
                    "forecast": train_forecasts[example_idx],
                    "truth": train_truth[example_idx],
                    "date": str(pd.Timestamp(train_dates[example_idx])),
                    "sentiment_history": train_sentiment_cache[key],
                }
            )
        return examples

    val_teaching_examples = []
    for idx in range(len(dates_val)):
        val_teaching_examples.append(build_examples_for_val(idx))
        if (idx + 1) % 50 == 0 or idx + 1 == len(dates_val):
            print(f"[LLM] Built examples for {idx + 1}/{len(dates_val)} validation samples", flush=True)

    val_histories = y_val_hist[:, -prompt_history_points:]
    llm_dir = output_dir / "llm"
    llm_dir.mkdir(parents=True, exist_ok=True)
    refiner = LLMRefiner(
        llm_cfg,
        cache_dir=llm_dir / "cache",
        log_dir=llm_dir / "logs",
    )

    print("[LLM] Running validation refinement for TSM+LLM-COT-SENT-RF", flush=True)
    val_yhat_raw, metadata = refiner.refine_batch(
        method="TSM+LLM-COT-SENT-RF",
        histories=val_histories,
        date_arrays=val_date_arrays,
        tsm_forecasts=tsm_pred_val,
        pred_len=pred_len,
        exogenous_summaries=[None for _ in range(len(dates_val))],
        price_bases=y_val_base,
        teaching_examples=val_teaching_examples,
        sentiment_histories=val_sentiment_histories,
    )

    blend_grid_cfg = llm_cfg.get("blend_grid", {}) or {}
    blend_weights = [float(w) for w in blend_grid_cfg.get("weights", [0.0, 0.25, 0.5, 0.75, 1.0])]
    blend_schedule = str(blend_grid_cfg.get("schedule", "ramp")).lower()
    key_horizons = [int(h) for h in blend_grid_cfg.get("key_horizons", [1, 5, 10, 20, 30])]
    min_weight = float(blend_grid_cfg.get("min_weight", 0.0))
    power = float(blend_grid_cfg.get("power", 1.0))

    val_summary, val_mse_by_h, val_best_by_h = evaluate_blend_grid(
        y_true=y_val_true,
        base_pred=tsm_pred_val,
        llm_pred=val_yhat_raw,
        weights=blend_weights,
        schedule=blend_schedule,
        key_horizons=key_horizons,
        min_weight=min_weight,
        power=power,
    )
    best_w = float(val_summary.loc[val_summary["mse_path"].idxmin(), "w"])
    val_yhat_blend = blend_forecasts(
        base_pred=tsm_pred_val,
        llm_pred=val_yhat_raw,
        strength=best_w,
        schedule=blend_schedule,
        pred_len=pred_len,
        min_weight=min_weight,
        power=power,
    )
    val_yhat_blend_h1base = val_yhat_blend.copy()
    val_yhat_blend_h1base[:, 0] = tsm_pred_val[:, 0]

    val_rows = [
        _metric_row("tsm", y_val_true, tsm_pred_val),
        _metric_row("cot_sent_rf_raw", y_val_true, val_yhat_raw),
        _metric_row(f"cot_sent_rf_blend_ramp_w{best_w:.2f}", y_val_true, val_yhat_blend),
        _metric_row(f"cot_sent_rf_blend_ramp_w{best_w:.2f}_h1base", y_val_true, val_yhat_blend_h1base),
    ]
    val_df = pd.DataFrame(val_rows).sort_values("path_mse")
    best_variant = str(val_df.iloc[0]["variant"])

    test_yhat_blend = blend_forecasts(
        base_pred=test_tsm_saved,
        llm_pred=test_yhat_raw_saved,
        strength=best_w,
        schedule=blend_schedule,
        pred_len=pred_len,
        min_weight=min_weight,
        power=power,
    )
    test_yhat_blend_h1base = test_yhat_blend.copy()
    test_yhat_blend_h1base[:, 0] = test_tsm_saved[:, 0]
    test_rows = [
        _metric_row("tsm", test_y_true_saved, test_tsm_saved),
        _metric_row("cot_sent_rf_raw", test_y_true_saved, test_yhat_raw_saved),
        _metric_row(f"cot_sent_rf_blend_ramp_w{best_w:.2f}", test_y_true_saved, test_yhat_blend),
        _metric_row(f"cot_sent_rf_blend_ramp_w{best_w:.2f}_h1base", test_y_true_saved, test_yhat_blend_h1base),
    ]
    test_df = pd.DataFrame(test_rows).sort_values("path_mse")

    selected_test_variant = (
        test_yhat_raw_saved if best_variant == "cot_sent_rf_raw"
        else test_yhat_blend_h1base if best_variant.endswith("_h1base")
        else test_yhat_blend if "blend_ramp" in best_variant
        else test_tsm_saved
    )

    val_df.to_csv(output_dir / "val_variant_comparison.csv", index=False)
    test_df.to_csv(output_dir / "test_variant_comparison.csv", index=False)
    val_summary.to_csv(output_dir / "val_blend_grid_summary.csv", index=False)
    val_mse_by_h.to_csv(output_dir / "val_blend_grid_mse_by_horizon.csv", index=False)
    val_best_by_h.to_csv(output_dir / "val_blend_grid_best_w_by_horizon.csv", index=False)
    np.savez_compressed(
        output_dir / "val_cot_sent_rf_raw_predictions.npz",
        dates=np.asarray([str(x) for x in dates_val], dtype="U"),
        y_true=np.asarray(y_val_true, dtype=float),
        tsm_pred=np.asarray(tsm_pred_val, dtype=float),
        yhat=np.asarray(val_yhat_raw, dtype=float),
    )
    np.savez_compressed(
        output_dir / "test_selected_variant_predictions.npz",
        dates=np.asarray(saved_dates, dtype="U"),
        y_true=np.asarray(test_y_true_saved, dtype=float),
        tsm_pred=np.asarray(test_tsm_saved, dtype=float),
        yhat=np.asarray(selected_test_variant, dtype=float),
    )
    with (output_dir / "val_llm_metadata.jsonl").open("w") as handle:
        for row in metadata:
            handle.write(json.dumps(row) + "\n")

    selection_summary = {
        "base_run": str(base_run),
        "best_w_path": best_w,
        "blend_schedule": blend_schedule,
        "blend_min_weight": min_weight,
        "blend_power": power,
        "best_variant_on_val": best_variant,
        "val_results": {row["variant"]: row["path_mse"] for row in val_rows},
        "test_results": {row["variant"]: row["path_mse"] for row in test_rows},
        "selected_test_path_mse": float(compute_path_metrics(test_y_true_saved, selected_test_variant)["mse_path"]),
    }
    with (output_dir / "selection_summary.json").open("w") as handle:
        json.dump(selection_summary, handle, indent=2)

    print(json.dumps(selection_summary, indent=2), flush=True)
    print(f"Wrote tuning artifacts to {output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
