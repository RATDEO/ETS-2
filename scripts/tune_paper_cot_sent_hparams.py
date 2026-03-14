#!/usr/bin/env python3
"""
Leakage-free tuning for CoT-SENT prompt/example hyperparameters.

Workflow:
1. Load a saved clean paper-style run and its TSM checkpoint.
2. Recompute TSM forecasts from the checkpoint.
3. Search CoT-SENT candidates on a validation subset only.
4. Re-evaluate the top candidates on the full validation split.
5. Run the selected best candidate once on the held-out test split.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from data import select_feature_columns
from data.windows import StandardScaler, TimeSeriesDataset, WindowConfig, make_windows, split_windows
from eval.metrics import compute_metrics_by_horizon, compute_path_metrics
from llm.refine import LLMRefiner
from models.tsm import TSMForecaster
from run_experiment import (
    build_history_dates,
    build_sentiment_histories,
    load_daily_sentiment,
    returns_to_prices,
    select_eval_indices,
)

try:
    from torch.utils.data import DataLoader
except ImportError as exc:  # pragma: no cover
    raise SystemExit("PyTorch is required for tuning.") from exc


@dataclass(frozen=True)
class CoTSentCandidate:
    name: str
    history_points: int
    prompt_history_points: int
    prompt_sentiment_points: int
    sentiment_points: int
    k_examples: int
    example_selection: str
    feature_window: int
    lookback_days: int | None
    retain_context: bool
    strict_json_prompt: bool = False


CANDIDATES: list[CoTSentCandidate] = [
    CoTSentCandidate(
        name="paper_sim_k5_h18_lb365_ctx",
        history_points=18,
        prompt_history_points=18,
        prompt_sentiment_points=18,
        sentiment_points=18,
        k_examples=5,
        example_selection="similarity",
        feature_window=18,
        lookback_days=365,
        retain_context=True,
    ),
    CoTSentCandidate(
        name="sim_k10_h18_lb365_ctx",
        history_points=18,
        prompt_history_points=18,
        prompt_sentiment_points=18,
        sentiment_points=18,
        k_examples=10,
        example_selection="similarity",
        feature_window=18,
        lookback_days=365,
        retain_context=True,
    ),
    CoTSentCandidate(
        name="sim_k5_h24_lb365_ctx",
        history_points=24,
        prompt_history_points=24,
        prompt_sentiment_points=24,
        sentiment_points=24,
        k_examples=5,
        example_selection="similarity",
        feature_window=24,
        lookback_days=365,
        retain_context=True,
    ),
    CoTSentCandidate(
        name="sim_k5_h18_lb180_ctx",
        history_points=18,
        prompt_history_points=18,
        prompt_sentiment_points=18,
        sentiment_points=18,
        k_examples=5,
        example_selection="similarity",
        feature_window=18,
        lookback_days=180,
        retain_context=True,
    ),
    CoTSentCandidate(
        name="recent_high_error_k5_h18_lb365_ctx",
        history_points=18,
        prompt_history_points=18,
        prompt_sentiment_points=18,
        sentiment_points=18,
        k_examples=5,
        example_selection="recent_high_error",
        feature_window=18,
        lookback_days=365,
        retain_context=True,
    ),
    CoTSentCandidate(
        name="sim_k5_h18_lb365_nocontext",
        history_points=18,
        prompt_history_points=18,
        prompt_sentiment_points=18,
        sentiment_points=18,
        k_examples=5,
        example_selection="similarity",
        feature_window=18,
        lookback_days=365,
        retain_context=False,
    ),
    CoTSentCandidate(
        name="error_stratified_k5_h18_lb365_ctx",
        history_points=18,
        prompt_history_points=18,
        prompt_sentiment_points=18,
        sentiment_points=18,
        k_examples=5,
        example_selection="error_stratified",
        feature_window=18,
        lookback_days=365,
        retain_context=True,
    ),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tune CoT-SENT hyperparameters on a clean saved run.")
    parser.add_argument(
        "--base-run",
        type=Path,
        default=Path("runs/20260228_145058_40d91f"),
        help="Existing clean paper-style CoT-SENT run directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default under reports/cot_sent_hparam_tune).",
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
    parser.add_argument(
        "--val-subset-samples",
        type=int,
        default=60,
        help="Validation subset size for stage-1 candidate search.",
    )
    parser.add_argument(
        "--top-k-full-val",
        type=int,
        default=2,
        help="How many stage-1 winners to rerun on the full validation split.",
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


def _error_profile(forecast: np.ndarray, truth: np.ndarray) -> np.ndarray:
    horizon_idx = [0, 4, 19, 29]
    horizon_idx = [idx for idx in horizon_idx if idx < len(forecast) and idx < len(truth)]
    errs = np.asarray(forecast, dtype=float)[horizon_idx] - np.asarray(truth, dtype=float)[horizon_idx]
    if len(errs) < 4:
        errs = np.pad(errs, (0, 4 - len(errs)))
    return errs.astype(float)


def _select_error_stratified_indices(
    candidate_indices: np.ndarray,
    pool_dates: np.ndarray,
    pool_error_profiles: np.ndarray,
    k_examples: int,
) -> np.ndarray:
    if len(candidate_indices) <= k_examples:
        return candidate_indices[np.argsort(pool_dates[candidate_indices])]

    cand_profiles = np.asarray(pool_error_profiles[candidate_indices], dtype=float)
    center = np.mean(cand_profiles, axis=0, keepdims=True)
    scale = np.std(cand_profiles, axis=0, keepdims=True)
    scale[scale < 1e-6] = 1.0
    norm_profiles = (cand_profiles - center) / scale
    norms = np.linalg.norm(norm_profiles, axis=1)

    selected_pos: list[int] = [int(np.argmin(norms))]
    while len(selected_pos) < k_examples:
        remaining = [i for i in range(len(candidate_indices)) if i not in selected_pos]
        if not remaining:
            break
        scored: list[tuple[float, float, int]] = []
        for pos in remaining:
            min_dist = min(
                float(np.linalg.norm(norm_profiles[pos] - norm_profiles[sel]))
                for sel in selected_pos
            )
            recency = float(candidate_indices[pos])
            scored.append((min_dist, recency, pos))
        scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
        selected_pos.append(int(scored[0][2]))

    selected = candidate_indices[np.asarray(selected_pos, dtype=int)]
    return selected[np.argsort(pool_dates[selected])]


def _metric_row(name: str, y_true: np.ndarray, yhat: np.ndarray) -> dict[str, float | str]:
    metrics = compute_metrics_by_horizon(y_true, yhat, [1, 5, 20, 30]).reset_index()
    row: dict[str, float | str] = {"name": name, "path_mse": float(compute_path_metrics(y_true, yhat)["mse_path"])}
    for _, metric_row in metrics.iterrows():
        row[f"h{int(metric_row['horizon'])}_mse"] = float(metric_row["mse"])
    return row


def _predict_returns(
    tsm: TSMForecaster,
    scaler: StandardScaler,
    x_enc: np.ndarray,
    x_dec: np.ndarray,
    y: np.ndarray,
) -> np.ndarray:
    dataset = TimeSeriesDataset(
        scaler.transform(x_enc),
        scaler.transform(x_dec),
        (y - scaler.mean_[0]) / scaler.std_[0],
    )
    loader = DataLoader(dataset, batch_size=32)
    pred_scaled, _ = tsm.predict(loader)
    return scaler.inverse_transform_target(pred_scaled)


def _prepare_bundle(base_run: Path) -> dict[str, Any]:
    with (base_run / "config_resolved.yaml").open("r") as handle:
        cfg = yaml.safe_load(handle)

    target_mode = str((cfg.get("target", {}) or {}).get("mode", "price"))
    if target_mode != "returns":
        raise ValueError("This tuning script currently expects target.mode=returns.")

    panel = pd.read_parquet(base_run / "data" / "panel.parquet")
    target_col = "y_return"
    feature_cols = select_feature_columns(
        panel,
        target_col=target_col,
        max_exogenous_features=int((cfg.get("features", {}) or {}).get("max_exogenous_features_model", 10)),
        preferred_feature_order=((cfg.get("features", {}) or {}).get("preferred_feature_order")),
    )
    price_idx = feature_cols.index("y")

    ts_cfg = cfg.get("time_series", {}) or {}
    window_config = WindowConfig(
        seq_len=int(ts_cfg.get("seq_len", 60)),
        label_len=int(ts_cfg.get("label_len", 15)),
        pred_len=int(ts_cfg.get("pred_len", 30)),
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

    scaler = StandardScaler().fit(x_train_enc)
    tsm_cfg = {
        "time_series": dict(cfg.get("time_series", {}) or {}),
        "model": dict(cfg.get("model", {}) or {}),
    }
    tsm_cfg["model"]["enc_in"] = int(x_train_enc.shape[-1])
    tsm_cfg["model"]["dec_in"] = int(x_train_dec.shape[-1])
    tsm = TSMForecaster(tsm_cfg, device="auto")
    tsm.load(base_run / "models" / "tsm_checkpoint.pt")

    y_train_hist = x_train_enc[:, :, price_idx]
    y_val_hist = x_val_enc[:, :, price_idx]
    y_test_hist = x_test_enc[:, :, price_idx]
    y_train_base = y_train_hist[:, -1]
    y_val_base = y_val_hist[:, -1]
    y_test_base = y_test_hist[:, -1]
    y_train_true = returns_to_prices(y_train, y_train_base)
    y_val_true = returns_to_prices(y_val, y_val_base)
    y_test_true = returns_to_prices(y_test, y_test_base)

    print("[TSM] Recomputing train/val/test forecasts from saved checkpoint", flush=True)
    tsm_pred_train = returns_to_prices(_predict_returns(tsm, scaler, x_train_enc, x_train_dec, y_train), y_train_base)
    tsm_pred_val = returns_to_prices(_predict_returns(tsm, scaler, x_val_enc, x_val_dec, y_val), y_val_base)
    tsm_pred_test = returns_to_prices(_predict_returns(tsm, scaler, x_test_enc, x_test_dec, y_test), y_test_base)

    panel_dates = pd.to_datetime(panel["date"]).to_numpy()
    date_to_idx = {pd.Timestamp(d): i for i, d in enumerate(panel_dates)}

    return {
        "cfg": cfg,
        "splits": splits,
        "window_config": window_config,
        "panel_dates": panel_dates,
        "date_to_idx": date_to_idx,
        "y_train_hist": y_train_hist,
        "y_val_hist": y_val_hist,
        "y_test_hist": y_test_hist,
        "y_train_true": y_train_true,
        "y_val_true": y_val_true,
        "y_test_true": y_test_true,
        "y_val_base": y_val_base,
        "y_test_base": y_test_base,
        "tsm_pred_train": tsm_pred_train,
        "tsm_pred_val": tsm_pred_val,
        "tsm_pred_test": tsm_pred_test,
    }


def _select_example_indices(
    pool_dates: np.ndarray,
    pool_features: np.ndarray,
    pool_errors: np.ndarray,
    pool_error_profiles: np.ndarray,
    ref_date: Any,
    ref_history: np.ndarray,
    candidate: CoTSentCandidate,
) -> np.ndarray:
    ref_ts = pd.Timestamp(ref_date).to_datetime64()
    pos = int(np.searchsorted(pool_dates, ref_ts, side="left"))
    if pos <= 0:
        return np.array([], dtype=int)

    candidate_indices = np.arange(pos, dtype=int)
    if candidate.lookback_days is not None:
        cutoff = np.datetime64(pd.Timestamp(ref_ts) - pd.Timedelta(days=candidate.lookback_days))
        candidate_indices = candidate_indices[pool_dates[candidate_indices] >= cutoff]
        if candidate_indices.size == 0:
            candidate_indices = np.arange(pos, dtype=int)

    if candidate.example_selection == "recent_high_error":
        cand_errors = pool_errors[candidate_indices]
        order = np.argsort(cand_errors)[-candidate.k_examples :]
        selected = candidate_indices[order]
        return selected[np.argsort(pool_dates[selected])]

    if candidate.example_selection == "error_stratified":
        return _select_error_stratified_indices(
            candidate_indices=candidate_indices,
            pool_dates=pool_dates,
            pool_error_profiles=pool_error_profiles,
            k_examples=candidate.k_examples,
        )

    sample_feat = _history_features(ref_history, candidate.feature_window)
    cand_feats = pool_features[candidate_indices]
    distances = np.linalg.norm(cand_feats - sample_feat, axis=1)
    order = np.argsort(distances)[: candidate.k_examples]
    selected = candidate_indices[order]
    return selected[np.argsort(pool_dates[selected])]


def _build_examples(
    bundle: dict[str, Any],
    candidate: CoTSentCandidate,
    split_name: str,
    eval_indices: np.ndarray,
    sentiment_map: dict,
) -> tuple[list[np.ndarray], list[list[str]], list[np.ndarray], list[list[dict[str, Any]]]]:
    splits = bundle["splits"]
    panel_dates = bundle["panel_dates"]
    date_to_idx = bundle["date_to_idx"]

    if split_name == "val":
        ref_dates = splits["val"]["dates"]
        ref_hist = bundle["y_val_hist"]
        pool_dates = pd.to_datetime(splits["train"]["dates"]).to_numpy()
        pool_hist = bundle["y_train_hist"]
        pool_forecasts = bundle["tsm_pred_train"]
        pool_truth = bundle["y_train_true"]
    elif split_name == "test":
        ref_dates = splits["test"]["dates"]
        ref_hist = bundle["y_test_hist"]
        pool_dates = pd.to_datetime(np.concatenate([splits["train"]["dates"], splits["val"]["dates"]])).to_numpy()
        pool_hist = np.concatenate([bundle["y_train_hist"], bundle["y_val_hist"]])
        pool_forecasts = np.concatenate([bundle["tsm_pred_train"], bundle["tsm_pred_val"]])
        pool_truth = np.concatenate([bundle["y_train_true"], bundle["y_val_true"]])
    else:
        raise ValueError(f"Unknown split: {split_name}")

    pool_order = np.argsort(pool_dates)
    pool_dates = pool_dates[pool_order]
    pool_hist = pool_hist[pool_order]
    pool_forecasts = pool_forecasts[pool_order]
    pool_truth = pool_truth[pool_order]
    pool_errors = np.mean((pool_forecasts - pool_truth) ** 2, axis=1)
    pool_error_profiles = np.vstack(
        [_error_profile(fc, tr) for fc, tr in zip(pool_forecasts, pool_truth, strict=False)]
    )
    pool_features = np.vstack([_history_features(hist, candidate.feature_window) for hist in pool_hist])

    eval_dates = ref_dates[eval_indices]
    histories = ref_hist[eval_indices][:, -candidate.history_points :]
    date_arrays = build_history_dates(
        panel_dates,
        date_to_idx,
        eval_dates,
        candidate.history_points,
    )
    sentiment_histories = build_sentiment_histories(
        date_arrays,
        sentiment_map,
        candidate.sentiment_points,
    )

    sentiment_cache: dict[str, np.ndarray] = {}
    teaching_examples: list[list[dict[str, Any]]] = []
    for sample_idx in eval_indices:
        example_indices = _select_example_indices(
            pool_dates=pool_dates,
            pool_features=pool_features,
            pool_errors=pool_errors,
            pool_error_profiles=pool_error_profiles,
            ref_date=ref_dates[sample_idx],
            ref_history=ref_hist[sample_idx],
            candidate=candidate,
        )
        examples: list[dict[str, Any]] = []
        for ex_idx in example_indices:
            example_date = pd.Timestamp(pool_dates[ex_idx]).normalize()
            cache_key = str(example_date)
            if cache_key not in sentiment_cache:
                ex_dates = build_history_dates(
                    panel_dates,
                    date_to_idx,
                    np.array([example_date]),
                    candidate.history_points,
                )[0]
                sentiment_cache[cache_key] = build_sentiment_histories(
                    [ex_dates],
                    sentiment_map,
                    candidate.sentiment_points,
                )[0]
            examples.append(
                {
                    "history": pool_hist[ex_idx],
                    "forecast": pool_forecasts[ex_idx],
                    "truth": pool_truth[ex_idx],
                    "date": str(pd.Timestamp(pool_dates[ex_idx])),
                    "sentiment_history": sentiment_cache[cache_key],
                }
            )
        teaching_examples.append(examples)
    return histories, date_arrays, sentiment_histories, teaching_examples


def _evaluate_candidate(
    bundle: dict[str, Any],
    base_llm_cfg: dict[str, Any],
    candidate: CoTSentCandidate,
    split_name: str,
    eval_indices: np.ndarray,
    output_dir: Path,
    method: str = "TSM+LLM-COT-SENT-RF",
) -> dict[str, Any]:
    llm_cfg = copy.deepcopy(base_llm_cfg)
    llm_cfg["history_points"] = candidate.history_points
    llm_cfg["prompt_history_points"] = candidate.prompt_history_points
    llm_cfg["prompt_sentiment_points"] = candidate.prompt_sentiment_points
    llm_cfg.setdefault("sentiment", {})
    llm_cfg["sentiment"]["history_points"] = candidate.sentiment_points
    llm_cfg.setdefault("cot_rf", {})
    llm_cfg["cot_rf"]["k_examples"] = candidate.k_examples
    llm_cfg["cot_rf"]["example_selection"] = candidate.example_selection
    llm_cfg["cot_rf"]["feature_window"] = candidate.feature_window
    llm_cfg["cot_rf"]["lookback_days"] = candidate.lookback_days
    llm_cfg["cot_rf"]["retain_context"] = candidate.retain_context
    llm_cfg["cot_rf"]["strict_json_prompt"] = candidate.strict_json_prompt

    sentiment_cfg = llm_cfg.get("sentiment", {}) or {}
    sentiment_path = Path(sentiment_cfg.get("path", "data/news/daily_sentiment.csv"))
    if not sentiment_path.is_absolute():
        sentiment_path = ROOT / sentiment_path
    sentiment_map = load_daily_sentiment(
        str(sentiment_path),
        date_col=sentiment_cfg.get("date_col", "seendate"),
        score_col=sentiment_cfg.get("score_col", "sent_score"),
    )

    histories, date_arrays, sentiment_histories, teaching_examples = _build_examples(
        bundle=bundle,
        candidate=candidate,
        split_name=split_name,
        eval_indices=eval_indices,
        sentiment_map=sentiment_map,
    )

    if split_name == "val":
        base_pred = bundle["tsm_pred_val"][eval_indices]
        y_true = bundle["y_val_true"][eval_indices]
        price_bases = bundle["y_val_base"][eval_indices]
    else:
        base_pred = bundle["tsm_pred_test"][eval_indices]
        y_true = bundle["y_test_true"][eval_indices]
        price_bases = bundle["y_test_base"][eval_indices]

    run_dir = output_dir / candidate.name / split_name
    run_dir.mkdir(parents=True, exist_ok=True)
    refiner = LLMRefiner(
        llm_cfg,
        cache_dir=run_dir / "cache",
        log_dir=run_dir / "logs",
    )
    yhat, metadata = refiner.refine_batch(
        method=method,
        histories=np.asarray(histories),
        date_arrays=date_arrays,
        tsm_forecasts=base_pred,
        pred_len=base_pred.shape[1],
        exogenous_summaries=[None for _ in range(len(eval_indices))],
        price_bases=price_bases,
        teaching_examples=teaching_examples,
        sentiment_histories=sentiment_histories,
    )

    result = _metric_row(candidate.name, y_true, yhat)
    result["split"] = split_name
    result["n_samples"] = int(len(eval_indices))
    result["candidate"] = candidate.name
    result["metadata_sample"] = metadata[0] if metadata else {}
    np.savez_compressed(
        run_dir / f"{split_name}_predictions.npz",
        dates=np.asarray([str(d) for d in (bundle["splits"][split_name]["dates"][eval_indices])], dtype="U"),
        y_true=np.asarray(y_true, dtype=float),
        tsm_pred=np.asarray(base_pred, dtype=float),
        yhat=np.asarray(yhat, dtype=float),
    )
    with (run_dir / "summary.json").open("w") as handle:
        json.dump(
            {
                **{k: v for k, v in result.items() if isinstance(v, (str, int, float))},
                "candidate_config": asdict(candidate),
            },
            handle,
            indent=2,
        )
    return result


def main() -> int:
    args = parse_args()
    base_run = args.base_run if args.base_run.is_absolute() else (ROOT / args.base_run)
    if not base_run.exists():
        raise FileNotFoundError(f"Run not found: {base_run}")

    if args.output_dir is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = ROOT / "reports" / "cot_sent_hparam_tune" / stamp
    else:
        output_dir = args.output_dir if args.output_dir.is_absolute() else (ROOT / args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    bundle = _prepare_bundle(base_run)
    base_llm_cfg = copy.deepcopy(bundle["cfg"].get("llm", {}) or {})
    base_llm_cfg["api_key"] = args.llm_api_key or os.environ.get("OPENAI_API_KEY") or base_llm_cfg.get("api_key") or "deo"
    base_llm_cfg["base_url"] = args.llm_base_url or os.environ.get("OPENAI_BASE_URL") or base_llm_cfg.get("base_url")
    base_llm_cfg["methods"] = ["TSM+LLM-COT-SENT-RF"]
    base_llm_cfg["max_samples"] = 100000

    val_n = len(bundle["splits"]["val"]["dates"])
    subset_eval_indices = select_eval_indices(
        np.arange(val_n, dtype=int),
        max_samples=int(min(max(args.val_subset_samples, 1), val_n)),
        strategy="spaced",
        seed=42,
    )
    full_val_indices = np.arange(val_n, dtype=int)
    full_test_indices = np.arange(len(bundle["splits"]["test"]["dates"]), dtype=int)

    manifest = {
        "base_run": str(base_run),
        "val_subset_samples": int(len(subset_eval_indices)),
        "top_k_full_val": int(args.top_k_full_val),
        "candidates": [asdict(c) for c in CANDIDATES],
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"[SEARCH] stage1 subset size={len(subset_eval_indices)} candidates={len(CANDIDATES)}", flush=True)
    stage1_results = []
    for candidate in CANDIDATES:
        print(f"[SEARCH] subset {candidate.name}", flush=True)
        result = _evaluate_candidate(
            bundle=bundle,
            base_llm_cfg=base_llm_cfg,
            candidate=candidate,
            split_name="val",
            eval_indices=subset_eval_indices,
            output_dir=output_dir / "stage1_subset",
        )
        stage1_results.append(result)
        pd.DataFrame(stage1_results).sort_values("path_mse").to_csv(
            output_dir / "stage1_subset_results.csv",
            index=False,
        )
        print(f"[SEARCH] subset done {candidate.name} path_mse={result['path_mse']:.6f}", flush=True)

    stage1_df = pd.DataFrame(stage1_results).sort_values("path_mse")
    top_names = stage1_df["candidate"].head(int(min(args.top_k_full_val, len(stage1_df)))).tolist()
    top_candidates = [candidate for candidate in CANDIDATES if candidate.name in top_names]

    print(f"[SEARCH] stage2 full validation top={top_names}", flush=True)
    stage2_results = []
    for candidate in top_candidates:
        print(f"[SEARCH] full val {candidate.name}", flush=True)
        result = _evaluate_candidate(
            bundle=bundle,
            base_llm_cfg=base_llm_cfg,
            candidate=candidate,
            split_name="val",
            eval_indices=full_val_indices,
            output_dir=output_dir / "stage2_full_val",
        )
        stage2_results.append(result)
        pd.DataFrame(stage2_results).sort_values("path_mse").to_csv(
            output_dir / "stage2_full_val_results.csv",
            index=False,
        )
        print(f"[SEARCH] full val done {candidate.name} path_mse={result['path_mse']:.6f}", flush=True)

    stage2_df = pd.DataFrame(stage2_results).sort_values("path_mse")
    best_name = str(stage2_df.iloc[0]["candidate"])
    best_candidate = next(candidate for candidate in CANDIDATES if candidate.name == best_name)

    print(f"[SEARCH] stage3 test best={best_name}", flush=True)
    test_result = _evaluate_candidate(
        bundle=bundle,
        base_llm_cfg=base_llm_cfg,
        candidate=best_candidate,
        split_name="test",
        eval_indices=full_test_indices,
        output_dir=output_dir / "stage3_test",
    )
    pd.DataFrame([test_result]).to_csv(output_dir / "stage3_test_result.csv", index=False)

    baseline_test_npz = np.load(
        base_run / "predictions" / "TSM+LLM-COT-SENT-RF_pred_test_subset.npz",
        allow_pickle=True,
    )
    baseline_test = _metric_row(
        "baseline_paper_raw",
        np.asarray(baseline_test_npz["y_true"], dtype=float),
        np.asarray(baseline_test_npz["yhat"], dtype=float),
    )
    baseline_test["split"] = "test"
    baseline_test["candidate"] = "baseline_paper_raw"
    pd.DataFrame([baseline_test, test_result]).to_csv(output_dir / "final_test_comparison.csv", index=False)

    summary = {
        "base_run": str(base_run),
        "best_candidate": asdict(best_candidate),
        "stage1_best": stage1_df.head(int(min(args.top_k_full_val, len(stage1_df)))).to_dict(orient="records"),
        "stage2_best": stage2_df.iloc[0].to_dict(),
        "baseline_test": baseline_test,
        "best_candidate_test": test_result,
    }
    (output_dir / "selection_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    print(f"Wrote tuning outputs to {output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
