#!/usr/bin/env python3
"""
Compare CoT-Sent-RF test MSE when swapping the daily sentiment series.

This script reuses an existing run's:
- window datasets (train/val/test .npz)
- trained TSM checkpoint
- CoT-Sent-RF configuration (k_examples/selection/lookback/etc.)

It then:
1) Recomputes TSM forecasts (train/val/test) from the saved checkpoint so we can
   rebuild teaching-example pools.
2) Runs LLM refinement on the test split using a provided daily sentiment CSV.
3) Compares path MSE (and a few horizon MSEs) to the baseline run's saved
   `TSM+LLM-COT-SENT-RF_pred_test_subset.npz`.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
from datetime import datetime
import sys
import threading

import numpy as np
import pandas as pd
import yaml


def _load_npz(path: Path) -> dict:
    z = np.load(path, allow_pickle=True)
    return {k: z[k] for k in z.files}


def _history_features(hist: np.ndarray, feature_window: int) -> np.ndarray:
    window = hist[-feature_window:] if len(hist) >= feature_window else hist
    mean = float(np.mean(window)) if len(window) else 0.0
    std = float(np.std(window)) if len(window) else 0.0
    if len(window) > 1:
        slope = float(np.polyfit(np.arange(len(window)), window, 1)[0])
    else:
        slope = 0.0
    return np.array([mean, std, slope], dtype=float)


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Re-run CoT-Sent-RF LLM refinement on an existing run using an alternate daily sentiment series, and compare MSEs."
    )
    parser.add_argument("--base-run", required=True, help="Existing run directory under runs/")
    parser.add_argument("--sentiment-path", required=True, help="Daily sentiment CSV to use")
    parser.add_argument(
        "--sentiment-feature-cols",
        default="",
        help=(
            "Optional comma-separated feature columns to use from sentiment CSV "
            "(e.g. sent_score,sent_change,news_volume)."
        ),
    )
    parser.add_argument(
        "--out-run",
        default=None,
        help="Output run dir (default: runs/YYYYMMDD_HHMMSS_sentiment_compare)",
    )
    parser.add_argument(
        "--force-refine-all",
        action="store_true",
        help="Refine all test windows (do not reuse baseline predictions for unchanged sentiment windows).",
    )
    parser.add_argument(
        "--paper-replication-mode",
        action="store_true",
        help="Enforce strict paper-style CoT-SENT-RF settings: h=18, k=5, retained context.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of parallel LLM refinement workers (default: 1).",
    )
    parser.add_argument(
        "--strict-json-prompt",
        action="store_true",
        help=(
            "Use opt-in strict JSON CoT-SENT-RF prompts (reflection/apply) without "
            "changing baseline prompt defaults."
        ),
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    # Allow `import src.*` when running as a script.
    sys.path.insert(0, str(repo_root))
    sys.path.insert(0, str(repo_root / "src"))
    base_run = (repo_root / args.base_run).resolve() if not Path(args.base_run).is_absolute() else Path(args.base_run)
    sentiment_path = (repo_root / args.sentiment_path).resolve() if not Path(args.sentiment_path).is_absolute() else Path(args.sentiment_path)
    if not base_run.exists():
        raise FileNotFoundError(f"Base run not found: {base_run}")
    if not sentiment_path.exists():
        raise FileNotFoundError(f"Sentiment file not found: {sentiment_path}")

    if args.out_run:
        out_run = (repo_root / args.out_run).resolve() if not Path(args.out_run).is_absolute() else Path(args.out_run)
    else:
        stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        out_run = repo_root / "runs" / f"{stamp}_sentiment_compare"

    # ---------------------------------------------------------------------
    # Load base config + panel schema (to reconstruct feature ordering)
    # ---------------------------------------------------------------------
    with open(base_run / "config_resolved.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    with open(base_run / "data" / "panel_schema.json", "r") as f:
        panel_schema = json.load(f)

    target_mode = (cfg.get("target", {}) or {}).get("mode", "price")
    target_col = "y_return" if target_mode == "returns" else "y"

    pred_len = int((cfg.get("time_series", {}) or {}).get("pred_len", 30))
    horizons = [int(x) for x in ((cfg.get("time_series", {}) or {}).get("horizons") or [1, 5, 20, 30])]

    llm_cfg = cfg.get("llm", {}) or {}
    method = "TSM+LLM-COT-SENT-RF"
    history_points = int(llm_cfg.get("history_points", 18))
    sentiment_cfg = llm_cfg.get("sentiment", {}) or {}
    sentiment_points = int(sentiment_cfg.get("history_points", 18))
    cot_cfg = llm_cfg.get("cot_rf", {}) or {}
    k_examples = int(cot_cfg.get("k_examples", 5))
    selection_mode = str(cot_cfg.get("example_selection", "recent")).lower()
    feature_window = int(cot_cfg.get("feature_window", 18))
    lookback_days = cot_cfg.get("lookback_days")
    lookback_days = int(lookback_days) if lookback_days is not None else None
    sentiment_feature_cols = [
        c.strip() for c in str(args.sentiment_feature_cols).split(",") if c.strip()
    ]

    if args.paper_replication_mode:
        history_points = 18
        sentiment_points = 18
        k_examples = 5
        selection_mode = "similarity"
        if lookback_days is None:
            lookback_days = 365

    # ---------------------------------------------------------------------
    # Recreate unscaled windows/splits from the saved panel.
    #
    # Note: runs/*/data/datasets/*.npz are saved in *scaled* form (for convenience),
    # but TSM training/inference in run_experiment.py uses the *unscaled* windows.
    # We therefore rebuild windows from panel.parquet to match the original run.
    # ---------------------------------------------------------------------
    from src.data.windows import WindowConfig, make_windows, split_windows

    panel = pd.read_parquet(base_run / "data" / "panel.parquet")
    if "date" not in panel.columns:
        raise ValueError("panel.parquet missing 'date' column")
    if target_col not in panel.columns:
        raise ValueError(f"panel.parquet missing target column: {target_col}")

    # Match run_experiment.py feature selection exactly (panel column order matters).
    feature_candidates = [c for c in panel.columns if c not in ["date", target_col]]
    if "y" in feature_candidates:
        feature_candidates.remove("y")
        feature_candidates = ["y"] + feature_candidates
    feature_cols = [target_col] + feature_candidates[:10]

    if target_mode == "returns":
        if "y" not in feature_cols:
            raise ValueError("Expected price feature 'y' in feature_cols for returns mode.")
        price_idx = feature_cols.index("y")
    else:
        price_idx = 0

    ts_cfg = cfg.get("time_series", {}) or {}
    seq_len = int(ts_cfg.get("seq_len", 60))
    label_len = int(ts_cfg.get("label_len", 15))

    window_config = WindowConfig(
        seq_len=seq_len,
        label_len=label_len,
        pred_len=pred_len,
        target_col=target_col,
        feature_cols=feature_cols,
    )
    X_enc, X_dec, y, dates = make_windows(panel, window_config, mode="MS")

    split_cfg = cfg.get("split", {}) or {}
    splits = split_windows(
        X_enc,
        X_dec,
        y,
        dates,
        train_end=split_cfg.get("train_end", "2023-06-30"),
        val_end=split_cfg.get("val_end", "2024-06-30"),
    )

    X_enc_train, X_dec_train, y_train, dates_train = (
        splits["train"]["X_enc"],
        splits["train"]["X_dec"],
        splits["train"]["y"],
        splits["train"]["dates"],
    )
    X_enc_val, X_dec_val, y_val, dates_val = (
        splits["val"]["X_enc"],
        splits["val"]["X_dec"],
        splits["val"]["y"],
        splits["val"]["dates"],
    )
    X_enc_test, X_dec_test, y_test, dates_test = (
        splits["test"]["X_enc"],
        splits["test"]["X_dec"],
        splits["test"]["y"],
        splits["test"]["dates"],
    )

    y_train_hist = X_enc_train[:, :, price_idx]
    y_val_hist = X_enc_val[:, :, price_idx]
    y_test_hist = X_enc_test[:, :, price_idx]

    if target_mode == "returns":
        from src.run_experiment import returns_to_prices

        y_train_base = y_train_hist[:, -1]
        y_val_base = y_val_hist[:, -1]
        y_test_base = y_test_hist[:, -1]
        y_train_true = returns_to_prices(y_train, y_train_base)
        y_val_true = returns_to_prices(y_val, y_val_base)
        y_test_true = returns_to_prices(y_test, y_test_base)
    else:
        y_train_true, y_val_true, y_test_true = y_train, y_val, y_test
        y_train_base = y_train_hist[:, -1]
        y_val_base = y_val_hist[:, -1]
        y_test_base = y_test_hist[:, -1]

    # ---------------------------------------------------------------------
    # Recompute TSM forecasts for train/val/test from checkpoint so we can
    # rebuild the teaching-example pool consistently.
    # ---------------------------------------------------------------------
    import torch  # noqa: F401
    from torch.utils.data import DataLoader
    from src.data.windows import StandardScaler
    from src.data.windows import TimeSeriesDataset
    from src.models.tsm import TSMForecaster

    local_scaler = StandardScaler().fit(X_enc_train)

    def _predict_returns(X_enc: np.ndarray, X_dec: np.ndarray, y: np.ndarray) -> np.ndarray:
        ds = TimeSeriesDataset(
            local_scaler.transform(X_enc),
            local_scaler.transform(X_dec),
            (y - local_scaler.mean_[0]) / local_scaler.std_[0],
        )
        loader = DataLoader(ds, batch_size=32)
        pred_scaled, _ = tsm.predict(loader)
        return local_scaler.inverse_transform_target(pred_scaled)

    tsm_config = {
        "time_series": dict(cfg.get("time_series", {}) or {}),
        "model": dict(cfg.get("model", {}) or {}),
    }
    tsm_config["model"]["enc_in"] = int(X_enc_train.shape[-1])
    tsm_config["model"]["dec_in"] = int(X_dec_train.shape[-1])
    tsm = TSMForecaster(tsm_config, device="auto")
    tsm.load(base_run / "models" / "tsm_checkpoint.pt")

    if target_mode != "returns":
        raise ValueError("This comparison script currently expects target.mode=returns.")

    from src.run_experiment import returns_to_prices

    tsm_pred_train_eval = returns_to_prices(_predict_returns(X_enc_train, X_dec_train, y_train), y_train_base)
    tsm_pred_val_eval = returns_to_prices(_predict_returns(X_enc_val, X_dec_val, y_val), y_val_base)
    tsm_pred_test_eval = returns_to_prices(_predict_returns(X_enc_test, X_dec_test, y_test), y_test_base)

    # ---------------------------------------------------------------------
    # Sanity check: ensure our reconstructed test truth/TSM forecasts align with
    # the baseline run's persisted subset predictions. If they don't, abort.
    # ---------------------------------------------------------------------
    base_npz = base_run / "predictions" / f"{method}_pred_test_subset.npz"
    if not base_npz.exists():
        raise FileNotFoundError(f"Missing baseline predictions: {base_npz}")
    z_base = np.load(base_npz, allow_pickle=True)
    y_true_base = np.asarray(z_base["y_true"], dtype=float)
    tsm_base = np.asarray(z_base["tsm_pred"], dtype=float)
    base_dates = [str(x) for x in np.asarray(z_base["dates"]).tolist()]
    our_dates = [str(d) for d in dates_test.tolist()]
    if base_dates != our_dates:
        raise ValueError("Test date alignment mismatch vs baseline run (dates differ).")
    if y_true_base.shape != y_test_true.shape:
        raise ValueError(f"y_true shape mismatch: baseline {y_true_base.shape} vs rebuilt {y_test_true.shape}")
    if float(np.mean((y_true_base - y_test_true) ** 2)) > 1e-8:
        raise ValueError("Rebuilt y_true does not match baseline y_true (MSE too large).")
    if tsm_base.shape != tsm_pred_test_eval.shape:
        raise ValueError(f"TSM pred shape mismatch: baseline {tsm_base.shape} vs rebuilt {tsm_pred_test_eval.shape}")
    tsm_mse = float(np.mean((tsm_base - tsm_pred_test_eval) ** 2))
    if tsm_mse > 1e-6:
        # Exact equality is not guaranteed across devices / torch versions (e.g. MPS vs CPU).
        # For an apples-to-apples comparison vs the baseline run, we use the baseline run's
        # persisted test TSM predictions going forward.
        print(f"WARNING: rebuilt TSM test preds differ from baseline (MSE={tsm_mse:.6g}); using baseline test TSM preds for LLM prompts/eval.")
    tsm_pred_test_eval = tsm_base

    # Only after sanity checks pass: create output directories.
    _ensure_dir(out_run / "predictions")
    _ensure_dir(out_run / "results")
    _ensure_dir(out_run / "llm" / "cache")
    _ensure_dir(out_run / "llm" / "logs")

    # ---------------------------------------------------------------------
    # Load panel dates (for sentiment alignment) + build sentiment histories
    # ---------------------------------------------------------------------
    panel_df = pd.read_parquet(base_run / "data" / "panel.parquet", columns=["date"])
    panel_dates = pd.to_datetime(panel_df["date"]).to_numpy()
    date_to_idx = {pd.Timestamp(d): i for i, d in enumerate(panel_dates)}

    from src.run_experiment import build_history_dates, build_sentiment_histories, load_daily_sentiment

    test_date_arrays = build_history_dates(panel_dates, date_to_idx, dates_test, history_points)
    try:
        sentiment_map = load_daily_sentiment(
            str(sentiment_path),
            date_col=sentiment_cfg.get("date_col", "seendate"),
            score_col=sentiment_cfg.get("score_col", "sent_score"),
            feature_cols=sentiment_feature_cols or None,
        )
    except TypeError:
        sentiment_map = load_daily_sentiment(
            str(sentiment_path),
            date_col=sentiment_cfg.get("date_col", "seendate"),
            score_col=sentiment_cfg.get("score_col", "sent_score"),
        )
    try:
        test_sentiment_histories = build_sentiment_histories(
            test_date_arrays,
            sentiment_map,
            sentiment_points,
            feature_cols=sentiment_feature_cols or None,
        )
    except TypeError:
        test_sentiment_histories = build_sentiment_histories(
            test_date_arrays,
            sentiment_map,
            sentiment_points,
        )
    base_sentiment_path = sentiment_cfg.get("path", "")
    if not base_sentiment_path:
        raise ValueError("Base config is missing llm.sentiment.path; cannot compare sentiment changes.")
    base_sentiment_path = Path(base_sentiment_path)
    if not base_sentiment_path.is_absolute():
        base_sentiment_path = repo_root / base_sentiment_path
    base_sentiment_map = load_daily_sentiment(
        str(base_sentiment_path),
        date_col=sentiment_cfg.get("date_col", "seendate"),
        score_col=sentiment_cfg.get("score_col", "sent_score"),
    )
    try:
        base_test_sentiment_histories = build_sentiment_histories(
            test_date_arrays,
            base_sentiment_map,
            sentiment_points,
            feature_cols=None,
        )
    except TypeError:
        base_test_sentiment_histories = build_sentiment_histories(
            test_date_arrays,
            base_sentiment_map,
            sentiment_points,
        )

    # ---------------------------------------------------------------------
    # Build teaching examples (same algorithm as run_experiment.py)
    # ---------------------------------------------------------------------
    all_dates = np.concatenate([dates_train, dates_val, dates_test])
    all_dates = pd.to_datetime(all_dates).to_numpy()
    all_histories = np.concatenate([y_train_hist, y_val_hist, y_test_hist])
    all_forecasts = np.concatenate([tsm_pred_train_eval, tsm_pred_val_eval, tsm_pred_test_eval])
    all_truth = np.concatenate([y_train_true, y_val_true, y_test_true])
    all_path_mse = np.mean((all_forecasts - all_truth) ** 2, axis=1)

    sorted_idx = np.argsort(all_dates)
    dates_sorted = all_dates[sorted_idx]

    if selection_mode != "similarity":
        raise ValueError(f"This script currently expects example_selection=similarity, got: {selection_mode}")

    all_history_features = np.vstack([_history_features(h, feature_window) for h in all_histories])

    # Cache sentiment histories for example dates (keyed by timestamp string)
    ex_sent_cache: dict[str, np.ndarray] = {}
    ex_sent_cache_base: dict[str, np.ndarray] = {}

    teaching_examples_subset: list[list[dict]] = []
    sentiment_changed_mask = np.zeros(len(dates_test), dtype=bool)
    for sample_idx in range(len(dates_test)):
        sample_changed = args.force_refine_all or not np.array_equal(
            test_sentiment_histories[sample_idx],
            base_test_sentiment_histories[sample_idx],
        )
        test_date = pd.Timestamp(dates_test[sample_idx]).to_datetime64()
        pos = int(np.searchsorted(dates_sorted, test_date, side="left"))
        if pos <= 0:
            teaching_examples_subset.append([])
            sentiment_changed_mask[sample_idx] = sample_changed
            continue

        candidate_indices = sorted_idx[:pos]
        if lookback_days is not None:
            cutoff = np.datetime64(pd.Timestamp(test_date) - pd.Timedelta(days=lookback_days))
            candidate_indices = candidate_indices[all_dates[candidate_indices] >= cutoff]
            if candidate_indices.size == 0:
                candidate_indices = sorted_idx[:pos]

        sample_feat = _history_features(y_test_hist[sample_idx], feature_window)
        cand_feats = all_history_features[candidate_indices]
        distances = np.linalg.norm(cand_feats - sample_feat, axis=1)
        order = np.argsort(distances)[:k_examples]
        example_indices = candidate_indices[order]

        examples: list[dict] = []
        for idx in example_indices:
            ex_date = pd.Timestamp(all_dates[idx])
            ex_key = str(ex_date.normalize())
            if ex_key in ex_sent_cache:
                sent_hist = ex_sent_cache[ex_key]
            else:
                ex_dates = build_history_dates(panel_dates, date_to_idx, np.array([ex_date]), history_points)[0]
                try:
                    sent_hist = build_sentiment_histories(
                        [ex_dates],
                        sentiment_map,
                        sentiment_points,
                        feature_cols=sentiment_feature_cols or None,
                    )[0]
                except TypeError:
                    sent_hist = build_sentiment_histories(
                        [ex_dates],
                        sentiment_map,
                        sentiment_points,
                    )[0]
                ex_sent_cache[ex_key] = sent_hist
            if ex_key in ex_sent_cache_base:
                sent_hist_base = ex_sent_cache_base[ex_key]
            else:
                ex_dates = build_history_dates(panel_dates, date_to_idx, np.array([ex_date]), history_points)[0]
                try:
                    sent_hist_base = build_sentiment_histories(
                        [ex_dates],
                        base_sentiment_map,
                        sentiment_points,
                        feature_cols=None,
                    )[0]
                except TypeError:
                    sent_hist_base = build_sentiment_histories(
                        [ex_dates],
                        base_sentiment_map,
                        sentiment_points,
                    )[0]
                ex_sent_cache_base[ex_key] = sent_hist_base
            if not np.array_equal(sent_hist, sent_hist_base):
                sample_changed = True

            examples.append(
                {
                    "history": all_histories[idx],
                    "forecast": all_forecasts[idx],
                    "truth": all_truth[idx],
                    "date": str(pd.Timestamp(all_dates[idx])),
                    "sentiment_history": sent_hist,
                }
            )
        teaching_examples_subset.append(examples)
        sentiment_changed_mask[sample_idx] = sample_changed

        if (sample_idx + 1) % 50 == 0:
            print(f"Built teaching examples for {sample_idx + 1}/{len(dates_test)} test samples")
    changed_windows = int(sentiment_changed_mask.sum())
    if args.force_refine_all:
        print(f"Force-refine enabled: refining all {len(dates_test)} windows.")
    else:
        print(
            f"Sentiment context changed in {changed_windows}/{len(dates_test)} windows "
            f"({changed_windows / max(1, len(dates_test)):.1%}); unchanged windows will reuse baseline yhat."
        )

    # ---------------------------------------------------------------------
    # Run LLM refinement on the full test split
    # ---------------------------------------------------------------------
    yhat_base = np.asarray(z_base["yhat"], dtype=float)
    # Override LLM sentiment path for traceability (it is not used by LLMRefiner directly).
    llm_cfg = dict(llm_cfg)
    llm_cfg["sentiment"] = dict(llm_cfg.get("sentiment", {}) or {})
    llm_cfg["sentiment"]["path"] = str(sentiment_path)
    if args.strict_json_prompt:
        llm_cfg["cot_rf"] = dict(llm_cfg.get("cot_rf", {}) or {})
        llm_cfg["cot_rf"]["strict_json_prompt"] = True
    if args.paper_replication_mode:
        llm_cfg["prompt_history_points"] = 18
        llm_cfg["prompt_sentiment_points"] = 18
        llm_cfg["history_points"] = 18
        llm_cfg["sentiment"]["history_points"] = 18
        llm_cfg["cot_rf"] = dict(llm_cfg.get("cot_rf", {}) or {})
        llm_cfg["cot_rf"]["paper_replication_mode"] = True
        llm_cfg["cot_rf"]["retain_context"] = True

    from src.llm import LLMRefiner
    histories_subset = y_test_hist[:, -history_points:]
    changed_idx = np.flatnonzero(sentiment_changed_mask)
    preds = np.asarray(yhat_base, dtype=float).copy()
    metadata = [{"reused_baseline": True} for _ in range(len(histories_subset))]
    if len(changed_idx):
        workers = max(1, int(args.workers))
        if workers == 1:
            refiner = LLMRefiner(
                llm_cfg,
                cache_dir=out_run / "llm" / "cache",
                log_dir=out_run / "llm" / "logs",
            )
            changed_preds, changed_metadata = refiner.refine_batch(
                method=method,
                histories=histories_subset[changed_idx],
                date_arrays=[test_date_arrays[int(i)] for i in changed_idx],
                tsm_forecasts=tsm_pred_test_eval[changed_idx],
                pred_len=pred_len,
                exogenous_summaries=[None for _ in range(len(changed_idx))],
                price_bases=y_test_base[changed_idx],
                teaching_examples=[teaching_examples_subset[int(i)] for i in changed_idx],
                sentiment_histories=[test_sentiment_histories[int(i)] for i in changed_idx],
            )
            preds[changed_idx] = changed_preds
            for local_i, global_i in enumerate(changed_idx):
                md = dict(changed_metadata[local_i]) if isinstance(changed_metadata[local_i], dict) else {}
                md["reused_baseline"] = False
                metadata[int(global_i)] = md
        else:
            print(f"Running parallel refinement with workers={workers}")
            thread_state = threading.local()

            def _get_thread_refiner() -> LLMRefiner:
                refiner = getattr(thread_state, "refiner", None)
                if refiner is None:
                    # Avoid log-file contention in threaded mode.
                    refiner = LLMRefiner(
                        llm_cfg,
                        cache_dir=out_run / "llm" / "cache",
                        log_dir=None,
                    )
                    thread_state.refiner = refiner
                return refiner

            def _run_one(global_i: int):
                i = int(global_i)
                refiner = _get_thread_refiner()
                forecast_i, metadata_i = refiner.refine(
                    method=method,
                    history=histories_subset[i],
                    dates=test_date_arrays[i],
                    tsm_forecast=tsm_pred_test_eval[i],
                    pred_len=pred_len,
                    exogenous_summary=None,
                    price_base=y_test_base[i],
                    teaching_examples=teaching_examples_subset[i],
                    sentiment_history=test_sentiment_histories[i],
                )
                md = dict(metadata_i) if isinstance(metadata_i, dict) else {}
                if forecast_i is None:
                    forecast_i = tsm_pred_test_eval[i]
                    md["fallback"] = True
                md["reused_baseline"] = False
                return i, np.asarray(forecast_i, dtype=float), md

            with ThreadPoolExecutor(max_workers=workers) as ex:
                futures = {ex.submit(_run_one, int(i)): int(i) for i in changed_idx}
                total = len(futures)
                for done_count, fut in enumerate(as_completed(futures), start=1):
                    i = futures[fut]
                    try:
                        idx, pred_i, md = fut.result()
                    except Exception as e:
                        idx = i
                        pred_i = np.asarray(tsm_pred_test_eval[idx], dtype=float)
                        md = {"fallback": True, "error": str(e), "reused_baseline": False}
                    preds[idx] = pred_i
                    metadata[idx] = md
                    if done_count % 25 == 0 or done_count == total:
                        print(f"Parallel refined {done_count}/{total} windows")
    print(f"Reused baseline predictions for {len(histories_subset) - len(changed_idx)} windows; refined {len(changed_idx)} windows.")

    # ---------------------------------------------------------------------
    # Compare against baseline run's saved predictions
    # ---------------------------------------------------------------------
    if y_true_base.shape != preds.shape:
        raise ValueError(f"Shape mismatch: baseline y_true {y_true_base.shape} vs new preds {preds.shape}")

    from src.eval.metrics import compute_path_metrics
    from src.run_experiment import mse_by_horizon

    base_path = compute_path_metrics(y_true_base, yhat_base).get("mse_path")
    new_path = compute_path_metrics(y_true_base, preds).get("mse_path")

    base_h = mse_by_horizon(y_true_base, yhat_base)
    new_h = mse_by_horizon(y_true_base, preds)

    def _trend_accuracy(y_true: np.ndarray, y_pred: np.ndarray, y_base: np.ndarray, hs: list[int]) -> dict:
        out = {}
        for h in hs:
            idx = int(h) - 1
            if idx < 0 or idx >= y_true.shape[1]:
                continue
            true_sign = np.sign(y_true[:, idx] - y_base)
            pred_sign = np.sign(y_pred[:, idx] - y_base)
            out[f"d{h}"] = float(np.mean(pred_sign == true_sign))
        return out

    trend_base = _trend_accuracy(y_true_base, yhat_base, y_test_base, [10, 20, 30])
    trend_new = _trend_accuracy(y_true_base, preds, y_test_base, [10, 20, 30])

    # Optional: apply the baseline run's selected blend weight (so the comparison
    # matches the way results are usually reported).
    blend_sel_path = base_run / "llm" / "blend_grid_selection_TSM_LLM-COT-SENT-RF.json"
    blend_summary = None
    if blend_sel_path.exists():
        with open(blend_sel_path, "r") as f:
            blend_sel = json.load(f)
        best_w = float(blend_sel.get("best_w_path", blend_sel.get("w", 0.0)))
        schedule = str((llm_cfg.get("blend_grid", {}) or {}).get("schedule", "ramp")).lower()
        min_w = float((llm_cfg.get("blend_grid", {}) or {}).get("min_weight", 0.0))
        power = float((llm_cfg.get("blend_grid", {}) or {}).get("power", 1.0))

        from src.run_experiment import blend_forecasts

        base_blend = blend_forecasts(tsm_base, yhat_base, best_w, schedule, pred_len, min_weight=min_w, power=power)
        new_blend = blend_forecasts(tsm_pred_test_eval, preds, best_w, schedule, pred_len, min_weight=min_w, power=power)
        base_blend_path = compute_path_metrics(y_true_base, base_blend).get("mse_path")
        new_blend_path = compute_path_metrics(y_true_base, new_blend).get("mse_path")

        blend_summary = {
            "best_w_path": best_w,
            "schedule": schedule,
            "min_weight": min_w,
            "power": power,
            "base_mse_path": float(base_blend_path),
            "new_mse_path": float(new_blend_path),
        }

    def _sel(h: np.ndarray, hs: list[int]) -> dict:
        out = {}
        for x in hs:
            idx = int(x) - 1
            if 0 <= idx < len(h):
                out[f"h{x}"] = float(h[idx])
        return out

    summary = {
        "base_run": str(base_run),
        "sentiment_path": str(sentiment_path),
        "method": method,
        "n_test": int(len(preds)),
        "n_refined_windows": int(len(changed_idx)),
        "n_reused_baseline_windows": int(len(preds) - len(changed_idx)),
        "force_refine_all": bool(args.force_refine_all),
        "paper_replication_mode": bool(args.paper_replication_mode),
        "mse_path_base": float(base_path),
        "mse_path_new": float(new_path),
        "mse_horizons_base": _sel(base_h, [1, 5, 20, 30]),
        "mse_horizons_new": _sel(new_h, [1, 5, 20, 30]),
        "trend_accuracy_base": trend_base,
        "trend_accuracy_new": trend_new,
        "blend_comparison": blend_summary,
    }

    with open(out_run / "results" / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Save new subset predictions in the same shape as run_experiment.
    np.savez_compressed(
        out_run / "predictions" / f"{method}_pred_test_subset.npz",
        method=np.array([method]),
        eval_indices=np.arange(len(preds), dtype=int),
        dates=np.asarray([str(d) for d in dates_test], dtype="U"),
        y_true=np.asarray(y_true_base, dtype=float),
        tsm_pred=np.asarray(tsm_pred_test_eval, dtype=float),
        yhat=np.asarray(preds, dtype=float),
    )

    # Persist metadata (can be large; keep it compact)
    with open(out_run / "results" / "llm_metadata.jsonl", "w") as f:
        for m in metadata:
            f.write(json.dumps(m) + "\n")

    print("\nSummary:")
    print(json.dumps(summary, indent=2))
    print(f"\nWrote: {out_run}")


if __name__ == "__main__":
    main()
