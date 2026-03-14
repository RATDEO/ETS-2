#!/usr/bin/env python3
"""
Run a leakage-safe paper-replication validation path.

This keeps the production gate untouched and evaluates a small set of
paper-aligned CoT-SENT-RF candidates on rolling validation folds before
optionally running the selected candidate on the held-out test split.
"""

from __future__ import annotations

import argparse
import copy
import json
from datetime import datetime
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from data.windows import StandardScaler, TimeSeriesDataset, WindowConfig, make_windows, split_windows
from eval.paper_replication import (
    PaperReplicationCandidate,
    aggregate_fold_rows,
    build_outer_folds,
    score_paper_candidate,
)
from llm.refine import LLMRefiner
from models.tsm import TSMForecaster
from run_experiment import build_history_dates, build_sentiment_histories, load_daily_sentiment, returns_to_prices
from news.official_event_features import build_official_event_summary_map, build_official_origin_feature_frame

try:
    from torch.utils.data import DataLoader
except ImportError as exc:  # pragma: no cover
    raise SystemExit("PyTorch is required for paper replication v2.") from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the paper_replication_v2 validation/test path.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("src/config/paper_replication_v2.yaml"),
        help="Replication config yaml.",
    )
    parser.add_argument(
        "--base-run",
        type=Path,
        default=None,
        help="Override base clean run directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Where to write outputs. Defaults under reports/paper_replication_v2/<timestamp>.",
    )
    parser.add_argument(
        "--llm-api-key",
        type=str,
        default=None,
        help="LLM API key. Falls back to config then deo.",
    )
    parser.add_argument(
        "--llm-base-url",
        type=str,
        default=None,
        help="Optional base URL override applied to all candidates.",
    )
    parser.add_argument(
        "--profile",
        type=str,
        default=None,
        help="Configured candidate profile name. Ignored if --candidate-names is provided.",
    )
    parser.add_argument(
        "--candidate-names",
        type=str,
        default=None,
        help="Comma-separated candidate names to run. Defaults to all configured candidates.",
    )
    parser.add_argument(
        "--max-samples-per-fold",
        type=int,
        default=None,
        help="Override fold sample cap.",
    )
    parser.add_argument(
        "--skip-selected-test",
        action="store_true",
        help="Run validation folds only and skip selected-candidate held-out test.",
    )
    parser.add_argument(
        "--test-max-samples",
        type=int,
        default=None,
        help="Optional cap for selected-candidate test evaluation. Defaults to config value.",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=None,
        help="Override LLM timeout for this replication run.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=None,
        help="Override LLM max_retries for this replication run.",
    )
    return parser.parse_args()


def _resolve_path(path: Path | str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else (ROOT / p)


def _predict_returns(
    tsm: TSMForecaster,
    scaler: StandardScaler,
    x_enc: np.ndarray,
    x_dec: np.ndarray,
    y: np.ndarray,
    batch_size: int = 32,
) -> np.ndarray:
    dataset = TimeSeriesDataset(
        scaler.transform(x_enc),
        scaler.transform(x_dec),
        (y - scaler.mean_[0]) / scaler.std_[0],
    )
    loader = DataLoader(dataset, batch_size=batch_size)
    pred_scaled, _ = tsm.predict(loader)
    return scaler.inverse_transform_target(pred_scaled)


def _history_features(history: np.ndarray, feature_window: int) -> np.ndarray:
    window = history[-feature_window:] if len(history) >= feature_window else history
    mean = float(np.mean(window)) if len(window) else 0.0
    std = float(np.std(window)) if len(window) else 0.0
    if len(window) > 1:
        slope = float(np.polyfit(np.arange(len(window)), window, 1)[0])
    else:
        slope = 0.0
    return np.array([mean, std, slope], dtype=float)


def _build_retrieval_feature_map(
    panel_dates: np.ndarray,
    panel_prices: np.ndarray,
    origin_dates: np.ndarray,
    sentiment_map: dict[pd.Timestamp, float],
    official_event_path: str | None = None,
    *,
    sentiment_fill_limit: int = 5,
    sentiment_window: int = 3,
    return_window: int = 5,
    volatility_window: int = 20,
) -> tuple[dict[pd.Timestamp, np.ndarray], int]:
    price_df = pd.DataFrame(
        {
            "date": pd.to_datetime(panel_dates).normalize(),
            "close_eur": pd.to_numeric(panel_prices, errors="coerce"),
        }
    )
    price_df = price_df.dropna(subset=["date", "close_eur"]).drop_duplicates("date").sort_values("date")

    sentiment_df = pd.DataFrame(
        {
            "date": pd.to_datetime(list(sentiment_map.keys())).normalize(),
            "sent_score": list(sentiment_map.values()),
        }
    )
    if sentiment_df.empty:
        sentiment_df = pd.DataFrame({"date": price_df["date"], "sent_score": 0.0})
    else:
        sentiment_df["sent_score"] = pd.to_numeric(sentiment_df["sent_score"], errors="coerce").fillna(0.0)
        sentiment_df = sentiment_df.dropna(subset=["date"]).drop_duplicates("date").sort_values("date")

    feature_df = price_df.merge(sentiment_df, on="date", how="left").sort_values("date").reset_index(drop=True)
    feature_df["sent_score"] = feature_df["sent_score"].ffill(limit=int(sentiment_fill_limit)).fillna(0.0)
    feature_df["sent_score_3d_ma"] = (
        feature_df["sent_score"].rolling(window=int(sentiment_window), min_periods=1).mean().fillna(0.0)
    )
    feature_df["ret_5d"] = feature_df["close_eur"].pct_change(int(return_window)).fillna(0.0)
    feature_df["hist_vol_20d"] = (
        np.log(feature_df["close_eur"]).diff().rolling(window=int(volatility_window), min_periods=2).std().fillna(0.0)
    )

    feature_cols = ["sent_score_3d_ma", "ret_5d", "hist_vol_20d"]
    origin_index = pd.DatetimeIndex(pd.to_datetime(origin_dates)).normalize()
    origin_frame = feature_df.set_index("date").reindex(origin_index).reset_index().rename(columns={"index": "date"})
    if official_event_path:
        official_df, _ = build_official_origin_feature_frame(origin_index, official_event_path)
        official_df["date"] = pd.to_datetime(official_df["date"]).dt.normalize()
        official_cols = [c for c in official_df.columns if c not in {"date", "official_supply_regime_30d"}]
        origin_frame = origin_frame.merge(official_df[["date"] + official_cols], on="date", how="left")
        origin_frame[official_cols] = origin_frame[official_cols].fillna(0.0)
        feature_cols.extend(official_cols)

    feature_frame = origin_frame[["date"] + feature_cols].copy()
    feature_frame[feature_cols] = feature_frame[feature_cols].replace([np.inf, -np.inf], 0.0).fillna(0.0)
    feature_map = {
        pd.Timestamp(row["date"]).normalize(): row[feature_cols].to_numpy(dtype=float)
        for _, row in feature_frame.iterrows()
    }
    return feature_map, len(feature_cols)


def _prepare_example_pool(
    pool_dates: np.ndarray,
    pool_histories: np.ndarray,
    pool_forecasts: np.ndarray,
    pool_truth: np.ndarray,
    selection_mode: str,
    feature_window: int,
    retrieval_feature_map: dict[pd.Timestamp, np.ndarray] | None = None,
    retrieval_feature_width: int = 0,
) -> dict[str, Any]:
    dates_arr = pd.to_datetime(pool_dates).to_numpy()
    order = np.argsort(dates_arr)
    prepared: dict[str, Any] = {
        "dates": dates_arr[order],
        "histories": pool_histories[order],
        "forecasts": pool_forecasts[order],
        "truth": pool_truth[order],
    }
    prepared["path_mse"] = np.mean((prepared["forecasts"] - prepared["truth"]) ** 2, axis=1)
    if selection_mode in {"similarity", "event_similarity"}:
        history_features = np.vstack([_history_features(hist, feature_window) for hist in prepared["histories"]])
        prepared["history_features"] = history_features
        if selection_mode == "event_similarity":
            width = int(retrieval_feature_width)
            zero = np.zeros(width, dtype=float)
            retrieval_features = np.vstack(
                [
                    np.asarray(
                        (retrieval_feature_map or {}).get(pd.Timestamp(date).normalize(), zero),
                        dtype=float,
                    )
                    for date in prepared["dates"]
                ]
            )
            combined = np.hstack([history_features, retrieval_features])
            means = combined.mean(axis=0)
            stds = combined.std(axis=0)
            stds = np.where(stds > 1e-8, stds, 1.0)
            prepared["event_similarity_features"] = (combined - means) / stds
            prepared["event_similarity_means"] = means
            prepared["event_similarity_stds"] = stds
            prepared["retrieval_feature_width"] = width
    return prepared


def _select_example_indices(
    pool: dict[str, Any],
    reference_date: Any,
    reference_history: np.ndarray,
    selection_mode: str,
    k_examples: int,
    feature_window: int,
    lookback_days: int | None,
    reference_retrieval_features: np.ndarray | None = None,
) -> np.ndarray:
    ref_ts = pd.Timestamp(reference_date).to_datetime64()
    pos = np.searchsorted(pool["dates"], ref_ts, side="left")
    if pos <= 0:
        return np.array([], dtype=int)

    candidate_indices = np.arange(pos, dtype=int)
    if selection_mode in ("recent_high_error", "similarity", "event_similarity") and lookback_days is not None:
        cutoff = np.datetime64(pd.Timestamp(ref_ts) - pd.Timedelta(days=lookback_days))
        candidate_indices = candidate_indices[pool["dates"][candidate_indices] >= cutoff]
        if candidate_indices.size == 0:
            candidate_indices = np.arange(pos, dtype=int)

    if selection_mode in ("high_error", "recent_high_error"):
        cand_errors = pool["path_mse"][candidate_indices]
        if len(candidate_indices) > k_examples:
            order = np.argsort(cand_errors)[-k_examples:]
            picked = candidate_indices[order]
        else:
            picked = candidate_indices
        return picked[np.argsort(pool["dates"][picked])]

    if selection_mode == "similarity":
        sample_feat = _history_features(reference_history, feature_window)
        cand_feats = pool["history_features"][candidate_indices]
        distances = np.linalg.norm(cand_feats - sample_feat, axis=1)
        order = np.argsort(distances)[:k_examples]
        picked = candidate_indices[order]
        return picked[np.argsort(pool["dates"][picked])]

    if selection_mode == "event_similarity":
        width = int(pool.get("retrieval_feature_width", 0))
        if reference_retrieval_features is None:
            reference_retrieval_features = np.zeros(width, dtype=float)
        sample_feat = np.concatenate(
            [
                _history_features(reference_history, feature_window),
                np.asarray(reference_retrieval_features, dtype=float),
            ]
        )
        sample_feat = (sample_feat - pool["event_similarity_means"]) / pool["event_similarity_stds"]
        cand_feats = pool["event_similarity_features"][candidate_indices]
        distances = np.linalg.norm(cand_feats - sample_feat, axis=1)
        order = np.argsort(distances)[:k_examples]
        picked = candidate_indices[order]
        return picked[np.argsort(pool["dates"][picked])]

    start = max(0, pos - k_examples)
    return np.arange(start, pos, dtype=int)


def _build_examples(
    pool: dict[str, Any],
    example_indices: np.ndarray,
    panel_dates: np.ndarray,
    date_to_idx: dict[pd.Timestamp, int],
    sentiment_map: dict[pd.Timestamp, float],
    history_points: int,
    sentiment_points: int,
    official_summary_map: dict[pd.Timestamp, dict[str, str]] | None = None,
) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    for idx in example_indices:
        ex_date = pd.Timestamp(pool["dates"][idx])
        ex_dates = build_history_dates(panel_dates, date_to_idx, np.array([ex_date]), history_points)[0]
        sent_hist = build_sentiment_histories([ex_dates], sentiment_map, sentiment_points)[0]
        examples.append(
            {
                "history": pool["histories"][idx],
                "forecast": pool["forecasts"][idx],
                "truth": pool["truth"][idx],
                "date": str(ex_date),
                "sentiment_history": sent_hist,
                "exogenous_summary": (
                    official_summary_map.get(ex_date.normalize()) if official_summary_map is not None else None
                ),
            }
        )
    return examples


def _load_candidates(cfg: dict[str, Any], requested_names: set[str] | None) -> list[PaperReplicationCandidate]:
    candidates = []
    for raw in cfg.get("candidates", []):
        candidate = PaperReplicationCandidate(**raw)
        if requested_names and candidate.name not in requested_names:
            continue
        candidates.append(candidate)
    if not candidates:
        raise ValueError("No paper_replication_v2 candidates selected.")
    return candidates


def _resolve_requested_names(
    rep_cfg: dict[str, Any],
    profile_name: str | None,
    candidate_names_arg: str | None,
) -> tuple[set[str] | None, str | None]:
    if candidate_names_arg:
        requested_names = {name.strip() for name in candidate_names_arg.split(",") if name.strip()}
        return requested_names, None

    profiles = dict(rep_cfg.get("profiles", {}) or {})
    resolved_profile = profile_name or rep_cfg.get("default_profile")
    if not resolved_profile:
        return None, None
    if resolved_profile not in profiles:
        known = ", ".join(sorted(profiles)) or "<none>"
        raise ValueError(f"Unknown paper_replication_v2 profile '{resolved_profile}'. Known profiles: {known}")
    requested_names = {str(name).strip() for name in profiles[resolved_profile] if str(name).strip()}
    if not requested_names:
        raise ValueError(f"paper_replication_v2 profile '{resolved_profile}' does not contain any candidates.")
    return requested_names, str(resolved_profile)


def _candidate_llm_config(
    base_llm_cfg: dict[str, Any],
    candidate: PaperReplicationCandidate,
    api_key: str | None,
    base_url_override: str | None,
    timeout_seconds: int | None = None,
    max_retries: int | None = None,
) -> dict[str, Any]:
    llm_cfg = copy.deepcopy(base_llm_cfg)
    llm_cfg["methods"] = [candidate.method]
    llm_cfg["history_points"] = int(candidate.history_points)
    llm_cfg["prompt_history_points"] = int(candidate.prompt_history_points)
    llm_cfg["prompt_sentiment_points"] = int(candidate.prompt_sentiment_points)
    llm_cfg["api_key"] = api_key or llm_cfg.get("api_key") or "deo"
    if candidate.base_url:
        llm_cfg["base_url"] = candidate.base_url
    elif base_url_override:
        llm_cfg["base_url"] = base_url_override
    if candidate.model:
        llm_cfg["model"] = candidate.model
    if timeout_seconds is not None:
        llm_cfg["timeout_seconds"] = int(timeout_seconds)
    if max_retries is not None:
        llm_cfg["max_retries"] = int(max_retries)
    cot_rf_cfg = dict(llm_cfg.get("cot_rf", {}) or {})
    cot_rf_cfg["k_examples"] = int(candidate.k_examples)
    cot_rf_cfg["example_selection"] = str(candidate.example_selection)
    cot_rf_cfg["feature_window"] = int(candidate.feature_window)
    cot_rf_cfg["lookback_days"] = (
        int(candidate.lookback_days) if candidate.lookback_days is not None else None
    )
    cot_rf_cfg["retain_context"] = bool(candidate.retain_context)
    cot_rf_cfg["strict_json_prompt"] = bool(candidate.strict_json_prompt)
    cot_rf_cfg["strict_json_response_format"] = bool(candidate.strict_json_response_format)
    llm_cfg["cot_rf"] = cot_rf_cfg
    sentiment_cfg = dict(llm_cfg.get("sentiment", {}) or {})
    sentiment_cfg["enabled"] = True
    sentiment_cfg["path"] = str(_resolve_path(candidate.sentiment_path))
    sentiment_cfg["history_points"] = int(candidate.sentiment_points)
    llm_cfg["sentiment"] = sentiment_cfg
    if candidate.method == "TSM+LLM-COT-SENT-RF-HDELTA":
        hdelta_cfg = dict(llm_cfg.get("hdelta", {}) or {})
        hdelta_cfg["key_horizons"] = [int(h) for h in candidate.hdelta_key_horizons]
        hdelta_cfg["max_adjustment_pct"] = float(candidate.hdelta_max_adjustment_pct or 3.0)
        hdelta_cfg["freeze_horizons"] = [int(h) for h in candidate.hdelta_freeze_horizons]
        hdelta_cfg["sentiment_secondary"] = bool(candidate.hdelta_sentiment_secondary)
        llm_cfg["hdelta"] = hdelta_cfg
    if candidate.method == "TSM+LLM-COT-SENT-RF-HPRICE":
        hprice_cfg = dict(llm_cfg.get("hprice", {}) or {})
        hprice_cfg["key_horizons"] = [int(h) for h in candidate.hprice_key_horizons]
        hprice_cfg["max_adjustment_pct"] = float(candidate.hprice_max_adjustment_pct or 1.0)
        hprice_cfg["freeze_horizons"] = [int(h) for h in candidate.hprice_freeze_horizons]
        hprice_cfg["sentiment_secondary"] = bool(candidate.hprice_sentiment_secondary)
        llm_cfg["hprice"] = hprice_cfg
    llm_cfg["blend_grid"] = {"enabled": False}
    llm_cfg["calibrate_blend"] = {"enabled": False}
    return llm_cfg


def _reuse_saved_test_artifact(
    base_run: Path,
    candidate: PaperReplicationCandidate,
    base_llm_cfg: dict[str, Any],
    test_histories: np.ndarray,
    paper_alpha: float,
    paper_history_window: int,
    paper_horizons: list[int],
) -> dict[str, float | str] | None:
    sentiment_cfg = dict(base_llm_cfg.get("sentiment", {}) or {})
    base_sentiment = str(_resolve_path(sentiment_cfg.get("path", "data/news/daily_sentiment.csv")))
    cot_rf_cfg = dict(base_llm_cfg.get("cot_rf", {}) or {})
    if candidate.method != "TSM+LLM-COT-SENT-RF":
        return None
    if _resolve_path(candidate.sentiment_path).as_posix() != Path(base_sentiment).as_posix():
        return None
    if int(candidate.history_points) != int(base_llm_cfg.get("history_points", 18)):
        return None
    if int(candidate.prompt_history_points) != int(base_llm_cfg.get("prompt_history_points", 18)):
        return None
    if int(candidate.prompt_sentiment_points) != int(base_llm_cfg.get("prompt_sentiment_points", 18)):
        return None
    if int(candidate.k_examples) != int(cot_rf_cfg.get("k_examples", 5)):
        return None
    if str(candidate.example_selection) != str(cot_rf_cfg.get("example_selection", "similarity")):
        return None
    if int(candidate.feature_window) != int(cot_rf_cfg.get("feature_window", 18)):
        return None
    base_lookback = cot_rf_cfg.get("lookback_days")
    if (int(candidate.lookback_days) if candidate.lookback_days is not None else None) != (
        int(base_lookback) if base_lookback is not None else None
    ):
        return None
    if bool(candidate.retain_context) != bool(cot_rf_cfg.get("retain_context", False)):
        return None
    if bool(candidate.strict_json_prompt) != bool(cot_rf_cfg.get("strict_json_prompt", False)):
        return None

    npz_path = base_run / "predictions" / "TSM+LLM-COT-SENT-RF_pred_test_subset.npz"
    if not npz_path.exists():
        return None

    z = np.load(npz_path, allow_pickle=True)
    y_true = np.asarray(z["y_true"], dtype=float)
    y_pred = np.asarray(z["yhat"], dtype=float)
    if y_true.shape[0] != test_histories.shape[0]:
        return None
    return score_paper_candidate(
        candidate.name,
        test_histories,
        y_true,
        y_pred,
        paper_alpha=paper_alpha,
        paper_history_window=paper_history_window,
        paper_horizons=paper_horizons,
    )


def main() -> int:
    args = parse_args()
    cfg_path = _resolve_path(args.config)
    with cfg_path.open("r") as handle:
        cfg = yaml.safe_load(handle)

    rep_cfg = dict(cfg.get("paper_replication_v2", {}) or {})
    base_run_path = args.base_run or rep_cfg.get("base_run", "runs/20260228_145058_40d91f")
    base_run = _resolve_path(base_run_path)
    if not base_run.exists():
        raise FileNotFoundError(f"Base run not found: {base_run}")

    if args.output_dir is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = ROOT / "reports" / "paper_replication_v2" / stamp
    else:
        output_dir = _resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    requested_names, resolved_profile = _resolve_requested_names(
        rep_cfg,
        args.profile,
        args.candidate_names,
    )
    candidates = _load_candidates(rep_cfg, requested_names)
    timeout_seconds = (
        int(args.timeout_seconds)
        if args.timeout_seconds is not None
        else (int(rep_cfg["timeout_seconds"]) if rep_cfg.get("timeout_seconds") is not None else None)
    )
    max_retries = (
        int(args.max_retries)
        if args.max_retries is not None
        else (int(rep_cfg["max_retries"]) if rep_cfg.get("max_retries") is not None else None)
    )

    with (base_run / "config_resolved.yaml").open("r") as handle:
        base_run_cfg = yaml.safe_load(handle)

    llm_api_key = args.llm_api_key or rep_cfg.get("llm_api_key") or base_run_cfg.get("llm", {}).get("api_key") or "deo"
    llm_base_url = args.llm_base_url or rep_cfg.get("llm_base_url")

    target_mode = str((base_run_cfg.get("target", {}) or {}).get("mode", "price"))
    if target_mode != "returns":
        raise ValueError("paper_replication_v2 currently expects target.mode=returns.")

    time_cfg = base_run_cfg.get("time_series", {}) or {}
    pred_len = int(time_cfg.get("pred_len", 30))
    seq_len = int(time_cfg.get("seq_len", 60))
    label_len = int(time_cfg.get("label_len", 15))
    target_col = "y_return"

    panel = pd.read_parquet(base_run / "data" / "panel.parquet")
    feature_candidates = [c for c in panel.columns if c not in ["date", target_col]]
    if "y" in feature_candidates:
        feature_candidates.remove("y")
        feature_candidates = ["y"] + feature_candidates
    feature_cols = [target_col] + feature_candidates[:10]
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
    split_cfg = base_run_cfg.get("split", {}) or {}
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
        "time_series": dict(base_run_cfg.get("time_series", {}) or {}),
        "model": dict(base_run_cfg.get("model", {}) or {}),
    }
    tsm_config["model"]["enc_in"] = int(x_train_enc.shape[-1])
    tsm_config["model"]["dec_in"] = int(x_train_dec.shape[-1])
    tsm = TSMForecaster(tsm_config, device="auto")
    tsm.load(base_run / "models" / "tsm_checkpoint.pt")

    print("[TSM] Recomputing train/val/test forecasts from saved checkpoint", flush=True)
    tsm_pred_train = returns_to_prices(_predict_returns(tsm, scaler, x_train_enc, x_train_dec, y_train), y_train_base)
    tsm_pred_val = returns_to_prices(_predict_returns(tsm, scaler, x_val_enc, x_val_dec, y_val), y_val_base)
    tsm_pred_test = returns_to_prices(_predict_returns(tsm, scaler, x_test_enc, x_test_dec, y_test), y_test_base)

    panel_dates = pd.to_datetime(panel["date"]).to_numpy()
    panel_prices = pd.to_numeric(panel["y"], errors="coerce").to_numpy()
    date_to_idx = {pd.Timestamp(d): i for i, d in enumerate(panel_dates)}

    n_folds = int(rep_cfg.get("outer_folds", 3))
    max_samples_per_fold = args.max_samples_per_fold
    if max_samples_per_fold is None:
        max_samples_per_fold = rep_cfg.get("max_samples_per_fold")
    fold_indices = build_outer_folds(len(dates_val), n_folds=n_folds, max_samples_per_fold=max_samples_per_fold)

    eval_cfg = dict(base_run_cfg.get("evaluation", {}) or {})
    paper_trend_cfg = dict(eval_cfg.get("paper_trend", {}) or {})
    paper_alpha = float(paper_trend_cfg.get("alpha", 0.02))
    paper_history_window = int(paper_trend_cfg.get("history_window", 18))
    paper_horizons = [int(h) for h in paper_trend_cfg.get("horizons", [10, 20, 30])]

    with (output_dir / "config_resolved.yaml").open("w") as handle:
        yaml.safe_dump(
            {
                "paper_replication_v2": rep_cfg,
                "base_run": str(base_run),
                "selected_profile": resolved_profile,
                "candidates": [candidate.__dict__ for candidate in candidates],
            },
            handle,
            sort_keys=False,
        )

    sentiment_cache: dict[str, dict[pd.Timestamp, float]] = {}
    official_summary_cache: dict[tuple[str, int, int], dict[pd.Timestamp, dict[str, str]]] = {}
    candidate_rows: list[dict[str, Any]] = []
    error_rows: list[dict[str, Any]] = []

    for candidate in candidates:
        print(f"[paper_replication_v2] candidate={candidate.name}", flush=True)
        llm_cfg = _candidate_llm_config(
            base_run_cfg.get("llm", {}) or {},
            candidate,
            llm_api_key,
            llm_base_url,
            timeout_seconds=timeout_seconds,
            max_retries=max_retries,
        )
        candidate_sentiment_path = str(_resolve_path(candidate.sentiment_path))
        sentiment_map = sentiment_cache.get(candidate_sentiment_path)
        if sentiment_map is None:
            sentiment_map = load_daily_sentiment(
                candidate_sentiment_path,
                date_col=llm_cfg["sentiment"].get("date_col", "seendate"),
                score_col=llm_cfg["sentiment"].get("score_col", "sent_score"),
            )
            sentiment_cache[candidate_sentiment_path] = sentiment_map
        retrieval_feature_map = None
        retrieval_feature_width = 0
        if candidate.example_selection == "event_similarity":
            retrieval_event_path = candidate.retrieval_official_event_path or candidate.official_event_path
            if retrieval_event_path is None:
                raise ValueError(
                    f"candidate '{candidate.name}' uses example_selection=event_similarity without "
                    "retrieval_official_event_path or official_event_path."
                )
            all_origin_dates = np.concatenate([dates_train, dates_val, dates_test])
            retrieval_feature_map, retrieval_feature_width = _build_retrieval_feature_map(
                panel_dates,
                panel_prices,
                all_origin_dates,
                sentiment_map,
                str(_resolve_path(retrieval_event_path)),
            )
        official_summary_map = None
        if candidate.official_event_path:
            official_key = (
                str(_resolve_path(candidate.official_event_path)),
                int(candidate.official_event_lookback_days),
                int(candidate.official_event_max_titles),
            )
            official_summary_map = official_summary_cache.get(official_key)
            if official_summary_map is None:
                all_origin_dates = np.concatenate([dates_train, dates_val, dates_test])
                official_summary_map = build_official_event_summary_map(
                    official_key[0],
                    all_origin_dates,
                    lookback_days=int(candidate.official_event_lookback_days),
                    max_titles=int(candidate.official_event_max_titles),
                )
                official_summary_cache[official_key] = official_summary_map

        candidate_cache_dir = output_dir / "llm_cache" / candidate.name
        candidate_log_dir = output_dir / "llm_logs" / candidate.name
        refiner = LLMRefiner(llm_cfg, cache_dir=candidate_cache_dir, log_dir=candidate_log_dir)

        val_date_arrays = build_history_dates(panel_dates, date_to_idx, dates_val, candidate.history_points)
        val_sentiment_histories = build_sentiment_histories(
            val_date_arrays,
            sentiment_map,
            candidate.sentiment_points,
        )
        val_exogenous_summaries = (
            [official_summary_map.get(pd.Timestamp(d).normalize()) for d in dates_val]
            if official_summary_map is not None
            else [None for _ in dates_val]
        )

        val_pool = _prepare_example_pool(
            np.concatenate([dates_train, dates_val]),
            np.concatenate([y_train_hist, y_val_hist]),
            np.concatenate([tsm_pred_train, tsm_pred_val]),
            np.concatenate([y_train_true, y_val_true]),
            candidate.example_selection,
            candidate.feature_window,
            retrieval_feature_map=retrieval_feature_map,
            retrieval_feature_width=retrieval_feature_width,
        )

        for fold_idx, eval_indices in enumerate(fold_indices, start=1):
            if len(eval_indices) == 0:
                continue
            teaching_examples = []
            for sample_idx in eval_indices:
                example_indices = _select_example_indices(
                    val_pool,
                    dates_val[sample_idx],
                    y_val_hist[sample_idx],
                    candidate.example_selection,
                    candidate.k_examples,
                    candidate.feature_window,
                    candidate.lookback_days,
                    reference_retrieval_features=(
                        retrieval_feature_map.get(pd.Timestamp(dates_val[sample_idx]).normalize())
                        if retrieval_feature_map is not None
                        else None
                    ),
                )
                teaching_examples.append(
                    _build_examples(
                        val_pool,
                        example_indices,
                        panel_dates,
                        date_to_idx,
                        sentiment_map,
                        candidate.history_points,
                        candidate.sentiment_points,
                        official_summary_map=official_summary_map,
                    )
                )

            try:
                predictions, _ = refiner.refine_batch(
                    method=candidate.method,
                    histories=y_val_hist[eval_indices],
                    date_arrays=[val_date_arrays[i] for i in eval_indices],
                    tsm_forecasts=tsm_pred_val[eval_indices],
                    pred_len=pred_len,
                    exogenous_summaries=[val_exogenous_summaries[i] for i in eval_indices],
                    price_bases=None,
                    teaching_examples=teaching_examples,
                    sentiment_histories=[val_sentiment_histories[i] for i in eval_indices],
                )
                row = score_paper_candidate(
                    candidate.name,
                    y_val_hist[eval_indices],
                    y_val_true[eval_indices],
                    predictions,
                    paper_alpha=paper_alpha,
                    paper_history_window=paper_history_window,
                    paper_horizons=paper_horizons,
                )
                row["fold"] = int(fold_idx)
                candidate_rows.append(row)
            except Exception as exc:
                error_rows.append(
                    {
                        "candidate": candidate.name,
                        "fold": int(fold_idx),
                        "error": str(exc),
                    }
                )
                print(
                    f"[paper_replication_v2] candidate={candidate.name} fold={fold_idx} failed: {exc}",
                    flush=True,
                )

    if error_rows:
        pd.DataFrame(error_rows).to_csv(output_dir / "val_fold_errors.csv", index=False)
    if not candidate_rows:
        raise RuntimeError("No validation-fold results were produced; see val_fold_errors.csv for details.")

    folds_df = pd.DataFrame(candidate_rows).sort_values(["candidate", "fold"], kind="stable")
    folds_df.to_csv(output_dir / "val_fold_metrics.csv", index=False)
    summary_df = aggregate_fold_rows(folds_df)
    summary_df.to_csv(output_dir / "val_candidate_summary.csv", index=False)

    if summary_df.empty:
        raise RuntimeError("No validation-fold results were produced.")

    best_row = summary_df.iloc[0].to_dict()
    selected_name = str(best_row["candidate"])
    selected_candidate = next(candidate for candidate in candidates if candidate.name == selected_name)
    selected_payload = {
        "selected_candidate": selected_name,
        "selection_metric": "mse_path_mean",
        "candidate_summary": best_row,
    }
    with (output_dir / "selected_candidate.json").open("w") as handle:
        json.dump(selected_payload, handle, indent=2)

    if args.skip_selected_test:
        return 0

    test_summary = _reuse_saved_test_artifact(
        base_run,
        selected_candidate,
        base_run_cfg.get("llm", {}) or {},
        y_test_hist,
        paper_alpha,
        paper_history_window,
        paper_horizons,
    )

    if test_summary is None:
        llm_cfg = _candidate_llm_config(
            base_run_cfg.get("llm", {}) or {},
            selected_candidate,
            llm_api_key,
            llm_base_url,
            timeout_seconds=timeout_seconds,
            max_retries=max_retries,
        )
        sentiment_map = sentiment_cache[str(_resolve_path(selected_candidate.sentiment_path))]
        retrieval_feature_map = None
        retrieval_feature_width = 0
        if selected_candidate.example_selection == "event_similarity":
            retrieval_event_path = (
                selected_candidate.retrieval_official_event_path or selected_candidate.official_event_path
            )
            if retrieval_event_path is None:
                raise ValueError(
                    f"candidate '{selected_candidate.name}' uses example_selection=event_similarity without "
                    "retrieval_official_event_path or official_event_path."
                )
            all_origin_dates = np.concatenate([dates_train, dates_val, dates_test])
            retrieval_feature_map, retrieval_feature_width = _build_retrieval_feature_map(
                panel_dates,
                panel_prices,
                all_origin_dates,
                sentiment_map,
                str(_resolve_path(retrieval_event_path)),
            )
        official_summary_map = None
        if selected_candidate.official_event_path:
            official_key = (
                str(_resolve_path(selected_candidate.official_event_path)),
                int(selected_candidate.official_event_lookback_days),
                int(selected_candidate.official_event_max_titles),
            )
            official_summary_map = official_summary_cache.get(official_key)
            if official_summary_map is None:
                all_origin_dates = np.concatenate([dates_train, dates_val, dates_test])
                official_summary_map = build_official_event_summary_map(
                    official_key[0],
                    all_origin_dates,
                    lookback_days=int(selected_candidate.official_event_lookback_days),
                    max_titles=int(selected_candidate.official_event_max_titles),
                )
                official_summary_cache[official_key] = official_summary_map
        test_date_arrays = build_history_dates(panel_dates, date_to_idx, dates_test, selected_candidate.history_points)
        test_sentiment_histories = build_sentiment_histories(
            test_date_arrays,
            sentiment_map,
            selected_candidate.sentiment_points,
        )
        test_exogenous_summaries = (
            [official_summary_map.get(pd.Timestamp(d).normalize()) for d in dates_test]
            if official_summary_map is not None
            else [None for _ in dates_test]
        )
        test_pool = _prepare_example_pool(
            np.concatenate([dates_train, dates_val]),
            np.concatenate([y_train_hist, y_val_hist]),
            np.concatenate([tsm_pred_train, tsm_pred_val]),
            np.concatenate([y_train_true, y_val_true]),
            selected_candidate.example_selection,
            selected_candidate.feature_window,
            retrieval_feature_map=retrieval_feature_map,
            retrieval_feature_width=retrieval_feature_width,
        )
        test_max_samples = args.test_max_samples
        if test_max_samples is None:
            test_max_samples = rep_cfg.get("test_max_samples")
        if test_max_samples is None or int(test_max_samples) <= 0:
            test_indices = np.arange(len(dates_test), dtype=int)
        else:
            test_indices = build_outer_folds(len(dates_test), n_folds=1, max_samples_per_fold=int(test_max_samples))[0]

        teaching_examples = []
        for sample_idx in test_indices:
            example_indices = _select_example_indices(
                test_pool,
                dates_test[sample_idx],
                y_test_hist[sample_idx],
                selected_candidate.example_selection,
                selected_candidate.k_examples,
                selected_candidate.feature_window,
                selected_candidate.lookback_days,
                reference_retrieval_features=(
                    retrieval_feature_map.get(pd.Timestamp(dates_test[sample_idx]).normalize())
                    if retrieval_feature_map is not None
                    else None
                ),
            )
            teaching_examples.append(
                _build_examples(
                    test_pool,
                    example_indices,
                    panel_dates,
                    date_to_idx,
                    sentiment_map,
                    selected_candidate.history_points,
                    selected_candidate.sentiment_points,
                    official_summary_map=official_summary_map,
                )
            )

        refiner = LLMRefiner(
            llm_cfg,
            cache_dir=output_dir / "llm_cache" / selected_candidate.name,
            log_dir=output_dir / "llm_logs" / selected_candidate.name,
        )
        predictions, _ = refiner.refine_batch(
            method=selected_candidate.method,
            histories=y_test_hist[test_indices],
            date_arrays=[test_date_arrays[i] for i in test_indices],
            tsm_forecasts=tsm_pred_test[test_indices],
            pred_len=pred_len,
            exogenous_summaries=[test_exogenous_summaries[i] for i in test_indices],
            price_bases=None,
            teaching_examples=teaching_examples,
            sentiment_histories=[test_sentiment_histories[i] for i in test_indices],
        )
        test_summary = score_paper_candidate(
            selected_candidate.name,
            y_test_hist[test_indices],
            y_test_true[test_indices],
            predictions,
            paper_alpha=paper_alpha,
            paper_history_window=paper_history_window,
            paper_horizons=paper_horizons,
        )
        test_summary["test_sample_cap"] = int(len(test_indices))
        test_summary["test_full_split"] = bool(len(test_indices) == len(dates_test))
    else:
        test_summary["test_sample_cap"] = int(len(dates_test))
        test_summary["test_full_split"] = True
        test_summary["reused_saved_artifact"] = True

    with (output_dir / "selected_test_summary.json").open("w") as handle:
        json.dump(test_summary, handle, indent=2)

    gap_summary = {
        "selected_candidate": selected_name,
        "selected_test_mse_path": float(test_summary["mse_path"]),
        "gap_vs_target_20": float(test_summary["mse_path"]) - 20.0,
        "gap_vs_repo_old_paper_sent_13_8262": float(test_summary["mse_path"]) - 13.8262,
        "gap_vs_repo_old_paper_cot_rf_12_5149": float(test_summary["mse_path"]) - 12.5149,
    }
    with (output_dir / "gap_summary.json").open("w") as handle:
        json.dump(gap_summary, handle, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
