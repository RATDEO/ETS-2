#!/usr/bin/env python3
"""Validate fixed regime gates on the contamination-safe rolling archive."""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval.metrics import compute_metrics_by_horizon, compute_path_metrics  # noqa: E402
from src.eval.regime_audit import (  # noqa: E402
    apply_fixed_gate,
    build_origin_regime_frame,
    overall_metrics,
)
from src.frontend_live_bundle import (  # noqa: E402
    TrainedArtifacts,
    _build_live_inputs,
    _history_features,
    _predict_prices,
    _select_origin_indices,
    train_tsm_on_prefix,
)
from src.run_experiment import load_daily_sentiment  # noqa: E402
from src.llm import LLMRefiner  # noqa: E402
from src.data import build_panel, select_feature_columns  # noqa: E402


PAPER_MODEL = "Baseline paper CoT-SENT"
HPRICE_4B_MODEL = "4B guarded HPRICE"
HDELTA_4B_MODEL = "4B raw HDELTA"
TSM_MODEL = "Raw tsm"
HDELTA_35B_MODEL = "35B raw guarded HDELTA"

SENTIMENT_ALIGNMENT_GATE = {
    "aligned": HDELTA_35B_MODEL,
    "divergent": HDELTA_4B_MODEL,
    "neutral": PAPER_MODEL,
}
TREND_VOL_GATE = {
    "down__high_vol": PAPER_MODEL,
    "down__low_vol": PAPER_MODEL,
    "up__high_vol": PAPER_MODEL,
    "up__low_vol": HDELTA_35B_MODEL,
    "flat__high_vol": HDELTA_4B_MODEL,
    "flat__low_vol": HDELTA_35B_MODEL,
}


@dataclass(frozen=True)
class ArchiveLLMCandidate:
    name: str
    display_name: str
    method: str
    base_url: str
    model: str
    sentiment_path: str = "data/news/daily_sentiment.csv"
    history_points: int = 18
    prompt_history_points: int = 18
    prompt_sentiment_points: int = 18
    sentiment_points: int = 18
    k_examples: int = 5
    example_selection: str = "similarity"
    feature_window: int = 18
    lookback_days: int = 365
    retain_context: bool = True
    strict_json_prompt: bool = False
    strict_json_response_format: bool = True
    hdelta_max_adjustment_pct: float | None = None
    hdelta_freeze_horizons: tuple[int, ...] = ()
    hdelta_sentiment_secondary: bool = False
    hprice_key_horizons: tuple[int, ...] = (1, 10, 20, 30)
    hprice_max_adjustment_pct: float | None = None
    hprice_freeze_horizons: tuple[int, ...] = ()
    hprice_sentiment_secondary: bool = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate simple regime gates on the rolling archive.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("src/config/paper_llm_cot_sent_rf_qwen_k5_h18_similarity_ctx_full.yaml"),
    )
    parser.add_argument("--data-dir", type=Path, default=Path("Data_auto"))
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--history-years", type=int, default=5)
    parser.add_argument("--origin-step", type=int, default=63)
    parser.add_argument("--max-origins", type=int, default=None)
    parser.add_argument("--llm-api-key", type=str, default=None)
    parser.add_argument("--timeout-seconds", type=float, default=120.0)
    parser.add_argument("--llm4b-base-url", type=str, default="http://192.168.1.140:9877/v1")
    parser.add_argument("--llm4b-model", type=str, default="qwen3-vl-4b-gpu")
    parser.add_argument("--llm35b-base-url", type=str, default="http://192.168.1.140:9881/v1")
    parser.add_argument("--llm35b-model", type=str, default="qwen3.5-35b-a3b-ud-q4-k-xl")
    parser.add_argument("--paper-base-url", type=str, default=None)
    parser.add_argument("--paper-model", type=str, default=None)
    parser.add_argument(
        "--paper-mode",
        choices=["fullpath", "hprice"],
        default="fullpath",
        help="Use the legacy full-path paper branch or the simplified guarded HPRICE branch for the paper role.",
    )
    parser.add_argument("--divergent-base-url", type=str, default=None)
    parser.add_argument("--divergent-model", type=str, default=None)
    parser.add_argument("--official-event-path", type=Path, default=Path("data/news/official/official_event_records_v1.csv"))
    parser.add_argument(
        "--divergent-mode",
        choices=["raw", "guarded"],
        default="raw",
        help="Use the raw or guarded HDELTA configuration for the divergent branch.",
    )
    parser.add_argument("--reuse-predictions-csv", type=Path, default=None)
    parser.add_argument(
        "--reuse-models",
        type=str,
        default=None,
        help="Comma-separated model display names to reuse from --reuse-predictions-csv.",
    )
    return parser.parse_args()


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r") as handle:
        return yaml.safe_load(handle)


def _make_output_dir(path: Path | None) -> Path:
    if path is not None:
        out = path if path.is_absolute() else (ROOT / path)
    else:
        out = ROOT / "reports" / "regime_gate_validation" / datetime.now().strftime("%Y%m%d_%H%M%S")
    out.mkdir(parents=True, exist_ok=True)
    return out


def _resolve_llm_cfg(base_cfg: dict[str, Any], candidate: ArchiveLLMCandidate, api_key: str, timeout_seconds: float) -> dict[str, Any]:
    llm_cfg = copy.deepcopy(base_cfg)
    llm_cfg["api_key"] = api_key
    llm_cfg["base_url"] = candidate.base_url
    llm_cfg["model"] = candidate.model
    llm_cfg["methods"] = [candidate.method]
    llm_cfg["max_samples"] = 100000
    llm_cfg["timeout_seconds"] = float(timeout_seconds)
    llm_cfg["history_points"] = candidate.history_points
    llm_cfg["prompt_history_points"] = candidate.prompt_history_points
    llm_cfg["prompt_sentiment_points"] = candidate.prompt_sentiment_points
    llm_cfg["temperature"] = 0.0
    llm_cfg.setdefault("cot_rf", {})
    llm_cfg["cot_rf"]["k_examples"] = candidate.k_examples
    llm_cfg["cot_rf"]["example_selection"] = candidate.example_selection
    llm_cfg["cot_rf"]["feature_window"] = candidate.feature_window
    llm_cfg["cot_rf"]["lookback_days"] = candidate.lookback_days
    llm_cfg["cot_rf"]["retain_context"] = candidate.retain_context
    llm_cfg["cot_rf"]["strict_json_prompt"] = candidate.strict_json_prompt
    llm_cfg["cot_rf"]["strict_json_response_format"] = candidate.strict_json_response_format
    llm_cfg.setdefault("sentiment", {})
    llm_cfg["sentiment"]["enabled"] = True
    llm_cfg["sentiment"]["path"] = candidate.sentiment_path
    llm_cfg["sentiment"]["date_col"] = "seendate"
    llm_cfg["sentiment"]["score_col"] = "sent_score"
    llm_cfg["sentiment"]["history_points"] = candidate.sentiment_points
    if candidate.method == "TSM+LLM-COT-SENT-RF-HDELTA":
        llm_cfg.setdefault("hdelta", {})
        llm_cfg["hdelta"]["key_horizons"] = [1, 5, 20, 30]
        llm_cfg["hdelta"]["max_adjustment_pct"] = float(candidate.hdelta_max_adjustment_pct or 3.0)
        llm_cfg["hdelta"]["freeze_horizons"] = list(candidate.hdelta_freeze_horizons)
        llm_cfg["hdelta"]["sentiment_secondary"] = bool(candidate.hdelta_sentiment_secondary)
    if candidate.method == "TSM+LLM-COT-SENT-RF-HPRICE":
        llm_cfg.setdefault("hprice", {})
        llm_cfg["hprice"]["key_horizons"] = [int(h) for h in candidate.hprice_key_horizons]
        llm_cfg["hprice"]["max_adjustment_pct"] = float(candidate.hprice_max_adjustment_pct or 1.0)
        llm_cfg["hprice"]["freeze_horizons"] = list(candidate.hprice_freeze_horizons)
        llm_cfg["hprice"]["sentiment_secondary"] = bool(candidate.hprice_sentiment_secondary)
    return llm_cfg


def _paper_role_definition(args: argparse.Namespace) -> ArchiveLLMCandidate:
    paper_base_url = args.paper_base_url or args.llm4b_base_url
    paper_model = args.paper_model or args.llm4b_model
    if str(args.paper_mode).lower() == "hprice":
        return ArchiveLLMCandidate(
            name="paper_hprice_4b",
            display_name=HPRICE_4B_MODEL,
            method="TSM+LLM-COT-SENT-RF-HPRICE",
            base_url=paper_base_url,
            model=paper_model,
            sentiment_path="data/news/daily_sentiment_qwen_votes3_daily3_gate_policy_shipping_event90_proxy.csv",
            retain_context=False,
            strict_json_prompt=True,
            hprice_key_horizons=(1, 10, 20, 30),
            hprice_max_adjustment_pct=1.0,
            hprice_freeze_horizons=(1,),
            hprice_sentiment_secondary=True,
        )
    return ArchiveLLMCandidate(
        name="paper_cot_sent",
        display_name=PAPER_MODEL,
        method="TSM+LLM-COT-SENT-RF",
        base_url=paper_base_url,
        model=paper_model,
        sentiment_path="data/news/daily_sentiment.csv",
        retain_context=True,
        strict_json_prompt=False,
    )


def _divergent_display_name(args: argparse.Namespace) -> str:
    divergent_base_url = args.divergent_base_url or args.llm4b_base_url
    divergent_model = args.divergent_model or args.llm4b_model
    divergent_mode = str(args.divergent_mode).lower()
    if (
        divergent_base_url == args.llm4b_base_url
        and divergent_model == args.llm4b_model
        and divergent_mode == "raw"
    ):
        return HDELTA_4B_MODEL
    if divergent_mode == "guarded":
        return f"{divergent_model} guarded HDELTA"
    return f"{divergent_model} raw HDELTA"


def _candidate_definitions(args: argparse.Namespace) -> list[ArchiveLLMCandidate]:
    divergent_base_url = args.divergent_base_url or args.llm4b_base_url
    divergent_model = args.divergent_model or args.llm4b_model
    divergent_guarded = str(args.divergent_mode).lower() == "guarded"
    return [
        _paper_role_definition(args),
        ArchiveLLMCandidate(
            name="hdelta_4b",
            display_name=_divergent_display_name(args),
            method="TSM+LLM-COT-SENT-RF-HDELTA",
            base_url=divergent_base_url,
            model=divergent_model,
            sentiment_path="data/news/daily_sentiment.csv",
            retain_context=not divergent_guarded,
            strict_json_prompt=divergent_guarded,
            hdelta_max_adjustment_pct=1.0 if divergent_guarded else 3.0,
            hdelta_freeze_horizons=(1,) if divergent_guarded else (),
            hdelta_sentiment_secondary=divergent_guarded,
        ),
        ArchiveLLMCandidate(
            name="hdelta_35b_guarded",
            display_name=HDELTA_35B_MODEL,
            method="TSM+LLM-COT-SENT-RF-HDELTA",
            base_url=args.llm35b_base_url,
            model=args.llm35b_model,
            sentiment_path="data/news/daily_sentiment.csv",
            retain_context=False,
            strict_json_prompt=True,
            hdelta_max_adjustment_pct=1.0,
            hdelta_freeze_horizons=(1,),
            hdelta_sentiment_secondary=True,
        ),
    ]


def _pool_slice(trained: TrainedArtifacts) -> slice:
    return slice(trained.train_slice.start, trained.val_slice.stop)


def _pool_predictions(trained: TrainedArtifacts, target_mode: str, batch_size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    idx = _pool_slice(trained)
    pool_pred = _predict_prices(
        model=trained.model,
        scaler=trained.scaler,
        X_enc=trained.windows.X_enc[idx],
        X_dec=trained.windows.X_dec[idx],
        base_prices=trained.windows.base_prices[idx],
        target_mode=target_mode,
        batch_size=batch_size,
    )
    if target_mode == "returns":
        pool_true = trained.windows.y[idx]
        from src.run_experiment import returns_to_prices  # local import to keep imports narrow

        pool_true_prices = returns_to_prices(pool_true, trained.windows.base_prices[idx])
    else:
        pool_true_prices = trained.windows.y[idx]
    pool_dates = pd.to_datetime(trained.windows.pred_dates[idx]).to_numpy()
    pool_histories = trained.windows.price_histories[idx]
    return pool_dates, pool_histories, pool_pred, pool_true_prices


def _sentiment_series_from_dates(dates: list[str], sentiment_map: Mapping[pd.Timestamp, float], sentiment_points: int) -> np.ndarray:
    values = [float(sentiment_map.get(pd.Timestamp(d).normalize(), 0.0)) for d in dates]
    values = values[-sentiment_points:]
    if len(values) < sentiment_points:
        values = [0.0] * (sentiment_points - len(values)) + values
    return np.asarray(values, dtype=float)


def _select_examples(
    *,
    pool_dates: np.ndarray,
    pool_histories: np.ndarray,
    pool_forecasts: np.ndarray,
    pool_truth: np.ndarray,
    origin_date: pd.Timestamp,
    current_history: np.ndarray,
    candidate: ArchiveLLMCandidate,
    panel_dates: np.ndarray,
    date_to_idx: dict[pd.Timestamp, int],
    sentiment_map: Mapping[pd.Timestamp, float],
) -> list[dict[str, Any]]:
    candidate_indices = np.arange(len(pool_dates), dtype=int)
    candidate_indices = candidate_indices[pd.to_datetime(pool_dates[candidate_indices]) < origin_date]
    if candidate.lookback_days is not None and candidate_indices.size:
        cutoff = np.datetime64(origin_date - pd.Timedelta(days=int(candidate.lookback_days)))
        lookback_idx = candidate_indices[pool_dates[candidate_indices] >= cutoff]
        if len(lookback_idx):
            candidate_indices = lookback_idx
    if candidate_indices.size == 0:
        return []

    sample_feat = _history_features(current_history, candidate.feature_window)
    cand_feats = np.vstack([_history_features(pool_histories[idx], candidate.feature_window) for idx in candidate_indices])
    distances = np.linalg.norm(cand_feats - sample_feat, axis=1)
    chosen = candidate_indices[np.argsort(distances)[: candidate.k_examples]]
    chosen = chosen[np.argsort(pool_dates[chosen])]

    examples: list[dict[str, Any]] = []
    for idx in chosen:
        example_pred_date = pd.Timestamp(pool_dates[idx]).normalize()
        example_history_dates = []
        pred_idx = date_to_idx.get(example_pred_date)
        if pred_idx is None:
            continue
        start_idx = max(0, pred_idx - candidate.history_points)
        example_history_dates = [str(pd.Timestamp(d).date()) for d in panel_dates[start_idx:pred_idx]]
        examples.append(
            {
                "history": np.asarray(pool_histories[idx][-candidate.history_points :], dtype=float),
                "forecast": np.asarray(pool_forecasts[idx], dtype=float),
                "truth": np.asarray(pool_truth[idx], dtype=float),
                "date": str(example_pred_date),
                "sentiment_history": _sentiment_series_from_dates(
                    example_history_dates,
                    sentiment_map,
                    candidate.sentiment_points,
                ),
            }
        )
    return examples


def _metric_row(name: str, y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    path = compute_path_metrics(y_true, y_pred)
    by_h = compute_metrics_by_horizon(y_true, y_pred, horizons=[1, 5, 20, 30])
    row = {
        "model": name,
        "n_origins": int(y_true.shape[0]),
        "path_mse": float(path["mse_path"]),
    }
    for h in [1, 5, 20, 30]:
        if h in by_h.index:
            row[f"h{h}_mse"] = float(by_h.loc[h, "mse"])
    return row


def _serialize_origin_rows(
    origin_date: str,
    target_dates: list[str],
    actual_prices: np.ndarray,
    last_observed_price: float,
    predictions: Mapping[str, np.ndarray],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for model_name, forecast in predictions.items():
        for i, target_date in enumerate(target_dates):
            rows.append(
                {
                    "origin_date": origin_date,
                    "target_date": str(target_date),
                    "model": model_name,
                    "horizon": int(i + 1),
                    "predicted_price": float(forecast[i]),
                    "actual_price": float(actual_prices[i]),
                    "last_observed_price": float(last_observed_price),
                }
            )
    return rows


def _gate_strategies(paper_model_name: str, divergent_model_name: str) -> dict[str, tuple[str, Mapping[str, str], str]]:
    return {
        "gate_sentiment_alignment_5d": (
            "sentiment_alignment_5d",
            {
                "aligned": HDELTA_35B_MODEL,
                "divergent": divergent_model_name,
                "neutral": paper_model_name,
            },
            paper_model_name,
        ),
        "gate_official_event_alignment_5d": (
            "official_event_alignment_5d",
            {
                "aligned": HDELTA_35B_MODEL,
                "divergent": divergent_model_name,
                "neutral": paper_model_name,
            },
            paper_model_name,
        ),
        "gate_official_supply_regime_30d": (
            "official_supply_regime_30d",
            {
                "bearish_supply": HDELTA_35B_MODEL,
                "bullish_supply": divergent_model_name,
                "neutral_supply": paper_model_name,
            },
            paper_model_name,
        ),
        "gate_trend_vol_regime": (
            "trend_vol_regime",
            {
                "down__high_vol": paper_model_name,
                "down__low_vol": paper_model_name,
                "up__high_vol": paper_model_name,
                "up__low_vol": HDELTA_35B_MODEL,
                "flat__high_vol": divergent_model_name,
                "flat__low_vol": HDELTA_35B_MODEL,
            },
            paper_model_name,
        ),
    }


def _parse_reuse_models(text: str | None) -> set[str]:
    if not text:
        return set()
    return {part.strip() for part in text.split(",") if part.strip()}


def _load_reused_prediction_arrays(
    csv_path: Path,
    pred_len: int,
    reuse_models: set[str],
) -> dict[str, dict[str, np.ndarray]]:
    if not reuse_models:
        return {}
    frame = pd.read_csv(csv_path)
    required = {"origin_date", "model", "horizon", "predicted_price"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns in reused predictions CSV: {sorted(missing)}")
    out: dict[str, dict[str, np.ndarray]] = {}
    frame = frame[frame["model"].isin(reuse_models)].copy()
    for (model_name, origin_date), grp in frame.groupby(["model", "origin_date"], sort=True):
        grp = grp.sort_values("horizon")
        if len(grp) != pred_len:
            raise ValueError(
                f"Reused predictions for model={model_name}, origin={origin_date} "
                f"have {len(grp)} rows, expected {pred_len}."
            )
        out.setdefault(str(model_name), {})[str(origin_date)] = grp["predicted_price"].to_numpy(dtype=float)
    missing_models = sorted(reuse_models - set(out))
    if missing_models:
        raise ValueError(f"Requested reuse models not found in reused predictions CSV: {missing_models}")
    return out


def main() -> int:
    args = parse_args()
    config_path = args.config if args.config.is_absolute() else (ROOT / args.config)
    data_dir = args.data_dir if args.data_dir.is_absolute() else (ROOT / args.data_dir)
    output_dir = _make_output_dir(args.output_dir)
    api_key = args.llm_api_key or os.environ.get("OPENAI_API_KEY") or "deo"

    raw_cfg = _load_yaml(config_path)
    raw_cfg = copy.deepcopy(raw_cfg)
    raw_cfg.setdefault("frontend_live_bundle", {})
    raw_cfg["frontend_live_bundle"]["history_years"] = int(args.history_years)
    raw_cfg["frontend_live_bundle"]["origin_step"] = int(args.origin_step)
    raw_cfg["output"]["generate_paper"] = False

    panel, _schema = build_panel(data_dir=data_dir, config=raw_cfg)
    target_mode = str(raw_cfg.get("target", {}).get("mode", "returns")).lower()
    target_col = "y_return" if target_mode == "returns" else "y"
    feature_cols = select_feature_columns(
        panel,
        target_col=target_col,
        max_exogenous_features=int((raw_cfg.get("features", {}) or {}).get("max_exogenous_features_model", 10)),
        preferred_feature_order=((raw_cfg.get("features", {}) or {}).get("preferred_feature_order")),
    )
    panel = panel[["date"] + [c for c in feature_cols if c in panel.columns]].copy()
    panel["date"] = pd.to_datetime(panel["date"])
    panel = panel.sort_values("date").reset_index(drop=True)

    seq_len = int(raw_cfg.get("time_series", {}).get("seq_len", 120))
    pred_len = int(raw_cfg.get("time_series", {}).get("pred_len", 30))
    reuse_models = _parse_reuse_models(args.reuse_models)
    reused_predictions = _load_reused_prediction_arrays(
        (args.reuse_predictions_csv if args.reuse_predictions_csv is None or args.reuse_predictions_csv.is_absolute() else (ROOT / args.reuse_predictions_csv)),
        pred_len=pred_len,
        reuse_models=reuse_models,
    ) if args.reuse_predictions_csv else {}
    archive_cfg = raw_cfg.get("frontend_live_bundle", {}) or {}
    origin_indices = _select_origin_indices(
        panel=panel,
        pred_len=pred_len,
        history_years=int(archive_cfg.get("history_years", 5)),
        origin_step=int(archive_cfg.get("origin_step", 63)),
        seq_len=seq_len,
        min_train_windows=int(archive_cfg.get("min_train_windows", 756)),
        min_val_windows=int(archive_cfg.get("min_val_windows", 126)),
    )
    if args.max_origins is not None:
        origin_indices = origin_indices[-int(args.max_origins) :]

    origin_dates = [pd.Timestamp(panel["date"].iloc[idx]).normalize() for idx in origin_indices]
    origin_regimes, regime_thresholds = build_origin_regime_frame(
        origin_dates,
        data_dir=data_dir,
        sentiment_path=ROOT / "data" / "news" / "daily_sentiment.csv",
        official_event_path=(args.official_event_path if args.official_event_path.is_absolute() else (ROOT / args.official_event_path)),
    )
    regime_by_date = origin_regimes.set_index("date")

    base_llm_cfg = copy.deepcopy(raw_cfg.get("llm", {}) or {})
    candidates = _candidate_definitions(args)
    paper_model_name = candidates[0].display_name
    divergent_model_name = next(
        candidate.display_name for candidate in candidates if candidate.name == "hdelta_4b"
    )
    gate_strategies = _gate_strategies(paper_model_name, divergent_model_name)
    refiners: dict[str, LLMRefiner] = {}
    for candidate in candidates:
        if candidate.display_name in reuse_models:
            continue
        llm_cfg = _resolve_llm_cfg(base_llm_cfg, candidate, api_key=api_key, timeout_seconds=args.timeout_seconds)
        refiner_dir = output_dir / "llm" / candidate.name
        refiners[candidate.display_name] = LLMRefiner(
            llm_cfg,
            cache_dir=refiner_dir / "cache",
            log_dir=refiner_dir / "logs",
        )

    sentiment_map = load_daily_sentiment(
        str(ROOT / "data" / "news" / "daily_sentiment.csv"),
        date_col="seendate",
        score_col="sent_score",
    )

    origin_rows: list[dict[str, Any]] = []
    model_forecasts: dict[str, list[np.ndarray]] = {TSM_MODEL: []}
    for candidate in candidates:
        model_forecasts[candidate.display_name] = []
    y_true_rows: list[np.ndarray] = []
    progress_rows: list[dict[str, Any]] = []

    for pos, origin_idx in enumerate(origin_indices, start=1):
        origin_date = pd.Timestamp(panel["date"].iloc[origin_idx]).normalize()
        prefix_panel = panel.iloc[: origin_idx + 1].copy()
        trained = train_tsm_on_prefix(prefix_panel=prefix_panel, config_raw=raw_cfg)
        batch_size = int(raw_cfg.get("model", {}).get("batch_size", 32))
        panel_dates = pd.to_datetime(prefix_panel["date"]).to_numpy()
        date_to_idx = {pd.Timestamp(d).normalize(): i for i, d in enumerate(panel_dates)}

        prefix_panel.attrs["pred_len"] = pred_len
        X_enc, X_dec, last_price, history_prices = _build_live_inputs(
            prefix_panel=prefix_panel,
            feature_cols=trained.windows.feature_cols,
            target_col=target_col,
            seq_len=seq_len,
            label_len=int(raw_cfg.get("time_series", {}).get("label_len", 30)),
        )
        base_pred = _predict_prices(
            model=trained.model,
            scaler=trained.scaler,
            X_enc=X_enc[None, :, :],
            X_dec=X_dec[None, :, :],
            base_prices=np.array([last_price], dtype=np.float32),
            target_mode=target_mode,
            batch_size=batch_size,
        )[0]

        actual_slice = prefix_panel.iloc[0:0]  # keep type stable
        actual_slice = panel.iloc[origin_idx + 1 : origin_idx + pred_len + 1]
        actual_prices = actual_slice["y"].to_numpy(dtype=float)
        target_dates = [str(pd.Timestamp(d).date()) for d in actual_slice["date"].tolist()]
        if len(actual_prices) != pred_len:
            raise ValueError(f"Origin {origin_date.date()} has incomplete future window.")

        origin_date_str = str(origin_date.date())
        if TSM_MODEL in reuse_models:
            reused_tsm = reused_predictions.get(TSM_MODEL, {}).get(origin_date_str)
            if reused_tsm is None:
                raise ValueError(f"Missing reused predictions for model={TSM_MODEL}, origin={origin_date_str}")
            origin_predictions: dict[str, np.ndarray] = {TSM_MODEL: np.asarray(reused_tsm, dtype=float)}
        else:
            origin_predictions = {TSM_MODEL: np.asarray(base_pred, dtype=float)}
        y_true_rows.append(np.asarray(actual_prices, dtype=float))

        pool_dates, pool_histories, pool_forecasts, pool_truth = _pool_predictions(
            trained=trained,
            target_mode=target_mode,
            batch_size=batch_size,
        )
        for candidate in candidates:
            if candidate.display_name in reuse_models:
                reused_pred = reused_predictions.get(candidate.display_name, {}).get(origin_date_str)
                if reused_pred is None:
                    raise ValueError(
                        f"Missing reused predictions for model={candidate.display_name}, origin={origin_date_str}"
                    )
                origin_predictions[candidate.display_name] = np.asarray(reused_pred, dtype=float)
                progress_rows.append(
                    {
                        "origin_date": origin_date_str,
                        "model": candidate.display_name,
                        "path_mse": float(np.mean((actual_prices - origin_predictions[candidate.display_name]) ** 2)),
                        "fallback_to_tsm": False,
                        "reused_prediction": True,
                    }
                )
                continue
            current_history = np.asarray(history_prices[-candidate.history_points :], dtype=float)
            current_history_dates = [
                str(pd.Timestamp(d).date())
                for d in pd.to_datetime(prefix_panel["date"]).tail(candidate.history_points).tolist()
            ]
            examples = _select_examples(
                pool_dates=pool_dates,
                pool_histories=pool_histories,
                pool_forecasts=pool_forecasts,
                pool_truth=pool_truth,
                origin_date=origin_date,
                current_history=current_history,
                candidate=candidate,
                panel_dates=panel_dates,
                date_to_idx=date_to_idx,
                sentiment_map=sentiment_map,
            )
            sentiment_history = _sentiment_series_from_dates(
                current_history_dates,
                sentiment_map,
                candidate.sentiment_points,
            )
            refined = None
            meta: dict[str, Any] = {}
            if examples:
                refined, meta = refiners[candidate.display_name].refine(
                    method=candidate.method,
                    history=current_history,
                    dates=current_history_dates,
                    tsm_forecast=np.asarray(base_pred, dtype=float),
                    pred_len=pred_len,
                    exogenous_summary=None,
                    price_base=float(last_price),
                    teaching_examples=examples,
                    sentiment_history=sentiment_history,
                )
            if refined is None:
                refined = np.asarray(base_pred, dtype=float)
                meta = {**meta, "fallback_to_tsm": True, "reason": meta.get("reason", "no_llm_output")}
            origin_predictions[candidate.display_name] = np.asarray(refined, dtype=float)
            progress_rows.append(
                {
                    "origin_date": origin_date_str,
                    "model": candidate.display_name,
                    "path_mse": float(np.mean((actual_prices - origin_predictions[candidate.display_name]) ** 2)),
                    "fallback_to_tsm": bool((meta or {}).get("fallback_to_tsm", False)),
                    "reused_prediction": False,
                }
            )

        for model_name in model_forecasts:
            model_forecasts[model_name].append(origin_predictions[model_name])
        origin_rows.extend(
            _serialize_origin_rows(
                origin_date=str(origin_date.date()),
                target_dates=target_dates,
                actual_prices=actual_prices,
                last_observed_price=float(last_price),
                predictions=origin_predictions,
            )
        )
        pd.DataFrame(progress_rows).to_csv(output_dir / "archive_origin_progress.csv", index=False)
        print(f"[{pos}/{len(origin_indices)}] {origin_date.date()} complete", flush=True)

    y_true = np.vstack(y_true_rows)
    prediction_arrays = {name: np.vstack(rows) for name, rows in model_forecasts.items()}
    overall_df = overall_metrics(y_true, prediction_arrays)

    gate_assignment_rows: list[pd.DataFrame] = []
    gate_predictions: dict[str, np.ndarray] = {}
    for gate_name, (regime_col, model_map, fallback_model) in gate_strategies.items():
        selected_pred, selection_df = apply_fixed_gate(
            origin_regimes,
            prediction_arrays,
            regime_column=regime_col,
            model_by_regime=model_map,
            fallback_model=fallback_model,
            strategy_name=gate_name,
        )
        gate_predictions[gate_name] = selected_pred
        gate_assignment_rows.append(selection_df)

    combined_predictions = {**prediction_arrays, **gate_predictions}
    summary_df = overall_metrics(y_true, combined_predictions)
    summary_df.to_csv(output_dir / "archive_gate_summary.csv", index=False)

    gate_assignment_rows = []
    for gate_name, (regime_col, model_map, fallback_model) in gate_strategies.items():
        _, selection_df = apply_fixed_gate(
            origin_regimes,
            prediction_arrays,
            regime_column=regime_col,
            model_by_regime=model_map,
            fallback_model=fallback_model,
            strategy_name=gate_name,
        )
        selection_df = selection_df.rename(
            columns={
                regime_col: f"{gate_name}__regime_value",
                "selected_model": f"{gate_name}__selected_model",
                "strategy_name": f"{gate_name}__strategy_name",
            }
        )
        gate_assignment_rows.append(selection_df)
    gate_assignments = gate_assignment_rows[0]
    for extra in gate_assignment_rows[1:]:
        gate_assignments = gate_assignments.merge(extra, on="date", how="outer")
    gate_assignments = gate_assignments.merge(origin_regimes, on="date", how="left")
    gate_assignments.to_csv(output_dir / "gate_assignments_by_origin.csv", index=False)

    archive_df = pd.DataFrame(origin_rows)
    archive_df.to_csv(output_dir / "benchmark_archive_predictions.csv", index=False)

    metadata = {
        "config": str(config_path),
        "data_dir": str(data_dir),
        "history_years": int(args.history_years),
        "origin_step": int(args.origin_step),
        "n_origins": int(len(origin_indices)),
        "origin_dates": [str(d.date()) for d in origin_dates],
        "regime_thresholds": regime_thresholds.__dict__,
        "gates": {
            "gate_sentiment_alignment_5d": gate_strategies["gate_sentiment_alignment_5d"][1],
            "gate_official_event_alignment_5d": gate_strategies["gate_official_event_alignment_5d"][1],
            "gate_official_supply_regime_30d": gate_strategies["gate_official_supply_regime_30d"][1],
            "gate_trend_vol_regime": gate_strategies["gate_trend_vol_regime"][1],
        },
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    summary_payload = {
        "benchmark_models": overall_df.to_dict(orient="records"),
        "with_gates": summary_df.to_dict(orient="records"),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")
    print(summary_df.to_string(index=False), flush=True)
    print(f"Wrote archive gate validation to {output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
