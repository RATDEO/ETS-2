from __future__ import annotations

import copy
import json
import logging
import re
import shlex
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import yaml

DEFAULT_METHOD_NAME = "TSM+LLM-COT-RF-HDELTA"
DEFAULT_MARKET_NAME = "UK ETS"
DEFAULT_MODEL_VERSION = "uk_ets_llm_4b_canonical_default"
HEADLINE_HORIZONS = (1, 5, 20, 30)
EXPECTED_EXPORT_FILES = [
    "latest.json",
    "horizon_1d.json",
    "horizon_5d.json",
    "horizon_20d.json",
    "horizon_30d.json",
    "manifest.json",
    "status.json",
]
ARCHIVE_COLUMNS = [
    "run_id",
    "forecast_made_on",
    "latest_observed_date",
    "latest_observed_price",
    "target_date",
    "step_index",
    "actual",
    "base_tsm_forecast",
    "llm_tsm_forecast",
    "model_version",
    "model_commit",
    "data_version",
    "is_realized",
    "generated_at",
    "record_source",
    "source_run_id",
]

logger = logging.getLogger(__name__)


def _iso_utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def empty_archive_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=ARCHIVE_COLUMNS)


def _normalize_archive_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return empty_archive_frame()

    out = frame.copy()
    for col in ARCHIVE_COLUMNS:
        if col not in out.columns:
            out[col] = np.nan

    date_cols = ["forecast_made_on", "latest_observed_date", "target_date", "generated_at"]
    for col in date_cols:
        out[col] = out[col].astype("string")

    str_cols = ["run_id", "model_version", "model_commit", "data_version", "record_source", "source_run_id"]
    for col in str_cols:
        out[col] = out[col].fillna("").astype("string")

    num_cols = ["latest_observed_price", "actual", "base_tsm_forecast", "llm_tsm_forecast"]
    for col in num_cols:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out["step_index"] = pd.to_numeric(out["step_index"], errors="coerce").astype("Int64")
    out["is_realized"] = out["is_realized"].fillna(False).astype(bool)

    extra_cols = [col for col in out.columns if col not in ARCHIVE_COLUMNS]
    out = out[ARCHIVE_COLUMNS + extra_cols]
    return out.sort_values(["latest_observed_date", "step_index", "generated_at"], kind="stable").reset_index(drop=True)


def load_archive(archive_path: str | Path) -> pd.DataFrame:
    path = Path(archive_path)
    if not path.exists():
        return empty_archive_frame()
    return _normalize_archive_frame(pd.read_csv(path))


def save_archive(frame: pd.DataFrame, archive_path: str | Path) -> Path:
    path = Path(archive_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    normalized = _normalize_archive_frame(frame)
    normalized.to_csv(path, index=False)
    return path


def _read_run_metadata(run_dir: str | Path) -> dict[str, Any]:
    run_dir = Path(run_dir).resolve()
    config_path = run_dir / "config_resolved.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing resolved config: {config_path}")
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    config_raw = {
        key: value
        for key, value in raw.items()
        if key not in {"run_id", "git_hash", "timestamp"}
    }
    return {
        "run_id": str(raw.get("run_id") or run_dir.name),
        "git_hash": str(raw.get("git_hash") or ""),
        "timestamp": str(raw.get("timestamp") or ""),
        "config_raw": config_raw,
    }


def _panel_with_prices(panel_path: str | Path) -> pd.DataFrame:
    panel = pd.read_parquet(panel_path, columns=["date", "y"]).copy()
    panel["date"] = pd.to_datetime(panel["date"]).dt.normalize()
    panel["y"] = pd.to_numeric(panel["y"], errors="coerce")
    panel = panel.dropna(subset=["date", "y"]).sort_values("date").drop_duplicates("date").reset_index(drop=True)
    return panel


def _future_realized_slice(panel: pd.DataFrame, origin_date: pd.Timestamp, pred_len: int) -> pd.DataFrame:
    date_to_idx = {pd.Timestamp(ts).normalize(): idx for idx, ts in enumerate(panel["date"])}
    origin_idx = date_to_idx.get(pd.Timestamp(origin_date).normalize())
    if origin_idx is None:
        raise ValueError(f"Origin date {origin_date.date()} not found in panel.")
    future = panel.iloc[origin_idx + 1 : origin_idx + 1 + int(pred_len)].copy()
    if len(future) < int(pred_len):
        raise ValueError(
            f"Origin {origin_date.date()} only has {len(future)} realized rows, need {pred_len}."
        )
    return future.reset_index(drop=True)


def _next_business_target_dates(origin_date: pd.Timestamp, pred_len: int) -> list[str]:
    return [
        str(ts.date())
        for ts in pd.bdate_range(pd.Timestamp(origin_date).normalize() + pd.Timedelta(days=1), periods=int(pred_len))
    ]


def _archive_row(
    *,
    run_id: str,
    forecast_made_on: str,
    latest_observed_date: str,
    latest_observed_price: float,
    target_date: str,
    step_index: int,
    actual: float | None,
    base_tsm_forecast: float,
    llm_tsm_forecast: float,
    model_version: str,
    model_commit: str,
    data_version: str,
    is_realized: bool,
    generated_at: str,
    record_source: str,
    source_run_id: str,
) -> dict[str, Any]:
    return {
        "run_id": str(run_id),
        "forecast_made_on": str(forecast_made_on),
        "latest_observed_date": str(latest_observed_date),
        "latest_observed_price": float(latest_observed_price),
        "target_date": str(target_date),
        "step_index": int(step_index),
        "actual": None if actual is None or pd.isna(actual) else float(actual),
        "base_tsm_forecast": float(base_tsm_forecast),
        "llm_tsm_forecast": float(llm_tsm_forecast),
        "model_version": str(model_version),
        "model_commit": str(model_commit or ""),
        "data_version": str(data_version or ""),
        "is_realized": bool(is_realized),
        "generated_at": str(generated_at),
        "record_source": str(record_source),
        "source_run_id": str(source_run_id),
    }


def build_backtest_seed_rows(
    run_dir: str | Path,
    *,
    method_name: str = DEFAULT_METHOD_NAME,
    model_version: str = DEFAULT_MODEL_VERSION,
) -> pd.DataFrame:
    run_dir = Path(run_dir).resolve()
    metadata = _read_run_metadata(run_dir)
    panel = _panel_with_prices(run_dir / "data" / "panel.parquet")
    pred_path = run_dir / "predictions" / f"{method_name}_pred_test_subset.npz"
    if not pred_path.exists():
        raise FileNotFoundError(f"Missing canonical prediction subset: {pred_path}")

    with np.load(pred_path, allow_pickle=True) as npz:
        origin_dates = pd.to_datetime(np.asarray(npz["dates"])).normalize()
        y_true = np.asarray(npz["y_true"], dtype=float)
        yhat = np.asarray(npz["yhat"], dtype=float)
        base_pred = np.asarray(npz["base_pred"], dtype=float)

    if yhat.ndim != 2:
        raise ValueError(f"Expected 2-D llm predictions, got shape {yhat.shape}.")
    if base_pred.shape != yhat.shape or y_true.shape != yhat.shape:
        raise ValueError(
            "Canonical subset arrays do not align: "
            f"base_pred={base_pred.shape}, yhat={yhat.shape}, y_true={y_true.shape}."
        )

    rows: list[dict[str, Any]] = []
    generated_at = metadata["timestamp"] or _iso_utc_now()
    for row_idx, origin_date in enumerate(origin_dates):
        origin_str = str(pd.Timestamp(origin_date).date())
        try:
            target_dates = [
                str(pd.Timestamp(ts).date())
                for ts in _future_realized_slice(panel, pd.Timestamp(origin_date), yhat.shape[1])["date"].tolist()
            ]
        except ValueError:
            target_dates = _next_business_target_dates(pd.Timestamp(origin_date), yhat.shape[1])
        latest_price = float(panel.loc[panel["date"] == pd.Timestamp(origin_date), "y"].iloc[0])
        source_run_id = str(run_dir.name)
        row_run_id = f"{source_run_id}:backfill:{origin_str}"
        for step_idx in range(yhat.shape[1]):
            rows.append(
                _archive_row(
                    run_id=row_run_id,
                    forecast_made_on=origin_str,
                    latest_observed_date=origin_str,
                    latest_observed_price=latest_price,
                    target_date=target_dates[step_idx],
                    step_index=step_idx + 1,
                    actual=float(y_true[row_idx, step_idx]),
                    base_tsm_forecast=float(base_pred[row_idx, step_idx]),
                    llm_tsm_forecast=float(yhat[row_idx, step_idx]),
                    model_version=model_version,
                    model_commit=metadata["git_hash"],
                    data_version=origin_str,
                    is_realized=True,
                    generated_at=generated_at,
                    record_source="historical_backfill",
                    source_run_id=source_run_id,
                )
            )
    return _normalize_archive_frame(pd.DataFrame(rows))


def _pool_predictions(trained: Any, target_mode: str, batch_size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    from .frontend_live_bundle import _predict_prices
    from .run_experiment import returns_to_prices

    idx = slice(trained.train_slice.start, trained.val_slice.stop)
    pred = _predict_prices(
        model=trained.model,
        scaler=trained.scaler,
        X_enc=trained.windows.X_enc[idx],
        X_dec=trained.windows.X_dec[idx],
        base_prices=trained.windows.base_prices[idx],
        target_mode=target_mode,
        batch_size=batch_size,
    )
    if target_mode == "returns":
        truth = returns_to_prices(trained.windows.y[idx], trained.windows.base_prices[idx])
    else:
        truth = trained.windows.y[idx]
    return (
        pd.to_datetime(trained.windows.pred_dates[idx]).to_numpy(),
        np.asarray(trained.windows.price_histories[idx], dtype=float),
        np.asarray(pred, dtype=float),
        np.asarray(truth, dtype=float),
    )


def _history_distance_features(history: np.ndarray, feature_window: int) -> np.ndarray:
    from .frontend_live_bundle import _history_features

    return _history_features(np.asarray(history, dtype=float), int(feature_window))


def _normalize_history_dates(date_values: Sequence[object], expected_len: int) -> list[str]:
    normalized = [str(pd.Timestamp(value).date()) for value in pd.to_datetime(list(date_values))]
    if expected_len <= 0:
        return []
    if len(normalized) >= expected_len:
        return normalized[-expected_len:]
    if not normalized:
        return []
    pad = [normalized[0]] * max(0, expected_len - len(normalized))
    return pad + normalized


def _sentiment_history_from_dates(
    date_values: Sequence[object],
    sentiment_map: Mapping[pd.Timestamp, float] | None,
    sentiment_points: int | None,
) -> list[float]:
    if sentiment_map is None or sentiment_points is None or int(sentiment_points) <= 0:
        return []
    normalized_dates = [pd.Timestamp(value).normalize() for value in pd.to_datetime(list(date_values))]
    sent_values = [float(sentiment_map.get(value, 0.0)) for value in normalized_dates]
    if len(sent_values) < int(sentiment_points):
        sent_values = ([0.0] * (int(sentiment_points) - len(sent_values))) + sent_values
    return list(np.asarray(sent_values[-int(sentiment_points):], dtype=float))


def _online_memory_result_slug(method_name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(method_name).strip())


def _load_online_memory_gate_selection(
    run_dir: str | Path | None,
    method_name: str,
) -> dict[str, Any] | None:
    if run_dir is None:
        return None
    run_dir = Path(run_dir).resolve()
    selection_path = run_dir / "llm" / f"online_memory_gate_selection_{_online_memory_result_slug(method_name)}.json"
    if not selection_path.exists():
        return None
    try:
        payload = json.loads(selection_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Failed to read online-memory gate selection %s: %s", selection_path, exc)
        return None
    if not isinstance(payload, dict):
        return None
    payload["selection_path"] = str(selection_path)
    return payload


def _merge_online_memory_gate_cfg(
    gate_cfg: Mapping[str, Any] | None,
    selection_payload: Mapping[str, Any] | None,
) -> dict[str, Any]:
    merged = dict(gate_cfg or {})
    if not selection_payload:
        return merged
    selected_gate_cfg = selection_payload.get("selected_gate_cfg")
    if isinstance(selected_gate_cfg, Mapping):
        merged.update(dict(selected_gate_cfg))
    return merged


def _resolve_live_learned_gate_cfg(
    gate_cfg: Mapping[str, Any] | None,
    selection_payload: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    learned_cfg = dict((gate_cfg or {}).get("learned", {}) or {})
    if selection_payload and str(selection_payload.get("gate_mode", "")).strip().lower() == "learned":
        learned_cfg.update(
            {
                "enabled": True,
                "model_type": selection_payload.get("model_type", learned_cfg.get("model_type", "logistic")),
                "feature_columns": selection_payload.get("feature_columns", learned_cfg.get("feature_columns", [])),
                "use_all_for_fit": True,
            }
        )
    return learned_cfg if learned_cfg.get("enabled", False) else None


def _reference_case_profile(
    *,
    history: np.ndarray,
    forecast: np.ndarray,
    window: np.ndarray | None,
    feature_window: int,
    augment_with_retrieval: bool,
    feature_cols: Sequence[str],
    target_col: str,
    preferred_features: Sequence[str] | None,
    retrieval_feature_columns: Sequence[str] | None,
    retrieval_max_features: int,
) -> np.ndarray:
    from .run_experiment import _case_profile_features, build_retrieval_feature_vector

    profile = _case_profile_features(
        np.asarray(history, dtype=float),
        np.asarray(forecast, dtype=float),
        feature_window=int(feature_window),
    )
    if not augment_with_retrieval or window is None:
        return profile
    retrieval_vector = build_retrieval_feature_vector(
        np.asarray(window, dtype=float),
        list(feature_cols),
        target_col,
        feature_window=int(feature_window),
        preferred_features=list(preferred_features) if preferred_features is not None else None,
        include_features=list(retrieval_feature_columns) if retrieval_feature_columns is not None else None,
        max_features=int(retrieval_max_features),
    )
    if retrieval_vector.size == 0:
        return profile
    return np.concatenate([profile, retrieval_vector])


def _candidate_example_indices(
    *,
    pool_dates: np.ndarray,
    pool_histories: np.ndarray,
    pool_forecasts: np.ndarray,
    pool_truth: np.ndarray,
    origin_date: pd.Timestamp,
    current_history: np.ndarray,
    selection_mode: str,
    k_examples: int,
    feature_window: int,
    lookback_days: int | None,
) -> np.ndarray:
    from .run_experiment import select_top_score_indices

    candidate_indices = np.arange(len(pool_dates), dtype=int)
    candidate_indices = candidate_indices[pd.to_datetime(pool_dates[candidate_indices]) < pd.Timestamp(origin_date)]
    if lookback_days is not None and candidate_indices.size:
        cutoff = np.datetime64(pd.Timestamp(origin_date) - pd.Timedelta(days=int(lookback_days)))
        candidate_indices = candidate_indices[pool_dates[candidate_indices] >= cutoff]
    if candidate_indices.size == 0:
        return np.asarray([], dtype=int)

    selection_mode = str(selection_mode or "recent").strip().lower()
    if selection_mode in {"high_error", "recent_high_error"}:
        path_mse = np.mean((pool_forecasts - pool_truth) ** 2, axis=1)
        chosen = select_top_score_indices(
            candidate_indices=candidate_indices,
            pool_dates=pool_dates,
            scores=path_mse,
            k_examples=k_examples,
        )
    elif selection_mode == "similarity":
        sample_feat = _history_distance_features(current_history, feature_window)
        cand_feats = np.vstack(
            [_history_distance_features(pool_histories[idx], feature_window) for idx in candidate_indices]
        )
        distances = np.linalg.norm(cand_feats - sample_feat, axis=1)
        chosen = candidate_indices[np.argsort(distances)[: int(min(k_examples, len(candidate_indices)))]]
        chosen = chosen[np.argsort(pool_dates[chosen])]
    else:
        chosen = candidate_indices[np.argsort(pool_dates[candidate_indices])][-int(min(k_examples, len(candidate_indices))):]
        chosen = chosen[np.argsort(pool_dates[chosen])]
    return np.asarray(chosen, dtype=int)


def _build_pool_example(
    *,
    idx: int,
    pool_dates: np.ndarray,
    pool_histories: np.ndarray,
    pool_forecasts: np.ndarray,
    pool_truth: np.ndarray,
    history_points: int,
    history_date_arrays: Sequence[Sequence[object]] | None = None,
    sentiment_map: Mapping[pd.Timestamp, float] | None = None,
    sentiment_points: int | None = None,
    pool_windows: np.ndarray | None = None,
    feature_cols: Sequence[str] | None = None,
    target_col: str | None = None,
    max_exogenous_features: int | None = None,
    preferred_features: Sequence[str] | None = None,
    selection_role: str | None = None,
    retrieval_tag: str | None = None,
) -> dict[str, Any]:
    from .run_experiment import build_exogenous_summary

    history = np.asarray(pool_histories[idx][-history_points:], dtype=float)
    example: dict[str, Any] = {
        "history": history,
        "forecast": np.asarray(pool_forecasts[idx], dtype=float),
        "truth": np.asarray(pool_truth[idx], dtype=float),
        "date": str(pd.Timestamp(pool_dates[idx]).date()),
        "sentiment_history": [],
    }
    if history_date_arrays is not None and len(history_date_arrays) > idx:
        history_dates = _normalize_history_dates(history_date_arrays[idx], len(history))
        example["sentiment_history"] = _sentiment_history_from_dates(
            history_dates,
            sentiment_map,
            sentiment_points,
        )
    if (
        pool_windows is not None
        and feature_cols is not None
        and target_col is not None
        and max_exogenous_features is not None
    ):
        example["exogenous_summary"] = build_exogenous_summary(
            window=np.asarray(pool_windows[idx], dtype=float),
            feature_cols=list(feature_cols),
            target_col=str(target_col),
            max_features=int(max_exogenous_features),
            preferred_features=list(preferred_features) if preferred_features is not None else None,
        )
    case_summary: dict[str, Any] = {}
    if selection_role:
        case_summary["selection_role"] = str(selection_role)
    if case_summary:
        example["case_summary"] = case_summary
    if retrieval_tag:
        example["retrieval_tag"] = str(retrieval_tag)
    return example


def _select_teaching_examples(
    *,
    pool_dates: np.ndarray,
    pool_histories: np.ndarray,
    pool_forecasts: np.ndarray,
    pool_truth: np.ndarray,
    origin_date: pd.Timestamp,
    current_history: np.ndarray,
    history_points: int,
    selection_mode: str,
    k_examples: int,
    feature_window: int,
    lookback_days: int | None,
    history_date_arrays: Sequence[Sequence[object]] | None = None,
    sentiment_map: Mapping[pd.Timestamp, float] | None = None,
    sentiment_points: int | None = None,
    pool_windows: np.ndarray | None = None,
    feature_cols: Sequence[str] | None = None,
    target_col: str | None = None,
    max_exogenous_features: int | None = None,
    preferred_features: Sequence[str] | None = None,
) -> list[dict[str, Any]]:
    chosen = _candidate_example_indices(
        pool_dates=pool_dates,
        pool_histories=pool_histories,
        pool_forecasts=pool_forecasts,
        pool_truth=pool_truth,
        origin_date=origin_date,
        current_history=current_history,
        selection_mode=selection_mode,
        k_examples=k_examples,
        feature_window=feature_window,
        lookback_days=lookback_days,
    )
    if chosen.size == 0:
        return []

    examples: list[dict[str, Any]] = []
    for idx in chosen:
        examples.append(
            _build_pool_example(
                idx=int(idx),
                pool_dates=pool_dates,
                pool_histories=pool_histories,
                pool_forecasts=pool_forecasts,
                pool_truth=pool_truth,
                history_points=history_points,
                history_date_arrays=history_date_arrays,
                sentiment_map=sentiment_map,
                sentiment_points=sentiment_points,
                pool_windows=pool_windows,
                feature_cols=feature_cols,
                target_col=target_col,
                max_exogenous_features=max_exogenous_features,
                preferred_features=preferred_features,
            )
        )
    return examples


def _build_online_memory_example(
    record: Mapping[str, Any],
    selection_role: str,
) -> dict[str, Any]:
    case_summary = dict(record.get("case_summary") or {})
    case_summary["selection_role"] = str(selection_role)
    case_summary["retrieval_tag"] = str(record.get("retrieval_tag", ""))
    case_summary["memory_helpfulness_score"] = f"{float(record.get('helpfulness_score', 0.0)):+.4f}"
    case_summary["memory_admission_label"] = str(record.get("admission_label", ""))
    case_summary["memory_rank"] = int(record.get("memory_rank", 0))
    case_summary["memory_similarity"] = f"{float(record.get('memory_similarity', 0.0)):.3f}"
    if record.get("memory_horizon") is not None:
        case_summary["memory_horizon"] = int(record.get("memory_horizon"))
    return {
        "history": np.asarray(record.get("history", []), dtype=float),
        "forecast": np.asarray(record.get("forecast", []), dtype=float),
        "truth": np.asarray(record.get("truth", []), dtype=float),
        "date": str(record.get("date", "")),
        "sentiment_history": list(record.get("sentiment_history") or []),
        "exogenous_summary": record.get("exogenous_summary"),
        "case_summary": case_summary,
        "retrieval_tag": str(record.get("retrieval_tag") or ""),
    }


def _fit_live_learned_gate_bundle(
    *,
    gate_feature_rows: Sequence[Mapping[str, Any]],
    y_true: np.ndarray,
    base_pred: np.ndarray,
    llm_pred: np.ndarray,
    gate_cfg: Mapping[str, Any] | None,
    selection_payload: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    from .run_experiment import (
        build_online_memory_gate_feature_frame,
        build_online_memory_gate_labels,
        fit_online_memory_learned_gate,
    )

    learned_cfg = _resolve_live_learned_gate_cfg(gate_cfg, selection_payload)
    if learned_cfg is None:
        return None

    feature_df = build_online_memory_gate_feature_frame([dict(row) for row in gate_feature_rows])
    if feature_df.empty:
        return None

    labels, _ = build_online_memory_gate_labels(
        y_true=np.asarray(y_true, dtype=float),
        base_pred=np.asarray(base_pred, dtype=float),
        llm_pred=np.asarray(llm_pred, dtype=float),
        label_cfg=learned_cfg,
    )
    fit_cfg = dict(learned_cfg)
    fit_cfg["use_all_for_fit"] = True
    bundle, _ = fit_online_memory_learned_gate(
        feature_df=feature_df,
        labels=np.asarray(labels, dtype=int),
        learned_cfg=fit_cfg,
    )
    if bundle is None:
        return None

    if selection_payload and str(selection_payload.get("gate_mode", "")).strip().lower() == "learned":
        bundle["threshold"] = float(selection_payload.get("threshold", learned_cfg.get("threshold", 0.5)))
        bundle["regime_thresholds"] = dict(selection_payload.get("regime_thresholds") or {})
        bundle["regime_threshold_cfg"] = dict(selection_payload.get("regime_threshold_cfg") or {})
        bundle["selection_path"] = str(selection_payload.get("selection_path", ""))
    else:
        bundle["threshold"] = float(learned_cfg.get("threshold", 0.5))
        bundle["regime_thresholds"] = dict(learned_cfg.get("regime_thresholds") or {})
        bundle["regime_threshold_cfg"] = dict(learned_cfg.get("regime_threshold_cfg") or {})
    return bundle


def _simulate_live_online_memory_history(
    *,
    method_name: str,
    refiner: Any,
    pool_dates: np.ndarray,
    pool_histories: np.ndarray,
    pool_forecasts: np.ndarray,
    pool_truth: np.ndarray,
    pool_windows: np.ndarray | None,
    panel_dates: np.ndarray,
    date_to_idx: Mapping[pd.Timestamp, int],
    llm_cfg: Mapping[str, Any],
    target_col: str,
    feature_cols: Sequence[str],
    max_exogenous_features: int,
    preferred_feature_order: Sequence[str] | None,
    sentiment_map: Mapping[pd.Timestamp, float] | None,
) -> dict[str, Any]:
    from .run_experiment import (
        _profile_regime_tag,
        build_history_dates,
        build_online_memory_gate_feature_row,
        build_realized_availability_dates,
        compute_online_memory_helpfulness,
        compute_online_memory_horizon_gains,
        classify_online_memory_admission,
        select_online_memory_records,
    )
    from .run_experiment import build_exogenous_summary

    cot_cfg = dict(llm_cfg.get("cot_rf", {}) or {})
    online_memory_policy_cfg = dict(cot_cfg.get("online_memory_policy", {}) or {})
    history_points = int(llm_cfg.get("history_points", 18))
    feature_window = int(cot_cfg.get("feature_window", 18))
    selection_mode = str(cot_cfg.get("example_selection", "recent"))
    k_examples = int(cot_cfg.get("k_examples", 5))
    lookback_days = cot_cfg.get("lookback_days")
    lookback_days = int(lookback_days) if lookback_days is not None else None
    sentiment_points = int((llm_cfg.get("sentiment", {}) or {}).get("history_points", history_points))

    support_examples = int(online_memory_policy_cfg.get("support_examples", k_examples))
    positive_examples = int(online_memory_policy_cfg.get("positive_examples", 2))
    negative_examples = int(online_memory_policy_cfg.get("negative_examples", 1))
    max_total_examples = int(
        online_memory_policy_cfg.get(
            "max_total_examples",
            support_examples + positive_examples + negative_examples,
        )
    )
    positive_margin = float(online_memory_policy_cfg.get("positive_margin", 0.01))
    negative_margin = float(online_memory_policy_cfg.get("negative_margin", -0.01))
    warmup_min_realized = int(online_memory_policy_cfg.get("warmup_min_realized", 4))
    store_neutral = bool(online_memory_policy_cfg.get("store_neutral", False))
    admission_cfg = dict(online_memory_policy_cfg.get("admission", {}) or {})
    regime_cfg = dict(online_memory_policy_cfg.get("regime", {}) or {})
    prototype_cfg = dict(online_memory_policy_cfg.get("prototype", {}) or {})
    horizon_cfg = dict(online_memory_policy_cfg.get("horizon_specific", {}) or {})
    horizon_enabled = bool(horizon_cfg.get("enabled", False))
    split_memory_banks = bool(horizon_cfg.get("split_memory_banks", False))
    admission_horizons = sorted(
        {
            int(value)
            for value in (horizon_cfg.get("admission_horizons") or ([20, 30] if horizon_enabled else [len(pool_forecasts[0])]))
            if 1 <= int(value) <= int(pool_forecasts.shape[1])
        }
    )
    if not admission_horizons:
        admission_horizons = [min(int(pool_forecasts.shape[1]), 30)]
    positive_examples_per_horizon = int(
        horizon_cfg.get("positive_examples_per_horizon", max(1, positive_examples))
    )
    negative_examples_per_horizon = int(
        horizon_cfg.get("negative_examples_per_horizon", max(1, negative_examples))
    )
    regime_enabled = bool(regime_cfg.get("enabled", False))
    regime_min_overlap = int(regime_cfg.get("min_overlap", 3))
    prototype_enabled = bool(prototype_cfg.get("enabled", False))
    prototype_top_pool_size = int(
        prototype_cfg.get("top_pool_size", max(positive_examples + negative_examples, 4))
    )
    augment_case_profiles_with_retrieval = bool(cot_cfg.get("augment_case_profiles_with_retrieval", False))
    retrieval_feature_columns = cot_cfg.get("retrieval_feature_columns")
    retrieval_max_features = int(
        cot_cfg.get(
            "retrieval_max_features",
            max(len(retrieval_feature_columns or []), max_exogenous_features),
        )
    )

    history_date_arrays = build_history_dates(
        panel_dates=np.asarray(panel_dates),
        date_to_idx=dict(date_to_idx),
        pred_dates=np.asarray(pool_dates),
        history_points=history_points,
    )
    availability_dates = {
        int(horizon): build_realized_availability_dates(
            panel_dates=np.asarray(panel_dates),
            date_to_idx=dict(date_to_idx),
            pred_dates=np.asarray(pool_dates),
            realization_horizon=int(horizon),
        )
        for horizon in admission_horizons
    }

    predictions = np.asarray(pool_forecasts, dtype=float).copy()
    gate_feature_rows: list[dict[str, Any]] = []
    online_memory_records: list[dict[str, Any]] = []

    for row_idx, origin_date in enumerate(pd.to_datetime(pool_dates)):
        full_history = np.asarray(pool_histories[row_idx], dtype=float)
        history = np.asarray(full_history[-history_points:], dtype=float)
        history_dates = _normalize_history_dates(history_date_arrays[row_idx], len(history))
        sample_window = None if pool_windows is None else np.asarray(pool_windows[row_idx], dtype=float)
        sample_profile = _reference_case_profile(
            history=full_history,
            forecast=np.asarray(pool_forecasts[row_idx], dtype=float),
            window=sample_window,
            feature_window=feature_window,
            augment_with_retrieval=augment_case_profiles_with_retrieval,
            feature_cols=feature_cols,
            target_col=target_col,
            preferred_features=preferred_feature_order,
            retrieval_feature_columns=retrieval_feature_columns,
            retrieval_max_features=retrieval_max_features,
        )
        reference_tag = _profile_regime_tag(sample_profile)
        available_records = [
            dict(record)
            for record in online_memory_records
            if pd.Timestamp(record.get("availability_date")).to_datetime64() <= origin_date.to_datetime64()
            and str(record.get("admission_label", "")).strip().lower() in {"positive", "negative"}
        ]

        support = _select_teaching_examples(
            pool_dates=pool_dates,
            pool_histories=pool_histories,
            pool_forecasts=pool_forecasts,
            pool_truth=pool_truth,
            origin_date=pd.Timestamp(origin_date),
            current_history=full_history,
            history_points=history_points,
            selection_mode=selection_mode,
            k_examples=support_examples,
            feature_window=feature_window,
            lookback_days=lookback_days,
            history_date_arrays=history_date_arrays,
            sentiment_map=sentiment_map,
            sentiment_points=sentiment_points,
            pool_windows=pool_windows,
            feature_cols=feature_cols,
            target_col=target_col,
            max_exogenous_features=max_exogenous_features,
            preferred_features=preferred_feature_order,
        )

        if split_memory_banks and horizon_enabled:
            positive_records_by_horizon: dict[int, list[dict[str, Any]]] = {}
            negative_records_by_horizon: dict[int, list[dict[str, Any]]] = {}
            for memory_horizon in admission_horizons:
                positive_records_by_horizon[int(memory_horizon)] = select_online_memory_records(
                    available_records,
                    sample_profile,
                    label="positive",
                    k_examples=positive_examples_per_horizon,
                    reference_tag=reference_tag if regime_enabled else None,
                    min_tag_overlap=regime_min_overlap,
                    target_horizons=[int(memory_horizon)],
                    prototype_enabled=prototype_enabled,
                    prototype_top_pool_size=prototype_top_pool_size,
                )
                negative_records_by_horizon[int(memory_horizon)] = select_online_memory_records(
                    available_records,
                    sample_profile,
                    label="negative",
                    k_examples=negative_examples_per_horizon,
                    reference_tag=reference_tag if regime_enabled else None,
                    min_tag_overlap=regime_min_overlap,
                    target_horizons=[int(memory_horizon)],
                    prototype_enabled=prototype_enabled,
                    prototype_top_pool_size=prototype_top_pool_size,
                )
            positive_records = [
                dict(record)
                for horizon in admission_horizons
                for record in positive_records_by_horizon.get(int(horizon), [])
            ]
            negative_records = [
                dict(record)
                for horizon in admission_horizons
                for record in negative_records_by_horizon.get(int(horizon), [])
            ]
        else:
            positive_records = select_online_memory_records(
                available_records,
                sample_profile,
                label="positive",
                k_examples=positive_examples,
                reference_tag=reference_tag if regime_enabled else None,
                min_tag_overlap=regime_min_overlap,
                target_horizons=admission_horizons if horizon_enabled else None,
                prototype_enabled=prototype_enabled,
                prototype_top_pool_size=prototype_top_pool_size,
            )
            negative_records = select_online_memory_records(
                available_records,
                sample_profile,
                label="negative",
                k_examples=negative_examples,
                reference_tag=reference_tag if regime_enabled else None,
                min_tag_overlap=regime_min_overlap,
                target_horizons=admission_horizons if horizon_enabled else None,
                prototype_enabled=prototype_enabled,
                prototype_top_pool_size=prototype_top_pool_size,
            )
            positive_records_by_horizon = {}
            negative_records_by_horizon = {}

        feature_row = build_online_memory_gate_feature_row(
            positive_records=positive_records,
            negative_records=negative_records,
            base_forecast=np.asarray(pool_forecasts[row_idx], dtype=float),
            current_price=float(history[-1]) if history.size else 0.0,
            warmup_ready=len(available_records) >= warmup_min_realized,
            reference_profile=sample_profile,
            support_example_count=len(support),
        )
        gate_feature_rows.append(dict(feature_row))

        memory_examples: list[dict[str, Any]] = []
        if split_memory_banks and horizon_enabled:
            for memory_horizon in admission_horizons:
                for record in positive_records_by_horizon.get(int(memory_horizon), []):
                    memory_examples.append(
                        _build_online_memory_example(record, f"positive_memory_h{int(memory_horizon)}")
                    )
                for record in negative_records_by_horizon.get(int(memory_horizon), []):
                    memory_examples.append(
                        _build_online_memory_example(record, f"negative_memory_h{int(memory_horizon)}")
                    )
        else:
            memory_examples.extend(
                _build_online_memory_example(record, "positive_memory")
                for record in positive_records
            )
            memory_examples.extend(
                _build_online_memory_example(record, "negative_memory")
                for record in negative_records
            )
        teaching_examples = list(support) + memory_examples
        if max_total_examples > 0:
            teaching_examples = teaching_examples[:max_total_examples]

        refined = None
        if teaching_examples:
            sample_exogenous = None
            if sample_window is not None:
                sample_exogenous = build_exogenous_summary(
                    window=sample_window,
                    feature_cols=list(feature_cols),
                    target_col=str(target_col),
                    max_features=int(max_exogenous_features),
                    preferred_features=list(preferred_feature_order) if preferred_feature_order is not None else None,
                )
            sample_sentiment = _sentiment_history_from_dates(
                history_dates,
                sentiment_map,
                sentiment_points,
            )
            refined, _meta = refiner.refine(
                method=method_name,
                history=history,
                dates=history_dates,
                tsm_forecast=np.asarray(pool_forecasts[row_idx], dtype=float),
                pred_len=int(pool_forecasts.shape[1]),
                exogenous_summary=sample_exogenous,
                price_base=float(history[-1]) if history.size else 0.0,
                teaching_examples=teaching_examples,
                sentiment_history=np.asarray(sample_sentiment, dtype=float) if sample_sentiment else None,
            )
        if refined is None:
            refined = np.asarray(pool_forecasts[row_idx], dtype=float)
        refined = np.asarray(refined, dtype=float)
        if refined.shape != np.asarray(pool_forecasts[row_idx], dtype=float).shape:
            refined = np.asarray(pool_forecasts[row_idx], dtype=float)
        predictions[row_idx] = refined

        helpfulness_score = compute_online_memory_helpfulness(
            np.asarray(pool_forecasts[row_idx], dtype=float),
            refined,
            np.asarray(pool_truth[row_idx], dtype=float),
            path_weight=float(admission_cfg.get("path_weight", 0.40)),
            h5_weight=float(admission_cfg.get("h5_weight", 0.05)),
            h20_weight=float(admission_cfg.get("h20_weight", 0.30)),
            h30_weight=float(admission_cfg.get("h30_weight", 0.25)),
        )
        horizon_gains = compute_online_memory_horizon_gains(
            np.asarray(pool_forecasts[row_idx], dtype=float),
            refined,
            np.asarray(pool_truth[row_idx], dtype=float),
        )
        memory_horizons = admission_horizons if horizon_enabled else [int(pool_forecasts.shape[1])]
        for memory_horizon in memory_horizons:
            horizon_idx = int(memory_horizon) - 1
            if horizon_idx >= pool_forecasts.shape[1] or horizon_idx >= pool_truth.shape[1]:
                continue
            record_helpfulness = float(
                horizon_gains.get(int(memory_horizon), helpfulness_score)
                if horizon_enabled
                else helpfulness_score
            )
            admission_label = classify_online_memory_admission(
                record_helpfulness,
                horizon_gains,
                positive_margin=positive_margin,
                negative_margin=negative_margin,
                positive_min_h20_gain=admission_cfg.get("positive_min_h20_gain"),
                positive_min_h30_gain=admission_cfg.get("positive_min_h30_gain"),
                positive_max_h5_damage=admission_cfg.get("positive_max_h5_damage"),
                negative_max_h20_gain=admission_cfg.get("negative_max_h20_gain"),
                negative_max_h30_gain=admission_cfg.get("negative_max_h30_gain"),
                target_horizon=int(memory_horizon) if horizon_enabled else None,
                positive_min_target_gain=admission_cfg.get("positive_min_target_gain"),
                negative_max_target_gain=admission_cfg.get("negative_max_target_gain"),
            )
            if admission_label not in {"positive", "negative"} and not store_neutral:
                continue
            availability = availability_dates.get(int(memory_horizon))
            availability_date = (
                str(pd.Timestamp(availability[row_idx]).date())
                if availability is not None and row_idx < len(availability)
                else str(pd.Timestamp(origin_date).date())
            )
            online_memory_records.append(
                {
                    "availability_date": availability_date,
                    "date": str(pd.Timestamp(origin_date).date()),
                    "history": np.asarray(history, dtype=float),
                    "forecast": np.asarray(refined, dtype=float),
                    "truth": np.asarray(pool_truth[row_idx], dtype=float),
                    "sentiment_history": _sentiment_history_from_dates(
                        history_dates,
                        sentiment_map,
                        sentiment_points,
                    ),
                    "exogenous_summary": (
                        build_exogenous_summary(
                            window=sample_window,
                            feature_cols=list(feature_cols),
                            target_col=str(target_col),
                            max_features=int(max_exogenous_features),
                            preferred_features=list(preferred_feature_order) if preferred_feature_order is not None else None,
                        )
                        if sample_window is not None
                        else None
                    ),
                    "case_profile": np.asarray(sample_profile, dtype=float),
                    "case_summary": {"support_example_count": int(len(support))},
                    "retrieval_tag": str(reference_tag),
                    "helpfulness_score": float(record_helpfulness),
                    "helpfulness_path_score": float(helpfulness_score),
                    "horizon_gains": {str(key): float(value) for key, value in horizon_gains.items()},
                    "admission_label": str(admission_label),
                    "memory_horizon": int(memory_horizon) if horizon_enabled else None,
                }
            )

    return {
        "records": online_memory_records,
        "gate_feature_rows": gate_feature_rows,
        "predictions": np.asarray(predictions, dtype=float),
        "history_date_arrays": history_date_arrays,
    }


def _build_current_online_memory_context(
    *,
    current_origin_date: pd.Timestamp,
    current_history: np.ndarray,
    current_forecast: np.ndarray,
    current_price: float,
    current_window: np.ndarray | None,
    current_history_dates: Sequence[object],
    support_examples: Sequence[dict[str, Any]],
    online_memory_records: Sequence[Mapping[str, Any]],
    llm_cfg: Mapping[str, Any],
    gate_cfg: Mapping[str, Any],
    learned_gate_bundle: Mapping[str, Any] | None,
    target_col: str,
    feature_cols: Sequence[str],
    max_exogenous_features: int,
    preferred_feature_order: Sequence[str] | None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    from .run_experiment import _profile_regime_tag, build_online_memory_gate_decision, select_online_memory_records

    cot_cfg = dict(llm_cfg.get("cot_rf", {}) or {})
    online_memory_policy_cfg = dict(cot_cfg.get("online_memory_policy", {}) or {})
    warmup_min_realized = int(online_memory_policy_cfg.get("warmup_min_realized", 4))
    positive_examples = int(online_memory_policy_cfg.get("positive_examples", 2))
    negative_examples = int(online_memory_policy_cfg.get("negative_examples", 1))
    max_total_examples = int(
        online_memory_policy_cfg.get(
            "max_total_examples",
            int(online_memory_policy_cfg.get("support_examples", len(support_examples)))
            + positive_examples
            + negative_examples,
        )
    )
    regime_cfg = dict(online_memory_policy_cfg.get("regime", {}) or {})
    prototype_cfg = dict(online_memory_policy_cfg.get("prototype", {}) or {})
    horizon_cfg = dict(online_memory_policy_cfg.get("horizon_specific", {}) or {})
    horizon_enabled = bool(horizon_cfg.get("enabled", False))
    split_memory_banks = bool(horizon_cfg.get("split_memory_banks", False))
    admission_horizons = sorted(
        {
            int(value)
            for value in (horizon_cfg.get("admission_horizons") or ([20, 30] if horizon_enabled else [len(current_forecast)]))
            if 1 <= int(value) <= int(len(current_forecast))
        }
    )
    if not admission_horizons:
        admission_horizons = [min(int(len(current_forecast)), 30)]
    positive_examples_per_horizon = int(
        horizon_cfg.get("positive_examples_per_horizon", max(1, positive_examples))
    )
    negative_examples_per_horizon = int(
        horizon_cfg.get("negative_examples_per_horizon", max(1, negative_examples))
    )
    regime_enabled = bool(regime_cfg.get("enabled", False))
    regime_min_overlap = int(regime_cfg.get("min_overlap", 3))
    prototype_enabled = bool(prototype_cfg.get("enabled", False))
    prototype_top_pool_size = int(
        prototype_cfg.get("top_pool_size", max(positive_examples + negative_examples, 4))
    )
    feature_window = int(cot_cfg.get("feature_window", 18))
    augment_case_profiles_with_retrieval = bool(cot_cfg.get("augment_case_profiles_with_retrieval", False))
    retrieval_feature_columns = cot_cfg.get("retrieval_feature_columns")
    retrieval_max_features = int(
        cot_cfg.get(
            "retrieval_max_features",
            max(len(retrieval_feature_columns or []), max_exogenous_features),
        )
    )
    reference_profile = _reference_case_profile(
        history=np.asarray(current_history, dtype=float),
        forecast=np.asarray(current_forecast, dtype=float),
        window=current_window,
        feature_window=feature_window,
        augment_with_retrieval=augment_case_profiles_with_retrieval,
        feature_cols=feature_cols,
        target_col=target_col,
        preferred_features=preferred_feature_order,
        retrieval_feature_columns=retrieval_feature_columns,
        retrieval_max_features=retrieval_max_features,
    )
    reference_tag = _profile_regime_tag(reference_profile)
    available_records = [
        dict(record)
        for record in online_memory_records
        if pd.Timestamp(record.get("availability_date")).to_datetime64() <= pd.Timestamp(current_origin_date).to_datetime64()
        and str(record.get("admission_label", "")).strip().lower() in {"positive", "negative"}
    ]

    if split_memory_banks and horizon_enabled:
        positive_records_by_horizon: dict[int, list[dict[str, Any]]] = {}
        negative_records_by_horizon: dict[int, list[dict[str, Any]]] = {}
        for memory_horizon in admission_horizons:
            positive_records_by_horizon[int(memory_horizon)] = select_online_memory_records(
                available_records,
                reference_profile,
                label="positive",
                k_examples=positive_examples_per_horizon,
                reference_tag=reference_tag if regime_enabled else None,
                min_tag_overlap=regime_min_overlap,
                target_horizons=[int(memory_horizon)],
                prototype_enabled=prototype_enabled,
                prototype_top_pool_size=prototype_top_pool_size,
            )
            negative_records_by_horizon[int(memory_horizon)] = select_online_memory_records(
                available_records,
                reference_profile,
                label="negative",
                k_examples=negative_examples_per_horizon,
                reference_tag=reference_tag if regime_enabled else None,
                min_tag_overlap=regime_min_overlap,
                target_horizons=[int(memory_horizon)],
                prototype_enabled=prototype_enabled,
                prototype_top_pool_size=prototype_top_pool_size,
            )
        positive_records = [
            dict(record)
            for horizon in admission_horizons
            for record in positive_records_by_horizon.get(int(horizon), [])
        ]
        negative_records = [
            dict(record)
            for horizon in admission_horizons
            for record in negative_records_by_horizon.get(int(horizon), [])
        ]
    else:
        positive_records = select_online_memory_records(
            available_records,
            reference_profile,
            label="positive",
            k_examples=positive_examples,
            reference_tag=reference_tag if regime_enabled else None,
            min_tag_overlap=regime_min_overlap,
            target_horizons=admission_horizons if horizon_enabled else None,
            prototype_enabled=prototype_enabled,
            prototype_top_pool_size=prototype_top_pool_size,
        )
        negative_records = select_online_memory_records(
            available_records,
            reference_profile,
            label="negative",
            k_examples=negative_examples,
            reference_tag=reference_tag if regime_enabled else None,
            min_tag_overlap=regime_min_overlap,
            target_horizons=admission_horizons if horizon_enabled else None,
            prototype_enabled=prototype_enabled,
            prototype_top_pool_size=prototype_top_pool_size,
        )
        positive_records_by_horizon = {}
        negative_records_by_horizon = {}

    gate_decision = build_online_memory_gate_decision(
        positive_records=positive_records,
        negative_records=negative_records,
        base_forecast=np.asarray(current_forecast, dtype=float),
        current_price=float(current_price),
        warmup_ready=len(available_records) >= warmup_min_realized,
        gate_cfg=dict(gate_cfg),
        learned_gate_bundle=dict(learned_gate_bundle) if learned_gate_bundle is not None else None,
        reference_profile=reference_profile,
        support_example_count=len(support_examples),
    )

    memory_examples: list[dict[str, Any]] = []
    if split_memory_banks and horizon_enabled:
        for memory_horizon in admission_horizons:
            for record in positive_records_by_horizon.get(int(memory_horizon), []):
                memory_examples.append(
                    _build_online_memory_example(record, f"positive_memory_h{int(memory_horizon)}")
                )
            for record in negative_records_by_horizon.get(int(memory_horizon), []):
                memory_examples.append(
                    _build_online_memory_example(record, f"negative_memory_h{int(memory_horizon)}")
                )
    else:
        memory_examples.extend(
            _build_online_memory_example(record, "positive_memory")
            for record in positive_records
        )
        memory_examples.extend(
            _build_online_memory_example(record, "negative_memory")
            for record in negative_records
        )

    teaching_examples = list(support_examples) + memory_examples
    if max_total_examples > 0:
        teaching_examples = teaching_examples[:max_total_examples]
    gate_decision = {
        **gate_decision,
        "support_example_count": int(len(support_examples)),
        "positive_example_count": int(len(positive_records)),
        "negative_example_count": int(len(negative_records)),
        "available_record_count": int(len(available_records)),
        "reference_tag": str(reference_tag),
        "selection_history_end": str(pd.Timestamp(current_origin_date).date()),
        "selection_history_points": int(len(current_history_dates)),
    }
    return teaching_examples, gate_decision


def _load_live_feature_panel(
    *,
    config_raw: Mapping[str, Any],
    data_dir: str | Path,
    run_dir: str | Path,
) -> pd.DataFrame:
    from .data import build_panel, select_feature_columns

    target_mode = str(config_raw.get("target", {}).get("mode", "returns")).lower()
    target_col = "y_return" if target_mode == "returns" else "y"
    feature_cfg = config_raw.get("features", {}) or {}
    max_exogenous_features = int(feature_cfg.get("max_exogenous_features_model", 10))
    preferred_feature_order = feature_cfg.get("preferred_feature_order")

    panel, _schema = build_panel(data_dir=Path(data_dir), config=config_raw, save_path=Path(run_dir) / "data")
    feature_cols = select_feature_columns(
        panel,
        target_col=target_col,
        max_exogenous_features=max_exogenous_features,
        preferred_feature_order=preferred_feature_order,
    )
    panel = panel[["date"] + [col for col in feature_cols if col in panel.columns]].copy()
    panel["date"] = pd.to_datetime(panel["date"]).dt.normalize()
    return panel.sort_values("date").reset_index(drop=True)


def _build_live_rows_from_feature_panel(
    *,
    config_raw: Mapping[str, Any],
    panel: pd.DataFrame,
    run_dir: str | Path,
    artifact_run_dir: str | Path | None = None,
    model_version: str = DEFAULT_MODEL_VERSION,
    record_source: str = "live_run",
    enable_llm: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    from .frontend_live_bundle import _build_live_inputs, _predict_prices, train_tsm_on_prefix
    from .llm import LLMRefiner
    from .run_experiment import (
        build_exogenous_summary,
        infer_llm_market_name,
        load_daily_sentiment,
        seed_llm_response_cache,
    )

    run_dir = Path(run_dir).resolve()
    artifact_run_dir = Path(artifact_run_dir).resolve() if artifact_run_dir is not None else None
    run_metadata = _read_run_metadata(run_dir) if (run_dir / "config_resolved.yaml").exists() else {
        "git_hash": "",
        "timestamp": "",
    }
    target_mode = str(config_raw.get("target", {}).get("mode", "returns")).lower()
    target_col = "y_return" if target_mode == "returns" else "y"
    pred_len = int(config_raw.get("time_series", {}).get("pred_len", 30))
    seq_len = int(config_raw.get("time_series", {}).get("seq_len", 120))
    label_len = int(config_raw.get("time_series", {}).get("label_len", 30))
    feature_cfg = config_raw.get("features", {}) or {}
    preferred_feature_order = feature_cfg.get("preferred_feature_order")

    trained = train_tsm_on_prefix(prefix_panel=panel.copy(), config_raw=dict(config_raw))
    batch_size = int(config_raw.get("model", {}).get("batch_size", 32))

    live_panel = panel.copy()
    live_panel.attrs["pred_len"] = pred_len
    X_enc, X_dec, last_price, history_prices = _build_live_inputs(
        prefix_panel=live_panel,
        feature_cols=trained.windows.feature_cols,
        target_col=target_col,
        seq_len=seq_len,
        label_len=label_len,
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

    llm_cfg = copy.deepcopy(config_raw.get("llm", {}) or {})
    llm_cfg.setdefault("currency", config_raw.get("target", {}).get("currency", "GBP"))
    llm_cfg.setdefault("market_name", infer_llm_market_name(config_raw.get("target", {})))
    if "hdelta" in config_raw:
        llm_cfg["hdelta"] = copy.deepcopy(config_raw.get("hdelta", {}) or {})

    methods = [str(method) for method in llm_cfg.get("methods", []) if str(method).strip()]
    method_name = methods[0] if methods else DEFAULT_METHOD_NAME
    llm_pred = np.asarray(base_pred, dtype=float)
    llm_metadata: dict[str, Any] = {"fallback_to_tsm": True}

    if enable_llm and methods:
        history_points = int(llm_cfg.get("history_points", min(pred_len, len(history_prices))))
        cot_cfg = llm_cfg.get("cot_rf", {}) or {}
        sentiment_cfg = llm_cfg.get("sentiment", {}) or {}
        sentiment_map: dict[pd.Timestamp, float] | None = None
        if sentiment_cfg.get("enabled", False):
            sentiment_map = load_daily_sentiment(
                str(Path(sentiment_cfg.get("path", "data/news/daily_sentiment.csv")).resolve()),
                date_col=str(sentiment_cfg.get("date_col", "seendate")),
                score_col=str(sentiment_cfg.get("score_col", "sent_score")),
            )
        sent_points = int(sentiment_cfg.get("history_points", history_points))

        cache_dir = run_dir / "llm" / "live_forecast" / "cache"
        log_dir = run_dir / "llm" / "live_forecast" / "logs"
        if artifact_run_dir is not None and artifact_run_dir != run_dir:
            try:
                copied = seed_llm_response_cache(cache_dir, [artifact_run_dir])
                if copied:
                    logger.info("Seeded %d cached LLM responses from %s", copied, artifact_run_dir)
            except Exception as exc:
                logger.warning("Failed to seed LLM response cache from %s: %s", artifact_run_dir, exc)

        refiner = LLMRefiner(
            config=llm_cfg,
            cache_dir=cache_dir,
            log_dir=log_dir,
        )

        current_origin_date = pd.Timestamp(panel["date"].iloc[-1]).normalize()
        current_history = np.asarray(history_prices[-history_points:], dtype=float)
        current_history_dates = [
            str(ts.date())
            for ts in pd.to_datetime(panel["date"]).tail(history_points).tolist()
        ]
        current_exogenous_summary = build_exogenous_summary(
            window=np.asarray(X_enc, dtype=float),
            feature_cols=trained.windows.feature_cols,
            target_col=target_col,
            max_features=int(llm_cfg.get("max_exogenous_features", 6)),
            preferred_features=preferred_feature_order,
        )
        current_sentiment = _sentiment_history_from_dates(
            current_history_dates,
            sentiment_map,
            sent_points,
        )
        current_sentiment_history = (
            np.asarray(current_sentiment, dtype=float) if current_sentiment else None
        )

        online_memory_policy_cfg = dict(cot_cfg.get("online_memory_policy", {}) or {})
        online_memory_policy_enabled = bool(
            str(cot_cfg.get("test_pool_mode", "")).strip().lower() == "online_realized_memory"
            and online_memory_policy_cfg.get("enabled", False)
        )

        if online_memory_policy_enabled:
            val_idx = np.arange(trained.val_slice.start, trained.val_slice.stop)
            val_dates = pd.to_datetime(trained.windows.pred_dates[val_idx]).to_numpy()
            val_histories = np.asarray(trained.windows.price_histories[val_idx], dtype=float)
            val_base_pred = np.asarray(trained.val_pred_prices, dtype=float)
            val_truth = np.asarray(trained.val_true_prices, dtype=float)
            val_windows = np.asarray(trained.windows.X_enc[val_idx], dtype=float)
            panel_dates = pd.to_datetime(panel["date"]).to_numpy()
            date_to_idx = {
                pd.Timestamp(date_value).normalize(): idx
                for idx, date_value in enumerate(pd.to_datetime(panel_dates))
            }
            selection_payload = _load_online_memory_gate_selection(
                artifact_run_dir or run_dir,
                method_name,
            )
            active_gate_cfg = _merge_online_memory_gate_cfg(
                online_memory_policy_cfg.get("gate", {}) or {},
                selection_payload,
            )
            simulated = _simulate_live_online_memory_history(
                method_name=method_name,
                refiner=refiner,
                pool_dates=val_dates,
                pool_histories=val_histories,
                pool_forecasts=val_base_pred,
                pool_truth=val_truth,
                pool_windows=val_windows,
                panel_dates=panel_dates,
                date_to_idx=date_to_idx,
                llm_cfg=llm_cfg,
                target_col=target_col,
                feature_cols=trained.windows.feature_cols,
                max_exogenous_features=int(llm_cfg.get("max_exogenous_features", 6)),
                preferred_feature_order=preferred_feature_order,
                sentiment_map=sentiment_map,
            )
            learned_gate_bundle = _fit_live_learned_gate_bundle(
                gate_feature_rows=simulated["gate_feature_rows"],
                y_true=val_truth,
                base_pred=val_base_pred,
                llm_pred=np.asarray(simulated["predictions"], dtype=float),
                gate_cfg=active_gate_cfg,
                selection_payload=selection_payload,
            )
            support_examples = _select_teaching_examples(
                pool_dates=val_dates,
                pool_histories=val_histories,
                pool_forecasts=val_base_pred,
                pool_truth=val_truth,
                origin_date=current_origin_date,
                current_history=np.asarray(history_prices, dtype=float),
                history_points=history_points,
                selection_mode=str(cot_cfg.get("example_selection", "recent")),
                k_examples=int(online_memory_policy_cfg.get("support_examples", cot_cfg.get("k_examples", 5))),
                feature_window=int(cot_cfg.get("feature_window", 18)),
                lookback_days=(
                    int(cot_cfg["lookback_days"])
                    if cot_cfg.get("lookback_days") is not None
                    else None
                ),
                history_date_arrays=simulated["history_date_arrays"],
                sentiment_map=sentiment_map,
                sentiment_points=sent_points,
                pool_windows=val_windows,
                feature_cols=trained.windows.feature_cols,
                target_col=target_col,
                max_exogenous_features=int(llm_cfg.get("max_exogenous_features", 6)),
                preferred_features=preferred_feature_order,
            )
            teaching_examples, gate_decision = _build_current_online_memory_context(
                current_origin_date=current_origin_date,
                current_history=np.asarray(history_prices, dtype=float),
                current_forecast=np.asarray(base_pred, dtype=float),
                current_price=float(last_price),
                current_window=np.asarray(X_enc, dtype=float),
                current_history_dates=current_history_dates,
                support_examples=support_examples,
                online_memory_records=simulated["records"],
                llm_cfg=llm_cfg,
                gate_cfg=active_gate_cfg,
                learned_gate_bundle=learned_gate_bundle,
                target_col=target_col,
                feature_cols=trained.windows.feature_cols,
                max_exogenous_features=int(llm_cfg.get("max_exogenous_features", 6)),
                preferred_feature_order=preferred_feature_order,
            )
            llm_metadata = {
                "fallback_to_tsm": True,
                "online_memory_gate": gate_decision,
                "online_memory_policy_enabled": True,
                "online_memory_record_count": int(len(simulated["records"])),
                "online_memory_history_cases": int(len(val_dates)),
                "online_memory_selection_path": (
                    str(selection_payload.get("selection_path", ""))
                    if selection_payload
                    else ""
                ),
                "online_memory_learned_gate_enabled": bool(learned_gate_bundle is not None),
            }
            if gate_decision.get("apply_llm", True) and teaching_examples:
                refined, refine_meta = refiner.refine(
                    method=method_name,
                    history=current_history,
                    dates=current_history_dates,
                    tsm_forecast=np.asarray(base_pred, dtype=float),
                    pred_len=pred_len,
                    exogenous_summary=current_exogenous_summary,
                    price_base=float(last_price),
                    teaching_examples=teaching_examples,
                    sentiment_history=current_sentiment_history,
                )
                if refined is not None:
                    llm_pred = np.asarray(refined, dtype=float)
                    llm_metadata = {**llm_metadata, **dict(refine_meta or {})}
                    llm_metadata["fallback_to_tsm"] = False
                else:
                    llm_metadata["error"] = "Live online-memory refinement returned no forecast."
            elif not gate_decision.get("apply_llm", True):
                llm_metadata["reason"] = str(gate_decision.get("reason", "memory_gate_blocked"))
            else:
                llm_metadata["reason"] = "no_teaching_examples"
        else:
            pool_dates, pool_histories, pool_forecasts, pool_truth = _pool_predictions(
                trained,
                target_mode,
                batch_size,
            )
            examples = _select_teaching_examples(
                pool_dates=pool_dates,
                pool_histories=pool_histories,
                pool_forecasts=pool_forecasts,
                pool_truth=pool_truth,
                origin_date=current_origin_date,
                current_history=np.asarray(history_prices, dtype=float),
                history_points=history_points,
                selection_mode=str(cot_cfg.get("example_selection", "recent")),
                k_examples=int(cot_cfg.get("k_examples", 5)),
                feature_window=int(cot_cfg.get("feature_window", 18)),
                lookback_days=(
                    int(cot_cfg["lookback_days"])
                    if cot_cfg.get("lookback_days") is not None
                    else None
                ),
            )
            if examples:
                refined, llm_metadata = refiner.refine(
                    method=method_name,
                    history=current_history,
                    dates=current_history_dates,
                    tsm_forecast=np.asarray(base_pred, dtype=float),
                    pred_len=pred_len,
                    exogenous_summary=current_exogenous_summary,
                    price_base=float(last_price),
                    teaching_examples=examples,
                    sentiment_history=current_sentiment_history,
                )
                if refined is not None:
                    llm_pred = np.asarray(refined, dtype=float)
                    llm_metadata = dict(llm_metadata or {})
                    llm_metadata["fallback_to_tsm"] = False

    latest_date = pd.Timestamp(panel["date"].iloc[-1]).normalize()
    latest_date_str = str(latest_date.date())
    target_dates = _next_business_target_dates(latest_date, pred_len)
    generated_at = _iso_utc_now()

    rows = [
        _archive_row(
            run_id=str(Path(run_dir).name),
            forecast_made_on=latest_date_str,
            latest_observed_date=latest_date_str,
            latest_observed_price=float(last_price),
            target_date=target_dates[idx],
            step_index=idx + 1,
            actual=None,
            base_tsm_forecast=float(base_pred[idx]),
            llm_tsm_forecast=float(llm_pred[idx]),
            model_version=model_version,
            model_commit=str(run_metadata.get("git_hash") or ""),
            data_version=latest_date_str,
            is_realized=False,
            generated_at=generated_at,
            record_source=record_source,
            source_run_id=str(Path(run_dir).name),
        )
        for idx in range(pred_len)
    ]
    live_meta = {
        "origin_date": latest_date_str,
        "latest_observed_price": float(last_price),
        "base_forecast": np.asarray(base_pred, dtype=float),
        "llm_forecast": np.asarray(llm_pred, dtype=float),
        "target_dates": target_dates,
        "llm_method": method_name,
        "llm_metadata": llm_metadata,
    }
    return _normalize_archive_frame(pd.DataFrame(rows)), panel[["date", "y"]].copy(), live_meta


def build_current_live_rows(
    *,
    config_raw: dict[str, Any],
    data_dir: str | Path,
    run_dir: str | Path,
    artifact_run_dir: str | Path | None = None,
    model_version: str = DEFAULT_MODEL_VERSION,
    record_source: str = "live_run",
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    run_dir = Path(run_dir).resolve()
    panel = _load_live_feature_panel(
        config_raw=config_raw,
        data_dir=data_dir,
        run_dir=run_dir,
    )
    return _build_live_rows_from_feature_panel(
        config_raw=config_raw,
        panel=panel,
        run_dir=run_dir,
        artifact_run_dir=artifact_run_dir,
        model_version=model_version,
        record_source=record_source,
    )


def build_recent_live_history_rows(
    *,
    config_raw: dict[str, Any],
    data_dir: str | Path,
    run_dir: str | Path,
    start_after_date: str | pd.Timestamp | None,
    artifact_run_dir: str | Path | None = None,
    model_version: str = DEFAULT_MODEL_VERSION,
    record_source: str = "historical_live_backfill",
    include_latest_origin: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    run_dir = Path(run_dir).resolve()
    panel = _load_live_feature_panel(
        config_raw=config_raw,
        data_dir=data_dir,
        run_dir=run_dir,
    )
    panel_prices = panel[["date", "y"]].copy()
    if panel.empty or len(panel) < 2:
        return empty_archive_frame(), panel_prices

    pred_len = int(config_raw.get("time_series", {}).get("pred_len", 30))
    seq_len = int(config_raw.get("time_series", {}).get("seq_len", 120))
    archive_cfg = config_raw.get("frontend_live_bundle", {}) or {}
    min_train_windows = int(archive_cfg.get("min_train_windows", 756))
    min_val_windows = int(archive_cfg.get("min_val_windows", 126))
    earliest_origin_idx = seq_len + pred_len + min_train_windows + min_val_windows - 2

    cutoff = (
        pd.Timestamp(start_after_date).normalize()
        if start_after_date is not None
        else pd.Timestamp.min.normalize()
    )
    last_origin_idx = len(panel) - 1 if include_latest_origin else len(panel) - 2
    if last_origin_idx < 0:
        return empty_archive_frame(), panel_prices

    origin_indices = [
        idx
        for idx in range(max(0, earliest_origin_idx), last_origin_idx + 1)
        if pd.Timestamp(panel["date"].iloc[idx]).normalize() > cutoff
    ]
    if not origin_indices:
        return empty_archive_frame(), panel_prices

    row_frames: list[pd.DataFrame] = []
    for origin_idx in origin_indices:
        prefix_panel = panel.iloc[: origin_idx + 1].copy().reset_index(drop=True)
        rows, _prefix_prices, _meta = _build_live_rows_from_feature_panel(
            config_raw=config_raw,
            panel=prefix_panel,
            run_dir=run_dir,
            artifact_run_dir=artifact_run_dir,
            model_version=model_version,
            record_source=record_source,
            enable_llm=False,
        )
        row_frames.append(rows)

    if not row_frames:
        return empty_archive_frame(), panel_prices
    return _normalize_archive_frame(pd.concat(row_frames, ignore_index=True)), panel_prices


def merge_archive_rows(existing: pd.DataFrame, incoming: pd.DataFrame) -> pd.DataFrame:
    if incoming is None or incoming.empty:
        return _normalize_archive_frame(existing)
    if existing is None or existing.empty:
        return _normalize_archive_frame(incoming)

    left = _normalize_archive_frame(existing)
    right = _normalize_archive_frame(incoming)
    combined = pd.concat([left, right], ignore_index=True)
    combined = combined.drop_duplicates(
        subset=["latest_observed_date", "step_index"],
        keep="last",
    )
    return _normalize_archive_frame(combined)


def backfill_archive_actuals(archive_df: pd.DataFrame, panel: pd.DataFrame) -> pd.DataFrame:
    out = _normalize_archive_frame(archive_df)
    if out.empty:
        return out
    work_panel = panel.copy()
    work_panel["date"] = pd.to_datetime(work_panel["date"]).dt.normalize()
    work_panel["y"] = pd.to_numeric(work_panel["y"], errors="coerce")
    actual_map = {
        str(pd.Timestamp(row.date).date()): float(row.y)
        for row in work_panel.dropna(subset=["date", "y"]).itertuples(index=False)
    }
    missing_mask = out["actual"].isna()
    for idx in out.index[missing_mask]:
        actual = actual_map.get(str(out.at[idx, "target_date"]))
        if actual is None:
            continue
        out.at[idx, "actual"] = float(actual)
        out.at[idx, "is_realized"] = True
    return _normalize_archive_frame(out)


def _latest_origin_rows(archive_df: pd.DataFrame) -> pd.DataFrame:
    if archive_df.empty:
        raise ValueError("Archive is empty; no live forecast rows available.")
    latest_date = archive_df["latest_observed_date"].dropna().astype(str).max()
    subset = archive_df[archive_df["latest_observed_date"].astype(str) == str(latest_date)].copy()
    if subset.empty:
        raise ValueError("Could not resolve latest live forecast rows from archive.")
    return subset.sort_values("step_index").reset_index(drop=True)


def export_public_site_files(
    archive_df: pd.DataFrame,
    export_dir: str | Path,
    *,
    market_name: str = DEFAULT_MARKET_NAME,
) -> dict[str, Any]:
    export_dir = Path(export_dir).resolve()
    export_dir.mkdir(parents=True, exist_ok=True)

    archive_df = _normalize_archive_frame(archive_df)
    live_rows = _latest_origin_rows(archive_df)
    latest_generated_at = str(live_rows["generated_at"].dropna().iloc[-1])
    latest_observed_date = str(live_rows["latest_observed_date"].iloc[0])
    latest_observed_price = float(live_rows["latest_observed_price"].iloc[0])
    model_version = str(live_rows["model_version"].iloc[0] or DEFAULT_MODEL_VERSION)

    forecasts_payload: dict[str, Any] = {}
    for horizon in HEADLINE_HORIZONS:
        row = live_rows[live_rows["step_index"] == int(horizon)]
        if row.empty:
            continue
        item = row.iloc[0]
        llm_value = float(item["llm_tsm_forecast"])
        forecasts_payload[f"{horizon}d"] = {
            "target_date": str(item["target_date"]),
            "base_tsm_value": float(item["base_tsm_forecast"]),
            "llm_tsm_value": llm_value,
            "predicted_change_abs": llm_value - latest_observed_price,
            "predicted_change_pct": (
                ((llm_value / latest_observed_price) - 1.0) * 100.0
                if abs(latest_observed_price) > 1e-8
                else 0.0
            ),
        }

    latest_payload = {
        "market": market_name,
        "generated_at": latest_generated_at,
        "latest_observed_date": latest_observed_date,
        "latest_observed_price": latest_observed_price,
        "model_version": model_version,
        "forecasts": forecasts_payload,
    }
    _write_json(export_dir / "latest.json", latest_payload)

    horizon_counts: dict[str, int] = {}
    for horizon in HEADLINE_HORIZONS:
        realized = archive_df[
            (archive_df["step_index"] == int(horizon))
            & (archive_df["is_realized"].fillna(False))
            & archive_df["actual"].notna()
        ].copy()
        realized = realized.sort_values(["target_date", "generated_at"], kind="stable")
        realized = realized.drop_duplicates(subset=["forecast_made_on", "step_index"], keep="first")
        series = [
            {
                "target_date": str(row.target_date),
                "forecast_made_on": str(row.forecast_made_on),
                "actual": float(row.actual),
                "base_tsm_forecast": float(row.base_tsm_forecast),
                "llm_tsm_forecast": float(row.llm_tsm_forecast),
            }
            for row in realized.itertuples(index=False)
        ]
        payload = {
            "market": market_name,
            "horizon_days": int(horizon),
            "generated_at": latest_generated_at,
            "model_version": model_version,
            "series": series,
        }
        _write_json(export_dir / f"horizon_{horizon}d.json", payload)
        horizon_counts[f"h{horizon}"] = int(len(series))

    manifest_payload = {
        "generated_at": latest_generated_at,
        "files": EXPECTED_EXPORT_FILES,
        "latest_observed_date": latest_observed_date,
        "model_version": model_version,
    }
    status_payload = {
        "generated_at": latest_generated_at,
        "latest_observed_date": latest_observed_date,
        "model_version": model_version,
        "archive_rows": int(len(archive_df)),
        "realized_rows": int(archive_df["actual"].notna().sum()),
        "current_live_rows": int(len(live_rows)),
        "horizon_counts": horizon_counts,
    }
    _write_json(export_dir / "manifest.json", manifest_payload)
    _write_json(export_dir / "status.json", status_payload)

    return {
        "latest": latest_payload,
        "status": status_payload,
        "files": list(EXPECTED_EXPORT_FILES),
    }


def deploy_exports_via_rsync(
    *,
    export_dir: str | Path,
    remote_host: str,
    remote_staging_dir: str,
    remote_live_dir: str,
    expected_files: list[str] | None = None,
    run_label: str | None = None,
    dry_run: bool = False,
) -> None:
    export_dir = Path(export_dir).resolve()
    expected_files = list(expected_files or EXPECTED_EXPORT_FILES)
    missing = [name for name in expected_files if not (export_dir / name).exists()]
    if missing:
        raise FileNotFoundError(f"Refusing to deploy; missing export files: {missing}")

    run_label = str(run_label or datetime.now().strftime("%Y%m%d_%H%M%S"))
    remote_staging_path = f"{remote_staging_dir.rstrip('/')}/{run_label}"
    rsync_cmd = ["rsync", "-az", "--delete"]
    if dry_run:
        rsync_cmd.append("--dry-run")
    rsync_cmd.extend([f"{export_dir}/", f"{remote_host}:{remote_staging_path}/"])
    subprocess.run(rsync_cmd, check=True)

    if dry_run:
        return

    checks = " && ".join(
        f"test -f {shlex.quote(remote_staging_path.rstrip('/') + '/' + name)}"
        for name in expected_files
    )
    promote = (
        f"{checks} && mkdir -p {shlex.quote(remote_live_dir)} "
        f"&& rsync -az --delete {shlex.quote(remote_staging_path)}/ {shlex.quote(remote_live_dir)}/"
    )
    subprocess.run(["ssh", remote_host, promote], check=True)


def build_deploy_preview(
    *,
    export_dir: str | Path,
    remote_host: str,
    remote_staging_dir: str,
    remote_live_dir: str,
    run_label: str,
) -> dict[str, str]:
    export_dir = Path(export_dir).resolve()
    staging_path = f"{remote_staging_dir.rstrip('/')}/{run_label}"
    rsync_cmd = f"rsync -az --delete {shlex.quote(str(export_dir) + '/')} {shlex.quote(remote_host + ':' + staging_path + '/')}"
    validate_and_promote = (
        f"ssh {shlex.quote(remote_host)} "
        f"{shlex.quote(f'mkdir -p {remote_live_dir} && rsync -az --delete {staging_path}/ {remote_live_dir}/')}"
    )
    return {
        "rsync": rsync_cmd,
        "promote": validate_and_promote,
        "staging_path": staging_path,
    }
