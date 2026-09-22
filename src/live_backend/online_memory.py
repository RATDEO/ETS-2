from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from .common import logger


def _history_distance_features(history: np.ndarray, feature_window: int) -> np.ndarray:
    from ..frontend_live_bundle import _history_features

    return _history_features(np.asarray(history, dtype=float), int(feature_window))


def _normalize_history_dates(date_values: Sequence[object], expected_len: int) -> list[str]:
    normalized = [str(pd.Timestamp(value).date()) for value in pd.to_datetime(list(date_values))]
    if expected_len <= 0:
        return []
    if len(normalized) >= expected_len:
        return normalized[-expected_len:]
    if not normalized:
        return []
    return ([normalized[0]] * max(0, expected_len - len(normalized))) + normalized


def _sentiment_history_from_dates(
    date_values: Sequence[object],
    sentiment_map: Mapping[pd.Timestamp, float] | None,
    sentiment_points: int | None,
) -> list[float]:
    if sentiment_map is None or sentiment_points is None or int(sentiment_points) <= 0:
        return []
    normalized_dates = [pd.Timestamp(value).normalize() for value in pd.to_datetime(list(date_values))]
    values = [float(sentiment_map.get(value, 0.0)) for value in normalized_dates]
    if len(values) < int(sentiment_points):
        values = ([0.0] * (int(sentiment_points) - len(values))) + values
    return list(np.asarray(values[-int(sentiment_points):], dtype=float))


def _online_memory_result_slug(method_name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(method_name).strip())


def load_online_memory_gate_selection(
    run_dir: str | Path | None,
    method_name: str,
) -> dict[str, Any] | None:
    if run_dir is None:
        return None
    run_path = Path(run_dir).resolve()
    selection_path = run_path / "llm" / f"online_memory_gate_selection_{_online_memory_result_slug(method_name)}.json"
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


def merge_online_memory_gate_cfg(
    gate_cfg: Mapping[str, Any] | None,
    selection_payload: Mapping[str, Any] | None,
) -> dict[str, Any]:
    merged = dict(gate_cfg or {})
    selected_gate_cfg = (selection_payload or {}).get("selected_gate_cfg")
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
    from ..run_experiment import _case_profile_features, build_retrieval_feature_vector

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
    from ..run_experiment import select_top_score_indices

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
        return select_top_score_indices(
            candidate_indices=candidate_indices,
            pool_dates=pool_dates,
            scores=path_mse,
            k_examples=k_examples,
        )
    if selection_mode == "similarity":
        sample_features = _history_distance_features(current_history, feature_window)
        candidate_features = np.vstack(
            [_history_distance_features(pool_histories[idx], feature_window) for idx in candidate_indices]
        )
        distances = np.linalg.norm(candidate_features - sample_features, axis=1)
        chosen = candidate_indices[np.argsort(distances)[: int(min(k_examples, len(candidate_indices)))]]
        return chosen[np.argsort(pool_dates[chosen])]

    chosen = candidate_indices[np.argsort(pool_dates[candidate_indices])][-int(min(k_examples, len(candidate_indices))):]
    return chosen[np.argsort(pool_dates[chosen])]


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
    from ..run_experiment import build_exogenous_summary

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
    if pool_windows is not None and feature_cols is not None and target_col is not None and max_exogenous_features is not None:
        example["exogenous_summary"] = build_exogenous_summary(
            window=np.asarray(pool_windows[idx], dtype=float),
            feature_cols=list(feature_cols),
            target_col=str(target_col),
            max_features=int(max_exogenous_features),
            preferred_features=list(preferred_features) if preferred_features is not None else None,
        )
    if selection_role:
        example["case_summary"] = {"selection_role": str(selection_role)}
    if retrieval_tag:
        example["retrieval_tag"] = str(retrieval_tag)
    return example


def select_teaching_examples(
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
    return [
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
        for idx in chosen
    ]


def _build_online_memory_example(record: Mapping[str, Any], selection_role: str) -> dict[str, Any]:
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
    from ..run_experiment import (
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
    from ..run_experiment import (
        _profile_regime_tag,
        build_exogenous_summary,
        build_history_dates,
        build_online_memory_gate_feature_row,
        build_realized_availability_dates,
        classify_online_memory_admission,
        compute_online_memory_helpfulness,
        compute_online_memory_horizon_gains,
        select_online_memory_records,
    )

    cot_cfg = dict(llm_cfg.get("cot_rf", {}) or {})
    policy_cfg = dict(cot_cfg.get("online_memory_policy", {}) or {})
    history_points = int(llm_cfg.get("history_points", 18))
    feature_window = int(cot_cfg.get("feature_window", 18))
    sentiment_points = int((llm_cfg.get("sentiment", {}) or {}).get("history_points", history_points))
    support_examples = int(policy_cfg.get("support_examples", cot_cfg.get("k_examples", 5)))
    positive_examples = int(policy_cfg.get("positive_examples", 2))
    negative_examples = int(policy_cfg.get("negative_examples", 1))
    max_total_examples = int(policy_cfg.get("max_total_examples", support_examples + positive_examples + negative_examples))
    positive_margin = float(policy_cfg.get("positive_margin", 0.01))
    negative_margin = float(policy_cfg.get("negative_margin", -0.01))
    warmup_min_realized = int(policy_cfg.get("warmup_min_realized", 4))

    history_date_arrays = build_history_dates(
        panel_dates=np.asarray(panel_dates),
        date_to_idx=dict(date_to_idx),
        pred_dates=np.asarray(pool_dates),
        history_points=history_points,
    )
    availability_dates = build_realized_availability_dates(
        panel_dates=np.asarray(panel_dates),
        date_to_idx=dict(date_to_idx),
        pred_dates=np.asarray(pool_dates),
        realization_horizon=int(pool_forecasts.shape[1]),
    )

    predictions = np.asarray(pool_forecasts, dtype=float).copy()
    gate_feature_rows: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []

    for row_index, origin_date in enumerate(pd.to_datetime(pool_dates)):
        full_history = np.asarray(pool_histories[row_index], dtype=float)
        history = np.asarray(full_history[-history_points:], dtype=float)
        history_dates = _normalize_history_dates(history_date_arrays[row_index], len(history))
        sample_window = None if pool_windows is None else np.asarray(pool_windows[row_index], dtype=float)
        sample_profile = _reference_case_profile(
            history=full_history,
            forecast=np.asarray(pool_forecasts[row_index], dtype=float),
            window=sample_window,
            feature_window=feature_window,
            augment_with_retrieval=bool(cot_cfg.get("augment_case_profiles_with_retrieval", False)),
            feature_cols=feature_cols,
            target_col=target_col,
            preferred_features=preferred_feature_order,
            retrieval_feature_columns=cot_cfg.get("retrieval_feature_columns"),
            retrieval_max_features=int(cot_cfg.get("retrieval_max_features", max_exogenous_features)),
        )
        reference_tag = _profile_regime_tag(sample_profile)
        available_records = [
            dict(record)
            for record in records
            if pd.Timestamp(record.get("availability_date")).to_datetime64() <= origin_date.to_datetime64()
            and str(record.get("admission_label", "")).strip().lower() in {"positive", "negative"}
        ]
        support = select_teaching_examples(
            pool_dates=pool_dates,
            pool_histories=pool_histories,
            pool_forecasts=pool_forecasts,
            pool_truth=pool_truth,
            origin_date=pd.Timestamp(origin_date),
            current_history=full_history,
            history_points=history_points,
            selection_mode=str(cot_cfg.get("example_selection", "recent")),
            k_examples=support_examples,
            feature_window=feature_window,
            lookback_days=int(cot_cfg["lookback_days"]) if cot_cfg.get("lookback_days") is not None else None,
            history_date_arrays=history_date_arrays,
            sentiment_map=sentiment_map,
            sentiment_points=sentiment_points,
            pool_windows=pool_windows,
            feature_cols=feature_cols,
            target_col=target_col,
            max_exogenous_features=max_exogenous_features,
            preferred_features=preferred_feature_order,
        )
        positive_records = select_online_memory_records(
            available_records,
            sample_profile,
            label="positive",
            k_examples=positive_examples,
            reference_tag=reference_tag,
            min_tag_overlap=0,
        )
        negative_records = select_online_memory_records(
            available_records,
            sample_profile,
            label="negative",
            k_examples=negative_examples,
            reference_tag=reference_tag,
            min_tag_overlap=0,
        )
        feature_row = build_online_memory_gate_feature_row(
            positive_records=positive_records,
            negative_records=negative_records,
            base_forecast=np.asarray(pool_forecasts[row_index], dtype=float),
            current_price=float(history[-1]) if history.size else 0.0,
            warmup_ready=len(available_records) >= warmup_min_realized,
            reference_profile=sample_profile,
            support_example_count=len(support),
        )
        gate_feature_rows.append(dict(feature_row))

        teaching_examples = list(support)
        teaching_examples.extend(_build_online_memory_example(record, "positive_memory") for record in positive_records)
        teaching_examples.extend(_build_online_memory_example(record, "negative_memory") for record in negative_records)
        if max_total_examples > 0:
            teaching_examples = teaching_examples[:max_total_examples]

        refined = None
        if teaching_examples:
            refined, _meta = refiner.refine(
                method=method_name,
                history=history,
                dates=history_dates,
                tsm_forecast=np.asarray(pool_forecasts[row_index], dtype=float),
                pred_len=int(pool_forecasts.shape[1]),
                exogenous_summary=(
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
                price_base=float(history[-1]) if history.size else 0.0,
                teaching_examples=teaching_examples,
                sentiment_history=np.asarray(_sentiment_history_from_dates(history_dates, sentiment_map, sentiment_points), dtype=float),
            )
        if refined is None:
            refined = np.asarray(pool_forecasts[row_index], dtype=float)
        refined = np.asarray(refined, dtype=float)
        if refined.shape != np.asarray(pool_forecasts[row_index], dtype=float).shape:
            refined = np.asarray(pool_forecasts[row_index], dtype=float)
        predictions[row_index] = refined

        helpfulness_score = compute_online_memory_helpfulness(
            np.asarray(pool_forecasts[row_index], dtype=float),
            refined,
            np.asarray(pool_truth[row_index], dtype=float),
        )
        horizon_gains = compute_online_memory_horizon_gains(
            np.asarray(pool_forecasts[row_index], dtype=float),
            refined,
            np.asarray(pool_truth[row_index], dtype=float),
        )
        admission_label = classify_online_memory_admission(
            helpfulness_score,
            horizon_gains,
            positive_margin=positive_margin,
            negative_margin=negative_margin,
        )
        if admission_label in {"positive", "negative"}:
            records.append(
                {
                    "availability_date": str(pd.Timestamp(availability_dates[row_index]).date()),
                    "date": str(pd.Timestamp(origin_date).date()),
                    "history": np.asarray(history, dtype=float),
                    "forecast": np.asarray(refined, dtype=float),
                    "truth": np.asarray(pool_truth[row_index], dtype=float),
                    "sentiment_history": _sentiment_history_from_dates(history_dates, sentiment_map, sentiment_points),
                    "exogenous_summary": None,
                    "case_profile": np.asarray(sample_profile, dtype=float),
                    "case_summary": {"support_example_count": int(len(support))},
                    "retrieval_tag": str(reference_tag),
                    "helpfulness_score": float(helpfulness_score),
                    "horizon_gains": {str(key): float(value) for key, value in horizon_gains.items()},
                    "admission_label": str(admission_label),
                }
            )

    return {
        "records": records,
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
    from ..run_experiment import _profile_regime_tag, build_online_memory_gate_decision, select_online_memory_records

    cot_cfg = dict(llm_cfg.get("cot_rf", {}) or {})
    policy_cfg = dict(cot_cfg.get("online_memory_policy", {}) or {})
    positive_examples = int(policy_cfg.get("positive_examples", 2))
    negative_examples = int(policy_cfg.get("negative_examples", 1))
    max_total_examples = int(
        policy_cfg.get("max_total_examples", int(policy_cfg.get("support_examples", len(support_examples))) + positive_examples + negative_examples)
    )
    warmup_min_realized = int(policy_cfg.get("warmup_min_realized", 4))
    feature_window = int(cot_cfg.get("feature_window", 18))

    reference_profile = _reference_case_profile(
        history=np.asarray(current_history, dtype=float),
        forecast=np.asarray(current_forecast, dtype=float),
        window=current_window,
        feature_window=feature_window,
        augment_with_retrieval=bool(cot_cfg.get("augment_case_profiles_with_retrieval", False)),
        feature_cols=feature_cols,
        target_col=target_col,
        preferred_features=preferred_feature_order,
        retrieval_feature_columns=cot_cfg.get("retrieval_feature_columns"),
        retrieval_max_features=int(cot_cfg.get("retrieval_max_features", max_exogenous_features)),
    )
    reference_tag = _profile_regime_tag(reference_profile)
    available_records = [
        dict(record)
        for record in online_memory_records
        if pd.Timestamp(record.get("availability_date")).to_datetime64() <= pd.Timestamp(current_origin_date).to_datetime64()
        and str(record.get("admission_label", "")).strip().lower() in {"positive", "negative"}
    ]
    positive_records = select_online_memory_records(
        available_records,
        reference_profile,
        label="positive",
        k_examples=positive_examples,
        reference_tag=reference_tag,
        min_tag_overlap=0,
    )
    negative_records = select_online_memory_records(
        available_records,
        reference_profile,
        label="negative",
        k_examples=negative_examples,
        reference_tag=reference_tag,
        min_tag_overlap=0,
    )
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

    teaching_examples = list(support_examples)
    teaching_examples.extend(_build_online_memory_example(record, "positive_memory") for record in positive_records)
    teaching_examples.extend(_build_online_memory_example(record, "negative_memory") for record in negative_records)
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
