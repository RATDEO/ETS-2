from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from .common import (
    DEFAULT_METHOD_NAME,
    DEFAULT_MODEL_VERSION,
    archive_row,
    empty_archive_frame,
    iso_utc_now,
    next_business_target_dates,
    normalize_archive_frame,
    read_run_metadata,
)
from .online_memory import (
    _build_current_online_memory_context,
    _fit_live_learned_gate_bundle,
    _sentiment_history_from_dates,
    _simulate_live_online_memory_history,
    load_online_memory_gate_selection,
    merge_online_memory_gate_cfg,
    select_teaching_examples,
)


def _pool_predictions(trained: Any, target_mode: str, batch_size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    from ..frontend_live_bundle import _predict_prices
    from ..run_experiment import returns_to_prices

    index_slice = slice(trained.train_slice.start, trained.val_slice.stop)
    predictions = _predict_prices(
        model=trained.model,
        scaler=trained.scaler,
        X_enc=trained.windows.X_enc[index_slice],
        X_dec=trained.windows.X_dec[index_slice],
        base_prices=trained.windows.base_prices[index_slice],
        target_mode=target_mode,
        batch_size=batch_size,
    )
    if target_mode == "returns":
        truth = returns_to_prices(trained.windows.y[index_slice], trained.windows.base_prices[index_slice])
    else:
        truth = trained.windows.y[index_slice]
    return (
        pd.to_datetime(trained.windows.pred_dates[index_slice]).to_numpy(),
        np.asarray(trained.windows.price_histories[index_slice], dtype=float),
        np.asarray(predictions, dtype=float),
        np.asarray(truth, dtype=float),
    )


def _load_live_feature_panel(
    *,
    config_raw: Mapping[str, Any],
    data_dir: str | Path,
    run_dir: str | Path,
) -> pd.DataFrame:
    from ..data import build_panel, select_feature_columns

    target_mode = str(config_raw.get("target", {}).get("mode", "returns")).lower()
    target_col = "y_return" if target_mode == "returns" else "y"
    feature_cfg = config_raw.get("features", {}) or {}
    panel, _schema = build_panel(data_dir=Path(data_dir), config=config_raw, save_path=Path(run_dir) / "data")
    feature_cols = select_feature_columns(
        panel,
        target_col=target_col,
        max_exogenous_features=int(feature_cfg.get("max_exogenous_features_model", 10)),
        preferred_feature_order=feature_cfg.get("preferred_feature_order"),
    )
    panel = panel[["date"] + [column for column in feature_cols if column in panel.columns]].copy()
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
    from ..frontend_live_bundle import _build_live_inputs, _predict_prices, train_tsm_on_prefix
    from ..llm import LLMRefiner
    from ..run_experiment import build_exogenous_summary, infer_llm_market_name, load_daily_sentiment, seed_llm_response_cache

    run_path = Path(run_dir).resolve()
    artifact_path = Path(artifact_run_dir).resolve() if artifact_run_dir is not None else None
    metadata = read_run_metadata(run_path) if (run_path / "config_resolved.yaml").exists() else {"git_hash": "", "timestamp": ""}

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

        cache_dir = run_path / "llm" / "live_forecast" / "cache"
        log_dir = run_path / "llm" / "live_forecast" / "logs"
        if artifact_path is not None and artifact_path != run_path:
            try:
                seed_llm_response_cache(cache_dir, [artifact_path])
            except Exception:
                pass

        refiner = LLMRefiner(config=llm_cfg, cache_dir=cache_dir, log_dir=log_dir)
        current_origin_date = pd.Timestamp(panel["date"].iloc[-1]).normalize()
        current_history = np.asarray(history_prices[-history_points:], dtype=float)
        current_history_dates = [str(ts.date()) for ts in pd.to_datetime(panel["date"]).tail(history_points).tolist()]
        current_exogenous_summary = build_exogenous_summary(
            window=np.asarray(X_enc, dtype=float),
            feature_cols=trained.windows.feature_cols,
            target_col=target_col,
            max_features=int(llm_cfg.get("max_exogenous_features", 6)),
            preferred_features=preferred_feature_order,
        )
        current_sentiment = _sentiment_history_from_dates(current_history_dates, sentiment_map, sent_points)
        current_sentiment_history = np.asarray(current_sentiment, dtype=float) if current_sentiment else None

        policy_cfg = dict(cot_cfg.get("online_memory_policy", {}) or {})
        policy_enabled = bool(
            str(cot_cfg.get("test_pool_mode", "")).strip().lower() == "online_realized_memory"
            and policy_cfg.get("enabled", False)
        )
        if policy_enabled:
            val_index = np.arange(trained.val_slice.start, trained.val_slice.stop)
            panel_dates = pd.to_datetime(panel["date"]).to_numpy()
            date_to_idx = {pd.Timestamp(value).normalize(): idx for idx, value in enumerate(pd.to_datetime(panel_dates))}
            selection_payload = load_online_memory_gate_selection(artifact_path or run_path, method_name)
            gate_cfg = merge_online_memory_gate_cfg(policy_cfg.get("gate", {}) or {}, selection_payload)
            simulated = _simulate_live_online_memory_history(
                method_name=method_name,
                refiner=refiner,
                pool_dates=pd.to_datetime(trained.windows.pred_dates[val_index]).to_numpy(),
                pool_histories=np.asarray(trained.windows.price_histories[val_index], dtype=float),
                pool_forecasts=np.asarray(trained.val_pred_prices, dtype=float),
                pool_truth=np.asarray(trained.val_true_prices, dtype=float),
                pool_windows=np.asarray(trained.windows.X_enc[val_index], dtype=float),
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
                y_true=np.asarray(trained.val_true_prices, dtype=float),
                base_pred=np.asarray(trained.val_pred_prices, dtype=float),
                llm_pred=np.asarray(simulated["predictions"], dtype=float),
                gate_cfg=gate_cfg,
                selection_payload=selection_payload,
            )
            support_examples = select_teaching_examples(
                pool_dates=pd.to_datetime(trained.windows.pred_dates[val_index]).to_numpy(),
                pool_histories=np.asarray(trained.windows.price_histories[val_index], dtype=float),
                pool_forecasts=np.asarray(trained.val_pred_prices, dtype=float),
                pool_truth=np.asarray(trained.val_true_prices, dtype=float),
                origin_date=current_origin_date,
                current_history=np.asarray(history_prices, dtype=float),
                history_points=history_points,
                selection_mode=str(cot_cfg.get("example_selection", "recent")),
                k_examples=int(policy_cfg.get("support_examples", cot_cfg.get("k_examples", 5))),
                feature_window=int(cot_cfg.get("feature_window", 18)),
                lookback_days=int(cot_cfg["lookback_days"]) if cot_cfg.get("lookback_days") is not None else None,
                history_date_arrays=simulated["history_date_arrays"],
                sentiment_map=sentiment_map,
                sentiment_points=sent_points,
                pool_windows=np.asarray(trained.windows.X_enc[val_index], dtype=float),
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
                gate_cfg=gate_cfg,
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
                "online_memory_history_cases": int(len(val_index)),
                "online_memory_selection_path": str((selection_payload or {}).get("selection_path", "")),
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
                    llm_metadata = {**llm_metadata, **dict(refine_meta or {}), "fallback_to_tsm": False}
        else:
            pool_dates, pool_histories, pool_forecasts, pool_truth = _pool_predictions(trained, target_mode, batch_size)
            examples = select_teaching_examples(
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
                lookback_days=int(cot_cfg["lookback_days"]) if cot_cfg.get("lookback_days") is not None else None,
            )
            if examples:
                refined, refine_meta = refiner.refine(
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
                    llm_metadata = {**dict(refine_meta or {}), "fallback_to_tsm": False}

    latest_date = pd.Timestamp(panel["date"].iloc[-1]).normalize()
    latest_date_str = str(latest_date.date())
    target_dates = next_business_target_dates(latest_date, pred_len)
    generated_at = iso_utc_now()

    rows = [
        archive_row(
            run_id=str(run_path.name),
            forecast_made_on=latest_date_str,
            latest_observed_date=latest_date_str,
            latest_observed_price=float(last_price),
            target_date=target_dates[index],
            step_index=index + 1,
            actual=None,
            base_tsm_forecast=float(base_pred[index]),
            llm_tsm_forecast=float(llm_pred[index]),
            model_version=model_version,
            model_commit=str(metadata.get("git_hash") or ""),
            data_version=latest_date_str,
            is_realized=False,
            generated_at=generated_at,
            record_source=record_source,
            source_run_id=str(run_path.name),
        )
        for index in range(pred_len)
    ]
    return (
        normalize_archive_frame(pd.DataFrame(rows)),
        panel[["date", "y"]].copy(),
        {
            "origin_date": latest_date_str,
            "latest_observed_price": float(last_price),
            "base_forecast": np.asarray(base_pred, dtype=float),
            "llm_forecast": np.asarray(llm_pred, dtype=float),
            "target_dates": target_dates,
            "llm_method": method_name,
            "llm_metadata": llm_metadata,
        },
    )


def _config_with_previous_llm_system(config_raw: Mapping[str, Any]) -> dict[str, Any]:
    row_config = copy.deepcopy(dict(config_raw))
    llm_cfg = copy.deepcopy(row_config.get("llm", {}) or {})
    cot_cfg = copy.deepcopy(llm_cfg.get("cot_rf", {}) or {})
    policy_cfg = copy.deepcopy(cot_cfg.get("online_memory_policy", {}) or {})
    policy_cfg["enabled"] = False
    cot_cfg["online_memory_policy"] = policy_cfg
    cot_cfg["test_pool_mode"] = ""
    llm_cfg["cot_rf"] = cot_cfg
    row_config["llm"] = llm_cfg
    return row_config


def build_current_live_rows(
    *,
    config_raw: dict[str, Any],
    data_dir: str | Path,
    run_dir: str | Path,
    artifact_run_dir: str | Path | None = None,
    model_version: str = DEFAULT_MODEL_VERSION,
    record_source: str = "live_run",
    use_previous_llm_system: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    row_config = _config_with_previous_llm_system(config_raw) if use_previous_llm_system else copy.deepcopy(config_raw)
    panel = _load_live_feature_panel(config_raw=row_config, data_dir=data_dir, run_dir=Path(run_dir).resolve())
    return _build_live_rows_from_feature_panel(
        config_raw=row_config,
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
    use_previous_llm_system: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    run_path = Path(run_dir).resolve()
    panel = _load_live_feature_panel(config_raw=config_raw, data_dir=data_dir, run_dir=run_path)
    panel_prices = panel[["date", "y"]].copy()
    if panel.empty or len(panel) < 2:
        return empty_archive_frame(), panel_prices

    pred_len = int(config_raw.get("time_series", {}).get("pred_len", 30))
    seq_len = int(config_raw.get("time_series", {}).get("seq_len", 120))
    archive_cfg = config_raw.get("frontend_live_bundle", {}) or {}
    earliest_origin_index = (
        seq_len
        + pred_len
        + int(archive_cfg.get("min_train_windows", 756))
        + int(archive_cfg.get("min_val_windows", 126))
        - 2
    )
    cutoff = pd.Timestamp(start_after_date).normalize() if start_after_date is not None else pd.Timestamp.min.normalize()
    last_origin_index = len(panel) - 1 if include_latest_origin else len(panel) - 2
    if last_origin_index < 0:
        return empty_archive_frame(), panel_prices

    origin_indices = [
        index
        for index in range(max(0, earliest_origin_index), last_origin_index + 1)
        if pd.Timestamp(panel["date"].iloc[index]).normalize() > cutoff
    ]
    if not origin_indices:
        return empty_archive_frame(), panel_prices

    row_config = _config_with_previous_llm_system(config_raw) if use_previous_llm_system else copy.deepcopy(config_raw)
    row_frames = []
    for origin_index in origin_indices:
        prefix_panel = panel.iloc[: origin_index + 1].copy().reset_index(drop=True)
        try:
            rows, _prices, _meta = _build_live_rows_from_feature_panel(
                config_raw=row_config,
                panel=prefix_panel,
                run_dir=run_path,
                artifact_run_dir=artifact_run_dir,
                model_version=model_version,
                record_source=record_source,
            )
        except ValueError as exc:
            if "Insufficient clean windows" not in str(exc):
                raise
            continue
        row_frames.append(rows)

    if not row_frames:
        return empty_archive_frame(), panel_prices
    return normalize_archive_frame(pd.concat(row_frames, ignore_index=True)), panel_prices
