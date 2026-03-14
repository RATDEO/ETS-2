from __future__ import annotations

import copy
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from .data import build_panel, select_feature_columns
from .eval.metrics import compute_metrics_by_horizon, compute_path_metrics
from .eval.regime_audit import build_origin_regime_frame
from .frontend_live_bundle import (
    DEFAULT_LOCAL_LLM_API_KEY,
    DEFAULT_LOCAL_LLM_BASE_URL,
    TrainedArtifacts,
    _build_live_inputs,
    _collect_actual_history,
    _copy_outputs,
    _history_features,
    _iso_utc_now,
    _predict_prices,
    _select_origin_indices,
    _write_json,
    train_tsm_on_prefix,
)
from .llm import LLMRefiner
from .run_experiment import load_daily_sentiment

logger = logging.getLogger(__name__)

PAPER_MODEL = "Baseline paper CoT-SENT"
HDELTA_4B_MODEL = "4B raw HDELTA"
TSM_MODEL = "Raw tsm"
HDELTA_35B_MODEL = "35B raw guarded HDELTA"
GATE_MODEL = "gate_sentiment_alignment_5d"

MODEL_DISPLAY_NAMES = {
    PAPER_MODEL: "Baseline paper CoT-SENT",
    HDELTA_4B_MODEL: "4B Raw HDELTA",
    TSM_MODEL: "Raw TSM",
    HDELTA_35B_MODEL: "35B Raw Guarded HDELTA",
    GATE_MODEL: "Sentiment-Alignment Gate",
}

SENTIMENT_ALIGNMENT_GATE = {
    "aligned": HDELTA_35B_MODEL,
    "divergent": HDELTA_4B_MODEL,
    "neutral": PAPER_MODEL,
}


@dataclass(frozen=True)
class ArchiveLLMCandidate:
    display_name: str
    method: str
    base_url: str
    model: str
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


def _display_name(model_name: str) -> str:
    return MODEL_DISPLAY_NAMES.get(model_name, model_name)


def _metric_row(name: str, y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    path = compute_path_metrics(y_true, y_pred)
    by_h = compute_metrics_by_horizon(y_true, y_pred, horizons=[1, 5, 20, 30])
    row = {
        "model": name,
        "display_name": _display_name(name),
        "n_origins": int(y_true.shape[0]),
        "path_mse": float(path["mse_path"]),
        "path_rmse": float(path["rmse_path"]),
        "path_mae": float(path["mae_path"]),
    }
    for h in [1, 5, 20, 30]:
        if h in by_h.index:
            row[f"h{h}_mse"] = float(by_h.loc[h, "mse"])
            row[f"h{h}_rmse"] = float(by_h.loc[h, "rmse"])
            row[f"h{h}_mae"] = float(by_h.loc[h, "mae"])
    return row


def _candidate_definitions(
    llm4b_model: str,
    llm4b_base_url: str,
    llm35b_model: str,
    llm35b_base_url: str,
) -> list[ArchiveLLMCandidate]:
    return [
        ArchiveLLMCandidate(
            display_name=PAPER_MODEL,
            method="TSM+LLM-COT-SENT-RF",
            base_url=llm4b_base_url,
            model=llm4b_model,
        ),
        ArchiveLLMCandidate(
            display_name=HDELTA_4B_MODEL,
            method="TSM+LLM-COT-SENT-RF-HDELTA",
            base_url=llm4b_base_url,
            model=llm4b_model,
            hdelta_max_adjustment_pct=3.0,
            hdelta_freeze_horizons=(),
            hdelta_sentiment_secondary=False,
        ),
        ArchiveLLMCandidate(
            display_name=HDELTA_35B_MODEL,
            method="TSM+LLM-COT-SENT-RF-HDELTA",
            base_url=llm35b_base_url,
            model=llm35b_model,
            retain_context=False,
            strict_json_prompt=True,
            hdelta_max_adjustment_pct=1.0,
            hdelta_freeze_horizons=(1,),
            hdelta_sentiment_secondary=True,
        ),
    ]


def _resolve_llm_cfg(base_cfg: dict[str, Any], candidate: ArchiveLLMCandidate, api_key: str | None, timeout_seconds: float) -> dict[str, Any]:
    llm_cfg = copy.deepcopy(base_cfg)
    llm_cfg["api_key"] = api_key or DEFAULT_LOCAL_LLM_API_KEY
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
    llm_cfg["sentiment"]["path"] = "data/news/daily_sentiment.csv"
    llm_cfg["sentiment"]["date_col"] = "seendate"
    llm_cfg["sentiment"]["score_col"] = "sent_score"
    llm_cfg["sentiment"]["history_points"] = candidate.sentiment_points
    if candidate.method == "TSM+LLM-COT-SENT-RF-HDELTA":
        llm_cfg.setdefault("hdelta", {})
        llm_cfg["hdelta"]["key_horizons"] = [1, 5, 20, 30]
        llm_cfg["hdelta"]["max_adjustment_pct"] = float(candidate.hdelta_max_adjustment_pct or 3.0)
        llm_cfg["hdelta"]["freeze_horizons"] = list(candidate.hdelta_freeze_horizons)
        llm_cfg["hdelta"]["sentiment_secondary"] = bool(candidate.hdelta_sentiment_secondary)
    return llm_cfg


def _pool_slice(trained: TrainedArtifacts) -> slice:
    return slice(trained.train_slice.start, trained.val_slice.stop)


def _pool_predictions(trained: TrainedArtifacts, target_mode: str, batch_size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    from .run_experiment import returns_to_prices

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
    pool_true = returns_to_prices(trained.windows.y[idx], trained.windows.base_prices[idx]) if target_mode == "returns" else trained.windows.y[idx]
    return (
        pd.to_datetime(trained.windows.pred_dates[idx]).to_numpy(),
        trained.windows.price_histories[idx],
        pool_pred,
        np.asarray(pool_true, dtype=float),
    )


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
        candidate_indices = candidate_indices[pool_dates[candidate_indices] >= cutoff]
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
                "date": str(example_pred_date.date()),
                "sentiment_history": _sentiment_series_from_dates(example_history_dates, sentiment_map, candidate.sentiment_points),
            }
        )
    return examples


def _refine_models_for_prefix(
    *,
    prefix_panel: pd.DataFrame,
    full_panel: pd.DataFrame,
    trained: TrainedArtifacts,
    target_mode: str,
    target_col: str,
    seq_len: int,
    label_len: int,
    pred_len: int,
    batch_size: int,
    candidates: list[ArchiveLLMCandidate],
    refiners: Mapping[str, LLMRefiner],
    sentiment_map: Mapping[pd.Timestamp, float],
) -> tuple[dict[str, np.ndarray], float, list[dict[str, Any]]]:
    prefix_panel = prefix_panel.copy()
    prefix_panel.attrs["pred_len"] = pred_len
    X_enc, X_dec, last_price, history_prices = _build_live_inputs(
        prefix_panel=prefix_panel,
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

    origin_predictions: dict[str, np.ndarray] = {TSM_MODEL: np.asarray(base_pred, dtype=float)}
    metadata_rows: list[dict[str, Any]] = []

    pool_dates, pool_histories, pool_forecasts, pool_truth = _pool_predictions(
        trained=trained,
        target_mode=target_mode,
        batch_size=batch_size,
    )
    panel_dates = pd.to_datetime(prefix_panel["date"]).to_numpy()
    date_to_idx = {pd.Timestamp(d).normalize(): i for i, d in enumerate(panel_dates)}

    for candidate in candidates:
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
            origin_date=pd.Timestamp(prefix_panel["date"].iloc[-1]).normalize(),
            current_history=current_history,
            candidate=candidate,
            panel_dates=panel_dates,
            date_to_idx=date_to_idx,
            sentiment_map=sentiment_map,
        )
        sentiment_history = _sentiment_series_from_dates(current_history_dates, sentiment_map, candidate.sentiment_points)
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
            meta = {**meta, "fallback_to_tsm": True}
        origin_predictions[candidate.display_name] = np.asarray(refined, dtype=float)
        metadata_rows.append({"model": candidate.display_name, "metadata": meta})

    return origin_predictions, float(last_price), metadata_rows


def _archive_rows_for_origin(
    origin_date: str,
    target_dates: list[str],
    actual_prices: np.ndarray | None,
    last_observed_price: float,
    predictions: Mapping[str, np.ndarray],
    regime_value: str,
    selected_model: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    gate_pred = predictions[selected_model]
    model_preds = {**predictions, GATE_MODEL: gate_pred}
    for model_name, forecast in model_preds.items():
        for i, target_date in enumerate(target_dates):
            actual_price = None if actual_prices is None else float(actual_prices[i])
            rows.append(
                {
                    "origin_date": origin_date,
                    "target_date": str(target_date),
                    "model": model_name,
                    "display_name": _display_name(model_name),
                    "horizon": int(i + 1),
                    "predicted_price": float(forecast[i]),
                    "actual_price": actual_price,
                    "last_observed_price": float(last_observed_price),
                    "is_realized": actual_prices is not None,
                    "gate_selected_model": selected_model,
                    "gate_regime_value": regime_value,
                }
            )
    return rows


def _summarize_archive(archive_df: pd.DataFrame) -> dict[str, Any]:
    models = []
    for model_name, grp in archive_df.groupby("model", sort=False):
        grp = grp.dropna(subset=["actual_price"]).copy()
        if grp.empty:
            continue
        y_true = []
        y_pred = []
        for _, origin_grp in grp.groupby("origin_date", sort=True):
            origin_grp = origin_grp.sort_values("horizon")
            y_true.append(origin_grp["actual_price"].to_numpy(dtype=float))
            y_pred.append(origin_grp["predicted_price"].to_numpy(dtype=float))
        metrics = _metric_row(model_name, np.vstack(y_true), np.vstack(y_pred))
        models.append(metrics)
    return {"models": sorted(models, key=lambda row: row["path_mse"])}


def _build_archive_json(archive_df: pd.DataFrame) -> dict[str, Any]:
    origins = []
    for origin_date, grp in archive_df.groupby("origin_date", sort=True):
        grp = grp.sort_values(["model", "horizon"])
        last_observed_price = float(grp["last_observed_price"].iloc[0])
        gate_regime_value = str(grp["gate_regime_value"].dropna().iloc[0])
        gate_selected_model = str(grp["gate_selected_model"].dropna().iloc[0])
        actual_path = []
        actual_grp = grp[grp["model"] == PAPER_MODEL]
        if not actual_grp.empty:
            for row in actual_grp.sort_values("horizon").itertuples(index=False):
                actual_path.append(
                    {
                        "horizon": int(row.horizon),
                        "target_date": str(row.target_date),
                        "actual_price": float(row.actual_price),
                    }
                )
        models = []
        for model_name, mg in grp.groupby("model", sort=False):
            forecast = [
                {
                    "horizon": int(row.horizon),
                    "target_date": str(row.target_date),
                    "predicted_price": float(row.predicted_price),
                }
                for row in mg.sort_values("horizon").itertuples(index=False)
            ]
            model_payload = {
                "model": model_name,
                "display_name": _display_name(model_name),
                "forecast": forecast,
            }
            if model_name == GATE_MODEL:
                model_payload["selected_underlying_model"] = gate_selected_model
                model_payload["regime_value"] = gate_regime_value
            models.append(model_payload)
        origins.append(
            {
                "origin_date": origin_date,
                "last_observed_price": last_observed_price,
                "gate_regime_value": gate_regime_value,
                "gate_selected_model": gate_selected_model,
                "actual_path": actual_path,
                "models": models,
            }
        )
    return {"origins": origins}


def generate_regime_gate_bundle(
    config_raw: dict[str, Any],
    data_dir: str | Path,
    run_dir: str | Path,
    output_dir: str | Path | None = None,
    publish_dir: str | Path | None = None,
    llm_api_key: str | None = None,
    llm4b_model: str = "qwen3-vl-4b-gpu",
    llm4b_base_url: str = DEFAULT_LOCAL_LLM_BASE_URL,
    llm35b_model: str = "qwen3.5-35b-a3b-ud-q4-k-xl",
    llm35b_base_url: str = "http://192.168.1.140:9881/v1",
    timeout_seconds: float = 120.0,
    max_origins: int | None = None,
) -> dict[str, Any]:
    run_dir = Path(run_dir).resolve()
    output_dir = Path(output_dir).resolve() if output_dir else run_dir / "frontend_live_bundle"
    (run_dir / "results").mkdir(parents=True, exist_ok=True)
    (run_dir / "predictions").mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    target_mode = str(config_raw.get("target", {}).get("mode", "returns")).lower()
    target_col = "y_return" if target_mode == "returns" else "y"
    seq_len = int(config_raw.get("time_series", {}).get("seq_len", 120))
    label_len = int(config_raw.get("time_series", {}).get("label_len", 30))
    pred_len = int(config_raw.get("time_series", {}).get("pred_len", 30))

    panel, _schema = build_panel(data_dir=Path(data_dir), config=config_raw, save_path=run_dir / "data")
    feature_cols = select_feature_columns(
        panel,
        target_col=target_col,
        max_exogenous_features=int((config_raw.get("features", {}) or {}).get("max_exogenous_features_model", 10)),
        preferred_feature_order=((config_raw.get("features", {}) or {}).get("preferred_feature_order")),
    )
    panel = panel[["date"] + [c for c in feature_cols if c in panel.columns]].copy()
    panel["date"] = pd.to_datetime(panel["date"])
    panel = panel.sort_values("date").reset_index(drop=True)

    archive_cfg = config_raw.get("frontend_live_bundle", {}) or {}
    origin_indices = _select_origin_indices(
        panel=panel,
        pred_len=pred_len,
        history_years=int(archive_cfg.get("history_years", 5)),
        origin_step=int(archive_cfg.get("origin_step", 63)),
        seq_len=seq_len,
        min_train_windows=int(archive_cfg.get("min_train_windows", 756)),
        min_val_windows=int(archive_cfg.get("min_val_windows", 126)),
    )
    if max_origins is not None:
        origin_indices = origin_indices[-int(max_origins) :]
    if not origin_indices:
        raise ValueError("No archive origins selected for regime gate bundle generation.")

    origin_dates = [pd.Timestamp(panel["date"].iloc[idx]).normalize() for idx in origin_indices]
    origin_regimes, regime_thresholds = build_origin_regime_frame(
        origin_dates,
        data_dir=data_dir,
        sentiment_path=Path("data/news/daily_sentiment.csv"),
    )
    regime_by_date = origin_regimes.set_index("date")

    base_llm_cfg = copy.deepcopy(config_raw.get("llm", {}) or {})
    candidates = _candidate_definitions(llm4b_model, llm4b_base_url, llm35b_model, llm35b_base_url)
    refiners: dict[str, LLMRefiner] = {}
    for candidate in candidates:
        llm_cfg = _resolve_llm_cfg(base_llm_cfg, candidate, api_key=llm_api_key, timeout_seconds=timeout_seconds)
        refiner_dir = run_dir / "llm" / candidate.display_name.replace(" ", "_")
        refiners[candidate.display_name] = LLMRefiner(
            llm_cfg,
            cache_dir=refiner_dir / "cache",
            log_dir=refiner_dir / "logs",
        )

    sentiment_map = load_daily_sentiment(
        str(Path("data/news/daily_sentiment.csv").resolve()),
        date_col="seendate",
        score_col="sent_score",
    )

    archive_rows: list[dict[str, Any]] = []
    progress_rows: list[dict[str, Any]] = []
    gate_assignment_rows: list[dict[str, Any]] = []

    for pos, origin_idx in enumerate(origin_indices, start=1):
        origin_date = pd.Timestamp(panel["date"].iloc[origin_idx]).normalize()
        prefix_panel = panel.iloc[: origin_idx + 1].copy()
        trained = train_tsm_on_prefix(prefix_panel=prefix_panel, config_raw=config_raw)
        batch_size = int(config_raw.get("model", {}).get("batch_size", 32))
        predictions, last_price, metadata_rows = _refine_models_for_prefix(
            prefix_panel=prefix_panel,
            full_panel=panel,
            trained=trained,
            target_mode=target_mode,
            target_col=target_col,
            seq_len=seq_len,
            label_len=label_len,
            pred_len=pred_len,
            batch_size=batch_size,
            candidates=candidates,
            refiners=refiners,
            sentiment_map=sentiment_map,
        )

        actual_slice = panel.iloc[origin_idx + 1 : origin_idx + pred_len + 1]
        actual_prices = actual_slice["y"].to_numpy(dtype=float)
        target_dates = [str(pd.Timestamp(d).date()) for d in actual_slice["date"].tolist()]
        regime_row = regime_by_date.loc[origin_date]
        regime_value = str(regime_row["sentiment_alignment_5d"])
        selected_model = SENTIMENT_ALIGNMENT_GATE.get(regime_value, PAPER_MODEL)
        gate_assignment_rows.append(
            {
                "date": str(origin_date.date()),
                "gate_sentiment_alignment_5d__regime_value": regime_value,
                "gate_sentiment_alignment_5d__selected_model": selected_model,
                **{k: (str(v.date()) if isinstance(v, pd.Timestamp) else v) for k, v in regime_row.to_dict().items()},
            }
        )

        archive_rows.extend(
            _archive_rows_for_origin(
                origin_date=str(origin_date.date()),
                target_dates=target_dates,
                actual_prices=actual_prices,
                last_observed_price=last_price,
                predictions=predictions,
                regime_value=regime_value,
                selected_model=selected_model,
            )
        )

        for model_name, forecast in {**predictions, GATE_MODEL: predictions[selected_model]}.items():
            progress_rows.append(
                {
                    "origin_date": str(origin_date.date()),
                    "model": model_name,
                    "path_mse": float(np.mean((actual_prices - forecast) ** 2)),
                    "selected_underlying_model": selected_model if model_name == GATE_MODEL else None,
                    "gate_regime_value": regime_value if model_name == GATE_MODEL else None,
                }
            )
        pd.DataFrame(progress_rows).to_csv(run_dir / "results" / "archive_origin_metrics_partial.csv", index=False)
        print(f"[{pos}/{len(origin_indices)}] {origin_date.date()} complete", flush=True)

    # current forward forecast
    latest_date = pd.Timestamp(panel["date"].iloc[-1]).normalize()
    current_regime_df, _ = build_origin_regime_frame(
        [latest_date],
        data_dir=data_dir,
        sentiment_path=Path("data/news/daily_sentiment.csv"),
    )
    current_regime = str(current_regime_df.iloc[0]["sentiment_alignment_5d"])
    current_selected_model = SENTIMENT_ALIGNMENT_GATE.get(current_regime, PAPER_MODEL)
    current_trained = train_tsm_on_prefix(prefix_panel=panel.copy(), config_raw=config_raw)
    current_predictions, current_last_price, _ = _refine_models_for_prefix(
        prefix_panel=panel.copy(),
        full_panel=panel,
        trained=current_trained,
        target_mode=target_mode,
        target_col=target_col,
        seq_len=seq_len,
        label_len=label_len,
        pred_len=pred_len,
        batch_size=int(config_raw.get("model", {}).get("batch_size", 32)),
        candidates=candidates,
        refiners=refiners,
        sentiment_map=sentiment_map,
    )
    current_target_dates = [
        str(d.date()) for d in pd.bdate_range(latest_date + pd.Timedelta(days=1), periods=pred_len)
    ]

    archive_df = pd.DataFrame(archive_rows)
    archive_df.to_csv(run_dir / "predictions" / "forecast_archive.csv", index=False)
    archive_df.to_csv(output_dir / "forecast_archive.csv", index=False)
    archive_json = _build_archive_json(archive_df)
    archive_summary = _summarize_archive(archive_df)
    pd.DataFrame(progress_rows).to_csv(run_dir / "results" / "archive_origin_metrics.csv", index=False)
    pd.DataFrame(archive_summary.get("models", [])).to_csv(run_dir / "results" / "archive_model_summary.csv", index=False)
    pd.DataFrame(gate_assignment_rows).to_csv(run_dir / "results" / "gate_assignments_by_origin.csv", index=False)
    pd.DataFrame(gate_assignment_rows).to_csv(output_dir / "gate_assignments_by_origin.csv", index=False)

    actual_history = _collect_actual_history(panel, int(archive_cfg.get("actual_history_days", 365)))
    current_models = []
    for model_name, forecast in {**current_predictions, GATE_MODEL: current_predictions[current_selected_model]}.items():
        payload = {
            "model": model_name,
            "display_name": _display_name(model_name),
            "forecast": [
                {
                    "horizon": int(i + 1),
                    "target_date": current_target_dates[i],
                    "predicted_price": float(forecast[i]),
                }
                for i in range(pred_len)
            ],
        }
        if model_name == GATE_MODEL:
            payload["selected_underlying_model"] = current_selected_model
            payload["regime_value"] = current_regime
        current_models.append(payload)

    current_price_payload = {
        "artifact_version": 2,
        "exported_at_utc": _iso_utc_now(),
        "run_id": run_dir.name,
        "instrument": "EUA_FUTURES",
        "display_name": "EUA Futures",
        "price": float(panel["y"].iloc[-1]),
        "as_of_date": str(latest_date.date()),
        "source_frequency": "daily_close",
        "delayed": True,
    }
    current_forecast_payload = {
        "artifact_version": 2,
        "exported_at_utc": _iso_utc_now(),
        "run_id": run_dir.name,
        "forecast_kind": "live_forward_daily_close",
        "strategy": GATE_MODEL,
        "selected_model": current_selected_model,
        "regime_family": "sentiment_alignment_5d",
        "regime_value": current_regime,
        "origin_date": str(latest_date.date()),
        "data_cutoff_date": str(latest_date.date()),
        "last_observed_price": float(current_last_price),
        "models": current_models,
    }
    actuals_recent_payload = {
        "artifact_version": 2,
        "exported_at_utc": _iso_utc_now(),
        "run_id": run_dir.name,
        "data_cutoff_date": str(latest_date.date()),
        "history": actual_history,
    }
    status_payload = {
        "artifact_version": 2,
        "exported_at_utc": _iso_utc_now(),
        "run_id": run_dir.name,
        "contamination_safe": True,
        "strategy": GATE_MODEL,
        "selected_model": current_selected_model,
        "regime_family": "sentiment_alignment_5d",
        "regime_value": current_regime,
        "data_cutoff_date": str(latest_date.date()),
        "archive_start_date": str(origin_dates[0].date()),
        "archive_end_date": str(origin_dates[-1].date()),
        "requested_history_years": int(archive_cfg.get("history_years", 5)),
        "archive_origin_step": int(archive_cfg.get("origin_step", 63)),
        "actual_history_days": int(archive_cfg.get("actual_history_days", 365)),
        "n_archive_origins": int(len(origin_indices)),
        "models": [model["model"] for model in current_models],
    }
    manifest_payload = {
        "artifact_version": 2,
        "exported_at_utc": _iso_utc_now(),
        "run_id": run_dir.name,
        "source_run_dir": str(run_dir),
        "strategy": GATE_MODEL,
        "files": [
            "current_price.json",
            "actuals_recent.json",
            "current_forecast.json",
            "forecast_archive.csv",
            "forecast_archive.json",
            "model_summary.json",
            "status.json",
            "manifest.json",
            "gate_assignments_by_origin.csv",
        ],
    }

    _write_json(output_dir / "current_price.json", current_price_payload)
    _write_json(output_dir / "actuals_recent.json", actuals_recent_payload)
    _write_json(output_dir / "current_forecast.json", current_forecast_payload)
    _write_json(
        output_dir / "forecast_archive.json",
        {
            "artifact_version": 2,
            "exported_at_utc": _iso_utc_now(),
            "run_id": run_dir.name,
            "strategy": GATE_MODEL,
            **archive_json,
        },
    )
    _write_json(
        output_dir / "model_summary.json",
        {
            "artifact_version": 2,
            "exported_at_utc": _iso_utc_now(),
            "run_id": run_dir.name,
            "strategy": GATE_MODEL,
            **archive_summary,
        },
    )
    _write_json(output_dir / "status.json", status_payload)
    _write_json(output_dir / "manifest.json", manifest_payload)

    if publish_dir is not None:
        _copy_outputs(output_dir, Path(publish_dir).resolve(), manifest_payload["files"])

    return {
        "run_dir": run_dir,
        "output_dir": output_dir,
        "current_price": current_price_payload,
        "archive_summary": archive_summary,
        "status": status_payload,
    }
