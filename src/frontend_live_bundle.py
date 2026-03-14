from __future__ import annotations

import copy
import json
import logging
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .data import build_panel, select_feature_columns
from .data.windows import StandardScaler, TimeSeriesDataset
from .eval.metrics import compute_metrics_by_horizon, compute_path_metrics
from .llm import LLMRefiner
from .models.tsm import TSMForecaster
from .run_experiment import blend_forecasts, returns_to_prices

logger = logging.getLogger(__name__)

DEFAULT_BEST_BLEND_MODEL = "TSM+LLM-COT-RF_blend_ramp_bestval_path_h1base"
DEFAULT_MODEL_DISPLAY_NAMES = {
    "tsm": "Base TSM",
    DEFAULT_BEST_BLEND_MODEL: "TSM + CoT Ramp Final Blend",
}
DEFAULT_LOCAL_LLM_BASE_URL = "http://192.168.1.140:9877/v1"
DEFAULT_LOCAL_LLM_API_KEY = "deo"
DEFAULT_INSTRUMENT_DISPLAY_NAMES = {
    "EUA_FUTURES": "EUA Futures",
    "UKA_FUTURES": "UKA Futures",
}


@dataclass
class RollingArchiveConfig:
    history_years: int = 5
    origin_step: int = 63
    actual_history_days: int = 365
    val_windows: int = 252
    min_train_windows: int = 756
    min_val_windows: int = 126


@dataclass
class PreparedWindows:
    feature_cols: list[str]
    target_col: str
    target_mode: str
    dates: np.ndarray
    X_enc: np.ndarray
    X_dec: np.ndarray
    y: np.ndarray
    pred_dates: np.ndarray
    base_prices: np.ndarray
    price_histories: np.ndarray


@dataclass
class TrainedArtifacts:
    model: TSMForecaster
    scaler: StandardScaler
    windows: PreparedWindows
    train_slice: slice
    val_slice: slice
    val_pred_prices: np.ndarray
    val_true_prices: np.ndarray


@dataclass
class ForecastResult:
    model_name: str
    display_name: str
    origin_date: str
    last_observed_price: float
    forecast_prices: np.ndarray
    target_dates: list[str]
    actual_prices: np.ndarray | None = None
    metadata: dict[str, Any] | None = None


def _iso_utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _copy_outputs(output_dir: Path, publish_dir: Path, filenames: list[str]) -> None:
    publish_dir.mkdir(parents=True, exist_ok=True)
    for name in filenames:
        shutil.copy2(output_dir / name, publish_dir / name)


def _display_name(model_name: str) -> str:
    return DEFAULT_MODEL_DISPLAY_NAMES.get(model_name, model_name)


def _instrument_display_name(instrument: str) -> str:
    return DEFAULT_INSTRUMENT_DISPLAY_NAMES.get(instrument, str(instrument).replace("_", " ").title())


def _select_feature_cols(panel: pd.DataFrame, target_col: str, max_features: int = 10) -> list[str]:
    return select_feature_columns(
        panel,
        target_col=target_col,
        max_exogenous_features=max_features,
    )


def _prepare_panel(panel: pd.DataFrame, target_col: str, feature_cols: list[str]) -> pd.DataFrame:
    out = panel.sort_values("date").drop_duplicates("date").copy()
    out["date"] = pd.to_datetime(out["date"])
    out = out[out["y"].notna()].copy()
    if target_col in out.columns:
        out = out[out[target_col].notna()].copy()
    needed = ["date"] + [c for c in feature_cols if c in out.columns]
    return out[needed].reset_index(drop=True)


def _fill_feature_frame(frame: pd.DataFrame) -> pd.DataFrame:
    filled = frame.copy()
    for col in filled.columns:
        filled[col] = pd.to_numeric(filled[col], errors="coerce")
    filled = filled.ffill()
    col_mean = filled.mean(skipna=True).fillna(0.0)
    return filled.fillna(col_mean)


def _future_fill_value(name: str, last_value: float, target_col: str) -> float:
    lname = str(name).lower()
    if name == target_col and target_col.endswith("_return"):
        return 0.0
    if lname.endswith("_return") or lname.endswith("_pct"):
        return 0.0
    if lname.startswith("is_") or lname.endswith("_flag"):
        return 0.0
    return float(last_value)


def make_clean_windows(
    panel: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    seq_len: int,
    label_len: int,
    pred_len: int,
    target_mode: str,
) -> PreparedWindows:
    if target_col not in panel.columns:
        raise ValueError(f"Missing target column: {target_col}")
    if "y" not in panel.columns:
        raise ValueError("Panel must include observed price column 'y'")

    work = panel[["date"] + feature_cols].copy()
    work = _fill_feature_frame(work[feature_cols]).assign(date=pd.to_datetime(panel["date"]))
    work = work[["date"] + feature_cols]

    data = work[feature_cols].to_numpy(dtype=np.float32)
    target = work[target_col].to_numpy(dtype=np.float32)
    dates = pd.to_datetime(work["date"]).to_numpy()
    price_idx = feature_cols.index("y")
    prices = data[:, price_idx].astype(np.float32)

    n_windows = len(work) - seq_len - pred_len + 1
    if n_windows <= 0:
        raise ValueError(
            f"Not enough rows for clean windows: need at least {seq_len + pred_len}, have {len(work)}"
        )

    X_enc = np.zeros((n_windows, seq_len, len(feature_cols)), dtype=np.float32)
    X_dec = np.zeros((n_windows, label_len + pred_len, len(feature_cols)), dtype=np.float32)
    y = np.zeros((n_windows, pred_len), dtype=np.float32)
    pred_dates = np.empty((n_windows,), dtype="datetime64[ns]")
    base_prices = np.zeros((n_windows,), dtype=np.float32)
    price_histories = np.zeros((n_windows, seq_len), dtype=np.float32)

    for i in range(n_windows):
        enc_start = i
        enc_end = i + seq_len
        y_start = enc_end
        y_end = enc_end + pred_len
        X_enc[i] = data[enc_start:enc_end]
        context = data[enc_end - label_len:enc_end]
        last_row = data[enc_end - 1]
        future_block = np.repeat(last_row[None, :], pred_len, axis=0).astype(np.float32)
        for j, name in enumerate(feature_cols):
            future_block[:, j] = _future_fill_value(name, last_row[j], target_col)
        X_dec[i] = np.concatenate([context, future_block], axis=0)
        y[i] = target[y_start:y_end]
        pred_dates[i] = dates[y_start]
        base_prices[i] = prices[enc_end - 1]
        price_histories[i] = prices[enc_start:enc_end]

    return PreparedWindows(
        feature_cols=feature_cols,
        target_col=target_col,
        target_mode=target_mode,
        dates=dates,
        X_enc=X_enc,
        X_dec=X_dec,
        y=y,
        pred_dates=pred_dates,
        base_prices=base_prices,
        price_histories=price_histories,
    )


def _split_train_val_indices(
    n_windows: int,
    val_windows: int,
    min_train_windows: int,
    min_val_windows: int,
) -> tuple[slice, slice]:
    if n_windows < min_train_windows + min_val_windows:
        raise ValueError(
            f"Insufficient clean windows ({n_windows}) for train/val split; "
            f"need at least {min_train_windows + min_val_windows}."
        )
    val_count = min(int(val_windows), n_windows - min_train_windows)
    val_count = max(int(min_val_windows), val_count)
    train_count = n_windows - val_count
    if train_count < min_train_windows:
        raise ValueError(
            f"Insufficient training windows after split ({train_count} < {min_train_windows})."
        )
    return slice(0, train_count), slice(train_count, n_windows)


def _scale_target(y: np.ndarray, scaler: StandardScaler) -> np.ndarray:
    return ((y - scaler.mean_[0]) / scaler.std_[0]).astype(np.float32)


def _make_loader(X_enc: np.ndarray, X_dec: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool):
    from torch.utils.data import DataLoader

    dataset = TimeSeriesDataset(X_enc, X_dec, y)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)


def _predict_scaled(model: TSMForecaster, X_enc: np.ndarray, X_dec: np.ndarray, batch_size: int) -> np.ndarray:
    from torch.utils.data import DataLoader

    dummy_y = np.zeros((len(X_enc), model.config.get("time_series", {}).get("pred_len", X_dec.shape[1])), dtype=np.float32)
    dataset = TimeSeriesDataset(X_enc, X_dec, dummy_y)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    pred_scaled, _ = model.predict(loader)
    return pred_scaled


def _predict_prices(
    model: TSMForecaster,
    scaler: StandardScaler,
    X_enc: np.ndarray,
    X_dec: np.ndarray,
    base_prices: np.ndarray,
    target_mode: str,
    batch_size: int,
) -> np.ndarray:
    pred_scaled = _predict_scaled(model, scaler.transform(X_enc), scaler.transform(X_dec), batch_size=batch_size)
    pred = scaler.inverse_transform_target(pred_scaled)
    if target_mode == "returns":
        return returns_to_prices(pred, base_prices)
    return pred


def _history_features(history: np.ndarray, feature_window: int) -> np.ndarray:
    window = history[-feature_window:] if len(history) >= feature_window else history
    mean = float(np.mean(window)) if len(window) else 0.0
    std = float(np.std(window)) if len(window) else 0.0
    if len(window) > 1:
        slope = float(np.polyfit(np.arange(len(window)), window, 1)[0])
    else:
        slope = 0.0
    return np.array([mean, std, slope], dtype=float)


def _select_teaching_examples(
    trained: TrainedArtifacts,
    current_history: np.ndarray,
    current_origin_date: pd.Timestamp,
    llm_cfg: dict[str, Any],
) -> list[dict[str, Any]]:
    cot_cfg = llm_cfg.get("cot_rf", {}) or {}
    k_examples = int(cot_cfg.get("k_examples", 5))
    selection_mode = str(cot_cfg.get("example_selection", "recent")).lower()
    feature_window = int(cot_cfg.get("feature_window", 18))
    lookback_days = cot_cfg.get("lookback_days")
    lookback_days = int(lookback_days) if lookback_days is not None else None

    val_idx = np.arange(trained.val_slice.start, trained.val_slice.stop)
    val_pred_dates = pd.to_datetime(trained.windows.pred_dates[val_idx])
    val_histories = trained.windows.price_histories[val_idx]
    val_truth = trained.val_true_prices
    val_forecasts = trained.val_pred_prices

    if lookback_days is not None:
        cutoff = current_origin_date - pd.Timedelta(days=lookback_days)
        mask = val_pred_dates >= cutoff
        if mask.any():
            val_pred_dates = val_pred_dates[mask]
            val_histories = val_histories[mask]
            val_truth = val_truth[mask]
            val_forecasts = val_forecasts[mask]

    if len(val_pred_dates) == 0:
        return []

    if selection_mode == "similarity":
        sample_feat = _history_features(current_history, feature_window)
        cand_feats = np.vstack([_history_features(hist, feature_window) for hist in val_histories])
        distances = np.linalg.norm(cand_feats - sample_feat, axis=1)
        choice = np.argsort(distances)[:k_examples]
        choice = np.sort(choice)
    else:
        choice = np.arange(max(0, len(val_pred_dates) - k_examples), len(val_pred_dates))

    examples = []
    for idx in choice:
        examples.append(
            {
                "history": val_histories[idx],
                "forecast": val_forecasts[idx],
                "truth": val_truth[idx],
                "date": str(pd.Timestamp(val_pred_dates[idx]).date()),
                "sentiment_history": [],
            }
        )
    return examples


def train_tsm_on_prefix(prefix_panel: pd.DataFrame, config_raw: dict[str, Any]) -> TrainedArtifacts:
    target_mode = str(config_raw.get("target", {}).get("mode", "returns")).lower()
    target_col = "y_return" if target_mode == "returns" else "y"
    feature_cols = _select_feature_cols(prefix_panel, target_col=target_col)
    prepared_panel = _prepare_panel(prefix_panel, target_col=target_col, feature_cols=feature_cols)
    windows = make_clean_windows(
        panel=prepared_panel,
        feature_cols=feature_cols,
        target_col=target_col,
        seq_len=int(config_raw.get("time_series", {}).get("seq_len", 120)),
        label_len=int(config_raw.get("time_series", {}).get("label_len", 30)),
        pred_len=int(config_raw.get("time_series", {}).get("pred_len", 30)),
        target_mode=target_mode,
    )

    archive_cfg = config_raw.get("frontend_live_bundle", {}) or {}
    train_slice, val_slice = _split_train_val_indices(
        n_windows=len(windows.y),
        val_windows=int(archive_cfg.get("val_windows", 252)),
        min_train_windows=int(archive_cfg.get("min_train_windows", 756)),
        min_val_windows=int(archive_cfg.get("min_val_windows", 126)),
    )

    scaler = StandardScaler()
    scaler.fit(windows.X_enc[train_slice])

    batch_size = int(config_raw.get("model", {}).get("batch_size", 32))
    train_loader = _make_loader(
        scaler.transform(windows.X_enc[train_slice]),
        scaler.transform(windows.X_dec[train_slice]),
        _scale_target(windows.y[train_slice], scaler),
        batch_size=batch_size,
        shuffle=True,
    )
    val_loader = _make_loader(
        scaler.transform(windows.X_enc[val_slice]),
        scaler.transform(windows.X_dec[val_slice]),
        _scale_target(windows.y[val_slice], scaler),
        batch_size=batch_size,
        shuffle=False,
    )

    tsm_config = copy.deepcopy(config_raw)
    tsm_config.setdefault("model", {})["enc_in"] = windows.X_enc.shape[-1]
    tsm_config.setdefault("model", {})["dec_in"] = windows.X_dec.shape[-1]

    model = TSMForecaster(tsm_config)
    model.fit(
        train_loader,
        val_loader,
        epochs=int(tsm_config.get("model", {}).get("max_epochs", 100)),
        patience=int(tsm_config.get("model", {}).get("early_stopping_patience", 10)),
        save_path=None,
    )

    val_pred_prices = _predict_prices(
        model=model,
        scaler=scaler,
        X_enc=windows.X_enc[val_slice],
        X_dec=windows.X_dec[val_slice],
        base_prices=windows.base_prices[val_slice],
        target_mode=target_mode,
        batch_size=batch_size,
    )
    if target_mode == "returns":
        val_true_prices = returns_to_prices(windows.y[val_slice], windows.base_prices[val_slice])
    else:
        val_true_prices = windows.y[val_slice]

    return TrainedArtifacts(
        model=model,
        scaler=scaler,
        windows=windows,
        train_slice=train_slice,
        val_slice=val_slice,
        val_pred_prices=val_pred_prices,
        val_true_prices=val_true_prices,
    )


def _build_live_inputs(
    prefix_panel: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    seq_len: int,
    label_len: int,
) -> tuple[np.ndarray, np.ndarray, float, np.ndarray]:
    prepared = _prepare_panel(prefix_panel, target_col=target_col, feature_cols=feature_cols)
    work = _fill_feature_frame(prepared[feature_cols])
    data = work[feature_cols].to_numpy(dtype=np.float32)
    if len(data) < seq_len:
        raise ValueError(f"Need at least {seq_len} rows for live input, have {len(data)}")

    X_enc = data[-seq_len:]
    context = data[-label_len:]
    last_row = data[-1]
    future_block = np.repeat(last_row[None, :], int(prefix_panel.attrs.get("pred_len", 30)), axis=0).astype(np.float32)
    for j, name in enumerate(feature_cols):
        future_block[:, j] = _future_fill_value(name, last_row[j], target_col)
    X_dec = np.concatenate([context, future_block], axis=0)
    price_idx = feature_cols.index("y")
    last_price = float(data[-1, price_idx])
    history_prices = data[-seq_len:, price_idx].astype(np.float32)
    return X_enc, X_dec, last_price, history_prices


def _predict_current_forecast(
    trained: TrainedArtifacts,
    prefix_panel: pd.DataFrame,
    config_raw: dict[str, Any],
    llm_refiner: LLMRefiner | None,
) -> tuple[ForecastResult, ForecastResult | None]:
    target_mode = str(config_raw.get("target", {}).get("mode", "returns")).lower()
    target_col = "y_return" if target_mode == "returns" else "y"
    seq_len = int(config_raw.get("time_series", {}).get("seq_len", 120))
    label_len = int(config_raw.get("time_series", {}).get("label_len", 30))
    pred_len = int(config_raw.get("time_series", {}).get("pred_len", 30))
    batch_size = int(config_raw.get("model", {}).get("batch_size", 32))

    prefix_panel = prefix_panel.copy()
    prefix_panel.attrs["pred_len"] = pred_len
    X_enc, X_dec, last_price, current_history = _build_live_inputs(
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

    origin_date = str(pd.Timestamp(prefix_panel["date"].iloc[-1]).date())
    future_dates = [
        str(d.date())
        for d in pd.bdate_range(pd.Timestamp(origin_date) + pd.Timedelta(days=1), periods=pred_len)
    ]
    tsm_result = ForecastResult(
        model_name="tsm",
        display_name=_display_name("tsm"),
        origin_date=origin_date,
        last_observed_price=float(last_price),
        forecast_prices=base_pred,
        target_dates=future_dates,
        actual_prices=None,
        metadata={"contamination_safe": True},
    )

    if llm_refiner is None:
        return tsm_result, None

    history_points = min(int(config_raw.get("llm", {}).get("history_points", 18)), len(current_history))
    history = current_history[-history_points:]
    history_dates = [
        str(d.date())
        for d in pd.to_datetime(prefix_panel["date"]).tail(history_points).tolist()
    ]
    teaching_examples = _select_teaching_examples(
        trained=trained,
        current_history=current_history,
        current_origin_date=pd.Timestamp(origin_date),
        llm_cfg=config_raw.get("llm", {}),
    )
    if not teaching_examples:
        logger.warning("No teaching examples available for current blend forecast; skipping LLM blend.")
        return tsm_result, None

    llm_pred, llm_meta = llm_refiner.refine(
        method="TSM+LLM-COT-RF",
        history=history,
        dates=history_dates,
        tsm_forecast=base_pred,
        pred_len=pred_len,
        exogenous_summary=None,
        price_base=float(last_price),
        teaching_examples=teaching_examples,
        sentiment_history=None,
    )
    if llm_pred is None:
        logger.warning("LLM refinement failed for current forecast; skipping blend.")
        return tsm_result, None

    blended = blend_forecasts(
        base_pred=base_pred[None, :],
        llm_pred=np.asarray(llm_pred, dtype=float)[None, :],
        strength=0.5,
        schedule="ramp",
        pred_len=pred_len,
        min_weight=0.15,
        power=1.0,
    )[0]
    blended[0] = base_pred[0]

    blend_result = ForecastResult(
        model_name=DEFAULT_BEST_BLEND_MODEL,
        display_name=_display_name(DEFAULT_BEST_BLEND_MODEL),
        origin_date=origin_date,
        last_observed_price=float(last_price),
        forecast_prices=blended,
        target_dates=future_dates,
        actual_prices=None,
        metadata={
            "llm_success": True,
            "blend_schedule": "ramp",
            "blend_strength": 0.5,
            "blend_min_weight": 0.15,
            "blend_power": 1.0,
            "h1_override_model": "tsm",
            "llm_metadata": llm_meta,
        },
    )
    return tsm_result, blend_result


def _run_origin_forecast(
    panel: pd.DataFrame,
    origin_idx: int,
    config_raw: dict[str, Any],
    llm_refiner: LLMRefiner | None,
) -> list[ForecastResult]:
    pred_len = int(config_raw.get("time_series", {}).get("pred_len", 30))
    prefix_panel = panel.iloc[: origin_idx + 1].copy()
    trained = train_tsm_on_prefix(prefix_panel=prefix_panel, config_raw=config_raw)
    target_mode = str(config_raw.get("target", {}).get("mode", "returns")).lower()
    target_col = "y_return" if target_mode == "returns" else "y"
    seq_len = int(config_raw.get("time_series", {}).get("seq_len", 120))
    label_len = int(config_raw.get("time_series", {}).get("label_len", 30))
    batch_size = int(config_raw.get("model", {}).get("batch_size", 32))

    prefix_panel = prefix_panel.copy()
    prefix_panel.attrs["pred_len"] = pred_len
    X_enc, X_dec, last_price, current_history = _build_live_inputs(
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

    origin_date = str(pd.Timestamp(panel["date"].iloc[origin_idx]).date())
    actual_slice = panel.iloc[origin_idx + 1 : origin_idx + pred_len + 1]
    actual_prices = actual_slice["y"].to_numpy(dtype=float)
    target_dates = [str(pd.Timestamp(d).date()) for d in actual_slice["date"].tolist()]

    results = [
        ForecastResult(
            model_name="tsm",
            display_name=_display_name("tsm"),
            origin_date=origin_date,
            last_observed_price=float(last_price),
            forecast_prices=base_pred,
            target_dates=target_dates,
            actual_prices=actual_prices,
            metadata={"contamination_safe": True},
        )
    ]

    if llm_refiner is None:
        return results

    history_points = min(int(config_raw.get("llm", {}).get("history_points", 18)), len(current_history))
    history = current_history[-history_points:]
    history_dates = [str(d.date()) for d in pd.to_datetime(prefix_panel["date"]).tail(history_points).tolist()]
    teaching_examples = _select_teaching_examples(
        trained=trained,
        current_history=current_history,
        current_origin_date=pd.Timestamp(origin_date),
        llm_cfg=config_raw.get("llm", {}),
    )
    if not teaching_examples:
        logger.warning("No teaching examples available at origin %s; skipping blend forecast.", origin_date)
        return results

    llm_pred, llm_meta = llm_refiner.refine(
        method="TSM+LLM-COT-RF",
        history=history,
        dates=history_dates,
        tsm_forecast=base_pred,
        pred_len=pred_len,
        exogenous_summary=None,
        price_base=float(last_price),
        teaching_examples=teaching_examples,
        sentiment_history=None,
    )
    if llm_pred is None:
        logger.warning("LLM refinement failed at origin %s; skipping blend forecast.", origin_date)
        return results

    blended = blend_forecasts(
        base_pred=base_pred[None, :],
        llm_pred=np.asarray(llm_pred, dtype=float)[None, :],
        strength=0.5,
        schedule="ramp",
        pred_len=pred_len,
        min_weight=0.15,
        power=1.0,
    )[0]
    blended[0] = base_pred[0]

    results.append(
        ForecastResult(
            model_name=DEFAULT_BEST_BLEND_MODEL,
            display_name=_display_name(DEFAULT_BEST_BLEND_MODEL),
            origin_date=origin_date,
            last_observed_price=float(last_price),
            forecast_prices=blended,
            target_dates=target_dates,
            actual_prices=actual_prices,
            metadata={
                "llm_success": True,
                "blend_schedule": "ramp",
                "blend_strength": 0.5,
                "blend_min_weight": 0.15,
                "blend_power": 1.0,
                "h1_override_model": "tsm",
                "llm_metadata": llm_meta,
            },
        )
    )
    return results


def _collect_actual_history(panel: pd.DataFrame, history_days: int) -> list[dict[str, Any]]:
    tail = panel[["date", "y"]].dropna().tail(int(history_days)).copy()
    tail["date"] = pd.to_datetime(tail["date"]).dt.strftime("%Y-%m-%d")
    return [{"date": str(row.date), "price": float(row.y)} for row in tail.itertuples(index=False)]


def _build_archive_rows(results: list[ForecastResult]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for result in results:
        actuals = result.actual_prices
        for idx, target_date in enumerate(result.target_dates):
            actual = None if actuals is None else float(actuals[idx])
            rows.append(
                {
                    "origin_date": result.origin_date,
                    "target_date": str(target_date),
                    "model": result.model_name,
                    "display_name": result.display_name,
                    "horizon": int(idx + 1),
                    "predicted_price": float(result.forecast_prices[idx]),
                    "actual_price": actual,
                    "last_observed_price": float(result.last_observed_price),
                    "is_realized": bool(actual is not None),
                }
            )
    return pd.DataFrame(rows)


def _build_archive_json(results: list[ForecastResult]) -> dict[str, Any]:
    grouped: list[dict[str, Any]] = []
    by_origin: dict[str, list[ForecastResult]] = {}
    for result in results:
        by_origin.setdefault(result.origin_date, []).append(result)

    for origin_date in sorted(by_origin):
        items = by_origin[origin_date]
        actual = items[0].actual_prices if items else None
        target_dates = items[0].target_dates if items else []
        grouped.append(
            {
                "origin_date": origin_date,
                "last_observed_price": float(items[0].last_observed_price),
                "actual_path": [
                    {
                        "horizon": int(i + 1),
                        "target_date": str(target_dates[i]),
                        "actual_price": float(actual[i]),
                    }
                    for i in range(len(target_dates))
                ]
                if actual is not None
                else [],
                "models": [
                    {
                        "model": item.model_name,
                        "display_name": item.display_name,
                        "forecast": [
                            {
                                "horizon": int(i + 1),
                                "target_date": str(item.target_dates[i]),
                                "predicted_price": float(item.forecast_prices[i]),
                            }
                            for i in range(len(item.target_dates))
                        ],
                    }
                    for item in items
                ],
            }
        )
    return {"origins": grouped}


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
        y_true_arr = np.vstack(y_true)
        y_pred_arr = np.vstack(y_pred)
        metrics_by_h = compute_metrics_by_horizon(y_true_arr, y_pred_arr, horizons=list(range(1, y_true_arr.shape[1] + 1)))
        path_metrics = compute_path_metrics(y_true_arr, y_pred_arr)
        key_h = {}
        for h in (1, 5, 20, 30):
            if h in metrics_by_h.index:
                row = metrics_by_h.loc[h]
                key_h[f"h{h}"] = {
                    "mse": float(row["mse"]),
                    "rmse": float(row["rmse"]),
                    "mae": float(row["mae"]),
                    "mape": float(row.get("mape", np.nan)),
                }
        models.append(
            {
                "model": model_name,
                "display_name": _display_name(model_name),
                "n_origins": int(grp["origin_date"].nunique()),
                "n_rows": int(len(grp)),
                "path_mse": float(path_metrics["mse_path"]),
                "path_rmse": float(path_metrics["rmse_path"]),
                "path_mae": float(path_metrics["mae_path"]),
                "horizons": key_h,
            }
        )
    return {"models": models}


def _resolve_llm_config(config_raw: dict[str, Any], llm_model: str, llm_api_key: str | None, llm_base_url: str | None) -> dict[str, Any]:
    llm = copy.deepcopy(config_raw.get("llm", {}))
    llm.update(
        {
            "provider": "openai",
            "model": llm_model,
            "temperature": 0.0,
            "max_tokens": 512,
            "timeout_seconds": 300,
            "methods": ["TSM+LLM-COT-RF"],
            "history_points": 18,
            "include_summary_stats": True,
            "max_retries": 3,
            "cache_enabled": True,
            "max_samples": 100000,
            "subset": {"strategy": "random", "seed": 42},
            "max_exogenous_features": 6,
            "prompt_history_points": 18,
            "prompt_sentiment_points": 18,
            "api_key": llm_api_key or DEFAULT_LOCAL_LLM_API_KEY,
            "base_url": llm_base_url or DEFAULT_LOCAL_LLM_BASE_URL,
            "cot_rf": {
                "k_examples": 5,
                "example_selection": "similarity",
                "feature_window": 18,
                "lookback_days": 365,
                "retain_context": True,
            },
            "sentiment": {
                "enabled": False,
                "path": "data/news/daily_sentiment.csv",
                "date_col": "seendate",
                "score_col": "sent_score",
                "history_points": 18,
            },
        }
    )
    return llm


def _select_origin_indices(
    panel: pd.DataFrame,
    pred_len: int,
    history_years: int,
    origin_step: int,
    seq_len: int,
    min_train_windows: int,
    min_val_windows: int,
) -> list[int]:
    max_origin_idx = len(panel) - pred_len - 1
    if max_origin_idx <= 0:
        raise ValueError("Not enough data to create realized archive origins.")

    earliest_idx = seq_len + pred_len + min_train_windows + min_val_windows - 2
    requested_start_date = pd.Timestamp(panel["date"].max()) - pd.DateOffset(years=int(history_years))

    candidate_indices = [
        idx
        for idx in range(max(0, earliest_idx), max_origin_idx + 1)
        if pd.Timestamp(panel["date"].iloc[idx]) >= requested_start_date
    ]
    if not candidate_indices:
        candidate_indices = list(range(max(0, earliest_idx), max_origin_idx + 1))
    if not candidate_indices:
        raise ValueError("No feasible archive origins for the requested history window.")

    selected = candidate_indices[:: max(1, int(origin_step))]
    if selected[-1] != candidate_indices[-1]:
        selected.append(candidate_indices[-1])
    return selected


def generate_frontend_live_bundle(
    config_raw: dict[str, Any],
    data_dir: str | Path,
    run_dir: str | Path,
    output_dir: str | Path | None = None,
    publish_dir: str | Path | None = None,
    llm_enabled: bool = True,
    llm_model: str = "qwen3-vl-4b-gpu",
    llm_api_key: str | None = None,
    llm_base_url: str | None = None,
    max_origins: int | None = None,
) -> dict[str, Any]:
    run_dir = Path(run_dir).resolve()
    output_dir = Path(output_dir).resolve() if output_dir else run_dir / "frontend_live_bundle"
    output_dir.mkdir(parents=True, exist_ok=True)

    target_mode = str(config_raw.get("target", {}).get("mode", "returns")).lower()
    target_col = "y_return" if target_mode == "returns" else "y"
    seq_len = int(config_raw.get("time_series", {}).get("seq_len", 120))
    pred_len = int(config_raw.get("time_series", {}).get("pred_len", 30))

    panel, schema = build_panel(data_dir=Path(data_dir), config=config_raw, save_path=run_dir / "data")
    panel = _prepare_panel(panel, target_col=target_col, feature_cols=_select_feature_cols(panel, target_col))

    archive_cfg = RollingArchiveConfig(
        history_years=int(config_raw.get("frontend_live_bundle", {}).get("history_years", 5)),
        origin_step=int(config_raw.get("frontend_live_bundle", {}).get("origin_step", 63)),
        actual_history_days=int(config_raw.get("frontend_live_bundle", {}).get("actual_history_days", 365)),
        val_windows=int(config_raw.get("frontend_live_bundle", {}).get("val_windows", 252)),
        min_train_windows=int(config_raw.get("frontend_live_bundle", {}).get("min_train_windows", 756)),
        min_val_windows=int(config_raw.get("frontend_live_bundle", {}).get("min_val_windows", 126)),
    )

    llm_refiner = None
    if llm_enabled:
        llm_cfg = _resolve_llm_config(config_raw, llm_model=llm_model, llm_api_key=llm_api_key, llm_base_url=llm_base_url)
        config_raw = copy.deepcopy(config_raw)
        config_raw["llm"] = llm_cfg
        llm_refiner = LLMRefiner(
            config=config_raw["llm"],
            cache_dir=run_dir / "llm" / "cache",
            log_dir=run_dir / "llm" / "logs",
        )

    origin_indices = _select_origin_indices(
        panel=panel,
        pred_len=pred_len,
        history_years=archive_cfg.history_years,
        origin_step=archive_cfg.origin_step,
        seq_len=seq_len,
        min_train_windows=archive_cfg.min_train_windows,
        min_val_windows=archive_cfg.min_val_windows,
    )
    if max_origins is not None:
        origin_indices = origin_indices[-int(max_origins):]

    logger.info("Generating contamination-safe archive for %d origins.", len(origin_indices))
    archive_results: list[ForecastResult] = []
    archive_progress: list[dict[str, Any]] = []
    for pos, origin_idx in enumerate(origin_indices, start=1):
        origin_date = str(pd.Timestamp(panel["date"].iloc[origin_idx]).date())
        logger.info("[%d/%d] Origin %s", pos, len(origin_indices), origin_date)
        origin_results = _run_origin_forecast(
            panel=panel,
            origin_idx=origin_idx,
            config_raw=config_raw,
            llm_refiner=llm_refiner,
        )
        archive_results.extend(origin_results)
        for result in origin_results:
            if result.actual_prices is None:
                continue
            mse = float(np.mean((np.asarray(result.forecast_prices) - np.asarray(result.actual_prices)) ** 2))
            archive_progress.append(
                {
                    "origin_date": result.origin_date,
                    "model": result.model_name,
                    "display_name": result.display_name,
                    "path_mse": mse,
                }
            )
        if archive_results:
            partial_df = _build_archive_rows(archive_results)
            partial_df.to_csv(run_dir / "predictions" / "forecast_archive_partial.csv", index=False)
        if archive_progress:
            pd.DataFrame(archive_progress).to_csv(run_dir / "results" / "archive_origin_metrics_partial.csv", index=False)

    current_trained = train_tsm_on_prefix(prefix_panel=panel.copy(), config_raw=config_raw)
    current_tsm, current_blend = _predict_current_forecast(
        trained=current_trained,
        prefix_panel=panel.copy(),
        config_raw=config_raw,
        llm_refiner=llm_refiner,
    )

    actual_history = _collect_actual_history(panel, archive_cfg.actual_history_days)
    instrument_name = str(config_raw.get("target", {}).get("instrument", "EUA_FUTURES"))
    instrument_display = _instrument_display_name(instrument_name)

    current_price_payload = {
        "artifact_version": 1,
        "exported_at_utc": _iso_utc_now(),
        "run_id": run_dir.name,
        "instrument": instrument_name,
        "display_name": instrument_display,
        "price": float(panel["y"].iloc[-1]),
        "as_of_date": str(pd.Timestamp(panel["date"].iloc[-1]).date()),
        "source_frequency": "daily_close",
        "delayed": True,
    }

    archive_df = _build_archive_rows(archive_results)
    archive_summary = _summarize_archive(archive_df)
    archive_json = _build_archive_json(archive_results)

    current_models = [
        {
            "model": current_tsm.model_name,
            "display_name": current_tsm.display_name,
            "forecast": [
                {
                    "horizon": int(i + 1),
                    "target_date": str(current_tsm.target_dates[i]),
                    "predicted_price": float(current_tsm.forecast_prices[i]),
                }
                for i in range(len(current_tsm.target_dates))
            ],
        }
    ]
    if current_blend is not None:
        current_models.append(
            {
                "model": current_blend.model_name,
                "display_name": current_blend.display_name,
                "forecast": [
                    {
                        "horizon": int(i + 1),
                        "target_date": str(current_blend.target_dates[i]),
                        "predicted_price": float(current_blend.forecast_prices[i]),
                    }
                    for i in range(len(current_blend.target_dates))
                ],
            }
        )

    current_forecast_payload = {
        "artifact_version": 1,
        "exported_at_utc": _iso_utc_now(),
        "run_id": run_dir.name,
        "forecast_kind": "live_forward_daily_close",
        "origin_date": current_tsm.origin_date,
        "data_cutoff_date": str(pd.Timestamp(panel["date"].iloc[-1]).date()),
        "last_observed_price": float(current_tsm.last_observed_price),
        "models": current_models,
    }
    actuals_recent_payload = {
        "artifact_version": 1,
        "exported_at_utc": _iso_utc_now(),
        "run_id": run_dir.name,
        "data_cutoff_date": str(pd.Timestamp(panel["date"].iloc[-1]).date()),
        "history": actual_history,
    }
    archive_start_date = str(pd.Timestamp(panel["date"].iloc[origin_indices[0]]).date())
    archive_end_date = str(pd.Timestamp(panel["date"].iloc[origin_indices[-1]]).date())

    status_payload = {
        "artifact_version": 1,
        "exported_at_utc": _iso_utc_now(),
        "run_id": run_dir.name,
        "contamination_safe": True,
        "target_mode": target_mode,
        "data_cutoff_date": str(pd.Timestamp(panel["date"].iloc[-1]).date()),
        "archive_start_date": archive_start_date,
        "archive_end_date": archive_end_date,
        "requested_history_years": archive_cfg.history_years,
        "archive_origin_step": archive_cfg.origin_step,
        "actual_history_days": archive_cfg.actual_history_days,
        "n_archive_origins": int(len(origin_indices)),
        "models": [model["model"] for model in current_models],
        "daily_ticker": {
            "frequency": "daily_close",
            "delayed": True,
            "as_of_date": str(pd.Timestamp(panel["date"].iloc[-1]).date()),
        },
    }
    manifest_payload = {
        "artifact_version": 1,
        "exported_at_utc": _iso_utc_now(),
        "run_id": run_dir.name,
        "source_run_dir": str(run_dir),
        "files": [
            "current_price.json",
            "actuals_recent.json",
            "current_forecast.json",
            "forecast_archive.csv",
            "forecast_archive.json",
            "model_summary.json",
            "status.json",
            "manifest.json",
        ],
    }

    archive_df.to_csv(run_dir / "predictions" / "forecast_archive.csv", index=False)
    archive_df.to_csv(output_dir / "forecast_archive.csv", index=False)
    pd.DataFrame(archive_progress).to_csv(run_dir / "results" / "archive_origin_metrics.csv", index=False)
    pd.DataFrame(archive_summary.get("models", [])).to_csv(run_dir / "results" / "archive_model_summary.csv", index=False)

    _write_json(output_dir / "current_price.json", current_price_payload)
    _write_json(output_dir / "actuals_recent.json", actuals_recent_payload)
    _write_json(output_dir / "current_forecast.json", current_forecast_payload)
    _write_json(output_dir / "forecast_archive.json", {
        "artifact_version": 1,
        "exported_at_utc": _iso_utc_now(),
        "run_id": run_dir.name,
        **archive_json,
    })
    _write_json(output_dir / "model_summary.json", {
        "artifact_version": 1,
        "exported_at_utc": _iso_utc_now(),
        "run_id": run_dir.name,
        "archive_start_date": status_payload["archive_start_date"],
        "archive_end_date": status_payload["archive_end_date"],
        "n_archive_origins": int(len(origin_indices)),
        **archive_summary,
    })
    _write_json(output_dir / "status.json", status_payload)
    _write_json(output_dir / "manifest.json", manifest_payload)

    if publish_dir is not None:
        publish_dir = Path(publish_dir).resolve()
        _copy_outputs(output_dir, publish_dir, manifest_payload["files"])

    return {
        "run_dir": run_dir,
        "output_dir": output_dir,
        "current_price": current_price_payload,
        "archive_summary": archive_summary,
        "status": status_payload,
    }
