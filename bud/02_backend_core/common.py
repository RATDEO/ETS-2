from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

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
    "metrics.json",
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


def iso_utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def empty_archive_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=ARCHIVE_COLUMNS)


def normalize_archive_frame(frame: pd.DataFrame | None) -> pd.DataFrame:
    if frame is None or frame.empty:
        return empty_archive_frame()

    out = frame.copy()
    for column in ARCHIVE_COLUMNS:
        if column not in out.columns:
            out[column] = np.nan

    for column in ["forecast_made_on", "latest_observed_date", "target_date", "generated_at"]:
        out[column] = out[column].astype("string")

    for column in ["run_id", "model_version", "model_commit", "data_version", "record_source", "source_run_id"]:
        out[column] = out[column].fillna("").astype("string")

    for column in ["latest_observed_price", "actual", "base_tsm_forecast", "llm_tsm_forecast"]:
        out[column] = pd.to_numeric(out[column], errors="coerce")

    out["step_index"] = pd.to_numeric(out["step_index"], errors="coerce").astype("Int64")
    out["is_realized"] = out["is_realized"].fillna(False).astype(bool)

    extra_columns = [column for column in out.columns if column not in ARCHIVE_COLUMNS]
    out = out[ARCHIVE_COLUMNS + extra_columns]
    return (
        out.sort_values(["latest_observed_date", "step_index", "generated_at"], kind="stable")
        .reset_index(drop=True)
    )


def read_run_metadata(run_dir: str | Path) -> dict[str, Any]:
    run_path = Path(run_dir).resolve()
    config_path = run_path / "config_resolved.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing resolved config: {config_path}")

    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    config_raw = {
        key: value
        for key, value in raw.items()
        if key not in {"run_id", "git_hash", "timestamp"}
    }
    return {
        "run_id": str(raw.get("run_id") or run_path.name),
        "git_hash": str(raw.get("git_hash") or ""),
        "timestamp": str(raw.get("timestamp") or ""),
        "config_raw": config_raw,
    }


def panel_with_prices(panel_path: str | Path) -> pd.DataFrame:
    panel = pd.read_parquet(panel_path, columns=["date", "y"]).copy()
    panel["date"] = pd.to_datetime(panel["date"]).dt.normalize()
    panel["y"] = pd.to_numeric(panel["y"], errors="coerce")
    return (
        panel.dropna(subset=["date", "y"])
        .sort_values("date")
        .drop_duplicates("date")
        .reset_index(drop=True)
    )


def future_realized_slice(
    panel: pd.DataFrame,
    origin_date: pd.Timestamp,
    pred_len: int,
) -> pd.DataFrame:
    date_to_index = {pd.Timestamp(ts).normalize(): index for index, ts in enumerate(panel["date"])}
    origin_index = date_to_index.get(pd.Timestamp(origin_date).normalize())
    if origin_index is None:
        raise ValueError(f"Origin date {origin_date.date()} not found in panel.")

    future = panel.iloc[origin_index + 1 : origin_index + 1 + int(pred_len)].copy()
    if len(future) < int(pred_len):
        raise ValueError(
            f"Origin {origin_date.date()} only has {len(future)} realized rows, need {pred_len}."
        )
    return future.reset_index(drop=True)


def next_business_target_dates(origin_date: pd.Timestamp, pred_len: int) -> list[str]:
    return [
        str(ts.date())
        for ts in pd.bdate_range(pd.Timestamp(origin_date).normalize() + pd.Timedelta(days=1), periods=int(pred_len))
    ]


def archive_row(
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
