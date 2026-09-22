from __future__ import annotations

import shlex
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .common import (
    DEFAULT_MARKET_NAME,
    DEFAULT_MODEL_VERSION,
    EXPECTED_EXPORT_FILES,
    HEADLINE_HORIZONS,
    archive_row,
    empty_archive_frame,
    future_realized_slice,
    iso_utc_now,
    next_business_target_dates,
    normalize_archive_frame,
    panel_with_prices,
    read_run_metadata,
    write_json,
)


def _load_matching_full_archive_seed(model_version: str) -> pd.DataFrame | None:
    archive_dir = Path(__file__).resolve().parents[2] / "uk_ets" / "live_forecast" / "archive"
    if not archive_dir.exists():
        return None

    best_match: pd.DataFrame | None = None
    best_rows = -1
    requested_model_version = str(model_version or "").strip()

    for path in sorted(archive_dir.glob("forecast_history*.csv")):
        try:
            frame = normalize_archive_frame(pd.read_csv(path))
        except Exception:
            continue
        if frame.empty:
            continue

        if requested_model_version:
            frame = frame[frame["model_version"].astype(str) == requested_model_version].copy()
        if frame.empty or not frame["actual"].notna().any():
            continue

        if len(frame) > best_rows:
            best_match = frame
            best_rows = len(frame)

    return best_match


def load_archive(archive_path: str | Path) -> pd.DataFrame:
    path = Path(archive_path)
    if not path.exists():
        return empty_archive_frame()
    return normalize_archive_frame(pd.read_csv(path))


def save_archive(frame: pd.DataFrame, archive_path: str | Path) -> Path:
    path = Path(archive_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    normalize_archive_frame(frame).to_csv(path, index=False)
    return path


def build_backtest_seed_rows(
    run_dir: str | Path,
    *,
    method_name: str = "TSM+LLM-COT-RF-HDELTA",
    model_version: str = DEFAULT_MODEL_VERSION,
) -> pd.DataFrame:
    matching_archive = _load_matching_full_archive_seed(model_version=model_version)
    if matching_archive is not None:
        return matching_archive

    run_path = Path(run_dir).resolve()
    metadata = read_run_metadata(run_path)
    panel = panel_with_prices(run_path / "data" / "panel.parquet")
    prediction_path = run_path / "predictions" / f"{method_name}_pred_test_subset.npz"
    if not prediction_path.exists():
        raise FileNotFoundError(f"Missing canonical prediction subset: {prediction_path}")

    with np.load(prediction_path, allow_pickle=True) as npz:
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
    generated_at = metadata["timestamp"] or iso_utc_now()
    for row_index, origin_date in enumerate(origin_dates):
        origin_str = str(pd.Timestamp(origin_date).date())
        try:
            target_dates = [
                str(pd.Timestamp(ts).date())
                for ts in future_realized_slice(panel, pd.Timestamp(origin_date), yhat.shape[1])["date"].tolist()
            ]
        except ValueError:
            target_dates = next_business_target_dates(pd.Timestamp(origin_date), yhat.shape[1])

        latest_price = float(panel.loc[panel["date"] == pd.Timestamp(origin_date), "y"].iloc[0])
        source_run_id = str(run_path.name)
        row_run_id = f"{source_run_id}:backfill:{origin_str}"
        for step_index in range(yhat.shape[1]):
            rows.append(
                archive_row(
                    run_id=row_run_id,
                    forecast_made_on=origin_str,
                    latest_observed_date=origin_str,
                    latest_observed_price=latest_price,
                    target_date=target_dates[step_index],
                    step_index=step_index + 1,
                    actual=float(y_true[row_index, step_index]),
                    base_tsm_forecast=float(base_pred[row_index, step_index]),
                    llm_tsm_forecast=float(yhat[row_index, step_index]),
                    model_version=model_version,
                    model_commit=metadata["git_hash"],
                    data_version=origin_str,
                    is_realized=True,
                    generated_at=generated_at,
                    record_source="historical_backfill",
                    source_run_id=source_run_id,
                )
            )
    return normalize_archive_frame(pd.DataFrame(rows))


def _mse_summary(frame: pd.DataFrame, forecast_column: str) -> dict[str, Any] | None:
    if frame.empty:
        return None

    actual = pd.to_numeric(frame["actual"], errors="coerce").to_numpy(dtype=float)
    forecast = pd.to_numeric(frame[forecast_column], errors="coerce").to_numpy(dtype=float)
    squared_error = np.square(forecast - actual)
    mse = float(np.mean(squared_error))
    return {
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "mae": float(np.mean(np.abs(forecast - actual))),
    }


def _build_metrics_payload(
    archive: pd.DataFrame,
    *,
    market_name: str,
    latest_generated_at: str,
    latest_observed_date: str,
    model_version: str,
) -> dict[str, Any]:
    realized = archive[
        archive["is_realized"].fillna(False)
        & archive["actual"].notna()
        & archive["step_index"].notna()
    ].copy()
    realized["step_index"] = pd.to_numeric(realized["step_index"], errors="coerce").astype(int)

    max_horizon = int(realized["step_index"].max()) if not realized.empty else 0
    origin_counts = (
        realized.groupby("forecast_made_on")["step_index"].nunique().sort_index()
        if not realized.empty
        else pd.Series(dtype=int)
    )
    complete_origins = origin_counts[origin_counts == max_horizon].index.tolist() if max_horizon else []
    complete_rows = realized[realized["forecast_made_on"].isin(complete_origins)].copy()

    base_path_summary = _mse_summary(complete_rows, "base_tsm_forecast")
    llm_path_summary = _mse_summary(complete_rows, "llm_tsm_forecast")
    if base_path_summary is None or llm_path_summary is None:
        path_summary: dict[str, Any] | None = None
    else:
        base_mse = float(base_path_summary["mse"])
        llm_mse = float(llm_path_summary["mse"])
        path_summary = {
            "complete_origin_count": int(len(complete_origins)),
            "horizon_count": int(max_horizon),
            "base_tsm_mse": base_mse,
            "llm_tsm_mse": llm_mse,
            "uplift_pct": float(((base_mse - llm_mse) / base_mse) * 100.0) if abs(base_mse) > 1e-12 else 0.0,
        }

    horizons: dict[str, Any] = {}
    for horizon in sorted(realized["step_index"].unique().tolist()):
        horizon_rows = realized[realized["step_index"] == int(horizon)].copy()
        base_summary = _mse_summary(horizon_rows, "base_tsm_forecast")
        llm_summary = _mse_summary(horizon_rows, "llm_tsm_forecast")
        if base_summary is None or llm_summary is None:
            continue
        base_mse = float(base_summary["mse"])
        llm_mse = float(llm_summary["mse"])
        horizons[f"{int(horizon)}d"] = {
            "settled_count": int(len(horizon_rows)),
            "base_tsm_mse": base_mse,
            "llm_tsm_mse": llm_mse,
            "uplift_pct": float(((base_mse - llm_mse) / base_mse) * 100.0) if abs(base_mse) > 1e-12 else 0.0,
        }

    return {
        "market": market_name,
        "generated_at": latest_generated_at,
        "latest_observed_date": latest_observed_date,
        "model_version": model_version,
        "settled_origin_count": int(realized["forecast_made_on"].nunique()),
        "settled_row_count": int(len(realized)),
        "path": path_summary,
        "horizons": horizons,
    }


def merge_archive_rows(existing: pd.DataFrame, incoming: pd.DataFrame) -> pd.DataFrame:
    if incoming is None or incoming.empty:
        return normalize_archive_frame(existing)
    if existing is None or existing.empty:
        return normalize_archive_frame(incoming)

    combined = pd.concat(
        [normalize_archive_frame(existing), normalize_archive_frame(incoming)],
        ignore_index=True,
    )
    combined = combined.drop_duplicates(subset=["latest_observed_date", "step_index"], keep="last")
    return normalize_archive_frame(combined)


def backfill_archive_actuals(archive_df: pd.DataFrame, panel: pd.DataFrame) -> pd.DataFrame:
    out = normalize_archive_frame(archive_df)
    if out.empty:
        return out

    work_panel = panel.copy()
    work_panel["date"] = pd.to_datetime(work_panel["date"]).dt.normalize()
    work_panel["y"] = pd.to_numeric(work_panel["y"], errors="coerce")
    actual_map = {
        str(pd.Timestamp(row.date).date()): float(row.y)
        for row in work_panel.dropna(subset=["date", "y"]).itertuples(index=False)
    }
    for row_index in out.index[out["actual"].isna()]:
        actual = actual_map.get(str(out.at[row_index, "target_date"]))
        if actual is None:
            continue
        out.at[row_index, "actual"] = float(actual)
        out.at[row_index, "is_realized"] = True
    return normalize_archive_frame(out)


def export_public_site_files(
    archive_df: pd.DataFrame,
    export_dir: str | Path,
    *,
    market_name: str = DEFAULT_MARKET_NAME,
) -> dict[str, Any]:
    export_path = Path(export_dir).resolve()
    export_path.mkdir(parents=True, exist_ok=True)

    archive = normalize_archive_frame(archive_df)
    if archive.empty:
        raise ValueError("Archive is empty; no live forecast rows available.")

    latest_date = archive["latest_observed_date"].dropna().astype(str).max()
    live_rows = archive[archive["latest_observed_date"].astype(str) == str(latest_date)].copy()
    if live_rows.empty:
        raise ValueError("Could not resolve latest live forecast rows from archive.")
    live_rows = live_rows.sort_values("step_index").reset_index(drop=True)

    latest_generated_at = str(live_rows["generated_at"].dropna().iloc[-1])
    latest_observed_date = str(live_rows["latest_observed_date"].iloc[0])
    latest_observed_price = float(live_rows["latest_observed_price"].iloc[0])
    model_version = str(live_rows["model_version"].iloc[0] or DEFAULT_MODEL_VERSION)

    forecasts: dict[str, Any] = {}
    for horizon in HEADLINE_HORIZONS:
        row = live_rows[live_rows["step_index"] == int(horizon)]
        if row.empty:
            continue
        item = row.iloc[0]
        llm_value = float(item["llm_tsm_forecast"])
        forecasts[f"{horizon}d"] = {
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
        "forecasts": forecasts,
    }
    write_json(export_path / "latest.json", latest_payload)

    horizon_counts: dict[str, int] = {}
    for horizon in HEADLINE_HORIZONS:
        realized = archive[
            (archive["step_index"] == int(horizon))
            & archive["is_realized"].fillna(False)
            & archive["actual"].notna()
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
        write_json(
            export_path / f"horizon_{horizon}d.json",
            {
                "market": market_name,
                "horizon_days": int(horizon),
                "generated_at": latest_generated_at,
                "model_version": model_version,
                "series": series,
            },
        )
        horizon_counts[f"h{horizon}"] = len(series)

    metrics_payload = _build_metrics_payload(
        archive,
        market_name=market_name,
        latest_generated_at=latest_generated_at,
        latest_observed_date=latest_observed_date,
        model_version=model_version,
    )
    write_json(export_path / "metrics.json", metrics_payload)

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
        "archive_rows": int(len(archive)),
        "realized_rows": int(archive["actual"].notna().sum()),
        "current_live_rows": int(len(live_rows)),
        "horizon_counts": horizon_counts,
    }
    write_json(export_path / "manifest.json", manifest_payload)
    write_json(export_path / "status.json", status_payload)

    return {
        "latest": latest_payload,
        "metrics": metrics_payload,
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
    export_path = Path(export_dir).resolve()
    expected = list(expected_files or EXPECTED_EXPORT_FILES)
    missing = [name for name in expected if not (export_path / name).exists()]
    if missing:
        raise FileNotFoundError(f"Refusing to deploy; missing export files: {missing}")

    run_label = str(run_label or datetime.now().strftime("%Y%m%d_%H%M%S"))
    remote_staging_path = f"{remote_staging_dir.rstrip('/')}/{run_label}"

    rsync_command = ["rsync", "-az", "--delete"]
    if dry_run:
        rsync_command.append("--dry-run")
    rsync_command.extend([f"{export_path}/", f"{remote_host}:{remote_staging_path}/"])
    subprocess.run(rsync_command, check=True)

    if dry_run:
        return

    checks = " && ".join(
        f"test -f {shlex.quote(remote_staging_path.rstrip('/') + '/' + name)}"
        for name in expected
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
    export_path = Path(export_dir).resolve()
    staging_path = f"{remote_staging_dir.rstrip('/')}/{run_label}"
    return {
        "rsync": (
            f"rsync -az --delete {shlex.quote(str(export_path) + '/')} "
            f"{shlex.quote(remote_host + ':' + staging_path + '/')}"
        ),
        "promote": (
            f"ssh {shlex.quote(remote_host)} "
            f"{shlex.quote(f'mkdir -p {remote_live_dir} && rsync -az --delete {staging_path}/ {remote_live_dir}/')}"
        ),
        "staging_path": staging_path,
    }
