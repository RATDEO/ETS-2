#!/usr/bin/env python3
"""Run overlap holdouts to fill the missing 30-business-day tails in the live archive."""

from __future__ import annotations

import shutil
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.live_backend.common import (
    archive_row,
    future_realized_slice,
    next_business_target_dates,
    normalize_archive_frame,
    panel_with_prices,
    read_run_metadata,
)
from src.run_experiment import run_experiment
from src.uk_ets_live_deployment import export_public_site_files, load_archive, merge_archive_rows, save_archive


CONFIG_PATH = "uk_ets/config/uk_ets_llm_27b_live_reference.yaml"
MODEL_VERSION = "uk_ets_llm_27b_live_reference"
METHOD_NAME = "TSM+LLM-COT-RF-HDELTA"
ACTIVE_ARCHIVE = PROJECT_ROOT / "uk_ets" / "live_forecast" / "archive" / "forecast_history.csv"
ACTIVE_EXPORT_DIR = PROJECT_ROOT / "uk_ets" / "live_forecast" / "exports"
PORTFOLIO_DIR = Path("/Users/davidwilkinson/Desktop/portfolio/app/public/data/ukets")
DATA_DIR = PROJECT_ROOT / "uk_ets" / "Data_auto_uk"


@dataclass(frozen=True)
class OverlapWindow:
    name: str
    train_end: str
    val_end: str
    test_end: str


WINDOWS = [
    OverlapWindow("w4_gap", "2021-10-16", "2023-05-10", "2023-08-01"),
    OverlapWindow("w3_gap", "2022-06-20", "2024-01-12", "2024-04-04"),
    OverlapWindow("w2_gap", "2023-02-22", "2024-09-16", "2024-12-06"),
    OverlapWindow("w1_gap", "2023-10-27", "2025-05-20", "2025-08-11"),
]


def _copy_export_tree(src: Path, dst: Path) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    for child in src.iterdir():
        target = dst / child.name
        if child.is_dir():
            if target.exists():
                shutil.rmtree(target)
            shutil.copytree(child, target)
        else:
            shutil.copy2(child, target)


def _build_seed_rows_from_run(run_dir: Path, *, model_version: str) -> pd.DataFrame:
    metadata = read_run_metadata(run_dir)
    panel = panel_with_prices(run_dir / "data" / "panel.parquet")
    prediction_path = run_dir / "predictions" / f"{METHOD_NAME}_pred_test_subset.npz"
    if not prediction_path.exists():
        raise FileNotFoundError(f"Missing prediction subset: {prediction_path}")

    with np.load(prediction_path, allow_pickle=True) as npz:
        origin_dates = pd.to_datetime(np.asarray(npz["dates"])).normalize()
        y_true = np.asarray(npz["y_true"], dtype=float)
        yhat = np.asarray(npz["yhat"], dtype=float)
        base_pred = np.asarray(npz["base_pred"], dtype=float)

    rows: list[dict[str, object]] = []
    generated_at = metadata["timestamp"] or datetime.now(timezone.utc).isoformat()
    source_run_id = run_dir.name
    for row_index, origin_date in enumerate(origin_dates):
        origin_ts = pd.Timestamp(origin_date)
        origin_str = str(origin_ts.date())
        try:
            target_dates = [
                str(pd.Timestamp(ts).date())
                for ts in future_realized_slice(panel, origin_ts, yhat.shape[1])["date"].tolist()
            ]
        except ValueError:
            target_dates = next_business_target_dates(origin_ts, yhat.shape[1])

        latest_price = float(panel.loc[panel["date"] == origin_ts, "y"].iloc[0])
        row_run_id = f"{source_run_id}:overlap:{origin_str}"
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
                    record_source="historical_overlap_backfill",
                    source_run_id=source_run_id,
                )
            )
    return normalize_archive_frame(pd.DataFrame(rows))


def _run_overlap_window(window: OverlapWindow) -> Path:
    overrides = {
        "split": {
            "train_end": window.train_end,
            "val_end": window.val_end,
            "test_end": window.test_end,
        },
        "target": {
            "mode": "returns",
            "max_date": window.test_end,
        },
        "output": {
            "generate_paper": False,
            "write_project_paper": False,
        },
    }
    run_dir = run_experiment(
        config_path=CONFIG_PATH,
        overrides=overrides,
        data_dir=str(DATA_DIR),
    )
    return Path(run_dir).resolve()


def main() -> int:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    archive_backup = ACTIVE_ARCHIVE.parent / "backups" / f"forecast_history_{stamp}_before_overlap_fill.csv"
    export_backup = ACTIVE_EXPORT_DIR.parent / "backups" / f"exports_{stamp}_before_overlap_fill"
    portfolio_backup = PORTFOLIO_DIR.parent / "backups" / f"ukets_{stamp}_before_overlap_fill"
    temp_archive = ACTIVE_ARCHIVE.with_name(f"forecast_history_overlap_fill_{stamp}.csv")
    temp_export_dir = ACTIVE_EXPORT_DIR.parent / f"exports_overlap_fill_{stamp}"

    archive_backup.parent.mkdir(parents=True, exist_ok=True)
    export_backup.parent.mkdir(parents=True, exist_ok=True)
    portfolio_backup.parent.mkdir(parents=True, exist_ok=True)

    run_dirs: list[Path] = []
    archive_df = load_archive(ACTIVE_ARCHIVE)
    for window in WINDOWS:
        run_dir = _run_overlap_window(window)
        run_dirs.append(run_dir)
        seed_rows = _build_seed_rows_from_run(run_dir, model_version=MODEL_VERSION)
        archive_df = merge_archive_rows(archive_df, seed_rows)
        print(f"{window.name}: {run_dir}")

    save_archive(archive_df, temp_archive)
    export_result = export_public_site_files(archive_df, temp_export_dir)

    shutil.copy2(ACTIVE_ARCHIVE, archive_backup)
    shutil.copytree(ACTIVE_EXPORT_DIR, export_backup)
    shutil.copytree(PORTFOLIO_DIR, portfolio_backup)

    shutil.copy2(temp_archive, ACTIVE_ARCHIVE)
    _copy_export_tree(temp_export_dir, ACTIVE_EXPORT_DIR)
    _copy_export_tree(temp_export_dir, PORTFOLIO_DIR)

    print(f"Temp archive: {temp_archive}")
    print(f"Temp export dir: {temp_export_dir}")
    print(f"Archive backup: {archive_backup}")
    print(f"Export backup: {export_backup}")
    print(f"Portfolio backup: {portfolio_backup}")
    print(f"Run dirs: {[str(path) for path in run_dirs]}")
    print(f"Archive rows: {export_result['status']['archive_rows']}")
    print(f"Horizon counts: {export_result['status']['horizon_counts']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
