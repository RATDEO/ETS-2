#!/usr/bin/env python3
"""Extend the active 27B live archive backward with completed 27B window runs."""

from __future__ import annotations

import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.uk_ets_live_deployment import (
    build_backtest_seed_rows,
    export_public_site_files,
    load_archive,
    merge_archive_rows,
    save_archive,
)
ACTIVE_ARCHIVE = PROJECT_ROOT / "uk_ets" / "live_forecast" / "archive" / "forecast_history.csv"
ACTIVE_EXPORT_DIR = PROJECT_ROOT / "uk_ets" / "live_forecast" / "exports"
PORTFOLIO_DIR = Path("/Users/davidwilkinson/Desktop/portfolio/app/public/data/ukets")
MODEL_VERSION = "uk_ets_llm_27b_live_reference"
WINDOW_RUNS = [
    "20260331_214230_22bef9",  # W4
    "20260331_205229_5c7de8",  # W3
    "20260331_200333_fac9f0",  # W2
    "20260331_192217_2f8fbb",  # W1
]


def _backup_tree(src: Path, dst: Path) -> None:
    if dst.exists():
        raise FileExistsError(f"Backup already exists: {dst}")
    shutil.copytree(src, dst)


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


def main() -> int:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    temp_archive = ACTIVE_ARCHIVE.with_name(f"forecast_history_27b_extended_{stamp}.csv")
    temp_export_dir = ACTIVE_EXPORT_DIR.parent / f"exports_27b_extended_{stamp}"

    archive_df = load_archive(ACTIVE_ARCHIVE)
    for run_id in WINDOW_RUNS:
        seed_rows = build_backtest_seed_rows(
            PROJECT_ROOT / "runs" / run_id,
            model_version=MODEL_VERSION,
        )
        archive_df = merge_archive_rows(archive_df, seed_rows)

    save_archive(archive_df, temp_archive)
    export_result = export_public_site_files(archive_df, temp_export_dir)

    archive_backup = ACTIVE_ARCHIVE.parent / "backups" / f"forecast_history_{stamp}_before_27b_extension.csv"
    export_backup = ACTIVE_EXPORT_DIR.parent / "backups" / f"exports_{stamp}_before_27b_extension"
    portfolio_backup = PORTFOLIO_DIR.parent / "backups" / f"ukets_{stamp}_before_27b_extension"
    archive_backup.parent.mkdir(parents=True, exist_ok=True)
    export_backup.parent.mkdir(parents=True, exist_ok=True)
    portfolio_backup.parent.mkdir(parents=True, exist_ok=True)

    shutil.copy2(ACTIVE_ARCHIVE, archive_backup)
    _backup_tree(ACTIVE_EXPORT_DIR, export_backup)
    _backup_tree(PORTFOLIO_DIR, portfolio_backup)

    shutil.copy2(temp_archive, ACTIVE_ARCHIVE)
    _copy_export_tree(temp_export_dir, ACTIVE_EXPORT_DIR)
    _copy_export_tree(temp_export_dir, PORTFOLIO_DIR)

    print(f"Temp archive: {temp_archive}")
    print(f"Temp export dir: {temp_export_dir}")
    print(f"Archive backup: {archive_backup}")
    print(f"Export backup: {export_backup}")
    print(f"Portfolio backup: {portfolio_backup}")
    print(f"Latest observed date: {export_result['latest']['latest_observed_date']}")
    print(f"Model version: {export_result['latest']['model_version']}")
    print(f"Archive rows: {export_result['status']['archive_rows']}")
    print(f"Horizon counts: {export_result['status']['horizon_counts']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
