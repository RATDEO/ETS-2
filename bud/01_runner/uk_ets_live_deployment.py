"""Readable compatibility surface for the live UK ETS backend."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import pandas as pd

from .live_backend import archive as _archive
from .live_backend import forecast as _forecast
from .live_backend import online_memory as _online_memory
from .live_backend.common import DEFAULT_MARKET_NAME, DEFAULT_METHOD_NAME, DEFAULT_MODEL_VERSION

build_backtest_seed_rows = _archive.build_backtest_seed_rows
load_archive = _archive.load_archive
save_archive = _archive.save_archive
merge_archive_rows = _archive.merge_archive_rows
backfill_archive_actuals = _archive.backfill_archive_actuals
export_public_site_files = _archive.export_public_site_files
deploy_exports_via_rsync = _archive.deploy_exports_via_rsync
build_deploy_preview = _archive.build_deploy_preview

_fit_live_learned_gate_bundle = _online_memory._fit_live_learned_gate_bundle
_simulate_live_online_memory_history = _online_memory._simulate_live_online_memory_history
_build_current_online_memory_context = _online_memory._build_current_online_memory_context

_load_live_feature_panel = _forecast._load_live_feature_panel
_build_live_rows_from_feature_panel = _forecast._build_live_rows_from_feature_panel
_config_with_previous_llm_system = _forecast._config_with_previous_llm_system


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
    row_config = (
        _config_with_previous_llm_system(config_raw)
        if use_previous_llm_system
        else copy.deepcopy(config_raw)
    )
    panel = _load_live_feature_panel(
        config_raw=row_config,
        data_dir=data_dir,
        run_dir=Path(run_dir).resolve(),
    )
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
        return _archive.empty_archive_frame(), panel_prices

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
    cutoff = (
        pd.Timestamp(start_after_date).normalize()
        if start_after_date is not None
        else pd.Timestamp.min.normalize()
    )
    last_origin_index = len(panel) - 1 if include_latest_origin else len(panel) - 2
    if last_origin_index < 0:
        return _archive.empty_archive_frame(), panel_prices

    origin_indices = [
        index
        for index in range(max(0, earliest_origin_index), last_origin_index + 1)
        if pd.Timestamp(panel["date"].iloc[index]).normalize() > cutoff
    ]
    if not origin_indices:
        return _archive.empty_archive_frame(), panel_prices

    row_config = (
        _config_with_previous_llm_system(config_raw)
        if use_previous_llm_system
        else copy.deepcopy(config_raw)
    )
    row_frames: list[pd.DataFrame] = []
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
        return _archive.empty_archive_frame(), panel_prices
    return _archive.normalize_archive_frame(pd.concat(row_frames, ignore_index=True)), panel_prices


__all__ = [
    "DEFAULT_MARKET_NAME",
    "DEFAULT_METHOD_NAME",
    "DEFAULT_MODEL_VERSION",
    "build_backtest_seed_rows",
    "build_current_live_rows",
    "build_recent_live_history_rows",
    "load_archive",
    "save_archive",
    "merge_archive_rows",
    "backfill_archive_actuals",
    "export_public_site_files",
    "deploy_exports_via_rsync",
    "build_deploy_preview",
    "_load_live_feature_panel",
    "_build_live_rows_from_feature_panel",
    "_build_current_online_memory_context",
    "_fit_live_learned_gate_bundle",
    "_simulate_live_online_memory_history",
]
