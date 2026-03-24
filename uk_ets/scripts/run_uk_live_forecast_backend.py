#!/usr/bin/env python3
"""Run the UK ETS live-forecast backend flow for portfolio exports."""

from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config
from src.run_experiment import run_experiment
from src.uk_ets_live_deployment import (
    DEFAULT_MODEL_VERSION,
    build_backtest_seed_rows,
    build_current_live_rows,
    build_recent_live_history_rows,
    deploy_exports_via_rsync,
    export_public_site_files,
    load_archive,
    merge_archive_rows,
    backfill_archive_actuals,
    save_archive,
)


def _resolve(path_value: str) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path.resolve()
    return (PROJECT_ROOT / path).resolve()


def _make_lightweight_live_run_dir() -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir = (PROJECT_ROOT / "runs" / f"live_backend_{stamp}").resolve()
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "data").mkdir(parents=True, exist_ok=True)
    return run_dir


def _run_bootstrap(
    *,
    output_dir: Path,
    start_date: str,
    end_date: str,
    include_news: bool,
    news_start: str,
    news_end: str,
    ice_report_recaptcha_token: str | None,
    require_actual_auctions: bool,
) -> None:
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "uk_ets" / "scripts" / "bootstrap_uk_ets_data_sources.py"),
        "--output-dir",
        str(output_dir),
        "--start-date",
        start_date,
        "--end-date",
        end_date,
    ]
    if ice_report_recaptcha_token:
        cmd.extend(["--ice-report-recaptcha-token", ice_report_recaptcha_token])
    if require_actual_auctions:
        cmd.append("--require-actual-auctions")
    if include_news:
        cmd.extend(
            [
                "--include-news",
                "--news-start",
                news_start,
                "--news-end",
                news_end,
            ]
        )
    subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)


def main() -> int:
    today = date.today().isoformat()
    parser = argparse.ArgumentParser(description="Run the UK ETS live-forecast backend flow.")
    parser.add_argument("--mode", choices=["main", "repair"], default="main")
    parser.add_argument("--config", default="uk_ets/config/uk_ets_llm_4b_canonical_default.yaml")
    parser.add_argument("--data-dir", default="uk_ets/Data_auto_uk")
    parser.add_argument("--archive-path", default="uk_ets/live_forecast/archive/forecast_history.csv")
    parser.add_argument("--export-dir", default="uk_ets/live_forecast/exports")
    parser.add_argument("--run-dir", default=None, help="Reuse an existing experiment run directory.")
    parser.add_argument("--model-version", default=DEFAULT_MODEL_VERSION)
    parser.add_argument("--target-mode", choices=["price", "returns"], default="returns")
    parser.add_argument("--start-date", default="2021-05-19")
    parser.add_argument("--end-date", default=today)
    parser.add_argument("--skip-bootstrap", action="store_true")
    parser.add_argument("--skip-experiment", action="store_true")
    parser.add_argument("--include-news", action="store_true")
    parser.add_argument("--news-start", default="2025-01-01")
    parser.add_argument("--news-end", default=today)
    parser.add_argument("--ice-report-recaptcha-token", default="")
    parser.add_argument("--require-actual-auctions", action="store_true")
    parser.add_argument("--remote-host", default=None)
    parser.add_argument("--remote-staging-dir", default=None)
    parser.add_argument("--remote-live-dir", default=None)
    parser.add_argument("--deploy-dry-run", action="store_true")
    args = parser.parse_args()

    data_dir = _resolve(args.data_dir)
    archive_path = _resolve(args.archive_path)
    export_dir = _resolve(args.export_dir)
    config_path = _resolve(args.config)

    if not args.skip_bootstrap:
        data_dir.mkdir(parents=True, exist_ok=True)
        _run_bootstrap(
            output_dir=data_dir,
            start_date=args.start_date,
            end_date=args.end_date,
            include_news=args.include_news,
            news_start=args.news_start,
            news_end=args.news_end,
            ice_report_recaptcha_token=(args.ice_report_recaptcha_token or "").strip() or None,
            require_actual_auctions=bool(args.require_actual_auctions),
        )

    if args.run_dir and not args.skip_experiment:
        raise SystemExit("Use either --run-dir or a fresh experiment run, not both.")
    if not args.run_dir and args.skip_experiment:
        raise SystemExit("--skip-experiment requires --run-dir so the backend can seed from a canonical run.")

    seed_run_dir: Path
    live_run_dir: Path
    if args.run_dir:
        seed_run_dir = _resolve(args.run_dir)
        live_run_dir = _make_lightweight_live_run_dir()
    else:
        overrides = {
            "target": {"mode": args.target_mode},
            "output": {
                "generate_paper": False,
                "write_project_paper": False,
            },
        }
        seed_run_dir = Path(
            run_experiment(
                config_path=str(config_path),
                overrides=overrides,
                data_dir=str(data_dir),
            )
        ).resolve()
        live_run_dir = seed_run_dir

    config_raw = load_config(str(config_path), overrides={"target": {"mode": args.target_mode}}).raw

    archive_df = load_archive(archive_path)
    seed_rows = build_backtest_seed_rows(
        seed_run_dir,
        model_version=args.model_version,
    )
    archive_df = merge_archive_rows(archive_df, seed_rows)
    seed_max_origin = (
        pd.to_datetime(seed_rows["latest_observed_date"]).max()
        if not seed_rows.empty
        else None
    )
    recent_history_rows, recent_history_panel = build_recent_live_history_rows(
        config_raw=config_raw,
        data_dir=data_dir,
        run_dir=live_run_dir,
        start_after_date=seed_max_origin,
        artifact_run_dir=seed_run_dir,
        model_version=args.model_version,
        record_source="historical_live_backfill",
    )
    archive_df = merge_archive_rows(archive_df, recent_history_rows)
    current_rows, live_panel, live_meta = build_current_live_rows(
        config_raw=config_raw,
        data_dir=data_dir,
        run_dir=live_run_dir,
        artifact_run_dir=seed_run_dir,
        model_version=args.model_version,
        record_source=f"{args.mode}_live_run",
    )
    archive_df = merge_archive_rows(archive_df, current_rows)
    backfill_panel = pd.concat([recent_history_panel, live_panel], ignore_index=True)
    backfill_panel = (
        backfill_panel.sort_values("date")
        .drop_duplicates(subset=["date"], keep="last")
        .reset_index(drop=True)
    )
    archive_df = backfill_archive_actuals(archive_df, backfill_panel)
    save_archive(archive_df, archive_path)

    export_result = export_public_site_files(
        archive_df,
        export_dir,
    )

    if args.remote_host or args.remote_staging_dir or args.remote_live_dir:
        if not (args.remote_host and args.remote_staging_dir and args.remote_live_dir):
            raise SystemExit(
                "Remote deployment requires --remote-host, --remote-staging-dir, and --remote-live-dir together."
            )
        deploy_exports_via_rsync(
            export_dir=export_dir,
            remote_host=str(args.remote_host),
            remote_staging_dir=str(args.remote_staging_dir),
            remote_live_dir=str(args.remote_live_dir),
            run_label=live_run_dir.name,
            dry_run=bool(args.deploy_dry_run),
        )

    print(f"Mode: {args.mode}")
    print(f"Canonical seed run dir: {seed_run_dir}")
    if live_run_dir != seed_run_dir:
        print(f"Live run dir: {live_run_dir}")
    print(f"Archive CSV: {archive_path}")
    print(f"Export dir: {export_dir}")
    print(f"Latest observed date: {export_result['latest']['latest_observed_date']}")
    print(
        "Latest 20d LLM forecast: "
        f"{export_result['latest']['forecasts'].get('20d', {}).get('llm_tsm_value', 'n/a')}"
    )
    print(f"Live LLM method: {live_meta['llm_method']}")
    if args.remote_host:
        print(
            f"Deploy target: {args.remote_host}:{args.remote_live_dir} "
            f"(dry_run={bool(args.deploy_dry_run)})"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
