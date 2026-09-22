#!/usr/bin/env python3
"""Run one UK ETS automated pipeline pass: data refresh + experiment."""

from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import date
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.run_experiment import run_experiment


def _resolve(path_value: str) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path.resolve()
    return (PROJECT_ROOT / path).resolve()


def _run_bootstrap(
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
    parser = argparse.ArgumentParser(description="Run UK ETS automated data + one experiment pass.")
    parser.add_argument("--config", default="uk_ets/config/uk_ets_default.yaml")
    parser.add_argument("--data-dir", default="uk_ets/Data_auto_uk")
    parser.add_argument("--start-date", default="2021-05-19")
    parser.add_argument("--end-date", default=today)
    parser.add_argument("--target-mode", choices=["price", "returns"], default="returns")
    parser.add_argument("--llm-base-model", default=None)
    parser.add_argument("--disable-llm", action="store_true")
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--use-residual-wrapper", action="store_true")
    parser.add_argument("--residual-alpha", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--skip-bootstrap", action="store_true")
    parser.add_argument("--include-news", action="store_true")
    parser.add_argument("--news-start", default="2025-01-01")
    parser.add_argument("--news-end", default=today)
    parser.add_argument("--ice-report-recaptcha-token", default="")
    parser.add_argument("--require-actual-auctions", action="store_true")
    args = parser.parse_args()

    config_path = _resolve(args.config)
    data_dir = _resolve(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    if not args.skip_bootstrap:
        _run_bootstrap(
            output_dir=data_dir,
            start_date=args.start_date,
            end_date=args.end_date,
            include_news=args.include_news,
            news_start=args.news_start,
            news_end=args.news_end,
            ice_report_recaptcha_token=(args.ice_report_recaptcha_token or "").strip() or None,
            require_actual_auctions=args.require_actual_auctions,
        )

    overrides: dict = {
        "target": {"mode": args.target_mode},
        "output": {"write_project_paper": False},
    }
    if args.llm_base_model:
        overrides.setdefault("llm", {})["base_model"] = args.llm_base_model
    if args.disable_llm:
        overrides.setdefault("llm", {}).update({"methods": [], "max_samples": 0})
    if args.num_workers is not None:
        overrides.setdefault("compute", {})["num_workers"] = args.num_workers
    if args.use_residual_wrapper:
        overrides.setdefault("model", {})["use_residual_wrapper"] = True
    if args.residual_alpha is not None:
        overrides.setdefault("model", {})["residual_alpha"] = args.residual_alpha
    if args.seed is not None:
        overrides.setdefault("reproducibility", {})["seed"] = args.seed

    run_dir = run_experiment(
        config_path=str(config_path),
        overrides=overrides,
        data_dir=str(data_dir),
    )
    print(f"UK ETS pipeline run complete. Run dir: {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
