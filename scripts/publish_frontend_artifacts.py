#!/usr/bin/env python3
"""Publish frontend-ready artifacts from a completed run into a frontend public directory."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.frontend_export import export_frontend_artifacts


def _find_latest_completed_run(runs_root: Path) -> Path:
    candidates: list[tuple[float, Path]] = []
    for run_dir in runs_root.iterdir():
        if not run_dir.is_dir():
            continue
        metrics_path = run_dir / "results" / "path_metrics.csv"
        if metrics_path.exists():
            candidates.append((metrics_path.stat().st_mtime, run_dir))
    if not candidates:
        raise FileNotFoundError(f"No completed runs found under: {runs_root}")
    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates[0][1]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Publish frontend-ready artifacts into a frontend public directory."
    )
    parser.add_argument(
        "--publish-dir",
        required=True,
        help="Frontend static/public directory to receive JSON artifacts.",
    )
    parser.add_argument(
        "--run-dir",
        default=None,
        help="Completed backend run directory. If omitted, uses the latest completed run.",
    )
    parser.add_argument(
        "--runs-root",
        default="runs",
        help="Runs root used when --run-dir is omitted (default: runs).",
    )
    parser.add_argument(
        "--model-name",
        default=None,
        help="Optional exported model override.",
    )
    parser.add_argument(
        "--history-days",
        type=int,
        default=90,
        help="Recent history rows to include (default: 90).",
    )
    args = parser.parse_args()

    if args.run_dir:
        run_dir = Path(args.run_dir).resolve()
    else:
        run_dir = _find_latest_completed_run(Path(args.runs_root).resolve())

    publish_dir = Path(args.publish_dir).resolve()
    export_dir = export_frontend_artifacts(
        run_dir=run_dir,
        publish_dir=publish_dir,
        model_name=args.model_name,
        history_days=args.history_days,
    )
    print(f"Published frontend artifacts from run: {run_dir}")
    print(f"Export directory: {export_dir}")
    print(f"Frontend publish directory: {publish_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
