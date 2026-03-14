#!/usr/bin/env python3
"""Export frontend-ready JSON artifacts from a completed run."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.frontend_export import export_frontend_artifacts


def main() -> int:
    parser = argparse.ArgumentParser(description="Export frontend-ready artifacts from a run directory.")
    parser.add_argument("--run-dir", required=True, help="Run directory to export from.")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory for exported JSON files (default: <run-dir>/frontend_export).",
    )
    parser.add_argument(
        "--publish-dir",
        default=None,
        help="Optional frontend static/public directory to copy exported files into.",
    )
    parser.add_argument(
        "--model-name",
        default=None,
        help="Optional model override. Default is the best available frontend priority model.",
    )
    parser.add_argument(
        "--history-days",
        type=int,
        default=90,
        help="Recent history length to include (default: 90 rows).",
    )
    args = parser.parse_args()

    out_dir = export_frontend_artifacts(
        run_dir=args.run_dir,
        output_dir=args.output_dir,
        publish_dir=args.publish_dir,
        model_name=args.model_name,
        history_days=args.history_days,
    )
    print(f"Frontend artifacts exported to: {out_dir}")
    if args.publish_dir:
        print(f"Frontend artifacts published to: {Path(args.publish_dir).resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
