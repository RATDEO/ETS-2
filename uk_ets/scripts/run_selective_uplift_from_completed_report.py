#!/usr/bin/env python3
"""Run helpful-case mining and selective uplift evaluation from a completed W1-W4 report."""

from __future__ import annotations

import argparse
import subprocess
from datetime import datetime
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report-results", required=True, type=str, help="Path to completed W1-W4 results.csv with window/train_end/val_end/test_end/run_dir")
    parser.add_argument("--out-root", type=str, default="")
    args = parser.parse_args()

    results_path = Path(args.report_results).expanduser().resolve()
    if args.out_root:
        out_root = Path(args.out_root).expanduser().resolve()
    else:
        out_root = (
            PROJECT_ROOT
            / "reports"
            / "uk_ets_selective_uplift_from_completed_report"
            / datetime.now().strftime("%Y%m%d_%H%M%S")
        ).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(results_path)
    required = ["window", "train_end", "val_end", "test_end", "run_dir"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"report-results missing required columns: {missing}")

    runs_csv = out_root / "runs.csv"
    df[required].to_csv(runs_csv, index=False)

    helpful_out = out_root / "helpful_cases"
    uplift_out = out_root / "uplift_models"

    subprocess.run(
        [
            "python",
            str(PROJECT_ROOT / "uk_ets" / "scripts" / "build_selective_residual_helpful_case_dataset.py"),
            "--runs-csv",
            str(runs_csv),
            "--out-dir",
            str(helpful_out),
        ],
        check=True,
    )
    subprocess.run(
        [
            "python",
            str(PROJECT_ROOT / "uk_ets" / "scripts" / "evaluate_selective_uplift_models_w1_w4.py"),
            "--source-dataset",
            str(helpful_out / "helpful_case_dataset.csv"),
            "--out-dir",
            str(uplift_out),
        ],
        check=True,
    )
    print(out_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
