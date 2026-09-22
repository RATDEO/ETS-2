#!/usr/bin/env python3
"""Run the frozen 27B reference policy across repeated-holdout windows W1-W4."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]

import sys

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.run_experiment import run_experiment


CONFIG_PATH = "uk_ets/config/uk_ets_llm_27b_current_default_reference.yaml"
DATA_DIR = "uk_ets/Data_auto_uk"
METHOD_NAME = "TSM+LLM-COT-RF-HDELTA"


@dataclass(frozen=True)
class WindowSpec:
    name: str
    train_end: str
    val_end: str
    test_end: str


WINDOWS = [
    WindowSpec("W1", "2023-10-27", "2024-10-26", "2025-06-30"),
    WindowSpec("W2", "2023-02-22", "2024-02-22", "2024-10-26"),
    WindowSpec("W3", "2022-06-20", "2023-06-20", "2024-02-22"),
    WindowSpec("W4", "2021-10-16", "2022-10-16", "2023-06-20"),
]


def _run_window(window: WindowSpec) -> Path:
    overrides: dict[str, Any] = {
        "split": {
            "train_end": window.train_end,
            "val_end": window.val_end,
            "test_end": window.test_end,
        }
    }
    run_dir = run_experiment(
        config_path=CONFIG_PATH,
        overrides=overrides,
        data_dir=DATA_DIR,
    )
    return Path(run_dir).resolve()


def _extract_metrics(run_dir: Path) -> dict[str, Any]:
    path_df = pd.read_csv(run_dir / "results" / "path_metrics.csv")
    horizon_df = pd.read_csv(run_dir / "results" / "metrics_by_horizon.csv")
    long_short_df = pd.read_csv(run_dir / "results" / "long_short_metrics.csv")
    pred_df = pd.read_parquet(run_dir / "predictions" / "tsm_pred_test.parquet")

    path_rows = path_df[path_df["model"].isin(["tsm", METHOD_NAME])].set_index("model")
    horizon_rows = horizon_df[horizon_df["model"].isin(["tsm", METHOD_NAME])].set_index(
        ["model", "horizon"]
    )
    sharpe_rows = long_short_df[long_short_df["model"].isin(["tsm", METHOD_NAME])].set_index(
        ["model", "horizon"]
    )

    tsm_path = float(path_rows.loc["tsm", "mse_path"])
    llm_path = float(path_rows.loc[METHOD_NAME, "mse_path"])
    tsm_h20 = float(horizon_rows.loc[("tsm", 20), "mse"])
    llm_h20 = float(horizon_rows.loc[(METHOD_NAME, 20), "mse"])
    tsm_h30 = float(horizon_rows.loc[("tsm", 30), "mse"])
    llm_h30 = float(horizon_rows.loc[(METHOD_NAME, 30), "mse"])

    origin_dates = pd.to_datetime(pred_df["date"]).sort_values()
    return {
        "origin_start": str(origin_dates.iloc[0].date()),
        "origin_end": str(origin_dates.iloc[-1].date()),
        "n_origins": int(origin_dates.nunique()),
        "tsm_path_mse": tsm_path,
        "llm_path_mse": llm_path,
        "path_improvement_pct": 1.0 - (llm_path / tsm_path),
        "tsm_h20_mse": tsm_h20,
        "llm_h20_mse": llm_h20,
        "h20_improvement_pct": 1.0 - (llm_h20 / tsm_h20),
        "tsm_h30_mse": tsm_h30,
        "llm_h30_mse": llm_h30,
        "h30_improvement_pct": 1.0 - (llm_h30 / tsm_h30),
        "tsm_h20_sharpe": float(
            sharpe_rows.loc[("tsm", 20), "sharpe_non_overlap_offset_avg"]
        ),
        "llm_h20_sharpe": float(
            sharpe_rows.loc[(METHOD_NAME, 20), "sharpe_non_overlap_offset_avg"]
        ),
        "tsm_h30_sharpe": float(
            sharpe_rows.loc[("tsm", 30), "sharpe_non_overlap_offset_avg"]
        ),
        "llm_h30_sharpe": float(
            sharpe_rows.loc[(METHOD_NAME, 30), "sharpe_non_overlap_offset_avg"]
        ),
    }


def _write_summary(out_dir: Path, df: pd.DataFrame) -> None:
    lines = [
        "# 27B Reference W1-W4 Sweep",
        "",
        f"Generated: {datetime.now().isoformat()}",
        "",
        f"Config: `{CONFIG_PATH}`",
        f"Data dir: `{DATA_DIR}`",
        "",
        "## Results",
        "",
        df.to_markdown(index=False),
        "",
        "## Mean",
        "",
        df[
            [
                "tsm_path_mse",
                "llm_path_mse",
                "path_improvement_pct",
                "h20_improvement_pct",
                "h30_improvement_pct",
                "tsm_h20_sharpe",
                "llm_h20_sharpe",
                "tsm_h30_sharpe",
                "llm_h30_sharpe",
            ]
        ]
        .mean()
        .to_frame("mean")
        .to_markdown(),
    ]
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    out_dir = (
        PROJECT_ROOT
        / "reports"
        / "uk_ets_27b_reference_w1_w4"
        / datetime.now().strftime("%Y%m%d_%H%M%S")
    ).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for window in WINDOWS:
        print(f"Running {window.name} ({window.train_end} / {window.val_end} / {window.test_end})", flush=True)
        run_dir = _run_window(window)
        metrics = _extract_metrics(run_dir)
        rows.append(
            {
                "window": window.name,
                "train_end": window.train_end,
                "val_end": window.val_end,
                "test_end": window.test_end,
                "run_dir": str(run_dir),
                **metrics,
            }
        )
        pd.DataFrame(rows).to_csv(out_dir / "window_results.csv", index=False)

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "window_results.csv", index=False)
    _write_summary(out_dir, df)
    print(out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
