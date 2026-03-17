#!/usr/bin/env python3
"""Repeated production holdouts with the regularized live TSM backbone."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]

import sys

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.eval.return_metrics import assess_llm_training_cutoff
from src.run_experiment import run_experiment


CONFIG_PATH = "uk_ets/config/uk_ets_llm_4b_tsm_live_gbdt_top20.yaml"
DATA_DIR = "uk_ets/Data_auto_uk"
ANCHOR_RUN_DIR = PROJECT_ROOT / "runs" / "20260316_170659_fb765d"
METHOD_NAME = "TSM+LLM-COT-RF-HDELTA"
N_WINDOWS = 5


@dataclass(frozen=True)
class HoldoutWindow:
    idx: int
    train_end: pd.Timestamp
    val_end: pd.Timestamp
    test_end: pd.Timestamp

    @property
    def name(self) -> str:
        return f"W{self.idx}_{self.test_start.date()}_{self.test_end.date()}"

    @property
    def test_start(self) -> pd.Timestamp:
        return self.val_end + pd.Timedelta(days=1)


def _load_anchor_split() -> dict[str, pd.Timestamp]:
    with (ANCHOR_RUN_DIR / "config_resolved.yaml").open() as f:
        cfg = yaml.safe_load(f)
    split = cfg["split"]
    return {
        "train_end": pd.Timestamp(split["train_end"]),
        "val_end": pd.Timestamp(split["val_end"]),
        "test_end": pd.Timestamp(split["test_end"]),
    }


def _build_windows(n_windows: int) -> list[HoldoutWindow]:
    split = _load_anchor_split()
    test_span = split["test_end"] - split["val_end"]
    windows: list[HoldoutWindow] = []
    for idx in range(n_windows):
        shift = test_span * idx
        windows.append(
            HoldoutWindow(
                idx=idx,
                train_end=split["train_end"] - shift,
                val_end=split["val_end"] - shift,
                test_end=split["test_end"] - shift,
            )
        )
    return windows


def _run_window(window: HoldoutWindow) -> Path:
    overrides: dict[str, Any] = {
        "split": {
            "train_end": str(window.train_end.date()),
            "val_end": str(window.val_end.date()),
            "test_end": str(window.test_end.date()),
        },
        "model": {
            "learning_rate": 0.001,
            "max_epochs": 40,
            "early_stopping_patience": 8,
            "dropout": 0.5,
            "weight_decay": 0.02,
        },
    }
    run_dir = run_experiment(config_path=CONFIG_PATH, overrides=overrides, data_dir=DATA_DIR)
    return Path(run_dir).resolve()


def _read_path_metrics(run_dir: Path) -> pd.DataFrame:
    return pd.read_csv(run_dir / "results" / "path_metrics.csv")


def _read_horizon_metrics(run_dir: Path) -> pd.DataFrame:
    return pd.read_csv(run_dir / "results" / "metrics_by_horizon.csv")


def _extract_path_mse(path_df: pd.DataFrame, method_name: str) -> float:
    row = path_df[path_df["model"] == method_name]
    if row.empty:
        raise RuntimeError(f"Missing {method_name} in path metrics")
    return float(row.iloc[0]["mse_path"])


def _extract_horizon_mse(h_df: pd.DataFrame, method_name: str) -> dict[str, float]:
    rows = h_df[h_df["model"] == method_name]
    if rows.empty:
        raise RuntimeError(f"Missing {method_name} in horizon metrics")
    out: dict[str, float] = {}
    for horizon in (1, 5, 20, 30):
        row = rows[rows["horizon"] == horizon]
        if not row.empty:
            out[f"h{horizon}"] = float(row.iloc[0]["mse"])
    return out


def _cutoff_note(run_dir: Path) -> dict[str, object]:
    pred = pd.read_parquet(run_dir / "predictions" / "tsm_pred_test.parquet")
    dates = pd.to_datetime(pred["date"])
    return assess_llm_training_cutoff(dates, cutoff_date="2025-01-01")


def _write_summary(out_dir: Path, df: pd.DataFrame) -> None:
    llm_better = int((df["llm_path_mse"] < df["tsm_path_mse"]).sum())
    mean_gain = float((1.0 - (df["llm_path_mse"] / df["tsm_path_mse"])).mean())
    median_gain = float((1.0 - (df["llm_path_mse"] / df["tsm_path_mse"])).median())

    lines = [
        "# Repeated Production Holdout Benchmark (Regularized Base)",
        "",
        f"Generated: {datetime.now().isoformat()}",
        "",
        "Protocol:",
        "- Same repeated holdout geometry as the production benchmark.",
        "- UK-specific data root forced to `uk_ets/Data_auto_uk`.",
        "- Frozen method: `DLinear + learned_gbdt(top20) + selective LLM`.",
        "- Base DLinear regularized to `lr=0.001`, `max_epochs=40`, `patience=8`, `dropout=0.5`, `weight_decay=0.02`.",
        "",
        "## Window Results",
        "",
        df.to_markdown(index=False),
        "",
        "## Summary",
        f"- LLM beat raw `TSM` on path MSE in `{llm_better}/{len(df)}` repeated holdouts.",
        f"- Mean path-MSE improvement: `{mean_gain:.2%}`.",
        f"- Median path-MSE improvement: `{median_gain:.2%}`.",
        f"- Mean `h20` improvement: `{df['h20_improvement_pct'].mean():.2%}`.",
        f"- Mean `h30` improvement: `{df['h30_improvement_pct'].mean():.2%}`.",
    ]
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    out_dir = (
        PROJECT_ROOT
        / "reports"
        / "uk_ets_repeated_production_holdout_regularized"
        / datetime.now().strftime("%Y%m%d_%H%M%S")
    ).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    windows = _build_windows(N_WINDOWS)
    rows: list[dict[str, object]] = []
    for window in windows:
        run_dir = _run_window(window)
        path_df = _read_path_metrics(run_dir)
        horizon_df = _read_horizon_metrics(run_dir)
        tsm_path = _extract_path_mse(path_df, "tsm_llm_subset")
        llm_path = _extract_path_mse(path_df, METHOD_NAME)
        tsm_h = _extract_horizon_mse(horizon_df, "tsm_llm_subset")
        llm_h = _extract_horizon_mse(horizon_df, METHOD_NAME)
        cutoff = _cutoff_note(run_dir)
        row = {
            "window": window.name,
            "train_end": str(window.train_end.date()),
            "val_end": str(window.val_end.date()),
            "test_end": str(window.test_end.date()),
            "run_dir": str(run_dir),
            "tsm_path_mse": tsm_path,
            "llm_path_mse": llm_path,
            "path_improvement_pct": 1.0 - (llm_path / tsm_path),
            "tsm_h1": tsm_h.get("h1"),
            "llm_h1": llm_h.get("h1"),
            "tsm_h5": tsm_h.get("h5"),
            "llm_h5": llm_h.get("h5"),
            "tsm_h20": tsm_h.get("h20"),
            "llm_h20": llm_h.get("h20"),
            "h20_improvement_pct": 1.0 - (llm_h.get("h20") / tsm_h.get("h20")),
            "tsm_h30": tsm_h.get("h30"),
            "llm_h30": llm_h.get("h30"),
            "h30_improvement_pct": 1.0 - (llm_h.get("h30") / tsm_h.get("h30")),
            "llm_cutoff_status": cutoff["status"],
            "llm_cutoff_date": cutoff["cutoff_date"],
            "llm_n_pre_cutoff": cutoff["n_pre_cutoff"],
            "llm_n_post_cutoff": cutoff["n_post_cutoff"],
        }
        rows.append(row)
        pd.DataFrame(rows).to_csv(out_dir / "window_results.csv", index=False)

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "window_results.csv", index=False)
    _write_summary(out_dir, df)
    print(out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
