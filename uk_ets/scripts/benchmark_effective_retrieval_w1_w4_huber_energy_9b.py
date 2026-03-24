#!/usr/bin/env python3
"""Benchmark the current wide8 UK schema on the swapped 9B endpoint."""

from __future__ import annotations

import argparse
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
from uk_ets.scripts.benchmark_effective_retrieval_w1_w4_huber_energy import (
    ENERGY_INTERACTIONS,
    METHOD_NAME,
)

CONFIG_PATH = "uk_ets/config/uk_ets_llm_4b_current_default.yaml"
DATA_DIR = "uk_ets/Data_auto_uk"
BASELINE_4B_RESULTS = (
    PROJECT_ROOT
    / "reports"
    / "uk_ets_effective_retrieval_w1_w4_huber_energy"
    / "20260320_174901"
    / "results.csv"
)
MODEL_9B = "qwen3.5-9b-ud-q4-k-xl"


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


def _merge(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    out = dict(a)
    for k, v in b.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


def _base_overrides() -> dict[str, Any]:
    return {
        "time_series": {
            "seq_len": 20,
            "label_len": 10,
        },
        "model": {
            "learning_rate": 0.001,
            "max_epochs": 40,
            "early_stopping_patience": 8,
            "dropout": 0.5,
            "weight_decay": 0.02,
            "dlinear_individual": False,
            "loss_type": "huber",
            "loss_huber_beta": 0.5,
        },
        "features": {
            "max_exogenous_features_model": len(ENERGY_INTERACTIONS),
            "preferred_feature_order": ENERGY_INTERACTIONS,
        },
        "llm": {
            "model": MODEL_9B,
            "cache_enabled": False,
            "cot_rf": {
                "k_examples": 10,
                "n_similarity_examples": 5,
                "online_memory_policy": {
                    "support_examples": 8,
                    "positive_examples": 3,
                    "negative_examples": 1,
                    "max_total_examples": 12,
                },
            },
        },
        "hdelta": {
            "case_match_top_k": 8,
        },
    }


def _run_window(window: WindowSpec) -> Path:
    overrides = _merge(
        _base_overrides(),
        {
            "split": {
                "train_end": window.train_end,
                "val_end": window.val_end,
                "test_end": window.test_end,
            }
        },
    )
    return Path(run_experiment(config_path=CONFIG_PATH, overrides=overrides, data_dir=DATA_DIR)).resolve()


def _extract_metrics(run_dir: Path) -> dict[str, float]:
    path_df = pd.read_csv(run_dir / "results" / "path_metrics.csv")
    horizon_df = pd.read_csv(run_dir / "results" / "metrics_by_horizon.csv")
    rows = path_df[path_df["model"].isin(["tsm_llm_subset", METHOD_NAME])]
    h_rows = horizon_df[horizon_df["model"].isin(["tsm_llm_subset", METHOD_NAME])]
    out: dict[str, float] = {}
    for model_name, prefix in [("tsm_llm_subset", "tsm"), (METHOD_NAME, "llm")]:
        out[f"{prefix}_path_mse"] = float(rows[rows["model"] == model_name].iloc[0]["mse_path"])
        by_h = h_rows[h_rows["model"] == model_name].set_index("horizon")
        out[f"{prefix}_h20"] = float(by_h.loc[20, "mse"])
        out[f"{prefix}_h30"] = float(by_h.loc[30, "mse"])
    out["path_improvement_pct"] = 1.0 - (out["llm_path_mse"] / out["tsm_path_mse"])
    out["h20_improvement_pct"] = 1.0 - (out["llm_h20"] / out["tsm_h20"])
    out["h30_improvement_pct"] = 1.0 - (out["llm_h30"] / out["tsm_h30"])
    return out


def _write_plan(out_dir: Path) -> None:
    lines = [
        "# 9B Wide8 Benchmark Plan",
        "",
        "## Objective",
        "- Re-run the current best 4B UK live schema on the swapped 9B endpoint.",
        "- Hold everything else fixed: improved Huber + energy base, wide effective retrieval, same live heuristic policy.",
        "",
        "## Endpoint Controls",
        f"- Model label forced to `{MODEL_9B}`.",
        "- Cache disabled to prevent stale 4B reuse on the same base URL.",
        "",
        "## Comparison Baseline",
        "- Last completed 4B current-schema benchmark: `effective_retrieval_wide8`.",
    ]
    (out_dir / "plan.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_summary(out_dir: Path, df: pd.DataFrame) -> None:
    mean_9b = df[["tsm_path_mse", "llm_path_mse", "path_improvement_pct", "h20_improvement_pct", "h30_improvement_pct"]].mean().to_dict()
    baseline = pd.read_csv(BASELINE_4B_RESULTS)
    baseline = baseline[baseline["candidate"] == "effective_retrieval_wide8"].copy()
    mean_4b = baseline[["tsm_path_mse", "llm_path_mse", "path_improvement_pct", "h20_improvement_pct", "h30_improvement_pct"]].mean().to_dict()
    compare = pd.DataFrame(
        [
            {"run_family": "4b_wide8", **mean_4b},
            {"run_family": "9b_wide8", **mean_9b},
        ]
    )
    lines = [
        "# 9B Wide8 Benchmark Summary",
        "",
        f"Generated: {datetime.now().isoformat()}",
        "",
        "## By Window",
        "",
        df.to_markdown(index=False),
        "",
        "## 4B vs 9B Mean Comparison",
        "",
        compare.to_markdown(index=False),
    ]
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=str, default="")
    args = parser.parse_args()
    if args.out_dir:
        out_dir = Path(args.out_dir).expanduser().resolve()
    else:
        out_dir = (
            PROJECT_ROOT
            / "reports"
            / "uk_ets_effective_retrieval_w1_w4_huber_energy_9b"
            / datetime.now().strftime("%Y%m%d_%H%M%S")
        ).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_plan(out_dir)

    rows = []
    for window in WINDOWS:
        print(f"Running 9B wide8 {window.name}", flush=True)
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
        pd.DataFrame(rows).to_csv(out_dir / "results.csv", index=False)

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "results.csv", index=False)
    _write_summary(out_dir, df)
    print(out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
