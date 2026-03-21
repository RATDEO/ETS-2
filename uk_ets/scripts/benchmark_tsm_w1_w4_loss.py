#!/usr/bin/env python3
"""Benchmark loss shaping candidates on difficult windows W1-W4."""

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


DATA_DIR = "uk_ets/Data_auto_uk"
CONFIG_PATH = "uk_ets/config/uk_ets_llm_4b_tsm_live_gbdt_top20.yaml"
TAIL_WEIGHTS = {1: 1.0, 5: 1.0, 20: 2.0, 30: 3.0}


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


def _candidate_overrides() -> dict[str, dict[str, Any]]:
    base_model = {
        "learning_rate": 0.001,
        "max_epochs": 40,
        "early_stopping_patience": 8,
        "dropout": 0.5,
        "weight_decay": 0.02,
        "dlinear_individual": False,
    }
    return {
        "mse_shared_seq20_reg": {
            "llm": {"methods": []},
            "time_series": {"seq_len": 20, "label_len": 10},
            "model": {
                **base_model,
                "loss_type": "mse",
            },
        },
        "huber_shared_seq20_reg": {
            "llm": {"methods": []},
            "time_series": {"seq_len": 20, "label_len": 10},
            "model": {
                **base_model,
                "loss_type": "huber",
                "loss_huber_beta": 0.5,
            },
        },
        "weighted_mse_shared_seq20_reg": {
            "llm": {"methods": []},
            "time_series": {"seq_len": 20, "label_len": 10},
            "model": {
                **base_model,
                "loss_type": "weighted_mse",
                "loss_horizon_weights": TAIL_WEIGHTS,
            },
        },
        "weighted_huber_shared_seq20_reg": {
            "llm": {"methods": []},
            "time_series": {"seq_len": 20, "label_len": 10},
            "model": {
                **base_model,
                "loss_type": "weighted_huber",
                "loss_huber_beta": 0.5,
                "loss_horizon_weights": TAIL_WEIGHTS,
            },
        },
    }


def _run_candidate(window: WindowSpec, candidate_name: str, base_overrides: dict[str, Any]) -> Path:
    overrides = _merge(
        base_overrides,
        {
            "split": {
                "train_end": window.train_end,
                "val_end": window.val_end,
                "test_end": window.test_end,
            }
        },
    )
    return Path(
        run_experiment(
            config_path=CONFIG_PATH,
            overrides=overrides,
            data_dir=DATA_DIR,
        )
    ).resolve()


def _extract_metrics(run_dir: Path) -> dict[str, float]:
    path_df = pd.read_csv(run_dir / "results" / "path_metrics.csv")
    horizon_df = pd.read_csv(run_dir / "results" / "metrics_by_horizon.csv")
    tsm_row = path_df[path_df["model"] == "tsm"].iloc[0]
    h_rows = horizon_df[horizon_df["model"] == "tsm"].set_index("horizon")
    return {
        "path_mse": float(tsm_row["mse_path"]),
        "h1": float(h_rows.loc[1, "mse"]),
        "h5": float(h_rows.loc[5, "mse"]),
        "h20": float(h_rows.loc[20, "mse"]),
        "h30": float(h_rows.loc[30, "mse"]),
    }


def _write_summary(out_dir: Path, df: pd.DataFrame) -> None:
    mean_df = (
        df.groupby("candidate")[["path_mse", "h1", "h5", "h20", "h30"]]
        .mean()
        .sort_values("path_mse")
        .reset_index()
    )
    baseline = mean_df[mean_df["candidate"] == "mse_shared_seq20_reg"].iloc[0]
    best = mean_df.iloc[0]
    gain = 1.0 - (float(best["path_mse"]) / float(baseline["path_mse"]))
    best_rows = df.loc[df.groupby("window")["path_mse"].idxmin()].sort_values("window")
    lines = [
        "# W1-W4 Loss Shaping Benchmark",
        "",
        f"Generated: {datetime.now().isoformat()}",
        "",
        "## Setup",
        "",
        "- Base only: `llm.methods=[]`.",
        "- Fixed shared regularized `DLinear`: `seq_len=20`, `label_len=10`, `lr=0.001`, `epochs=40`, `patience=8`, `dropout=0.5`, `weight_decay=0.02`.",
        "- Tested `mse`, `huber`, `weighted_mse`, and `weighted_huber`.",
        f"- Tail weights: `{TAIL_WEIGHTS}`.",
        "",
        "## Results",
        "",
        df.to_markdown(index=False),
        "",
        "## Mean By Candidate",
        "",
        mean_df.to_markdown(index=False),
        "",
        "## Best Candidate Per Window",
        "",
        best_rows.to_markdown(index=False),
        "",
        "## Headline",
        "",
        f"- Best mean candidate: `{best['candidate']}` with path MSE `{best['path_mse']:.6f}`.",
        f"- Baseline `mse_shared_seq20_reg` path MSE: `{baseline['path_mse']:.6f}`.",
        f"- Improvement vs baseline: `{gain:.2%}`.",
    ]
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    out_dir = (
        PROJECT_ROOT
        / "reports"
        / "uk_ets_tsm_w1_w4_loss_benchmark"
        / datetime.now().strftime("%Y%m%d_%H%M%S")
    ).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for candidate_name, overrides in _candidate_overrides().items():
        for window in WINDOWS:
            run_dir = _run_candidate(window, candidate_name, overrides)
            metrics = _extract_metrics(run_dir)
            rows.append(
                {
                    "candidate": candidate_name,
                    "window": window.name,
                    "train_end": window.train_end,
                    "val_end": window.val_end,
                    "test_end": window.test_end,
                    "run_dir": str(run_dir),
                    "loss_type": overrides["model"]["loss_type"],
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
