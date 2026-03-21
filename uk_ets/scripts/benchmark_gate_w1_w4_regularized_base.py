#!/usr/bin/env python3
"""Benchmark live gate feature schemas on W1-W4 using the regularized TSM base."""

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


CONFIG_PATH = "uk_ets/config/uk_ets_llm_4b_tsm_live_gbdt_top20.yaml"
DATA_DIR = "uk_ets/Data_auto_uk"
METHOD_NAME = "TSM+LLM-COT-RF-HDELTA"

TOP20 = [
    "positive_count_h30",
    "positive_count",
    "positive_signal_h30",
    "positive_signal",
    "net_signal",
    "negative_mean_helpfulness",
    "base_move_h5_pct",
    "profile_fc_h5",
    "negative_count",
    "positive_mean_helpfulness",
    "profile_vol_pct",
    "positive_mean_similarity",
    "positive_best_similarity",
    "profile_change_5",
    "positive_count_h20",
    "negative_signal",
    "base_move_h20_pct",
    "profile_fc_h20",
    "negative_signal_h30",
    "base_move_h30_pct",
]

TOP25 = TOP20 + [
    "profile_fc_h30",
    "negative_best_similarity",
    "negative_mean_similarity",
    "negative_count_h30",
    "negative_count_h20",
]

TOP29 = [
    "positive_count",
    "negative_count",
    "positive_signal",
    "negative_signal",
    "net_signal",
    "positive_count_h20",
    "positive_count_h30",
    "negative_count_h20",
    "negative_count_h30",
    "positive_signal_h20",
    "positive_signal_h30",
    "negative_signal_h20",
    "negative_signal_h30",
    "positive_best_similarity",
    "negative_best_similarity",
    "positive_mean_similarity",
    "negative_mean_similarity",
    "positive_mean_helpfulness",
    "negative_mean_helpfulness",
    "support_example_count",
    "base_move_h5_pct",
    "base_move_h20_pct",
    "base_move_h30_pct",
    "profile_change_5",
    "profile_change_20",
    "profile_vol_pct",
    "profile_fc_h5",
    "profile_fc_h20",
    "profile_fc_h30",
]


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


def _candidate_overrides() -> dict[str, dict[str, Any]]:
    base_model = {
        "learning_rate": 0.001,
        "max_epochs": 40,
        "early_stopping_patience": 8,
        "dropout": 0.5,
        "weight_decay": 0.02,
    }
    return {
        "gate_top20": {
            "model": base_model,
            "llm": {
                "cot_rf": {
                    "online_memory_policy": {
                        "gate": {
                            "learned": {
                                "feature_columns": TOP20,
                            }
                        }
                    }
                }
            },
        },
        "gate_top25": {
            "model": base_model,
            "llm": {
                "cot_rf": {
                    "online_memory_policy": {
                        "gate": {
                            "learned": {
                                "feature_columns": TOP25,
                            }
                        }
                    }
                }
            },
        },
        "gate_top29": {
            "model": base_model,
            "llm": {
                "cot_rf": {
                    "online_memory_policy": {
                        "gate": {
                            "learned": {
                                "feature_columns": TOP29,
                            }
                        }
                    }
                }
            },
        },
    }


def _merge(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    out = dict(a)
    for k, v in b.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


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


def _write_summary(out_dir: Path, df: pd.DataFrame) -> None:
    mean_df = (
        df.groupby("candidate")[["tsm_path_mse", "llm_path_mse", "path_improvement_pct", "h20_improvement_pct", "h30_improvement_pct"]]
        .mean()
        .sort_values("path_improvement_pct", ascending=False)
        .reset_index()
    )
    lines = [
        "# W1-W4 Gate Feature Benchmark (Regularized Base)",
        "",
        f"Generated: {datetime.now().isoformat()}",
        "",
        "Frozen base:",
        "- UK data root: `uk_ets/Data_auto_uk`",
        "- DLinear regularized: `lr=0.001`, `max_epochs=40`, `patience=8`, `dropout=0.5`, `weight_decay=0.02`",
        "",
        "## Results",
        "",
        df.to_markdown(index=False),
        "",
        "## Mean By Candidate",
        "",
        mean_df.to_markdown(index=False),
    ]
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=str, default="")
    parser.add_argument("--candidates", type=str, default="")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.out_dir:
        out_dir = Path(args.out_dir).expanduser().resolve()
    else:
        out_dir = (
            PROJECT_ROOT
            / "reports"
            / "uk_ets_gate_w1_w4_regularized_base"
            / datetime.now().strftime("%Y%m%d_%H%M%S")
        ).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    results_path = out_dir / "results.csv"
    if results_path.exists():
        rows = pd.read_csv(results_path).to_dict(orient="records")
    else:
        rows = []
    completed = {(str(row["candidate"]), str(row["window"])) for row in rows}

    all_candidates = _candidate_overrides()
    if args.candidates:
        requested = [item.strip() for item in args.candidates.split(",") if item.strip()]
        candidate_map = {name: all_candidates[name] for name in requested}
    else:
        candidate_map = all_candidates

    for candidate_name, overrides in candidate_map.items():
        for window in WINDOWS:
            key = (candidate_name, window.name)
            if key in completed:
                print(f"Skipping completed {candidate_name} {window.name}", flush=True)
                continue
            print(f"Running {candidate_name} {window.name}", flush=True)
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
                    **metrics,
                }
            )
            pd.DataFrame(rows).to_csv(results_path, index=False)

    df = pd.DataFrame(rows)
    df.to_csv(results_path, index=False)
    _write_summary(out_dir, df)
    print(out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
