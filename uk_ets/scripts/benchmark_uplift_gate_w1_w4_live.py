#!/usr/bin/env python3
"""Full live rerun benchmark for uplift-style learned gates on W1-W4."""

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
BASELINE_RESULTS_PATH = (
    PROJECT_ROOT
    / "reports"
    / "uk_ets_gate_w1_w4_regularized_base"
    / "20260319_000250"
    / "results.csv"
)

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


@dataclass(frozen=True)
class WindowSpec:
    name: str
    train_end: str
    val_end: str
    test_end: str


WINDOWS = [
    WindowSpec("W4", "2021-10-16", "2022-10-16", "2023-06-20"),
    WindowSpec("W3", "2022-06-20", "2023-06-20", "2024-02-22"),
    WindowSpec("W2", "2023-02-22", "2024-02-22", "2024-10-26"),
    WindowSpec("W1", "2023-10-27", "2024-10-26", "2025-06-30"),
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
        "model": {
            "learning_rate": 0.001,
            "max_epochs": 40,
            "early_stopping_patience": 8,
            "dropout": 0.5,
            "weight_decay": 0.02,
        },
        "llm": {
            "cot_rf": {
                "online_memory_policy": {
                    "gate": {
                        "learned": {
                            "feature_columns": TOP20,
                            "scope": "full",
                            "max_samples": 0,
                            "bootstrap_include_splits": ["val", "test"],
                        }
                    }
                }
            }
        },
    }


def _candidate_overrides() -> dict[str, dict[str, Any]]:
    base = _base_overrides()
    return {
        "uplift_logistic_live": _merge(
            base,
            {
                "llm": {
                    "cot_rf": {
                        "online_memory_policy": {
                            "gate": {
                                "learned": {
                                    "model_type": "logistic",
                                    "probability_threshold_grid": [0.5, 0.6, 0.7, 0.8],
                                }
                            }
                        }
                    }
                }
            },
        ),
        "uplift_gbdt_live": _merge(
            base,
            {
                "llm": {
                    "cot_rf": {
                        "online_memory_policy": {
                            "gate": {
                                "learned": {
                                    "model_type": "hist_gbdt",
                                    "probability_threshold_grid": [0.5, 0.6, 0.7, 0.8],
                                }
                            }
                        }
                    }
                }
            },
        ),
    }


def _run_candidate(window: WindowSpec, overrides: dict[str, Any]) -> Path:
    merged = _merge(
        overrides,
        {
            "split": {
                "train_end": window.train_end,
                "val_end": window.val_end,
                "test_end": window.test_end,
            }
        },
    )
    return Path(run_experiment(config_path=CONFIG_PATH, overrides=merged, data_dir=DATA_DIR)).resolve()


def _extract_metrics(run_dir: Path) -> dict[str, Any]:
    path_df = pd.read_csv(run_dir / "results" / "path_metrics.csv")
    horizon_df = pd.read_csv(run_dir / "results" / "metrics_by_horizon.csv")
    rows = path_df[path_df["model"].isin(["tsm_llm_subset", METHOD_NAME])]
    h_rows = horizon_df[horizon_df["model"].isin(["tsm_llm_subset", METHOD_NAME])]
    out: dict[str, Any] = {}
    for model_name, prefix in [("tsm_llm_subset", "tsm"), (METHOD_NAME, "llm")]:
        out[f"{prefix}_path_mse"] = float(rows[rows["model"] == model_name].iloc[0]["mse_path"])
        by_h = h_rows[h_rows["model"] == model_name].set_index("horizon")
        out[f"{prefix}_h20"] = float(by_h.loc[20, "mse"])
        out[f"{prefix}_h30"] = float(by_h.loc[30, "mse"])
    out["path_improvement_pct"] = 1.0 - (out["llm_path_mse"] / out["tsm_path_mse"])
    out["h20_improvement_pct"] = 1.0 - (out["llm_h20"] / out["tsm_h20"])
    out["h30_improvement_pct"] = 1.0 - (out["llm_h30"] / out["tsm_h30"])

    gate_dir = run_dir / "results" / "online_memory_gate" / "TSM_LLM-COT-RF-HDELTA"
    probs_path = gate_dir / "test_learned_gate_probabilities.csv"
    if probs_path.exists():
        probs_df = pd.read_csv(probs_path)
        out["apply_rate"] = float(probs_df["apply_llm_selected"].mean())
        out["mean_apply_probability"] = float(probs_df["apply_probability"].mean())
        out["selected_thresholds_seen"] = ",".join(
            str(x) for x in sorted({float(v) for v in probs_df["selected_threshold"].dropna().tolist()})
        )
    else:
        out["apply_rate"] = float("nan")
        out["mean_apply_probability"] = float("nan")
        out["selected_thresholds_seen"] = ""

    selection_path = run_dir / "llm" / "online_memory_gate_selection_TSM_LLM-COT-RF-HDELTA.json"
    if selection_path.exists():
        import json

        meta = json.loads(selection_path.read_text())
        out["bootstrap_rows"] = int(meta.get("bootstrap_rows", 0))
        out["bootstrap_run_dirs"] = ",".join(meta.get("bootstrap_run_dirs", []) or [])
        out["selected_threshold"] = float(meta.get("threshold", float("nan")))
    else:
        out["bootstrap_rows"] = 0
        out["bootstrap_run_dirs"] = ""
        out["selected_threshold"] = float("nan")
    return out


def _seed_baseline_rows() -> list[dict[str, Any]]:
    if not BASELINE_RESULTS_PATH.exists():
        return []
    df = pd.read_csv(BASELINE_RESULTS_PATH)
    df = df[df["candidate"] == "gate_top20"].copy()
    rows: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        rows.append(
            {
                "candidate": "gate_top20_baseline",
                "window": str(row["window"]),
                "train_end": str(row["train_end"]),
                "val_end": str(row["val_end"]),
                "test_end": str(row["test_end"]),
                "run_dir": str(row["run_dir"]),
                "tsm_path_mse": float(row["tsm_path_mse"]),
                "llm_path_mse": float(row["llm_path_mse"]),
                "tsm_h20": float(row["tsm_h20"]),
                "llm_h20": float(row["llm_h20"]),
                "tsm_h30": float(row["tsm_h30"]),
                "llm_h30": float(row["llm_h30"]),
                "path_improvement_pct": float(row["path_improvement_pct"]),
                "h20_improvement_pct": float(row["h20_improvement_pct"]),
                "h30_improvement_pct": float(row["h30_improvement_pct"]),
                "apply_rate": float("nan"),
                "mean_apply_probability": float("nan"),
                "selected_thresholds_seen": "",
                "bootstrap_rows": 0,
                "bootstrap_run_dirs": "",
                "selected_threshold": float("nan"),
            }
        )
    return rows


def _baseline_run_dir_by_window() -> dict[str, str]:
    rows = _seed_baseline_rows()
    return {str(row["window"]): str(row["run_dir"]) for row in rows}


def _write_plan(out_dir: Path) -> None:
    lines = [
        "# W1-W4 Full Live Uplift Gate Benchmark Plan",
        "",
        f"Generated: {datetime.now().isoformat()}",
        "",
        "## Objective",
        "",
        "Run a true live rerun of the `W1-W4` benchmark where the learned gate is fit on prior completed windows plus the current window's validation rows, then used directly inside the full causal LLM pipeline.",
        "",
        "Frozen stack:",
        "- UK data root: `uk_ets/Data_auto_uk`",
        "- Regularized shared DLinear base: `lr=0.001`, `epochs=40`, `patience=8`, `dropout=0.5`, `weight_decay=0.02`",
        "- Same `top20` gate feature schema",
        "- Same 4B LLM refinement path",
        "",
        "## Candidates",
        "",
        "- `uplift_logistic_live`",
        "- `uplift_gbdt_live`",
        "",
        "## Training protocol",
        "",
        "- Process windows in chronological order `W4 -> W3 -> W2 -> W1`",
        "- For each candidate, bootstrap learned-gate training rows from that candidate's prior completed windows",
        "- Fit the gate on prior bootstrap rows plus the current window's validation rows",
        "- Select threshold from the current window's validation rows only",
        "",
        "## Expected outcome",
        "",
        "- More realistic than the conservative offline suppression test because the gate can both add and remove applications in the live run",
        "- Best chance of success: `uplift_gbdt_live`",
        "- Success condition: beat the current `gate_top20` mean `W1-W4` path MSE `54.613995`",
        "- Realistic expected lift if it works: about `+0.2%` to `+1.5%`",
    ]
    (out_dir / "plan.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_summary(out_dir: Path, df: pd.DataFrame) -> None:
    mean_cols = [
        "llm_path_mse",
        "path_improvement_pct",
        "h20_improvement_pct",
        "h30_improvement_pct",
        "apply_rate",
    ]
    mean_df = df.groupby("candidate")[mean_cols].mean().sort_values("llm_path_mse").reset_index()
    lines = [
        "# W1-W4 Full Live Uplift Gate Benchmark",
        "",
        f"Generated: {datetime.now().isoformat()}",
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
    out_dir = (
        Path(args.out_dir).expanduser().resolve()
        if args.out_dir
        else (PROJECT_ROOT / "reports" / "uk_ets_uplift_gate_w1_w4_live" / datetime.now().strftime("%Y%m%d_%H%M%S")).resolve()
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_plan(out_dir)

    results_path = out_dir / "results.csv"
    if results_path.exists():
        rows = pd.read_csv(results_path).to_dict(orient="records")
    else:
        rows = _seed_baseline_rows()
        if rows:
            pd.DataFrame(rows).to_csv(results_path, index=False)

    completed = {(str(row["candidate"]), str(row["window"])) for row in rows}
    all_candidates = _candidate_overrides()
    baseline_run_dir_map = _baseline_run_dir_by_window()
    if args.candidates:
        requested = [item.strip() for item in args.candidates.split(",") if item.strip()]
        candidate_map = {name: all_candidates[name] for name in requested}
    else:
        candidate_map = all_candidates

    for candidate_name, base_overrides in candidate_map.items():
        history_run_dirs: list[str] = []
        candidate_existing = [row for row in rows if str(row["candidate"]) == candidate_name]
        candidate_existing_by_window = {str(row["window"]): row for row in candidate_existing}
        for window in WINDOWS:
            existing = candidate_existing_by_window.get(window.name)
            if existing is not None:
                if str(existing.get("run_dir", "")).strip():
                    history_run_dirs.append(str(existing["run_dir"]))
                continue
            overrides = _merge(
                base_overrides,
                {
                    "llm": {
                        "cache_seed_run_dirs": [
                            baseline_run_dir_map.get(window.name, "")
                        ],
                    },
                },
            )
            overrides = _merge(
                overrides,
                {
                    "llm": {
                        "cot_rf": {
                            "online_memory_policy": {
                                "gate": {
                                    "learned": {
                                        "bootstrap_from_run_dirs": history_run_dirs,
                                    }
                                }
                            }
                        }
                    }
                },
            )
            print(f"Running {candidate_name} {window.name}", flush=True)
            run_dir = _run_candidate(window, overrides)
            metrics = _extract_metrics(run_dir)
            row = {
                "candidate": candidate_name,
                "window": window.name,
                "train_end": window.train_end,
                "val_end": window.val_end,
                "test_end": window.test_end,
                "run_dir": str(run_dir),
                **metrics,
            }
            rows.append(row)
            pd.DataFrame(rows).to_csv(results_path, index=False)
            history_run_dirs.append(str(run_dir))

    df = pd.DataFrame(rows)
    df.to_csv(results_path, index=False)
    _write_summary(out_dir, df)
    print(out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
