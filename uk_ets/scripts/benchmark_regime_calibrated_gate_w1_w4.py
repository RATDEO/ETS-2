#!/usr/bin/env python3
"""Benchmark regime-calibrated learned-gate thresholds on W1-W4."""

from __future__ import annotations

import argparse
import json
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


def _base_model_overrides() -> dict[str, Any]:
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
                        }
                    }
                }
            }
        },
    }


def _candidate_overrides() -> dict[str, dict[str, Any]]:
    base = _base_model_overrides()
    return {
        "regime_cal_move_default": _merge(
            base,
            {
                "llm": {
                    "cot_rf": {
                        "online_memory_policy": {
                            "gate": {
                                "learned": {
                                    "regime_calibration": {
                                        "enabled": True,
                                        "metric": "mse_path",
                                        "min_bucket_rows": 8,
                                        "max_abs_base_h20_pct": 2.5,
                                        "max_abs_base_h30_pct": 4.0,
                                        "use_profile_vol": False,
                                    }
                                }
                            }
                        }
                    }
                }
            },
        ),
        "regime_cal_move_loose": _merge(
            base,
            {
                "llm": {
                    "cot_rf": {
                        "online_memory_policy": {
                            "gate": {
                                "learned": {
                                    "regime_calibration": {
                                        "enabled": True,
                                        "metric": "mse_path",
                                        "min_bucket_rows": 8,
                                        "max_abs_base_h20_pct": 3.0,
                                        "max_abs_base_h30_pct": 5.0,
                                        "use_profile_vol": False,
                                    }
                                }
                            }
                        }
                    }
                }
            },
        ),
        "regime_cal_move_tight": _merge(
            base,
            {
                "llm": {
                    "cot_rf": {
                        "online_memory_policy": {
                            "gate": {
                                "learned": {
                                    "regime_calibration": {
                                        "enabled": True,
                                        "metric": "mse_path",
                                        "min_bucket_rows": 8,
                                        "max_abs_base_h20_pct": 2.0,
                                        "max_abs_base_h30_pct": 3.0,
                                        "use_profile_vol": False,
                                    }
                                }
                            }
                        }
                    }
                }
            },
        ),
        "regime_cal_move_vol_default": _merge(
            base,
            {
                "llm": {
                    "cot_rf": {
                        "online_memory_policy": {
                            "gate": {
                                "learned": {
                                    "regime_calibration": {
                                        "enabled": True,
                                        "metric": "mse_path",
                                        "min_bucket_rows": 8,
                                        "max_abs_base_h20_pct": 2.5,
                                        "max_abs_base_h30_pct": 4.0,
                                        "use_profile_vol": True,
                                        "max_profile_vol_pct": 3.8,
                                    }
                                }
                            }
                        }
                    }
                }
            },
        ),
        "regime_cal_move_vol_loose": _merge(
            base,
            {
                "llm": {
                    "cot_rf": {
                        "online_memory_policy": {
                            "gate": {
                                "learned": {
                                    "regime_calibration": {
                                        "enabled": True,
                                        "metric": "mse_path",
                                        "min_bucket_rows": 8,
                                        "max_abs_base_h20_pct": 3.0,
                                        "max_abs_base_h30_pct": 5.0,
                                        "use_profile_vol": True,
                                        "max_profile_vol_pct": 4.0,
                                    }
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

    gate_dir = run_dir / "results" / "online_memory_gate" / METHOD_NAME
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

    regime_path = gate_dir / "val_learned_gate_regime_thresholds.csv"
    if regime_path.exists():
        regime_df = pd.read_csv(regime_path)
        out["regime_thresholds"] = json.dumps(
            {
                str(row["regime_bucket"]): float(row["selected_threshold"])
                for _, row in regime_df.iterrows()
            },
            sort_keys=True,
        )
    else:
        out["regime_thresholds"] = ""
    return out


def _seed_baseline_rows() -> list[dict[str, Any]]:
    if not BASELINE_RESULTS_PATH.exists():
        return []
    df = pd.read_csv(BASELINE_RESULTS_PATH)
    df = df[df["candidate"] == "gate_top20"].copy()
    if df.empty:
        return []
    rows: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        run_dir = Path(str(row["run_dir"]))
        extra = _extract_metrics(run_dir)
        rows.append(
            {
                "candidate": "gate_top20_baseline",
                "window": str(row["window"]),
                "train_end": str(row["train_end"]),
                "val_end": str(row["val_end"]),
                "test_end": str(row["test_end"]),
                "run_dir": str(run_dir),
                **extra,
            }
        )
    return rows


def _write_plan(out_dir: Path) -> None:
    lines = [
        "# W1-W4 Regime-Calibrated Gate Benchmark Plan",
        "",
        f"Generated: {datetime.now().isoformat()}",
        "",
        "## Objective",
        "",
        "Test whether the current `top20` learned gate improves on hard historical windows when the apply threshold is calibrated by simple regime bucket rather than kept global.",
        "",
        "Frozen stack:",
        "- UK-specific data root: `uk_ets/Data_auto_uk`",
        "- Regularized shared DLinear base: `lr=0.001`, `epochs=40`, `patience=8`, `dropout=0.5`, `weight_decay=0.02`",
        "- Same 4B LLM refinement stack",
        "- Same `top20` learned-gate features",
        "",
        "## Benchmark windows",
        "",
        "- `W1`: 2024-10-27 to 2025-06-30",
        "- `W2`: 2024-02-23 to 2024-10-26",
        "- `W3`: 2023-06-21 to 2024-02-22",
        "- `W4`: 2022-10-17 to 2023-06-20",
        "",
        "## Candidates",
        "",
        "- `gate_top20_baseline`: current global-threshold learned gate",
        "- `regime_cal_move_default`: moderate/large move buckets at `h20=2.5%`, `h30=4.0%`",
        "- `regime_cal_move_loose`: same idea with looser `3.0%` / `5.0%` caps",
        "- `regime_cal_move_tight`: tighter `2.0%` / `3.0%` caps",
        "- `regime_cal_move_vol_default`: move buckets plus `profile_vol_pct > 3.8` split",
        "- `regime_cal_move_vol_loose`: move buckets plus a looser volatility split",
        "",
        "## Expected outcome",
        "",
        "Hard safe-regime blocking already failed, so the expected gain here is modest rather than dramatic.",
        "",
        "Expected ranking:",
        "- Best candidate likely `regime_cal_move_default` or `regime_cal_move_vol_default`",
        "- Expected mean `W1-W4` path-MSE lift versus `gate_top20_baseline`: roughly `+0.1%` to `+0.8%`",
        "- Success condition: improve mean `llm_path_mse` below the current baseline `54.613995` without collapsing the only clearly positive historical window (`W3`)",
        "",
        "## Interpretation rule",
        "",
        "If none of the regime-calibrated variants beat the current baseline, then the next bottleneck is not threshold calibration and we should move on to a stronger regime/uplift model rather than more threshold engineering.",
    ]
    (out_dir / "plan.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_summary(out_dir: Path, df: pd.DataFrame) -> None:
    mean_cols = [
        "tsm_path_mse",
        "llm_path_mse",
        "path_improvement_pct",
        "h20_improvement_pct",
        "h30_improvement_pct",
        "apply_rate",
        "mean_apply_probability",
    ]
    mean_df = df.groupby("candidate")[mean_cols].mean().sort_values("llm_path_mse").reset_index()
    lines = [
        "# W1-W4 Regime-Calibrated Gate Benchmark",
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
        else (PROJECT_ROOT / "reports" / "uk_ets_regime_calibrated_gate_w1_w4" / datetime.now().strftime("%Y%m%d_%H%M%S")).resolve()
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
            run_dir = _run_candidate(window, overrides)
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
