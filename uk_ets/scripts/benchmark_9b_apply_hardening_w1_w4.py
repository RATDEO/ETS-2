#!/usr/bin/env python3
"""Benchmark 9B apply-stage hardening variants on W1-W4."""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/xdg")

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
MODEL_9B = "qwen3.5-9b-ud-q4-k-xl"
BASELINE_9B_RESULTS = (
    PROJECT_ROOT
    / "reports"
    / "uk_ets_effective_retrieval_w1_w4_huber_energy_9b"
    / "20260322_002454"
    / "results.csv"
)


@dataclass(frozen=True)
class WindowSpec:
    name: str
    train_end: str
    val_end: str
    test_end: str


@dataclass(frozen=True)
class Candidate:
    name: str
    change_summary: str
    aim: str
    expected_low: float
    expected_high: float
    llm_overrides: dict[str, Any]


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
                "strict_json_response_format": True,
                "enable_numeric_tool": True,
                "force_numeric_tool": True,
                "enable_delta_verifier_tool": True,
                "force_delta_verifier_tool": True,
                "max_tool_rounds": 4,
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


def _extract_apply_stats(run_dir: Path) -> dict[str, float]:
    log_path = run_dir / "llm" / "logs" / "llm_calls.jsonl"
    apply_total = 0
    apply_fail = 0
    reflect_total = 0
    reflect_fail = 0
    tool_counts: Counter[str] = Counter()
    if log_path.exists():
        with open(log_path, "r", encoding="utf-8") as handle:
            for line in handle:
                obj = json.loads(line)
                method = str(obj.get("method", ""))
                meta = obj.get("metadata") or {}
                if method.endswith(":apply"):
                    apply_total += 1
                    if not bool(meta.get("success", False)):
                        apply_fail += 1
                    tool_counts.update(meta.get("tool_calls") or [])
                elif method.endswith(":reflect"):
                    reflect_total += 1
                    if not bool(meta.get("success", False)):
                        reflect_fail += 1
    return {
        "apply_total": float(apply_total),
        "apply_fail": float(apply_fail),
        "apply_fail_rate": (float(apply_fail) / float(apply_total)) if apply_total else 0.0,
        "reflect_total": float(reflect_total),
        "reflect_fail": float(reflect_fail),
        "numeric_tool_calls": float(tool_counts.get("get_numeric_analysis", 0)),
        "verifier_tool_calls": float(tool_counts.get("verify_hdelta_adjustments", 0)),
    }


def _run_window(window: WindowSpec, candidate: Candidate) -> Path:
    overrides = _merge(
        _base_overrides(),
        candidate.llm_overrides,
    )
    overrides = _merge(
        overrides,
        {
            "split": {
                "train_end": window.train_end,
                "val_end": window.val_end,
                "test_end": window.test_end,
            }
        },
    )
    return Path(run_experiment(config_path=CONFIG_PATH, overrides=overrides, data_dir=DATA_DIR)).resolve()


def _write_plan(out_dir: Path, candidates: list[Candidate]) -> None:
    lines = [
        "# 9B Apply Hardening Sweep Plan",
        "",
        f"Generated: {datetime.now().isoformat()}",
        "",
        "## Objective",
        "- Keep the improved Huber + energy base and wide8 retrieval fixed.",
        "- Harden only the 9B apply stage, because the completed 9B benchmark showed apply-loop failures but no endpoint/schema transport failures.",
        "",
        "## Baseline",
        "- Completed 9B wide8 report: `reports/uk_ets_effective_retrieval_w1_w4_huber_energy_9b/20260322_002454/results.csv`.",
        "",
        "## Candidates",
        "",
        "| Candidate | Aim | Expected uplift vs 9B baseline | Change |",
        "|---|---|---:|---|",
    ]
    for c in candidates:
        lines.append(
            f"| {c.name} | {c.aim} | {c.expected_low:.2%} to {c.expected_high:.2%} | {c.change_summary} |"
        )
    (out_dir / "plan.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_summary(out_dir: Path, df: pd.DataFrame) -> None:
    baseline = pd.read_csv(BASELINE_9B_RESULTS)
    baseline_means = baseline[["tsm_path_mse", "llm_path_mse", "path_improvement_pct"]].mean().to_dict()
    summary_rows = []
    for candidate, group in df.groupby("candidate"):
        mean_row = group[["tsm_path_mse", "llm_path_mse", "path_improvement_pct", "apply_fail_rate", "numeric_tool_calls", "verifier_tool_calls"]].mean().to_dict()
        mean_row["candidate"] = candidate
        mean_row["delta_vs_baseline_9b"] = float(mean_row["llm_path_mse"] - baseline_means["llm_path_mse"])
        summary_rows.append(mean_row)
    summary_df = pd.DataFrame(summary_rows).sort_values("llm_path_mse")
    lines = [
        "# 9B Apply Hardening Sweep Summary",
        "",
        f"Generated: {datetime.now().isoformat()}",
        "",
        "## By Window",
        "",
        df.to_markdown(index=False),
        "",
        "## Mean Comparison",
        "",
        summary_df.to_markdown(index=False),
    ]
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=str, default="")
    args = parser.parse_args()
    out_dir = (
        Path(args.out_dir).expanduser().resolve()
        if args.out_dir
        else (PROJECT_ROOT / "reports" / "uk_ets_9b_apply_hardening_w1_w4" / datetime.now().strftime("%Y%m%d_%H%M%S")).resolve()
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    candidates = [
        Candidate(
            name="optional_verifier",
            change_summary="Keep numeric tool forced but make the verifier optional; retain wide8 retrieval and strict JSON.",
            aim="Reduce apply-loop dead-ends caused by repeated verifier chaining while preserving numeric grounding.",
            expected_low=0.001,
            expected_high=0.006,
            llm_overrides={
                "llm": {
                    "cot_rf": {
                        "force_delta_verifier_tool": False,
                    }
                }
            },
        ),
        Candidate(
            name="numeric_only_compact",
            change_summary="Disable the verifier and shrink the apply tool loop to numeric-only with max_tool_rounds=3.",
            aim="Test whether the 9B model performs better when the apply stage is reduced to one deterministic tool and a shorter loop.",
            expected_low=0.002,
            expected_high=0.010,
            llm_overrides={
                "llm": {
                    "cot_rf": {
                        "enable_delta_verifier_tool": False,
                        "force_delta_verifier_tool": False,
                        "max_tool_rounds": 3,
                    }
                }
            },
        ),
        Candidate(
            name="optional_both_compact",
            change_summary="Keep both tools enabled but make both optional and reduce max_tool_rounds to 3.",
            aim="Test whether the 9B model improves when it can skip tools entirely on easy cases without getting trapped in required tool loops.",
            expected_low=-0.002,
            expected_high=0.006,
            llm_overrides={
                "llm": {
                    "cot_rf": {
                        "force_numeric_tool": False,
                        "force_delta_verifier_tool": False,
                        "max_tool_rounds": 3,
                    }
                }
            },
        ),
    ]
    _write_plan(out_dir, candidates)

    rows: list[dict[str, Any]] = []
    for candidate in candidates:
        for window in WINDOWS:
            print(f"Running {candidate.name} {window.name}", flush=True)
            run_dir = _run_window(window, candidate)
            rows.append(
                {
                    "candidate": candidate.name,
                    "window": window.name,
                    "train_end": window.train_end,
                    "val_end": window.val_end,
                    "test_end": window.test_end,
                    "run_dir": str(run_dir),
                    **_extract_metrics(run_dir),
                    **_extract_apply_stats(run_dir),
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
