#!/usr/bin/env python3
"""Run multi-base online-memory LLM sweeps for UK ETS and write a detailed report."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]

import sys

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.run_experiment import llm_result_name, run_experiment


METHOD_NAME = "TSM+LLM-COT-RF-HDELTA"
HORIZONS = (1, 5, 20, 30)


@dataclass(frozen=True)
class Candidate:
    name: str
    endpoint: str
    config_path: str
    base_model: str
    objective: str
    expectation: str
    overrides: dict[str, Any]


def _online_policy(
    *,
    support_examples: int,
    positive_examples: int,
    negative_examples: int,
    max_total_examples: int,
    positive_margin: float,
    negative_margin: float,
    warmup_min_realized: int,
    gate: dict[str, Any],
) -> dict[str, Any]:
    return {
        "enabled": True,
        "support_examples": int(support_examples),
        "positive_examples": int(positive_examples),
        "negative_examples": int(negative_examples),
        "max_total_examples": int(max_total_examples),
        "positive_margin": float(positive_margin),
        "negative_margin": float(negative_margin),
        "warmup_min_realized": int(warmup_min_realized),
        "gate": dict(gate),
    }


def _candidate_definitions() -> list[Candidate]:
    common_4b = "uk_ets/config/uk_ets_llm_4b_current_default.yaml"
    common_35b = "uk_ets/config/uk_ets_llm_35b_current_default.yaml"
    candidates: list[Candidate] = [
        Candidate(
            name="4b_tsm_verifier_conservative",
            endpoint="4b",
            config_path=common_4b,
            base_model="tsm",
            objective="Recover the late-horizon TSM edge in online memory while preventing noisy recent cases from entering the bank.",
            expectation="Expected path MSE: 24.90-25.35. Expected gain vs base TSM: 1.1%-2.9%.",
            overrides={
                "llm": {
                    "base_model": "tsm",
                    "cot_rf": {
                        "test_pool_mode": "online_realized_memory",
                        "realized_memory_horizon": 30,
                        "enable_numeric_tool": True,
                        "force_numeric_tool": True,
                        "enable_delta_verifier_tool": True,
                        "force_delta_verifier_tool": True,
                        "online_memory_policy": _online_policy(
                            support_examples=3,
                            positive_examples=2,
                            negative_examples=1,
                            max_total_examples=6,
                            positive_margin=0.015,
                            negative_margin=-0.015,
                            warmup_min_realized=5,
                            gate={
                                "min_positive_examples": 1,
                                "min_positive_signal": 0.015,
                                "min_net_signal": 0.005,
                                "max_negative_signal": 0.06,
                                "min_abs_base_h20_pct": 0.20,
                            },
                        ),
                    },
                },
            },
        ),
        Candidate(
            name="4b_tsm_numeric_only_conservative",
            endpoint="4b",
            config_path=common_4b,
            base_model="tsm",
            objective="Test whether the verifier is over-regularizing online TSM memory and whether numeric-only corrections carry the same signal with less clipping.",
            expectation="Expected path MSE: 25.00-25.45. Expected gain vs base TSM: 0.7%-2.5%.",
            overrides={
                "llm": {
                    "base_model": "tsm",
                    "cot_rf": {
                        "test_pool_mode": "online_realized_memory",
                        "realized_memory_horizon": 30,
                        "enable_numeric_tool": True,
                        "force_numeric_tool": True,
                        "enable_delta_verifier_tool": False,
                        "force_delta_verifier_tool": False,
                        "online_memory_policy": _online_policy(
                            support_examples=3,
                            positive_examples=2,
                            negative_examples=1,
                            max_total_examples=6,
                            positive_margin=0.015,
                            negative_margin=-0.015,
                            warmup_min_realized=5,
                            gate={
                                "min_positive_examples": 1,
                                "min_positive_signal": 0.015,
                                "min_net_signal": 0.005,
                                "max_negative_signal": 0.07,
                                "min_abs_base_h20_pct": 0.20,
                            },
                        ),
                    },
                },
            },
        ),
        Candidate(
            name="4b_linear_ridge_numeric_only",
            endpoint="4b",
            config_path=common_4b,
            base_model="linear_ridge",
            objective="Exploit ridge's already-stable shape with long-horizon numeric correction and a cautious online memory gate.",
            expectation="Expected path MSE: 22.05-22.35. Expected gain vs base ridge: 0.6%-1.9%.",
            overrides={
                "llm": {
                    "base_model": "linear_ridge",
                    "cot_rf": {
                        "test_pool_mode": "online_realized_memory",
                        "realized_memory_horizon": 30,
                        "enable_numeric_tool": True,
                        "force_numeric_tool": True,
                        "enable_delta_verifier_tool": False,
                        "force_delta_verifier_tool": False,
                        "online_memory_policy": _online_policy(
                            support_examples=3,
                            positive_examples=2,
                            negative_examples=1,
                            max_total_examples=6,
                            positive_margin=0.010,
                            negative_margin=-0.010,
                            warmup_min_realized=4,
                            gate={
                                "min_positive_examples": 1,
                                "min_positive_signal": 0.010,
                                "min_net_signal": 0.002,
                                "max_negative_signal": 0.05,
                                "min_abs_base_h20_pct": 0.10,
                            },
                        ),
                    },
                },
            },
        ),
        Candidate(
            name="4b_linear_ridge_numeric_verifier",
            endpoint="4b",
            config_path=common_4b,
            base_model="linear_ridge",
            objective="Test whether ridge benefits from extra verifier clipping once online memory selects only historically helpful regimes.",
            expectation="Expected path MSE: 22.10-22.40. Expected gain vs base ridge: 0.4%-1.7%.",
            overrides={
                "llm": {
                    "base_model": "linear_ridge",
                    "cot_rf": {
                        "test_pool_mode": "online_realized_memory",
                        "realized_memory_horizon": 30,
                        "enable_numeric_tool": True,
                        "force_numeric_tool": True,
                        "enable_delta_verifier_tool": True,
                        "force_delta_verifier_tool": True,
                        "online_memory_policy": _online_policy(
                            support_examples=3,
                            positive_examples=2,
                            negative_examples=1,
                            max_total_examples=6,
                            positive_margin=0.010,
                            negative_margin=-0.010,
                            warmup_min_realized=4,
                            gate={
                                "min_positive_examples": 1,
                                "min_positive_signal": 0.012,
                                "min_net_signal": 0.002,
                                "max_negative_signal": 0.05,
                                "min_abs_base_h20_pct": 0.10,
                            },
                        ),
                    },
                },
            },
        ),
        Candidate(
            name="4b_linear_lasso_numeric_only",
            endpoint="4b",
            config_path=common_4b,
            base_model="linear_lasso",
            objective="Use a stricter online gate to protect the noisier lasso base while still letting the LLM repair large long-horizon miss cases.",
            expectation="Expected path MSE: 23.40-23.95. Expected gain vs base lasso: 1.2%-3.5%.",
            overrides={
                "llm": {
                    "base_model": "linear_lasso",
                    "cot_rf": {
                        "test_pool_mode": "online_realized_memory",
                        "realized_memory_horizon": 30,
                        "enable_numeric_tool": True,
                        "force_numeric_tool": True,
                        "enable_delta_verifier_tool": False,
                        "force_delta_verifier_tool": False,
                        "online_memory_policy": _online_policy(
                            support_examples=3,
                            positive_examples=2,
                            negative_examples=1,
                            max_total_examples=6,
                            positive_margin=0.015,
                            negative_margin=-0.015,
                            warmup_min_realized=5,
                            gate={
                                "min_positive_examples": 1,
                                "min_positive_signal": 0.015,
                                "min_net_signal": 0.005,
                                "max_negative_signal": 0.05,
                                "min_abs_base_h20_pct": 0.20,
                            },
                        ),
                    },
                },
            },
        ),
        Candidate(
            name="4b_linear_lasso_numeric_verifier",
            endpoint="4b",
            config_path=common_4b,
            base_model="linear_lasso",
            objective="Check whether verifier clipping stabilizes lasso enough to turn noisy residuals into a cleaner online correction pathway.",
            expectation="Expected path MSE: 23.30-23.90. Expected gain vs base lasso: 1.4%-3.9%.",
            overrides={
                "llm": {
                    "base_model": "linear_lasso",
                    "cot_rf": {
                        "test_pool_mode": "online_realized_memory",
                        "realized_memory_horizon": 30,
                        "enable_numeric_tool": True,
                        "force_numeric_tool": True,
                        "enable_delta_verifier_tool": True,
                        "force_delta_verifier_tool": True,
                        "online_memory_policy": _online_policy(
                            support_examples=3,
                            positive_examples=2,
                            negative_examples=1,
                            max_total_examples=6,
                            positive_margin=0.015,
                            negative_margin=-0.015,
                            warmup_min_realized=5,
                            gate={
                                "min_positive_examples": 1,
                                "min_positive_signal": 0.016,
                                "min_net_signal": 0.006,
                                "max_negative_signal": 0.05,
                                "min_abs_base_h20_pct": 0.20,
                            },
                        ),
                    },
                },
            },
        ),
        Candidate(
            name="4b_naive_persistence_relaxed",
            endpoint="4b",
            config_path=common_4b,
            base_model="naive_persistence",
            objective="See whether online memory can salvage a weak naive base by only applying LLM corrections on historically large profitable long-horizon miss cases.",
            expectation="Expected path MSE: 29.50-31.20. Expected gain vs naive: 2.7%-8.0%.",
            overrides={
                "llm": {
                    "base_model": "naive_persistence",
                    "cot_rf": {
                        "test_pool_mode": "online_realized_memory",
                        "realized_memory_horizon": 30,
                        "enable_numeric_tool": True,
                        "force_numeric_tool": True,
                        "enable_delta_verifier_tool": True,
                        "force_delta_verifier_tool": True,
                        "online_memory_policy": _online_policy(
                            support_examples=4,
                            positive_examples=2,
                            negative_examples=1,
                            max_total_examples=7,
                            positive_margin=0.020,
                            negative_margin=-0.015,
                            warmup_min_realized=5,
                            gate={
                                "min_positive_examples": 1,
                                "min_positive_signal": 0.015,
                                "min_net_signal": 0.005,
                                "max_negative_signal": 0.06,
                                "min_abs_base_h20_pct": 0.35,
                            },
                        ),
                    },
                },
            },
        ),
        Candidate(
            name="4b_seasonal_naive_relaxed",
            endpoint="4b",
            config_path=common_4b,
            base_model="seasonal_naive",
            objective="Test whether selective online memory can convert seasonal naive into a viable long-horizon correction target without overtrading.",
            expectation="Expected path MSE: 31.00-33.20. Expected gain vs seasonal naive: 3.0%-9.5%.",
            overrides={
                "llm": {
                    "base_model": "seasonal_naive",
                    "cot_rf": {
                        "test_pool_mode": "online_realized_memory",
                        "realized_memory_horizon": 30,
                        "enable_numeric_tool": True,
                        "force_numeric_tool": True,
                        "enable_delta_verifier_tool": True,
                        "force_delta_verifier_tool": True,
                        "online_memory_policy": _online_policy(
                            support_examples=4,
                            positive_examples=2,
                            negative_examples=1,
                            max_total_examples=7,
                            positive_margin=0.020,
                            negative_margin=-0.015,
                            warmup_min_realized=5,
                            gate={
                                "min_positive_examples": 1,
                                "min_positive_signal": 0.015,
                                "min_net_signal": 0.005,
                                "max_negative_signal": 0.06,
                                "min_abs_base_h20_pct": 0.35,
                            },
                        ),
                    },
                },
            },
        ),
        Candidate(
            name="35b_tsm_numeric_only_online",
            endpoint="35b",
            config_path=common_35b,
            base_model="tsm",
            objective="Port the curated online memory policy to the 35B non-thinking endpoint while keeping the tool stack lightweight.",
            expectation="Expected path MSE: 24.40-25.00. Expected gain vs base TSM: 2.5%-4.8%.",
            overrides={
                "llm": {
                    "base_model": "tsm",
                    "cot_rf": {
                        "test_pool_mode": "online_realized_memory",
                        "realized_memory_horizon": 30,
                        "enable_numeric_tool": True,
                        "force_numeric_tool": True,
                        "enable_delta_verifier_tool": False,
                        "force_delta_verifier_tool": False,
                        "online_memory_policy": _online_policy(
                            support_examples=3,
                            positive_examples=2,
                            negative_examples=1,
                            max_total_examples=6,
                            positive_margin=0.012,
                            negative_margin=-0.012,
                            warmup_min_realized=5,
                            gate={
                                "min_positive_examples": 1,
                                "min_positive_signal": 0.012,
                                "min_net_signal": 0.004,
                                "max_negative_signal": 0.06,
                                "min_abs_base_h20_pct": 0.20,
                            },
                        ),
                    },
                },
            },
        ),
        Candidate(
            name="35b_linear_ridge_numeric_only_online",
            endpoint="35b",
            config_path=common_35b,
            base_model="linear_ridge",
            objective="Use 35B as a higher-capacity long-horizon residual corrector for the strongest linear baseline under online memory.",
            expectation="Expected path MSE: 21.85-22.20. Expected gain vs base ridge: 1.3%-2.8%.",
            overrides={
                "llm": {
                    "base_model": "linear_ridge",
                    "cot_rf": {
                        "test_pool_mode": "online_realized_memory",
                        "realized_memory_horizon": 30,
                        "enable_numeric_tool": True,
                        "force_numeric_tool": True,
                        "enable_delta_verifier_tool": False,
                        "force_delta_verifier_tool": False,
                        "online_memory_policy": _online_policy(
                            support_examples=3,
                            positive_examples=2,
                            negative_examples=1,
                            max_total_examples=6,
                            positive_margin=0.010,
                            negative_margin=-0.010,
                            warmup_min_realized=4,
                            gate={
                                "min_positive_examples": 1,
                                "min_positive_signal": 0.010,
                                "min_net_signal": 0.002,
                                "max_negative_signal": 0.05,
                                "min_abs_base_h20_pct": 0.10,
                            },
                        ),
                    },
                },
            },
        ),
        Candidate(
            name="35b_linear_lasso_numeric_only_online",
            endpoint="35b",
            config_path=common_35b,
            base_model="linear_lasso",
            objective="See whether 35B can stabilize lasso's more volatile path enough for real online improvement without the verifier.",
            expectation="Expected path MSE: 23.00-23.70. Expected gain vs base lasso: 2.2%-5.1%.",
            overrides={
                "llm": {
                    "base_model": "linear_lasso",
                    "cot_rf": {
                        "test_pool_mode": "online_realized_memory",
                        "realized_memory_horizon": 30,
                        "enable_numeric_tool": True,
                        "force_numeric_tool": True,
                        "enable_delta_verifier_tool": False,
                        "force_delta_verifier_tool": False,
                        "online_memory_policy": _online_policy(
                            support_examples=3,
                            positive_examples=2,
                            negative_examples=1,
                            max_total_examples=6,
                            positive_margin=0.012,
                            negative_margin=-0.012,
                            warmup_min_realized=5,
                            gate={
                                "min_positive_examples": 1,
                                "min_positive_signal": 0.012,
                                "min_net_signal": 0.004,
                                "max_negative_signal": 0.05,
                                "min_abs_base_h20_pct": 0.20,
                            },
                        ),
                    },
                },
            },
        ),
        Candidate(
            name="35b_naive_persistence_relaxed_online",
            endpoint="35b",
            config_path=common_35b,
            base_model="naive_persistence",
            objective="Use 35B only on strong positive-memory long-horizon setups to see how much of the naive baseline can be repaired online.",
            expectation="Expected path MSE: 28.50-30.50. Expected gain vs naive: 4.8%-11.1%.",
            overrides={
                "llm": {
                    "base_model": "naive_persistence",
                    "cot_rf": {
                        "test_pool_mode": "online_realized_memory",
                        "realized_memory_horizon": 30,
                        "enable_numeric_tool": True,
                        "force_numeric_tool": True,
                        "enable_delta_verifier_tool": False,
                        "force_delta_verifier_tool": False,
                        "online_memory_policy": _online_policy(
                            support_examples=4,
                            positive_examples=2,
                            negative_examples=1,
                            max_total_examples=7,
                            positive_margin=0.015,
                            negative_margin=-0.012,
                            warmup_min_realized=5,
                            gate={
                                "min_positive_examples": 1,
                                "min_positive_signal": 0.015,
                                "min_net_signal": 0.005,
                                "max_negative_signal": 0.06,
                                "min_abs_base_h20_pct": 0.35,
                            },
                        ),
                    },
                },
            },
        ),
        Candidate(
            name="35b_seasonal_naive_relaxed_online",
            endpoint="35b",
            config_path=common_35b,
            base_model="seasonal_naive",
            objective="Test whether 35B can turn seasonal naive into a selective online correction target when only the most helpful memories are admitted.",
            expectation="Expected path MSE: 29.80-31.90. Expected gain vs seasonal naive: 6.8%-13.0%.",
            overrides={
                "llm": {
                    "base_model": "seasonal_naive",
                    "cot_rf": {
                        "test_pool_mode": "online_realized_memory",
                        "realized_memory_horizon": 30,
                        "enable_numeric_tool": True,
                        "force_numeric_tool": True,
                        "enable_delta_verifier_tool": False,
                        "force_delta_verifier_tool": False,
                        "online_memory_policy": _online_policy(
                            support_examples=4,
                            positive_examples=2,
                            negative_examples=1,
                            max_total_examples=7,
                            positive_margin=0.015,
                            negative_margin=-0.012,
                            warmup_min_realized=5,
                            gate={
                                "min_positive_examples": 1,
                                "min_positive_signal": 0.015,
                                "min_net_signal": 0.005,
                                "max_negative_signal": 0.06,
                                "min_abs_base_h20_pct": 0.35,
                            },
                        ),
                    },
                },
            },
        ),
    ]
    return candidates


def _count_llm_stage_calls(run_dir: Path, result_name: str) -> tuple[int, int]:
    log_path = run_dir / "llm" / "logs" / "llm_calls.jsonl"
    if not log_path.exists():
        return 0, 0
    reflect_calls = 0
    apply_calls = 0
    prefix_reflect = f"{METHOD_NAME}:reflect"
    prefix_apply = f"{METHOD_NAME}:apply"
    with log_path.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            method = str(payload.get("method", ""))
            if method == prefix_reflect:
                reflect_calls += 1
            elif method == prefix_apply:
                apply_calls += 1
    return reflect_calls, apply_calls


def _load_metric_row(df: pd.DataFrame, model_name: str) -> pd.Series:
    match = df.loc[df["model"] == model_name]
    if match.empty:
        raise KeyError(f"Missing model row: {model_name}")
    return match.iloc[0]


def _extract_run_metrics(candidate: Candidate, run_dir: Path) -> dict[str, Any]:
    path_df = pd.read_csv(run_dir / "results" / "path_metrics.csv")
    horizon_df = pd.read_csv(run_dir / "results" / "metrics_by_horizon.csv")
    result_name = llm_result_name(METHOD_NAME, candidate.base_model)

    base_row = _load_metric_row(path_df, candidate.base_model)
    llm_row = _load_metric_row(path_df, result_name)
    reflect_calls, apply_calls = _count_llm_stage_calls(run_dir, result_name)

    row: dict[str, Any] = {
        "candidate": candidate.name,
        "endpoint": candidate.endpoint,
        "base_model": candidate.base_model,
        "run_dir": str(run_dir),
        "result_name": result_name,
        "base_mse_path": float(base_row["mse_path"]),
        "llm_mse_path": float(llm_row["mse_path"]),
        "path_gain_abs": float(base_row["mse_path"]) - float(llm_row["mse_path"]),
        "path_gain_pct": (
            (float(base_row["mse_path"]) - float(llm_row["mse_path"]))
            / max(float(base_row["mse_path"]), 1e-8)
            * 100.0
        ),
        "reflect_calls": int(reflect_calls),
        "apply_calls": int(apply_calls),
        "approx_llm_applied_samples": int(apply_calls),
    }
    for horizon in HORIZONS:
        base_h = _load_metric_row(
            horizon_df.loc[horizon_df["horizon"] == horizon],
            candidate.base_model,
        )
        llm_h = _load_metric_row(
            horizon_df.loc[horizon_df["horizon"] == horizon],
            result_name,
        )
        row[f"base_h{horizon}_mse"] = float(base_h["mse"])
        row[f"llm_h{horizon}_mse"] = float(llm_h["mse"])
        row[f"gain_h{horizon}_abs"] = float(base_h["mse"]) - float(llm_h["mse"])
        row[f"gain_h{horizon}_pct"] = (
            (float(base_h["mse"]) - float(llm_h["mse"]))
            / max(float(base_h["mse"]), 1e-8)
            * 100.0
        )
    return row


def _write_markdown_report(
    out_dir: Path,
    candidates: list[Candidate],
    results_df: pd.DataFrame,
) -> None:
    by_candidate = {c.name: c for c in candidates}
    best_rows = (
        results_df.sort_values(["endpoint", "base_model", "llm_mse_path"])
        .groupby(["endpoint", "base_model"], as_index=False)
        .first()
    )

    lines: list[str] = [
        "# UK ETS Online-Memory Multi-Base LLM Sweep",
        "",
        f"Generated: {datetime.now().isoformat()}",
        "",
        "## Objectives",
        "",
        "1. Convert the new offline helpfulness concept into a deployable online-memory policy.",
        "2. Evaluate whether that policy improves the production-style online LLM refinement path, not just the frozen-memory benchmark.",
        "3. Tailor the LLM refinement path across all current baseline forecasters, not only TSM.",
        "4. Measure absolute and relative lift over each base model on the full 144-window UK holdout.",
        "",
        "## Method",
        "",
        "- Each run used `test_pool_mode=online_realized_memory`.",
        "- Only fully realized prior cases were eligible for online memory admission.",
        "- A realized case entered memory only if its ex-post LLM helpfulness score met the configured positive or negative threshold.",
        "- Positive and negative memories were stored separately and retrieved independently.",
        "- A deterministic live gate decided whether to apply the LLM using only pre-decision information plus admitted memory similarity/consensus.",
        "- Full 144-window holdout runs were used for all candidates.",
        "",
        "## Planned Candidates",
        "",
        "| Candidate | Endpoint | Base | Objective | Expected Result |",
        "|---|---|---|---|---|",
    ]
    for candidate in candidates:
        lines.append(
            f"| `{candidate.name}` | `{candidate.endpoint}` | `{candidate.base_model}` | {candidate.objective} | {candidate.expectation} |"
        )

    lines.extend(
        [
            "",
            "## Results",
            "",
            "| Candidate | Base | Endpoint | Base path MSE | LLM path MSE | Abs gain | % gain | h5 % | h20 % | h30 % | Approx applied |",
            "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for _, row in results_df.sort_values(["endpoint", "base_model", "llm_mse_path"]).iterrows():
        lines.append(
            "| `{candidate}` | `{base}` | `{endpoint}` | {base_mse:.6f} | {llm_mse:.6f} | {gain_abs:+.6f} | {gain_pct:+.2f}% | {h5:+.2f}% | {h20:+.2f}% | {h30:+.2f}% | {applied} |".format(
                candidate=row["candidate"],
                base=row["base_model"],
                endpoint=row["endpoint"],
                base_mse=row["base_mse_path"],
                llm_mse=row["llm_mse_path"],
                gain_abs=row["path_gain_abs"],
                gain_pct=row["path_gain_pct"],
                h5=row["gain_h5_pct"],
                h20=row["gain_h20_pct"],
                h30=row["gain_h30_pct"],
                applied=int(row["approx_llm_applied_samples"]),
            )
        )

    lines.extend(["", "## Best By Base", ""])
    for endpoint in ("4b", "35b"):
        subset = best_rows.loc[best_rows["endpoint"] == endpoint]
        if subset.empty:
            continue
        lines.append(f"### {endpoint.upper()}")
        lines.append("")
        for _, row in subset.iterrows():
            candidate = by_candidate[row["candidate"]]
            lines.append(f"- `{row['base_model']}`: `{row['candidate']}`")
            lines.append(
                "  base path MSE `{:.6f}` -> LLM `{:.6f}` ({:+.2f}%)".format(
                    row["base_mse_path"],
                    row["llm_mse_path"],
                    row["path_gain_pct"],
                )
            )
            lines.append(
                "  horizon gains: `h1 {:+.2f}%`, `h5 {:+.2f}%`, `h20 {:+.2f}%`, `h30 {:+.2f}%`".format(
                    row["gain_h1_pct"],
                    row["gain_h5_pct"],
                    row["gain_h20_pct"],
                    row["gain_h30_pct"],
                )
            )
            lines.append(f"  objective: {candidate.objective}")
            lines.append(f"  expected: {candidate.expectation}")
        lines.append("")

    lines.extend(
        [
            "## Interpretation",
            "",
            "- Positive path gain means the LLM improved the base model on the full holdout.",
            "- `Approx applied` is the number of apply-stage LLM calls; the remainder of the 144 windows were effectively gated to the base forecast.",
            "- This is a production-style single-holdout online-memory benchmark, not the stricter frozen multi-fold scientific benchmark.",
            "",
        ]
    )

    (out_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = PROJECT_ROOT / "reports" / "uk_ets_online_memory_multi_base" / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)

    candidates = _candidate_definitions()
    planned_rows = [
        {
            "candidate": c.name,
            "endpoint": c.endpoint,
            "config_path": c.config_path,
            "base_model": c.base_model,
            "objective": c.objective,
            "expectation": c.expectation,
            "overrides_json": json.dumps(c.overrides, sort_keys=True),
        }
        for c in candidates
    ]
    pd.DataFrame(planned_rows).to_csv(out_dir / "planned_candidates.csv", index=False)

    plan_lines = [
        "# Planned UK ETS Online-Memory Multi-Base Sweep",
        "",
        f"Generated: {datetime.now().isoformat()}",
        "",
        "| Candidate | Endpoint | Base | Objective | Expected Result |",
        "|---|---|---|---|---|",
    ]
    for c in candidates:
        plan_lines.append(
            f"| `{c.name}` | `{c.endpoint}` | `{c.base_model}` | {c.objective} | {c.expectation} |"
        )
    (out_dir / "plan.md").write_text("\n".join(plan_lines), encoding="utf-8")

    results: list[dict[str, Any]] = []
    for candidate in candidates:
        run_dir = Path(
            run_experiment(
                config_path=candidate.config_path,
                overrides=candidate.overrides,
                data_dir="uk_ets/Data_auto_uk",
            )
        ).resolve()
        row = _extract_run_metrics(candidate, run_dir)
        results.append(row)
        pd.DataFrame(results).to_csv(out_dir / "candidate_results.csv", index=False)

    results_df = pd.DataFrame(results)
    results_df.to_csv(out_dir / "candidate_results.csv", index=False)
    best_rows = (
        results_df.sort_values(["endpoint", "base_model", "llm_mse_path"])
        .groupby(["endpoint", "base_model"], as_index=False)
        .first()
    )
    best_rows.to_csv(out_dir / "best_by_base.csv", index=False)
    _write_markdown_report(out_dir, candidates, results_df)
    print(out_dir)


if __name__ == "__main__":
    main()
