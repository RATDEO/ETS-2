#!/usr/bin/env python3
"""Run online-memory policy improvement sweeps for UK ETS and write a detailed report."""

from __future__ import annotations

import argparse
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
BASELINE_REFERENCE_PATH = (
    PROJECT_ROOT
    / "reports"
    / "uk_ets_online_memory_multi_base"
    / "20260315_175555"
    / "best_by_base.csv"
)


@dataclass(frozen=True)
class Candidate:
    name: str
    endpoint: str
    base_model: str
    phase: str
    objective: str
    expectation: str
    config_path: str
    overrides: dict[str, Any]


def _deep_merge(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    merged = dict(left)
    for key, value in right.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(dict(merged[key]), value)
        else:
            merged[key] = value
    return merged


def _baseline_policy(endpoint: str, base_model: str) -> dict[str, Any]:
    endpoint = str(endpoint)
    base_model = str(base_model)
    common: dict[str, Any] = {
        "llm": {
            "base_model": base_model,
            "cot_rf": {
                "test_pool_mode": "online_realized_memory",
                "realized_memory_horizon": 30,
            },
        }
    }
    if endpoint == "4b":
        gate_map = {
            "tsm": dict(min_positive_examples=1, min_positive_signal=0.015, min_net_signal=0.005, max_negative_signal=0.06, min_abs_base_h20_pct=0.20),
            "linear_ridge": dict(min_positive_examples=1, min_positive_signal=0.012, min_net_signal=0.002, max_negative_signal=0.05, min_abs_base_h20_pct=0.10),
            "linear_lasso": dict(min_positive_examples=1, min_positive_signal=0.016, min_net_signal=0.006, max_negative_signal=0.05, min_abs_base_h20_pct=0.20),
            "naive_persistence": dict(min_positive_examples=1, min_positive_signal=0.015, min_net_signal=0.005, max_negative_signal=0.06, min_abs_base_h20_pct=0.35),
            "seasonal_naive": dict(min_positive_examples=1, min_positive_signal=0.015, min_net_signal=0.005, max_negative_signal=0.06, min_abs_base_h20_pct=0.35),
        }
        margin_map = {
            "tsm": (0.015, -0.015, 5, 3, 2, 1, 6),
            "linear_ridge": (0.010, -0.010, 4, 3, 2, 1, 6),
            "linear_lasso": (0.015, -0.015, 5, 3, 2, 1, 6),
            "naive_persistence": (0.020, -0.015, 5, 4, 2, 1, 7),
            "seasonal_naive": (0.020, -0.015, 5, 4, 2, 1, 7),
        }
        pos_margin, neg_margin, warmup, support, pos_k, neg_k, max_total = margin_map[base_model]
        return _deep_merge(
            common,
            {
                "llm": {
                    "cot_rf": {
                        "enable_numeric_tool": True,
                        "force_numeric_tool": True,
                        "enable_delta_verifier_tool": True,
                        "force_delta_verifier_tool": True,
                        "online_memory_policy": {
                            "enabled": True,
                            "support_examples": support,
                            "positive_examples": pos_k,
                            "negative_examples": neg_k,
                            "max_total_examples": max_total,
                            "positive_margin": pos_margin,
                            "negative_margin": neg_margin,
                            "warmup_min_realized": warmup,
                            "gate": gate_map[base_model],
                        },
                    },
                },
            },
        )

    gate_map = {
        "tsm": dict(min_positive_examples=1, min_positive_signal=0.012, min_net_signal=0.004, max_negative_signal=0.06, min_abs_base_h20_pct=0.20),
        "linear_ridge": dict(min_positive_examples=1, min_positive_signal=0.010, min_net_signal=0.002, max_negative_signal=0.05, min_abs_base_h20_pct=0.10),
        "linear_lasso": dict(min_positive_examples=1, min_positive_signal=0.012, min_net_signal=0.004, max_negative_signal=0.05, min_abs_base_h20_pct=0.20),
        "naive_persistence": dict(min_positive_examples=1, min_positive_signal=0.015, min_net_signal=0.005, max_negative_signal=0.06, min_abs_base_h20_pct=0.35),
        "seasonal_naive": dict(min_positive_examples=1, min_positive_signal=0.015, min_net_signal=0.005, max_negative_signal=0.06, min_abs_base_h20_pct=0.35),
    }
    margin_map = {
        "tsm": (0.012, -0.012, 5, 3, 2, 1, 6),
        "linear_ridge": (0.010, -0.010, 4, 3, 2, 1, 6),
        "linear_lasso": (0.012, -0.012, 5, 3, 2, 1, 6),
        "naive_persistence": (0.015, -0.012, 5, 4, 2, 1, 7),
        "seasonal_naive": (0.015, -0.012, 5, 4, 2, 1, 7),
    }
    pos_margin, neg_margin, warmup, support, pos_k, neg_k, max_total = margin_map[base_model]
    return _deep_merge(
        common,
        {
            "llm": {
                "cot_rf": {
                    "enable_numeric_tool": True,
                    "force_numeric_tool": True,
                    "enable_delta_verifier_tool": False,
                    "force_delta_verifier_tool": False,
                    "online_memory_policy": {
                        "enabled": True,
                        "support_examples": support,
                        "positive_examples": pos_k,
                        "negative_examples": neg_k,
                        "max_total_examples": max_total,
                        "positive_margin": pos_margin,
                        "negative_margin": neg_margin,
                        "warmup_min_realized": warmup,
                        "gate": gate_map[base_model],
                    },
                },
            },
        },
    )


def _single_policy_variants() -> dict[str, dict[str, Any]]:
    return {
        "baseline": {},
        "learned_gate": {
            "llm": {
                "cot_rf": {
                    "online_memory_policy": {
                        "gate": {
                            "tune_on_val": {
                                "enabled": True,
                                "scope": "recent_tail",
                                "recent_tail_fraction": 0.5,
                                "recent_tail_min_samples": 80,
                                "max_samples": 80,
                                "metric": "mse_path",
                            },
                            "tuning_grid": {
                                "min_positive_examples": [1, 2],
                                "min_positive_signal": [0.008, 0.012, 0.016, 0.020],
                                "min_net_signal": [0.000, 0.003, 0.006],
                                "max_negative_signal": [0.03, 0.05, 0.07],
                                "min_abs_base_h20_pct": [0.10, 0.20, 0.30],
                            },
                        },
                    },
                },
            },
        },
        "horizon_specific": {
            "llm": {
                "cot_rf": {
                    "online_memory_policy": {
                        "horizon_specific": {
                            "enabled": True,
                            "admission_horizons": [20, 30],
                        },
                    },
                },
            },
        },
        "regime_specific": {
            "llm": {
                "cot_rf": {
                    "online_memory_policy": {
                        "regime": {
                            "enabled": True,
                            "min_overlap": 4,
                        },
                    },
                },
            },
        },
        "high_utility_admission": {
            "llm": {
                "cot_rf": {
                    "online_memory_policy": {
                        "admission": {
                            "positive_min_h20_gain": 0.010,
                            "positive_min_h30_gain": 0.010,
                            "positive_max_h5_damage": -0.150,
                            "negative_max_h20_gain": -0.050,
                            "negative_max_h30_gain": -0.050,
                        },
                    },
                },
            },
        },
        "prototype_compression": {
            "llm": {
                "cot_rf": {
                    "online_memory_policy": {
                        "prototype": {
                            "enabled": True,
                            "top_pool_size": 8,
                        },
                    },
                },
            },
        },
    }


def _tsm_candidates(endpoints: list[str]) -> list[Candidate]:
    variants = _single_policy_variants()
    endpoint_info = {
        "4b": {
            "config_path": "uk_ets/config/uk_ets_llm_4b_current_default.yaml",
            "expectations": {
                "baseline": "Expected path MSE near 26.65-26.80. This should reproduce the current live-online 4B TSM baseline.",
                "learned_gate": "Expected path MSE 26.45-26.68. Offline-tuned thresholds should reduce false positives and modestly lift live TSM performance.",
                "horizon_specific": "Expected path MSE 26.50-26.70. Earlier h20 memory availability should help long horizons without materially changing h1/h5.",
                "regime_specific": "Expected path MSE 26.50-26.72. Regime filtering should cut contradictory memories and slightly improve net signal quality.",
                "high_utility_admission": "Expected path MSE 26.45-26.70. Promoting only truly helpful cases should reduce noisy bank growth.",
                "prototype_compression": "Expected path MSE 26.50-26.72. Fewer but cleaner memories should suit 4B.",
            },
        },
        "35b": {
            "config_path": "uk_ets/config/uk_ets_llm_35b_current_default.yaml",
            "expectations": {
                "baseline": "Expected path MSE near 26.55-26.72. This should reproduce the current live-online 35B TSM baseline.",
                "learned_gate": "Expected path MSE 26.30-26.58. 35B should benefit more from a validation-fit live gate than 4B.",
                "horizon_specific": "Expected path MSE 26.35-26.60. Faster h20 admission should let 35B exploit long-horizon improvements earlier.",
                "regime_specific": "Expected path MSE 26.35-26.62. Regime filtering should help 35B avoid low-similarity memories.",
                "high_utility_admission": "Expected path MSE 26.30-26.60. Higher-capacity 35B should benefit from more selective positive-memory promotion.",
                "prototype_compression": "Expected path MSE 26.40-26.64. Compression may help less than on 4B but should still control prompt clutter.",
            },
        },
    }
    objective_map = {
        "baseline": "Reproduce the current live-online TSM baseline under the refactored runner.",
        "learned_gate": "Convert offline helpfulness into validation-fit threshold rules for live gating.",
        "horizon_specific": "Admit h20 memories earlier than h30 to make online adaptation faster and more target-aligned.",
        "regime_specific": "Only retrieve online memories from matching regime buckets.",
        "high_utility_admission": "Promote only cases that materially help long horizons while avoiding h5 damage.",
        "prototype_compression": "Compress raw memory banks into a smaller, more diverse case set.",
    }
    candidates: list[Candidate] = []
    for endpoint, meta in endpoint_info.items():
        if endpoint not in endpoints:
            continue
        for variant_name, variant_overrides in variants.items():
            candidates.append(
                Candidate(
                    name=f"{endpoint}_tsm_{variant_name}",
                    endpoint=endpoint,
                    base_model="tsm",
                    phase="single_policy",
                    objective=objective_map[variant_name],
                    expectation=meta["expectations"][variant_name],
                    config_path=meta["config_path"],
                    overrides=_deep_merge(
                        _baseline_policy(endpoint, "tsm"),
                        variant_overrides,
                    ),
                )
            )
    return candidates


def _load_previous_best_by_base() -> pd.DataFrame:
    return pd.read_csv(BASELINE_REFERENCE_PATH)


def _count_llm_stage_calls(run_dir: Path) -> tuple[int, int]:
    log_path = run_dir / "llm" / "logs" / "llm_calls.jsonl"
    if not log_path.exists():
        return 0, 0
    reflect_calls = 0
    apply_calls = 0
    with log_path.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            method = str(payload.get("method", ""))
            if method.endswith(":reflect"):
                reflect_calls += 1
            elif method.endswith(":apply"):
                apply_calls += 1
    return reflect_calls, apply_calls


def _load_metric_row(df: pd.DataFrame, model_name: str) -> pd.Series:
    match = df.loc[df["model"] == model_name]
    if match.empty:
        raise KeyError(f"Missing model row: {model_name}")
    return match.iloc[0]


def _gate_selection_payload(run_dir: Path, result_name: str) -> dict[str, Any] | None:
    result_slug = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in result_name)
    path = run_dir / "llm" / f"online_memory_gate_selection_{result_slug}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _extract_run_metrics(
    candidate: Candidate,
    run_dir: Path,
    previous_best_lookup: dict[tuple[str, str], dict[str, Any]],
) -> dict[str, Any]:
    path_df = pd.read_csv(run_dir / "results" / "path_metrics.csv")
    horizon_df = pd.read_csv(run_dir / "results" / "metrics_by_horizon.csv")
    result_name = llm_result_name(METHOD_NAME, candidate.base_model)

    base_row = _load_metric_row(path_df, candidate.base_model)
    llm_row = _load_metric_row(path_df, result_name)
    reflect_calls, apply_calls = _count_llm_stage_calls(run_dir)
    prev = previous_best_lookup.get((candidate.endpoint, candidate.base_model))
    gate_selection = _gate_selection_payload(run_dir, result_name)

    row: dict[str, Any] = {
        "candidate": candidate.name,
        "endpoint": candidate.endpoint,
        "base_model": candidate.base_model,
        "phase": candidate.phase,
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
        "prev_best_llm_mse_path": float(prev["llm_mse_path"]) if prev is not None else np.nan,
        "vs_prev_best_abs": (
            float(prev["llm_mse_path"]) - float(llm_row["mse_path"])
            if prev is not None
            else np.nan
        ),
        "vs_prev_best_pct": (
            (float(prev["llm_mse_path"]) - float(llm_row["mse_path"]))
            / max(float(prev["llm_mse_path"]), 1e-8)
            * 100.0
            if prev is not None
            else np.nan
        ),
        "gate_selection": json.dumps(gate_selection, sort_keys=True) if gate_selection else "",
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


def _run_candidate(candidate: Candidate, previous_best_lookup: dict[tuple[str, str], dict[str, Any]]) -> dict[str, Any]:
    run_dir = Path(run_experiment(config_path=candidate.config_path, overrides=candidate.overrides))
    row = _extract_run_metrics(candidate, run_dir, previous_best_lookup)
    row["objective"] = candidate.objective
    row["expectation"] = candidate.expectation
    return row


def _single_success_policies(single_rows: list[dict[str, Any]], endpoint: str) -> list[str]:
    endpoint_rows = [
        row
        for row in single_rows
        if row["endpoint"] == endpoint and row["phase"] == "single_policy"
    ]
    baseline = next(row for row in endpoint_rows if row["candidate"].endswith("_baseline"))
    success_names = []
    for row in endpoint_rows:
        if row["candidate"].endswith("_baseline"):
            continue
        if float(row["llm_mse_path"]) < float(baseline["llm_mse_path"]):
            success_names.append(row["candidate"].split("_", 2)[-1])
    return success_names


def _combined_policy_overrides(success_variant_names: list[str]) -> dict[str, Any]:
    variants = _single_policy_variants()
    combined: dict[str, Any] = {}
    for variant_name in success_variant_names:
        variant_key = variant_name.replace("tsm_", "") if variant_name.startswith("tsm_") else variant_name
        variant = variants.get(variant_key)
        if variant is None:
            continue
        combined = _deep_merge(combined, variant)
    return combined


def _multi_base_candidates(success_map: dict[str, list[str]], endpoints: list[str]) -> list[Candidate]:
    bases = ["tsm", "linear_ridge", "linear_lasso", "naive_persistence", "seasonal_naive"]
    config_map = {
        "4b": "uk_ets/config/uk_ets_llm_4b_current_default.yaml",
        "35b": "uk_ets/config/uk_ets_llm_35b_current_default.yaml",
    }
    candidates: list[Candidate] = []
    for endpoint in endpoints:
        combined_overrides = _combined_policy_overrides(success_map.get(endpoint, []))
        for base_model in bases:
            objective = (
                "Port the successful TSM online-memory policy improvements onto "
                f"{base_model} and measure whether the live LLM edge broadens beyond TSM."
            )
            expectation = (
                "Expected incremental gain vs the current live-online baseline: "
                "0.05%-0.80% for strong bases, potentially higher for naive baselines."
            )
            candidates.append(
                Candidate(
                    name=f"{endpoint}_{base_model}_combined",
                    endpoint=endpoint,
                    base_model=base_model,
                    phase="combined_multi_base",
                    objective=objective,
                    expectation=expectation,
                    config_path=config_map[endpoint],
                    overrides=_deep_merge(
                        _baseline_policy(endpoint, base_model),
                        combined_overrides,
                    ),
                )
            )
    return candidates


def _write_plan(out_dir: Path, single_candidates: list[Candidate]) -> None:
    lines = ["# Online Memory Policy Sweep Plan", ""]
    lines.append("## Objectives")
    lines.append("- Replace heuristic online gating with validation-fit threshold rules.")
    lines.append("- Speed online learning with horizon-specific admission.")
    lines.append("- Reduce contradictory memories with regime-specific retrieval.")
    lines.append("- Improve memory quality with utility-gated admission.")
    lines.append("- Control prompt dilution with prototype compression.")
    lines.append("- Combine only the successful single-policy changes and port them across bases.")
    lines.append("")
    lines.append("## Pre-Registered Singles")
    for candidate in single_candidates:
        lines.append(f"- `{candidate.name}`: {candidate.objective} {candidate.expectation}")
    (out_dir / "plan.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_report(
    out_dir: Path,
    single_rows: list[dict[str, Any]],
    combined_rows: list[dict[str, Any]],
    success_map: dict[str, list[str]],
) -> None:
    single_df = pd.DataFrame(single_rows)
    combined_df = pd.DataFrame(combined_rows)

    lines = ["# Online Memory Policy Improvement Report", ""]
    lines.append("## Single-Policy Results")
    for endpoint in sorted(set(single_df["endpoint"].tolist())):
        lines.append(f"### {endpoint.upper()} TSM")
        endpoint_rows = single_df.loc[single_df["endpoint"] == endpoint].sort_values("llm_mse_path")
        baseline_row = endpoint_rows.loc[endpoint_rows["candidate"].str.endswith("_baseline")].iloc[0]
        for _, row in endpoint_rows.iterrows():
            lines.append(
                f"- `{row['candidate']}`: path MSE `{row['llm_mse_path']:.6f}` vs baseline `{baseline_row['llm_mse_path']:.6f}`; "
                f"delta `{baseline_row['llm_mse_path'] - row['llm_mse_path']:+.6f}`; "
                f"applied `{int(row['approx_llm_applied_samples'])}` windows. "
                f"Expected: {row['expectation']}"
            )
        successful = success_map.get(endpoint, [])
        lines.append(
            f"- Successful singles promoted for `{endpoint}`: "
            + (", ".join(f"`{name}`" for name in successful) if successful else "`none`")
        )
        lines.append("")

    lines.append("## Combined Multi-Base Results")
    for endpoint in sorted(set(combined_df["endpoint"].tolist())):
        lines.append(f"### {endpoint.upper()} Combined Policy")
        endpoint_rows = combined_df.loc[combined_df["endpoint"] == endpoint].sort_values("llm_mse_path")
        for _, row in endpoint_rows.iterrows():
            lines.append(
                f"- `{row['base_model']}`: path MSE `{row['llm_mse_path']:.6f}`; gain vs base `{row['path_gain_pct']:.3f}%`; "
                f"vs previous live baseline `{row['vs_prev_best_pct']:+.3f}%`; "
                f"`h20` gain `{row['gain_h20_pct']:.3f}%`; `h30` gain `{row['gain_h30_pct']:.3f}%`."
            )
        lines.append("")

    lines.append("## Artifacts")
    lines.append(f"- Singles CSV: `{(out_dir / 'single_policy_results.csv').name}`")
    lines.append(f"- Combined CSV: `{(out_dir / 'combined_multi_base_results.csv').name}`")
    (out_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--endpoints",
        default="4b",
        help="Comma-separated endpoints to run, e.g. '4b' or '4b,35b'. Default: 4b",
    )
    args = parser.parse_args()
    endpoints = [item.strip() for item in str(args.endpoints).split(",") if item.strip()]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = PROJECT_ROOT / "reports" / "uk_ets_online_memory_policy_improvement" / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)

    previous_best_df = _load_previous_best_by_base()
    previous_best_lookup = {
        (str(row["endpoint"]), str(row["base_model"])): dict(row)
        for _, row in previous_best_df.iterrows()
    }

    single_candidates = _tsm_candidates(endpoints)
    _write_plan(out_dir, single_candidates)

    single_rows: list[dict[str, Any]] = []
    for candidate in single_candidates:
        single_rows.append(_run_candidate(candidate, previous_best_lookup))
        pd.DataFrame(single_rows).to_csv(out_dir / "single_policy_results.csv", index=False)

    success_map = {
        endpoint: _single_success_policies(single_rows, endpoint)
        for endpoint in endpoints
    }

    combined_candidates = _multi_base_candidates(success_map, endpoints)
    combined_rows: list[dict[str, Any]] = []
    for candidate in combined_candidates:
        combined_rows.append(_run_candidate(candidate, previous_best_lookup))
        pd.DataFrame(combined_rows).to_csv(out_dir / "combined_multi_base_results.csv", index=False)

    _write_report(out_dir, single_rows, combined_rows, success_map)


if __name__ == "__main__":
    main()
