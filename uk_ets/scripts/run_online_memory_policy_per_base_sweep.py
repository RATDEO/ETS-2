#!/usr/bin/env python3
"""Run per-base online-memory policy sweeps for UK ETS and write a detailed report."""

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

from src.run_experiment import llm_result_name, run_experiment


METHOD_NAME = "TSM+LLM-COT-RF-HDELTA"
HORIZONS = (1, 5, 20, 30)
BASE_MODELS = ["tsm", "linear_ridge", "linear_lasso", "naive_persistence", "seasonal_naive"]


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


def _policy_variants() -> dict[str, dict[str, Any]]:
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


def _candidate_expectation(base_model: str, variant_name: str, endpoint: str) -> str:
    strength = "strong" if base_model in {"tsm", "linear_ridge"} else "weaker"
    endpoint_label = "4B" if endpoint == "4b" else "35B"
    if variant_name == "baseline":
        return f"Expected to reproduce the current {endpoint_label} online-memory baseline for {base_model}."
    if variant_name == "learned_gate":
        return f"Expected small lift if {base_model} benefits from validation-fit online gating; likely limited on {strength} bases."
    if variant_name == "horizon_specific":
        return f"Expected to help if {base_model} needs faster h20 adaptation without waiting for full h30 realization."
    if variant_name == "regime_specific":
        return f"Expected to help if {base_model} suffers from contradictory memories across regimes."
    if variant_name == "high_utility_admission":
        return f"Expected to help if {base_model} benefits from only promoting materially positive long-horizon memories."
    return f"Expected to help if {base_model} benefits from smaller, less cluttered online memory prompts."


def _single_candidates(endpoints: list[str], base_models: list[str]) -> list[Candidate]:
    variants = _policy_variants()
    objective_map = {
        "baseline": "Reproduce the current live-online baseline under the refactored runner.",
        "learned_gate": "Convert offline helpfulness into validation-fit threshold rules for live gating.",
        "horizon_specific": "Admit h20 memories earlier than h30 to make online adaptation faster and more target-aligned.",
        "regime_specific": "Only retrieve online memories from matching regime buckets.",
        "high_utility_admission": "Promote only cases that materially help long horizons while avoiding h5 damage.",
        "prototype_compression": "Compress raw memory banks into a smaller, more diverse case set.",
    }
    config_map = {
        "4b": "uk_ets/config/uk_ets_llm_4b_current_default.yaml",
        "35b": "uk_ets/config/uk_ets_llm_35b_current_default.yaml",
    }
    candidates: list[Candidate] = []
    for endpoint in endpoints:
        for base_model in base_models:
            for variant_name, variant_overrides in variants.items():
                candidates.append(
                    Candidate(
                        name=f"{endpoint}_{base_model}_{variant_name}",
                        endpoint=endpoint,
                        base_model=base_model,
                        phase="single_policy",
                        objective=objective_map[variant_name],
                        expectation=_candidate_expectation(base_model, variant_name, endpoint),
                        config_path=config_map[endpoint],
                        overrides=_deep_merge(
                            _baseline_policy(endpoint, base_model),
                            variant_overrides,
                        ),
                    )
                )
    return candidates


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


def _extract_run_metrics(candidate: Candidate, run_dir: Path) -> dict[str, Any]:
    path_df = pd.read_csv(run_dir / "results" / "path_metrics.csv")
    horizon_df = pd.read_csv(run_dir / "results" / "metrics_by_horizon.csv")
    result_name = llm_result_name(METHOD_NAME, candidate.base_model)

    base_row = _load_metric_row(path_df, candidate.base_model)
    llm_row = _load_metric_row(path_df, result_name)
    reflect_calls, apply_calls = _count_llm_stage_calls(run_dir)
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
        "gate_selection": json.dumps(gate_selection, sort_keys=True) if gate_selection else "",
        "objective": candidate.objective,
        "expectation": candidate.expectation,
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


def _run_candidate(candidate: Candidate) -> dict[str, Any]:
    run_dir = Path(run_experiment(config_path=candidate.config_path, overrides=candidate.overrides))
    return _extract_run_metrics(candidate, run_dir)


def _success_variants(single_rows: list[dict[str, Any]], endpoint: str, base_model: str) -> list[str]:
    endpoint_rows = [
        row
        for row in single_rows
        if row["endpoint"] == endpoint
        and row["base_model"] == base_model
        and row["phase"] == "single_policy"
    ]
    baseline = next(row for row in endpoint_rows if row["candidate"].endswith("_baseline"))
    success_names = []
    for row in endpoint_rows:
        if row["candidate"].endswith("_baseline"):
            continue
        if float(row["llm_mse_path"]) < float(baseline["llm_mse_path"]):
            prefix = f"{endpoint}_{base_model}_"
            candidate_name = str(row["candidate"])
            if candidate_name.startswith(prefix):
                success_names.append(candidate_name[len(prefix):])
            else:
                success_names.append(candidate_name)
    return success_names


def _combined_policy_overrides(success_variant_names: list[str]) -> dict[str, Any]:
    variants = _policy_variants()
    combined: dict[str, Any] = {}
    for variant_name in success_variant_names:
        variant = variants.get(variant_name)
        if variant is None:
            continue
        combined = _deep_merge(combined, variant)
    return combined


def _combined_candidates(success_map: dict[tuple[str, str], list[str]], endpoints: list[str], base_models: list[str]) -> list[Candidate]:
    config_map = {
        "4b": "uk_ets/config/uk_ets_llm_4b_current_default.yaml",
        "35b": "uk_ets/config/uk_ets_llm_35b_current_default.yaml",
    }
    candidates: list[Candidate] = []
    for endpoint in endpoints:
        for base_model in base_models:
            combined_overrides = _combined_policy_overrides(success_map.get((endpoint, base_model), []))
            candidates.append(
                Candidate(
                    name=f"{endpoint}_{base_model}_combined",
                    endpoint=endpoint,
                    base_model=base_model,
                    phase="combined_policy",
                    objective=(
                        f"Combine the successful online-memory policy improvements discovered for {base_model}."
                    ),
                    expectation=(
                        f"Expected to match or slightly beat the best single-policy {base_model} run if the policy gains are additive."
                    ),
                    config_path=config_map[endpoint],
                    overrides=_deep_merge(_baseline_policy(endpoint, base_model), combined_overrides),
                )
            )
    return candidates


def _best_single_lookup(single_rows: list[dict[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    lookup: dict[tuple[str, str], dict[str, Any]] = {}
    for row in single_rows:
        key = (row["endpoint"], row["base_model"])
        current = lookup.get(key)
        if current is None or float(row["llm_mse_path"]) < float(current["llm_mse_path"]):
            lookup[key] = row
    return lookup


def _write_plan(out_dir: Path, single_candidates: list[Candidate]) -> None:
    lines = ["# Per-Base Online Memory Policy Sweep Plan", ""]
    lines.append("## Objectives")
    lines.append("- Test each proposed online-memory policy one at a time on every base model.")
    lines.append("- Measure whether the policy helps the live causal setup relative to the same-base online baseline.")
    lines.append("- Combine only the winning policies for each base separately.")
    lines.append("- Verify whether policy gains transfer across strong and weak base forecasters.")
    lines.append("")
    lines.append("## Pre-Registered Singles")
    for candidate in single_candidates:
        lines.append(f"- `{candidate.name}`: {candidate.objective} {candidate.expectation}")
    (out_dir / "plan.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_report(
    out_dir: Path,
    single_rows: list[dict[str, Any]],
    combined_rows: list[dict[str, Any]],
    success_map: dict[tuple[str, str], list[str]],
) -> None:
    single_df = pd.DataFrame(single_rows)
    combined_df = pd.DataFrame(combined_rows)
    best_single = _best_single_lookup(single_rows)

    def variant_name(candidate_name: str, endpoint: str, base_model: str) -> str:
        prefix = f"{endpoint}_{base_model}_"
        if candidate_name.startswith(prefix):
            return candidate_name[len(prefix):]
        return candidate_name

    lines = ["# Per-Base Online Memory Policy Sweep Report", ""]
    lines.append("## Interpretation Note")
    lines.append("- These results should be compared within this sweep only.")
    lines.append("- Absolute path-MSE levels are not directly comparable to older reports that used different panel states and evaluation plumbing.")
    lines.append("")

    lines.append("## Single-Policy Results")
    for endpoint in sorted(set(single_df["endpoint"].tolist())):
        lines.append(f"### {endpoint.upper()}")
        for base_model in BASE_MODELS:
            base_rows = single_df.loc[
                (single_df["endpoint"] == endpoint)
                & (single_df["base_model"] == base_model)
            ].sort_values("llm_mse_path")
            baseline_row = base_rows.loc[base_rows["candidate"].str.endswith("_baseline")].iloc[0]
            lines.append(f"#### {base_model}")
            lines.append(
                f"- Base path MSE `{baseline_row['base_mse_path']:.6f}`; baseline online LLM path MSE `{baseline_row['llm_mse_path']:.6f}`."
            )
            for _, row in base_rows.iterrows():
                delta_vs_baseline = float(baseline_row["llm_mse_path"]) - float(row["llm_mse_path"])
                lines.append(
                    f"- `{row['candidate']}`: LLM path MSE `{row['llm_mse_path']:.6f}`; "
                    f"gain vs base `{row['path_gain_pct']:.3f}%`; "
                    f"delta vs same-base online baseline `{delta_vs_baseline:+.6f}`; "
                    f"applied `{int(row['approx_llm_applied_samples'])}` windows."
                )
            successful = success_map.get((endpoint, base_model), [])
            lines.append(
                "- Successful singles promoted: "
                + (", ".join(f"`{name}`" for name in successful) if successful else "`none`")
            )
            lines.append("")

    lines.append("## Combined Per-Base Results")
    for endpoint in sorted(set(combined_df["endpoint"].tolist())):
        lines.append(f"### {endpoint.upper()}")
        endpoint_rows = combined_df.loc[combined_df["endpoint"] == endpoint]
        for _, row in endpoint_rows.iterrows():
            key = (row["endpoint"], row["base_model"])
            best_row = best_single[key]
            delta_vs_best_single = float(best_row["llm_mse_path"]) - float(row["llm_mse_path"])
            lines.append(
                f"- `{row['base_model']}`: combined path MSE `{row['llm_mse_path']:.6f}`; "
                f"gain vs base `{row['path_gain_pct']:.3f}%`; "
                f"delta vs best single `{delta_vs_best_single:+.6f}`; "
                f"`h20` gain `{row['gain_h20_pct']:.3f}%`; `h30` gain `{row['gain_h30_pct']:.3f}%`."
            )
        lines.append("")

    lines.append("## Main Findings")
    for endpoint in sorted(set(combined_df["endpoint"].tolist())):
        lines.append(f"### {endpoint.upper()}")
        endpoint_rows = combined_df.loc[combined_df["endpoint"] == endpoint]
        winners = endpoint_rows.loc[endpoint_rows["path_gain_pct"] > 0].sort_values("path_gain_pct", ascending=False)
        if winners.empty:
            lines.append("- No base model improved under the combined per-base policy.")
        else:
            lines.append(
                "- Combined-policy positive bases: "
                + ", ".join(
                    f"`{row['base_model']}` ({row['path_gain_pct']:.3f}%)"
                    for _, row in winners.iterrows()
                )
            )
        best_single_rows = (
            single_df.loc[single_df["endpoint"] == endpoint]
            .sort_values(["base_model", "llm_mse_path"])
            .groupby("base_model", as_index=False)
            .first()
        )
        lines.append(
            "- Best single-policy by base: "
            + ", ".join(
                f"`{row['base_model']}` -> `{variant_name(str(row['candidate']), str(row['endpoint']), str(row['base_model']))}` ({row['path_gain_pct']:.3f}%)"
                for _, row in best_single_rows.iterrows()
            )
        )
        lines.append("")

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
    out_dir = PROJECT_ROOT / "reports" / "uk_ets_online_memory_per_base" / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)

    single_candidates = _single_candidates(endpoints, BASE_MODELS)
    _write_plan(out_dir, single_candidates)

    single_rows: list[dict[str, Any]] = []
    for candidate in single_candidates:
        single_rows.append(_run_candidate(candidate))
        pd.DataFrame(single_rows).to_csv(out_dir / "single_policy_results.csv", index=False)

    success_map = {
        (endpoint, base_model): _success_variants(single_rows, endpoint, base_model)
        for endpoint in endpoints
        for base_model in BASE_MODELS
    }

    combined_candidates = _combined_candidates(success_map, endpoints, BASE_MODELS)
    combined_rows: list[dict[str, Any]] = []
    for candidate in combined_candidates:
        combined_rows.append(_run_candidate(candidate))
        pd.DataFrame(combined_rows).to_csv(out_dir / "combined_policy_results.csv", index=False)

    summary_rows = []
    best_single = _best_single_lookup(single_rows)
    for endpoint in endpoints:
        for base_model in BASE_MODELS:
            best_row = best_single[(endpoint, base_model)]
            combined_row = next(
                row
                for row in combined_rows
                if row["endpoint"] == endpoint and row["base_model"] == base_model
            )
            summary_rows.append(
                {
                    "endpoint": endpoint,
                    "base_model": base_model,
                    "best_single_candidate": best_row["candidate"],
                    "best_single_llm_mse_path": best_row["llm_mse_path"],
                    "best_single_gain_pct": best_row["path_gain_pct"],
                    "combined_llm_mse_path": combined_row["llm_mse_path"],
                    "combined_gain_pct": combined_row["path_gain_pct"],
                    "combined_vs_best_single_abs": float(best_row["llm_mse_path"]) - float(combined_row["llm_mse_path"]),
                    "successful_variants": ",".join(success_map[(endpoint, base_model)]),
                }
            )
    pd.DataFrame(summary_rows).to_csv(out_dir / "best_by_base.csv", index=False)

    _write_report(out_dir, single_rows, combined_rows, success_map)


if __name__ == "__main__":
    main()
