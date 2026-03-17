#!/usr/bin/env python3
"""Run the first production live-policy build sweep for UK ETS TSM+LLM."""

from __future__ import annotations

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


@dataclass(frozen=True)
class Candidate:
    name: str
    objective: str
    expectation: str
    overrides: dict[str, Any]


def _deep_merge(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    merged = dict(left)
    for key, value in right.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(dict(merged[key]), value)
        else:
            merged[key] = value
    return merged


def _base_policy() -> dict[str, Any]:
    return {
        "llm": {
            "base_model": "tsm",
            "cot_rf": {
                "enable_numeric_tool": True,
                "force_numeric_tool": True,
                "enable_delta_verifier_tool": True,
                "force_delta_verifier_tool": True,
                "test_pool_mode": "online_realized_memory",
                "realized_memory_horizon": 30,
                "online_memory_policy": {
                    "enabled": True,
                    "support_examples": 3,
                    "positive_examples": 2,
                    "negative_examples": 1,
                    "max_total_examples": 6,
                    "positive_margin": 0.015,
                    "negative_margin": -0.015,
                    "warmup_min_realized": 5,
                    "gate": {
                        "min_positive_examples": 1,
                        "min_positive_signal": 0.015,
                        "min_net_signal": 0.005,
                        "max_negative_signal": 0.06,
                        "min_abs_base_h20_pct": 0.20,
                    },
                    "regime": {
                        "enabled": True,
                        "min_overlap": 4,
                    },
                },
            },
        },
    }


def _candidates() -> list[Candidate]:
    base = _base_policy()
    top_20_mi_features = [
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
    return [
        Candidate(
            name="baseline_regime_specific",
            objective="Reproduce the current best live-online TSM policy using regime-specific memory only.",
            expectation="Expected path MSE around 4.16-4.20. This is the live benchmark to beat.",
            overrides=base,
        ),
        Candidate(
            name="split_horizon_banks_regime",
            objective="Split positive/negative online memory into separate h20 and h30 banks while keeping regime filtering.",
            expectation="Expected path MSE around 4.13-4.18. Separate horizon banks should reduce contradictory long-horizon memories.",
            overrides=_deep_merge(
                base,
                {
                    "llm": {
                        "cot_rf": {
                            "online_memory_policy": {
                                "horizon_specific": {
                                    "enabled": True,
                                    "admission_horizons": [20, 30],
                                    "split_memory_banks": True,
                                    "positive_examples_per_horizon": 1,
                                    "negative_examples_per_horizon": 1,
                                }
                            }
                        }
                    }
                },
            ),
        ),
        Candidate(
            name="learned_logistic_regime",
            objective="Replace the heuristic apply rule with a validation-fit logistic live gate while keeping regime-specific memory.",
            expectation="Expected path MSE around 4.12-4.18. A learned gate should reduce false positives and preserve only useful LLM calls.",
            overrides=_deep_merge(
                base,
                {
                    "llm": {
                        "cot_rf": {
                            "online_memory_policy": {
                                "gate": {
                                    "learned": {
                                        "enabled": True,
                                        "model_type": "logistic",
                                        "scope": "recent_tail",
                                        "recent_tail_fraction": 0.5,
                                        "recent_tail_min_samples": 80,
                                        "max_samples": 80,
                                        "train_fraction": 0.67,
                                        "positive_margin": 0.0,
                                        "positive_min_h20_gain": 0.0,
                                        "positive_min_h30_gain": 0.0,
                                        "positive_max_h5_damage": -0.15,
                                        "probability_threshold_grid": [0.30, 0.40, 0.50, 0.60, 0.70],
                                        "metric": "mse_path",
                                    }
                                }
                            }
                        }
                    }
                },
            ),
        ),
        Candidate(
            name="learned_logistic_split_horizon_regime",
            objective="Combine a learned logistic apply gate with split h20/h30 memory banks and regime filtering.",
            expectation="Expected path MSE around 4.08-4.16. This is the main candidate to beat the current TSM live benchmark.",
            overrides=_deep_merge(
                base,
                {
                    "llm": {
                        "cot_rf": {
                            "online_memory_policy": {
                                "gate": {
                                    "learned": {
                                        "enabled": True,
                                        "model_type": "logistic",
                                        "scope": "recent_tail",
                                        "recent_tail_fraction": 0.5,
                                        "recent_tail_min_samples": 80,
                                        "max_samples": 80,
                                        "train_fraction": 0.67,
                                        "positive_margin": 0.0,
                                        "positive_min_h20_gain": 0.0,
                                        "positive_min_h30_gain": 0.0,
                                        "positive_max_h5_damage": -0.15,
                                        "probability_threshold_grid": [0.30, 0.40, 0.50, 0.60, 0.70],
                                        "metric": "mse_path",
                                    }
                                },
                                "horizon_specific": {
                                    "enabled": True,
                                    "admission_horizons": [20, 30],
                                    "split_memory_banks": True,
                                    "positive_examples_per_horizon": 1,
                                    "negative_examples_per_horizon": 1,
                                },
                            }
                        }
                    }
                },
            ),
        ),
        Candidate(
            name="learned_logistic_regime_high_utility",
            objective="Add stricter long-horizon utility-based promotion on top of the learned logistic gate without split horizon banks.",
            expectation="Expected path MSE around 4.00-4.12. If stricter promotion helps on its own, it should improve on the plain learned logistic gate.",
            overrides=_deep_merge(
                base,
                {
                    "llm": {
                        "cot_rf": {
                            "online_memory_policy": {
                                "gate": {
                                    "learned": {
                                        "enabled": True,
                                        "model_type": "logistic",
                                        "scope": "recent_tail",
                                        "recent_tail_fraction": 0.5,
                                        "recent_tail_min_samples": 80,
                                        "max_samples": 80,
                                        "train_fraction": 0.67,
                                        "positive_margin": 0.0,
                                        "positive_min_h20_gain": 0.0,
                                        "positive_min_h30_gain": 0.0,
                                        "positive_max_h5_damage": -0.15,
                                        "probability_threshold_grid": [0.30, 0.40, 0.50, 0.60, 0.70],
                                        "metric": "mse_path",
                                    }
                                },
                                "admission": {
                                    "positive_min_h20_gain": 0.01,
                                    "positive_min_h30_gain": 0.01,
                                    "positive_max_h5_damage": -0.15,
                                },
                            }
                        }
                    }
                },
            ),
        ),
        Candidate(
            name="learned_logistic_split_horizon_regime_high_utility",
            objective="Add stricter long-horizon utility-based promotion on top of the learned gate and split horizon banks.",
            expectation="Expected path MSE around 4.08-4.17. This may improve memory quality if noisy positive cases are currently diluting the bank.",
            overrides=_deep_merge(
                base,
                {
                    "llm": {
                        "cot_rf": {
                            "online_memory_policy": {
                                "gate": {
                                    "learned": {
                                        "enabled": True,
                                        "model_type": "logistic",
                                        "scope": "recent_tail",
                                        "recent_tail_fraction": 0.5,
                                        "recent_tail_min_samples": 80,
                                        "max_samples": 80,
                                        "train_fraction": 0.67,
                                        "positive_margin": 0.0,
                                        "positive_min_h20_gain": 0.0,
                                        "positive_min_h30_gain": 0.0,
                                        "positive_max_h5_damage": -0.15,
                                        "probability_threshold_grid": [0.30, 0.40, 0.50, 0.60, 0.70],
                                        "metric": "mse_path",
                                    }
                                },
                                "admission": {
                                    "positive_min_h20_gain": 0.01,
                                    "positive_min_h30_gain": 0.01,
                                    "positive_max_h5_damage": -0.15,
                                },
                                "horizon_specific": {
                                    "enabled": True,
                                    "admission_horizons": [20, 30],
                                    "split_memory_banks": True,
                                    "positive_examples_per_horizon": 1,
                                    "negative_examples_per_horizon": 1,
                                },
                            }
                        }
                    }
                },
            ),
        ),
        Candidate(
            name="learned_gbdt_regime",
            objective="Use a nonlinear boosted-tree live gate with regime-specific memory and no split horizon banks.",
            expectation="Expected path MSE around 3.70-3.80 with the promoted top-20 gate schema. If the gain is mostly from the gate, this should remain the strongest production candidate.",
            overrides=_deep_merge(
                base,
                {
                    "llm": {
                        "cot_rf": {
                            "online_memory_policy": {
                                "gate": {
                                    "learned": {
                                        "enabled": True,
                                        "model_type": "hist_gbdt",
                                        "scope": "recent_tail",
                                        "recent_tail_fraction": 0.5,
                                        "recent_tail_min_samples": 80,
                                        "max_samples": 80,
                                        "train_fraction": 0.67,
                                        "positive_margin": 0.0,
                                        "positive_min_h20_gain": 0.0,
                                        "positive_min_h30_gain": 0.0,
                                        "positive_max_h5_damage": -0.15,
                                        "probability_threshold_grid": [0.30, 0.40, 0.50, 0.60, 0.70],
                                        "metric": "mse_path",
                                        "max_depth": 3,
                                        "learning_rate": 0.05,
                                        "max_iter": 200,
                                        "feature_columns": list(top_20_mi_features),
                                    }
                                }
                            }
                        }
                    }
                },
            ),
        ),
        Candidate(
            name="learned_gbdt_split_horizon_regime",
            objective="Use a nonlinear boosted-tree live gate with split horizon banks and regime filtering.",
            expectation="Expected path MSE around 4.08-4.16. This is the more flexible learned-gate alternative if logistic is too linear.",
            overrides=_deep_merge(
                base,
                {
                    "llm": {
                        "cot_rf": {
                            "online_memory_policy": {
                                "gate": {
                                    "learned": {
                                        "enabled": True,
                                        "model_type": "hist_gbdt",
                                        "scope": "recent_tail",
                                        "recent_tail_fraction": 0.5,
                                        "recent_tail_min_samples": 80,
                                        "max_samples": 80,
                                        "train_fraction": 0.67,
                                        "positive_margin": 0.0,
                                        "positive_min_h20_gain": 0.0,
                                        "positive_min_h30_gain": 0.0,
                                        "positive_max_h5_damage": -0.15,
                                        "probability_threshold_grid": [0.30, 0.40, 0.50, 0.60, 0.70],
                                        "metric": "mse_path",
                                        "max_depth": 3,
                                        "learning_rate": 0.05,
                                        "max_iter": 200,
                                    }
                                },
                                "horizon_specific": {
                                    "enabled": True,
                                    "admission_horizons": [20, 30],
                                    "split_memory_banks": True,
                                    "positive_examples_per_horizon": 1,
                                    "negative_examples_per_horizon": 1,
                                },
                            }
                        }
                    }
                },
            ),
        ),
    ]


def _count_calls(run_dir: Path) -> tuple[int, int]:
    path = run_dir / "llm" / "logs" / "llm_calls.jsonl"
    if not path.exists():
        return 0, 0
    reflect = apply = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            payload = json.loads(line)
            method = str(payload.get("method", ""))
            if method.endswith(":reflect"):
                reflect += 1
            elif method.endswith(":apply"):
                apply += 1
    return reflect, apply


def _metric_rows(run_dir: Path) -> tuple[pd.Series, pd.Series, pd.DataFrame]:
    path_df = pd.read_csv(run_dir / "results" / "path_metrics.csv")
    horizon_df = pd.read_csv(run_dir / "results" / "metrics_by_horizon.csv")
    result_name = llm_result_name(METHOD_NAME, "tsm")
    base_row = path_df.loc[path_df["model"] == "tsm"].iloc[0]
    llm_row = path_df.loc[path_df["model"] == result_name].iloc[0]
    return base_row, llm_row, horizon_df


def _gate_selection(run_dir: Path) -> dict[str, Any] | None:
    result_slug = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in llm_result_name(METHOD_NAME, "tsm"))
    path = run_dir / "llm" / f"online_memory_gate_selection_{result_slug}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _run_candidate(candidate: Candidate) -> dict[str, Any]:
    try:
        run_dir = Path(
            run_experiment(
                config_path="uk_ets/config/uk_ets_llm_4b_current_default.yaml",
                overrides=candidate.overrides,
            )
        )
        base_row, llm_row, horizon_df = _metric_rows(run_dir)
        reflect_calls, apply_calls = _count_calls(run_dir)
        row: dict[str, Any] = {
            "candidate": candidate.name,
            "status": "ok",
            "run_dir": str(run_dir),
            "base_mse_path": float(base_row["mse_path"]),
            "llm_mse_path": float(llm_row["mse_path"]),
            "path_gain_abs": float(base_row["mse_path"]) - float(llm_row["mse_path"]),
            "path_gain_pct": (float(base_row["mse_path"]) - float(llm_row["mse_path"])) / max(float(base_row["mse_path"]), 1e-8) * 100.0,
            "reflect_calls": int(reflect_calls),
            "apply_calls": int(apply_calls),
            "objective": candidate.objective,
            "expectation": candidate.expectation,
            "gate_selection": json.dumps(_gate_selection(run_dir) or {}, sort_keys=True),
            "error": "",
        }
        result_name = llm_result_name(METHOD_NAME, "tsm")
        for horizon in (1, 5, 20, 30):
            base_h = horizon_df.loc[(horizon_df["model"] == "tsm") & (horizon_df["horizon"] == horizon)].iloc[0]
            llm_h = horizon_df.loc[(horizon_df["model"] == result_name) & (horizon_df["horizon"] == horizon)].iloc[0]
            row[f"base_h{horizon}_mse"] = float(base_h["mse"])
            row[f"llm_h{horizon}_mse"] = float(llm_h["mse"])
            row[f"gain_h{horizon}_pct"] = (float(base_h["mse"]) - float(llm_h["mse"])) / max(float(base_h["mse"]), 1e-8) * 100.0
        return row
    except Exception as exc:
        return {
            "candidate": candidate.name,
            "status": "failed",
            "run_dir": "",
            "base_mse_path": float("nan"),
            "llm_mse_path": float("nan"),
            "path_gain_abs": float("nan"),
            "path_gain_pct": float("nan"),
            "reflect_calls": 0,
            "apply_calls": 0,
            "objective": candidate.objective,
            "expectation": candidate.expectation,
            "gate_selection": "{}",
            "error": repr(exc),
        }


def main() -> None:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = PROJECT_ROOT / "reports" / "uk_ets_tsm_live_policy_build" / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)

    plan_lines = [
        "# TSM Live Policy Build Plan",
        "",
        "## Objective",
        "- Build the first production-oriented live TSM policy with a learned apply gate and split h20/h30 memory banks.",
        "- Compare each change against the current best live-online TSM benchmark.",
        "",
        "## Candidates",
    ]
    for candidate in _candidates():
        plan_lines.append(f"- `{candidate.name}`: {candidate.objective} {candidate.expectation}")
    (out_dir / "plan.md").write_text("\n".join(plan_lines) + "\n", encoding="utf-8")

    rows = []
    for candidate in _candidates():
        rows.append(_run_candidate(candidate))
        pd.DataFrame(rows).to_csv(out_dir / "candidate_results.csv", index=False)

    df = pd.DataFrame(rows)
    ok_df = df.loc[df["status"] == "ok"].sort_values("llm_mse_path")
    baseline = ok_df.loc[ok_df["candidate"] == "baseline_regime_specific"].iloc[0]
    best = ok_df.iloc[0]
    lines = [
        "# TSM Live Policy Build Report",
        "",
        f"Best candidate: `{best['candidate']}` at path MSE `{best['llm_mse_path']:.6f}`.",
        f"Baseline live champion: `baseline_regime_specific` at `{baseline['llm_mse_path']:.6f}`.",
        "",
        "## Results",
    ]
    for _, row in ok_df.iterrows():
        delta_vs_baseline = float(baseline["llm_mse_path"]) - float(row["llm_mse_path"])
        lines.append(
            f"- `{row['candidate']}`: path MSE `{row['llm_mse_path']:.6f}`, gain vs base `{row['path_gain_pct']:.3f}%`, "
            f"delta vs baseline `{delta_vs_baseline:+.6f}`, apply calls `{int(row['apply_calls'])}`, "
            f"`h20` gain `{row['gain_h20_pct']:.3f}%`, `h30` gain `{row['gain_h30_pct']:.3f}%`."
        )
    failed_df = df.loc[df["status"] != "ok"]
    if not failed_df.empty:
        lines.extend(["", "## Failures"])
        for _, row in failed_df.iterrows():
            lines.append(f"- `{row['candidate']}` failed: `{row['error']}`")
    lines.extend(
        [
            "",
            "## Artifacts",
            f"- Candidate table: [{(out_dir / 'candidate_results.csv').name}]({str(out_dir / 'candidate_results.csv')})",
            f"- Plan: [{(out_dir / 'plan.md').name}]({str(out_dir / 'plan.md')})",
        ]
    )
    (out_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
