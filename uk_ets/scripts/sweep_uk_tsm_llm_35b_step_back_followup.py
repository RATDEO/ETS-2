#!/usr/bin/env python3
"""Five-way follow-up sweep on top of the 35B step-back UK winner."""

from __future__ import annotations

import argparse
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

from src.run_experiment import run_experiment


METHOD_NAME = "TSM+LLM-COT-RF-HDELTA"


@dataclass(frozen=True)
class Candidate:
    name: str
    literature_label: str
    literature_url: str
    change_summary: str
    aim: str
    expected_low: float
    expected_high: float
    cot_rf: dict[str, Any]
    llm: dict[str, Any]
    hdelta: dict[str, Any]


def _load_path_metric(run_dir: Path, model_name: str) -> float:
    path_metrics = pd.read_csv(run_dir / "results" / "path_metrics.csv")
    return float(path_metrics[path_metrics["model"] == model_name].iloc[0]["mse_path"])


def _load_horizon_mse(run_dir: Path, model_name: str) -> dict[int, float]:
    horizon_metrics = pd.read_csv(run_dir / "results" / "metrics_by_horizon.csv")
    subset = horizon_metrics[horizon_metrics["model"] == model_name]
    return {int(row["horizon"]): float(row["mse"]) for _, row in subset.iterrows()}


def _load_llm_stats(run_dir: Path) -> dict[str, Any]:
    unique_apply = set()
    unique_reflect = set()
    reflect_valid_samples: list[int] = []
    reflect_samples: list[int] = []
    routed_count = 0
    log_path = run_dir / "llm" / "logs" / "llm_calls.jsonl"
    with open(log_path, "r") as handle:
        for line in handle:
            obj = json.loads(line)
            method = str(obj.get("method", ""))
            response = ((obj.get("response") or {}).get("content") or "").strip()
            metadata = obj.get("metadata") or {}
            if method.endswith(":apply") and response:
                unique_apply.add(response)
            if method.endswith(":reflect"):
                if response:
                    unique_reflect.add(response)
                reflect_samples.append(int(metadata.get("reflect_samples", 1)))
                reflect_valid_samples.append(int(metadata.get("reflect_valid_samples", 0)))
                if bool(metadata.get("reflect_uncertainty_routed", False)):
                    routed_count += 1
    return {
        "unique_apply_responses": len(unique_apply),
        "unique_reflect_responses": len(unique_reflect),
        "reflect_samples_used": ",".join(str(x) for x in sorted(set(reflect_samples))),
        "mean_reflect_valid_samples": float(np.mean(reflect_valid_samples)) if reflect_valid_samples else 0.0,
        "uncertainty_routed_reflect_calls": routed_count,
    }


def _classify_expectation(actual: float, expected_low: float, expected_high: float) -> str:
    if actual < expected_low:
        return "beat"
    if actual <= expected_high:
        return "within"
    return "missed"


def _extract_candidate_metrics(
    run_dir: Path,
    baseline_llm_path: float,
    baseline_horizons: dict[int, float],
) -> dict[str, Any]:
    path_metrics = pd.read_csv(run_dir / "results" / "path_metrics.csv")
    horizon_metrics = pd.read_csv(run_dir / "results" / "metrics_by_horizon.csv")

    tsm_base = path_metrics[path_metrics["model"] == "tsm"].iloc[0]
    llm_raw = path_metrics[path_metrics["model"] == METHOD_NAME].iloc[0]

    llm_h = horizon_metrics[horizon_metrics["model"] == METHOD_NAME].set_index("horizon")
    pred_npz = np.load(run_dir / "predictions" / f"{METHOD_NAME}_pred_test_subset.npz")
    deltas = pred_npz["yhat"] - pred_npz["base_pred"]
    llm_stats = _load_llm_stats(run_dir)

    return {
        "run_dir": str(run_dir),
        "base_mse_path": float(tsm_base["mse_path"]),
        "llm_mse_path": float(llm_raw["mse_path"]),
        "delta_vs_tsm": float(llm_raw["mse_path"] - tsm_base["mse_path"]),
        "delta_vs_baseline_llm": float(llm_raw["mse_path"] - baseline_llm_path),
        "h5_delta_vs_baseline": float(llm_h.loc[5, "mse"] - baseline_horizons[5]),
        "h20_delta_vs_baseline": float(llm_h.loc[20, "mse"] - baseline_horizons[20]),
        "h30_delta_vs_baseline": float(llm_h.loc[30, "mse"] - baseline_horizons[30]),
        "nonzero_count": int(np.count_nonzero(np.abs(deltas) > 1e-12)),
        "abs_mean_delta": float(np.abs(deltas).mean()),
        **llm_stats,
    }


def _write_report(
    out_dir: Path,
    rows: list[dict[str, Any]],
    baseline_run_dir: Path,
    baseline_path_mse: float,
    ridge_path_mse: float,
) -> None:
    pd.DataFrame(rows).to_csv(out_dir / "candidate_results.csv", index=False)

    lines = [
        "# UK ETS 35B Step-Back Follow-Up Sweep",
        "",
        f"Generated: {datetime.now().isoformat()}",
        "",
        "## Baseline",
        "",
        f"- Baseline run: `{baseline_run_dir}`",
        f"- Baseline raw `{METHOD_NAME}` path MSE: `{baseline_path_mse:.6f}`",
        f"- Reference `linear_ridge` path MSE: `{ridge_path_mse:.6f}`",
        "",
        "## Planned Variants",
        "",
        "| Candidate | Literature | Aim | Expected range |",
        "|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['name']} | [{row['literature_label']}]({row['literature_url']}) | {row['aim']} | {row['expected_low']:.2f}-{row['expected_high']:.2f} |"
        )

    lines.extend(
        [
            "",
            "## Results",
            "",
            "| Candidate | Actual | Vs baseline | Vs ridge | Outcome | h5 vs baseline | h20 vs baseline | h30 vs baseline | Reflect samples | Routed reflect calls | Unique reflect | Unique apply |",
            "|---|---:|---:|---:|---|---:|---:|---:|---|---:|---:|---:|",
        ]
    )
    for row in rows:
        lines.append(
            "| {name} | {llm_mse_path:.6f} | {delta_vs_baseline_llm:+.6f} | {delta_vs_ridge:+.6f} | {expectation_result} | {h5_delta_vs_baseline:+.6f} | {h20_delta_vs_baseline:+.6f} | {h30_delta_vs_baseline:+.6f} | {reflect_samples_used} | {uncertainty_routed_reflect_calls} | {unique_reflect_responses} | {unique_apply_responses} |".format(
                **row
            )
        )

    best = min(rows, key=lambda row: row["llm_mse_path"])
    lines.extend(
        [
            "",
            "## Best Candidate",
            "",
            f"- Candidate: `{best['name']}`",
            f"- Literature: [{best['literature_label']}]({best['literature_url']})",
            f"- Change: {best['change_summary']}",
            f"- Aim: {best['aim']}",
            f"- Expected range: `{best['expected_low']:.2f}-{best['expected_high']:.2f}`",
            f"- Actual raw path MSE: `{best['llm_mse_path']:.6f}`",
            f"- Delta vs baseline raw LLM: `{best['delta_vs_baseline_llm']:+.6f}`",
            f"- Delta vs ridge: `{best['delta_vs_ridge']:+.6f}`",
            f"- Run dir: `{best['run_dir']}`",
        ]
    )

    (out_dir / "summary.md").write_text("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a five-way follow-up sweep on the 35B step-back UK winner.")
    parser.add_argument("--config", default="uk_ets/config/uk_ets_llm_35b_step_back_regime.yaml")
    parser.add_argument("--data-dir", default="uk_ets/Data_auto_uk")
    parser.add_argument("--output-root", default="reports/uk_ets_llm_35b_step_back_followup")
    parser.add_argument(
        "--baseline-run",
        default="runs/20260309_234219_249d62",
        help="Existing step-back winner used as the raw LLM reference.",
    )
    args = parser.parse_args()

    config_path = (PROJECT_ROOT / args.config).resolve()
    data_dir = (PROJECT_ROOT / args.data_dir).resolve()
    baseline_run_dir = (PROJECT_ROOT / args.baseline_run).resolve()

    baseline_path_mse = _load_path_metric(baseline_run_dir, METHOD_NAME)
    baseline_horizons = _load_horizon_mse(baseline_run_dir, METHOD_NAME)
    ridge_path_mse = _load_path_metric(baseline_run_dir, "linear_ridge")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = (PROJECT_ROOT / args.output_root / timestamp).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    candidates = [
        Candidate(
            name="analogical_step_back",
            literature_label="Large Language Models as Analogical Reasoners",
            literature_url="https://arxiv.org/abs/2310.01714",
            change_summary="Keep the step-back regime abstraction, but explicitly transfer only the reusable pattern from the most analogous matched examples.",
            aim="Exploit the matched-example memory more directly without giving up the strong regime-first reasoning structure.",
            expected_low=21.36,
            expected_high=21.50,
            cot_rf={"reasoning_style": "analogical_step_back", "apply_style": "program_of_thought"},
            llm={},
            hdelta={},
        ),
        Candidate(
            name="thought_propagation_step_back",
            literature_label="Thought Propagation",
            literature_url="https://arxiv.org/abs/2310.03965",
            change_summary="Ask the model to propagate reusable reasoning patterns from matched examples after the step-back regime abstraction.",
            aim="Improve h20 and h30 by reusing successful analogous correction patterns rather than reasoning from scratch each time.",
            expected_low=21.34,
            expected_high=21.50,
            cot_rf={"reasoning_style": "thought_propagation", "apply_style": "program_of_thought"},
            llm={},
            hdelta={},
        ),
        Candidate(
            name="self_verification_step_back",
            literature_label="Self-Verification Improves Few-Shot CoT",
            literature_url="https://arxiv.org/abs/2212.09561",
            change_summary="Apply backward self-verification to the step-back regime plan before emitting horizon decisions.",
            aim="Keep the regime abstraction but reject horizon moves that do not survive a backward consistency check.",
            expected_low=21.37,
            expected_high=21.52,
            cot_rf={"reasoning_style": "self_verification", "apply_style": "program_of_thought"},
            llm={},
            hdelta={},
        ),
        Candidate(
            name="self_discover_step_back",
            literature_label="Self-Discover",
            literature_url="https://arxiv.org/abs/2402.03620",
            change_summary="Let the model internally choose the reasoning modules before applying the step-back decision process.",
            aim="See whether dynamic reasoning-structure selection can improve on a fixed step-back regime contract.",
            expected_low=21.40,
            expected_high=21.56,
            cot_rf={"reasoning_style": "self_discover", "apply_style": "program_of_thought"},
            llm={},
            hdelta={},
        ),
        Candidate(
            name="uncertainty_routed_step_back",
            literature_label="UnCert-CoT / Uncertainty of Thoughts",
            literature_url="https://arxiv.org/abs/2503.15341",
            change_summary="Keep the step-back prompt, but route extra three-sample reflective search only to uncertain matched-example cases.",
            aim="Preserve the current winner on easy windows while spending extra deliberation only where evidence is ambiguous.",
            expected_low=21.32,
            expected_high=21.48,
            cot_rf={
                "reasoning_style": "step_back",
                "apply_style": "program_of_thought",
                "reflect_uncertainty_routing": True,
                "reflect_samples": 1,
                "reflect_samples_low_uncertainty": 1,
                "reflect_samples_high_uncertainty": 3,
                "reflect_aggregation": "conservative_majority",
                "reflect_temperature": 0.15,
                "reflect_uncertainty_distance_threshold": 1.50,
                "reflect_uncertainty_min_dynamic_freezes": 2,
                "reflect_uncertainty_min_low_bound_horizons": 2,
                "reflect_uncertainty_low_bound_threshold": 0.20,
            },
            llm={"cache_enabled": False},
            hdelta={},
        ),
    ]

    rows: list[dict[str, Any]] = []
    for idx, candidate in enumerate(candidates, start=1):
        print(f"[{idx}/{len(candidates)}] Running {candidate.name} ...", flush=True)
        overrides = {
            "target": {"mode": "returns"},
            "output": {
                "write_project_paper": False,
                "generate_paper": False,
                "generate_plots": False,
                "save_model_checkpoints": False,
            },
            "compute": {"num_workers": 0, "pin_memory": False},
            "llm": {
                "base_model": "tsm",
                "blend_grid": {"enabled": False},
                "rule_gate": {"enabled": False},
                **candidate.llm,
                "cot_rf": candidate.cot_rf,
            },
            "hdelta": candidate.hdelta,
        }
        run_dir = Path(
            run_experiment(
                config_path=str(config_path),
                overrides=overrides,
                data_dir=str(data_dir),
            )
        )
        metrics = _extract_candidate_metrics(run_dir, baseline_path_mse, baseline_horizons)
        row = {
            "name": candidate.name,
            "literature_label": candidate.literature_label,
            "literature_url": candidate.literature_url,
            "change_summary": candidate.change_summary,
            "aim": candidate.aim,
            "expected_low": candidate.expected_low,
            "expected_high": candidate.expected_high,
            "delta_vs_ridge": float(metrics["llm_mse_path"] - ridge_path_mse),
            "expectation_result": _classify_expectation(
                metrics["llm_mse_path"],
                candidate.expected_low,
                candidate.expected_high,
            ),
            **metrics,
        }
        rows.append(row)
        print(
            f"  -> {candidate.name}: llm_mse_path={row['llm_mse_path']:.6f} "
            f"(vs_baseline={row['delta_vs_baseline_llm']:+.6f}, expectation={row['expectation_result']})",
            flush=True,
        )

    rows.sort(key=lambda row: row["llm_mse_path"])
    _write_report(out_dir, rows, baseline_run_dir, baseline_path_mse, ridge_path_mse)
    print(f"Sweep results written to {out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
