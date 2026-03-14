#!/usr/bin/env python3
"""Tool-routing and retrieval follow-up sweep for the UK 35B TSM+LLM pipeline."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import dataclass, asdict
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
    tool_counts: Counter[str] = Counter()
    log_path = run_dir / "llm" / "logs" / "llm_calls.jsonl"
    if not log_path.exists():
        return {
            "unique_apply_responses": 0,
            "unique_reflect_responses": 0,
            "numeric_tool_calls": 0,
            "verifier_tool_calls": 0,
            "case_retrieval_tool_calls": 0,
            "market_tool_calls": 0,
            "counterexample_tool_calls": 0,
        }

    with open(log_path, "r") as handle:
        for line in handle:
            obj = json.loads(line)
            method = str(obj.get("method", ""))
            response = ((obj.get("response") or {}).get("content") or "").strip()
            metadata = obj.get("metadata") or {}
            tool_counts.update(metadata.get("tool_calls") or [])
            if method.endswith(":apply") and response:
                unique_apply.add(response)
            if method.endswith(":reflect") and response:
                unique_reflect.add(response)

    return {
        "unique_apply_responses": len(unique_apply),
        "unique_reflect_responses": len(unique_reflect),
        "numeric_tool_calls": int(tool_counts.get("get_numeric_analysis", 0)),
        "verifier_tool_calls": int(tool_counts.get("verify_hdelta_adjustments", 0)),
        "case_retrieval_tool_calls": int(tool_counts.get("get_structured_case_retrieval", 0)),
        "market_tool_calls": int(tool_counts.get("get_market_microstructure_state", 0)),
        "counterexample_tool_calls": int(tool_counts.get("get_freeze_counterexample", 0)),
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
    ridge_path_mse: float,
    old_best_path_mse: float,
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
        "delta_vs_ridge": float(llm_raw["mse_path"] - ridge_path_mse),
        "delta_vs_old_best": float(llm_raw["mse_path"] - old_best_path_mse),
        "h5_delta_vs_baseline": float(llm_h.loc[5, "mse"] - baseline_horizons[5]),
        "h20_delta_vs_baseline": float(llm_h.loc[20, "mse"] - baseline_horizons[20]),
        "h30_delta_vs_baseline": float(llm_h.loc[30, "mse"] - baseline_horizons[30]),
        "nonzero_count": int(np.count_nonzero(np.abs(deltas) > 1e-12)),
        "abs_mean_delta": float(np.abs(deltas).mean()),
        "mean_delta": float(deltas.mean()),
        **llm_stats,
    }


def _write_plan(out_dir: Path, candidates: list[Candidate]) -> None:
    pd.DataFrame([asdict(candidate) for candidate in candidates]).to_csv(
        out_dir / "planned_candidates.csv",
        index=False,
    )
    lines = [
        "# UK ETS 35B Tool Follow-Up Sweep Plan",
        "",
        f"Generated: {datetime.now().isoformat()}",
        "",
        "## Planned Candidates",
        "",
        "| Candidate | Literature | Aim | Expected range | Change |",
        "|---|---|---|---|---|",
    ]
    for candidate in candidates:
        lines.append(
            f"| {candidate.name} | [{candidate.literature_label}]({candidate.literature_url}) | {candidate.aim} | "
            f"{candidate.expected_low:.2f}-{candidate.expected_high:.2f} | {candidate.change_summary} |"
        )
    (out_dir / "plan.md").write_text("\n".join(lines) + "\n")


def _write_report(
    out_dir: Path,
    rows: list[dict[str, Any]],
    baseline_run_dir: Path,
    baseline_path_mse: float,
    ridge_path_mse: float,
    old_best_path_mse: float,
) -> None:
    pd.DataFrame(rows).to_csv(out_dir / "candidate_results.csv", index=False)

    lines = [
        "# UK ETS 35B Tool Follow-Up Sweep",
        "",
        f"Generated: {datetime.now().isoformat()}",
        "",
        "## References",
        "",
        f"- Baseline run: `{baseline_run_dir}`",
        f"- Baseline raw `{METHOD_NAME}` path MSE: `{baseline_path_mse:.6f}`",
        f"- `linear_ridge` path MSE: `{ridge_path_mse:.6f}`",
        f"- Old best 35B no-tool step-back path MSE: `{old_best_path_mse:.6f}`",
        "",
        "## Results",
        "",
        "| Candidate | Actual | Vs baseline | Vs ridge | Vs old best | Outcome | h5 vs baseline | h20 vs baseline | h30 vs baseline | Numeric | Verifier | Retrieval | Market | Counterexample | Unique reflect | Unique apply |",
        "|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {name} | {llm_mse_path:.6f} | {delta_vs_baseline_llm:+.6f} | {delta_vs_ridge:+.6f} | {delta_vs_old_best:+.6f} | {expectation_result} | {h5_delta_vs_baseline:+.6f} | {h20_delta_vs_baseline:+.6f} | {h30_delta_vs_baseline:+.6f} | {numeric_tool_calls} | {verifier_tool_calls} | {case_retrieval_tool_calls} | {market_tool_calls} | {counterexample_tool_calls} | {unique_reflect_responses} | {unique_apply_responses} |".format(
                **row
            )
        )

    if rows:
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
                f"- Delta vs baseline: `{best['delta_vs_baseline_llm']:+.6f}`",
                f"- Delta vs ridge: `{best['delta_vs_ridge']:+.6f}`",
                f"- Delta vs old best: `{best['delta_vs_old_best']:+.6f}`",
                f"- Outcome vs expectation: `{best['expectation_result']}`",
                f"- Run dir: `{best['run_dir']}`",
            ]
        )

    (out_dir / "summary.md").write_text("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a 35B tool-routing follow-up sweep.")
    parser.add_argument("--config", default="uk_ets/config/uk_ets_llm_35b_current_default.yaml")
    parser.add_argument("--data-dir", default="uk_ets/Data_auto_uk")
    parser.add_argument("--output-root", default="reports/uk_ets_llm_35b_tool_followup")
    parser.add_argument(
        "--baseline-run",
        default="runs/20260313_141036_907f02",
        help="Current tool-backed 35B default run used as the baseline reference.",
    )
    parser.add_argument(
        "--old-best-run",
        default="runs/20260309_234219_249d62",
        help="Old best 35B no-tool step-back run used as the absolute reference.",
    )
    args = parser.parse_args()

    config_path = (PROJECT_ROOT / args.config).resolve()
    data_dir = (PROJECT_ROOT / args.data_dir).resolve()
    baseline_run_dir = (PROJECT_ROOT / args.baseline_run).resolve()
    old_best_run_dir = (PROJECT_ROOT / args.old_best_run).resolve()

    baseline_path_mse = _load_path_metric(baseline_run_dir, METHOD_NAME)
    baseline_horizons = _load_horizon_mse(baseline_run_dir, METHOD_NAME)
    ridge_path_mse = _load_path_metric(baseline_run_dir, "linear_ridge")
    old_best_path_mse = _load_path_metric(old_best_run_dir, METHOD_NAME)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = (PROJECT_ROOT / args.output_root / timestamp).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    candidates = [
        Candidate(
            name="optional_tools_default",
            literature_label="Toolformer",
            literature_url="https://arxiv.org/abs/2302.04761",
            change_summary="Keep the current default stack but make numeric and verifier tool use optional instead of required.",
            aim="Test whether 35B is being over-regularized by forced tool calls on easy windows.",
            expected_low=21.42,
            expected_high=21.56,
            cot_rf={"force_numeric_tool": False, "force_delta_verifier_tool": False},
            llm={},
            hdelta={},
        ),
        Candidate(
            name="numeric_only_default",
            literature_label="Program of Thoughts",
            literature_url="https://arxiv.org/abs/2211.12588",
            change_summary="Remove the verifier and keep only the numeric tool under the current least-to-most stack.",
            aim="Test whether exact arithmetic helps more than post-hoc clipping on 35B.",
            expected_low=21.38,
            expected_high=21.54,
            cot_rf={
                "enable_delta_verifier_tool": False,
                "force_delta_verifier_tool": False,
                "enable_numeric_tool": True,
                "force_numeric_tool": True,
            },
            llm={},
            hdelta={},
        ),
        Candidate(
            name="verifier_only_default",
            literature_label="Chain-of-Verification",
            literature_url="https://arxiv.org/abs/2309.11495",
            change_summary="Remove the numeric tool and keep only the deterministic verifier under the current least-to-most stack.",
            aim="Test whether 35B already has enough arithmetic and mainly needs bounded final corrections.",
            expected_low=21.43,
            expected_high=21.57,
            cot_rf={
                "enable_numeric_tool": False,
                "force_numeric_tool": False,
                "enable_delta_verifier_tool": True,
                "force_delta_verifier_tool": True,
            },
            llm={},
            hdelta={},
        ),
        Candidate(
            name="react_evidence_optional_tools",
            literature_label="ReAct",
            literature_url="https://arxiv.org/abs/2210.03629",
            change_summary="Switch the reflection style to evidence-seeking ReAct and let the model decide when to call numeric or verifier tools.",
            aim="Encourage evidence-driven tool use instead of mandatory tool loops.",
            expected_low=21.36,
            expected_high=21.55,
            cot_rf={
                "reasoning_style": "react_evidence",
                "force_numeric_tool": False,
                "force_delta_verifier_tool": False,
            },
            llm={},
            hdelta={},
        ),
        Candidate(
            name="plan_and_solve_optional_tools",
            literature_label="Plan-and-Solve",
            literature_url="https://arxiv.org/abs/2305.04091",
            change_summary="Use plan-and-solve reflection while keeping numeric and verifier tools optional.",
            aim="Reduce missing-step errors without forcing every sample through the same tool sequence.",
            expected_low=21.38,
            expected_high=21.56,
            cot_rf={
                "reasoning_style": "plan_and_solve",
                "force_numeric_tool": False,
                "force_delta_verifier_tool": False,
            },
            llm={},
            hdelta={},
        ),
        Candidate(
            name="skill_tag_recent_high_error_tools",
            literature_label="Skill-KNN",
            literature_url="https://arxiv.org/abs/2305.14210",
            change_summary="Switch example selection to skill-tag-matched recent hard cases while keeping the current tool stack forced.",
            aim="Improve demonstration relevance without changing the tool contract.",
            expected_low=21.36,
            expected_high=21.53,
            cot_rf={
                "example_selection": "skill_tag_recent_high_error",
                "k_examples": 6,
                "skill_tag_min_overlap": 4,
            },
            llm={},
            hdelta={},
        ),
        Candidate(
            name="balanced_long_horizon_tools",
            literature_label="Active Example Selection",
            literature_url="https://arxiv.org/abs/2211.04486",
            change_summary="Balance the teaching pool across h20-heavy and h30-heavy failures under the current tool stack.",
            aim="Avoid overfitting the demonstration bank to one long-horizon failure mode.",
            expected_low=21.39,
            expected_high=21.56,
            cot_rf={
                "example_selection": "balanced_long_horizon",
                "k_examples": 6,
            },
            llm={},
            hdelta={},
        ),
        Candidate(
            name="counterexample_recent_high_error_tools",
            literature_label="Chain of Hindsight",
            literature_url="https://arxiv.org/abs/2302.02676",
            change_summary="Add one low-error freeze counterexample to the recent high-error teaching set while keeping the current tool stack.",
            aim="Teach the model both when to correct and when to leave the base path alone.",
            expected_low=21.37,
            expected_high=21.55,
            cot_rf={
                "include_counterexample_freeze": True,
                "counterexample_mode": "low_error",
                "counterexample_quantile": 0.35,
                "k_examples": 6,
            },
            llm={},
            hdelta={},
        ),
        Candidate(
            name="step_back_optional_tools_k6",
            literature_label="Step-Back Prompting",
            literature_url="https://arxiv.org/abs/2310.06117",
            change_summary="Restore step-back with 6 examples but leave numeric and verifier tools available rather than required.",
            aim="Recover the old 35B regime-level strength without forcing the clipping behavior that hurt the hybrid.",
            expected_low=21.30,
            expected_high=21.50,
            cot_rf={
                "reasoning_style": "step_back",
                "k_examples": 6,
                "force_numeric_tool": False,
                "force_delta_verifier_tool": False,
            },
            llm={},
            hdelta={},
        ),
        Candidate(
            name="step_back_numeric_optional_k6",
            literature_label="Step-Back Prompting + Program of Thoughts",
            literature_url="https://arxiv.org/abs/2310.06117",
            change_summary="Use step-back with 6 examples, enable numeric analysis as-needed, and remove the verifier entirely.",
            aim="See whether exact arithmetic complements regime-level reasoning while the verifier is the component causing over-regularization.",
            expected_low=21.28,
            expected_high=21.48,
            cot_rf={
                "reasoning_style": "step_back",
                "k_examples": 6,
                "enable_numeric_tool": True,
                "force_numeric_tool": False,
                "enable_delta_verifier_tool": False,
                "force_delta_verifier_tool": False,
            },
            llm={},
            hdelta={},
        ),
    ]

    _write_plan(out_dir, candidates)

    rows: list[dict[str, Any]] = []
    for idx, candidate in enumerate(candidates, start=1):
        print(
            f"[{idx}/{len(candidates)}] Running {candidate.name} "
            f"(expected {candidate.expected_low:.2f}-{candidate.expected_high:.2f}) ...",
            flush=True,
        )
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
                "delta_calibration": {"enabled": False},
                "cot_rf": candidate.cot_rf,
                **candidate.llm,
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
        metrics = _extract_candidate_metrics(
            run_dir=run_dir,
            baseline_llm_path=baseline_path_mse,
            baseline_horizons=baseline_horizons,
            ridge_path_mse=ridge_path_mse,
            old_best_path_mse=old_best_path_mse,
        )
        row = {
            "name": candidate.name,
            "literature_label": candidate.literature_label,
            "literature_url": candidate.literature_url,
            "change_summary": candidate.change_summary,
            "aim": candidate.aim,
            "expected_low": candidate.expected_low,
            "expected_high": candidate.expected_high,
            "expectation_result": _classify_expectation(
                metrics["llm_mse_path"],
                candidate.expected_low,
                candidate.expected_high,
            ),
            **metrics,
        }
        rows.append(row)
        pd.DataFrame(rows).to_csv(out_dir / "candidate_results_partial.csv", index=False)
        print(
            f"  -> {candidate.name}: llm_mse_path={metrics['llm_mse_path']:.6f} "
            f"(vs_baseline={metrics['delta_vs_baseline_llm']:+.6f}, "
            f"vs_old_best={metrics['delta_vs_old_best']:+.6f}, "
            f"expectation={row['expectation_result']})",
            flush=True,
        )

    _write_report(
        out_dir=out_dir,
        rows=rows,
        baseline_run_dir=baseline_run_dir,
        baseline_path_mse=baseline_path_mse,
        ridge_path_mse=ridge_path_mse,
        old_best_path_mse=old_best_path_mse,
    )
    print(f"Sweep results written to {out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
