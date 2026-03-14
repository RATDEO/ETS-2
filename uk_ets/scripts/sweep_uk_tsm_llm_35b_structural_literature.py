#!/usr/bin/env python3
"""Structural literature-backed 35B sweep for UK TSM+LLM variants."""

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
    apply_samples_used = set()
    log_path = run_dir / "llm" / "logs" / "llm_calls.jsonl"
    with open(log_path, "r") as handle:
        for line in handle:
            obj = json.loads(line)
            method = str(obj.get("method", ""))
            response = ((obj.get("response") or {}).get("content") or "").strip()
            metadata = obj.get("metadata") or {}
            if method.endswith(":apply"):
                if response:
                    unique_apply.add(response)
                apply_samples_used.add(int(metadata.get("apply_samples", 1)))
            if method.endswith(":reflect"):
                if response:
                    unique_reflect.add(response)
                reflect_samples.append(int(metadata.get("reflect_samples", 1)))
                reflect_valid_samples.append(int(metadata.get("reflect_valid_samples", 0)))
    return {
        "unique_apply_responses": len(unique_apply),
        "unique_reflect_responses": len(unique_reflect),
        "apply_samples_used": ",".join(str(x) for x in sorted(apply_samples_used)),
        "reflect_samples_used": ",".join(str(x) for x in sorted(set(reflect_samples))),
        "mean_reflect_valid_samples": float(np.mean(reflect_valid_samples)) if reflect_valid_samples else 0.0,
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

    tsm_h = horizon_metrics[horizon_metrics["model"] == "tsm"].set_index("horizon")
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
        "mean_delta": float(deltas.mean()),
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
        "# UK ETS 35B Structural Literature Sweep",
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
            "| Candidate | Actual | Vs baseline | Vs ridge | Outcome vs expected | h5 vs baseline | h20 vs baseline | h30 vs baseline | Reflect samples | Mean valid reflect | Unique reflect | Unique apply |",
            "|---|---:|---:|---:|---|---:|---:|---:|---|---:|---:|---:|",
        ]
    )
    for row in rows:
        lines.append(
            "| {name} | {llm_mse_path:.6f} | {delta_vs_baseline_llm:+.6f} | {delta_vs_ridge:+.6f} | {expectation_result} | {h5_delta_vs_baseline:+.6f} | {h20_delta_vs_baseline:+.6f} | {h30_delta_vs_baseline:+.6f} | {reflect_samples_used} | {mean_reflect_valid_samples:.2f} | {unique_reflect_responses} | {unique_apply_responses} |".format(
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
                f"- Delta vs baseline raw LLM: `{best['delta_vs_baseline_llm']:+.6f}`",
                f"- Delta vs base TSM: `{best['delta_vs_tsm']:+.6f}`",
                f"- Delta vs ridge: `{best['delta_vs_ridge']:+.6f}`",
                f"- Outcome vs expectation: `{best['expectation_result']}`",
                f"- Run dir: `{best['run_dir']}`",
            ]
        )

    (out_dir / "summary.md").write_text("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a structural literature-backed 35B UK TSM+LLM sweep.")
    parser.add_argument("--config", default="uk_ets/config/uk_ets_llm_35b_recent_high_error_180.yaml")
    parser.add_argument("--data-dir", default="uk_ets/Data_auto_uk")
    parser.add_argument("--output-root", default="reports/uk_ets_llm_35b_structural_literature")
    parser.add_argument(
        "--baseline-run",
        default="runs/20260309_151421_2aca41",
        help="Existing best run used as the raw LLM reference.",
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
            name="self_refine_verifier",
            literature_label="Self-Refine",
            literature_url="https://arxiv.org/abs/2303.17651",
            change_summary="Add an explicit internal critique-and-revise reflection contract plus a verifier-style numeric apply step.",
            aim="Reduce unsupported horizon moves by forcing the model to criticize its own first-pass plan before output.",
            expected_low=21.60,
            expected_high=21.74,
            cot_rf={"reasoning_style": "self_refine", "apply_style": "verifier_program"},
            llm={},
            hdelta={},
        ),
        Candidate(
            name="chain_of_verification",
            literature_label="Chain-of-Verification",
            literature_url="https://arxiv.org/abs/2309.11495",
            change_summary="Require each horizon decision to be verified against the matched examples before adjustment.",
            aim="Suppress hallucinated sign changes and keep only evidence-backed horizon moves.",
            expected_low=21.58,
            expected_high=21.72,
            cot_rf={"reasoning_style": "chain_of_verification", "apply_style": "verifier_program"},
            llm={},
            hdelta={},
        ),
        Candidate(
            name="step_back_regime",
            literature_label="Step-Back Prompting",
            literature_url="https://arxiv.org/abs/2310.06117",
            change_summary="Force the reflection stage to infer a higher-level regime first, then map it to horizon decisions.",
            aim="Improve medium and long-horizon coherence by reasoning at the regime level before local deltas.",
            expected_low=21.63,
            expected_high=21.78,
            cot_rf={"reasoning_style": "step_back", "apply_style": "program_of_thought"},
            llm={},
            hdelta={},
        ),
        Candidate(
            name="skeleton_then_fill",
            literature_label="Skeleton-of-Thought",
            literature_url="https://arxiv.org/abs/2307.15337",
            change_summary="Separate the action skeleton from the sign and magnitude fill-in step.",
            aim="Reduce over-adjustment by deciding freeze-versus-adjust structure before sizing any move.",
            expected_low=21.66,
            expected_high=21.80,
            cot_rf={"reasoning_style": "skeleton", "apply_style": "minimal"},
            llm={},
            hdelta={},
        ),
        Candidate(
            name="self_ask_horizons",
            literature_label="Self-Ask",
            literature_url="https://arxiv.org/abs/2210.03350",
            change_summary="Break the reflection stage into explicit horizon-level self-questions before output.",
            aim="Make the h5 and h20 decisions more deliberate and less coupled to the dominant long-horizon narrative.",
            expected_low=21.64,
            expected_high=21.79,
            cot_rf={"reasoning_style": "self_ask", "apply_style": "program_of_thought"},
            llm={},
            hdelta={},
        ),
        Candidate(
            name="react_evidence",
            literature_label="ReAct",
            literature_url="https://arxiv.org/abs/2210.03629",
            change_summary="Use an explicit observe-reason-act structure over the matched-example evidence.",
            aim="Tighten the link between retrieved evidence and bounded horizon actions.",
            expected_low=21.60,
            expected_high=21.75,
            cot_rf={"reasoning_style": "react_evidence", "apply_style": "verifier_program"},
            llm={},
            hdelta={},
        ),
        Candidate(
            name="rarr_attribution",
            literature_label="RARR",
            literature_url="https://arxiv.org/abs/2210.08726",
            change_summary="Force explicit evidence attribution in horizon reasons and use a citation-sensitive apply stage.",
            aim="Cut unsupported corrections by making every non-zero move justify itself from the matched examples.",
            expected_low=21.56,
            expected_high=21.72,
            cot_rf={"reasoning_style": "rarr_attribution", "apply_style": "citation_bounded"},
            llm={},
            hdelta={},
        ),
        Candidate(
            name="self_rag_critique",
            literature_label="Self-RAG",
            literature_url="https://arxiv.org/abs/2310.11511",
            change_summary="Make the model retrieve the strongest matched-example evidence, critique it, then decide.",
            aim="Improve robustness by filtering adjustments through an evidence-sufficiency check rather than direct prompting alone.",
            expected_low=21.55,
            expected_high=21.71,
            cot_rf={"reasoning_style": "self_rag", "apply_style": "citation_bounded"},
            llm={},
            hdelta={},
        ),
        Candidate(
            name="tree_of_thought_consensus",
            literature_label="Tree of Thoughts",
            literature_url="https://arxiv.org/abs/2305.10601",
            change_summary="Sample three reflection plans at non-zero temperature and aggregate them conservatively before apply.",
            aim="Search over multiple horizon plans and keep only consensus structure that survives aggregation.",
            expected_low=21.50,
            expected_high=21.70,
            cot_rf={
                "reasoning_style": "tree_of_thought",
                "apply_style": "verifier_program",
                "reflect_samples": 3,
                "reflect_aggregation": "conservative_majority",
                "reflect_temperature": 0.2,
            },
            llm={"cache_enabled": False},
            hdelta={},
        ),
        Candidate(
            name="debate_consensus",
            literature_label="Multi-Agent Debate",
            literature_url="https://arxiv.org/abs/2305.19118",
            change_summary="Sample three reflection drafts under an internal debate contract and aggregate only the conservative consensus.",
            aim="Retain only the horizon decisions that survive a skeptic-versus-adjuster conflict.",
            expected_low=21.52,
            expected_high=21.72,
            cot_rf={
                "reasoning_style": "debate",
                "apply_style": "verifier_program",
                "reflect_samples": 3,
                "reflect_aggregation": "conservative_majority",
                "reflect_temperature": 0.15,
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
