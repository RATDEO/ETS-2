#!/usr/bin/env python3
"""Literature-backed structural sweep for the UK 4B TSM+LLM pipeline."""

from __future__ import annotations

import argparse
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
    change_summary: str
    aim: str
    literature_basis: str
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


def _extract_candidate_metrics(
    run_dir: Path,
    checkpoint_llm_path: float,
    checkpoint_horizons: dict[int, float],
) -> dict[str, Any]:
    path_metrics = pd.read_csv(run_dir / "results" / "path_metrics.csv")
    horizon_metrics = pd.read_csv(run_dir / "results" / "metrics_by_horizon.csv")

    tsm_base = path_metrics[path_metrics["model"] == "tsm"].iloc[0]
    llm_raw = path_metrics[path_metrics["model"] == METHOD_NAME].iloc[0]

    tsm_h = horizon_metrics[horizon_metrics["model"] == "tsm"].set_index("horizon")
    llm_h = horizon_metrics[horizon_metrics["model"] == METHOD_NAME].set_index("horizon")

    pred_npz = np.load(run_dir / "predictions" / f"{METHOD_NAME}_pred_test_subset.npz")
    deltas = pred_npz["yhat"] - pred_npz["base_pred"]

    return {
        "run_dir": str(run_dir),
        "base_mse_path": float(tsm_base["mse_path"]),
        "llm_mse_path": float(llm_raw["mse_path"]),
        "delta_vs_tsm": float(llm_raw["mse_path"] - tsm_base["mse_path"]),
        "delta_vs_checkpoint_llm": float(llm_raw["mse_path"] - checkpoint_llm_path),
        "h5_delta_vs_tsm": float(llm_h.loc[5, "mse"] - tsm_h.loc[5, "mse"]),
        "h20_delta_vs_tsm": float(llm_h.loc[20, "mse"] - tsm_h.loc[20, "mse"]),
        "h30_delta_vs_tsm": float(llm_h.loc[30, "mse"] - tsm_h.loc[30, "mse"]),
        "h5_delta_vs_checkpoint": float(llm_h.loc[5, "mse"] - checkpoint_horizons[5]),
        "h20_delta_vs_checkpoint": float(llm_h.loc[20, "mse"] - checkpoint_horizons[20]),
        "h30_delta_vs_checkpoint": float(llm_h.loc[30, "mse"] - checkpoint_horizons[30]),
        "nonzero_count": int(np.count_nonzero(np.abs(deltas) > 1e-12)),
        "abs_mean_delta": float(np.abs(deltas).mean()),
        "mean_delta": float(deltas.mean()),
    }


def _classify_expectation(actual: float, expected_low: float, expected_high: float) -> str:
    if actual < expected_low:
        return "beat"
    if actual <= expected_high:
        return "within"
    return "missed"


def _write_report(
    out_dir: Path,
    rows: list[dict[str, Any]],
    checkpoint_run_dir: Path,
    checkpoint_path_mse: float,
    ridge_path_mse: float,
) -> None:
    rows_df = pd.DataFrame(rows)
    rows_df.to_csv(out_dir / "candidate_results.csv", index=False)

    lines = [
        "# UK ETS 4B Literature Structural Sweep",
        "",
        f"Generated: {datetime.now().isoformat()}",
        "",
        "## Checkpoint",
        "",
        f"- Base checkpoint run: `{checkpoint_run_dir}`",
        f"- Checkpoint raw `{METHOD_NAME}` path MSE: `{checkpoint_path_mse:.6f}`",
        f"- Reference `linear_ridge` path MSE: `{ridge_path_mse:.6f}`",
        "",
        "## Candidate Results",
        "",
        "| Candidate | Change | Literature basis | Aim | Expected | Actual | Vs checkpoint | Outcome | h5 vs checkpoint | h20 vs checkpoint | h30 vs checkpoint |",
        "|---|---|---|---|---|---:|---:|---|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {name} | {change_summary} | {literature_basis} | {aim} | {expected_low:.2f}-{expected_high:.2f} | {llm_mse_path:.6f} | {delta_vs_checkpoint_llm:+.6f} | {expectation_result} | {h5_delta_vs_checkpoint:+.6f} | {h20_delta_vs_checkpoint:+.6f} | {h30_delta_vs_checkpoint:+.6f} |".format(
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
                f"- Change: {best['change_summary']}",
                f"- Literature basis: {best['literature_basis']}",
                f"- Aim: {best['aim']}",
                f"- Expected range: `{best['expected_low']:.2f}-{best['expected_high']:.2f}`",
                f"- Actual raw path MSE: `{best['llm_mse_path']:.6f}`",
                f"- Delta vs checkpoint raw LLM: `{best['delta_vs_checkpoint_llm']:+.6f}`",
                f"- Delta vs base TSM: `{best['delta_vs_tsm']:+.6f}`",
                f"- Delta vs ridge: `{best['llm_mse_path'] - ridge_path_mse:+.6f}`",
                f"- Outcome vs expectation: `{best['expectation_result']}`",
                f"- Run dir: `{best['run_dir']}`",
            ]
        )

    (out_dir / "summary.md").write_text("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a literature-backed structural UK 4B sweep.")
    parser.add_argument("--config", default="uk_ets/config/uk_ets_llm_4b_recent_high_error_180_k4.yaml")
    parser.add_argument("--data-dir", default="uk_ets/Data_auto_uk")
    parser.add_argument("--output-root", default="reports/uk_ets_llm_4b_literature_structural")
    parser.add_argument(
        "--checkpoint-run",
        default="runs/20260310_195100_69e04e",
        help="Existing 4B run used as the structural checkpoint reference.",
    )
    args = parser.parse_args()

    config_path = (PROJECT_ROOT / args.config).resolve()
    data_dir = (PROJECT_ROOT / args.data_dir).resolve()
    checkpoint_run_dir = (PROJECT_ROOT / args.checkpoint_run).resolve()

    checkpoint_path_mse = _load_path_metric(checkpoint_run_dir, METHOD_NAME)
    checkpoint_horizons = _load_horizon_mse(checkpoint_run_dir, METHOD_NAME)
    ridge_path_mse = _load_path_metric(checkpoint_run_dir, "linear_ridge")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = (PROJECT_ROOT / args.output_root / timestamp).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    candidates = [
        Candidate(
            name="control_recent_high_error_k4",
            change_summary="Re-run the current 4B winner unchanged as a control.",
            aim="Verify the new structural code path reproduces the current checkpoint before changing evidence policy.",
            literature_basis="Control",
            expected_low=22.14,
            expected_high=22.24,
            cot_rf={},
            llm={},
            hdelta={},
        ),
        Candidate(
            name="recent_long_error_k4",
            change_summary="Replace path-MSE retrieval with recent long-horizon error retrieval centered on h20/h30.",
            aim="Bias the demo bank toward the horizons where the 4B LLM actually adds value.",
            literature_basis="Time-series LLM papers emphasize horizon/task alignment more than generic ICL breadth.",
            expected_low=22.02,
            expected_high=22.14,
            cot_rf={"example_selection": "recent_long_error", "lookback_days": 180, "k_examples": 4},
            llm={},
            hdelta={},
        ),
        Candidate(
            name="recent_consistent_long_error_k4",
            change_summary="Keep only recent long-horizon cases whose h20 and h30 error signs agree.",
            aim="Remove mixed-tail demonstrations that teach contradictory long-horizon corrections.",
            literature_basis="Comparable demonstration work argues that conflicting examples degrade small-model in-context learning.",
            expected_low=21.98,
            expected_high=22.12,
            cot_rf={"example_selection": "recent_consistent_long_error", "lookback_days": 180, "k_examples": 4},
            llm={},
            hdelta={},
        ),
        Candidate(
            name="balanced_long_horizon_k4",
            change_summary="Split the evidence set between h20-heavy and h30-heavy mistakes.",
            aim="Avoid letting one long horizon dominate the entire teaching set.",
            literature_basis="Demonstration-selection literature repeatedly finds coverage and diversity help more than raw score ranking alone.",
            expected_low=22.00,
            expected_high=22.15,
            cot_rf={"example_selection": "balanced_long_horizon", "lookback_days": 180, "k_examples": 4},
            llm={},
            hdelta={},
        ),
        Candidate(
            name="prototype_recent_long_error_k4",
            change_summary="Take prototypes from the top recent long-error pool instead of the top few ranked examples directly.",
            aim="Reduce redundant demonstrations and improve coverage of distinct correction patterns.",
            literature_basis="Active-example and retrieval papers show redundancy reduction is important in small-context ICL.",
            expected_low=22.05,
            expected_high=22.18,
            cot_rf={
                "example_selection": "prototype_recent_long_error",
                "lookback_days": 180,
                "k_examples": 4,
                "prototype_pool_multiplier": 4,
            },
            llm={},
            hdelta={},
        ),
        Candidate(
            name="skill_tag_recent_high_error_k4",
            change_summary="Filter recent hard examples through coarse regime-skill tags before ranking them.",
            aim="Retrieve demonstrations that match the same market regime, not just the same error magnitude.",
            literature_basis="Skill-KNN suggests abstract skill matching beats surface-level similarity for few-shot selection.",
            expected_low=22.04,
            expected_high=22.18,
            cot_rf={
                "example_selection": "skill_tag_recent_high_error",
                "lookback_days": 180,
                "k_examples": 4,
                "skill_tag_min_overlap": 4,
            },
            llm={},
            hdelta={},
        ),
        Candidate(
            name="feedback_recent_high_error_k4",
            change_summary="Add explicit hindsight feedback sentences to each recent high-error teaching example.",
            aim="Convert raw historical mistakes into short natural-language lessons the 4B model can reuse more easily.",
            literature_basis="Chain of Hindsight shows compact feedback can outperform raw trajectories alone.",
            expected_low=21.98,
            expected_high=22.12,
            cot_rf={
                "include_hindsight_feedback": True,
                "k_examples": 4,
                "example_selection": "recent_high_error",
                "lookback_days": 180,
            },
            llm={},
            hdelta={},
        ),
        Candidate(
            name="support_plus_freeze_counterexample_k4",
            change_summary="Augment recent high-error support examples with one similar low-error freeze counterexample.",
            aim="Teach the 4B model both when to correct and when not to create tail drift.",
            literature_basis="Supportive/comparable demonstration work argues explicit counterexamples sharpen decision boundaries.",
            expected_low=21.96,
            expected_high=22.10,
            cot_rf={
                "include_counterexample_freeze": True,
                "counterexample_mode": "low_error",
                "counterexample_quantile": 0.35,
                "k_examples": 4,
                "example_selection": "recent_high_error",
                "lookback_days": 180,
            },
            llm={},
            hdelta={},
        ),
        Candidate(
            name="ridge_context_recent_high_error_k4",
            change_summary="Keep recent high-error retrieval but show the ridge forecast deltas as a teacher signal in the examples and current case.",
            aim="Let the LLM see what a stronger non-LLM forecaster would do without changing the base TSM path directly.",
            literature_basis="Teacher/distillation literature for small models suggests auxiliary teacher signals help more than added prompt complexity.",
            expected_low=21.92,
            expected_high=22.08,
            cot_rf={
                "include_aux_teacher_summary": True,
                "include_current_aux_teacher_summary": True,
                "aux_teacher_model": "linear_ridge",
                "aux_teacher_horizons": [5, 20, 30],
                "k_examples": 4,
                "example_selection": "recent_high_error",
                "lookback_days": 180,
            },
            llm={},
            hdelta={},
        ),
        Candidate(
            name="ridge_gain_selection_k4",
            change_summary="Retrieve examples where the ridge teacher most improved on the long horizon, and show teacher summaries.",
            aim="Prioritize demonstrations whose correction pattern is both difficult and actually recoverable by a stronger model.",
            literature_basis="Active example selection favors demonstrations with the highest expected utility, not the largest raw loss alone.",
            expected_low=21.88,
            expected_high=22.06,
            cot_rf={
                "include_aux_teacher_summary": True,
                "include_current_aux_teacher_summary": True,
                "aux_teacher_model": "linear_ridge",
                "aux_teacher_horizons": [5, 20, 30],
                "example_selection": "ridge_gain",
                "lookback_days": 180,
                "k_examples": 4,
            },
            llm={},
            hdelta={},
        ),
        Candidate(
            name="ridge_gain_plus_counterexample_k4",
            change_summary="Combine ridge-gain retrieval with one similar low-gain counterexample.",
            aim="Keep the teacher-guided upside while explicitly teaching the model when not to trust a correction pattern.",
            literature_basis="Teacher-guided selection plus comparable demos is the highest-upside structural combination from the reviewed literature.",
            expected_low=21.86,
            expected_high=22.05,
            cot_rf={
                "include_aux_teacher_summary": True,
                "include_current_aux_teacher_summary": True,
                "aux_teacher_model": "linear_ridge",
                "aux_teacher_horizons": [5, 20, 30],
                "example_selection": "ridge_gain",
                "lookback_days": 180,
                "k_examples": 4,
                "include_counterexample_freeze": True,
                "counterexample_mode": "low_gain",
                "counterexample_quantile": 0.35,
            },
            llm={},
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
            checkpoint_llm_path=checkpoint_path_mse,
            checkpoint_horizons=checkpoint_horizons,
        )
        row = {
            "name": candidate.name,
            "change_summary": candidate.change_summary,
            "aim": candidate.aim,
            "literature_basis": candidate.literature_basis,
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
        print(
            f"  -> {candidate.name}: llm_mse_path={metrics['llm_mse_path']:.6f} "
            f"(vs_checkpoint={metrics['delta_vs_checkpoint_llm']:+.6f}, "
            f"expectation={row['expectation_result']})",
            flush=True,
        )

    _write_report(
        out_dir=out_dir,
        rows=rows,
        checkpoint_run_dir=checkpoint_run_dir,
        checkpoint_path_mse=checkpoint_path_mse,
        ridge_path_mse=ridge_path_mse,
    )
    print(f"Sweep results written to {out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
