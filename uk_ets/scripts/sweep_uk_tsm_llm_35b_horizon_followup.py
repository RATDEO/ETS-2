#!/usr/bin/env python3
"""Horizon-focused 35B follow-up sweep for UK TSM+LLM variants."""

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


def _load_apply_stats(run_dir: Path) -> tuple[int, str]:
    unique_apply = set()
    apply_samples_used = set()
    log_path = run_dir / "llm" / "logs" / "llm_calls.jsonl"
    with open(log_path, "r") as handle:
        for line in handle:
            obj = json.loads(line)
            if not str(obj.get("method", "")).endswith(":apply"):
                continue
            response = ((obj.get("response") or {}).get("content") or "").strip()
            if response:
                unique_apply.add(response)
            metadata = obj.get("metadata") or {}
            apply_samples_used.add(int(metadata.get("apply_samples", 1)))
    return len(unique_apply), ",".join(str(x) for x in sorted(apply_samples_used))


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
    unique_apply_responses, apply_samples_used = _load_apply_stats(run_dir)

    return {
        "run_dir": str(run_dir),
        "base_mse_path": float(tsm_base["mse_path"]),
        "llm_mse_path": float(llm_raw["mse_path"]),
        "delta_vs_tsm": float(llm_raw["mse_path"] - tsm_base["mse_path"]),
        "delta_vs_baseline_llm": float(llm_raw["mse_path"] - baseline_llm_path),
        "h5_delta_vs_tsm": float(llm_h.loc[5, "mse"] - tsm_h.loc[5, "mse"]),
        "h20_delta_vs_tsm": float(llm_h.loc[20, "mse"] - tsm_h.loc[20, "mse"]),
        "h30_delta_vs_tsm": float(llm_h.loc[30, "mse"] - tsm_h.loc[30, "mse"]),
        "h5_delta_vs_baseline": float(llm_h.loc[5, "mse"] - baseline_horizons[5]),
        "h20_delta_vs_baseline": float(llm_h.loc[20, "mse"] - baseline_horizons[20]),
        "h30_delta_vs_baseline": float(llm_h.loc[30, "mse"] - baseline_horizons[30]),
        "nonzero_count": int(np.count_nonzero(np.abs(deltas) > 1e-12)),
        "abs_mean_delta": float(np.abs(deltas).mean()),
        "mean_delta": float(deltas.mean()),
        "unique_apply_responses": int(unique_apply_responses),
        "apply_samples_used": apply_samples_used,
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
        "# UK ETS 35B Horizon Follow-Up Sweep",
        "",
        f"Generated: {datetime.now().isoformat()}",
        "",
        "## Baseline",
        "",
        f"- Baseline run: `{baseline_run_dir}`",
        f"- Baseline raw `{METHOD_NAME}` path MSE: `{baseline_path_mse:.6f}`",
        f"- Reference `linear_ridge` path MSE: `{ridge_path_mse:.6f}`",
        "",
        "## Candidate Results",
        "",
        "| Candidate | Change | Aim | Expected | Actual | Vs baseline | Outcome vs expected | h5 vs baseline | h20 vs baseline | h30 vs baseline | Apply samples | Unique apply |",
        "|---|---|---|---|---:|---:|---|---:|---:|---:|---|---:|",
    ]
    for row in rows:
        lines.append(
            "| {name} | {change_summary} | {aim} | {expected_low:.2f}-{expected_high:.2f} | {llm_mse_path:.6f} | {delta_vs_baseline_llm:+.6f} | {expectation_result} | {h5_delta_vs_baseline:+.6f} | {h20_delta_vs_baseline:+.6f} | {h30_delta_vs_baseline:+.6f} | {apply_samples_used} | {unique_apply_responses} |".format(
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
                f"- Aim: {best['aim']}",
                f"- Expected range: `{best['expected_low']:.2f}-{best['expected_high']:.2f}`",
                f"- Actual raw path MSE: `{best['llm_mse_path']:.6f}`",
                f"- Delta vs baseline raw LLM: `{best['delta_vs_baseline_llm']:+.6f}`",
                f"- Delta vs base TSM: `{best['delta_vs_tsm']:+.6f}`",
                f"- Delta vs ridge: `{best['llm_mse_path'] - ridge_path_mse:+.6f}`",
                f"- Outcome vs expectation: `{best['expectation_result']}`",
                f"- Run dir: `{best['run_dir']}`",
            ]
        )

    (out_dir / "summary.md").write_text("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a horizon-focused 35B UK TSM+LLM sweep.")
    parser.add_argument("--config", default="uk_ets/config/uk_ets_llm_35b_recent_high_error_180.yaml")
    parser.add_argument("--data-dir", default="uk_ets/Data_auto_uk")
    parser.add_argument("--output-root", default="reports/uk_ets_llm_35b_horizon_followup")
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
            name="horizon_specific_matching",
            change_summary="Enable horizon-specific matched-example selection within the recent high-error teaching pool.",
            aim="Give h5 more local analogs without sacrificing the recent hard-case evidence that is helping h20 and h30.",
            expected_low=21.65,
            expected_high=21.74,
            cot_rf={},
            llm={},
            hdelta={
                "horizon_specific_matching": True,
                "horizon_match_top_k": 2,
                "horizon_match_union_limit": 6,
            },
        ),
        Candidate(
            name="structured_h5_guard",
            change_summary="Require higher structured confidence before keeping h5 adjustments, while keeping h20/h30 medium-confidence capable.",
            aim="Reduce short-horizon overcorrection but preserve the larger medium and long-horizon gains.",
            expected_low=21.66,
            expected_high=21.76,
            cot_rf={},
            llm={},
            hdelta={
                "structured_enforce_sign": True,
                "structured_cap_by_guidance": True,
                "structured_min_confidence_by_horizon": {"h5": "high", "h20": "medium", "h30": "medium"},
                "structured_magnitude_scale": {"tiny": 0.20, "small": 0.45, "medium": 0.75},
                "structured_confidence_scale": {"medium": 0.85, "high": 1.0},
            },
        ),
        Candidate(
            name="h5_shrink_selective",
            change_summary="Shrink h5 case bounds and demand stronger sign agreement only for h5.",
            aim="Keep h5 actionable when evidence is clean, but cut its adjustment size and frequency when the signal is noisy.",
            expected_low=21.67,
            expected_high=21.76,
            cot_rf={},
            llm={},
            hdelta={
                "horizon_overrides": {
                    "h5": {
                        "case_min_sign_agreement": 0.75,
                        "case_bound_scale": 0.50,
                        "case_min_bound_pct": 0.05,
                    }
                }
            },
        ),
        Candidate(
            name="hmatch_plus_h5_shrink",
            change_summary="Combine horizon-specific matching with a tighter, more selective h5 bound.",
            aim="Highest-upside variant: use better per-horizon matches while keeping h5 from reacting too aggressively.",
            expected_low=21.60,
            expected_high=21.72,
            cot_rf={},
            llm={},
            hdelta={
                "horizon_specific_matching": True,
                "horizon_match_top_k": 2,
                "horizon_match_union_limit": 6,
                "horizon_overrides": {
                    "h5": {
                        "case_min_sign_agreement": 0.75,
                        "case_bound_scale": 0.50,
                        "case_min_bound_pct": 0.05,
                    }
                },
            },
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
