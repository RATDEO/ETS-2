#!/usr/bin/env python3
"""Targeted 35B follow-up sweep for UK TSM+LLM variants."""

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
    unique_apply_responses, apply_samples_used = _load_apply_stats(run_dir)

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
        "unique_apply_responses": int(unique_apply_responses),
        "apply_samples_used": apply_samples_used,
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
        "# UK ETS 35B Targeted Follow-Up Sweep",
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
        "| Candidate | Change | Aim | Expected | Actual | Vs checkpoint | Outcome vs expected | h5 vs checkpoint | h20 vs checkpoint | h30 vs checkpoint | Apply samples | Unique apply |",
        "|---|---|---|---|---:|---:|---|---:|---:|---:|---|---:|",
    ]
    for row in rows:
        lines.append(
            "| {name} | {change_summary} | {aim} | {expected_low:.2f}-{expected_high:.2f} | {llm_mse_path:.6f} | {delta_vs_checkpoint_llm:+.6f} | {expectation_result} | {h5_delta_vs_checkpoint:+.6f} | {h20_delta_vs_checkpoint:+.6f} | {h30_delta_vs_checkpoint:+.6f} | {apply_samples_used} | {unique_apply_responses} |".format(
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
                f"- Delta vs checkpoint raw LLM: `{best['delta_vs_checkpoint_llm']:+.6f}`",
                f"- Delta vs base TSM: `{best['delta_vs_tsm']:+.6f}`",
                f"- Delta vs ridge: `{best['llm_mse_path'] - ridge_path_mse:+.6f}`",
                f"- Outcome vs expectation: `{best['expectation_result']}`",
                f"- Run dir: `{best['run_dir']}`",
            ]
        )

    (out_dir / "summary.md").write_text("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a targeted 35B UK TSM+LLM follow-up sweep.")
    parser.add_argument("--config", default="uk_ets/config/uk_ets_llm_35b_checkpoint.yaml")
    parser.add_argument("--data-dir", default="uk_ets/Data_auto_uk")
    parser.add_argument("--output-root", default="reports/uk_ets_llm_35b_followup")
    parser.add_argument(
        "--checkpoint-run",
        default="runs/20260309_122028_0e017f",
        help="Existing checkpoint run used as the raw LLM reference.",
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
            name="hybrid_recent180",
            change_summary="Keep similarity-error hybrid selection, but restrict the teaching pool to the most recent 180 days.",
            aim="Test whether shorter-horizon exemplar locality helps the 35B model without giving up structured similarity matching.",
            expected_low=22.10,
            expected_high=22.18,
            cot_rf={"lookback_days": 180},
            llm={},
            hdelta={},
        ),
        Candidate(
            name="recent_high_error_180",
            change_summary="Switch teaching examples to recent high-error cases from the last 180 days.",
            aim="Bias the model toward the recent UK failure modes where the base TSM is least reliable.",
            expected_low=22.06,
            expected_high=22.16,
            cot_rf={"example_selection": "recent_high_error", "lookback_days": 180},
            llm={},
            hdelta={},
        ),
        Candidate(
            name="self_consistency_k3",
            change_summary="Keep the checkpoint prompt stack, but aggregate 3 apply-stage samples with a median.",
            aim="Reduce numeric apply jitter while preserving the current reasoning and example policy.",
            expected_low=22.08,
            expected_high=22.16,
            cot_rf={"apply_samples": 3, "apply_aggregation": "median", "apply_temperature": 0.25},
            llm={"cache_enabled": False},
            hdelta={},
        ),
        Candidate(
            name="recent_high_error_180_k8",
            change_summary="Use recent high-error teaching examples and increase the evidence set from 6 to 8 examples.",
            aim="Exploit the 35B context window with more recent hard cases while keeping the evidence regime-local.",
            expected_low=22.02,
            expected_high=22.12,
            cot_rf={"example_selection": "recent_high_error", "lookback_days": 180, "k_examples": 8},
            llm={},
            hdelta={},
        ),
        Candidate(
            name="recent_high_error_180_sc3",
            change_summary="Combine recent high-error teaching examples with 3-sample median apply self-consistency.",
            aim="Combine the strongest example-pool shift with apply-stage variance reduction for the highest upside run.",
            expected_low=21.98,
            expected_high=22.10,
            cot_rf={
                "example_selection": "recent_high_error",
                "lookback_days": 180,
                "apply_samples": 3,
                "apply_aggregation": "median",
                "apply_temperature": 0.25,
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
        metrics = _extract_candidate_metrics(run_dir, checkpoint_path_mse, checkpoint_horizons)
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
            f"(vs_checkpoint={row['delta_vs_checkpoint_llm']:+.6f}, expectation={row['expectation_result']})",
            flush=True,
        )

    rows.sort(key=lambda row: row["llm_mse_path"])
    _write_report(out_dir, rows, checkpoint_run_dir, checkpoint_path_mse, ridge_path_mse)
    print(f"Sweep results written to {out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
