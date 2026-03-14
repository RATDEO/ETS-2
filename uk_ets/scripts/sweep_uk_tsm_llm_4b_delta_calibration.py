#!/usr/bin/env python3
"""Delta-calibration sweep for UK 4B TSM+LLM."""

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


RAW_METHOD_NAME = "TSM+LLM-COT-RF-HDELTA"
UK_RETRIEVAL_FEATURES = [
    "target_range_pct",
    "target_volume",
    "is_auction_day",
    "uk_icap_primary_print_day",
    "uk_icap_secondary_print_day",
]


@dataclass(frozen=True)
class Candidate:
    name: str
    result_model_name: str
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


def _extract_candidate_metrics(
    run_dir: Path,
    result_model_name: str,
    checkpoint_llm_path: float,
    checkpoint_horizons: dict[int, float],
) -> dict[str, Any]:
    path_metrics = pd.read_csv(run_dir / "results" / "path_metrics.csv")
    horizon_metrics = pd.read_csv(run_dir / "results" / "metrics_by_horizon.csv")

    tsm_base = path_metrics[path_metrics["model"] == "tsm"].iloc[0]
    llm_row = path_metrics[path_metrics["model"] == result_model_name].iloc[0]

    tsm_h = horizon_metrics[horizon_metrics["model"] == "tsm"].set_index("horizon")
    llm_h = horizon_metrics[horizon_metrics["model"] == result_model_name].set_index("horizon")

    npz_name = f"{RAW_METHOD_NAME}_pred_test_subset.npz"
    pred_npz = np.load(run_dir / "predictions" / npz_name)
    deltas = pred_npz["yhat"] - pred_npz["base_pred"]

    return {
        "run_dir": str(run_dir),
        "base_mse_path": float(tsm_base["mse_path"]),
        "llm_mse_path": float(llm_row["mse_path"]),
        "delta_vs_tsm": float(llm_row["mse_path"] - tsm_base["mse_path"]),
        "delta_vs_checkpoint_llm": float(llm_row["mse_path"] - checkpoint_llm_path),
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
        "# UK ETS 4B Delta Calibration Sweep",
        "",
        f"Generated: {datetime.now().isoformat()}",
        "",
        "## Checkpoint",
        "",
        f"- Base checkpoint run: `{checkpoint_run_dir}`",
        f"- Checkpoint raw `{RAW_METHOD_NAME}` path MSE: `{checkpoint_path_mse:.6f}`",
        f"- Reference `linear_ridge` path MSE: `{ridge_path_mse:.6f}`",
        "",
        "## Candidate Results",
        "",
        "| Candidate | Reported model | Change | Aim | Expected | Actual | Vs checkpoint | Outcome | h5 vs checkpoint | h20 vs checkpoint | h30 vs checkpoint |",
        "|---|---|---|---|---|---:|---:|---|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {name} | {result_model_name} | {change_summary} | {aim} | {expected_low:.2f}-{expected_high:.2f} | {llm_mse_path:.6f} | {delta_vs_checkpoint_llm:+.6f} | {expectation_result} | {h5_delta_vs_checkpoint:+.6f} | {h20_delta_vs_checkpoint:+.6f} | {h30_delta_vs_checkpoint:+.6f} |".format(
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
                f"- Reported model: `{best['result_model_name']}`",
                f"- Actual path MSE: `{best['llm_mse_path']:.6f}`",
                f"- Delta vs checkpoint: `{best['delta_vs_checkpoint_llm']:+.6f}`",
                f"- Delta vs base TSM: `{best['delta_vs_tsm']:+.6f}`",
                f"- Delta vs ridge: `{best['llm_mse_path'] - ridge_path_mse:+.6f}`",
                f"- Run dir: `{best['run_dir']}`",
            ]
        )

    (out_dir / "summary.md").write_text("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a UK 4B delta-calibration sweep.")
    parser.add_argument("--config", default="uk_ets/config/uk_ets_llm_4b_recent_high_error_180_k4.yaml")
    parser.add_argument("--data-dir", default="uk_ets/Data_auto_uk")
    parser.add_argument("--output-root", default="reports/uk_ets_llm_4b_delta_calibration")
    parser.add_argument("--checkpoint-run", default="runs/20260310_195100_69e04e")
    args = parser.parse_args()

    config_path = (PROJECT_ROOT / args.config).resolve()
    data_dir = (PROJECT_ROOT / args.data_dir).resolve()
    checkpoint_run_dir = (PROJECT_ROOT / args.checkpoint_run).resolve()

    checkpoint_path_mse = _load_path_metric(checkpoint_run_dir, RAW_METHOD_NAME)
    checkpoint_horizons = _load_horizon_mse(checkpoint_run_dir, RAW_METHOD_NAME)
    ridge_path_mse = _load_path_metric(checkpoint_run_dir, "linear_ridge")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = (PROJECT_ROOT / args.output_root / timestamp).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    candidates = [
        Candidate(
            name="control_recent_high_error_k4",
            result_model_name=RAW_METHOD_NAME,
            change_summary="Re-run the current 4B checkpoint without post-LLM calibration.",
            aim="Verify the new calibration-capable runner reproduces the checkpoint exactly before shrinking any horizons.",
            expected_low=22.14,
            expected_high=22.24,
            cot_rf={},
            llm={},
            hdelta={},
        ),
        Candidate(
            name="recent_high_error_k4_delta_shared_h20_h30",
            result_model_name=f"{RAW_METHOD_NAME}_delta_calibrated",
            change_summary="Apply one shared validation-trained shrink factor to the `h20` and `h30` deltas only.",
            aim="Keep the current checkpoint structure, but reduce any long-horizon overshoot without disturbing early horizons.",
            expected_low=22.10,
            expected_high=22.20,
            cot_rf={},
            llm={
                "delta_calibration": {
                    "enabled": True,
                    "target_horizons": [20, 30],
                    "shared": True,
                    "min_scale": 0.0,
                    "max_scale": 1.0,
                }
            },
            hdelta={},
        ),
        Candidate(
            name="recent_high_error_k4_delta_per_h5_h20_h30",
            result_model_name=f"{RAW_METHOD_NAME}_delta_calibrated",
            change_summary="Fit separate validation scales for `h5`, `h20`, and `h30` while leaving the rest of the path untouched.",
            aim="Test whether the 4B path needs horizon-specific trust rather than a single long-horizon shrink factor.",
            expected_low=22.08,
            expected_high=22.20,
            cot_rf={},
            llm={
                "delta_calibration": {
                    "enabled": True,
                    "target_horizons": [5, 20, 30],
                    "shared": False,
                    "min_scale": 0.0,
                    "max_scale": 1.0,
                }
            },
            hdelta={},
        ),
        Candidate(
            name="utility_mmr_augmented_delta_shared_h20_h30",
            result_model_name=f"{RAW_METHOD_NAME}_delta_calibrated",
            change_summary="Use the best structural retrieval challenger from the prior sweep, then apply shared long-horizon delta shrinkage.",
            aim="See whether the utility-MMR variant can recover its small `h5` gain once `h20/h30` are calibrated down on validation.",
            expected_low=22.08,
            expected_high=22.20,
            cot_rf={
                "example_selection": "utility_mmr",
                "lookback_days": 180,
                "k_examples": 4,
                "retrieval_feature_columns": UK_RETRIEVAL_FEATURES,
                "retrieval_max_features": len(UK_RETRIEVAL_FEATURES),
                "augment_case_profiles_with_retrieval": True,
                "include_example_exogenous_summary": True,
                "example_exogenous_max_features": 4,
            },
            llm={
                "delta_calibration": {
                    "enabled": True,
                    "target_horizons": [20, 30],
                    "shared": True,
                    "min_scale": 0.0,
                    "max_scale": 1.0,
                }
            },
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
        metrics = _extract_candidate_metrics(
            run_dir,
            candidate.result_model_name,
            checkpoint_path_mse,
            checkpoint_horizons,
        )
        row = {
            "name": candidate.name,
            "result_model_name": candidate.result_model_name,
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
