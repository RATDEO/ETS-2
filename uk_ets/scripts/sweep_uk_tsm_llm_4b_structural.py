#!/usr/bin/env python3
"""Structural UK 4B sweep focused on retrieval and teaching-example context."""

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
        "# UK ETS 4B Structural Sweep",
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
        "| Candidate | Change | Aim | Expected | Actual | Vs checkpoint | Outcome vs expected | h5 vs checkpoint | h20 vs checkpoint | h30 vs checkpoint | Nonzero deltas |",
        "|---|---|---|---|---:|---:|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {name} | {change_summary} | {aim} | {expected_low:.2f}-{expected_high:.2f} | {llm_mse_path:.6f} | {delta_vs_checkpoint_llm:+.6f} | {expectation_result} | {h5_delta_vs_checkpoint:+.6f} | {h20_delta_vs_checkpoint:+.6f} | {h30_delta_vs_checkpoint:+.6f} | {nonzero_count} |".format(
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
    parser = argparse.ArgumentParser(description="Run a structural UK 4B TSM+LLM sweep.")
    parser.add_argument("--config", default="uk_ets/config/uk_ets_llm_4b_recent_high_error_180_k4.yaml")
    parser.add_argument("--data-dir", default="uk_ets/Data_auto_uk")
    parser.add_argument("--output-root", default="reports/uk_ets_llm_4b_structural")
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
            change_summary="Re-run the current 4B winner unchanged as a control on the new pipeline code path.",
            aim="Confirm the structural runner changes preserve the existing recent-high-error baseline before adding retrieval/context layers.",
            expected_low=22.14,
            expected_high=22.26,
            cot_rf={},
            llm={},
            hdelta={},
        ),
        Candidate(
            name="recent_high_error_k4_example_exo",
            change_summary="Keep recent high-error selection, but attach UK exogenous summaries to each teaching example.",
            aim="Let the 4B model compare hard historical fixes with the matching UK auction and ICAP state instead of only prices and forecasts.",
            expected_low=22.08,
            expected_high=22.23,
            cot_rf={
                "include_example_exogenous_summary": True,
                "example_exogenous_max_features": 4,
                "retrieval_feature_columns": UK_RETRIEVAL_FEATURES,
                "retrieval_max_features": len(UK_RETRIEVAL_FEATURES),
            },
            llm={},
            hdelta={},
        ),
        Candidate(
            name="event_similarity_uk_context_k4",
            change_summary="Switch retrieval to UK event similarity using auction, volume, and ICAP print-state features.",
            aim="Select demonstrations by market-state similarity rather than only forecast error, while keeping the 4B prompt compact.",
            expected_low=22.10,
            expected_high=22.26,
            cot_rf={
                "example_selection": "event_similarity",
                "lookback_days": 180,
                "k_examples": 4,
                "retrieval_feature_columns": UK_RETRIEVAL_FEATURES,
                "retrieval_max_features": len(UK_RETRIEVAL_FEATURES),
            },
            llm={},
            hdelta={},
        ),
        Candidate(
            name="event_similarity_uk_context_k4_example_exo",
            change_summary="Combine UK event-similarity retrieval with exogenous summaries on the retrieved teaching examples.",
            aim="Align both retrieval and prompt evidence around the same UK market-state cues instead of mixing price-only retrieval with context-free examples.",
            expected_low=22.04,
            expected_high=22.20,
            cot_rf={
                "example_selection": "event_similarity",
                "lookback_days": 180,
                "k_examples": 4,
                "retrieval_feature_columns": UK_RETRIEVAL_FEATURES,
                "retrieval_max_features": len(UK_RETRIEVAL_FEATURES),
                "include_example_exogenous_summary": True,
                "example_exogenous_max_features": 4,
            },
            llm={},
            hdelta={},
        ),
        Candidate(
            name="similarity_error_hybrid_augmented_uk_context_k4",
            change_summary="Use similarity-error hybrid retrieval, but augment the case profile with UK exogenous state and include exogenous summaries on examples.",
            aim="Keep the hard-case bias that has been working, while making similarity matching aware of auctions and ICAP event structure.",
            expected_low=22.02,
            expected_high=22.18,
            cot_rf={
                "example_selection": "similarity_error_hybrid",
                "lookback_days": 180,
                "k_examples": 4,
                "n_similarity_examples": 2,
                "retrieval_feature_columns": UK_RETRIEVAL_FEATURES,
                "retrieval_max_features": len(UK_RETRIEVAL_FEATURES),
                "augment_case_profiles_with_retrieval": True,
                "include_example_exogenous_summary": True,
                "example_exogenous_max_features": 4,
            },
            llm={},
            hdelta={},
        ),
        Candidate(
            name="utility_mmr_augmented_uk_context_k4",
            change_summary="Use utility-MMR retrieval with UK-augmented case profiles and example exogenous summaries.",
            aim="Test whether a more diverse but still utility-ranked UK retrieval set beats plain recent-high-error on the smaller 4B model.",
            expected_low=22.06,
            expected_high=22.22,
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
