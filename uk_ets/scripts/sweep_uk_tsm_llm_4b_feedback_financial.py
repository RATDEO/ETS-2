#!/usr/bin/env python3
"""Feedback-focused structural sweep for the UK 4B TSM+LLM pipeline."""

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


def _load_portfolio_metrics(run_dir: Path, model_name: str) -> dict[int, dict[str, float]]:
    portfolio_metrics = pd.read_csv(run_dir / "results" / "long_short_metrics.csv")
    subset = portfolio_metrics[portfolio_metrics["model"] == model_name]
    results: dict[int, dict[str, float]] = {}
    for _, row in subset.iterrows():
        horizon = int(row["horizon"])
        results[horizon] = {
            "sharpe": float(row["sharpe"]) if pd.notna(row["sharpe"]) else np.nan,
            "mean_return": float(row["mean_return"]),
            "total_return_non_overlap": float(row.get("total_return_non_overlap", np.nan)),
            "annualized_return_non_overlap": float(row.get("annualized_return_non_overlap", np.nan)),
            "sharpe_non_overlap": float(row.get("sharpe_non_overlap", np.nan)),
            "max_drawdown_non_overlap": float(row.get("max_drawdown_non_overlap", np.nan)),
        }
    return results


def _extract_candidate_metrics(
    run_dir: Path,
    checkpoint_llm_path: float,
    checkpoint_horizons: dict[int, float],
    checkpoint_portfolio: dict[int, dict[str, float]],
) -> dict[str, Any]:
    path_metrics = pd.read_csv(run_dir / "results" / "path_metrics.csv")
    horizon_metrics = pd.read_csv(run_dir / "results" / "metrics_by_horizon.csv")
    portfolio_metrics = _load_portfolio_metrics(run_dir, METHOD_NAME)

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
        "h20_sharpe_non_overlap": float(portfolio_metrics[20]["sharpe_non_overlap"]),
        "h30_sharpe_non_overlap": float(portfolio_metrics[30]["sharpe_non_overlap"]),
        "h20_total_return_non_overlap": float(portfolio_metrics[20]["total_return_non_overlap"]),
        "h30_total_return_non_overlap": float(portfolio_metrics[30]["total_return_non_overlap"]),
        "h20_annualized_return_non_overlap": float(portfolio_metrics[20]["annualized_return_non_overlap"]),
        "h30_annualized_return_non_overlap": float(portfolio_metrics[30]["annualized_return_non_overlap"]),
        "h20_max_drawdown_non_overlap": float(portfolio_metrics[20]["max_drawdown_non_overlap"]),
        "h30_max_drawdown_non_overlap": float(portfolio_metrics[30]["max_drawdown_non_overlap"]),
        "h20_total_return_vs_checkpoint": float(
            portfolio_metrics[20]["total_return_non_overlap"]
            - checkpoint_portfolio[20]["total_return_non_overlap"]
        ),
        "h30_total_return_vs_checkpoint": float(
            portfolio_metrics[30]["total_return_non_overlap"]
            - checkpoint_portfolio[30]["total_return_non_overlap"]
        ),
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
    checkpoint_portfolio: dict[int, dict[str, float]],
) -> None:
    rows_df = pd.DataFrame(rows)
    rows_df.to_csv(out_dir / "candidate_results.csv", index=False)

    lines = [
        "# UK ETS 4B Feedback Financial Sweep",
        "",
        f"Generated: {datetime.now().isoformat()}",
        "",
        "## Checkpoint",
        "",
        f"- Base checkpoint run: `{checkpoint_run_dir}`",
        f"- Checkpoint raw `{METHOD_NAME}` path MSE: `{checkpoint_path_mse:.6f}`",
        f"- Reference `linear_ridge` path MSE: `{ridge_path_mse:.6f}`",
        f"- Checkpoint h20 non-overlap total return: `{checkpoint_portfolio[20]['total_return_non_overlap'] * 100:.2f}%`",
        f"- Checkpoint h30 non-overlap total return: `{checkpoint_portfolio[30]['total_return_non_overlap'] * 100:.2f}%`",
        "",
        "## Candidate Results",
        "",
        "| Candidate | Expected | Actual MSE | Outcome | h20 vs chkpt | h30 vs chkpt | h20 total | h30 total | h20 Sharpe | h30 Sharpe |",
        "|---|---|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {name} | {expected_low:.2f}-{expected_high:.2f} | {llm_mse_path:.6f} | {expectation_result} | {h20_delta_vs_checkpoint:+.6f} | {h30_delta_vs_checkpoint:+.6f} | {h20_total_return_non_overlap:.2%} | {h30_total_return_non_overlap:.2%} | {h20_sharpe_non_overlap:.3f} | {h30_sharpe_non_overlap:.3f} |".format(
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
                f"- h20 non-overlap total return: `{best['h20_total_return_non_overlap'] * 100:.2f}%`",
                f"- h20 non-overlap annualized return: `{best['h20_annualized_return_non_overlap'] * 100:.2f}%`",
                f"- h20 non-overlap Sharpe: `{best['h20_sharpe_non_overlap']:.3f}`",
                f"- h20 non-overlap max drawdown: `{best['h20_max_drawdown_non_overlap'] * 100:.2f}%`",
                f"- h30 non-overlap total return: `{best['h30_total_return_non_overlap'] * 100:.2f}%`",
                f"- h30 non-overlap annualized return: `{best['h30_annualized_return_non_overlap'] * 100:.2f}%`",
                f"- h30 non-overlap Sharpe: `{best['h30_sharpe_non_overlap']:.3f}`",
                f"- h30 non-overlap max drawdown: `{best['h30_max_drawdown_non_overlap'] * 100:.2f}%`",
                f"- Outcome vs expectation: `{best['expectation_result']}`",
                f"- Run dir: `{best['run_dir']}`",
            ]
        )

    (out_dir / "summary.md").write_text("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a feedback-focused UK 4B sweep with financial reporting.")
    parser.add_argument("--config", default="uk_ets/config/uk_ets_llm_4b_feedback_recent_high_error_k4.yaml")
    parser.add_argument("--data-dir", default="uk_ets/Data_auto_uk")
    parser.add_argument("--output-root", default="reports/uk_ets_llm_4b_feedback_financial")
    parser.add_argument(
        "--checkpoint-run",
        default="runs/20260311_045606_c3f654",
        help="Existing 4B feedback run used as the checkpoint reference.",
    )
    args = parser.parse_args()

    config_path = (PROJECT_ROOT / args.config).resolve()
    data_dir = (PROJECT_ROOT / args.data_dir).resolve()
    checkpoint_run_dir = (PROJECT_ROOT / args.checkpoint_run).resolve()

    checkpoint_path_mse = _load_path_metric(checkpoint_run_dir, METHOD_NAME)
    checkpoint_horizons = _load_horizon_mse(checkpoint_run_dir, METHOD_NAME)
    ridge_path_mse = _load_path_metric(checkpoint_run_dir, "linear_ridge")
    checkpoint_portfolio = _load_portfolio_metrics(checkpoint_run_dir, METHOD_NAME)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = (PROJECT_ROOT / args.output_root / timestamp).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    candidates = [
        Candidate(
            name="control_feedback_recent_high_error_k4",
            change_summary="Re-run the current 4B feedback winner unchanged as a control.",
            aim="Verify the feedback branch and new financial metrics reproduce the current checkpoint.",
            literature_basis="Control",
            expected_low=21.96,
            expected_high=22.04,
            cot_rf={},
            llm={},
            hdelta={},
        ),
        Candidate(
            name="feedback_skill_tag_recent_high_error_k4",
            change_summary="Combine hindsight feedback with regime-skill-tag filtering.",
            aim="Test whether feedback becomes more reusable when the examples also match the same coarse market regime.",
            literature_basis="Chain of Hindsight plus Skill-KNN style skill matching.",
            expected_low=21.90,
            expected_high=22.00,
            cot_rf={
                "include_hindsight_feedback": True,
                "example_selection": "skill_tag_recent_high_error",
                "lookback_days": 180,
                "k_examples": 4,
                "skill_tag_min_overlap": 4,
            },
            llm={},
            hdelta={},
        ),
        Candidate(
            name="feedback_plus_freeze_counterexample_k4",
            change_summary="Combine hindsight feedback with one similar low-error freeze counterexample.",
            aim="Keep the feedback gain while making the model more explicit about when not to force a correction.",
            literature_basis="Chain of Hindsight plus supportive/comparable demonstrations.",
            expected_low=21.92,
            expected_high=22.02,
            cot_rf={
                "include_hindsight_feedback": True,
                "include_counterexample_freeze": True,
                "counterexample_mode": "low_error",
                "counterexample_quantile": 0.35,
                "example_selection": "recent_high_error",
                "lookback_days": 180,
                "k_examples": 4,
            },
            llm={},
            hdelta={},
        ),
        Candidate(
            name="feedback_long_horizon_only_k4",
            change_summary="Keep hindsight feedback but restrict it to the long-horizon mistake description only.",
            aim="Reduce noisy short-horizon commentary and focus the lesson on h20/h30, where the 4B gains live.",
            literature_basis="Time-series LLM evidence says the useful correction signal is concentrated in the long tail.",
            expected_low=21.92,
            expected_high=22.04,
            cot_rf={
                "include_hindsight_feedback": True,
                "feedback_mode": "long_horizon_only",
                "example_selection": "recent_high_error",
                "lookback_days": 180,
                "k_examples": 4,
            },
            llm={},
            hdelta={},
        ),
        Candidate(
            name="feedback_compact_tags_k4",
            change_summary="Convert hindsight feedback into compact error tags instead of prose.",
            aim="Test whether the 4B model follows short structured lessons better than full sentences.",
            literature_basis="Feedback compression and rubric-style supervision for small models.",
            expected_low=21.90,
            expected_high=22.02,
            cot_rf={
                "include_hindsight_feedback": True,
                "feedback_mode": "compact_tags",
                "example_selection": "recent_high_error",
                "lookback_days": 180,
                "k_examples": 4,
            },
            llm={},
            hdelta={},
        ),
        Candidate(
            name="feedback_skill_tag_plus_counterexample_k4",
            change_summary="Combine hindsight feedback, skill-tag filtering, and one freeze counterexample.",
            aim="Test the strongest combined feedback branch: regime-matched examples, explicit lessons, and one do-not-adjust example.",
            literature_basis="Chain of Hindsight plus Skill-KNN plus comparable demonstrations.",
            expected_low=21.88,
            expected_high=22.00,
            cot_rf={
                "include_hindsight_feedback": True,
                "include_counterexample_freeze": True,
                "counterexample_mode": "low_error",
                "counterexample_quantile": 0.35,
                "example_selection": "skill_tag_recent_high_error",
                "lookback_days": 180,
                "k_examples": 4,
                "skill_tag_min_overlap": 4,
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
            checkpoint_portfolio=checkpoint_portfolio,
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
        checkpoint_portfolio=checkpoint_portfolio,
    )
    print(f"Sweep results written to {out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
