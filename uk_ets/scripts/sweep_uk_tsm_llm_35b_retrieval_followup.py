#!/usr/bin/env python3
"""Five-way retrieval and routing sweep on top of the 35B step-back UK winner."""

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
        "# UK ETS 35B Retrieval Follow-Up Sweep",
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
    parser = argparse.ArgumentParser(description="Run a five-way retrieval follow-up sweep on the 35B UK winner.")
    parser.add_argument("--config", default="uk_ets/config/uk_ets_llm_35b_step_back_regime.yaml")
    parser.add_argument("--data-dir", default="uk_ets/Data_auto_uk")
    parser.add_argument("--output-root", default="reports/uk_ets_llm_35b_retrieval_followup")
    parser.add_argument(
        "--baseline-run",
        default="runs/20260309_234219_249d62",
        help="Existing 35B step-back winner used as the raw LLM reference.",
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
            name="utility_score_retrieval",
            literature_label="Learning To Retrieve Prompts for In-Context Learning",
            literature_url="https://arxiv.org/abs/2112.08633",
            change_summary="Replace recent-high-error retrieval with a utility score combining similarity, correction value, and recency.",
            aim="Choose examples that are not just hard, but also close to the current regime and recent enough to transfer cleanly.",
            expected_low=21.38,
            expected_high=21.50,
            cot_rf={
                "reasoning_style": "step_back",
                "apply_style": "program_of_thought",
                "example_selection": "utility_score",
                "utility_similarity_weight": 0.45,
                "utility_error_weight": 0.35,
                "utility_recency_weight": 0.20,
                "lookback_days": 180,
                "k_examples": 6,
            },
            llm={},
            hdelta={},
        ),
        Candidate(
            name="utility_mmr_retrieval",
            literature_label="Active Example Selection for In-Context Learning",
            literature_url="https://arxiv.org/abs/2211.04486",
            change_summary="Keep the utility score, but apply MMR to avoid feeding the model six near-duplicate examples.",
            aim="Increase evidence coverage without sacrificing regime fit, especially at h20 and h30.",
            expected_low=21.34,
            expected_high=21.48,
            cot_rf={
                "reasoning_style": "step_back",
                "apply_style": "program_of_thought",
                "example_selection": "utility_mmr",
                "utility_similarity_weight": 0.45,
                "utility_error_weight": 0.35,
                "utility_recency_weight": 0.20,
                "mmr_lambda": 0.75,
                "lookback_days": 180,
                "k_examples": 6,
            },
            llm={},
            hdelta={},
        ),
        Candidate(
            name="regime_filtered_mmr",
            literature_label="Large Language Models as Analogical Reasoners",
            literature_url="https://arxiv.org/abs/2310.01714",
            change_summary="Filter the candidate bank to regime-compatible cases, then use utility-weighted MMR and explicit retrieval tags.",
            aim="Make the model reason over truly analogous UK cases instead of mixing incompatible regimes inside the prompt.",
            expected_low=21.32,
            expected_high=21.47,
            cot_rf={
                "reasoning_style": "step_back",
                "apply_style": "program_of_thought",
                "example_selection": "regime_mmr",
                "utility_similarity_weight": 0.40,
                "utility_error_weight": 0.30,
                "utility_recency_weight": 0.15,
                "utility_regime_weight": 0.25,
                "mmr_lambda": 0.72,
                "regime_filter_min_candidates": 12,
                "lookback_days": 180,
                "k_examples": 6,
            },
            llm={},
            hdelta={"include_retrieval_tag": True},
        ),
        Candidate(
            name="long_context_regime_mmr",
            literature_label="In-Context Learning with Long-Context Models",
            literature_url="https://arxiv.org/abs/2405.00200",
            change_summary="Use the regime-filtered MMR retriever, but expand the prompt to a larger diverse evidence set.",
            aim="Test whether the 35B model can exploit a wider UK example bank without collapsing into repetition.",
            expected_low=21.30,
            expected_high=21.48,
            cot_rf={
                "reasoning_style": "step_back",
                "apply_style": "program_of_thought",
                "example_selection": "regime_mmr",
                "utility_similarity_weight": 0.40,
                "utility_error_weight": 0.30,
                "utility_recency_weight": 0.10,
                "utility_regime_weight": 0.30,
                "mmr_lambda": 0.70,
                "regime_filter_min_candidates": 16,
                "lookback_days": 180,
                "k_examples": 10,
            },
            llm={},
            hdelta={"include_retrieval_tag": True},
        ),
        Candidate(
            name="agreement_routed_regime_mmr",
            literature_label="UnCert-CoT / Uncertainty of Thoughts",
            literature_url="https://arxiv.org/abs/2503.15341",
            change_summary="Use the regime-filtered MMR retriever, then spend extra reflection samples only when the matched examples disagree materially on long-horizon sign.",
            aim="Keep the current winner on easy cases while giving ambiguous h20/h30 cases more deliberate consensus search.",
            expected_low=21.28,
            expected_high=21.45,
            cot_rf={
                "reasoning_style": "step_back",
                "apply_style": "program_of_thought",
                "example_selection": "regime_mmr",
                "utility_similarity_weight": 0.40,
                "utility_error_weight": 0.30,
                "utility_recency_weight": 0.15,
                "utility_regime_weight": 0.25,
                "mmr_lambda": 0.72,
                "regime_filter_min_candidates": 12,
                "lookback_days": 180,
                "k_examples": 6,
                "reflect_uncertainty_routing": True,
                "reflect_samples": 1,
                "reflect_samples_low_uncertainty": 1,
                "reflect_samples_high_uncertainty": 3,
                "reflect_aggregation": "conservative_majority",
                "reflect_temperature": 0.15,
                "reflect_uncertainty_distance_threshold": 1.45,
                "reflect_uncertainty_distance_std_threshold": 0.25,
                "reflect_uncertainty_min_dynamic_freezes": 2,
                "reflect_uncertainty_min_low_bound_horizons": 2,
                "reflect_uncertainty_low_bound_threshold": 0.20,
                "reflect_uncertainty_sign_horizons": [20, 30],
                "reflect_uncertainty_min_sign_agreement": 0.70,
            },
            llm={"cache_enabled": False},
            hdelta={"include_retrieval_tag": True},
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
