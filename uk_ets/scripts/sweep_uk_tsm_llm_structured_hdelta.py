#!/usr/bin/env python3
"""Sweep UK TSM+LLM structured HDELTA candidates and summarize outcomes."""

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


@dataclass(frozen=True)
class Candidate:
    name: str
    stage: str
    hypothesis: str
    expected_outcome: str
    llm_cot_rf: dict[str, Any]
    hdelta: dict[str, Any]


def _extract_candidate_metrics(run_dir: Path) -> dict[str, Any]:
    path_metrics = pd.read_csv(run_dir / "results" / "path_metrics.csv")
    horizon_metrics = pd.read_csv(run_dir / "results" / "metrics_by_horizon.csv")

    tsm_base = path_metrics[path_metrics["model"] == "tsm"].iloc[0]
    llm_raw = path_metrics[path_metrics["model"] == "TSM+LLM-COT-RF-HDELTA"].iloc[0]

    tsm_h = horizon_metrics[horizon_metrics["model"] == "tsm"].set_index("horizon")
    llm_h = horizon_metrics[horizon_metrics["model"] == "TSM+LLM-COT-RF-HDELTA"].set_index("horizon")

    pred_npz = np.load(run_dir / "predictions" / "TSM+LLM-COT-RF-HDELTA_pred_test_subset.npz")
    deltas = pred_npz["yhat"] - pred_npz["base_pred"]

    apply_counter: dict[str, int] = {}
    structured_calls = 0
    log_path = run_dir / "llm" / "logs" / "llm_calls.jsonl"
    with open(log_path, "r") as f:
        for line in f:
            obj = json.loads(line)
            if not str(obj.get("method", "")).endswith(":apply"):
                continue
            content = ((obj.get("response") or {}).get("content") or "").strip()
            apply_counter[content] = apply_counter.get(content, 0) + 1
            metadata = obj.get("metadata") or {}
            if metadata.get("structured_horizon_reflection"):
                structured_calls += 1

    rules_path = run_dir / "llm" / "TSM_LLM-COT-RF-HDELTA_rules.jsonl"
    unique_rules = 0
    if rules_path.exists():
        seen_rules = set()
        with open(rules_path, "r") as f:
            for line in f:
                seen_rules.add(json.loads(line)["rules_text"])
        unique_rules = len(seen_rules)

    return {
        "run_dir": str(run_dir),
        "base_mse_path": float(tsm_base["mse_path"]),
        "llm_mse_path": float(llm_raw["mse_path"]),
        "delta_mse_path": float(llm_raw["mse_path"] - tsm_base["mse_path"]),
        "base_h5_mse": float(tsm_h.loc[5, "mse"]),
        "llm_h5_mse": float(llm_h.loc[5, "mse"]),
        "base_h20_mse": float(tsm_h.loc[20, "mse"]),
        "llm_h20_mse": float(llm_h.loc[20, "mse"]),
        "base_h30_mse": float(tsm_h.loc[30, "mse"]),
        "llm_h30_mse": float(llm_h.loc[30, "mse"]),
        "nonzero_count": int(np.count_nonzero(np.abs(deltas) > 1e-12)),
        "abs_mean_delta": float(np.abs(deltas).mean()),
        "mean_delta": float(deltas.mean()),
        "unique_apply_responses": len(apply_counter),
        "unique_rules": int(unique_rules),
        "structured_apply_calls": int(structured_calls),
        "top_apply_count": max(apply_counter.values()) if apply_counter else 0,
    }


def _write_report(out_dir: Path, rows: list[dict[str, Any]], best_full: dict[str, Any] | None) -> None:
    rows_df = pd.DataFrame(rows)
    rows_df.to_csv(out_dir / "candidate_results.csv", index=False)

    lines = [
        "# UK ETS Structured HDELTA Sweep",
        "",
        f"Generated: {datetime.now().isoformat()}",
        "",
        "## Candidate Results",
        "",
        "| Candidate | Stage | Expected | LLM Path MSE | Delta vs TSM | h5 Delta | h20 Delta | h30 Delta | Unique Apply | Structured Apply Calls |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {name} | {stage} | {expected_outcome} | {llm_mse_path:.6f} | {delta_mse_path:+.6f} | {h5_delta:+.6f} | {h20_delta:+.6f} | {h30_delta:+.6f} | {unique_apply_responses} | {structured_apply_calls} |".format(
                name=row["name"],
                stage=row["stage"],
                expected_outcome=row["expected_outcome"],
                llm_mse_path=row["llm_mse_path"],
                delta_mse_path=row["delta_mse_path"],
                h5_delta=row["llm_h5_mse"] - row["base_h5_mse"],
                h20_delta=row["llm_h20_mse"] - row["base_h20_mse"],
                h30_delta=row["llm_h30_mse"] - row["base_h30_mse"],
                unique_apply_responses=row["unique_apply_responses"],
                structured_apply_calls=row["structured_apply_calls"],
            )
        )

    if rows:
        best = rows[0]
        lines.extend(
            [
                "",
                "## Best Raw Candidate",
                "",
                f"- Candidate: `{best['name']}`",
                f"- Stage: `{best['stage']}`",
                f"- Run dir: `{best['run_dir']}`",
                f"- Raw TSM+LLM path MSE: `{best['llm_mse_path']:.6f}`",
                f"- Delta vs base TSM: `{best['delta_mse_path']:+.6f}`",
            ]
        )

    if best_full is not None:
        lines.extend(
            [
                "",
                "## Best Full Run",
                "",
                f"- Candidate: `{best_full['name']}`",
                f"- Run dir: `{best_full['run_dir']}`",
                f"- Raw TSM+LLM path MSE: `{best_full['llm_mse_path']:.6f}`",
                f"- Delta vs base TSM: `{best_full['delta_mse_path']:+.6f}`",
            ]
        )

    (out_dir / "summary.md").write_text("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Sweep UK TSM+LLM structured HDELTA candidates.")
    parser.add_argument("--config", default="uk_ets/config/uk_ets_llm_case_conditioned_hdelta.yaml")
    parser.add_argument("--data-dir", default="uk_ets/Data_auto_uk")
    parser.add_argument("--output-root", default="reports/uk_ets_structured_hdelta_sweep")
    parser.add_argument("--run-full-best", action="store_true")
    args = parser.parse_args()

    config_path = (PROJECT_ROOT / args.config).resolve()
    data_dir = (PROJECT_ROOT / args.data_dir).resolve()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = (PROJECT_ROOT / args.output_root / timestamp).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    candidates = [
        Candidate(
            name="structured_baseline",
            stage="baseline",
            hypothesis="Structured per-horizon reflection should reduce h30 drag versus free-text reflection.",
            expected_outcome="Should reproduce about 22.59 path MSE and remain the best tested baseline for this sweep.",
            llm_cot_rf={"structured_horizon_reflection": True},
            hdelta={},
        ),
        Candidate(
            name="structured_h30_gate",
            stage="h30_gate",
            hypothesis="Require stronger evidence before letting h30 move.",
            expected_outcome="Likely improve path MSE to roughly 22.55-22.59 if residual long-horizon noise is still the main problem.",
            llm_cot_rf={"structured_horizon_reflection": True},
            hdelta={
                "structured_enforce_sign": True,
                "structured_min_confidence_by_horizon": {"h30": "high"},
                "horizon_overrides": {
                    "h30": {
                        "case_min_examples": 3,
                        "case_min_sign_agreement": 0.80,
                        "case_min_mean_abs_error_pct": 0.10,
                        "case_bound_scale": 0.60,
                    }
                },
            },
        ),
        Candidate(
            name="structured_asymmetric",
            stage="asymmetric",
            hypothesis="Loosen h5 while tightening h30 to recover short-horizon gains without bringing back long-tail damage.",
            expected_outcome="Best chance at roughly 22.52-22.57 if the current structured run is under-adjusting h5 and over-policing h30.",
            llm_cot_rf={"structured_horizon_reflection": True},
            hdelta={
                "structured_enforce_sign": True,
                "structured_min_confidence_by_horizon": {"h30": "high"},
                "horizon_overrides": {
                    "h5": {
                        "case_min_examples": 1,
                        "case_min_sign_agreement": 0.50,
                        "case_min_mean_abs_error_pct": 0.05,
                        "case_bound_scale": 1.00,
                        "case_min_bound_pct": 0.10,
                    },
                    "h30": {
                        "case_min_examples": 3,
                        "case_min_sign_agreement": 0.80,
                        "case_min_mean_abs_error_pct": 0.12,
                        "case_bound_scale": 0.60,
                    },
                },
            },
        ),
        Candidate(
            name="structured_guided_caps",
            stage="guided_caps",
            hypothesis="Deterministic enforcement of structured sign/confidence/magnitude should turn good reflection guidance into cleaner numeric adjustments.",
            expected_outcome="Should land around 22.54-22.60 if apply-stage noise is still limiting the LLM improvement.",
            llm_cot_rf={"structured_horizon_reflection": True},
            hdelta={
                "structured_enforce_sign": True,
                "structured_cap_by_guidance": True,
                "structured_min_confidence_by_horizon": {"h30": "medium"},
                "structured_magnitude_scale": {"tiny": 0.20, "small": 0.45, "medium": 0.75},
                "structured_confidence_scale": {"low": 0.50, "medium": 0.80, "high": 1.00},
            },
        ),
        Candidate(
            name="structured_combo",
            stage="combo",
            hypothesis="Combine asymmetric horizon control with deterministic guided caps.",
            expected_outcome="Largest upside, roughly 22.48-22.55, but also the highest risk of over-freezing useful corrections.",
            llm_cot_rf={"structured_horizon_reflection": True},
            hdelta={
                "structured_enforce_sign": True,
                "structured_cap_by_guidance": True,
                "structured_min_confidence_by_horizon": {"h30": "high"},
                "structured_magnitude_scale": {"tiny": 0.20, "small": 0.45, "medium": 0.75},
                "structured_confidence_scale": {"low": 0.50, "medium": 0.80, "high": 1.00},
                "horizon_overrides": {
                    "h5": {
                        "case_min_examples": 1,
                        "case_min_sign_agreement": 0.50,
                        "case_min_mean_abs_error_pct": 0.05,
                        "case_bound_scale": 1.00,
                        "case_min_bound_pct": 0.10,
                    },
                    "h30": {
                        "case_min_examples": 3,
                        "case_min_sign_agreement": 0.80,
                        "case_min_mean_abs_error_pct": 0.12,
                        "case_bound_scale": 0.60,
                    },
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
            "compute": {"num_workers": 0},
            "llm": {
                "base_model": "tsm",
                "cot_rf": candidate.llm_cot_rf,
                "blend_grid": {"enabled": False},
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
        metrics = _extract_candidate_metrics(run_dir)
        row = {
            "name": candidate.name,
            "stage": candidate.stage,
            "hypothesis": candidate.hypothesis,
            "expected_outcome": candidate.expected_outcome,
            **metrics,
        }
        rows.append(row)
        print(
            f"  -> {candidate.name}: llm_mse_path={row['llm_mse_path']:.6f} "
            f"(delta_vs_tsm={row['delta_mse_path']:+.6f})",
            flush=True,
        )

    rows.sort(key=lambda row: row["llm_mse_path"])
    best = rows[0]
    best_full = None

    if args.run_full_best:
        print(f"Running full best candidate: {best['name']} ...", flush=True)
        candidate = next(c for c in candidates if c.name == best["name"])
        overrides = {
            "target": {"mode": "returns"},
            "output": {"write_project_paper": False, "generate_paper": False},
            "compute": {"num_workers": 0},
            "llm": {
                "base_model": "tsm",
                "cot_rf": candidate.llm_cot_rf,
                "blend_grid": {"enabled": True},
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
        best_full = {
            "name": best["name"],
            **_extract_candidate_metrics(run_dir),
        }

    _write_report(out_dir, rows, best_full)
    print(f"Sweep results written to {out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
