#!/usr/bin/env python3
"""Sweep UK TSM+LLM case-conditioned HDELTA candidates and summarize outcomes."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

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


def _deep_update(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    result = dict(base)
    for key, value in updates.items():
        if isinstance(result.get(key), dict) and isinstance(value, dict):
            result[key] = _deep_update(result[key], value)
        else:
            result[key] = value
    return result


def _load_yaml(path: Path) -> dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}


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
    frozen_counter: dict[str, int] = {}
    matched_counter: dict[int, int] = {}
    log_path = run_dir / "llm" / "logs" / "llm_calls.jsonl"
    with open(log_path, "r") as f:
        for line in f:
            obj = json.loads(line)
            if not str(obj.get("method", "")).endswith(":apply"):
                continue
            content = ((obj.get("response") or {}).get("content") or "").strip()
            apply_counter[content] = apply_counter.get(content, 0) + 1
            metadata = obj.get("metadata") or {}
            frozen_key = json.dumps(metadata.get("dynamic_frozen_horizons") or [])
            frozen_counter[frozen_key] = frozen_counter.get(frozen_key, 0) + 1
            matched_n = len(metadata.get("matched_teaching_dates") or [])
            matched_counter[matched_n] = matched_counter.get(matched_n, 0) + 1

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
        "top_apply_response": max(apply_counter.items(), key=lambda kv: kv[1])[0] if apply_counter else "",
        "top_apply_count": max(apply_counter.values()) if apply_counter else 0,
        "top_frozen_pattern": max(frozen_counter.items(), key=lambda kv: kv[1])[0] if frozen_counter else "[]",
        "top_frozen_count": max(frozen_counter.values()) if frozen_counter else 0,
        "top_matched_n": max(matched_counter.items(), key=lambda kv: kv[1])[0] if matched_counter else 0,
        "top_matched_count": max(matched_counter.values()) if matched_counter else 0,
    }


def _write_report(out_dir: Path, rows: list[dict[str, Any]], best_full: dict[str, Any] | None) -> None:
    rows_df = pd.DataFrame(rows)
    rows_df.to_csv(out_dir / "candidate_results.csv", index=False)

    lines = [
        "# UK ETS TSM+LLM Case-Conditioned Sweep",
        "",
        f"Generated: {datetime.now().isoformat()}",
        "",
        "## Candidate Results",
        "",
        "| Candidate | Stage | Hypothesis | Expected | LLM Path MSE | Delta vs TSM | h5 Delta | h20 Delta | h30 Delta | Unique Apply | Unique Rules |",
        "|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {name} | {stage} | {hypothesis} | {expected_outcome} | {llm_mse_path:.6f} | {delta_mse_path:+.6f} | {h5_delta:+.6f} | {h20_delta:+.6f} | {h30_delta:+.6f} | {unique_apply_responses} | {unique_rules} |".format(
                name=row["name"],
                stage=row["stage"],
                hypothesis=row["hypothesis"],
                expected_outcome=row["expected_outcome"],
                llm_mse_path=row["llm_mse_path"],
                delta_mse_path=row["delta_mse_path"],
                h5_delta=row["llm_h5_mse"] - row["base_h5_mse"],
                h20_delta=row["llm_h20_mse"] - row["base_h20_mse"],
                h30_delta=row["llm_h30_mse"] - row["base_h30_mse"],
                unique_apply_responses=row["unique_apply_responses"],
                unique_rules=row["unique_rules"],
            )
        )

    lines.extend(
        [
            "",
            "## Best Raw Candidate",
            "",
            f"- Candidate: `{rows[0]['name']}`",
            f"- Stage: `{rows[0]['stage']}`",
            f"- Run dir: `{rows[0]['run_dir']}`",
            f"- Raw TSM+LLM path MSE: `{rows[0]['llm_mse_path']:.6f}`",
            f"- Delta vs base TSM: `{rows[0]['delta_mse_path']:+.6f}`",
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
    parser = argparse.ArgumentParser(description="Sweep UK TSM+LLM case-conditioned candidates.")
    parser.add_argument("--config", default="uk_ets/config/uk_ets_llm_case_conditioned_hdelta.yaml")
    parser.add_argument("--data-dir", default="uk_ets/Data_auto_uk")
    parser.add_argument("--output-root", default="reports/uk_ets_llm_case_conditioned_sweep")
    parser.add_argument("--run-full-best", action="store_true")
    args = parser.parse_args()

    config_path = (PROJECT_ROOT / args.config).resolve()
    data_dir = (PROJECT_ROOT / args.data_dir).resolve()
    base_cfg = _load_yaml(config_path)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = (PROJECT_ROOT / args.output_root / timestamp).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    candidates = [
        Candidate(
            name="baseline_365_k6_tk3",
            stage="lookback",
            hypothesis="Control: current case-conditioned settings.",
            expected_outcome="Should reproduce the small positive TSM gain already observed.",
            llm_cot_rf={"lookback_days": 365, "k_examples": 6, "n_similarity_examples": 3},
            hdelta={"case_match_top_k": 3},
        ),
        Candidate(
            name="recent_270_k6_tk3",
            stage="lookback",
            hypothesis="Shorter teaching horizon should reduce stale pre-regime examples without losing too much evidence.",
            expected_outcome="Slightly better h30 than baseline, similar h5/h20 gains.",
            llm_cot_rf={"lookback_days": 270, "k_examples": 6, "n_similarity_examples": 3},
            hdelta={"case_match_top_k": 3},
        ),
        Candidate(
            name="recent_180_k6_tk3",
            stage="lookback",
            hypothesis="A 180-day horizon should better match the late-2025 UK regime for TSM correction.",
            expected_outcome="Best chance to cut h30 drift while keeping h5/h20 improvement.",
            llm_cot_rf={"lookback_days": 180, "k_examples": 6, "n_similarity_examples": 3},
            hdelta={"case_match_top_k": 3},
        ),
        Candidate(
            name="recent_120_k6_tk3",
            stage="lookback",
            hypothesis="A 120-day teaching horizon may isolate the most recent UK regime without starving the matcher.",
            expected_outcome="Could outperform 180 days if the market regime really shifted late in the sample.",
            llm_cot_rf={"lookback_days": 120, "k_examples": 6, "n_similarity_examples": 3},
            hdelta={"case_match_top_k": 3},
        ),
        Candidate(
            name="recent_90_k6_tk3",
            stage="lookback",
            hypothesis="Very short history may track the local regime best but risks sparse or noisy teaching evidence.",
            expected_outcome="Could improve h30, but may lose stability and diversity.",
            llm_cot_rf={"lookback_days": 90, "k_examples": 6, "n_similarity_examples": 3},
            hdelta={"case_match_top_k": 3},
        ),
        Candidate(
            name="recent_180_k4_tk2",
            stage="evidence",
            hypothesis="Smaller evidence set should force tighter local matching and reduce conflicting horizon guidance.",
            expected_outcome="Lower h30 error and fewer contradictory long-horizon adjustments.",
            llm_cot_rf={"lookback_days": 180, "k_examples": 4, "n_similarity_examples": 2},
            hdelta={"case_match_top_k": 2},
        ),
        Candidate(
            name="recent_180_k4_tk2_tightbound",
            stage="bounds",
            hypothesis="Tighter per-horizon bounds should preserve h5/h20 gains while reducing overcorrection at h30.",
            expected_outcome="Path MSE improves if h30 drag is mostly caused by excessive adjustment magnitude.",
            llm_cot_rf={"lookback_days": 180, "k_examples": 4, "n_similarity_examples": 2},
            hdelta={"case_match_top_k": 2, "case_bound_scale": 0.65, "max_adjustment_pct": 0.8},
        ),
        Candidate(
            name="recent_180_k4_tk2_strictsign",
            stage="consensus",
            hypothesis="Stricter sign agreement and more required evidence should freeze noisy long-horizon corrections.",
            expected_outcome="Smaller active coverage, but cleaner raw test path if contradictory examples are the main problem.",
            llm_cot_rf={"lookback_days": 180, "k_examples": 4, "n_similarity_examples": 2},
            hdelta={
                "case_match_top_k": 2,
                "case_min_examples": 3,
                "case_min_sign_agreement": 0.75,
                "case_min_mean_abs_error_pct": 0.12,
            },
        ),
        Candidate(
            name="recent_180_k4_tk2_localmatch",
            stage="matching",
            hypothesis="Shorter regime feature windows should sharpen case matching around the latest UK pattern.",
            expected_outcome="More diverse but more locally relevant rules, with improved h20/h30 sign stability.",
            llm_cot_rf={
                "lookback_days": 180,
                "k_examples": 4,
                "n_similarity_examples": 2,
                "feature_window": 12,
            },
            hdelta={"case_match_top_k": 2},
        ),
        Candidate(
            name="recent_120_k4_tk2_tightbound",
            stage="combined",
            hypothesis="Best recent-regime horizon plus smaller evidence set and tighter bounds may be the strongest anti-h30 configuration.",
            expected_outcome="Best chance to beat the current raw TSM+LLM result if recency and magnitude control both matter.",
            llm_cot_rf={"lookback_days": 120, "k_examples": 4, "n_similarity_examples": 2},
            hdelta={"case_match_top_k": 2, "case_bound_scale": 0.65, "max_adjustment_pct": 0.8},
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
            "output": {"write_project_paper": False},
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
