#!/usr/bin/env python3
"""Fast literature-backed sweep for UK TSM+LLM-CoT HDELTA variants."""

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
    literature: str
    aim: str
    expected_outcome: str
    cot_rf: dict[str, Any]
    llm: dict[str, Any]
    hdelta: dict[str, Any]


def _extract_candidate_metrics(run_dir: Path) -> dict[str, Any]:
    path_metrics = pd.read_csv(run_dir / "results" / "path_metrics.csv")
    horizon_metrics = pd.read_csv(run_dir / "results" / "metrics_by_horizon.csv")

    tsm_base = path_metrics[path_metrics["model"] == "tsm"].iloc[0]
    llm_raw = path_metrics[path_metrics["model"] == METHOD_NAME].iloc[0]

    tsm_h = horizon_metrics[horizon_metrics["model"] == "tsm"].set_index("horizon")
    llm_h = horizon_metrics[horizon_metrics["model"] == METHOD_NAME].set_index("horizon")

    pred_npz = np.load(run_dir / "predictions" / f"{METHOD_NAME}_pred_test_subset.npz")
    deltas = pred_npz["yhat"] - pred_npz["base_pred"]

    unique_apply = set()
    apply_samples_used = set()
    log_path = run_dir / "llm" / "logs" / "llm_calls.jsonl"
    with open(log_path, "r") as f:
        for line in f:
            obj = json.loads(line)
            if not str(obj.get("method", "")).endswith(":apply"):
                continue
            unique_apply.add(((obj.get("response") or {}).get("content") or "").strip())
            metadata = obj.get("metadata") or {}
            apply_samples_used.add(int(metadata.get("apply_samples", 1)))

    return {
        "run_dir": str(run_dir),
        "base_mse_path": float(tsm_base["mse_path"]),
        "llm_mse_path": float(llm_raw["mse_path"]),
        "delta_vs_tsm": float(llm_raw["mse_path"] - tsm_base["mse_path"]),
        "h5_delta": float(llm_h.loc[5, "mse"] - tsm_h.loc[5, "mse"]),
        "h20_delta": float(llm_h.loc[20, "mse"] - tsm_h.loc[20, "mse"]),
        "h30_delta": float(llm_h.loc[30, "mse"] - tsm_h.loc[30, "mse"]),
        "nonzero_count": int(np.count_nonzero(np.abs(deltas) > 1e-12)),
        "abs_mean_delta": float(np.abs(deltas).mean()),
        "mean_delta": float(deltas.mean()),
        "unique_apply_responses": int(len(unique_apply)),
        "apply_samples_used": ",".join(str(x) for x in sorted(apply_samples_used)),
    }


def _write_report(out_dir: Path, rows: list[dict[str, Any]]) -> None:
    rows_df = pd.DataFrame(rows)
    rows_df.to_csv(out_dir / "candidate_results.csv", index=False)

    lines = [
        "# UK ETS LLM Literature Fast Sweep",
        "",
        f"Generated: {datetime.now().isoformat()}",
        "",
        "This sweep screened literature-backed UK `TSM+LLM-COT-RF-HDELTA` variants on the exploratory UK test subset to iterate quickly.",
        "",
        "## Candidate Results",
        "",
        "| Candidate | Literature | Aim | Expected | LLM Path MSE | Delta vs TSM | h5 Delta | h20 Delta | h30 Delta | Apply Samples | Unique Apply |",
        "|---|---|---|---|---:|---:|---:|---:|---:|---|---:|",
    ]
    for row in rows:
        lines.append(
            "| {name} | {literature} | {aim} | {expected_outcome} | {llm_mse_path:.6f} | {delta_vs_tsm:+.6f} | {h5_delta:+.6f} | {h20_delta:+.6f} | {h30_delta:+.6f} | {apply_samples_used} | {unique_apply_responses} |".format(
                **row
            )
        )

    if rows:
        best = min(rows, key=lambda row: row["llm_mse_path"])
        lines.extend(
            [
                "",
                "## Best Screened Variant",
                "",
                f"- Candidate: `{best['name']}`",
                f"- Literature: `{best['literature']}`",
                f"- Aim: {best['aim']}",
                f"- Expected: {best['expected_outcome']}",
                f"- Run dir: `{best['run_dir']}`",
                f"- Raw path MSE: `{best['llm_mse_path']:.6f}`",
                f"- Delta vs base TSM: `{best['delta_vs_tsm']:+.6f}`",
            ]
        )

    (out_dir / "summary.md").write_text("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Fast UK TSM+LLM literature sweep.")
    parser.add_argument("--config", default="uk_ets/config/uk_ets_llm_coherence_guard.yaml")
    parser.add_argument("--data-dir", default="uk_ets/Data_auto_uk")
    parser.add_argument("--output-root", default="reports/uk_ets_llm_literature_fast")
    args = parser.parse_args()

    config_path = (PROJECT_ROOT / args.config).resolve()
    data_dir = (PROJECT_ROOT / args.data_dir).resolve()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = (PROJECT_ROOT / args.output_root / timestamp).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    memory_lines = [
        "h1 is usually already aligned; avoid changing it.",
        "Most UK gains come from h5 and h20, not from aggressive h30 rewrites.",
        "Unsupported positive h30 drift has repeatedly hurt UK test performance.",
        "If long-horizon evidence is mixed, keep h30 near zero.",
    ]

    candidates = [
        Candidate(
            name="auto_cot_error_stratified",
            literature="Auto-CoT / Zhang et al. 2022",
            aim="Broaden the teaching set to cover diverse historical error modes instead of mostly similar cases.",
            expected_outcome="Could improve h20/h30 if the current prompt is overfitting to a narrow case family; target roughly 22.55-22.61.",
            cot_rf={"example_selection": "error_stratified"},
            llm={},
            hdelta={},
        ),
        Candidate(
            name="active_prompt_recent_high_error",
            literature="Active-Prompt / Diao et al. 2023",
            aim="Bias teaching examples toward recent high-error UK cases where the base TSM is least reliable.",
            expected_outcome="Could help if recent hard regimes matter more than broad average cases; target roughly 22.56-22.62.",
            cot_rf={"example_selection": "recent_high_error", "lookback_days": 180},
            llm={},
            hdelta={},
        ),
        Candidate(
            name="least_to_most",
            literature="Least-to-Most / Zhou et al. 2022",
            aim="Force the reflection step to decide horizons sequentially so h30 cannot drift without earlier-horizon support.",
            expected_outcome="Should help if residual error still comes from horizon coupling; target roughly 22.54-22.60.",
            cot_rf={"reasoning_style": "least_to_most"},
            llm={},
            hdelta={},
        ),
        Candidate(
            name="plan_and_solve",
            literature="Plan-and-Solve / Wang et al. 2023",
            aim="Make reflection plan first and only then emit structured horizon decisions.",
            expected_outcome="Could stabilize structured guidance if the model benefits from an explicit plan phase; target roughly 22.55-22.61.",
            cot_rf={"reasoning_style": "plan_and_solve"},
            llm={},
            hdelta={},
        ),
        Candidate(
            name="program_of_thought_apply",
            literature="Program of Thoughts / Chen et al. 2022",
            aim="Make the apply stage behave more like bounded numeric computation than free-form adjustment writing.",
            expected_outcome="Could reduce apply-stage numeric noise and improve h20/h30; target roughly 22.53-22.59.",
            cot_rf={"apply_style": "program_of_thought"},
            llm={},
            hdelta={},
        ),
        Candidate(
            name="reflexion_memory",
            literature="Reflexion / Shinn et al. 2023",
            aim="Inject persistent UK-specific failure memory so the model avoids repeating known bad long-horizon behaviors.",
            expected_outcome="Could trim repeated h30 mistakes if the model responds to explicit memory; target roughly 22.54-22.60.",
            cot_rf={"reflection_memory": memory_lines},
            llm={},
            hdelta={},
        ),
        Candidate(
            name="harmonized_reasoning",
            literature="Self-Harmonized CoT / Fu et al. 2024",
            aim="Force the model to harmonize matched examples into one dominant pattern instead of chasing example-specific noise.",
            expected_outcome="Could help if the reflection stage is too sensitive to outliers; target roughly 22.54-22.60.",
            cot_rf={"reasoning_style": "harmonized"},
            llm={},
            hdelta={},
        ),
        Candidate(
            name="self_consistency_k3",
            literature="Self-Consistency / Wang et al. 2022",
            aim="Sample multiple apply-stage adjustment paths and aggregate them by median to reduce unstable numeric outputs.",
            expected_outcome="Highest upside among single-change variants if apply noise is still material; target roughly 22.50-22.58.",
            cot_rf={"apply_samples": 3, "apply_aggregation": "median", "apply_temperature": 0.25},
            llm={"cache_enabled": False},
            hdelta={},
        ),
        Candidate(
            name="least_to_most_pot",
            literature="Least-to-Most + Program of Thoughts",
            aim="Combine sequential horizon reasoning with a stricter numeric apply-stage contract.",
            expected_outcome="Could preserve h5/h20 gains while preventing noisy h30 moves; target roughly 22.50-22.57.",
            cot_rf={"reasoning_style": "least_to_most", "apply_style": "program_of_thought"},
            llm={},
            hdelta={},
        ),
        Candidate(
            name="hybrid_memory_sc",
            literature="Reflexion + Harmonized CoT + Self-Consistency",
            aim="Combine persistent failure memory, dominant-pattern reflection, and median aggregation of multiple apply samples.",
            expected_outcome="Best upside but also highest risk of over-constraining the model; target roughly 22.47-22.56.",
            cot_rf={
                "reasoning_style": "harmonized",
                "apply_style": "program_of_thought",
                "reflection_memory": memory_lines,
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
            "compute": {"num_workers": 0},
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
        metrics = _extract_candidate_metrics(run_dir)
        row = {
            "name": candidate.name,
            "literature": candidate.literature,
            "aim": candidate.aim,
            "expected_outcome": candidate.expected_outcome,
            **metrics,
        }
        rows.append(row)
        print(
            f"  -> {candidate.name}: llm_mse_path={row['llm_mse_path']:.6f} "
            f"(delta_vs_tsm={row['delta_vs_tsm']:+.6f})",
            flush=True,
        )

    rows.sort(key=lambda row: row["llm_mse_path"])
    _write_report(out_dir, rows)
    print(f"Sweep results written to {out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
