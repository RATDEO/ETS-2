#!/usr/bin/env python3
"""Rerun the clean TSM live-policy shortlist with explicit learned-gate artifacts."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]

import sys

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from uk_ets.scripts.run_tsm_live_policy_build import _candidates, _run_candidate


SHORTLIST = [
    "baseline_regime_specific",
    "learned_logistic_regime",
    "learned_gbdt_regime",
]


def main() -> None:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = PROJECT_ROOT / "reports" / "uk_ets_tsm_live_policy_shortlist" / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)

    candidate_map = {candidate.name: candidate for candidate in _candidates()}
    selected = [candidate_map[name] for name in SHORTLIST]

    plan_lines = [
        "# TSM Live Policy Shortlist",
        "",
        "## Objective",
        "- Rerun the live-online TSM shortlist after clarifying the learned-gate validation-selection artifact.",
        "- Keep only the production-relevant candidates: old heuristic baseline, learned logistic gate, and learned boosted-tree gate.",
        "",
        "## Candidates",
    ]
    for candidate in selected:
        plan_lines.append(f"- `{candidate.name}`: {candidate.objective} {candidate.expectation}")
    (out_dir / "plan.md").write_text("\n".join(plan_lines) + "\n", encoding="utf-8")

    rows = []
    for candidate in selected:
        rows.append(_run_candidate(candidate))
        pd.DataFrame(rows).to_csv(out_dir / "candidate_results.csv", index=False)

    df = pd.DataFrame(rows).sort_values("llm_mse_path").reset_index(drop=True)
    baseline = df.loc[df["candidate"] == "baseline_regime_specific"].iloc[0]
    best = df.iloc[0]

    report_lines = [
        "# TSM Live Policy Shortlist Report",
        "",
        f"Best candidate: `{best['candidate']}` at path MSE `{best['llm_mse_path']:.6f}`.",
        f"Baseline: `baseline_regime_specific` at `{baseline['llm_mse_path']:.6f}`.",
        "",
        "## Ranking",
    ]
    for _, row in df.iterrows():
        delta_vs_baseline = float(baseline["llm_mse_path"]) - float(row["llm_mse_path"])
        report_lines.append(
            f"- `{row['candidate']}`: path MSE `{row['llm_mse_path']:.6f}`, gain vs base `{row['path_gain_pct']:.3f}%`, "
            f"delta vs baseline `{delta_vs_baseline:+.6f}`, apply calls `{int(row['apply_calls'])}`, "
            f"`h20` gain `{row['gain_h20_pct']:.3f}%`, `h30` gain `{row['gain_h30_pct']:.3f}%`."
        )
    report_lines.extend(
        [
            "",
            "## Interpretation",
            "- This rerun uses the same live-online architecture as the full build, but the learned-gate selection artifact now records the threshold-selection split explicitly.",
            "- The goal is not to change the policy family again; it is to confirm which of the shortlisted gates survives a clean rerun.",
            "",
            "## Artifacts",
            f"- Candidate table: [{(out_dir / 'candidate_results.csv').name}]({out_dir / 'candidate_results.csv'})",
            f"- Plan: [{(out_dir / 'plan.md').name}]({out_dir / 'plan.md'})",
        ]
    )
    (out_dir / "report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    print(out_dir)


if __name__ == "__main__":
    main()
