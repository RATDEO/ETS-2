#!/usr/bin/env python3
"""Ablate learned-gate feature subsets for the live TSM online-memory winner."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif

PROJECT_ROOT = Path(__file__).resolve().parents[2]

import sys

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from uk_ets.scripts.run_tsm_live_policy_build import Candidate, _candidates, _deep_merge, _run_candidate


REFERENCE_RUN = PROJECT_ROOT / "runs" / "20260316_103639_143185"
RESULT_SLUG = "TSM_LLM-COT-RF-HDELTA"


@dataclass(frozen=True)
class FeatureCandidate:
    name: str
    objective: str
    expectation: str
    feature_columns: list[str]


def _load_reference_assets() -> tuple[list[str], pd.DataFrame, pd.DataFrame]:
    gate_json = json.loads(
        (
            REFERENCE_RUN / "llm" / "online_memory_gate_selection_TSM_LLM-COT-RF-HDELTA.json"
        ).read_text(encoding="utf-8")
    )
    feature_columns = list(gate_json.get("feature_columns") or [])
    feature_df = pd.read_csv(
        REFERENCE_RUN
        / "results"
        / "online_memory_gate"
        / RESULT_SLUG
        / "val_learned_gate_features.csv"
    )
    label_df = pd.read_csv(
        REFERENCE_RUN
        / "results"
        / "online_memory_gate"
        / RESULT_SLUG
        / "val_learned_gate_labels.csv"
    )
    return feature_columns, feature_df, label_df


def _rank_features(feature_columns: list[str], feature_df: pd.DataFrame, label_df: pd.DataFrame) -> pd.DataFrame:
    X = feature_df[feature_columns].astype(float).fillna(0.0)
    y = label_df["label"].astype(int).to_numpy()
    mi = mutual_info_classif(X, y, random_state=42)
    rank_df = pd.DataFrame(
        {
            "feature": feature_columns,
            "mutual_info": mi,
        }
    )

    # Add a simple signed correlation fallback for interpretability when MI ties.
    corr_rows = []
    for col in feature_columns:
        series = X[col]
        if float(series.std()) < 1e-12:
            corr = 0.0
        else:
            corr = float(np.corrcoef(series.to_numpy(), y)[0, 1])
            if not np.isfinite(corr):
                corr = 0.0
        corr_rows.append(corr)
    rank_df["label_corr"] = corr_rows
    rank_df["abs_label_corr"] = rank_df["label_corr"].abs()
    rank_df = rank_df.sort_values(
        ["mutual_info", "abs_label_corr", "feature"],
        ascending=[False, False, True],
    ).reset_index(drop=True)
    rank_df["rank"] = np.arange(1, len(rank_df) + 1)
    return rank_df


def _candidate_specs(feature_columns: list[str], rank_df: pd.DataFrame) -> list[FeatureCandidate]:
    feature_set = set(feature_columns)

    def keep(cols: list[str]) -> list[str]:
        return [col for col in cols if col in feature_set]

    counts_signals = keep(
        [
            "positive_count",
            "negative_count",
            "positive_signal",
            "negative_signal",
            "net_signal",
            "positive_count_h20",
            "positive_count_h30",
            "negative_count_h20",
            "negative_count_h30",
            "positive_signal_h20",
            "positive_signal_h30",
            "negative_signal_h20",
            "negative_signal_h30",
            "support_example_count",
        ]
    )
    similarity_helpfulness = keep(
        [
            "positive_best_similarity",
            "negative_best_similarity",
            "positive_mean_similarity",
            "negative_mean_similarity",
            "positive_mean_helpfulness",
            "negative_mean_helpfulness",
        ]
    )
    forecast_regime = keep(
        [
            "base_move_h5_pct",
            "base_move_h20_pct",
            "base_move_h30_pct",
            "profile_change_5",
            "profile_change_20",
            "profile_vol_pct",
            "profile_fc_h5",
            "profile_fc_h20",
            "profile_fc_h30",
        ]
    )
    memory_full = counts_signals + similarity_helpfulness
    ranked = rank_df["feature"].tolist()

    return [
        FeatureCandidate(
            name="full_29_control",
            objective="Rerun the current winning learned gbdt gate with the full 29-feature schema.",
            expectation="Expected path MSE around 3.70-3.78. This is the control and current bar to beat.",
            feature_columns=feature_columns,
        ),
        FeatureCandidate(
            name="top_25_mi",
            objective="Trim the weakest four validation-ranked features while keeping most of the full schema.",
            expectation="Expected path MSE around 3.70-3.80 if the current full schema has mild redundancy.",
            feature_columns=ranked[:25],
        ),
        FeatureCandidate(
            name="top_20_mi",
            objective="Use only the top 20 validation-ranked learned-gate features.",
            expectation="Expected path MSE around 3.70-3.82 if the winner mostly depends on the strongest memory and forecast features.",
            feature_columns=ranked[:20],
        ),
        FeatureCandidate(
            name="top_15_mi",
            objective="Force the gate down to a medium-sized 15-feature schema.",
            expectation="Expected path MSE around 3.74-3.88. This should reveal whether the full gate is overparameterized.",
            feature_columns=ranked[:15],
        ),
        FeatureCandidate(
            name="top_10_mi",
            objective="Force the gate to a compact 10-feature schema.",
            expectation="Expected path MSE around 3.80-3.98. If this still works, the gate can likely be simplified materially.",
            feature_columns=ranked[:10],
        ),
        FeatureCandidate(
            name="top_5_mi",
            objective="Stress-test whether only the highest-ranked features carry most of the live signal.",
            expectation="Expected path MSE around 3.95-4.20. Likely too sparse, but a useful lower bound.",
            feature_columns=ranked[:5],
        ),
        FeatureCandidate(
            name="memory_full_20",
            objective="Use only memory-bank evidence and similarity/helpfulness features, with no direct forecast-shape regime inputs.",
            expectation="Expected path MSE around 3.82-4.05. If this holds up, the gate is mostly a memory-quality model.",
            feature_columns=memory_full,
        ),
        FeatureCandidate(
            name="counts_signals_14",
            objective="Use only count/signal/support features, dropping similarity/helpfulness and forecast-shape inputs.",
            expectation="Expected path MSE around 3.90-4.12. This tests whether raw memory balance is enough on its own.",
            feature_columns=counts_signals,
        ),
        FeatureCandidate(
            name="forecast_regime_9",
            objective="Use only direct forecast-shape and regime-profile features, with no explicit memory-quality inputs.",
            expectation="Expected path MSE around 3.95-4.18. This tests whether the gate is really just a contextual base-forecast filter.",
            feature_columns=forecast_regime,
        ),
        FeatureCandidate(
            name="similarity_helpfulness_6",
            objective="Use only similarity/helpfulness quality features from the memory banks.",
            expectation="Expected path MSE around 4.00-4.20. This should be informative, but likely too weak on its own.",
            feature_columns=similarity_helpfulness,
        ),
    ]


def _build_candidate(spec: FeatureCandidate) -> Candidate:
    candidate_map = {candidate.name: candidate for candidate in _candidates()}
    gbdt = candidate_map["learned_gbdt_regime"]
    overrides = _deep_merge(
        gbdt.overrides,
        {
            "llm": {
                "cot_rf": {
                    "online_memory_policy": {
                        "gate": {
                            "learned": {
                                "feature_columns": list(spec.feature_columns),
                            }
                        }
                    }
                }
            }
        },
    )
    return Candidate(
        name=spec.name,
        objective=spec.objective,
        expectation=spec.expectation,
        overrides=overrides,
    )


def main() -> None:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = PROJECT_ROOT / "reports" / "uk_ets_tsm_live_gate_feature_ablation" / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)

    feature_columns, feature_df, label_df = _load_reference_assets()
    rank_df = _rank_features(feature_columns, feature_df, label_df)
    rank_df.to_csv(out_dir / "feature_ranking.csv", index=False)

    specs = _candidate_specs(feature_columns, rank_df)
    plan_lines = [
        "# TSM Live Gate Feature Ablation Plan",
        "",
        "## Objective",
        "- Test whether the current 29-feature learned gbdt live gate is actually optimal.",
        "- Compare semantic feature-family subsets against MI-ranked top-k subsets.",
        "- Keep the live TSM online-memory architecture fixed and only change the learned gate feature columns.",
        "",
        "## Ranking Source",
        f"- Validation feature ranking derived from [{(REFERENCE_RUN / 'results' / 'online_memory_gate' / RESULT_SLUG / 'val_learned_gate_features.csv').name}]({REFERENCE_RUN / 'results' / 'online_memory_gate' / RESULT_SLUG / 'val_learned_gate_features.csv'}) and [{(REFERENCE_RUN / 'results' / 'online_memory_gate' / RESULT_SLUG / 'val_learned_gate_labels.csv').name}]({REFERENCE_RUN / 'results' / 'online_memory_gate' / RESULT_SLUG / 'val_learned_gate_labels.csv'}).",
        "",
        "## Candidates",
    ]
    for spec in specs:
        plan_lines.append(
            f"- `{spec.name}` ({len(spec.feature_columns)} features): {spec.objective} {spec.expectation}"
        )
    (out_dir / "plan.md").write_text("\n".join(plan_lines) + "\n", encoding="utf-8")

    rows: list[dict[str, Any]] = []
    for spec in specs:
        row = _run_candidate(_build_candidate(spec))
        row["feature_count"] = len(spec.feature_columns)
        row["feature_columns"] = json.dumps(spec.feature_columns)
        rows.append(row)
        pd.DataFrame(rows).to_csv(out_dir / "candidate_results.csv", index=False)

    df = pd.DataFrame(rows).sort_values("llm_mse_path").reset_index(drop=True)
    control = df.loc[df["candidate"] == "full_29_control"].iloc[0]
    best = df.iloc[0]

    report_lines = [
        "# TSM Live Gate Feature Ablation Report",
        "",
        f"Best candidate: `{best['candidate']}` with `{int(best['feature_count'])}` features at path MSE `{best['llm_mse_path']:.6f}`.",
        f"Full 29-feature control: `{control['llm_mse_path']:.6f}`.",
        "",
        "## Ranking",
    ]
    for _, row in df.iterrows():
        delta_vs_control = float(control["llm_mse_path"]) - float(row["llm_mse_path"])
        report_lines.append(
            f"- `{row['candidate']}` ({int(row['feature_count'])} features): path MSE `{row['llm_mse_path']:.6f}`, "
            f"gain vs base `{row['path_gain_pct']:.3f}%`, delta vs full-29 `{delta_vs_control:+.6f}`, "
            f"`h20` gain `{row['gain_h20_pct']:.3f}%`, `h30` gain `{row['gain_h30_pct']:.3f}%`."
        )
    report_lines.extend(
        [
            "",
            "## Interpretation",
            "- If a smaller top-k subset matches or beats the full-29 control, the production gate can be simplified without losing live accuracy.",
            "- If the semantic family subsets underperform, the current winner is probably relying on cross-family interactions rather than one dominant feature block.",
            "",
            "## Artifacts",
            f"- Plan: [{(out_dir / 'plan.md').name}]({out_dir / 'plan.md'})",
            f"- Feature ranking: [{(out_dir / 'feature_ranking.csv').name}]({out_dir / 'feature_ranking.csv'})",
            f"- Candidate results: [{(out_dir / 'candidate_results.csv').name}]({out_dir / 'candidate_results.csv'})",
        ]
    )
    (out_dir / "report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    print(out_dir)


if __name__ == "__main__":
    main()
