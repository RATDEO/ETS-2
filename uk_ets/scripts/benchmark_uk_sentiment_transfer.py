#!/usr/bin/env python3
"""Benchmark EU-style SENT transfer variants on UK ETS W0-W4."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]

import sys

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.run_experiment import run_experiment


CONFIG_PATH = "uk_ets/config/uk_ets_llm_4b_current_default.yaml"
DATA_DIR = "uk_ets/Data_auto_uk"
METHOD_NAME = "TSM+LLM-COT-SENT-RF"

ENERGY_INTERACTIONS = [
    "target_range_pct",
    "target_volume",
    "y_vol_20d",
    "y_ma_5d",
    "y_momentum_20d",
    "is_auction_day",
    "uk_icap_secondary_print_day",
    "coal_brent_ratio",
    "coal_brent_ratio_z20",
    "uk_power_gas_vol_ratio_20d",
    "uk_gas_hdd18_surprise_interaction",
    "uk_hdd18_7d_ma",
    "uka_brent_ratio",
    "uka_brent_ratio_z20",
    "uk_gas_vol_20d",
    "uk_temp_mean_c",
]

SENTIMENT_FEATURES = [
    "sent_score",
    "sent_score_3d_ma",
    "sent_score_change_1d",
    "sent_news_volume",
]


@dataclass(frozen=True)
class WindowSpec:
    name: str
    train_end: str
    val_end: str
    test_end: str


@dataclass(frozen=True)
class Candidate:
    name: str
    objective: str
    expectation: str
    overrides: dict[str, Any]


WINDOWS = [
    WindowSpec("W0", "2024-06-30", "2025-06-30", "2026-03-04"),
    WindowSpec("W1", "2023-10-27", "2024-10-26", "2025-06-30"),
    WindowSpec("W2", "2023-02-22", "2024-02-22", "2024-10-26"),
    WindowSpec("W3", "2022-06-20", "2023-06-20", "2024-02-22"),
    WindowSpec("W4", "2021-10-16", "2022-10-16", "2023-06-20"),
]


def _merge(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    out = dict(a)
    for k, v in b.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


def _improved_base_overrides() -> dict[str, Any]:
    return {
        "time_series": {
            "seq_len": 20,
            "label_len": 10,
        },
        "model": {
            "learning_rate": 0.001,
            "max_epochs": 40,
            "early_stopping_patience": 8,
            "dropout": 0.5,
            "weight_decay": 0.02,
            "dlinear_individual": False,
            "loss_type": "huber",
            "loss_huber_beta": 0.5,
        },
        "features": {
            "max_exogenous_features_model": len(ENERGY_INTERACTIONS),
            "preferred_feature_order": ENERGY_INTERACTIONS,
        },
        "llm": {
            "methods": [],
        },
    }


def _sentiment_feature_overrides(sent_path: str) -> dict[str, Any]:
    return {
        "features": {
            "include_sentiment_features": True,
            "sentiment_path": sent_path,
            "sentiment_date_col": "seendate",
            "sentiment_score_col": "sent_score",
            "sentiment_windows": [3, 7],
            "include_sentiment_change": True,
            "include_source_sent_change": True,
            "include_sentiment_volume": True,
            "include_sentiment_interaction": False,
            "sentiment_volume_windows": [3],
            "max_exogenous_features_model": len(ENERGY_INTERACTIONS) + len(SENTIMENT_FEATURES),
            "preferred_feature_order": ENERGY_INTERACTIONS + SENTIMENT_FEATURES,
        },
        "llm": {
            "methods": [],
        },
    }


def _cot_sent_overrides(sent_path: str) -> dict[str, Any]:
    return {
        "llm": {
            "methods": [METHOD_NAME],
            "history_points": 18,
            "prompt_history_points": 18,
            "prompt_sentiment_points": 18,
            "cot_rf": {
                "k_examples": 5,
                "example_selection": "similarity",
                "feature_window": 18,
                "lookback_days": 365,
                "retain_context": True,
                "strict_json_prompt": False,
                "strict_json_response_format": True,
            },
            "sentiment": {
                "enabled": True,
                "path": sent_path,
                "date_col": "seendate",
                "score_col": "sent_score",
                "history_points": 18,
            },
            "blend_grid": {
                "enabled": False,
            },
        },
    }


def _candidates(raw_sent: str, imp_sent: str) -> list[Candidate]:
    return [
        Candidate(
            name="base_energy16",
            objective="Improved UK hard-regime base anchor with no sentiment.",
            expectation="Reference baseline.",
            overrides=_improved_base_overrides(),
        ),
        Candidate(
            name="sent_feature_raw",
            objective="Apply the EU Strategy-3 style sentiment-as-feature path using the UK raw daily sentiment series.",
            expectation="Expected impact: flat to mildly positive if UK news has any incremental statistical signal.",
            overrides=_merge(_improved_base_overrides(), _sentiment_feature_overrides(raw_sent)),
        ),
        Candidate(
            name="sent_feature_importance",
            objective="Sentiment-as-feature using the UK importance-weighted daily series.",
            expectation="Expected impact: slightly better than raw if headline relevance is noisy.",
            overrides=_merge(_improved_base_overrides(), _sentiment_feature_overrides(imp_sent)),
        ),
        Candidate(
            name="cot_sent_raw",
            objective="Apply the EU paper-style CoT-SENT-RF refiner to the UK improved base using the raw UK daily sentiment series.",
            expectation="Expected impact: likely small and regime-dependent; W0/W3 most likely to benefit if the UK corpus is usable.",
            overrides=_merge(_improved_base_overrides(), _cot_sent_overrides(raw_sent)),
        ),
        Candidate(
            name="cot_sent_importance",
            objective="Apply the EU paper-style CoT-SENT-RF refiner to the UK improved base using the importance-weighted UK daily sentiment series.",
            expectation="Expected impact: slightly higher upside than raw, but also more risk of over-thinning the already sparse UK news signal.",
            overrides=_merge(_improved_base_overrides(), _cot_sent_overrides(imp_sent)),
        ),
    ]


def _run_candidate(window: WindowSpec, candidate: Candidate) -> Path:
    overrides = _merge(
        candidate.overrides,
        {
            "split": {
                "train_end": window.train_end,
                "val_end": window.val_end,
                "test_end": window.test_end,
            }
        },
    )
    return Path(run_experiment(config_path=CONFIG_PATH, overrides=overrides, data_dir=DATA_DIR)).resolve()


def _extract_metrics(run_dir: Path, candidate_name: str) -> dict[str, float]:
    path_df = pd.read_csv(run_dir / "results" / "path_metrics.csv")
    horizon_df = pd.read_csv(run_dir / "results" / "metrics_by_horizon.csv")
    out: dict[str, float] = {}

    tsm_row = path_df[path_df["model"] == "tsm"].iloc[0]
    tsm_h = horizon_df[horizon_df["model"] == "tsm"].set_index("horizon")
    out["tsm_path_mse"] = float(tsm_row["mse_path"])
    out["tsm_h20"] = float(tsm_h.loc[20, "mse"])
    out["tsm_h30"] = float(tsm_h.loc[30, "mse"])

    if candidate_name.startswith("cot_sent_"):
        llm_row = path_df[path_df["model"] == METHOD_NAME].iloc[0]
        llm_h = horizon_df[horizon_df["model"] == METHOD_NAME].set_index("horizon")
        out["llm_path_mse"] = float(llm_row["mse_path"])
        out["llm_h20"] = float(llm_h.loc[20, "mse"])
        out["llm_h30"] = float(llm_h.loc[30, "mse"])
        out["path_improvement_pct"] = 1.0 - (out["llm_path_mse"] / out["tsm_path_mse"])
        out["h20_improvement_pct"] = 1.0 - (out["llm_h20"] / out["tsm_h20"])
        out["h30_improvement_pct"] = 1.0 - (out["llm_h30"] / out["tsm_h30"])
    return out


def _write_plan(out_dir: Path) -> None:
    lines = [
        "# UK SENT Transfer Plan",
        "",
        "## Objective",
        "- Apply the existing EU `SENT` pipeline to UK ETS, not a new sentiment family.",
        "- Compare both prompt-side `CoT-SENT-RF` and sentiment-as-feature against the corrected UK base.",
        "",
        "## Candidates",
    ]
    raw_sent = getattr(_write_plan, "_raw_sent", "uk_ets/Data_auto_uk/news/daily_sentiment_uk_qwen_votes3.csv")
    imp_sent = getattr(_write_plan, "_imp_sent", "uk_ets/Data_auto_uk/news/daily_sentiment_uk_qwen_votes3_importance.csv")
    lines.append("")
    lines.append("## Sentiment Inputs")
    lines.append(f"- Raw daily sentiment: `{raw_sent}`")
    lines.append(f"- Importance daily sentiment: `{imp_sent}`")
    lines.append("")
    for candidate in _candidates(raw_sent, imp_sent):
        lines.append(f"- `{candidate.name}`: {candidate.objective}")
        lines.append(f"  Expected outcome: {candidate.expectation}")
    (out_dir / "plan.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_summary(out_dir: Path, df: pd.DataFrame) -> None:
    lines = [
        "# UK SENT Transfer Benchmark",
        "",
        f"Generated: {datetime.now().isoformat()}",
        "",
        "## Results",
        "",
        df.to_markdown(index=False),
    ]
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="")
    parser.add_argument("--raw-sentiment-path", default="uk_ets/Data_auto_uk/news/daily_sentiment_uk_qwen_votes3.csv")
    parser.add_argument("--importance-sentiment-path", default="uk_ets/Data_auto_uk/news/daily_sentiment_uk_qwen_votes3_importance.csv")
    args = parser.parse_args()

    if args.out_dir:
        out_dir = Path(args.out_dir).expanduser().resolve()
    else:
        out_dir = (PROJECT_ROOT / "reports" / "uk_ets_sent_transfer" / datetime.now().strftime("%Y%m%d_%H%M%S")).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_plan._raw_sent = args.raw_sentiment_path
    _write_plan._imp_sent = args.importance_sentiment_path
    _write_plan(out_dir)

    results_path = out_dir / "results.csv"
    rows = pd.read_csv(results_path).to_dict(orient="records") if results_path.exists() else []
    completed = {(str(r["candidate"]), str(r["window"])) for r in rows}

    for candidate in _candidates(args.raw_sentiment_path, args.importance_sentiment_path):
        for window in WINDOWS:
            key = (candidate.name, window.name)
            if key in completed:
                print(f"Skipping completed {candidate.name} {window.name}", flush=True)
                continue
            print(f"Running {candidate.name} {window.name}", flush=True)
            run_dir = _run_candidate(window, candidate)
            rows.append(
                {
                    "candidate": candidate.name,
                    "window": window.name,
                    "train_end": window.train_end,
                    "val_end": window.val_end,
                    "test_end": window.test_end,
                    "run_dir": str(run_dir),
                    **_extract_metrics(run_dir, candidate.name),
                }
            )
            pd.DataFrame(rows).to_csv(results_path, index=False)

    df = pd.DataFrame(rows)
    df.to_csv(results_path, index=False)
    _write_summary(out_dir, df)
    print(out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
