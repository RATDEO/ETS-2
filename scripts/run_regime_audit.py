#!/usr/bin/env python3
"""Run a regime audit on the current clean benchmark forecast set."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval.regime_audit import (  # noqa: E402
    build_origin_regime_frame,
    gate_candidate_summary,
    oracle_gate_upper_bounds,
    overall_metrics,
    regime_metric_tables,
)


DEFAULT_PAPER_PATH = ROOT / "runs" / "20260228_145058_40d91f" / "predictions" / "TSM+LLM-COT-SENT-RF_pred_test_subset.npz"
DEFAULT_HDELTA_4B_PATH = ROOT / "reports" / "cot_sent_signal_tune" / "strategy2_hdelta_small_model_full_20260301" / "meta_blend_strict_eval_4b" / "test_predictions.npz"
DEFAULT_HDELTA_35B_PATH = ROOT / "reports" / "cot_sent_signal_tune" / "strategy2_hdelta_large_model_full_guard_1p0_20260301" / "meta_blend_strict_eval_35b" / "test_predictions.npz"
DEFAULT_DATA_DIR = ROOT / "Data_auto"
DEFAULT_SENTIMENT_PATH = ROOT / "data" / "news" / "daily_sentiment.csv"
DEFAULT_OUTPUT_ROOT = ROOT / "reports" / "regime_audit"

MODEL_LABELS = {
    "baseline_paper_cot_sent": "Baseline paper CoT-SENT",
    "hdelta_4b_raw": "4B raw HDELTA",
    "tsm": "Raw tsm",
    "hdelta_35b_guarded_raw": "35B raw guarded HDELTA",
}


def _load_npz(path: Path) -> Dict[str, np.ndarray]:
    with np.load(path, allow_pickle=True) as data:
        return {key: data[key] for key in data.files}


def _assert_aligned(reference_dates: np.ndarray, reference_y: np.ndarray, other: Dict[str, np.ndarray], label: str) -> None:
    other_dates = pd.to_datetime(other["dates"]).normalize().to_numpy()
    if not np.array_equal(reference_dates, other_dates):
        raise ValueError(f"Date mismatch for {label}")
    if not np.allclose(reference_y, other["y_true"]):
        raise ValueError(f"y_true mismatch for {label}")


def _write_summary_markdown(
    output_dir: Path,
    thresholds: dict,
    overall_df: pd.DataFrame,
    regime_path_df: pd.DataFrame,
    winners_df: pd.DataFrame,
    gate_summary: dict,
    oracle_df: pd.DataFrame,
) -> None:
    lines = [
        "# Benchmark Regime Audit",
        "",
        "## Overall",
        "",
        "```text",
        overall_df.to_string(index=False),
        "```",
        "",
        "## Thresholds",
        "",
        f"- Volatility median: `{thresholds['volatility_median']:.6f}`",
        f"- Sentiment abs median: `{thresholds['sentiment_abs_median']:.6f}`",
        f"- Return abs median: `{thresholds['return_abs_median']:.6f}`",
        "",
        "## Best Model By Regime Bucket",
        "",
    ]

    for regime_family, family_df in regime_path_df.groupby("regime_family"):
        best_rows = family_df.sort_values("path_mse").groupby("regime_value", as_index=False).first()
        lines.append(f"### {regime_family}")
        lines.append("")
        lines.append("```text")
        lines.append(best_rows[["regime_value", "model", "n_samples", "path_mse"]].to_string(index=False))
        lines.append("```")
        lines.append("")
        top_winners = winners_df[winners_df["regime_family"] == regime_family].copy()
        if not top_winners.empty:
            top_winners = top_winners.sort_values(["regime_value", "winner_share"], ascending=[True, False])
            top_winners = top_winners.groupby("regime_value", as_index=False).first()
            lines.append("```text")
            lines.append(
                top_winners[
                    ["regime_value", "model", "winner_count", "winner_share", "avg_margin_to_second"]
                ].to_string(index=False)
            )
            lines.append("```")
            lines.append("")

    lines.append("## Gating Candidate Summary")
    lines.append("")
    lines.append("```json")
    lines.append(json.dumps(gate_summary, indent=2))
    lines.append("```")
    lines.append("")
    if not oracle_df.empty:
        lines.append("## Oracle Upper Bounds")
        lines.append("")
        lines.append("Ex post only. These use the same sample to choose the best model per regime bucket.")
        lines.append("")
        lines.append("```text")
        lines.append(oracle_df[["regime_family", "oracle_path_mse", "n_buckets"]].to_string(index=False))
        lines.append("```")
        lines.append("")
    (output_dir / "summary.md").write_text("\n".join(lines))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run regime audit for benchmark forecast models.")
    parser.add_argument("--paper-path", type=Path, default=DEFAULT_PAPER_PATH)
    parser.add_argument("--hdelta-4b-path", type=Path, default=DEFAULT_HDELTA_4B_PATH)
    parser.add_argument("--hdelta-35b-path", type=Path, default=DEFAULT_HDELTA_35B_PATH)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--sentiment-path", type=Path, default=DEFAULT_SENTIMENT_PATH)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir or (DEFAULT_OUTPUT_ROOT / timestamp)
    output_dir.mkdir(parents=True, exist_ok=True)

    paper = _load_npz(args.paper_path)
    hdelta_4b = _load_npz(args.hdelta_4b_path)
    hdelta_35b = _load_npz(args.hdelta_35b_path)

    dates = pd.to_datetime(paper["dates"]).normalize().to_numpy()
    y_true = paper["y_true"]
    _assert_aligned(dates, y_true, hdelta_4b, "4B raw HDELTA")
    _assert_aligned(dates, y_true, hdelta_35b, "35B raw guarded HDELTA")

    predictions = {
        MODEL_LABELS["baseline_paper_cot_sent"]: paper["yhat"],
        MODEL_LABELS["hdelta_4b_raw"]: hdelta_4b["llm_pred"],
        MODEL_LABELS["tsm"]: hdelta_4b["tsm_pred"],
        MODEL_LABELS["hdelta_35b_guarded_raw"]: hdelta_35b["llm_pred"],
    }

    overall_df = overall_metrics(y_true, predictions)
    overall_df.to_csv(output_dir / "overall_metrics.csv", index=False)

    origin_df, thresholds = build_origin_regime_frame(
        dates,
        data_dir=args.data_dir,
        sentiment_path=args.sentiment_path,
    )
    origin_df.to_csv(output_dir / "origin_regimes.csv", index=False)

    regime_columns = [
        "volatility_regime",
        "trend_regime_20d",
        "sentiment_alignment_5d",
        "trend_vol_regime",
    ]
    regime_path_df, regime_horizon_df, winner_df = regime_metric_tables(
        origin_df,
        y_true,
        predictions,
        regime_columns=regime_columns,
    )
    regime_path_df.to_csv(output_dir / "regime_path_mse.csv", index=False)
    regime_horizon_df.to_csv(output_dir / "regime_horizon_mse.csv", index=False)
    winner_df.to_csv(output_dir / "regime_winner_counts.csv", index=False)
    oracle_df = oracle_gate_upper_bounds(
        origin_df,
        y_true,
        predictions,
        regime_columns=regime_columns,
    )
    oracle_df.to_csv(output_dir / "oracle_gate_upper_bounds.csv", index=False)

    gate_summary = gate_candidate_summary(regime_path_df)
    metadata = {
        "paper_path": str(args.paper_path),
        "hdelta_4b_path": str(args.hdelta_4b_path),
        "hdelta_35b_path": str(args.hdelta_35b_path),
        "data_dir": str(args.data_dir),
        "sentiment_path": str(args.sentiment_path),
        "thresholds": thresholds.__dict__,
        "recommended_families": gate_summary.get("recommended_families", []),
        "best_oracle_family": oracle_df.iloc[0]["regime_family"] if not oracle_df.empty else None,
        "best_oracle_path_mse": float(oracle_df.iloc[0]["oracle_path_mse"]) if not oracle_df.empty else None,
    }
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    with open(output_dir / "gate_candidate_summary.json", "w") as f:
        json.dump(gate_summary, f, indent=2)

    _write_summary_markdown(
        output_dir,
        thresholds.__dict__,
        overall_df,
        regime_path_df,
        winner_df,
        gate_summary,
        oracle_df,
    )

    print(f"Regime audit written to {output_dir}")
    print(overall_df.to_string(index=False))
    if not oracle_df.empty:
        print("\nOracle upper bounds (ex post, not deployable):")
        print(oracle_df[["regime_family", "oracle_path_mse", "n_buckets"]].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
