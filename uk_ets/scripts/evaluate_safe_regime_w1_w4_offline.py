#!/usr/bin/env python3
"""Offline safe-regime evaluation using completed gate_top20 W1-W4 runs."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
BASELINE_REPORT_DIR = (
    PROJECT_ROOT
    / "reports"
    / "uk_ets_gate_w1_w4_regularized_base"
    / "20260319_000250"
)
OUT_DIR = (
    PROJECT_ROOT
    / "reports"
    / "uk_ets_safe_regime_w1_w4_offline"
    / datetime.now().strftime("%Y%m%d_%H%M%S")
)


@dataclass(frozen=True)
class Variant:
    name: str
    max_abs_base_h20_pct: float | None = None
    max_abs_base_h30_pct: float | None = None
    max_profile_vol_pct: float | None = None


BASELINE_RUNS = {
    "W1": "20260319_000250_03d354",
    "W2": "20260319_002140_38080d",
    "W3": "20260319_003723_9eaac4",
    "W4": "20260319_005435_9b1195",
}

VARIANTS = [
    Variant("safe_h20_2p0_h30_3p0", max_abs_base_h20_pct=2.0, max_abs_base_h30_pct=3.0),
    Variant("safe_h20_2p5_h30_4p0", max_abs_base_h20_pct=2.5, max_abs_base_h30_pct=4.0),
    Variant("safe_h20_3p0_h30_5p0", max_abs_base_h20_pct=3.0, max_abs_base_h30_pct=5.0),
    Variant(
        "safe_h20_2p5_h30_4p0_vol_3p8",
        max_abs_base_h20_pct=2.5,
        max_abs_base_h30_pct=4.0,
        max_profile_vol_pct=3.8,
    ),
]


def _load_baseline_results() -> pd.DataFrame:
    return pd.read_csv(BASELINE_REPORT_DIR / "results.csv")


def _load_run_artifacts(run_id: str) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    run_dir = PROJECT_ROOT / "runs" / run_id
    gate_df = pd.read_csv(
        run_dir / "results" / "online_memory_gate" / "TSM_LLM-COT-RF-HDELTA" / "test_learned_gate_probabilities.csv"
    ).sort_values("row_idx").reset_index(drop=True)
    pred_npz = np.load(
        run_dir / "predictions" / "TSM+LLM-COT-RF-HDELTA_pred_test_subset.npz",
        allow_pickle=True,
    )
    y_true = np.asarray(pred_npz["y_true"], dtype=float)
    base_pred = np.asarray(pred_npz["base_pred"], dtype=float)
    llm_pred = np.asarray(pred_npz["yhat"], dtype=float)
    return gate_df, y_true, base_pred, llm_pred


def _safe_mask(gate_df: pd.DataFrame, variant: Variant) -> np.ndarray:
    mask = np.ones(len(gate_df), dtype=bool)
    if variant.max_abs_base_h20_pct is not None:
        mask &= gate_df["base_move_h20_pct"].abs().to_numpy(dtype=float) <= float(variant.max_abs_base_h20_pct)
    if variant.max_abs_base_h30_pct is not None:
        mask &= gate_df["base_move_h30_pct"].abs().to_numpy(dtype=float) <= float(variant.max_abs_base_h30_pct)
    if variant.max_profile_vol_pct is not None:
        mask &= gate_df["profile_vol_pct"].to_numpy(dtype=float) <= float(variant.max_profile_vol_pct)
    return mask


def _metrics(y_true: np.ndarray, base_pred: np.ndarray, llm_pred: np.ndarray) -> dict[str, float]:
    mse_path = float(np.mean((llm_pred - y_true) ** 2))
    h20 = float(np.mean((llm_pred[:, 19] - y_true[:, 19]) ** 2))
    h30 = float(np.mean((llm_pred[:, 29] - y_true[:, 29]) ** 2))
    base_path = float(np.mean((base_pred - y_true) ** 2))
    base_h20 = float(np.mean((base_pred[:, 19] - y_true[:, 19]) ** 2))
    base_h30 = float(np.mean((base_pred[:, 29] - y_true[:, 29]) ** 2))
    return {
        "tsm_path_mse": base_path,
        "tsm_h20": base_h20,
        "tsm_h30": base_h30,
        "llm_path_mse": mse_path,
        "llm_h20": h20,
        "llm_h30": h30,
        "path_improvement_pct": 1.0 - (mse_path / base_path),
        "h20_improvement_pct": 1.0 - (h20 / base_h20),
        "h30_improvement_pct": 1.0 - (h30 / base_h30),
    }


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    baseline_df = _load_baseline_results()
    rows: list[dict[str, Any]] = baseline_df.to_dict(orient="records")

    for window, run_id in BASELINE_RUNS.items():
        gate_df, y_true, base_pred, llm_pred = _load_run_artifacts(run_id)
        baseline_apply_mask = gate_df["apply_llm_selected"].astype(bool).to_numpy()
        for variant in VARIANTS:
            safe_ok = _safe_mask(gate_df, variant)
            final_apply = baseline_apply_mask & safe_ok
            final_pred = np.where(final_apply[:, None], llm_pred, base_pred)
            metrics = _metrics(y_true, base_pred, final_pred)
            rows.append(
                {
                    "candidate": variant.name,
                    "window": window,
                    "train_end": baseline_df.loc[baseline_df["window"] == window, "train_end"].iloc[0],
                    "val_end": baseline_df.loc[baseline_df["window"] == window, "val_end"].iloc[0],
                    "test_end": baseline_df.loc[baseline_df["window"] == window, "test_end"].iloc[0],
                    "run_dir": run_id,
                    "apply_rate": float(np.mean(final_apply)),
                    "additional_blocks": int(np.sum(baseline_apply_mask & ~safe_ok)),
                    **metrics,
                }
            )

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "results.csv", index=False)
    mean_df = (
        df.groupby("candidate")[["llm_path_mse", "path_improvement_pct", "h20_improvement_pct", "h30_improvement_pct"]]
        .mean()
        .sort_values("path_improvement_pct", ascending=False)
        .reset_index()
    )
    (OUT_DIR / "mean_by_candidate.csv").write_text(mean_df.to_csv(index=False), encoding="utf-8")
    lines = [
        "# Offline Safe-Regime Evaluation",
        "",
        f"Generated: {datetime.now().isoformat()}",
        "",
        "This evaluation reuses the completed `gate_top20` W1-W4 runs and applies a stricter",
        "safe-regime layer offline. Because the safe-regime layer can only block existing LLM",
        "applications, this is an exact evaluation of the additional filter and does not require",
        "new endpoint calls.",
        "",
        "## Mean By Candidate",
        "",
        mean_df.to_markdown(index=False),
        "",
        "## Full Results",
        "",
        df.to_markdown(index=False),
    ]
    (OUT_DIR / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(OUT_DIR)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
