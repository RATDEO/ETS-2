#!/usr/bin/env python3
"""Offline exact evaluation of conservative regime-calibrated gate thresholds on W1-W4."""

from __future__ import annotations

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

from src.run_experiment import (
    build_online_memory_regime_bucket,
    compute_metrics_by_horizon,
    compute_path_metrics,
    fit_online_memory_learned_gate,
    predict_online_memory_learned_gate_scores,
)


BASELINE_RESULTS_PATH = (
    PROJECT_ROOT
    / "reports"
    / "uk_ets_gate_w1_w4_regularized_base"
    / "20260319_000250"
    / "results.csv"
)
METHOD_NAME = "TSM+LLM-COT-RF-HDELTA"
METHOD_DIR_NAME = "TSM_LLM-COT-RF-HDELTA"


@dataclass(frozen=True)
class CandidateSpec:
    name: str
    regime_cfg: dict[str, Any]


CANDIDATES = [
    CandidateSpec(
        "regime_cal_move_default",
        {
            "max_abs_base_h20_pct": 2.5,
            "max_abs_base_h30_pct": 4.0,
            "use_profile_vol": False,
        },
    ),
    CandidateSpec(
        "regime_cal_move_vol_default",
        {
            "max_abs_base_h20_pct": 2.5,
            "max_abs_base_h30_pct": 4.0,
            "use_profile_vol": True,
            "max_profile_vol_pct": 3.8,
        },
    ),
]


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _extract_baseline_rows() -> pd.DataFrame:
    df = pd.read_csv(BASELINE_RESULTS_PATH)
    return df[df["candidate"] == "gate_top20"].copy().reset_index(drop=True)


def _threshold_grid(global_threshold: float) -> list[float]:
    grid = sorted(
        {
            round(max(0.0, min(0.95, global_threshold + delta)), 3)
            for delta in [0.0, 0.1, 0.2, 0.3]
        }
    )
    return list(grid)


def _fit_bucket_thresholds(
    *,
    select_df: pd.DataFrame,
    global_threshold: float,
    regime_cfg: dict[str, Any],
    min_bucket_rows: int = 6,
    min_selected_rows: int = 3,
) -> tuple[dict[str, float], pd.DataFrame]:
    bucket_series = build_online_memory_regime_bucket(select_df, regime_cfg=regime_cfg).astype(str)
    threshold_rows: list[dict[str, Any]] = []
    bucket_thresholds: dict[str, float] = {}
    grid = _threshold_grid(global_threshold)
    probs = select_df["apply_probability"].astype(float).to_numpy()
    labels = select_df["label"].astype(int).to_numpy()

    for bucket_name in bucket_series.unique():
        bucket_mask = bucket_series.to_numpy() == str(bucket_name)
        bucket_rows = int(np.sum(bucket_mask))
        if bucket_rows < min_bucket_rows:
            bucket_thresholds[str(bucket_name)] = global_threshold
            threshold_rows.append(
                {
                    "regime_bucket": str(bucket_name),
                    "selected_threshold": global_threshold,
                    "bucket_rows": bucket_rows,
                    "selected_rows": 0,
                    "positive_rate_selected": float("nan"),
                    "selection_mode": "fallback_global",
                }
            )
            continue

        best_threshold = global_threshold
        best_score = -np.inf
        best_selected = 0
        best_positive_rate = float("nan")
        for threshold in grid:
            selected = bucket_mask & (probs >= float(threshold))
            selected_rows = int(np.sum(selected))
            if selected_rows < min_selected_rows:
                continue
            positive_rate = float(np.mean(labels[selected]))
            score = positive_rate
            if score > best_score or (
                np.isclose(score, best_score) and selected_rows > best_selected
            ):
                best_score = score
                best_threshold = float(threshold)
                best_selected = selected_rows
                best_positive_rate = positive_rate

        selection_mode = "bucket_fit" if best_score > -np.inf else "fallback_global"
        bucket_thresholds[str(bucket_name)] = best_threshold
        threshold_rows.append(
            {
                "regime_bucket": str(bucket_name),
                "selected_threshold": best_threshold,
                "bucket_rows": bucket_rows,
                "selected_rows": best_selected,
                "positive_rate_selected": best_positive_rate,
                "selection_mode": selection_mode,
            }
        )
    return bucket_thresholds, pd.DataFrame(threshold_rows)


def _evaluate_candidate_for_run(run_dir: Path, candidate: CandidateSpec) -> dict[str, Any]:
    pred_npz = np.load(run_dir / "predictions" / f"{METHOD_NAME}_pred_test_subset.npz", allow_pickle=True)
    y_true = np.asarray(pred_npz["y_true"], dtype=float)
    base_pred = np.asarray(pred_npz["base_pred"], dtype=float)
    baseline_pred = np.asarray(pred_npz["yhat"], dtype=float)

    gate_dir = run_dir / "results" / "online_memory_gate" / METHOD_DIR_NAME
    val_features = pd.read_csv(gate_dir / "val_learned_gate_features.csv")
    val_labels = pd.read_csv(gate_dir / "val_learned_gate_labels.csv")
    test_probs_df = pd.read_csv(gate_dir / "test_learned_gate_probabilities.csv")
    test_features = pd.read_csv(gate_dir / "test_learned_gate_features.csv")

    selection_meta = _load_json(run_dir / "llm" / f"online_memory_gate_selection_{METHOD_DIR_NAME}.json")
    global_threshold = float(selection_meta["threshold"])
    learned_cfg = {
        "model_type": str(selection_meta["model_type"]),
        "feature_columns": list(selection_meta["feature_columns"]),
        "train_fraction": float(selection_meta["threshold_selection_train_size"])
        / max(1, int(selection_meta["threshold_selection_rows_used"]) + int(selection_meta["threshold_selection_train_size"])),
        "random_state": 42,
        "max_depth": 3,
        "learning_rate": 0.05,
        "max_iter": 200,
    }

    labels = val_labels["label"].astype(int).to_numpy()
    selection_bundle, selection_dataset = fit_online_memory_learned_gate(
        feature_df=val_features,
        labels=labels,
        learned_cfg=learned_cfg,
    )
    if selection_bundle is None or selection_dataset.empty:
        raise RuntimeError(f"Could not rebuild learned gate for {run_dir}")

    train_size = int(selection_bundle["train_size"])
    select_df = selection_dataset.iloc[train_size:].copy()
    select_df["apply_probability"] = predict_online_memory_learned_gate_scores(
        select_df.drop(columns=["label"]),
        selection_bundle,
    )
    bucket_thresholds, bucket_df = _fit_bucket_thresholds(
        select_df=select_df,
        global_threshold=global_threshold,
        regime_cfg=candidate.regime_cfg,
    )

    test_bucket = build_online_memory_regime_bucket(test_features, regime_cfg=candidate.regime_cfg).astype(str)
    baseline_apply = test_probs_df["apply_llm_selected"].astype(bool).to_numpy()
    test_probs = test_probs_df["apply_probability"].astype(float).to_numpy()
    candidate_apply = np.zeros(len(test_probs), dtype=bool)
    for i, bucket_name in enumerate(test_bucket.tolist()):
        threshold = float(bucket_thresholds.get(str(bucket_name), global_threshold))
        candidate_apply[i] = bool(baseline_apply[i] and (test_probs[i] >= threshold))

    candidate_pred = np.where(candidate_apply[:, None], baseline_pred, base_pred)
    path_metrics = compute_path_metrics(y_true, candidate_pred)
    by_h = compute_metrics_by_horizon(y_true, candidate_pred, [1, 5, 20, 30])
    return {
        "llm_path_mse": float(path_metrics["mse_path"]),
        "llm_h20": float(by_h.loc[20, "mse"]),
        "llm_h30": float(by_h.loc[30, "mse"]),
        "apply_rate": float(np.mean(candidate_apply)),
        "mean_apply_probability": float(np.mean(test_probs)),
        "selected_thresholds_seen": ",".join(str(v) for v in sorted(set(bucket_thresholds.values()))),
        "regime_thresholds": json.dumps(bucket_thresholds, sort_keys=True),
        "bucket_selection": bucket_df.to_dict(orient="records"),
        "global_threshold": global_threshold,
    }


def _write_plan(out_dir: Path) -> None:
    lines = [
        "# W1-W4 Conservative Regime Calibration Plan",
        "",
        f"Generated: {datetime.now().isoformat()}",
        "",
        "## Objective",
        "",
        "Test whether the current `top20` learned gate improves on `W1-W4` when we keep the same forecast stack and only raise the apply threshold in harder regime buckets.",
        "",
        "This is an exact offline evaluation because it only blocks additional LLM applications relative to the completed `gate_top20` runs.",
        "",
        "## Method",
        "",
        "- Reuse the completed regularized-base `gate_top20` runs",
        "- Rebuild the learned GBDT gate from saved validation features/labels using the same configuration",
        "- Use the validation selection slice only to choose regime-bucket thresholds",
        "- Constrain every bucket threshold to be `>=` the run's original global threshold",
        "- Apply the calibrated thresholds to the saved test probabilities",
        "- Replace newly blocked rows with raw `TSM` predictions and recompute metrics exactly",
        "",
        "## Candidates",
        "",
        "- `regime_cal_move_default`: `moderate_move` vs `large_move` buckets using `|h20| > 2.5%` or `|h30| > 4.0%`",
        "- `regime_cal_move_vol_default`: the same move split plus `profile_vol_pct > 3.8` into a 4-bucket scheme",
        "",
        "## Expected outcome",
        "",
        "- Hard safe-regime blocking already failed, so expected gains are modest",
        "- Success condition: beat the current `gate_top20` mean `W1-W4` path MSE `54.613995`",
        "- Expected lift if this works: roughly `+0.1%` to `+0.5%` on mean `W1-W4` path MSE",
    ]
    (out_dir / "plan.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_summary(out_dir: Path, df: pd.DataFrame) -> None:
    mean_cols = [
        "tsm_path_mse",
        "llm_path_mse",
        "path_improvement_pct",
        "h20_improvement_pct",
        "h30_improvement_pct",
        "apply_rate",
    ]
    mean_df = df.groupby("candidate")[mean_cols].mean().sort_values("llm_path_mse").reset_index()
    lines = [
        "# W1-W4 Conservative Regime Calibration Results",
        "",
        f"Generated: {datetime.now().isoformat()}",
        "",
        "## Results",
        "",
        df.drop(columns=["bucket_selection"]).to_markdown(index=False),
        "",
        "## Mean By Candidate",
        "",
        mean_df.to_markdown(index=False),
    ]
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    out_dir = (
        PROJECT_ROOT
        / "reports"
        / "uk_ets_regime_calibration_w1_w4_offline"
        / datetime.now().strftime("%Y%m%d_%H%M%S")
    ).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_plan(out_dir)

    baseline_df = _extract_baseline_rows()
    rows: list[dict[str, Any]] = []
    for _, base_row in baseline_df.iterrows():
        rows.append(
            {
                "candidate": "gate_top20_baseline",
                "window": str(base_row["window"]),
                "train_end": str(base_row["train_end"]),
                "val_end": str(base_row["val_end"]),
                "test_end": str(base_row["test_end"]),
                "run_dir": str(base_row["run_dir"]),
                "tsm_path_mse": float(base_row["tsm_path_mse"]),
                "llm_path_mse": float(base_row["llm_path_mse"]),
                "llm_h20": float(base_row["llm_h20"]),
                "llm_h30": float(base_row["llm_h30"]),
                "path_improvement_pct": float(base_row["path_improvement_pct"]),
                "h20_improvement_pct": float(base_row["h20_improvement_pct"]),
                "h30_improvement_pct": float(base_row["h30_improvement_pct"]),
                "apply_rate": float("nan"),
                "mean_apply_probability": float("nan"),
                "selected_thresholds_seen": "",
                "regime_thresholds": "",
                "bucket_selection": [],
                "global_threshold": float("nan"),
            }
        )

    for candidate in CANDIDATES:
        for _, base_row in baseline_df.iterrows():
            run_dir = Path(str(base_row["run_dir"]))
            metrics = _evaluate_candidate_for_run(run_dir, candidate)
            rows.append(
                {
                    "candidate": candidate.name,
                    "window": str(base_row["window"]),
                    "train_end": str(base_row["train_end"]),
                    "val_end": str(base_row["val_end"]),
                    "test_end": str(base_row["test_end"]),
                    "run_dir": str(run_dir),
                    "tsm_path_mse": float(base_row["tsm_path_mse"]),
                    **metrics,
                    "path_improvement_pct": 1.0 - (float(metrics["llm_path_mse"]) / float(base_row["tsm_path_mse"])),
                    "h20_improvement_pct": 1.0 - (float(metrics["llm_h20"]) / float(base_row["tsm_h20"])),
                    "h30_improvement_pct": 1.0 - (float(metrics["llm_h30"]) / float(base_row["tsm_h30"])),
                }
            )

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "results.csv", index=False)
    _write_summary(out_dir, df)
    print(out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
