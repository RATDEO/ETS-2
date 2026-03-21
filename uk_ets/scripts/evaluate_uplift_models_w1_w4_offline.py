#!/usr/bin/env python3
"""Offline conservative uplift-model benchmark on the completed W1-W4 runs."""

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

from src.run_experiment import compute_metrics_by_horizon, compute_path_metrics


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
    model_type: str


CANDIDATES = [
    CandidateSpec("uplift_logistic", "logistic"),
    CandidateSpec("uplift_gbdt", "gbdt"),
]


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _baseline_df() -> pd.DataFrame:
    df = pd.read_csv(BASELINE_RESULTS_PATH)
    df = df[df["candidate"] == "gate_top20"].copy()
    df["test_end"] = pd.to_datetime(df["test_end"])
    return df.sort_values("test_end").reset_index(drop=True)


def _load_window_artifacts(run_dir: Path) -> dict[str, Any]:
    gate_dir = run_dir / "results" / "online_memory_gate" / METHOD_DIR_NAME
    pred_npz = np.load(run_dir / "predictions" / f"{METHOD_NAME}_pred_test_subset.npz", allow_pickle=True)
    meta = _load_json(run_dir / "llm" / f"online_memory_gate_selection_{METHOD_DIR_NAME}.json")
    return {
        "selection_meta": meta,
        "val_features": pd.read_csv(gate_dir / "val_learned_gate_features.csv"),
        "val_labels": pd.read_csv(gate_dir / "val_learned_gate_labels.csv"),
        "test_features": pd.read_csv(gate_dir / "test_learned_gate_features.csv"),
        "test_labels": pd.read_csv(gate_dir / "test_learned_gate_labels.csv"),
        "test_probs": pd.read_csv(gate_dir / "test_learned_gate_probabilities.csv"),
        "y_true": np.asarray(pred_npz["y_true"], dtype=float),
        "base_pred": np.asarray(pred_npz["base_pred"], dtype=float),
        "baseline_pred": np.asarray(pred_npz["yhat"], dtype=float),
    }


def _feature_columns(artifacts: dict[str, Any]) -> list[str]:
    return list(artifacts["selection_meta"]["feature_columns"])


def _prepare_training_dataset(history: list[dict[str, Any]], current: dict[str, Any]) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for item in history:
        h_val = item["val_features"].copy()
        h_val["label"] = item["val_labels"]["label"].astype(int).to_numpy()
        h_val["gain_path"] = item["val_labels"]["gain_path"].astype(float).to_numpy()
        h_val["source"] = "history_val"
        parts.append(h_val)

        h_test = item["test_features"].copy()
        h_test["label"] = item["test_labels"]["label"].astype(int).to_numpy()
        h_test["gain_path"] = item["test_labels"]["gain_path"].astype(float).to_numpy()
        h_test["source"] = "history_test"
        parts.append(h_test)

    cur_val = current["val_features"].copy()
    cur_val["label"] = current["val_labels"]["label"].astype(int).to_numpy()
    cur_val["gain_path"] = current["val_labels"]["gain_path"].astype(float).to_numpy()
    cur_val["source"] = "current_val"
    parts.append(cur_val)

    return pd.concat(parts, ignore_index=True)


def _build_model(model_type: str):
    if model_type == "logistic":
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        return make_pipeline(
            StandardScaler(),
            LogisticRegression(
                random_state=42,
                max_iter=1000,
                class_weight="balanced",
                C=1.0,
                solver="lbfgs",
            ),
        )
    if model_type == "gbdt":
        from sklearn.ensemble import HistGradientBoostingClassifier

        return HistGradientBoostingClassifier(
            max_depth=3,
            learning_rate=0.05,
            max_iter=200,
            random_state=42,
        )
    raise ValueError(f"Unsupported model_type: {model_type}")


def _fit_model_and_select_threshold(
    train_df: pd.DataFrame,
    feature_columns: list[str],
    model_type: str,
) -> tuple[Any, float, pd.DataFrame]:
    n_rows = len(train_df)
    if n_rows < 24:
        fit_df = train_df.copy()
        select_df = train_df.copy()
    else:
        fit_size = max(12, int(round(n_rows * 0.75)))
        fit_size = min(fit_size, n_rows - 1)
        fit_df = train_df.iloc[:fit_size].copy()
        select_df = train_df.iloc[fit_size:].copy()
        if select_df.empty:
            select_df = fit_df.copy()

    X_fit = fit_df[feature_columns].astype(float).fillna(0.0).to_numpy()
    y_fit = fit_df["label"].astype(int).to_numpy()

    model = _build_model(model_type)
    model.fit(X_fit, y_fit)

    X_select = select_df[feature_columns].astype(float).fillna(0.0).to_numpy()
    if hasattr(model, "predict_proba"):
        probs = np.asarray(model.predict_proba(X_select)[:, 1], dtype=float)
    else:
        probs = np.asarray(model.predict(X_select), dtype=float)

    grid = [0.5, 0.6, 0.7, 0.8]
    min_selected = max(6, int(round(len(select_df) * 0.08)))
    best_threshold = 0.5
    best_score = -np.inf
    threshold_rows: list[dict[str, Any]] = []

    for threshold in grid:
        selected = probs >= float(threshold)
        selected_rows = int(np.sum(selected))
        if selected_rows < min_selected:
            threshold_rows.append(
                {
                    "threshold": float(threshold),
                    "selected_rows": selected_rows,
                    "mean_gain_path": float("nan"),
                    "positive_rate": float("nan"),
                    "eligible": False,
                }
            )
            continue
        mean_gain = float(select_df.loc[selected, "gain_path"].mean())
        positive_rate = float(select_df.loc[selected, "label"].mean())
        threshold_rows.append(
            {
                "threshold": float(threshold),
                "selected_rows": selected_rows,
                "mean_gain_path": mean_gain,
                "positive_rate": positive_rate,
                "eligible": True,
            }
        )
        score = mean_gain
        if score > best_score or (np.isclose(score, best_score) and selected_rows > 0):
            best_score = score
            best_threshold = float(threshold)

    final_model = _build_model(model_type)
    X_full = train_df[feature_columns].astype(float).fillna(0.0).to_numpy()
    y_full = train_df["label"].astype(int).to_numpy()
    final_model.fit(X_full, y_full)
    return final_model, best_threshold, pd.DataFrame(threshold_rows)


def _predict_probs(model: Any, feature_df: pd.DataFrame, feature_columns: list[str]) -> np.ndarray:
    X = feature_df[feature_columns].astype(float).fillna(0.0).to_numpy()
    if hasattr(model, "predict_proba"):
        return np.asarray(model.predict_proba(X)[:, 1], dtype=float)
    return np.asarray(model.predict(X), dtype=float)


def _evaluate_candidate_window(
    *,
    candidate: CandidateSpec,
    history: list[dict[str, Any]],
    current: dict[str, Any],
    baseline_row: pd.Series,
) -> dict[str, Any]:
    train_df = _prepare_training_dataset(history, current)
    feature_columns = _feature_columns(current)
    model, threshold, threshold_df = _fit_model_and_select_threshold(
        train_df=train_df,
        feature_columns=feature_columns,
        model_type=candidate.model_type,
    )

    test_probs = _predict_probs(model, current["test_features"], feature_columns)
    baseline_apply = current["test_probs"]["apply_llm_selected"].astype(bool).to_numpy()
    candidate_apply = baseline_apply & (test_probs >= threshold)

    candidate_pred = np.where(candidate_apply[:, None], current["baseline_pred"], current["base_pred"])
    path_metrics = compute_path_metrics(current["y_true"], candidate_pred)
    by_h = compute_metrics_by_horizon(current["y_true"], candidate_pred, [1, 5, 20, 30])

    return {
        "tsm_path_mse": float(baseline_row["tsm_path_mse"]),
        "llm_path_mse": float(path_metrics["mse_path"]),
        "llm_h20": float(by_h.loc[20, "mse"]),
        "llm_h30": float(by_h.loc[30, "mse"]),
        "path_improvement_pct": 1.0 - (float(path_metrics["mse_path"]) / float(baseline_row["tsm_path_mse"])),
        "h20_improvement_pct": 1.0 - (float(by_h.loc[20, "mse"]) / float(baseline_row["tsm_h20"])),
        "h30_improvement_pct": 1.0 - (float(by_h.loc[30, "mse"]) / float(baseline_row["tsm_h30"])),
        "apply_rate": float(np.mean(candidate_apply)),
        "mean_uplift_probability": float(np.mean(test_probs)),
        "selected_threshold": float(threshold),
        "training_rows": int(len(train_df)),
        "training_positive_rate": float(train_df["label"].mean()),
        "threshold_selection": threshold_df.to_dict(orient="records"),
    }


def _write_plan(out_dir: Path) -> None:
    lines = [
        "# W1-W4 Uplift Model Benchmark Plan",
        "",
        f"Generated: {datetime.now().isoformat()}",
        "",
        "## Objective",
        "",
        "Test stronger learned uplift models for the online apply policy using the completed `W1-W4` runs, while keeping the underlying forecast stack fixed.",
        "",
        "## What is being tested",
        "",
        "- `uplift_logistic`: logistic classifier that predicts whether the LLM helps",
        "- `uplift_gbdt`: nonlinear GBDT classifier for the same target",
        "",
        "## Training protocol",
        "",
        "- Windows are processed chronologically: `W4 -> W3 -> W2 -> W1`",
        "- For each target window, the uplift model is trained on:",
        "  - all prior windows' validation and test rows",
        "  - the current window's validation rows",
        "- Thresholds are selected on a chronological holdout inside that training pool",
        "",
        "## Important constraint",
        "",
        "This is a conservative exact offline benchmark.",
        "",
        "Because we only have saved final predictions for the completed `gate_top20` runs, these uplift models can only block additional LLM applications relative to the baseline. They cannot add new LLM applications where the baseline skipped them.",
        "",
        "## Expected outcome",
        "",
        "- The logistic uplift model should be a stable low-variance baseline, but may be flat",
        "- The GBDT uplift model has the better chance of improving `W1-W4`",
        "- Success condition: beat the current `gate_top20` mean `W1-W4` path MSE `54.613995`",
        "- Expected lift if the idea works even conservatively: roughly `+0.2%` to `+1.0%` on mean `W1-W4` path MSE",
    ]
    (out_dir / "plan.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_summary(out_dir: Path, df: pd.DataFrame) -> None:
    mean_df = (
        df.groupby("candidate")[["llm_path_mse", "path_improvement_pct", "h20_improvement_pct", "h30_improvement_pct", "apply_rate"]]
        .mean()
        .sort_values("llm_path_mse")
        .reset_index()
    )
    lines = [
        "# W1-W4 Uplift Model Benchmark",
        "",
        f"Generated: {datetime.now().isoformat()}",
        "",
        "## Results",
        "",
        df.drop(columns=["threshold_selection"]).to_markdown(index=False),
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
        / "uk_ets_uplift_models_w1_w4_offline"
        / datetime.now().strftime("%Y%m%d_%H%M%S")
    ).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_plan(out_dir)

    baseline_df = _baseline_df()
    artifacts_by_window: dict[str, dict[str, Any]] = {}
    for _, row in baseline_df.iterrows():
        artifacts_by_window[str(row["window"])] = _load_window_artifacts(Path(str(row["run_dir"])))

    rows: list[dict[str, Any]] = []
    for _, row in baseline_df.iterrows():
        rows.append(
            {
                "candidate": "gate_top20_baseline",
                "window": str(row["window"]),
                "train_end": str(row["train_end"]),
                "val_end": str(row["val_end"]),
                "test_end": str(row["test_end"].date()),
                "run_dir": str(row["run_dir"]),
                "tsm_path_mse": float(row["tsm_path_mse"]),
                "llm_path_mse": float(row["llm_path_mse"]),
                "llm_h20": float(row["llm_h20"]),
                "llm_h30": float(row["llm_h30"]),
                "path_improvement_pct": float(row["path_improvement_pct"]),
                "h20_improvement_pct": float(row["h20_improvement_pct"]),
                "h30_improvement_pct": float(row["h30_improvement_pct"]),
                "apply_rate": float(row.get("apply_rate", np.nan)),
                "mean_uplift_probability": float("nan"),
                "selected_threshold": float("nan"),
                "training_rows": 0,
                "training_positive_rate": float("nan"),
                "threshold_selection": [],
            }
        )

    history: list[dict[str, Any]] = []
    for _, row in baseline_df.iterrows():
        window = str(row["window"])
        current = artifacts_by_window[window]
        for candidate in CANDIDATES:
            metrics = _evaluate_candidate_window(
                candidate=candidate,
                history=history,
                current=current,
                baseline_row=row,
            )
            rows.append(
                {
                    "candidate": candidate.name,
                    "window": window,
                    "train_end": str(row["train_end"]),
                    "val_end": str(row["val_end"]),
                    "test_end": str(row["test_end"].date()),
                    "run_dir": str(row["run_dir"]),
                    **metrics,
                }
            )
        history.append(current)

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "results.csv", index=False)
    _write_summary(out_dir, df)

    analysis_lines = [
        "# Final Analysis",
        "",
        "This benchmark tests stronger learned uplift policies under a conservative exact offline constraint.",
        "",
        "Because only the completed `gate_top20` final predictions are available, the uplift models can suppress harmful baseline applications but cannot add new LLM applications where the baseline skipped them.",
        "",
        "That means any improvement here is meaningful, but failure does not fully rule out a stronger live rerun where the uplift model can both add and remove applications.",
    ]
    (out_dir / "final_analysis.md").write_text("\n".join(analysis_lines) + "\n", encoding="utf-8")
    print(out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
