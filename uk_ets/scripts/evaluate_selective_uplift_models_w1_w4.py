#!/usr/bin/env python3
"""Evaluate continuous uplift models for selective UK ETS refinement."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE_DATASET = (
    PROJECT_ROOT
    / "reports"
    / "uk_ets_selective_residual_helpful_cases"
    / "20260321_180909"
    / "helpful_case_dataset.csv"
)

FEATURE_COLUMNS = [
    "y_vol_20d",
    "target_range_pct",
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
    "base_move_h20_pct",
    "base_move_h30_pct",
    "base_bias_h20",
    "base_bias_h30",
    "matched_teaching_count",
    "support_example_count",
    "positive_memory_count",
    "negative_memory_count",
    "teaching_examples",
    "apply_prompt_length",
    "llm_applied",
]


def _safe_pct(base: float, refined: float) -> float:
    if abs(base) <= 1e-12:
        return np.nan
    return 1.0 - (refined / base)


def _build_logistic() -> Pipeline:
    return Pipeline(
        steps=[
            (
                "prep",
                ColumnTransformer(
                    transformers=[
                        (
                            "num",
                            Pipeline(
                                steps=[
                                    ("impute", SimpleImputer(strategy="median")),
                                    ("scale", StandardScaler()),
                                ]
                            ),
                            FEATURE_COLUMNS,
                        )
                    ]
                ),
            ),
            (
                "model",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    random_state=0,
                ),
            ),
        ]
    )


def _build_gbdt_classifier() -> Pipeline:
    return Pipeline(
        steps=[
            (
                "prep",
                ColumnTransformer(
                    transformers=[
                        ("num", SimpleImputer(strategy="median"), FEATURE_COLUMNS),
                    ]
                ),
            ),
            (
                "model",
                HistGradientBoostingClassifier(
                    max_depth=3,
                    learning_rate=0.05,
                    max_iter=200,
                    min_samples_leaf=20,
                    random_state=0,
                ),
            ),
        ]
    )


def _build_gbdt_regressor() -> Pipeline:
    return Pipeline(
        steps=[
            (
                "prep",
                ColumnTransformer(
                    transformers=[
                        ("num", SimpleImputer(strategy="median"), FEATURE_COLUMNS),
                    ]
                ),
            ),
            (
                "model",
                HistGradientBoostingRegressor(
                    loss="squared_error",
                    max_depth=3,
                    learning_rate=0.05,
                    max_iter=200,
                    min_samples_leaf=20,
                    random_state=0,
                ),
            ),
        ]
    )


def _threshold_search_classifier(model: Pipeline, train_df: pd.DataFrame, test_df: pd.DataFrame, target_col: str) -> tuple[np.ndarray, float]:
    model.fit(train_df[FEATURE_COLUMNS], train_df[target_col].astype(int))
    train_scores = model.predict_proba(train_df[FEATURE_COLUMNS])[:, 1]
    thresholds = np.linspace(0.35, 0.65, 7)
    best_threshold = thresholds[0]
    best_score = -np.inf
    for threshold in thresholds:
        keep = train_scores >= threshold
        filtered = np.where(keep, train_df["llm_path_mse"], train_df["base_path_mse"])
        mean_uplift = train_df["base_path_mse"].mean() - filtered.mean()
        if mean_uplift > best_score:
            best_score = mean_uplift
            best_threshold = threshold
    test_scores = model.predict_proba(test_df[FEATURE_COLUMNS])[:, 1]
    keep_test = test_scores >= best_threshold
    return keep_test, float(best_threshold)


def _threshold_search_regressor(model: Pipeline, train_df: pd.DataFrame, test_df: pd.DataFrame, target_col: str) -> tuple[np.ndarray, float]:
    model.fit(train_df[FEATURE_COLUMNS], train_df[target_col].astype(float))
    train_scores = model.predict(train_df[FEATURE_COLUMNS])
    thresholds = np.array([0.0, 0.01, 0.025, 0.05, 0.1])
    best_threshold = float(thresholds[0])
    best_score = -np.inf
    for threshold in thresholds:
        keep = train_scores > threshold
        filtered = np.where(keep, train_df["llm_path_mse"], train_df["base_path_mse"])
        mean_uplift = train_df["base_path_mse"].mean() - filtered.mean()
        if mean_uplift > best_score:
            best_score = mean_uplift
            best_threshold = float(threshold)
    test_scores = model.predict(test_df[FEATURE_COLUMNS])
    keep_test = test_scores > best_threshold
    return keep_test, best_threshold


def _write_plan(out_dir: Path) -> None:
    lines = [
        "# Continuous Uplift Model Plan",
        "",
        "## Objective",
        "- Replace coarse hard regime buckets with a continuous out-of-sample uplift score.",
        "- Keep the benchmark strictly leave-one-window-out and causal in the same sense as the previous selective filter test.",
        "",
        "## Features",
        "- Pre-decision only: volatility, range, momentum, auction flags, energy interaction features, base move/bias features, retrieval counts, prompt-length metadata, and upstream `llm_applied` state.",
        "- No realized error fields and no post-response adjustment fields are used as predictors.",
        "",
        "## Variants",
        "- `uplift_logistic_helpful_strict`: balanced logistic classifier on `helpful_strict`.",
        "- `uplift_gbdt_helpful_strict`: histogram GBDT classifier on `helpful_strict`.",
        "- `uplift_gbdt_path_reg`: histogram GBDT regressor on realized `path_uplift_abs`.",
        "",
        "## Selection Protocol",
        "- For each held-out window, fit the model on the other three windows only.",
        "- Tune the keep-threshold on the same training windows only by maximizing mean training uplift versus the improved base.",
        "- Freeze the threshold and score the held-out window.",
        "",
        "## Expected Outcomes",
        "- Logistic classifier: likely conservative; expected to beat the hard regime filter but may still trail always-on `wide8`.",
        "- GBDT classifier: best chance of preserving `W3` while trimming `W2/W4`; target is modest improvement over `wide8` of `0.1%` to `0.5%`.",
        "- GBDT regressor: highest variance; may help if uplift magnitude is learnable, but could overfit on the small sample.",
        "",
        "## Primary Endpoint",
        "- Mean W1-W4 path MSE versus the improved base and the current always-on `wide8` refiner.",
    ]
    (out_dir / "plan.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _evaluate_variant(name: str, model: Pipeline, cases: pd.DataFrame, mode: str, target_col: str) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for target_window in sorted(cases["window"].unique()):
        train_df = cases[cases["window"] != target_window].copy()
        test_df = cases[cases["window"] == target_window].copy()

        if mode == "classifier":
            keep_mask, threshold = _threshold_search_classifier(model, train_df, test_df, target_col)
        else:
            keep_mask, threshold = _threshold_search_regressor(model, train_df, test_df, target_col)

        filtered_path = np.where(keep_mask, test_df["llm_path_mse"], test_df["base_path_mse"])
        base_path = float(test_df["base_path_mse"].mean())
        wide8_path = float(test_df["llm_path_mse"].mean())
        filtered_path_mse = float(np.mean(filtered_path))
        rows.append(
            {
                "variant": name,
                "window": target_window,
                "threshold": float(threshold),
                "keep_share": float(np.mean(keep_mask)),
                "base_path_mse": base_path,
                "wide8_path_mse": wide8_path,
                "filtered_path_mse": filtered_path_mse,
                "uplift_vs_base_pct": _safe_pct(base_path, filtered_path_mse),
                "uplift_vs_wide8_pct": _safe_pct(wide8_path, filtered_path_mse),
            }
        )
    return pd.DataFrame(rows)


def _write_outputs(out_dir: Path, results: pd.DataFrame) -> None:
    overall = (
        results.groupby("variant", as_index=False)[
            ["base_path_mse", "wide8_path_mse", "filtered_path_mse", "uplift_vs_base_pct", "uplift_vs_wide8_pct", "keep_share"]
        ]
        .mean()
    )
    lines = [
        "# Continuous Uplift Model Summary",
        "",
        f"Generated: {datetime.now().isoformat()}",
        "",
        "## Overall",
        "",
        overall.to_markdown(index=False),
        "",
        "## By Window",
        "",
        results.to_markdown(index=False),
        "",
    ]
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    best = overall.sort_values("filtered_path_mse").iloc[0]
    analysis = [
        "# Continuous Uplift Model Analysis",
        "",
        f"- Best variant by mean W1-W4 path MSE: `{best['variant']}`",
        f"- Mean filtered path MSE: `{best['filtered_path_mse']:.4f}`",
        f"- Mean uplift vs improved base: `{best['uplift_vs_base_pct']:.4%}`",
        f"- Mean uplift vs always-on wide8: `{best['uplift_vs_wide8_pct']:.4%}`",
        "",
        "Interpretation should focus on whether the best continuous uplift model can beat the improved base out of sample. If it cannot, the result still narrows the failure mode: the current residual signal is too weak or too poorly represented for case-level scoring on this sample size.",
    ]
    (out_dir / "final_analysis.md").write_text("\n".join(analysis) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dataset", type=str, default=str(DEFAULT_SOURCE_DATASET))
    parser.add_argument("--out-dir", type=str, default="")
    args = parser.parse_args()

    if args.out_dir:
        out_dir = Path(args.out_dir).expanduser().resolve()
    else:
        out_dir = (
            PROJECT_ROOT
            / "reports"
            / "uk_ets_selective_uplift_models_w1_w4"
            / datetime.now().strftime("%Y%m%d_%H%M%S")
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_plan(out_dir)

    cases = pd.read_csv(Path(args.source_dataset).expanduser().resolve())

    variants = [
        ("uplift_logistic_helpful_strict", _build_logistic(), "classifier", "helpful_strict"),
        ("uplift_gbdt_helpful_strict", _build_gbdt_classifier(), "classifier", "helpful_strict"),
        ("uplift_gbdt_path_reg", _build_gbdt_regressor(), "regressor", "path_uplift_abs"),
    ]
    results = pd.concat(
        [_evaluate_variant(name, model, cases, mode, target) for name, model, mode, target in variants],
        ignore_index=True,
    )
    results.to_csv(out_dir / "results.csv", index=False)
    _write_outputs(out_dir, results)
    print(out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
