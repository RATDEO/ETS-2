#!/usr/bin/env python3
"""Run the production-candidate Qwen scalar-ramp residual stacker.

This is a two-level forecast. The frozen ridge remains the level-zero model.
Level-one training labels come only from purged, expanding-window ridge
predictions, so they are genuine historical out-of-sample errors. Qwen-labeled
headline event features are compared with matched base-only and numeric-only
ablations on the validation split. The frozen winner is refit and applied once
to the held-out test split.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import yaml
from scipy.stats import ttest_rel, wilcoxon

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from data.panel import select_feature_columns
from eval.metrics import compute_path_metrics
from eval.residual_event_model import (
    ScalarRampCandidate,
    block_columns,
    build_origin_feature_frame,
    extract_optimal_ramp_targets,
    fit_scalar_ramp_candidate,
    predict_scalar_ramp_candidate,
    summarize_prediction,
)
from models.baselines import LinearBaseline
from news.event_panel import build_event_panel
from run_experiment import returns_to_prices

from run_ridge_qwen_residual_stacker_v1 import (
    fold_statistics,
    load_saved_split,
    load_yaml,
    prior_panel_dates,
    resolve_path,
    sha256_file,
    subset_comparisons,
)


DEFAULT_CONFIG = ROOT / "uk_ets" / "config" / "uk_ets_qwen_scalar_ramp_stacker_v1.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--base-run", type=Path, default=None)
    parser.add_argument("--events-path", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def candidate_family(candidate: ScalarRampCandidate) -> str:
    if candidate.estimator == "identity":
        return "identity"
    if "event_core" in candidate.feature_blocks:
        return "qwen"
    if "market" in candidate.feature_blocks:
        return "numeric"
    return "base_only"


def candidate_columns(candidate: ScalarRampCandidate, blocks: dict[str, list[str]]) -> list[str]:
    columns: list[str] = []
    for block in candidate.feature_blocks:
        columns.extend(blocks[block])
    return list(dict.fromkeys(columns))


def make_candidates(cfg: dict[str, Any]) -> list[ScalarRampCandidate]:
    candidate_cfg = cfg.get("candidates", {}) or {}
    candidates = [ScalarRampCandidate("identity", "identity", ("base",))]
    families = {
        "base_only": ("base",),
        "numeric": ("base", "market"),
        "qwen": ("base", "market", "event_core"),
    }
    for family, feature_blocks in families.items():
        for lookback in candidate_cfg.get("lookbacks", [150, 350, 504]):
            for alpha in candidate_cfg.get("ridge_alphas", [100.0, 1000.0, 10000.0]):
                for shrinkage in candidate_cfg.get("shrinkages", [1.0, 1.5]):
                    candidates.append(
                        ScalarRampCandidate(
                            name=f"ramp_{family}_lb{int(lookback)}_a{float(alpha):g}_s{float(shrinkage):g}",
                            estimator="ridge",
                            feature_blocks=feature_blocks,
                            lookback=int(lookback),
                            alpha=float(alpha),
                            max_adjustment_pct=float(candidate_cfg.get("max_adjustment_pct", 5.0)),
                            shrinkage=float(shrinkage),
                            min_adjustment_pct=float(candidate_cfg.get("min_adjustment_pct", 0.0)),
                            ramp_power=float(candidate_cfg.get("ramp_power", 1.0)),
                        )
                    )
    return candidates


def replay_frozen_base(
    base_run: Path,
    base_cfg: dict[str, Any],
    panel: pd.DataFrame,
) -> tuple[dict[str, dict[str, np.ndarray]], LinearBaseline, list[str], int, float]:
    feature_cfg = base_cfg.get("features", {}) or {}
    feature_columns = select_feature_columns(
        panel,
        target_col="y_return",
        max_exogenous_features=int(feature_cfg.get("max_exogenous_features_model", 10)),
        preferred_feature_order=feature_cfg.get("preferred_feature_order"),
    )
    price_idx = feature_columns.index("y")
    with (base_run / "data" / "datasets" / "scaler.pkl").open("rb") as handle:
        scaler_payload = pickle.load(handle)
    splits = {
        name: load_saved_split(base_run, name, scaler_payload, price_idx)
        for name in ("train", "val", "test")
    }
    pred_len = int((base_cfg.get("time_series", {}) or {}).get("pred_len", 30))
    model = LinearBaseline(pred_len=pred_len, model_type="ridge")
    model.fit(splits["train"]["histories"], splits["train"]["y_true"])
    for split in ("val", "test"):
        splits[split]["base_pred"] = model.predict(splits[split]["histories"])

    saved_metrics = pd.read_csv(base_run / "results" / "path_metrics.csv")
    saved_mse = float(saved_metrics.loc[saved_metrics["model"] == "linear_ridge", "mse_path"].iloc[0])
    replay_mse = float(compute_path_metrics(splits["test"]["y_true"], splits["test"]["base_pred"])["mse_path"])
    if not np.isclose(replay_mse, saved_mse, rtol=0.0, atol=2e-5):
        raise RuntimeError(f"Frozen ridge replay mismatch: saved={saved_mse}, replay={replay_mse}")
    return splits, model, feature_columns, price_idx, replay_mse


def make_train_oof_base(
    histories: np.ndarray,
    y_true: np.ndarray,
    initial_apply_start: int,
    fold_size: int,
    purge_size: int,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    indices: list[np.ndarray] = []
    predictions: list[np.ndarray] = []
    audit: list[dict[str, int]] = []
    for apply_start in range(initial_apply_start, len(histories), fold_size):
        apply_end = min(apply_start + fold_size, len(histories))
        fit_end = apply_start - purge_size
        if fit_end < 1:
            raise ValueError("OOF base fold has no fit observations after purge.")
        model = LinearBaseline(pred_len=y_true.shape[1], model_type="ridge")
        model.fit(histories[:fit_end], y_true[:fit_end])
        apply_idx = np.arange(apply_start, apply_end, dtype=int)
        predictions.append(model.predict(histories[apply_idx]))
        indices.append(apply_idx)
        audit.append(
            {
                "fit_start": 0,
                "fit_end": fit_end,
                "apply_start": apply_start,
                "apply_end": apply_end,
                "purge_size": purge_size,
            }
        )
    return np.concatenate(indices), np.concatenate(predictions), pd.DataFrame(audit)


def metric_row(
    name: str,
    family: str,
    y_true: np.ndarray,
    prediction: np.ndarray,
    base: np.ndarray,
) -> dict[str, Any]:
    row = summarize_prediction(name, y_true, prediction)
    base_mse = float(np.mean((y_true - base) ** 2))
    row["family"] = family
    row["improvement_vs_base_pct"] = 100.0 * (base_mse - float(row["path_mse"])) / base_mse
    return row


def select_candidate(
    validation: pd.DataFrame,
    candidates: dict[str, ScalarRampCandidate],
    cfg: dict[str, Any],
) -> ScalarRampCandidate:
    policy = cfg.get("selection", {}) or {}
    eligible = validation.loc[validation["variant"] != "identity"].copy()
    eligible = eligible.loc[
        (eligible["improvement_vs_base_pct"] >= float(policy.get("min_improvement_pct", 1.0)))
        & (eligible["fold_win_share"] >= float(policy.get("min_fold_win_share", 0.5)))
    ].sort_values(["path_mse", "n_features", "variant"])
    return candidates["identity"] if eligible.empty else candidates[str(eligible.iloc[0]["variant"])]


def coefficient_table(bundle: Any, feature_columns: list[str]) -> pd.DataFrame:
    if bundle.model is None or not hasattr(bundle.model[-1], "coef_"):
        return pd.DataFrame(columns=["feature", "coefficient", "abs_coefficient", "block"])
    coefficients = np.asarray(bundle.model[-1].coef_, dtype=float).reshape(-1)
    rows = []
    for feature, coefficient in zip(feature_columns, coefficients, strict=True):
        if feature.startswith("base_"):
            block = "base"
        elif feature.startswith("mkt_"):
            block = "market"
        else:
            block = "qwen_event"
        rows.append(
            {
                "feature": feature,
                "coefficient": coefficient,
                "abs_coefficient": abs(coefficient),
                "block": block,
            }
        )
    return pd.DataFrame(rows).sort_values("abs_coefficient", ascending=False).reset_index(drop=True)


def main() -> int:
    args = parse_args()
    config_path = resolve_path(args.config)
    cfg = load_yaml(config_path)
    base_run = resolve_path(args.base_run or cfg["base_run"])
    events_path = resolve_path(args.events_path or cfg["events_path"])
    output_dir = (
        ROOT / "reports" / "qwen_scalar_ramp_stacker_v1" / datetime.now().strftime("%Y%m%d_%H%M%S")
        if args.output_dir is None
        else resolve_path(args.output_dir)
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    base_cfg = load_yaml(base_run / "config_resolved.yaml")
    if str((base_cfg.get("target", {}) or {}).get("mode")) != "returns":
        raise ValueError("This stacker requires a returns-target frozen base run.")
    panel = pd.read_parquet(base_run / "data" / "panel.parquet")
    splits, base_model, feature_columns, price_idx, replay_mse = replay_frozen_base(base_run, base_cfg, panel)
    pred_len = splits["test"]["y_true"].shape[1]

    oof_cfg = cfg.get("base_oof", {}) or {}
    oof_indices, oof_base, oof_audit = make_train_oof_base(
        splits["train"]["histories"],
        splits["train"]["y_true"],
        initial_apply_start=int(oof_cfg.get("initial_apply_start", 250)),
        fold_size=int(oof_cfg.get("fold_size", 50)),
        purge_size=int(oof_cfg.get("purge_size", pred_len)),
    )
    oof_audit.to_csv(output_dir / "base_oof_fold_audit.csv", index=False)

    event_cfg = cfg.get("event_features", {}) or {}
    event_panel = build_event_panel(
        events_path,
        calendar=panel["date"],
        rolling_windows=event_cfg.get("rolling_windows", [3, 7]),
        taxonomy_version=str(event_cfg.get("taxonomy_version", "v2")),
    )
    train_origins = prior_panel_dates(panel["date"], splits["train"]["dates"][oof_indices])
    val_origins = prior_panel_dates(panel["date"], splits["val"]["dates"])
    test_origins = prior_panel_dates(panel["date"], splits["test"]["dates"])
    train_frame = build_origin_feature_frame(panel, event_panel, train_origins, oof_base)
    val_frame = build_origin_feature_frame(panel, event_panel, val_origins, splits["val"]["base_pred"])
    test_frame = build_origin_feature_frame(panel, event_panel, test_origins, splits["test"]["base_pred"])
    blocks = block_columns(train_frame)
    if not blocks["event_core"]:
        raise RuntimeError("The configured headline file produced no Qwen event features.")

    cap = float((cfg.get("candidates", {}) or {}).get("max_adjustment_pct", 5.0))
    power = float((cfg.get("candidates", {}) or {}).get("ramp_power", 1.0))
    train_targets = extract_optimal_ramp_targets(
        splits["train"]["y_true"][oof_indices], oof_base, cap, power
    )
    val_targets = extract_optimal_ramp_targets(
        splits["val"]["y_true"], splits["val"]["base_pred"], cap, power
    )

    candidates = make_candidates(cfg)
    candidate_map = {candidate.name: candidate for candidate in candidates}
    validation_rows: list[dict[str, Any]] = []
    validation_predictions: dict[str, np.ndarray] = {}
    score_fold_size = int((cfg.get("selection", {}) or {}).get("score_fold_size", 38))
    for candidate in candidates:
        columns = candidate_columns(candidate, blocks)
        bundle = fit_scalar_ramp_candidate(
            candidate,
            train_frame[columns].to_numpy(dtype=float),
            train_targets,
            pred_len=pred_len,
        )
        prediction = predict_scalar_ramp_candidate(
            bundle,
            val_frame[columns].to_numpy(dtype=float),
            splits["val"]["base_pred"],
        )
        validation_predictions[candidate.name] = prediction
        row = metric_row(
            candidate.name,
            candidate_family(candidate),
            splits["val"]["y_true"],
            prediction,
            splits["val"]["base_pred"],
        )
        wins, n_folds, median_improvement = fold_statistics(
            splits["val"]["y_true"], splits["val"]["base_pred"], prediction, score_fold_size
        )
        row.update(
            {
                "fold_wins": wins,
                "n_folds": n_folds,
                "fold_win_share": wins / n_folds,
                "median_fold_improvement_pct": median_improvement,
                "n_features": len(columns),
            }
        )
        validation_rows.append(row)

    validation = pd.DataFrame(validation_rows).sort_values("path_mse").reset_index(drop=True)
    selected = select_candidate(validation, candidate_map, cfg)
    best_by_family = {
        family: candidate_map[str(validation.loc[validation["family"] == family].iloc[0]["variant"])]
        for family in ("base_only", "numeric", "qwen")
    }

    final_candidates = {
        "selected": selected,
        "best_base_only": best_by_family["base_only"],
        "best_numeric": best_by_family["numeric"],
        "best_qwen": best_by_family["qwen"],
    }
    final_predictions: dict[str, np.ndarray] = {}
    final_bundles: dict[str, dict[str, Any]] = {}
    for label, candidate in final_candidates.items():
        columns = candidate_columns(candidate, blocks)
        X_fit = np.vstack(
            [train_frame[columns].to_numpy(dtype=float), val_frame[columns].to_numpy(dtype=float)]
        )
        y_fit = np.concatenate([train_targets, val_targets])
        bundle = fit_scalar_ramp_candidate(candidate, X_fit, y_fit, pred_len=pred_len)
        final_predictions[label] = predict_scalar_ramp_candidate(
            bundle,
            test_frame[columns].to_numpy(dtype=float),
            splits["test"]["base_pred"],
        )
        final_bundles[label] = {"bundle": bundle, "feature_columns": columns}

    test_rows = [
        metric_row(
            "ridge_base", "base", splits["test"]["y_true"], splits["test"]["base_pred"], splits["test"]["base_pred"]
        )
    ]
    for label, prediction in final_predictions.items():
        test_rows.append(
            metric_row(
                label,
                candidate_family(final_candidates[label]),
                splits["test"]["y_true"],
                prediction,
                splits["test"]["base_pred"],
            )
        )
    test_comparison = pd.DataFrame(test_rows).sort_values("path_mse").reset_index(drop=True)

    selected_error = np.mean((splits["test"]["y_true"] - final_predictions["selected"]) ** 2, axis=1)
    base_error = np.mean((splits["test"]["y_true"] - splits["test"]["base_pred"]) ** 2, axis=1)
    t_result = ttest_rel(base_error, selected_error)
    try:
        w_result = wilcoxon(base_error, selected_error)
        wilcoxon_stat, wilcoxon_p = float(w_result.statistic), float(w_result.pvalue)
    except ValueError:
        wilcoxon_stat, wilcoxon_p = float("nan"), float("nan")
    significance = pd.DataFrame(
        [
            {"test": "paired_t", "statistic": float(t_result.statistic), "p_value": float(t_result.pvalue)},
            {"test": "wilcoxon", "statistic": wilcoxon_stat, "p_value": wilcoxon_p},
        ]
    )

    comparison_runs = [resolve_path(value) for value in cfg.get("comparison_runs", [])]
    subset_table = subset_comparisons(
        comparison_runs,
        splits["test"]["dates"],
        splits["test"]["y_true"],
        splits["test"]["base_pred"],
        final_predictions,
    )
    coefficients = coefficient_table(
        final_bundles["selected"]["bundle"], final_bundles["selected"]["feature_columns"]
    )

    validation.to_csv(output_dir / "validation_selection.csv", index=False)
    test_comparison.to_csv(output_dir / "test_comparison.csv", index=False)
    subset_table.to_csv(output_dir / "test_seed_subsets.csv", index=False)
    significance.to_csv(output_dir / "significance_tests.csv", index=False)
    coefficients.to_csv(output_dir / "selected_standardized_coefficients.csv", index=False)
    train_frame.to_parquet(output_dir / "oof_train_features.parquet", index=False)
    val_frame.to_parquet(output_dir / "validation_features.parquet", index=False)
    test_frame.to_parquet(output_dir / "test_features.parquet", index=False)
    np.savez_compressed(
        output_dir / "predictions.npz",
        dates=splits["test"]["dates"].astype(str),
        origin_dates=test_origins.astype(str).to_numpy(),
        y_true=splits["test"]["y_true"],
        ridge_base=splits["test"]["base_pred"],
        **final_predictions,
    )

    source_hash = sha256_file(events_path)
    joblib.dump(
        {
            "version": 1,
            "architecture": "qwen_scalar_ramp_residual_stacker",
            "base_model": base_model,
            "model_feature_columns": feature_columns,
            "price_feature_index": price_idx,
            "event_source": str(events_path),
            "event_source_sha256": source_hash,
            "event_taxonomy_version": str(event_cfg.get("taxonomy_version", "v2")),
            **final_bundles,
        },
        output_dir / "stacker_bundle.joblib",
    )

    selection_payload = {
        "base_run": str(base_run.relative_to(ROOT)),
        "event_source": str(events_path.relative_to(ROOT)),
        "event_source_sha256": source_hash,
        "qwen_role": "precomputed per-headline sentiment, importance, and event taxonomy features",
        "new_qwen_calls_made": False,
        "selected_candidate": asdict(selected),
        "best_by_family": {family: asdict(candidate) for family, candidate in best_by_family.items()},
        "oof_train_rows": int(len(oof_indices)),
        "validation_rows": int(len(splits["val"]["y_true"])),
        "test_rows": int(len(splits["test"]["y_true"])),
        "test_used_for_selection": False,
        "test_interval_previously_inspected_during_project_research": True,
        "ridge_replay_mse": replay_mse,
    }
    (output_dir / "selection.json").write_text(json.dumps(selection_payload, indent=2), encoding="utf-8")
    (output_dir / "config_snapshot.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

    selected_val = validation.loc[validation["variant"] == selected.name].iloc[0]
    selected_test = test_comparison.loc[test_comparison["variant"] == "selected"].iloc[0]
    numeric_test = test_comparison.loc[test_comparison["variant"] == "best_numeric"].iloc[0]
    base_only_test = test_comparison.loc[test_comparison["variant"] == "best_base_only"].iloc[0]
    qwen_test = test_comparison.loc[test_comparison["variant"] == "best_qwen"].iloc[0]
    qwen_coef_share = (
        float(coefficients.loc[coefficients["block"] == "qwen_event", "abs_coefficient"].sum())
        / max(float(coefficients["abs_coefficient"].sum()), 1e-12)
    )
    summary = f"""# Qwen scalar-ramp residual stacker v1

## Outcome

- Frozen candidate: `{selected.name}` ({candidate_family(selected)})
- Validation path MSE improvement: {float(selected_val['improvement_vs_base_pct']):+.3f}%
- Held-out test path MSE: {replay_mse:.6f} → {float(selected_test['path_mse']):.6f} ({float(selected_test['improvement_vs_base_pct']):+.3f}%)
- Qwen-event share of absolute standardized coefficient mass: {100.0 * qwen_coef_share:.1f}%
- Test labels used by this script for model or threshold selection: no

## Matched held-out ablation

| Variant | Path MSE | Improvement vs ridge |
|---|---:|---:|
| Ridge base | {replay_mse:.6f} | 0.000% |
| Base-shape-only stacker | {float(base_only_test['path_mse']):.6f} | {float(base_only_test['improvement_vs_base_pct']):+.3f}% |
| Numeric-only stacker | {float(numeric_test['path_mse']):.6f} | {float(numeric_test['improvement_vs_base_pct']):+.3f}% |
| Qwen-enriched stacker | {float(qwen_test['path_mse']):.6f} | {float(qwen_test['improvement_vs_base_pct']):+.3f}% |

## Protocol

The level-one model was trained on {len(oof_indices)} purged out-of-sample ridge forecasts, selected on all {len(splits['val']['y_true'])} validation origins, refit using only training and validation outcomes, and evaluated on {len(splits['test']['y_true'])} test origins. No new Qwen calls were made in this run; it consumes the frozen Qwen headline-label file identified by SHA-256 in `selection.json`.

Research caveat: this test date interval has been inspected in earlier project experiments and during architecture development. The final script is test-blind, but the result should be treated as a retrospective holdout comparison rather than a pristine never-seen benchmark. Confirm it prospectively or on a newly acquired period before making a production-performance claim.

Paired test p-values: t-test {float(t_result.pvalue):.6f}; Wilcoxon {wilcoxon_p:.6f}.
"""
    (output_dir / "summary.md").write_text(summary, encoding="utf-8")

    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "selected": selected.name,
                "validation_improvement_pct": float(selected_val["improvement_vs_base_pct"]),
                "test": test_comparison.to_dict(orient="records"),
                "significance": significance.to_dict(orient="records"),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
