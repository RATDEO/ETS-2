#!/usr/bin/env python3
"""Leakage-safe supervised refinement over the frozen ridge price forecast.

The LLM is used upstream to turn headlines into stable daily sentiment,
importance, and event-taxonomy features. This script tests those Qwen-derived
features against matched non-LLM residual models using purged expanding-window
validation, then evaluates the frozen choice once on the held-out test split.
"""

from __future__ import annotations

import argparse
import hashlib
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

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from data.panel import select_feature_columns
from eval.metrics import compute_path_metrics
from eval.residual_event_model import (
    ResidualCandidate,
    block_columns,
    build_origin_feature_frame,
    extract_anchor_residual_targets,
    fit_residual_candidate,
    predict_residual_candidate,
    summarize_prediction,
    walk_forward_candidate_predictions,
)
from models.baselines import LinearBaseline
from news.event_panel import build_event_panel
from run_experiment import returns_to_prices


DEFAULT_CONFIG = ROOT / "uk_ets" / "config" / "uk_ets_qwen_residual_stacker_v1.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--base-run", type=Path, default=None)
    parser.add_argument("--events-path", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def resolve_path(path: str | Path) -> Path:
    value = Path(path).expanduser()
    return value if value.is_absolute() else ROOT / value


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    return payload or {}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_saved_split(
    base_run: Path,
    split: str,
    scaler_payload: dict[str, np.ndarray],
    price_idx: int,
) -> dict[str, np.ndarray]:
    saved = np.load(base_run / "data" / "datasets" / f"{split}.npz", allow_pickle=True)
    X_enc = saved["X_enc"] * scaler_payload["std_"] + scaler_payload["mean_"]
    returns = saved["y"] * scaler_payload["std_"][0] + scaler_payload["mean_"][0]
    histories = X_enc[:, :, price_idx]
    y_true = returns_to_prices(returns, histories[:, -1])
    return {
        "histories": np.asarray(histories, dtype=float),
        "y_true": np.asarray(y_true, dtype=float),
        "dates": pd.to_datetime(saved["dates"]).to_numpy(),
    }


def prior_panel_dates(panel_dates: pd.Series, prediction_dates: np.ndarray) -> pd.DatetimeIndex:
    panel_index = pd.DatetimeIndex(pd.to_datetime(panel_dates)).normalize().sort_values().unique()
    prediction_index = pd.DatetimeIndex(pd.to_datetime(prediction_dates)).normalize()
    positions = panel_index.searchsorted(prediction_index, side="left") - 1
    if np.any(positions < 0):
        raise ValueError("At least one prediction date has no prior panel observation.")
    return pd.DatetimeIndex(panel_index[positions])


def candidate_columns(
    candidate: ResidualCandidate,
    blocks: dict[str, list[str]],
) -> list[str]:
    columns: list[str] = []
    for block in candidate.feature_blocks:
        if block not in blocks:
            raise KeyError(f"Unknown feature block {block!r} for {candidate.name}.")
        columns.extend(blocks[block])
    return list(dict.fromkeys(columns))


def candidate_family(candidate: ResidualCandidate) -> str:
    if candidate.estimator == "identity":
        return "identity"
    return "qwen" if "event_core" in candidate.feature_blocks else "numeric"


def make_candidates(cfg: dict[str, Any]) -> list[ResidualCandidate]:
    candidate_cfg = cfg.get("candidates", {}) or {}
    cap = float(candidate_cfg.get("max_adjustment_pct", 5.0))
    alphas = [float(value) for value in candidate_cfg.get("ridge_alphas", [10.0, 100.0, 1000.0])]
    shrinkages = [float(value) for value in candidate_cfg.get("shrinkages", [0.25, 0.5, 1.0])]
    candidates = [ResidualCandidate("identity", "identity", ("base",))]

    block_sets = {
        "base": ("base",),
        "numeric": ("base", "market"),
        "qwen": ("base", "market", "event_core"),
    }
    for family, feature_blocks in block_sets.items():
        for alpha in alphas:
            for shrinkage in shrinkages:
                candidates.append(
                    ResidualCandidate(
                        name=f"ridge_{family}_a{alpha:g}_s{shrinkage:g}",
                        estimator="ridge",
                        feature_blocks=feature_blocks,
                        target_kind="pct",
                        alpha=alpha,
                        max_adjustment_pct=cap,
                        shrinkage=shrinkage,
                    )
                )

    hgb_cfg = candidate_cfg.get("hgb", {}) or {}
    if bool(hgb_cfg.get("enabled", True)):
        for family in ("numeric", "qwen"):
            for shrinkage in shrinkages:
                candidates.append(
                    ResidualCandidate(
                        name=f"hgb_{family}_s{shrinkage:g}",
                        estimator="hgb",
                        feature_blocks=block_sets[family],
                        target_kind="pct",
                        learning_rate=float(hgb_cfg.get("learning_rate", 0.04)),
                        max_depth=int(hgb_cfg.get("max_depth", 2)),
                        max_iter=int(hgb_cfg.get("max_iter", 200)),
                        min_samples_leaf=int(hgb_cfg.get("min_samples_leaf", 12)),
                        l2_regularization=float(hgb_cfg.get("l2_regularization", 2.0)),
                        max_adjustment_pct=cap,
                        shrinkage=shrinkage,
                    )
                )
    return candidates


def metric_row(
    variant: str,
    family: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    base_pred: np.ndarray,
) -> dict[str, Any]:
    row = summarize_prediction(variant, y_true, y_pred)
    base_mse = float(compute_path_metrics(y_true, base_pred)["mse_path"])
    row["family"] = family
    row["improvement_vs_base_pct"] = 100.0 * (base_mse - float(row["path_mse"])) / base_mse
    return row


def fold_statistics(
    y_true: np.ndarray,
    base_pred: np.ndarray,
    candidate_pred: np.ndarray,
    fold_size: int,
) -> tuple[int, int, float]:
    wins = 0
    fold_improvements: list[float] = []
    for start in range(0, len(y_true), fold_size):
        end = min(start + fold_size, len(y_true))
        base_mse = float(np.mean((y_true[start:end] - base_pred[start:end]) ** 2))
        candidate_mse = float(np.mean((y_true[start:end] - candidate_pred[start:end]) ** 2))
        wins += int(candidate_mse < base_mse)
        fold_improvements.append(100.0 * (base_mse - candidate_mse) / base_mse)
    return wins, len(fold_improvements), float(np.median(fold_improvements))


def select_candidate(
    validation: pd.DataFrame,
    candidates_by_name: dict[str, ResidualCandidate],
    cfg: dict[str, Any],
) -> ResidualCandidate:
    identity_mse = float(validation.loc[validation["variant"] == "identity", "path_mse"].iloc[0])
    eligible = validation.loc[validation["variant"] != "identity"].sort_values("path_mse")
    policy = cfg.get("selection", {}) or {}
    min_improvement = float(policy.get("min_improvement_pct", 0.0))
    min_fold_win_share = float(policy.get("min_fold_win_share", 0.5))
    eligible = eligible.loc[
        (eligible["improvement_vs_base_pct"] >= min_improvement)
        & (eligible["fold_win_share"] >= min_fold_win_share)
    ]
    if eligible.empty:
        return candidates_by_name["identity"]
    chosen = eligible.iloc[0]
    if float(chosen["path_mse"]) >= identity_mse:
        return candidates_by_name["identity"]
    return candidates_by_name[str(chosen["variant"])]


def subset_comparisons(
    comparison_runs: list[Path],
    full_dates: np.ndarray,
    y_true: np.ndarray,
    base_pred: np.ndarray,
    variants: dict[str, np.ndarray],
) -> pd.DataFrame:
    date_to_index = {str(pd.Timestamp(value).date()): idx for idx, value in enumerate(full_dates)}
    rows: list[dict[str, Any]] = []
    for run in comparison_runs:
        files = sorted((run / "predictions").glob("*test_subset.npz"))
        if not files:
            continue
        subset = np.load(files[0], allow_pickle=True)
        subset_dates = [str(pd.Timestamp(str(value)).date()) for value in subset["dates"]]
        indices = np.asarray([date_to_index[value] for value in subset_dates], dtype=int)
        run_label = run.name
        for variant, pred in {"ridge_base": base_pred, **variants}.items():
            row = metric_row(
                variant,
                "base" if variant == "ridge_base" else variant,
                y_true[indices],
                pred[indices],
                base_pred[indices],
            )
            row.update({"comparison_run": run_label, "n_samples": len(indices)})
            rows.append(row)
    return pd.DataFrame(rows)


def main() -> int:
    args = parse_args()
    config_path = resolve_path(args.config)
    stack_cfg = load_yaml(config_path)
    base_run = resolve_path(args.base_run or stack_cfg["base_run"])
    events_path = resolve_path(args.events_path or stack_cfg["events_path"])
    if args.output_dir is None:
        output_dir = ROOT / "reports" / "ridge_qwen_residual_stacker_v1" / datetime.now().strftime("%Y%m%d_%H%M%S")
    else:
        output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    base_cfg = load_yaml(base_run / "config_resolved.yaml")
    panel = pd.read_parquet(base_run / "data" / "panel.parquet")
    target_mode = str((base_cfg.get("target", {}) or {}).get("mode", "price"))
    if target_mode != "returns":
        raise ValueError("The frozen base run must use target.mode=returns.")

    features_cfg = base_cfg.get("features", {}) or {}
    feature_columns = select_feature_columns(
        panel,
        target_col="y_return",
        max_exogenous_features=int(features_cfg.get("max_exogenous_features_model", 10)),
        preferred_feature_order=features_cfg.get("preferred_feature_order"),
    )
    price_idx = feature_columns.index("y")
    with (base_run / "data" / "datasets" / "scaler.pkl").open("rb") as handle:
        scaler_payload = pickle.load(handle)
    splits = {
        name: load_saved_split(base_run, name, scaler_payload, price_idx)
        for name in ("train", "val", "test")
    }

    pred_len = int((base_cfg.get("time_series", {}) or {}).get("pred_len", 30))
    base_model = LinearBaseline(pred_len=pred_len, model_type="ridge")
    base_model.fit(splits["train"]["histories"], splits["train"]["y_true"])
    val_base = base_model.predict(splits["val"]["histories"])
    test_base = base_model.predict(splits["test"]["histories"])

    saved_metrics = pd.read_csv(base_run / "results" / "path_metrics.csv")
    frozen_mse = float(saved_metrics.loc[saved_metrics["model"] == "linear_ridge", "mse_path"].iloc[0])
    replay_mse = float(compute_path_metrics(splits["test"]["y_true"], test_base)["mse_path"])
    if not np.isclose(replay_mse, frozen_mse, rtol=0.0, atol=2e-5):
        raise RuntimeError(f"Ridge replay mismatch: saved={frozen_mse}, replay={replay_mse}")

    event_panel = build_event_panel(
        events_path,
        calendar=panel["date"],
        rolling_windows=stack_cfg.get("event_features", {}).get("rolling_windows", [3, 7]),
        taxonomy_version=str(stack_cfg.get("event_features", {}).get("taxonomy_version", "v2")),
    )
    val_origins = prior_panel_dates(panel["date"], splits["val"]["dates"])
    test_origins = prior_panel_dates(panel["date"], splits["test"]["dates"])
    val_frame = build_origin_feature_frame(panel, event_panel, val_origins, val_base)
    test_frame = build_origin_feature_frame(panel, event_panel, test_origins, test_base)
    blocks = block_columns(val_frame)
    if not blocks["event_core"]:
        raise RuntimeError("No Qwen event-core columns were built.")

    candidates = make_candidates(stack_cfg)
    candidates_by_name = {candidate.name: candidate for candidate in candidates}
    validation_cfg = stack_cfg.get("validation", {}) or {}
    initial_train_size = int(validation_cfg.get("initial_train_size", 96))
    fold_size = int(validation_cfg.get("fold_size", 24))
    purge_size = int(validation_cfg.get("purge_size", pred_len))
    anchor_horizons = tuple(int(value) for value in stack_cfg.get("anchor_horizons", [5, 10, 20, 30]))
    val_anchor_pct = extract_anchor_residual_targets(
        splits["val"]["y_true"],
        val_base,
        anchor_horizons=anchor_horizons,
        target_kind="pct",
    )

    validation_rows: list[dict[str, Any]] = []
    oof_predictions: dict[str, np.ndarray] = {}
    audit_folds: list[dict[str, Any]] = []
    expected_indices: np.ndarray | None = None
    for candidate in candidates:
        columns = candidate_columns(candidate, blocks)
        X_val = val_frame[columns].to_numpy(dtype=float)
        indices, predictions, folds = walk_forward_candidate_predictions(
            candidate,
            X_val,
            val_anchor_pct,
            val_base,
            initial_train_size=initial_train_size,
            fold_size=fold_size,
            purge_size=purge_size,
            pred_len=pred_len,
            anchor_horizons=anchor_horizons,
        )
        if expected_indices is None:
            expected_indices = indices
            audit_folds = [asdict(fold) for fold in folds]
        elif not np.array_equal(indices, expected_indices):
            raise RuntimeError("Candidates produced inconsistent validation indices.")
        oof_predictions[candidate.name] = predictions
        row = metric_row(
            candidate.name,
            candidate_family(candidate),
            splits["val"]["y_true"][indices],
            predictions,
            val_base[indices],
        )
        wins, n_folds, median_fold_improvement = fold_statistics(
            splits["val"]["y_true"][indices],
            val_base[indices],
            predictions,
            fold_size,
        )
        row.update(
            {
                "fold_wins": wins,
                "n_folds": n_folds,
                "fold_win_share": wins / n_folds,
                "median_fold_improvement_pct": median_fold_improvement,
                "n_features": len(columns),
            }
        )
        validation_rows.append(row)

    validation = pd.DataFrame(validation_rows).sort_values("path_mse").reset_index(drop=True)
    selected = select_candidate(validation, candidates_by_name, stack_cfg)
    best_by_family: dict[str, ResidualCandidate] = {}
    for family in ("numeric", "qwen"):
        family_rows = validation.loc[validation["family"] == family].sort_values("path_mse")
        best_by_family[family] = candidates_by_name[str(family_rows.iloc[0]["variant"])]

    final_predictions: dict[str, np.ndarray] = {}
    fitted_bundles: dict[str, Any] = {}
    for label, candidate in {
        "selected": selected,
        "best_numeric": best_by_family["numeric"],
        "best_qwen": best_by_family["qwen"],
    }.items():
        columns = candidate_columns(candidate, blocks)
        bundle = fit_residual_candidate(
            candidate,
            val_frame[columns].to_numpy(dtype=float),
            val_anchor_pct,
            pred_len=pred_len,
            anchor_horizons=anchor_horizons,
        )
        final_predictions[label] = predict_residual_candidate(
            bundle,
            test_frame[columns].to_numpy(dtype=float),
            test_base,
        )
        fitted_bundles[label] = {"bundle": bundle, "feature_columns": columns}

    test_rows = [
        metric_row("ridge_base", "base", splits["test"]["y_true"], test_base, test_base)
    ]
    for label, pred in final_predictions.items():
        family = candidate_family({
            "selected": selected,
            "best_numeric": best_by_family["numeric"],
            "best_qwen": best_by_family["qwen"],
        }[label])
        test_rows.append(metric_row(label, family, splits["test"]["y_true"], pred, test_base))
    test_comparison = pd.DataFrame(test_rows).sort_values("path_mse").reset_index(drop=True)

    comparison_runs = [resolve_path(value) for value in stack_cfg.get("comparison_runs", [])]
    subset_table = subset_comparisons(
        comparison_runs,
        splits["test"]["dates"],
        splits["test"]["y_true"],
        test_base,
        final_predictions,
    )

    validation.to_csv(output_dir / "validation_walk_forward.csv", index=False)
    test_comparison.to_csv(output_dir / "test_comparison.csv", index=False)
    subset_table.to_csv(output_dir / "test_seed_subsets.csv", index=False)
    pd.DataFrame(audit_folds).to_csv(output_dir / "validation_fold_audit.csv", index=False)
    val_frame.to_parquet(output_dir / "validation_features.parquet", index=False)
    test_frame.to_parquet(output_dir / "test_features.parquet", index=False)
    np.savez_compressed(
        output_dir / "predictions.npz",
        dates=splits["test"]["dates"].astype(str),
        origin_dates=test_origins.astype(str).to_numpy(),
        y_true=splits["test"]["y_true"],
        ridge_base=test_base,
        selected=final_predictions["selected"],
        best_numeric=final_predictions["best_numeric"],
        best_qwen=final_predictions["best_qwen"],
    )
    joblib.dump(
        {
            "version": 1,
            "base_model": base_model,
            "model_feature_columns": feature_columns,
            "price_feature_index": price_idx,
            "event_source_sha256": sha256_file(events_path),
            "event_taxonomy_version": str(stack_cfg.get("event_features", {}).get("taxonomy_version", "v2")),
            "selected": fitted_bundles["selected"],
            "best_numeric": fitted_bundles["best_numeric"],
            "best_qwen": fitted_bundles["best_qwen"],
        },
        output_dir / "stacker_bundle.joblib",
    )

    selection_payload = {
        "base_run": str(base_run.relative_to(ROOT)),
        "events_path": str(events_path.relative_to(ROOT)),
        "events_sha256": sha256_file(events_path),
        "qwen_role": "precomputed headline sentiment/importance/event features; no test-time labels",
        "selected_candidate": asdict(selected),
        "best_numeric_candidate": asdict(best_by_family["numeric"]),
        "best_qwen_candidate": asdict(best_by_family["qwen"]),
        "validation_evaluated_rows": int(len(expected_indices)),
        "validation_first_evaluated_date": str(pd.Timestamp(splits["val"]["dates"][expected_indices[0]]).date()),
        "validation_last_evaluated_date": str(pd.Timestamp(splits["val"]["dates"][expected_indices[-1]]).date()),
        "purge_size": purge_size,
        "test_was_used_for_selection": False,
        "ridge_replay_mse": replay_mse,
        "ridge_saved_mse": frozen_mse,
    }
    (output_dir / "selection.json").write_text(json.dumps(selection_payload, indent=2), encoding="utf-8")
    (output_dir / "config_snapshot.yaml").write_text(yaml.safe_dump(stack_cfg, sort_keys=False), encoding="utf-8")

    val_best = validation.iloc[0]
    selected_test = test_comparison.loc[test_comparison["variant"] == "selected"].iloc[0]
    numeric_test = test_comparison.loc[test_comparison["variant"] == "best_numeric"].iloc[0]
    qwen_test = test_comparison.loc[test_comparison["variant"] == "best_qwen"].iloc[0]
    summary = f"""# Ridge + Qwen supervised residual stacker v1

## Frozen selection

- Selected candidate: `{selected.name}` ({candidate_family(selected)})
- Best validation candidate: `{val_best['variant']}` ({val_best['family']})
- Validation protocol: {len(audit_folds)} expanding folds, {purge_size}-row purge, {len(expected_indices)} OOF predictions
- Test labels used for selection: no
- Qwen role: precomputed headline sentiment, importance, and event-taxonomy features

## Full held-out test ({len(splits['test']['y_true'])} origins)

| Variant | Path MSE | Improvement vs ridge |
|---|---:|---:|
| Ridge base | {replay_mse:.6f} | 0.000% |
| Frozen selected | {float(selected_test['path_mse']):.6f} | {float(selected_test['improvement_vs_base_pct']):+.3f}% |
| Best numeric-only | {float(numeric_test['path_mse']):.6f} | {float(numeric_test['improvement_vs_base_pct']):+.3f}% |
| Best Qwen-enriched | {float(qwen_test['path_mse']):.6f} | {float(qwen_test['improvement_vs_base_pct']):+.3f}% |

The numeric and Qwen rows use identical model families and validation rules; their difference is the incremental value of Qwen-derived event features.
"""
    (output_dir / "summary.md").write_text(summary, encoding="utf-8")

    print(json.dumps({
        "output_dir": str(output_dir),
        "selected": selected.name,
        "test": test_comparison.to_dict(orient="records"),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
