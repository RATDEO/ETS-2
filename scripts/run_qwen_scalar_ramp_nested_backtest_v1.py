#!/usr/bin/env python3
"""Nested multi-regime backtest for the Qwen scalar-ramp refinement stage."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from scipy.stats import ttest_rel, wilcoxon

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from data.panel import select_feature_columns
from data.windows import WindowConfig, make_windows
from eval.residual_event_model import (
    block_columns,
    build_origin_feature_frame,
    extract_optimal_ramp_targets,
    fit_scalar_ramp_candidate,
    predict_scalar_ramp_candidate,
)
from models.baselines import LinearBaseline
from news.event_panel import build_event_panel
from run_experiment import returns_to_prices

from run_qwen_scalar_ramp_stacker_v1 import (
    candidate_columns,
    candidate_family,
    make_candidates,
    metric_row,
    select_candidate,
)
from run_ridge_qwen_residual_stacker_v1 import fold_statistics, load_yaml, resolve_path, sha256_file


DEFAULT_CONFIG = ROOT / "uk_ets" / "config" / "uk_ets_qwen_scalar_ramp_nested_backtest_v1.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def load_full_windows(
    base_run: Path,
) -> tuple[pd.DataFrame, dict[str, Any], list[str], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    cfg = load_yaml(base_run / "config_resolved.yaml")
    panel = pd.read_parquet(base_run / "data" / "panel.parquet")
    features_cfg = cfg.get("features", {}) or {}
    feature_columns = select_feature_columns(
        panel,
        target_col="y_return",
        max_exogenous_features=int(features_cfg.get("max_exogenous_features_model", 10)),
        preferred_feature_order=features_cfg.get("preferred_feature_order"),
    )
    time_cfg = cfg.get("time_series", {}) or {}
    window_cfg = WindowConfig(
        seq_len=int(time_cfg.get("seq_len", 20)),
        label_len=int(time_cfg.get("label_len", 10)),
        pred_len=int(time_cfg.get("pred_len", 30)),
        target_col="y_return",
        feature_cols=feature_columns,
    )
    X_enc, _, future_returns, dates, metadata = make_windows(
        panel, window_cfg, mode="MS", return_metadata=True
    )
    price_idx = feature_columns.index("y")
    histories = np.asarray(X_enc[:, :, price_idx], dtype=float)
    y_true = returns_to_prices(np.asarray(future_returns, dtype=float), histories[:, -1])
    return (
        panel,
        cfg,
        feature_columns,
        histories,
        np.asarray(y_true, dtype=float),
        pd.to_datetime(dates).to_numpy(),
        pd.to_datetime(metadata.last_input_dates).to_numpy(),
    )


def expanding_base_oof(
    histories: np.ndarray,
    y_true: np.ndarray,
    first_apply: int,
    fold_size: int,
    purge_size: int,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    index_parts: list[np.ndarray] = []
    prediction_parts: list[np.ndarray] = []
    audit_rows: list[dict[str, int]] = []
    for apply_start in range(first_apply, len(histories), fold_size):
        apply_end = min(apply_start + fold_size, len(histories))
        fit_end = apply_start - purge_size
        model = LinearBaseline(pred_len=y_true.shape[1], model_type="ridge")
        model.fit(histories[:fit_end], y_true[:fit_end])
        apply_indices = np.arange(apply_start, apply_end, dtype=int)
        prediction_parts.append(model.predict(histories[apply_indices]))
        index_parts.append(apply_indices)
        audit_rows.append(
            {
                "fit_start": 0,
                "fit_end": fit_end,
                "apply_start": apply_start,
                "apply_end": apply_end,
                "purge_size": purge_size,
            }
        )
    return np.concatenate(index_parts), np.concatenate(prediction_parts), pd.DataFrame(audit_rows)


def validate_outer_folds(
    starts: list[int],
    n_samples: int,
    test_size: int,
    purge_size: int,
) -> None:
    if starts != sorted(set(starts)):
        raise ValueError("outer_starts must be unique and sorted.")
    for fold_idx, start in enumerate(starts):
        if start - purge_size < 1 or start + test_size > n_samples:
            raise ValueError(f"Outer fold {fold_idx} is outside the available windows.")
        if fold_idx and start < starts[fold_idx - 1] + test_size + purge_size:
            raise ValueError("Outer test folds must be separated by at least the purge size.")


def main() -> int:
    args = parse_args()
    config_path = resolve_path(args.config)
    cfg = load_yaml(config_path)
    base_run = resolve_path(cfg["base_run"])
    events_path = resolve_path(cfg["events_path"])
    output_dir = (
        ROOT / "reports" / "qwen_scalar_ramp_nested_backtest_v1" / datetime.now().strftime("%Y%m%d_%H%M%S")
        if args.output_dir is None
        else resolve_path(args.output_dir)
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    panel, base_cfg, feature_columns, histories, y_true, dates, origin_dates = load_full_windows(base_run)
    pred_len = y_true.shape[1]
    if str((base_cfg.get("target", {}) or {}).get("mode")) != "returns":
        raise ValueError("Nested stacker backtest requires target.mode=returns.")

    oof_cfg = cfg.get("base_oof", {}) or {}
    purge_size = int(oof_cfg.get("purge_size", pred_len))
    oof_indices, oof_base, oof_audit = expanding_base_oof(
        histories,
        y_true,
        first_apply=int(oof_cfg.get("initial_apply_start", 250)),
        fold_size=int(oof_cfg.get("fold_size", 30)),
        purge_size=purge_size,
    )
    oof_audit.to_csv(output_dir / "base_oof_fold_audit.csv", index=False)

    event_cfg = cfg.get("event_features", {}) or {}
    event_panel = build_event_panel(
        events_path,
        calendar=panel["date"],
        rolling_windows=event_cfg.get("rolling_windows", [3, 7]),
        taxonomy_version=str(event_cfg.get("taxonomy_version", "v2")),
    )
    oof_frame = build_origin_feature_frame(
        panel,
        event_panel,
        origin_dates[oof_indices],
        oof_base,
    )
    blocks = block_columns(oof_frame)
    if not blocks["event_core"]:
        raise RuntimeError("No Qwen event-core features were built.")

    candidate_cfg = cfg.get("candidates", {}) or {}
    cap = float(candidate_cfg.get("max_adjustment_pct", 5.0))
    power = float(candidate_cfg.get("ramp_power", 1.0))
    oof_targets = extract_optimal_ramp_targets(
        y_true[oof_indices], oof_base, max_adjustment_pct=cap, ramp_power=power
    )
    candidates = make_candidates(cfg)
    candidate_map = {candidate.name: candidate for candidate in candidates}

    outer_cfg = cfg.get("outer_folds", {}) or {}
    outer_starts = [int(value) for value in outer_cfg.get("starts", [])]
    outer_test_size = int(outer_cfg.get("test_size", 60))
    validation_size = int(outer_cfg.get("validation_size", 120))
    validation_blocks = int(outer_cfg.get("validation_blocks", 1))
    level_one_purge = int(outer_cfg.get("level_one_purge_size", pred_len))
    if validation_blocks < 1:
        raise ValueError("validation_blocks must be positive.")
    validate_outer_folds(outer_starts, len(histories), outer_test_size, purge_size)

    validation_rows: list[dict[str, Any]] = []
    validation_block_rows: list[dict[str, Any]] = []
    selection_rows: list[dict[str, Any]] = []
    fold_metric_rows: list[dict[str, Any]] = []
    aggregate_true: list[np.ndarray] = []
    aggregate_base: list[np.ndarray] = []
    aggregate_dates: list[np.ndarray] = []
    aggregate_indices: list[np.ndarray] = []
    aggregate_predictions: dict[str, list[np.ndarray]] = {
        "selected": [],
        "best_base_only": [],
        "best_numeric": [],
        "best_qwen": [],
    }

    for fold_number, test_start in enumerate(outer_starts, start=1):
        test_end = test_start + outer_test_size
        known_index_end = test_start - purge_size
        known_positions = np.flatnonzero(oof_indices < known_index_end)
        total_validation_rows = validation_size * validation_blocks
        if len(known_positions) <= total_validation_rows + level_one_purge:
            raise ValueError(f"Outer fold {fold_number} has insufficient OOF history.")
        validation_position_blocks = [
            known_positions[
                len(known_positions) - total_validation_rows + block_idx * validation_size:
                len(known_positions) - total_validation_rows + (block_idx + 1) * validation_size
            ]
            for block_idx in range(validation_blocks)
        ]
        recent_validation_positions = validation_position_blocks[-1]
        recent_validation_first_index = int(oof_indices[recent_validation_positions[0]])
        recent_training_positions = np.flatnonzero(
            oof_indices < recent_validation_first_index - level_one_purge
        )

        base_model = LinearBaseline(pred_len=pred_len, model_type="ridge")
        base_model.fit(histories[:known_index_end], y_true[:known_index_end])
        test_indices = np.arange(test_start, test_end, dtype=int)
        test_base = base_model.predict(histories[test_indices])
        test_frame = build_origin_feature_frame(
            panel,
            event_panel,
            origin_dates[test_indices],
            test_base,
        )

        fold_validation: list[dict[str, Any]] = []
        score_fold_size = int((cfg.get("selection", {}) or {}).get("score_fold_size", 30))
        for candidate in candidates:
            columns = candidate_columns(candidate, blocks)
            true_parts: list[np.ndarray] = []
            base_parts: list[np.ndarray] = []
            prediction_parts: list[np.ndarray] = []
            block_improvements: list[float] = []
            wins = 0
            n_score_folds = 0
            fold_improvements: list[float] = []
            for validation_block, validation_positions in enumerate(validation_position_blocks, start=1):
                validation_first_index = int(oof_indices[validation_positions[0]])
                training_positions = np.flatnonzero(
                    oof_indices < validation_first_index - level_one_purge
                )
                if len(training_positions) < 1:
                    raise ValueError(
                        f"Outer fold {fold_number}, validation block {validation_block} has no training rows."
                    )
                bundle = fit_scalar_ramp_candidate(
                    candidate,
                    oof_frame.iloc[training_positions][columns].to_numpy(dtype=float),
                    oof_targets[training_positions],
                    pred_len=pred_len,
                )
                validation_prediction = predict_scalar_ramp_candidate(
                    bundle,
                    oof_frame.iloc[validation_positions][columns].to_numpy(dtype=float),
                    oof_base[validation_positions],
                )
                block_row = metric_row(
                    candidate.name,
                    candidate_family(candidate),
                    y_true[oof_indices[validation_positions]],
                    validation_prediction,
                    oof_base[validation_positions],
                )
                block_wins, block_n_folds, block_median = fold_statistics(
                    y_true[oof_indices[validation_positions]],
                    oof_base[validation_positions],
                    validation_prediction,
                    score_fold_size,
                )
                block_row.update(
                    {
                        "outer_fold": fold_number,
                        "validation_block": validation_block,
                        "training_rows": len(training_positions),
                        "fold_wins": block_wins,
                        "n_folds": block_n_folds,
                        "fold_win_share": block_wins / block_n_folds,
                        "median_fold_improvement_pct": block_median,
                        "n_features": len(columns),
                    }
                )
                validation_block_rows.append(block_row)
                true_parts.append(y_true[oof_indices[validation_positions]])
                base_parts.append(oof_base[validation_positions])
                prediction_parts.append(validation_prediction)
                block_improvements.append(float(block_row["improvement_vs_base_pct"]))
                wins += block_wins
                n_score_folds += block_n_folds
                fold_improvements.append(block_median)

            row = metric_row(
                candidate.name,
                candidate_family(candidate),
                np.concatenate(true_parts),
                np.concatenate(prediction_parts),
                np.concatenate(base_parts),
            )
            row.update(
                {
                    "outer_fold": fold_number,
                    "fold_wins": wins,
                    "n_folds": n_score_folds,
                    "fold_win_share": wins / n_score_folds,
                    "median_fold_improvement_pct": float(np.median(fold_improvements)),
                    "min_validation_block_improvement_pct": float(min(block_improvements)),
                    "max_validation_block_improvement_pct": float(max(block_improvements)),
                    "n_features": len(columns),
                }
            )
            fold_validation.append(row)
            validation_rows.append(row)

        validation_table = pd.DataFrame(fold_validation).sort_values("path_mse").reset_index(drop=True)
        selection_input = validation_table.copy()
        selection_cfg = cfg.get("selection", {}) or {}
        if bool(selection_cfg.get("require_all_validation_blocks_improve", False)):
            min_block_improvement = float(selection_cfg.get("min_block_improvement_pct", 0.0))
            unstable = selection_input["min_validation_block_improvement_pct"] < min_block_improvement
            selection_input.loc[unstable, "improvement_vs_base_pct"] = -np.inf
        selected = select_candidate(selection_input, candidate_map, cfg)
        best_by_family = {
            family: candidate_map[
                str(validation_table.loc[validation_table["family"] == family].iloc[0]["variant"])
            ]
            for family in ("base_only", "numeric", "qwen")
        }
        final_candidates = {
            "selected": selected,
            "best_base_only": best_by_family["base_only"],
            "best_numeric": best_by_family["numeric"],
            "best_qwen": best_by_family["qwen"],
        }

        fold_predictions: dict[str, np.ndarray] = {}
        for label, candidate in final_candidates.items():
            columns = candidate_columns(candidate, blocks)
            final_bundle = fit_scalar_ramp_candidate(
                candidate,
                oof_frame.iloc[known_positions][columns].to_numpy(dtype=float),
                oof_targets[known_positions],
                pred_len=pred_len,
            )
            fold_predictions[label] = predict_scalar_ramp_candidate(
                final_bundle,
                test_frame[columns].to_numpy(dtype=float),
                test_base,
            )

        base_fold_mse = float(np.mean((y_true[test_indices] - test_base) ** 2))
        fold_metric_rows.append(
            {
                **metric_row("ridge_base", "base", y_true[test_indices], test_base, test_base),
                "outer_fold": fold_number,
                "test_start": str(pd.Timestamp(dates[test_start]).date()),
                "test_end": str(pd.Timestamp(dates[test_end - 1]).date()),
                "selected_candidate": selected.name,
            }
        )
        for label, prediction in fold_predictions.items():
            row = metric_row(
                label,
                candidate_family(final_candidates[label]),
                y_true[test_indices],
                prediction,
                test_base,
            )
            row.update(
                {
                    "outer_fold": fold_number,
                    "test_start": str(pd.Timestamp(dates[test_start]).date()),
                    "test_end": str(pd.Timestamp(dates[test_end - 1]).date()),
                    "selected_candidate": selected.name,
                }
            )
            fold_metric_rows.append(row)

        selected_validation = validation_table.loc[validation_table["variant"] == selected.name].iloc[0]
        selected_test_mse = float(np.mean((y_true[test_indices] - fold_predictions["selected"]) ** 2))
        selection_rows.append(
            {
                "outer_fold": fold_number,
                "test_start_index": test_start,
                "test_end_index": test_end,
                "test_start": str(pd.Timestamp(dates[test_start]).date()),
                "test_end": str(pd.Timestamp(dates[test_end - 1]).date()),
                "base_fit_end_exclusive": known_index_end,
                "level_one_train_rows": len(recent_training_positions),
                "validation_rows": total_validation_rows,
                "validation_blocks": validation_blocks,
                "known_oof_rows_at_test": len(known_positions),
                "selected_candidate": selected.name,
                "selected_family": candidate_family(selected),
                "validation_improvement_pct": float(selected_validation["improvement_vs_base_pct"]),
                "test_improvement_pct": 100.0 * (base_fold_mse - selected_test_mse) / base_fold_mse,
            }
        )
        aggregate_true.append(y_true[test_indices])
        aggregate_base.append(test_base)
        aggregate_dates.append(dates[test_indices])
        aggregate_indices.append(test_indices)
        for label, prediction in fold_predictions.items():
            aggregate_predictions[label].append(prediction)

    y_aggregate = np.concatenate(aggregate_true)
    base_aggregate = np.concatenate(aggregate_base)
    dates_aggregate = np.concatenate(aggregate_dates)
    indices_aggregate = np.concatenate(aggregate_indices)
    predictions_aggregate = {
        label: np.concatenate(parts) for label, parts in aggregate_predictions.items()
    }
    aggregate_rows = [metric_row("ridge_base", "base", y_aggregate, base_aggregate, base_aggregate)]
    for label, prediction in predictions_aggregate.items():
        families = (
            "adaptive"
            if label == "selected"
            else {"best_base_only": "base_only", "best_numeric": "numeric", "best_qwen": "qwen"}[label]
        )
        aggregate_rows.append(metric_row(label, families, y_aggregate, prediction, base_aggregate))
    aggregate_metrics = pd.DataFrame(aggregate_rows).sort_values("path_mse").reset_index(drop=True)

    base_errors = np.mean((y_aggregate - base_aggregate) ** 2, axis=1)
    selected_errors = np.mean((y_aggregate - predictions_aggregate["selected"]) ** 2, axis=1)
    t_result = ttest_rel(base_errors, selected_errors)
    try:
        w_result = wilcoxon(base_errors, selected_errors)
        w_stat, w_p = float(w_result.statistic), float(w_result.pvalue)
    except ValueError:
        w_stat, w_p = float("nan"), float("nan")
    significance = pd.DataFrame(
        [
            {"test": "paired_t", "statistic": float(t_result.statistic), "p_value": float(t_result.pvalue)},
            {"test": "wilcoxon", "statistic": w_stat, "p_value": w_p},
        ]
    )

    validation_table_all = pd.DataFrame(validation_rows)
    validation_blocks_table = pd.DataFrame(validation_block_rows)
    selection_table = pd.DataFrame(selection_rows)
    fold_metrics = pd.DataFrame(fold_metric_rows)
    validation_table_all.to_csv(output_dir / "nested_validation_candidates.csv", index=False)
    validation_blocks_table.to_csv(output_dir / "nested_validation_blocks.csv", index=False)
    selection_table.to_csv(output_dir / "outer_fold_selections.csv", index=False)
    fold_metrics.to_csv(output_dir / "outer_fold_metrics.csv", index=False)
    aggregate_metrics.to_csv(output_dir / "aggregate_metrics.csv", index=False)
    significance.to_csv(output_dir / "significance_tests.csv", index=False)
    np.savez_compressed(
        output_dir / "predictions.npz",
        window_indices=indices_aggregate,
        dates=dates_aggregate.astype(str),
        y_true=y_aggregate,
        ridge_base=base_aggregate,
        **predictions_aggregate,
    )

    audit_payload = {
        "base_run": str(base_run.relative_to(ROOT)),
        "event_source": str(events_path.relative_to(ROOT)),
        "event_source_sha256": sha256_file(events_path),
        "qwen_role": "frozen headline sentiment, importance, and event-taxonomy features",
        "new_qwen_calls_made": False,
        "outer_folds": len(outer_starts),
        "outer_test_rows": int(len(y_aggregate)),
        "outer_starts": outer_starts,
        "outer_test_size": outer_test_size,
        "base_purge_size": purge_size,
        "level_one_validation_size": validation_size,
        "level_one_validation_blocks": validation_blocks,
        "level_one_purge_size": level_one_purge,
        "test_labels_used_for_fold_selection": False,
        "candidates": [asdict(candidate) for candidate in candidates],
    }
    (output_dir / "audit.json").write_text(json.dumps(audit_payload, indent=2), encoding="utf-8")
    (output_dir / "config_snapshot.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

    selected_aggregate = aggregate_metrics.loc[aggregate_metrics["variant"] == "selected"].iloc[0]
    qwen_aggregate = aggregate_metrics.loc[aggregate_metrics["variant"] == "best_qwen"].iloc[0]
    numeric_aggregate = aggregate_metrics.loc[aggregate_metrics["variant"] == "best_numeric"].iloc[0]
    qwen_selected_count = int((selection_table["selected_family"] == "qwen").sum())
    improving_fold_count = int((selection_table["test_improvement_pct"] > 0.0).sum())
    fold_lines = "\n".join(
        f"| {int(row.outer_fold)} | {row.test_start} → {row.test_end} | {row.selected_family} | "
        f"{row.validation_improvement_pct:+.2f}% | {row.test_improvement_pct:+.2f}% |"
        for row in selection_table.itertuples(index=False)
    )
    summary = f"""# Qwen scalar-ramp nested backtest v1

## Aggregate result

- Outer test origins: {len(y_aggregate)} across {len(outer_starts)} non-overlapping market periods
- Adaptive selected refinement: {float(selected_aggregate['path_mse']):.6f} MSE ({float(selected_aggregate['improvement_vs_base_pct']):+.3f}% vs ridge)
- Best-Qwen-per-fold comparator: {float(qwen_aggregate['path_mse']):.6f} MSE ({float(qwen_aggregate['improvement_vs_base_pct']):+.3f}%)
- Best-numeric-per-fold comparator: {float(numeric_aggregate['path_mse']):.6f} MSE ({float(numeric_aggregate['improvement_vs_base_pct']):+.3f}%)
- Qwen selected by the validation policy: {qwen_selected_count}/{len(outer_starts)} folds
- Selected policy improved the next outer period: {improving_fold_count}/{len(outer_starts)} folds
- Paired p-values: t-test {float(t_result.pvalue):.6f}; Wilcoxon {w_p:.6f}

## Fold stability

| Fold | Test period | Selected family | Validation | Next-period test |
|---:|---|---|---:|---:|
{fold_lines}

## Leakage controls

Every outer ridge fit excludes the 30 preceding overlapping target windows. Level-one candidate training is separately purged by 30 windows before each validation block. Candidate family, lookback, regularization, and shrinkage are chosen without outer test outcomes. Outer test blocks are separated by 30-window resolution gaps.

This is a retrospective nested backtest, not a prospective live result. It is more robust than the single terminal split because every reported test block is downstream of an independently repeated selection step.
"""
    (output_dir / "summary.md").write_text(summary, encoding="utf-8")

    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "aggregate": aggregate_metrics.to_dict(orient="records"),
                "selections": selection_table.to_dict(orient="records"),
                "significance": significance.to_dict(orient="records"),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
