#!/usr/bin/env python3
"""
Run stricter outer rolling validation for the simple meta-blend, then refit on full
validation and score the held-out test set.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from eval.meta_blend import MetaBlender, rolling_meta_blend_cv
from eval.metrics import compute_metrics_by_horizon, compute_path_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run strict outer rolling validation for the simple TSM+LLM meta-blend."
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Run directory containing stage2_full_val and stage3_test prediction npz files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: <run-dir>/meta_blend_strict_eval).",
    )
    parser.add_argument(
        "--outer-folds",
        type=int,
        default=4,
        help="Number of outer rolling folds across validation.",
    )
    parser.add_argument(
        "--inner-folds",
        type=int,
        default=4,
        help="Number of inner blocked folds for alpha selection.",
    )
    return parser.parse_args()


def _metric_row(name: str, y_true: np.ndarray, yhat: np.ndarray) -> dict[str, float | str]:
    metrics = compute_metrics_by_horizon(y_true, yhat, [1, 5, 20, 30]).reset_index()
    row: dict[str, float | str] = {"name": name, "path_mse": float(compute_path_metrics(y_true, yhat)["mse_path"])}
    for _, metric_row in metrics.iterrows():
        row[f"h{int(metric_row['horizon'])}_mse"] = float(metric_row["mse"])
    return row


def main() -> int:
    args = parse_args()
    run_dir = args.run_dir if args.run_dir.is_absolute() else (ROOT / args.run_dir)
    output_dir = args.output_dir if args.output_dir is not None else (run_dir / "meta_blend_strict_eval")
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    val_npz = np.load(
        run_dir / "stage2_full_val" / "baseline_daily_sentiment" / "val" / "val_predictions.npz",
        allow_pickle=True,
    )
    test_npz = np.load(
        run_dir / "stage3_test" / "baseline_daily_sentiment" / "test" / "test_predictions.npz",
        allow_pickle=True,
    )

    val_true = np.asarray(val_npz["y_true"], dtype=float)
    val_tsm = np.asarray(val_npz["tsm_pred"], dtype=float)
    val_llm = np.asarray(val_npz["yhat"], dtype=float)
    val_dates = np.asarray(val_npz["dates"])

    test_true = np.asarray(test_npz["y_true"], dtype=float)
    test_tsm = np.asarray(test_npz["tsm_pred"], dtype=float)
    test_llm = np.asarray(test_npz["yhat"], dtype=float)

    cv_pred, fold_results = rolling_meta_blend_cv(
        val_tsm,
        val_llm,
        val_true,
        n_outer_folds=args.outer_folds,
        n_inner_folds=args.inner_folds,
    )
    eval_mask = np.isfinite(cv_pred).all(axis=1)
    cv_rows = [
        _metric_row("tsm", val_true[eval_mask], val_tsm[eval_mask]),
        _metric_row("llm_hdelta", val_true[eval_mask], val_llm[eval_mask]),
        _metric_row("meta_blend_outer_cv", val_true[eval_mask], cv_pred[eval_mask]),
    ]
    pd.DataFrame(cv_rows).to_csv(output_dir / "outer_cv_comparison.csv", index=False)
    pd.DataFrame([result.__dict__ for result in fold_results]).to_csv(output_dir / "outer_fold_results.csv", index=False)
    np.savez_compressed(
        output_dir / "outer_cv_predictions.npz",
        dates=val_dates[eval_mask],
        y_true=val_true[eval_mask],
        tsm_pred=val_tsm[eval_mask],
        llm_pred=val_llm[eval_mask],
        blend_pred=cv_pred[eval_mask],
    )

    blender = MetaBlender(n_folds=args.inner_folds)
    blender.fit(val_tsm, val_llm, val_true)
    test_blend = blender.predict(test_tsm, test_llm)

    test_rows = [
        _metric_row("tsm", test_true, test_tsm),
        _metric_row("llm_hdelta", test_true, test_llm),
        _metric_row("meta_blend_fullval_fit", test_true, test_blend),
    ]
    pd.DataFrame(test_rows).to_csv(output_dir / "test_comparison.csv", index=False)
    pd.DataFrame(blender.summary_rows()).to_csv(output_dir / "blend_coefficients.csv", index=False)
    np.savez_compressed(
        output_dir / "test_predictions.npz",
        dates=np.asarray(test_npz["dates"]),
        y_true=test_true,
        tsm_pred=test_tsm,
        llm_pred=test_llm,
        blend_pred=test_blend,
    )

    summary = {
        "run_dir": str(run_dir),
        "outer_cv_eval_samples": int(eval_mask.sum()),
        "outer_cv_results": {row["name"]: row for row in cv_rows},
        "test_results": {row["name"]: row for row in test_rows},
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
