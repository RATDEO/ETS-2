#!/usr/bin/env python3
"""Run a scientific-style UK ETS trading evaluation with embargoed folds."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]

import sys

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.eval.return_metrics import (
    assess_llm_training_cutoff,
    compute_daily_mtm_portfolio_metrics,
    compute_primary_horizon_signals,
    evaluate_signal_threshold_candidates,
)
from src.run_experiment import run_experiment


METHOD_NAME = "TSM+LLM-COT-RF-HDELTA"
DEFAULT_CONFIG = "uk_ets/config/uk_ets_llm_4b_current_default.yaml"
DEFAULT_DATA_DIR = "uk_ets/Data_auto_uk"
DEFAULT_FOLD_RUNS = {
    "2023H2": "runs/20260312_021028_41ad8a",
    "2024H1": "runs/20260312_021906_a19c17",
    "2024H2": "runs/20260312_022747_2f56c6",
    "2025H1": "runs/20260312_023621_733f60",
    "2025H2_2026Q1_holdout": "runs/20260311_135144_bb26cf",
}


@dataclass(frozen=True)
class Fold:
    name: str
    train_end: str
    val_end: str
    test_end: str
    existing_run_dir: Optional[str] = None


def _default_folds(config_path: str) -> list[Fold]:
    use_archived = Path(config_path).name == Path(DEFAULT_CONFIG).name
    return [
        Fold(
            name="2023H2",
            train_end="2022-12-31",
            val_end="2023-06-30",
            test_end="2023-12-31",
            existing_run_dir=DEFAULT_FOLD_RUNS["2023H2"] if use_archived else None,
        ),
        Fold(
            name="2024H1",
            train_end="2023-06-30",
            val_end="2023-12-31",
            test_end="2024-06-30",
            existing_run_dir=DEFAULT_FOLD_RUNS["2024H1"] if use_archived else None,
        ),
        Fold(
            name="2024H2",
            train_end="2023-12-31",
            val_end="2024-06-30",
            test_end="2024-12-31",
            existing_run_dir=DEFAULT_FOLD_RUNS["2024H2"] if use_archived else None,
        ),
        Fold(
            name="2025H1",
            train_end="2024-06-30",
            val_end="2024-12-31",
            test_end="2025-06-30",
            existing_run_dir=DEFAULT_FOLD_RUNS["2025H1"] if use_archived else None,
        ),
        Fold(
            name="2025H2_2026Q1_holdout",
            train_end="2024-06-30",
            val_end="2025-06-30",
            test_end="2026-03-04",
            existing_run_dir=DEFAULT_FOLD_RUNS["2025H2_2026Q1_holdout"] if use_archived else None,
        ),
    ]


def _resolve_project_path(path_str: str) -> Path:
    path = Path(path_str).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (PROJECT_ROOT / path).resolve()


def _extract_price_paths_from_parquet(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    y_true_cols = [f"y_true_t_plus_{h}" for h in range(1, 31)]
    yhat_cols = [f"yhat_t_plus_{h}" for h in range(1, 31)]
    return (
        pd.to_datetime(df["date"]).to_numpy(),
        df[y_true_cols].to_numpy(dtype=float),
        df[yhat_cols].to_numpy(dtype=float),
    )


def _load_model_predictions(
    run_dir: Path,
    model_name: str,
    split: str = "test",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    split = str(split).lower()
    if split not in {"val", "test"}:
        raise ValueError(f"Unsupported split for prediction loading: {split}")
    if model_name == "tsm":
        pred_df = pd.read_parquet(run_dir / "predictions" / f"tsm_pred_{split}.parquet")
        return _extract_price_paths_from_parquet(pred_df)
    if model_name == METHOD_NAME:
        pred_npz = np.load(
            run_dir / "predictions" / (
                f"{METHOD_NAME}_pred_val_full.npz"
                if split == "val"
                else f"{METHOD_NAME}_pred_test_subset.npz"
            ),
            allow_pickle=True,
        )
        return (
            pd.to_datetime(pred_npz["dates"]).to_numpy(),
            np.asarray(pred_npz["y_true"], dtype=float),
            np.asarray(pred_npz["yhat"], dtype=float),
        )
    raise ValueError(f"Unsupported model for scientific evaluation: {model_name}")


def _load_base_prices(run_dir: Path, prediction_dates: np.ndarray) -> np.ndarray:
    panel = pd.read_parquet(run_dir / "data" / "panel.parquet", columns=["date", "y"])
    panel["date"] = pd.to_datetime(panel["date"])
    panel = panel.sort_values("date").reset_index(drop=True)
    date_to_pos = {ts: i for i, ts in enumerate(panel["date"])}
    values = panel["y"].to_numpy(dtype=float)

    base_prices: list[float] = []
    for date_value in pd.to_datetime(prediction_dates):
        pos = date_to_pos.get(pd.Timestamp(date_value))
        if pos is None or pos == 0:
            raise ValueError(f"Could not resolve base price before forecast date {date_value}")
        base_prices.append(float(values[pos - 1]))
    return np.asarray(base_prices, dtype=float)


def _load_panel_prices(run_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    panel = pd.read_parquet(run_dir / "data" / "panel.parquet", columns=["date", "y"])
    panel["date"] = pd.to_datetime(panel["date"])
    panel = panel.sort_values("date").reset_index(drop=True)
    return panel["date"].to_numpy(), panel["y"].to_numpy(dtype=float)


def _path_mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean((np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float)) ** 2))


def _evaluate_model(
    *,
    run_dir: Path,
    fold: Fold,
    model_name: str,
    embargo_days: int,
    primary_horizon: int,
    signal_threshold: float,
    execution_lag_days: int,
    transaction_cost_bps: float,
    slippage_bps: float,
    cutoff_date: str,
    calibrate_threshold_on_val: bool,
    threshold_quantiles: Sequence[float],
    threshold_objective: str,
    threshold_min_executed_trades: int,
    trim_to_active_window: bool,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    prediction_dates, y_true, y_pred = _load_model_predictions(run_dir, model_name, split="test")
    base_prices = _load_base_prices(run_dir, prediction_dates)
    panel_dates, panel_prices = _load_panel_prices(run_dir)

    embargo_cutoff = pd.Timestamp(fold.val_end) + pd.Timedelta(days=int(embargo_days))
    eval_mask = pd.to_datetime(prediction_dates) > embargo_cutoff
    if not np.any(eval_mask):
        raise ValueError(
            f"No test windows remain for {fold.name} after applying a {embargo_days}-day embargo."
        )

    eval_dates = pd.to_datetime(prediction_dates[eval_mask]).to_numpy()
    eval_true = np.asarray(y_true[eval_mask], dtype=float)
    eval_pred = np.asarray(y_pred[eval_mask], dtype=float)
    eval_base = np.asarray(base_prices[eval_mask], dtype=float)

    selected_threshold = float(signal_threshold)
    threshold_summary = pd.DataFrame()
    if calibrate_threshold_on_val:
        val_dates, val_true, val_pred = _load_model_predictions(run_dir, model_name, split="val")
        val_base = _load_base_prices(run_dir, val_dates)
        val_embargo_cutoff = pd.Timestamp(fold.train_end) + pd.Timedelta(days=int(embargo_days))
        val_mask = pd.to_datetime(val_dates) > val_embargo_cutoff
        if not np.any(val_mask):
            raise ValueError(
                f"No validation windows remain for {fold.name} after applying a {embargo_days}-day embargo."
            )
        val_eval_dates = pd.to_datetime(val_dates[val_mask]).to_numpy()
        val_eval_pred = np.asarray(val_pred[val_mask], dtype=float)
        val_eval_base = np.asarray(val_base[val_mask], dtype=float)
        val_abs_returns = np.abs(
            (val_eval_pred[:, primary_horizon - 1].astype(float) / np.clip(val_eval_base, 1e-8, None)) - 1.0
        )
        quantile_candidates = [0.0]
        quantile_candidates.extend(
            float(np.quantile(val_abs_returns, q))
            for q in threshold_quantiles
            if 0.0 <= float(q) <= 1.0
        )
        threshold_summary, selected_threshold = evaluate_signal_threshold_candidates(
            y_pred_prices=val_eval_pred,
            base_prices=val_eval_base,
            decision_dates=val_eval_dates,
            panel_dates=panel_dates,
            panel_prices=panel_prices,
            horizon=primary_horizon,
            threshold_candidates=quantile_candidates,
            execution_lag_days=execution_lag_days,
            hold_days=primary_horizon,
            transaction_cost_bps=transaction_cost_bps,
            slippage_bps=slippage_bps,
            max_concurrent_positions=primary_horizon,
            objective=threshold_objective,
            min_executed_trades=threshold_min_executed_trades,
            trim_to_active_window=trim_to_active_window,
        )
        threshold_summary["fold"] = fold.name
        threshold_summary["model"] = model_name
        threshold_summary["selection_split"] = "val"
        threshold_summary["selected"] = (
            threshold_summary["signal_threshold"].astype(float) == float(selected_threshold)
        )

    signals = compute_primary_horizon_signals(
        y_pred_prices=eval_pred,
        base_prices=eval_base,
        horizon=primary_horizon,
        signal_threshold=selected_threshold,
    )
    mtm_df, mtm_metrics = compute_daily_mtm_portfolio_metrics(
        close_prices=panel_prices,
        dates=panel_dates,
        decision_dates=eval_dates,
        signals=signals,
        hold_days=primary_horizon,
        execution_lag_days=execution_lag_days,
        transaction_cost_bps=transaction_cost_bps,
        slippage_bps=slippage_bps,
        max_concurrent_positions=primary_horizon,
        trim_to_active_window=trim_to_active_window,
    )
    cutoff_note = assess_llm_training_cutoff(eval_dates, cutoff_date=cutoff_date)

    row = {
        "fold": fold.name,
        "train_end": fold.train_end,
        "val_end": fold.val_end,
        "test_end": fold.test_end,
        "run_dir": str(run_dir),
        "model": model_name,
        "primary_horizon": int(primary_horizon),
        "embargo_days": int(embargo_days),
        "n_raw_test_windows": int(len(prediction_dates)),
        "n_eval_windows_after_embargo": int(len(eval_dates)),
        "path_mse_embargoed": _path_mse(eval_true, eval_pred),
        "selected_signal_threshold": float(selected_threshold),
        "threshold_objective": str(threshold_objective),
        "threshold_calibrated_on_val": bool(calibrate_threshold_on_val),
        **mtm_metrics,
        "llm_cutoff_status": cutoff_note["status"],
        "llm_cutoff_date": cutoff_note["cutoff_date"],
        "llm_n_pre_cutoff": cutoff_note["n_pre_cutoff"],
        "llm_n_post_cutoff": cutoff_note["n_post_cutoff"],
        "llm_cutoff_note": cutoff_note["note"],
    }

    mtm_out = mtm_df.copy()
    mtm_out["fold"] = fold.name
    mtm_out["model"] = model_name
    mtm_out["primary_horizon"] = int(primary_horizon)
    mtm_out["selected_signal_threshold"] = float(selected_threshold)
    return row, mtm_out, threshold_summary


def _resolve_run_dir(
    fold: Fold,
    *,
    config_path: str,
    data_dir: str,
    run_missing: bool,
    ignore_archived_runs: bool,
    export_val_predictions: bool,
) -> Path:
    if fold.existing_run_dir and not ignore_archived_runs:
        existing = (PROJECT_ROOT / fold.existing_run_dir).resolve()
        if existing.exists():
            return existing
    if not run_missing:
        raise FileNotFoundError(
            f"No archived run is available for fold {fold.name}, and --run-missing was not set."
        )

    overrides = {
        "split": {
            "train_end": fold.train_end,
            "val_end": fold.val_end,
            "test_end": fold.test_end,
        },
        "llm": {
            "export_val_predictions": bool(export_val_predictions),
        },
    }
    return Path(run_experiment(config_path=config_path, overrides=overrides, data_dir=data_dir)).resolve()


def _write_summary(
    out_dir: Path,
    results_df: pd.DataFrame,
    primary_horizon: int,
    execution_lag_days: int,
    transaction_cost_bps: float,
    slippage_bps: float,
    cutoff_date: str,
    trim_to_active_window: bool,
    calibrate_threshold_on_val: bool,
    threshold_objective: str,
) -> None:
    lines = [
        "# UK ETS Scientific Trading Evaluation",
        "",
        f"Generated: {datetime.now().isoformat()}",
        "",
        "Protocol choices:",
        f"- Primary trading horizon: `h{primary_horizon}`",
        f"- Execution lag: `{execution_lag_days}` trading day(s) after the forecast decision date",
        f"- Holding period: `{primary_horizon}` close-to-close sessions",
        f"- Capital allocation: one staggered trade slot per horizon day, capped at `{primary_horizon}` concurrent positions",
        f"- Transaction cost: `{transaction_cost_bps:.1f}` bps per side",
        f"- Slippage stress: `{slippage_bps:.1f}` additional bps per side",
        f"- Portfolio window trimmed to active trading interval: `{trim_to_active_window}`",
        f"- Signal threshold calibrated on validation only: `{calibrate_threshold_on_val}`",
        f"- Threshold selection objective: `{threshold_objective}`",
        "",
        "LLM training-cutoff note:",
        f"- Cutoff used: `{cutoff_date}`",
        "- Prediction sets whose evaluation dates lie entirely on or after this cutoff are treated as materially cleaner with respect to possible LLM pretraining contamination.",
        "- Prediction sets earlier than this remain scientifically usable for research, but the contamination question is more ambiguous.",
        "",
        "## Fold Results",
        "",
        results_df.to_markdown(index=False),
        "",
    ]

    grouped = results_df.groupby("model", sort=False)
    lines.extend(["## Model Summary", ""])
    for model_name, grp in grouped:
        lines.extend(
            [
                f"- `{model_name}`: mean embargoed path MSE `{grp['path_mse_embargoed'].mean():.6f}`, "
                f"mean total return `{grp['total_return'].mean():.2%}`, mean annualized return `{grp['annualized_return'].mean():.2%}`, "
                f"mean Sharpe `{grp['daily_sharpe_annualized'].mean():.3f}`, mean max drawdown `{grp['max_drawdown'].mean():.2%}`",
                f"  LLM cutoff statuses: {', '.join(sorted(grp['llm_cutoff_status'].astype(str).unique()))}",
            ]
        )

    lines.extend(
        [
            "",
            "Interpretation:",
            "- These results are stricter than the earlier offset-averaged horizon backtests because they use embargoed forecast windows and a daily marked-to-market portfolio construction.",
            "- They are still historical evidence. The first confirmatory economic test remains prospective paper trading after the strategy freeze.",
        ]
    )
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _run_single_existing(
    *,
    run_dir: Path,
    fold_name: str,
    train_end: str,
    val_end: str,
    test_end: str,
    output_root: Path,
    embargo_days: int,
    primary_horizon: int,
    signal_threshold: float,
    execution_lag_days: int,
    transaction_cost_bps: float,
    slippage_bps: float,
    cutoff_date: str,
    trim_to_active_window: bool,
    calibrate_threshold_on_val: bool,
    threshold_quantiles: Sequence[float],
    threshold_objective: str,
    threshold_min_executed_trades: int,
) -> Path:
    out_dir = (output_root / datetime.now().strftime("%Y%m%d_%H%M%S")).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    fold = Fold(
        name=fold_name,
        train_end=train_end,
        val_end=val_end,
        test_end=test_end,
        existing_run_dir=str(run_dir),
    )

    rows = []
    daily_rows = []
    threshold_rows = []
    for model_name in ("tsm", METHOD_NAME):
        row, daily_df, threshold_df = _evaluate_model(
            run_dir=run_dir,
            fold=fold,
            model_name=model_name,
            embargo_days=embargo_days,
            primary_horizon=primary_horizon,
            signal_threshold=signal_threshold,
            execution_lag_days=execution_lag_days,
            transaction_cost_bps=transaction_cost_bps,
            slippage_bps=slippage_bps,
            cutoff_date=cutoff_date,
            calibrate_threshold_on_val=calibrate_threshold_on_val,
            threshold_quantiles=threshold_quantiles,
            threshold_objective=threshold_objective,
            threshold_min_executed_trades=threshold_min_executed_trades,
            trim_to_active_window=trim_to_active_window,
        )
        rows.append(row)
        daily_rows.append(daily_df)
        if not threshold_df.empty:
            threshold_rows.append(threshold_df)

    results_df = pd.DataFrame(rows)
    daily_df = pd.concat(daily_rows, ignore_index=True)
    results_df.to_csv(out_dir / "fold_results.csv", index=False)
    daily_df.to_csv(out_dir / "daily_mtm_returns.csv", index=False)
    if threshold_rows:
        pd.concat(threshold_rows, ignore_index=True).to_csv(
            out_dir / "threshold_selection.csv",
            index=False,
        )
    _write_summary(
        out_dir=out_dir,
        results_df=results_df,
        primary_horizon=primary_horizon,
        execution_lag_days=execution_lag_days,
        transaction_cost_bps=transaction_cost_bps,
        slippage_bps=slippage_bps,
        cutoff_date=cutoff_date,
        trim_to_active_window=trim_to_active_window,
        calibrate_threshold_on_val=calibrate_threshold_on_val,
        threshold_objective=threshold_objective,
    )
    return out_dir


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a scientific-style UK ETS trading evaluation.")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-root", default="reports/uk_ets_scientific_trading_eval")
    parser.add_argument("--run-missing", action="store_true", help="Run missing folds instead of requiring archived runs.")
    parser.add_argument(
        "--ignore-archived-runs",
        action="store_true",
        help="Ignore any archived run hints and regenerate folds under the current frozen config.",
    )
    parser.add_argument("--primary-horizon", type=int, default=20)
    parser.add_argument("--signal-threshold", type=float, default=0.0)
    parser.add_argument("--embargo-days", type=int, default=30)
    parser.add_argument("--execution-lag-days", type=int, default=1)
    parser.add_argument("--transaction-cost-bps", type=float, default=10.0)
    parser.add_argument("--slippage-bps", type=float, default=5.0)
    parser.add_argument("--cutoff-date", default="2025-01-01")
    parser.add_argument("--trim-to-active-window", action="store_true")
    parser.add_argument("--calibrate-threshold-on-val", action="store_true")
    parser.add_argument("--threshold-objective", default="daily_sharpe_annualized")
    parser.add_argument("--threshold-min-executed-trades", type=int, default=10)
    parser.add_argument("--threshold-quantiles", default="0.0,0.25,0.5,0.6,0.7,0.8,0.9")
    parser.add_argument("--max-folds", type=int, default=None)
    parser.add_argument("--evaluate-run-dir", default=None, help="Evaluate one existing run instead of the full fold set.")
    parser.add_argument("--evaluate-fold-name", default="existing_run")
    parser.add_argument("--evaluate-train-end", default="2024-06-30")
    parser.add_argument("--evaluate-val-end", default="2025-06-30")
    parser.add_argument("--evaluate-test-end", default="2026-03-04")
    args = parser.parse_args()

    output_root = (PROJECT_ROOT / args.output_root).resolve()

    if args.evaluate_run_dir:
        out_dir = _run_single_existing(
            run_dir=_resolve_project_path(args.evaluate_run_dir),
            fold_name=args.evaluate_fold_name,
            train_end=args.evaluate_train_end,
            val_end=args.evaluate_val_end,
            test_end=args.evaluate_test_end,
            output_root=output_root,
            embargo_days=args.embargo_days,
            primary_horizon=args.primary_horizon,
            signal_threshold=args.signal_threshold,
            execution_lag_days=args.execution_lag_days,
            transaction_cost_bps=args.transaction_cost_bps,
            slippage_bps=args.slippage_bps,
            cutoff_date=args.cutoff_date,
            trim_to_active_window=bool(args.trim_to_active_window),
            calibrate_threshold_on_val=bool(args.calibrate_threshold_on_val),
            threshold_quantiles=[
                float(x) for x in str(args.threshold_quantiles).split(",") if str(x).strip()
            ],
            threshold_objective=str(args.threshold_objective),
            threshold_min_executed_trades=int(args.threshold_min_executed_trades),
        )
        print(out_dir)
        return 0

    folds = _default_folds(args.config)
    if args.max_folds is not None:
        folds = folds[: int(args.max_folds)]

    out_dir = (output_root / datetime.now().strftime("%Y%m%d_%H%M%S")).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    daily_rows = []
    threshold_rows = []
    threshold_quantiles = [
        float(x) for x in str(args.threshold_quantiles).split(",") if str(x).strip()
    ]
    for fold in folds:
        run_dir = _resolve_run_dir(
            fold,
            config_path=str(_resolve_project_path(args.config)),
            data_dir=str(_resolve_project_path(args.data_dir)),
            run_missing=bool(args.run_missing),
            ignore_archived_runs=bool(args.ignore_archived_runs),
            export_val_predictions=bool(args.calibrate_threshold_on_val),
        )
        for model_name in ("tsm", METHOD_NAME):
            row, daily_df, threshold_df = _evaluate_model(
                run_dir=run_dir,
                fold=fold,
                model_name=model_name,
                embargo_days=args.embargo_days,
                primary_horizon=args.primary_horizon,
                signal_threshold=args.signal_threshold,
                execution_lag_days=args.execution_lag_days,
                transaction_cost_bps=args.transaction_cost_bps,
                slippage_bps=args.slippage_bps,
                cutoff_date=args.cutoff_date,
                calibrate_threshold_on_val=bool(args.calibrate_threshold_on_val),
                threshold_quantiles=threshold_quantiles,
                threshold_objective=str(args.threshold_objective),
                threshold_min_executed_trades=int(args.threshold_min_executed_trades),
                trim_to_active_window=bool(args.trim_to_active_window),
            )
            rows.append(row)
            daily_rows.append(daily_df)
            if not threshold_df.empty:
                threshold_rows.append(threshold_df)

    results_df = pd.DataFrame(rows)
    daily_df = pd.concat(daily_rows, ignore_index=True) if daily_rows else pd.DataFrame()
    results_df.to_csv(out_dir / "fold_results.csv", index=False)
    daily_df.to_csv(out_dir / "daily_mtm_returns.csv", index=False)
    if threshold_rows:
        pd.concat(threshold_rows, ignore_index=True).to_csv(
            out_dir / "threshold_selection.csv",
            index=False,
        )
    _write_summary(
        out_dir=out_dir,
        results_df=results_df,
        primary_horizon=args.primary_horizon,
        execution_lag_days=args.execution_lag_days,
        transaction_cost_bps=args.transaction_cost_bps,
        slippage_bps=args.slippage_bps,
        cutoff_date=args.cutoff_date,
        trim_to_active_window=bool(args.trim_to_active_window),
        calibrate_threshold_on_val=bool(args.calibrate_threshold_on_val),
        threshold_objective=str(args.threshold_objective),
    )
    print(out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
