#!/usr/bin/env python3
"""Run rolling-origin evaluation for the UK 4B forecast winner."""

from __future__ import annotations

import argparse
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

from src.eval.return_metrics import compute_long_short_portfolio_metrics
from src.run_experiment import run_experiment


METHOD_NAME = "TSM+LLM-COT-RF-HDELTA"
HORIZONS = [1, 5, 20, 30]


@dataclass(frozen=True)
class Fold:
    name: str
    train_end: str
    val_end: str
    test_end: str
    existing_run_dir: str | None = None


def _extract_price_paths_from_parquet(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    y_true_cols = [f"y_true_t_plus_{h}" for h in range(1, 31)]
    yhat_cols = [f"yhat_t_plus_{h}" for h in range(1, 31)]
    return (
        pd.to_datetime(df["date"]).to_numpy(),
        df[y_true_cols].to_numpy(dtype=float),
        df[yhat_cols].to_numpy(dtype=float),
    )


def _load_base_prices(run_dir: Path, dates: np.ndarray) -> np.ndarray:
    panel = pd.read_parquet(run_dir / "data" / "panel.parquet", columns=["date", "y"])
    panel["date"] = pd.to_datetime(panel["date"])
    panel = panel.sort_values("date").reset_index(drop=True)
    date_to_pos = {ts: i for i, ts in enumerate(panel["date"])}
    values = panel["y"].to_numpy(dtype=float)

    base_prices: list[float] = []
    for date_value in pd.to_datetime(dates):
        pos = date_to_pos.get(pd.Timestamp(date_value))
        if pos is None or pos == 0:
            raise ValueError(f"Could not resolve base price before forecast date {date_value}")
        base_prices.append(float(values[pos - 1]))
    return np.asarray(base_prices, dtype=float)


def _load_path_metric(run_dir: Path, model_name: str) -> tuple[float, int]:
    df = pd.read_csv(run_dir / "results" / "path_metrics.csv")
    row = df[df["model"] == model_name].iloc[0]
    return float(row["mse_path"]), int(row["n_samples"])


def _load_model_horizon_rows(run_dir: Path, model_name: str) -> pd.DataFrame:
    df = pd.read_csv(run_dir / "results" / "long_short_metrics.csv")
    subset = df[df["model"] == model_name].copy()
    if not subset.empty and "net_total_return_offset_avg" in subset.columns:
        return subset
    # Historical archived runs may predate the upgraded financial columns.
    if model_name != METHOD_NAME and model_name != "tsm":
        return subset
    if model_name == "tsm":
        pred_df = pd.read_parquet(run_dir / "predictions" / "tsm_pred_test.parquet")
        dates, y_true, y_pred = _extract_price_paths_from_parquet(pred_df)
    else:
        pred_npz = np.load(
            run_dir / "predictions" / f"{METHOD_NAME}_pred_test_subset.npz",
            allow_pickle=True,
        )
        dates = pred_npz["dates"]
        y_true = pred_npz["y_true"]
        y_pred = pred_npz["yhat"]
    base_prices = _load_base_prices(run_dir, dates)
    recomputed = compute_long_short_portfolio_metrics(
        y_true_prices=y_true,
        y_pred_prices=y_pred,
        base_prices=base_prices,
        horizons=HORIZONS,
        signal_threshold=0.0,
        average_non_overlap_offsets=True,
        transaction_cost_bps=10.0,
    ).reset_index()
    recomputed["model"] = model_name
    return recomputed


def _collect_fold_result(fold: Fold, run_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for model_name in ("tsm", METHOD_NAME):
        path_mse, n_samples = _load_path_metric(run_dir, model_name)
        horizon_df = _load_model_horizon_rows(run_dir, model_name)
        for horizon in (20, 30):
            row = horizon_df[horizon_df["horizon"] == horizon].iloc[0]
            rows.append(
                {
                    "fold": fold.name,
                    "train_end": fold.train_end,
                    "val_end": fold.val_end,
                    "test_end": fold.test_end,
                    "run_dir": str(run_dir),
                    "model": model_name,
                    "path_mse": path_mse,
                    "n_test_samples": n_samples,
                    "horizon": horizon,
                    "n_trades_non_overlap": int(row["n_trades_non_overlap"]),
                    "trade_rate_non_overlap": float(row["trade_rate_non_overlap"]),
                    "net_total_return_offset_avg": float(
                        row["total_return_non_overlap_cost_adj_offset_avg"]
                    ),
                    "net_annualized_return_offset_avg": float(
                        row["annualized_return_non_overlap_cost_adj_offset_avg"]
                    ),
                    "net_sharpe_offset_avg": float(row["sharpe_non_overlap_cost_adj_offset_avg"]),
                    "net_max_drawdown_offset_avg": float(
                        row["max_drawdown_non_overlap_cost_adj_offset_avg"]
                    ),
                }
            )
    return rows


def _write_report(out_dir: Path, rows: list[dict[str, Any]]) -> None:
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "fold_results.csv", index=False)

    summary_rows: list[dict[str, Any]] = []
    for horizon in (20, 30):
        tsm = df[(df["model"] == "tsm") & (df["horizon"] == horizon)].copy()
        llm = df[(df["model"] == METHOD_NAME) & (df["horizon"] == horizon)].copy()
        merged = llm.merge(
            tsm,
            on="fold",
            suffixes=("_llm", "_tsm"),
        )
        summary_rows.append(
            {
                "horizon": horizon,
                "folds": int(len(merged)),
                "total_test_samples": int(merged["n_test_samples_llm"].sum()),
                "total_non_overlap_trades": int(merged["n_trades_non_overlap_llm"].sum()),
                "llm_mean_path_mse": float(merged["path_mse_llm"].mean()),
                "tsm_mean_path_mse": float(merged["path_mse_tsm"].mean()),
                "llm_beats_tsm_path_folds": int(np.sum(merged["path_mse_llm"] < merged["path_mse_tsm"])),
                "llm_mean_net_total_return": float(merged["net_total_return_offset_avg_llm"].mean()),
                "tsm_mean_net_total_return": float(merged["net_total_return_offset_avg_tsm"].mean()),
                "llm_mean_net_sharpe": float(merged["net_sharpe_offset_avg_llm"].mean()),
                "tsm_mean_net_sharpe": float(merged["net_sharpe_offset_avg_tsm"].mean()),
                "llm_beats_tsm_return_folds": int(
                    np.sum(merged["net_total_return_offset_avg_llm"] > merged["net_total_return_offset_avg_tsm"])
                ),
            }
        )
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(out_dir / "summary_metrics.csv", index=False)

    lines = [
        "# UK ETS Rolling-Origin Forecast Check",
        "",
        f"Generated: {datetime.now().isoformat()}",
        "",
        "Aim:",
        "- Increase evidential confidence by testing the live 4B forecast winner on multiple disjoint UK test blocks.",
        "- Expected outcome: if the LLM edge is real, it should beat base TSM on path MSE in most folds and accumulate materially more than 8 h20 trades / 5 h30 trades overall.",
        "",
        "## Horizon Summary",
        "",
    ]
    for _, row in summary_df.iterrows():
        lines.extend(
            [
                f"- h{int(row['horizon'])}: {int(row['folds'])} folds, {int(row['total_test_samples'])} total test samples, "
                f"{int(row['total_non_overlap_trades'])} non-overlap trades",
                f"  LLM mean path MSE `{row['llm_mean_path_mse']:.6f}` vs TSM `{row['tsm_mean_path_mse']:.6f}`",
                f"  LLM beat TSM on path MSE in `{int(row['llm_beats_tsm_path_folds'])}/{int(row['folds'])}` folds",
                f"  LLM mean net return `{row['llm_mean_net_total_return']:.2%}` vs TSM `{row['tsm_mean_net_total_return']:.2%}`",
                f"  LLM mean net Sharpe `{row['llm_mean_net_sharpe']:.3f}` vs TSM `{row['tsm_mean_net_sharpe']:.3f}`",
                f"  LLM beat TSM on net return in `{int(row['llm_beats_tsm_return_folds'])}/{int(row['folds'])}` folds",
            ]
        )

    lines.extend(
        [
            "",
            "## Fold Results",
            "",
            df.to_markdown(index=False),
            "",
            "Interpretation:",
            "- This is more informative than a single terminal holdout because the test blocks are disjoint in time.",
            "- It is still not a production-grade trading study; the next step after this is a longer rolling history or a monthly rebalance study with cost/slippage stress tests.",
        ]
    )
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run rolling-origin validation for the UK 4B forecast winner.")
    parser.add_argument(
        "--config",
        default="uk_ets/config/uk_ets_llm_4b_feedback_skill_tag_recent_high_error_k4.yaml",
    )
    parser.add_argument("--data-dir", default="uk_ets/Data_auto_uk")
    parser.add_argument(
        "--output-root",
        default="reports/uk_ets_llm_4b_rolling_origin_forecast",
    )
    args = parser.parse_args()

    config_path = (PROJECT_ROOT / args.config).resolve()
    data_dir = (PROJECT_ROOT / args.data_dir).resolve()

    folds = [
        Fold(
            name="2023H2",
            train_end="2022-12-31",
            val_end="2023-06-30",
            test_end="2023-12-31",
        ),
        Fold(
            name="2024H1",
            train_end="2023-06-30",
            val_end="2023-12-31",
            test_end="2024-06-30",
        ),
        Fold(
            name="2024H2",
            train_end="2023-12-31",
            val_end="2024-06-30",
            test_end="2024-12-31",
        ),
        Fold(
            name="2025H1",
            train_end="2024-06-30",
            val_end="2024-12-31",
            test_end="2025-06-30",
        ),
        Fold(
            name="2025H2_2026Q1_holdout",
            train_end="2024-06-30",
            val_end="2025-06-30",
            test_end="2026-03-04",
            existing_run_dir="runs/20260311_135144_bb26cf",
        ),
    ]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = (PROJECT_ROOT / args.output_root / timestamp).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for fold in folds:
        if fold.existing_run_dir:
            run_dir = (PROJECT_ROOT / fold.existing_run_dir).resolve()
        else:
            run_dir = Path(
                run_experiment(
                    config_path=str(config_path),
                    overrides={
                        "split": {
                            "train_end": fold.train_end,
                            "val_end": fold.val_end,
                            "test_end": fold.test_end,
                        },
                        "output": {"write_project_paper": False},
                    },
                    data_dir=str(data_dir),
                )
            ).resolve()
        rows.extend(_collect_fold_result(fold, run_dir))

    _write_report(out_dir, rows)
    print(out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
