#!/usr/bin/env python3
"""Recompute UK ETS trading metrics from archived prediction artifacts."""

from __future__ import annotations

import argparse
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


METHOD_NAME = "TSM+LLM-COT-RF-HDELTA"
HORIZONS = [1, 5, 20, 30]


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

    base_prices = []
    for date_value in pd.to_datetime(dates):
        pos = date_to_pos.get(pd.Timestamp(date_value))
        if pos is None or pos == 0:
            raise ValueError(f"Could not resolve base price before forecast date {date_value}")
        base_prices.append(values[pos - 1])
    return np.asarray(base_prices, dtype=float)


def _load_path_mse(run_dir: Path, model_name: str) -> float:
    path_metrics = pd.read_csv(run_dir / "results" / "path_metrics.csv")
    row = path_metrics[path_metrics["model"] == model_name].iloc[0]
    return float(row["mse_path"])


def _load_legacy_portfolio_row(run_dir: Path, model_name: str, horizon: int) -> dict[str, float]:
    df = pd.read_csv(run_dir / "results" / "long_short_metrics.csv")
    subset = df[(df["model"] == model_name) & (df["horizon"] == horizon)]
    if subset.empty:
        return {}
    row = subset.iloc[0]
    return {
        "legacy_total_return_non_overlap": float(row.get("total_return_non_overlap", np.nan)),
        "legacy_annualized_return_non_overlap": float(
            row.get("annualized_return_non_overlap", np.nan)
        ),
        "legacy_sharpe_non_overlap": float(row.get("sharpe_non_overlap", np.nan)),
        "legacy_max_drawdown_non_overlap": float(
            row.get("max_drawdown_non_overlap", np.nan)
        ),
    }


def _collect_model_rows(
    *,
    run_dir: Path,
    label: str,
    model_name: str,
    dates: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    transaction_cost_bps: float,
) -> list[dict[str, Any]]:
    base_prices = _load_base_prices(run_dir, dates)
    portfolio_df = compute_long_short_portfolio_metrics(
        y_true_prices=y_true,
        y_pred_prices=y_pred,
        base_prices=base_prices,
        horizons=HORIZONS,
        signal_threshold=0.0,
        average_non_overlap_offsets=True,
        transaction_cost_bps=transaction_cost_bps,
    ).reset_index()

    path_mse = _load_path_mse(run_dir, model_name)
    rows: list[dict[str, Any]] = []
    for _, row in portfolio_df.iterrows():
        horizon = int(row["horizon"])
        legacy = _load_legacy_portfolio_row(run_dir, model_name, horizon)
        rows.append(
            {
                "run_label": label,
                "run_dir": str(run_dir),
                "model": model_name,
                "path_mse": path_mse,
                "horizon": horizon,
                "transaction_cost_bps": transaction_cost_bps,
                "gross_total_return_offset_avg": float(row["total_return_non_overlap_offset_avg"]),
                "gross_annualized_return_offset_avg": float(
                    row["annualized_return_non_overlap_offset_avg"]
                ),
                "gross_sharpe_offset_avg": float(row["sharpe_non_overlap_offset_avg"]),
                "gross_max_drawdown_offset_avg": float(
                    row["max_drawdown_non_overlap_offset_avg"]
                ),
                "gross_total_return_offset_min": float(row["total_return_non_overlap_offset_min"]),
                "gross_total_return_offset_max": float(row["total_return_non_overlap_offset_max"]),
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
                "net_total_return_offset_min": float(
                    row["total_return_non_overlap_cost_adj_offset_min"]
                ),
                "net_total_return_offset_max": float(
                    row["total_return_non_overlap_cost_adj_offset_max"]
                ),
                **legacy,
            }
        )
    return rows


def _build_summary(rows_df: pd.DataFrame, out_dir: Path, transaction_cost_bps: float) -> None:
    focus = rows_df[rows_df["horizon"].isin([5, 20, 30])].copy()
    lines = [
        "# UK ETS Financial Outcomes",
        "",
        f"Generated: {datetime.now().isoformat()}",
        "",
        "Methodology:",
        f"- Non-overlapping backtests averaged across all valid start offsets for each horizon.",
        f"- Transaction costs assumed at `{transaction_cost_bps:.1f}` bps per side.",
        "- Legacy offset-0, no-cost metrics are included where the archived run already stored them.",
        "",
        "## Forecast Winner",
        "",
    ]

    forecast_rows = focus[
        (focus["run_label"] == "forecast_winner") & (focus["model"] == METHOD_NAME)
    ].sort_values("horizon")
    for _, row in forecast_rows.iterrows():
        lines.append(
            "- h{horizon}: path MSE `{path_mse:.6f}`, gross offset-avg return `{gross_total_return_offset_avg:.2%}`, "
            "net offset-avg return `{net_total_return_offset_avg:.2%}`, net annualized `{net_annualized_return_offset_avg:.2%}`, "
            "net Sharpe `{net_sharpe_offset_avg:.3f}`, net max drawdown `{net_max_drawdown_offset_avg:.2%}`".format(
                **row
            )
        )

    lines.extend(["", "## Trading Winner", ""])
    trading_rows = focus[
        (focus["run_label"] == "trading_winner") & (focus["model"] == METHOD_NAME)
    ].sort_values("horizon")
    for _, row in trading_rows.iterrows():
        lines.append(
            "- h{horizon}: path MSE `{path_mse:.6f}`, gross offset-avg return `{gross_total_return_offset_avg:.2%}`, "
            "net offset-avg return `{net_total_return_offset_avg:.2%}`, net annualized `{net_annualized_return_offset_avg:.2%}`, "
            "net Sharpe `{net_sharpe_offset_avg:.3f}`, net max drawdown `{net_max_drawdown_offset_avg:.2%}`".format(
                **row
            )
        )

    lines.extend(["", "## Base TSM Reference", ""])
    tsm_rows = focus[
        (focus["run_label"] == "forecast_winner") & (focus["model"] == "tsm")
    ].sort_values("horizon")
    for _, row in tsm_rows.iterrows():
        lines.append(
            "- h{horizon}: path MSE `{path_mse:.6f}`, gross offset-avg return `{gross_total_return_offset_avg:.2%}`, "
            "net offset-avg return `{net_total_return_offset_avg:.2%}`, net annualized `{net_annualized_return_offset_avg:.2%}`, "
            "net Sharpe `{net_sharpe_offset_avg:.3f}`, net max drawdown `{net_max_drawdown_offset_avg:.2%}`".format(
                **row
            )
        )

    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- Offset averaging matters most at long horizons; single-offset h30 returns were materially more fragile.",
            "- Cost-adjusted returns remain illustrative; they do not include slippage beyond the fixed round-trip cost assumption.",
        ]
    )

    (out_dir / "summary.md").write_text("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Report offset-averaged, cost-adjusted UK ETS trading metrics.")
    parser.add_argument(
        "--forecast-run",
        default="runs/20260311_135144_bb26cf",
        help="Run directory for the best-forecast 4B candidate.",
    )
    parser.add_argument(
        "--trading-run",
        default="runs/20260311_144036_eb8115",
        help="Run directory for the best-trading 4B candidate.",
    )
    parser.add_argument(
        "--transaction-cost-bps",
        type=float,
        default=10.0,
        help="Per-side transaction cost in basis points.",
    )
    parser.add_argument(
        "--output-root",
        default="reports/uk_ets_financial_outcomes",
        help="Directory where the report will be written.",
    )
    args = parser.parse_args()

    out_dir = (PROJECT_ROOT / args.output_root / datetime.now().strftime("%Y%m%d_%H%M%S")).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []

    forecast_run = (PROJECT_ROOT / args.forecast_run).resolve()
    forecast_tsm_df = pd.read_parquet(forecast_run / "predictions" / "tsm_pred_test.parquet")
    forecast_dates, forecast_y_true, forecast_tsm_pred = _extract_price_paths_from_parquet(
        forecast_tsm_df
    )
    rows.extend(
        _collect_model_rows(
            run_dir=forecast_run,
            label="forecast_winner",
            model_name="tsm",
            dates=forecast_dates,
            y_true=forecast_y_true,
            y_pred=forecast_tsm_pred,
            transaction_cost_bps=args.transaction_cost_bps,
        )
    )
    forecast_llm = np.load(
        forecast_run / "predictions" / f"{METHOD_NAME}_pred_test_subset.npz",
        allow_pickle=True,
    )
    rows.extend(
        _collect_model_rows(
            run_dir=forecast_run,
            label="forecast_winner",
            model_name=METHOD_NAME,
            dates=forecast_llm["dates"],
            y_true=forecast_llm["y_true"],
            y_pred=forecast_llm["yhat"],
            transaction_cost_bps=args.transaction_cost_bps,
        )
    )

    trading_run = (PROJECT_ROOT / args.trading_run).resolve()
    trading_llm = np.load(
        trading_run / "predictions" / f"{METHOD_NAME}_pred_test_subset.npz",
        allow_pickle=True,
    )
    rows.extend(
        _collect_model_rows(
            run_dir=trading_run,
            label="trading_winner",
            model_name=METHOD_NAME,
            dates=trading_llm["dates"],
            y_true=trading_llm["y_true"],
            y_pred=trading_llm["yhat"],
            transaction_cost_bps=args.transaction_cost_bps,
        )
    )

    rows_df = pd.DataFrame(rows)
    rows_df.to_csv(out_dir / "financial_metrics.csv", index=False)
    _build_summary(rows_df, out_dir, args.transaction_cost_bps)
    print(out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
