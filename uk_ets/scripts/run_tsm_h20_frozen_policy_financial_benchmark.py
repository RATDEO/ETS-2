#!/usr/bin/env python3
"""Run the frozen h20 linear-size financial benchmark across UK folds."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]

import sys

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.eval.return_metrics import assess_llm_training_cutoff
from uk_ets.scripts.run_tsm_h20_financial_benchmark import (
    CUTOFF_DATE,
    EMBARGO_DAYS,
    METHOD_NAME,
    _apply_embargo,
)
from uk_ets.scripts.run_tsm_h20_trading_policy_build import (
    EXECUTION_LAG_DAYS,
    PRIMARY_HORIZON,
    SLIPPAGE_BPS,
    TRANSACTION_COST_BPS,
    _forecast_returns,
    _load_npz,
    _make_weights,
    _realized_returns,
    _weighted_portfolio_metrics,
)


POLICY_SOURCE_DIR = (
    PROJECT_ROOT / "reports" / "uk_ets_tsm_h20_trading_policy_build" / "20260316_165334"
)


@dataclass(frozen=True)
class FoldSpec:
    name: str
    train_end: str
    val_end: str
    test_end: str
    run_dir: str


def _folds() -> list[FoldSpec]:
    return [
        FoldSpec("2023H2", "2022-12-31", "2023-06-30", "2023-12-31", "runs/20260316_194225_3a0273"),
        FoldSpec("2024H1", "2023-06-30", "2023-12-31", "2024-06-30", "runs/20260316_200457_fdd803"),
        FoldSpec("2024H2", "2023-12-31", "2024-06-30", "2024-12-31", "runs/20260316_202338_216dc9"),
        FoldSpec("2025H1", "2024-06-30", "2024-12-31", "2025-06-30", "runs/20260316_210400_3446ce"),
        FoldSpec(
            "2025H2_2026Q1_holdout",
            "2024-06-30",
            "2025-06-30",
            "2026-03-04",
            "runs/20260316_170659_fb765d",
        ),
    ]


def _load_frozen_scales() -> dict[str, float]:
    df = pd.read_csv(POLICY_SOURCE_DIR / "policy_results.csv")
    rows = df[(df["horizon"] == PRIMARY_HORIZON) & (df["policy"] == "linear_size")]
    scales: dict[str, float] = {}
    for model_name in ("raw_tsm", "learned_gbdt"):
        row = rows[rows["model"] == model_name]
        if row.empty:
            raise RuntimeError(f"Missing frozen linear_size selection for {model_name}")
        scales[model_name] = float(row.iloc[0]["selected_scale_or_threshold"])
    return scales


def _evaluate_fold_model(
    *,
    fold: FoldSpec,
    model_name: str,
    scale: float,
    out_dir: Path,
) -> dict[str, object]:
    run_dir = (PROJECT_ROOT / fold.run_dir).resolve()
    pred_path = run_dir / "predictions" / f"{METHOD_NAME}_pred_test_subset.npz"
    data = _load_npz(pred_path)
    panel = pd.read_parquet(run_dir / "data" / "panel.parquet")
    panel_dates = panel["date"]
    panel_prices = panel["y"]

    test_mask = _apply_embargo(data["dates"], fold.val_end, EMBARGO_DAYS)
    test_true = data["y_true"][test_mask]
    test_base = data["base_pred"][test_mask]
    test_dates = pd.to_datetime(data["dates"][test_mask])

    if model_name == "raw_tsm":
        test_pred = test_base
    elif model_name == "learned_gbdt":
        test_pred = data["yhat"][test_mask]
    else:
        raise ValueError(f"Unsupported model_name: {model_name}")

    base_prices = test_base[:, 0]
    test_returns = _forecast_returns(test_pred, base_prices, PRIMARY_HORIZON)
    test_realized = _realized_returns(test_true, base_prices, PRIMARY_HORIZON)
    test_weights = _make_weights(
        "linear_size",
        test_returns,
        threshold_or_scale=scale,
        prob_floor=0.0,
    )
    mtm_df, test_metrics, trade_df = _weighted_portfolio_metrics(
        close_prices=panel_prices,
        dates=panel_dates,
        decision_dates=test_dates,
        raw_weights=test_weights,
        hold_days=PRIMARY_HORIZON,
        execution_lag_days=EXECUTION_LAG_DAYS,
        transaction_cost_bps=TRANSACTION_COST_BPS,
        slippage_bps=SLIPPAGE_BPS,
    )
    nonzero_mask = np.abs(test_weights) > 1e-12
    directional_accuracy = (
        float(np.mean(np.sign(test_weights[nonzero_mask]) == np.sign(test_realized[nonzero_mask])))
        if np.any(nonzero_mask)
        else np.nan
    )
    cutoff_note = assess_llm_training_cutoff(test_dates, cutoff_date=CUTOFF_DATE)

    model_dir = out_dir / fold.name / model_name
    model_dir.mkdir(parents=True, exist_ok=True)
    trade_df.to_csv(model_dir / "trade_ledger.csv", index=False)
    mtm_df.to_csv(model_dir / "daily_mtm.csv", index=False)
    pd.DataFrame(
        {
            "date": test_dates,
            "weight": test_weights,
            "forecast_return": test_returns,
            "realized_return": test_realized,
        }
    ).to_csv(model_dir / "test_weights.csv", index=False)

    return {
        "fold": fold.name,
        "train_end": fold.train_end,
        "val_end": fold.val_end,
        "test_end": fold.test_end,
        "run_dir": str(run_dir),
        "model": model_name,
        "policy": "linear_size_frozen",
        "primary_horizon": PRIMARY_HORIZON,
        "embargo_days": EMBARGO_DAYS,
        "execution_lag_days": EXECUTION_LAG_DAYS,
        "transaction_cost_bps": TRANSACTION_COST_BPS,
        "slippage_bps": SLIPPAGE_BPS,
        "n_test_after_embargo": int(len(test_dates)),
        "frozen_scale": float(scale),
        "test_sharpe": float(test_metrics["daily_sharpe_annualized"]),
        "test_total_return": float(test_metrics["total_return"]),
        "test_annualized_return": float(test_metrics["annualized_return"]),
        "test_max_drawdown": float(test_metrics["max_drawdown"]),
        "test_n_trades": int(test_metrics["n_executed_trades"]),
        "test_avg_abs_weight": float(test_metrics["avg_abs_weight"]),
        "test_directional_accuracy_nonzero": float(directional_accuracy),
        "path_mse_embargoed": float(np.mean((test_true - test_pred) ** 2)),
        "llm_cutoff_status": cutoff_note["status"],
        "llm_cutoff_date": cutoff_note["cutoff_date"],
        "llm_n_pre_cutoff": cutoff_note["n_pre_cutoff"],
        "llm_n_post_cutoff": cutoff_note["n_post_cutoff"],
        "llm_cutoff_note": cutoff_note["note"],
    }


def _write_summary(out_dir: Path, results_df: pd.DataFrame, scales: dict[str, float]) -> None:
    pivot = results_df.pivot(index="fold", columns="model", values=["test_sharpe", "test_total_return", "path_mse_embargoed"])
    wins = []
    if ("test_sharpe", "raw_tsm") in pivot.columns and ("test_sharpe", "learned_gbdt") in pivot.columns:
        wins.append(
            f"- `learned_gbdt` beat `raw_tsm` on Sharpe in "
            f"`{int((pivot[('test_sharpe', 'learned_gbdt')] > pivot[('test_sharpe', 'raw_tsm')]).sum())}/{len(pivot)}` folds."
        )
    if ("test_total_return", "raw_tsm") in pivot.columns and ("test_total_return", "learned_gbdt") in pivot.columns:
        wins.append(
            f"- `learned_gbdt` beat `raw_tsm` on total return in "
            f"`{int((pivot[('test_total_return', 'learned_gbdt')] > pivot[('test_total_return', 'raw_tsm')]).sum())}/{len(pivot)}` folds."
        )
    if ("path_mse_embargoed", "raw_tsm") in pivot.columns and ("path_mse_embargoed", "learned_gbdt") in pivot.columns:
        wins.append(
            f"- `learned_gbdt` beat `raw_tsm` on embargoed path MSE in "
            f"`{int((pivot[('path_mse_embargoed', 'learned_gbdt')] < pivot[('path_mse_embargoed', 'raw_tsm')]).sum())}/{len(pivot)}` folds."
        )

    lines = [
        "# TSM H20 Frozen-Policy Financial Benchmark",
        "",
        f"Generated: {datetime.now().isoformat()}",
        "",
        "Protocol:",
        "- Forecast stack is frozen at the promoted live production candidate:",
        "  - `DLinear + learned_gbdt(top20) + selective LLM` for `learned_gbdt`.",
        "  - raw base `TSM` extracted from the same fold runs for `raw_tsm`.",
        "- Execution family is frozen to `linear_size` at `h20`.",
        f"- Frozen scale for `raw_tsm`: `{scales['raw_tsm']:.15f}`.",
        f"- Frozen scale for `learned_gbdt`: `{scales['learned_gbdt']:.15f}`.",
        f"- Test-only evaluation with `{EMBARGO_DAYS}`-day embargo, `{EXECUTION_LAG_DAYS}`-day lag, `{TRANSACTION_COST_BPS:.1f}` bps costs, `{SLIPPAGE_BPS:.1f}` bps slippage.",
        "",
        "## Fold Results",
        "",
        results_df.to_markdown(index=False),
        "",
        "## Model Summary",
    ]

    for model_name, grp in results_df.groupby("model", sort=False):
        lines.append(
            f"- `{model_name}`: mean path MSE `{grp['path_mse_embargoed'].mean():.6f}`, "
            f"mean total return `{grp['test_total_return'].mean():.2%}`, "
            f"mean annualized return `{grp['test_annualized_return'].mean():.2%}`, "
            f"mean Sharpe `{grp['test_sharpe'].mean():.3f}`, "
            f"mean max drawdown `{grp['test_max_drawdown'].mean():.2%}`."
        )

    if wins:
        lines.extend(["", "## Fold Wins", *wins])

    post_cutoff = results_df[results_df["llm_cutoff_status"] == "post_cutoff_only"].copy()
    if not post_cutoff.empty:
        lines.extend(["", "## Post-2025 Summary"])
        for model_name, grp in post_cutoff.groupby("model", sort=False):
            lines.append(
                f"- `{model_name}`: mean total return `{grp['test_total_return'].mean():.2%}`, "
                f"mean annualized return `{grp['test_annualized_return'].mean():.2%}`, "
                f"mean Sharpe `{grp['test_sharpe'].mean():.3f}` across `{len(grp)}` cleaner fold(s)."
            )

    (out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    out_dir = (
        PROJECT_ROOT
        / "reports"
        / "uk_ets_tsm_h20_financial_benchmark_frozen"
        / datetime.now().strftime("%Y%m%d_%H%M%S")
    ).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    scales = _load_frozen_scales()
    rows: list[dict[str, object]] = []
    for fold in _folds():
        for model_name in ("raw_tsm", "learned_gbdt"):
            rows.append(
                _evaluate_fold_model(
                    fold=fold,
                    model_name=model_name,
                    scale=scales[model_name],
                    out_dir=out_dir,
                )
            )
            pd.DataFrame(rows).to_csv(out_dir / "fold_results.csv", index=False)

    results_df = pd.DataFrame(rows).sort_values(["test_end", "model"]).reset_index(drop=True)
    results_df.to_csv(out_dir / "fold_results.csv", index=False)
    _write_summary(out_dir, results_df, scales)
    print(out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
