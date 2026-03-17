#!/usr/bin/env python3
"""Run a full rolling financial benchmark for the selected live h20 trading policy."""

from __future__ import annotations

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

from src.eval.return_metrics import assess_llm_training_cutoff
from src.run_experiment import run_experiment
from uk_ets.scripts.run_tsm_h20_trading_policy_build import (
    EXECUTION_LAG_DAYS,
    PRIMARY_HORIZON,
    SLIPPAGE_BPS,
    TRANSACTION_COST_BPS,
    _evaluate_policy,
    _forecast_returns,
    _load_learned_probs,
    _load_npz,
    _policy_families,
    _realized_returns,
    _selected_candidates,
)


CONFIG_PATH = "uk_ets/config/uk_ets_llm_4b_tsm_live_gbdt_top20.yaml"
METHOD_NAME = "TSM+LLM-COT-RF-HDELTA"
EMBARGO_DAYS = 30
CUTOFF_DATE = "2025-01-01"


@dataclass(frozen=True)
class Fold:
    name: str
    train_end: str
    val_end: str
    test_end: str
    existing_learned_run_dir: str | None = None


def _default_folds() -> list[Fold]:
    return [
        Fold("2023H2", "2022-12-31", "2023-06-30", "2023-12-31"),
        Fold("2024H1", "2023-06-30", "2023-12-31", "2024-06-30"),
        Fold("2024H2", "2023-12-31", "2024-06-30", "2024-12-31"),
        Fold("2025H1", "2024-06-30", "2024-12-31", "2025-06-30"),
        Fold(
            "2025H2_2026Q1_holdout",
            "2024-06-30",
            "2025-06-30",
            "2026-03-04",
            existing_learned_run_dir="runs/20260316_170659_fb765d",
        ),
    ]


def _resolve_project_path(path_str: str) -> Path:
    path = Path(path_str).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (PROJECT_ROOT / path).resolve()


def _run_fold_candidate(run_name: str, base_overrides: dict[str, Any], fold: Fold) -> Path:
    overrides = {
        "split": {
            "train_end": fold.train_end,
            "val_end": fold.val_end,
            "test_end": fold.test_end,
        },
    }
    merged = dict(base_overrides)
    merged["split"] = overrides["split"]
    return Path(run_experiment(config_path=CONFIG_PATH, overrides=merged)).resolve()


def _resolve_fold_run(fold: Fold) -> Path:
    selected = _selected_candidates()
    if fold.existing_learned_run_dir:
        learned = _resolve_project_path(fold.existing_learned_run_dir)
        if learned.exists():
            return learned
    return _run_fold_candidate("learned_gbdt_regime", selected["learned_gbdt_regime"], fold)


def _apply_embargo(dates: np.ndarray, cutoff_date: str, embargo_days: int) -> np.ndarray:
    embargo_cutoff = pd.Timestamp(cutoff_date) + pd.Timedelta(days=int(embargo_days))
    return pd.to_datetime(dates) > embargo_cutoff


def _evaluate_fold_model(
    *,
    model_name: str,
    fold: Fold,
    learned_run: Path,
    out_dir: Path,
) -> dict[str, Any]:
    learned_val = _load_npz(learned_run / "predictions" / f"{METHOD_NAME}_pred_val_full.npz")
    learned_test = _load_npz(learned_run / "predictions" / f"{METHOD_NAME}_pred_test_subset.npz")

    if model_name == "raw_tsm":
        val_data = learned_val
        test_data = learned_test
        val_pred = val_data["base_pred"]
        test_pred = test_data["base_pred"]
        val_probs = None
        test_probs = None
    elif model_name == "learned_gbdt":
        val_data = learned_val
        test_data = learned_test
        val_pred = val_data["yhat"]
        test_pred = test_data["yhat"]
        val_probs, test_probs = _load_learned_probs(learned_run)
    else:
        raise ValueError(f"Unsupported model_name: {model_name}")

    panel = pd.read_parquet(learned_run / "data" / "panel.parquet")
    panel_dates = panel["date"]
    panel_prices = panel["y"]

    val_mask = _apply_embargo(val_data["dates"], fold.train_end, EMBARGO_DAYS)
    test_mask = _apply_embargo(test_data["dates"], fold.val_end, EMBARGO_DAYS)

    val_base = val_data["base_pred"][val_mask]
    test_base = test_data["base_pred"][test_mask]
    val_pred = val_pred[val_mask]
    test_pred = test_pred[test_mask]
    val_true = val_data["y_true"][val_mask]
    test_true = test_data["y_true"][test_mask]
    val_dates = pd.to_datetime(val_data["dates"][val_mask])
    test_dates = pd.to_datetime(test_data["dates"][test_mask])
    if val_probs is not None:
        val_probs = val_probs[val_mask]
        test_probs = test_probs[test_mask]

    val_base_prices = val_base[:, 0]
    test_base_prices = test_base[:, 0]
    val_returns = _forecast_returns(val_pred, val_base_prices, PRIMARY_HORIZON)
    test_returns = _forecast_returns(test_pred, test_base_prices, PRIMARY_HORIZON)
    val_base_returns = _forecast_returns(val_base, val_base_prices, PRIMARY_HORIZON)
    test_base_returns = _forecast_returns(test_base, test_base_prices, PRIMARY_HORIZON)
    val_realized = _realized_returns(val_true, val_base_prices, PRIMARY_HORIZON)
    test_realized = _realized_returns(test_true, test_base_prices, PRIMARY_HORIZON)

    linear_policy = next(policy for policy in _policy_families() if policy.name == "linear_size")
    best_params, val_metrics, test_metrics, test_weights, mtm_df, trade_df = _evaluate_policy(
        policy=linear_policy,
        val_returns=val_returns,
        test_returns=test_returns,
        val_base_returns=val_base_returns,
        test_base_returns=test_base_returns,
        val_probs=val_probs,
        test_probs=test_probs,
        val_realized=val_realized,
        test_realized=test_realized,
        val_dates=val_dates,
        test_dates=test_dates,
        panel_dates=panel_dates,
        panel_prices=panel_prices,
        horizon=PRIMARY_HORIZON,
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
            "base_return": test_base_returns,
            "realized_return": test_realized,
            "apply_probability": test_probs if test_probs is not None else np.nan,
        }
    ).to_csv(model_dir / "test_weights.csv", index=False)

    return {
        "fold": fold.name,
        "train_end": fold.train_end,
        "val_end": fold.val_end,
        "test_end": fold.test_end,
        "learned_run_dir": str(learned_run),
        "model": model_name,
        "policy": "linear_size",
        "primary_horizon": PRIMARY_HORIZON,
        "embargo_days": EMBARGO_DAYS,
        "execution_lag_days": EXECUTION_LAG_DAYS,
        "transaction_cost_bps": TRANSACTION_COST_BPS,
        "slippage_bps": SLIPPAGE_BPS,
        "n_val_after_embargo": int(len(val_dates)),
        "n_test_after_embargo": int(len(test_dates)),
        "selected_scale": float(best_params[0]),
        "selected_prob_floor": float(best_params[1]),
        "val_sharpe": float(val_metrics["daily_sharpe_annualized"]),
        "val_total_return": float(val_metrics["total_return"]),
        "val_n_trades": int(val_metrics["n_executed_trades"]),
        "test_sharpe": float(test_metrics["daily_sharpe_annualized"]),
        "test_total_return": float(test_metrics["total_return"]),
        "test_annualized_return": float(test_metrics["annualized_return"]),
        "test_max_drawdown": float(test_metrics["max_drawdown"]),
        "test_n_trades": int(test_metrics["n_executed_trades"]),
        "test_avg_abs_weight": float(test_metrics["avg_abs_weight"]),
        "test_directional_accuracy_nonzero": float(test_metrics["directional_accuracy_nonzero"]),
        "path_mse_embargoed": float(np.mean((test_true - test_pred) ** 2)),
        "llm_cutoff_status": cutoff_note["status"],
        "llm_cutoff_date": cutoff_note["cutoff_date"],
        "llm_n_pre_cutoff": cutoff_note["n_pre_cutoff"],
        "llm_n_post_cutoff": cutoff_note["n_post_cutoff"],
        "llm_cutoff_note": cutoff_note["note"],
    }


def _write_summary(out_dir: Path, results_df: pd.DataFrame) -> None:
    lines = [
        "# TSM H20 Financial Benchmark",
        "",
        f"Generated: {datetime.now().isoformat()}",
        "",
        "Protocol:",
        f"- Frozen execution family: `linear_size` at `h{PRIMARY_HORIZON}`.",
        f"- Forecast stack for promoted model: `DLinear + learned_gbdt(top20) + selective LLM`.",
        f"- Validation-only scale calibration with `{EMBARGO_DAYS}`-day embargo.",
        f"- Execution lag: `{EXECUTION_LAG_DAYS}` trading day.",
        f"- Costs: `{TRANSACTION_COST_BPS:.1f}` bps + `{SLIPPAGE_BPS:.1f}` bps slippage per side.",
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
    out_dir = (PROJECT_ROOT / "reports" / "uk_ets_tsm_h20_financial_benchmark" / datetime.now().strftime("%Y%m%d_%H%M%S")).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for fold in _default_folds():
        learned_run = _resolve_fold_run(fold)
        for model_name in ("raw_tsm", "learned_gbdt"):
            rows.append(
                _evaluate_fold_model(
                    model_name=model_name,
                    fold=fold,
                    learned_run=learned_run,
                    out_dir=out_dir,
                )
            )
            pd.DataFrame(rows).to_csv(out_dir / "fold_results.csv", index=False)

    results_df = pd.DataFrame(rows)
    _write_summary(out_dir, results_df)
    print(out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
