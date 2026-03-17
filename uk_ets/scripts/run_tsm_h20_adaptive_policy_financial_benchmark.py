#!/usr/bin/env python3
"""Adaptive h20 execution benchmark with online policy refits."""

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
    TradingPolicy,
    _candidate_grid,
    _forecast_returns,
    _load_npz,
    _make_weights,
    _realized_returns,
    _weighted_portfolio_metrics,
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


def _adaptive_policy_family() -> list[TradingPolicy]:
    return [
        TradingPolicy(
            name="sign_threshold",
            objective="Adaptive thresholded sign execution.",
            expectation="Should protect against weak forecasts when edge is mostly directional.",
        ),
        TradingPolicy(
            name="linear_size",
            objective="Adaptive linear sizing on forecast magnitude.",
            expectation="Should work best if improved magnitude calibration is the main edge.",
        ),
        TradingPolicy(
            name="tanh_size",
            objective="Adaptive smooth sizing on forecast magnitude.",
            expectation="Should reduce exposure when extreme forecasts are noisy.",
        ),
    ]


def _evaluate_candidate(
    *,
    policy: TradingPolicy,
    history_returns: np.ndarray,
    history_realized: np.ndarray,
    history_dates: pd.Index,
    panel_dates: pd.Series,
    panel_prices: pd.Series,
) -> tuple[float, dict[str, float]]:
    best_score = -np.inf
    best_params = (0.0, 0.0)
    best_metrics: dict[str, float] | None = None
    for threshold_or_scale, prob_floor in _candidate_grid(
        policy.name,
        history_returns,
        base_returns=None,
        apply_probs=None,
    ):
        weights = _make_weights(
            policy.name,
            history_returns,
            threshold_or_scale=threshold_or_scale,
            prob_floor=prob_floor,
        )
        _, metrics, _ = _weighted_portfolio_metrics(
            close_prices=panel_prices,
            dates=panel_dates,
            decision_dates=history_dates,
            raw_weights=weights,
            hold_days=PRIMARY_HORIZON,
            execution_lag_days=EXECUTION_LAG_DAYS,
            transaction_cost_bps=TRANSACTION_COST_BPS,
            slippage_bps=SLIPPAGE_BPS,
        )
        score = float(metrics["daily_sharpe_annualized"])
        if not np.isfinite(score):
            score = -np.inf
        if metrics["n_executed_trades"] < max(5, PRIMARY_HORIZON // 4):
            score = -np.inf
        if score > best_score:
            best_score = score
            best_params = (float(threshold_or_scale), float(prob_floor))
            best_metrics = metrics
    if best_metrics is None:
        best_metrics = {
            "daily_sharpe_annualized": np.nan,
            "total_return": np.nan,
            "annualized_return": np.nan,
            "max_drawdown": np.nan,
            "n_executed_trades": 0,
            "avg_abs_weight": 0.0,
        }
    return best_params[0], best_metrics


def _select_online_policy(
    *,
    history_returns: np.ndarray,
    history_realized: np.ndarray,
    history_dates: pd.Index,
    panel_dates: pd.Series,
    panel_prices: pd.Series,
) -> tuple[str, float, dict[str, float]]:
    best_policy_name = "sign_threshold"
    best_scale = 0.0
    best_metrics: dict[str, float] | None = None
    best_score = -np.inf
    for policy in _adaptive_policy_family():
        scale, metrics = _evaluate_candidate(
            policy=policy,
            history_returns=history_returns,
            history_realized=history_realized,
            history_dates=history_dates,
            panel_dates=panel_dates,
            panel_prices=panel_prices,
        )
        score = float(metrics["daily_sharpe_annualized"])
        if not np.isfinite(score):
            score = -np.inf
        if score > best_score:
            best_score = score
            best_policy_name = policy.name
            best_scale = float(scale)
            best_metrics = metrics
    assert best_metrics is not None
    return best_policy_name, best_scale, best_metrics


def _online_weights(
    *,
    val_returns: np.ndarray,
    val_realized: np.ndarray,
    val_dates: pd.Index,
    test_returns: np.ndarray,
    test_realized: np.ndarray,
    test_dates: pd.Index,
    panel_dates: pd.Series,
    panel_prices: pd.Series,
) -> tuple[np.ndarray, pd.DataFrame]:
    weights = np.zeros(len(test_returns), dtype=float)
    selection_rows: list[dict[str, Any]] = []
    for i in range(len(test_returns)):
        matured_test_cutoff = max(0, i - PRIMARY_HORIZON)
        hist_returns = np.concatenate([val_returns, test_returns[:matured_test_cutoff]])
        hist_realized = np.concatenate([val_realized, test_realized[:matured_test_cutoff]])
        hist_dates = pd.Index(np.concatenate([val_dates.to_numpy(), test_dates[:matured_test_cutoff].to_numpy()]))
        policy_name, scale, metrics = _select_online_policy(
            history_returns=hist_returns,
            history_realized=hist_realized,
            history_dates=hist_dates,
            panel_dates=panel_dates,
            panel_prices=panel_prices,
        )
        current_weight = _make_weights(
            policy_name,
            np.asarray([test_returns[i]], dtype=float),
            threshold_or_scale=scale,
            prob_floor=0.0,
        )[0]
        weights[i] = float(current_weight)
        selection_rows.append(
            {
                "decision_date": str(pd.Timestamp(test_dates[i]).date()),
                "selected_policy": policy_name,
                "selected_scale": float(scale),
                "history_size": int(len(hist_returns)),
                "history_test_matured": int(matured_test_cutoff),
                "history_sharpe": float(metrics["daily_sharpe_annualized"]),
                "history_total_return": float(metrics["total_return"]),
                "weight": float(current_weight),
                "forecast_return": float(test_returns[i]),
                "realized_return": float(test_realized[i]),
            }
        )
    return weights, pd.DataFrame(selection_rows)


def _evaluate_fold_model(*, fold: FoldSpec, model_name: str, out_dir: Path) -> dict[str, object]:
    run_dir = (PROJECT_ROOT / fold.run_dir).resolve()
    val_data = _load_npz(run_dir / "predictions" / f"{METHOD_NAME}_pred_val_full.npz")
    test_data = _load_npz(run_dir / "predictions" / f"{METHOD_NAME}_pred_test_subset.npz")
    panel = pd.read_parquet(run_dir / "data" / "panel.parquet")
    panel_dates = panel["date"]
    panel_prices = panel["y"]

    val_mask = _apply_embargo(val_data["dates"], fold.train_end, EMBARGO_DAYS)
    test_mask = _apply_embargo(test_data["dates"], fold.val_end, EMBARGO_DAYS)

    val_true = val_data["y_true"][val_mask]
    test_true = test_data["y_true"][test_mask]
    val_base = val_data["base_pred"][val_mask]
    test_base = test_data["base_pred"][test_mask]
    val_dates = pd.to_datetime(val_data["dates"][val_mask])
    test_dates = pd.to_datetime(test_data["dates"][test_mask])

    if model_name == "raw_tsm":
        val_pred = val_base
        test_pred = test_base
    elif model_name == "learned_gbdt":
        val_pred = val_data["yhat"][val_mask]
        test_pred = test_data["yhat"][test_mask]
    else:
        raise ValueError(f"Unsupported model_name: {model_name}")

    val_base_prices = val_base[:, 0]
    test_base_prices = test_base[:, 0]
    val_returns = _forecast_returns(val_pred, val_base_prices, PRIMARY_HORIZON)
    test_returns = _forecast_returns(test_pred, test_base_prices, PRIMARY_HORIZON)
    val_realized = _realized_returns(val_true, val_base_prices, PRIMARY_HORIZON)
    test_realized = _realized_returns(test_true, test_base_prices, PRIMARY_HORIZON)

    test_weights, selection_df = _online_weights(
        val_returns=val_returns,
        val_realized=val_realized,
        val_dates=val_dates,
        test_returns=test_returns,
        test_realized=test_realized,
        test_dates=test_dates,
        panel_dates=panel_dates,
        panel_prices=panel_prices,
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
    selection_df.to_csv(model_dir / "online_policy_selection.csv", index=False)
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
        "policy_mode": "adaptive_online_refit",
        "policy_family_set": "sign_threshold,linear_size,tanh_size",
        "primary_horizon": PRIMARY_HORIZON,
        "embargo_days": EMBARGO_DAYS,
        "execution_lag_days": EXECUTION_LAG_DAYS,
        "transaction_cost_bps": TRANSACTION_COST_BPS,
        "slippage_bps": SLIPPAGE_BPS,
        "n_val_after_embargo": int(len(val_dates)),
        "n_test_after_embargo": int(len(test_dates)),
        "test_sharpe": float(test_metrics["daily_sharpe_annualized"]),
        "test_total_return": float(test_metrics["total_return"]),
        "test_annualized_return": float(test_metrics["annualized_return"]),
        "test_max_drawdown": float(test_metrics["max_drawdown"]),
        "test_n_trades": int(test_metrics["n_executed_trades"]),
        "test_avg_abs_weight": float(test_metrics["avg_abs_weight"]),
        "test_directional_accuracy_nonzero": float(directional_accuracy),
        "path_mse_embargoed": float(np.mean((test_true - test_pred) ** 2)),
        "adaptive_policy_linear_share": float((selection_df["selected_policy"] == "linear_size").mean()),
        "adaptive_policy_tanh_share": float((selection_df["selected_policy"] == "tanh_size").mean()),
        "adaptive_policy_sign_share": float((selection_df["selected_policy"] == "sign_threshold").mean()),
        "llm_cutoff_status": cutoff_note["status"],
        "llm_cutoff_date": cutoff_note["cutoff_date"],
        "llm_n_pre_cutoff": cutoff_note["n_pre_cutoff"],
        "llm_n_post_cutoff": cutoff_note["n_post_cutoff"],
        "llm_cutoff_note": cutoff_note["note"],
    }


def _write_summary(out_dir: Path, results_df: pd.DataFrame) -> None:
    pivot = results_df.pivot(index="fold", columns="model", values=["test_sharpe", "test_total_return", "path_mse_embargoed"])
    lines = [
        "# TSM H20 Adaptive-Policy Financial Benchmark",
        "",
        f"Generated: {datetime.now().isoformat()}",
        "",
        "Protocol:",
        "- Forecast stack is causal and fold-specific.",
        "- Execution layer is reselected online at each test decision date.",
        "- Refit history uses only embargoed validation decisions plus prior test decisions whose `h20` outcome is already realized.",
        "- Policy family set is deliberately constrained to `sign_threshold`, `linear_size`, and `tanh_size`.",
        f"- Costs use `{TRANSACTION_COST_BPS:.1f}` bps transaction cost and `{SLIPPAGE_BPS:.1f}` bps slippage with `{EXECUTION_LAG_DAYS}`-day lag.",
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
    lines.extend(
        [
            "",
            "## Fold Wins",
            f"- `learned_gbdt` beat `raw_tsm` on Sharpe in `{int((pivot[('test_sharpe', 'learned_gbdt')] > pivot[('test_sharpe', 'raw_tsm')]).sum())}/{len(pivot)}` folds.",
            f"- `learned_gbdt` beat `raw_tsm` on total return in `{int((pivot[('test_total_return', 'learned_gbdt')] > pivot[('test_total_return', 'raw_tsm')]).sum())}/{len(pivot)}` folds.",
            f"- `learned_gbdt` beat `raw_tsm` on embargoed path MSE in `{int((pivot[('path_mse_embargoed', 'learned_gbdt')] < pivot[('path_mse_embargoed', 'raw_tsm')]).sum())}/{len(pivot)}` folds.",
        ]
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
    out_dir = (
        PROJECT_ROOT
        / "reports"
        / "uk_ets_tsm_h20_financial_benchmark_adaptive"
        / datetime.now().strftime("%Y%m%d_%H%M%S")
    ).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    for fold in _folds():
        for model_name in ("raw_tsm", "learned_gbdt"):
            rows.append(_evaluate_fold_model(fold=fold, model_name=model_name, out_dir=out_dir))
            pd.DataFrame(rows).to_csv(out_dir / "fold_results.csv", index=False)

    results_df = pd.DataFrame(rows).sort_values(["test_end", "model"]).reset_index(drop=True)
    results_df.to_csv(out_dir / "fold_results.csv", index=False)
    _write_summary(out_dir, results_df)
    print(out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
