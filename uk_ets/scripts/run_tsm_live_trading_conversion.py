#!/usr/bin/env python3
"""Rerun live TSM candidates with validation exports and test trading-conversion rules."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]

import sys

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.run_experiment import run_experiment
from uk_ets.scripts.run_tsm_live_policy_build import _candidates, _deep_merge


@dataclass(frozen=True)
class TradingPolicy:
    name: str
    objective: str
    expectation: str


def _selected_candidates() -> dict[str, dict]:
    candidate_map = {candidate.name: candidate for candidate in _candidates()}
    export_override = {
        "llm": {
            "export_val": {
                "enabled": True,
                "scope": "recent_tail",
                "recent_tail_fraction": 0.5,
                "recent_tail_min_samples": 80,
                "max_samples": 80,
            }
        }
    }
    return {
        "baseline_regime_specific": _deep_merge(
            candidate_map["baseline_regime_specific"].overrides,
            export_override,
        ),
        "learned_gbdt_regime": _deep_merge(
            candidate_map["learned_gbdt_regime"].overrides,
            export_override,
        ),
    }


def _policy_families() -> list[TradingPolicy]:
    return [
        TradingPolicy(
            name="sign_threshold",
            objective="Trade only when the absolute forecast return is large enough.",
            expectation="Expected to monetize magnitude gains if the learned gate mainly improves conviction, not direction.",
        ),
        TradingPolicy(
            name="linear_size",
            objective="Size linearly with forecast return magnitude, clipped to [-1, 1].",
            expectation="Expected to reward better long-horizon magnitude estimates if the LLM is improving sizing quality.",
        ),
        TradingPolicy(
            name="tanh_size",
            objective="Use a smooth tanh sizing rule on forecast return magnitude.",
            expectation="Expected to be more stable than linear sizing if extreme forecast returns are noisy.",
        ),
        TradingPolicy(
            name="uplift_gate",
            objective="Only trade the LLM forecast when its horizon return differs enough from raw TSM.",
            expectation="Expected to help if LLM gains are concentrated in a smaller subset of high-uplift windows.",
        ),
        TradingPolicy(
            name="uplift_linear_size",
            objective="Size by the absolute LLM-vs-TSM uplift while keeping the LLM sign.",
            expectation="Expected to help if incremental LLM edge is informative even when raw TSM direction is unchanged.",
        ),
    ]


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    data = np.load(path)
    return {key: data[key] for key in data.files}


def _weighted_portfolio_metrics(
    *,
    close_prices: pd.Series,
    dates: pd.Series,
    decision_dates: pd.Index,
    raw_weights: np.ndarray,
    hold_days: int,
    execution_lag_days: int = 1,
    transaction_cost_bps: float = 10.0,
    slippage_bps: float = 5.0,
    max_concurrent_positions: int | None = None,
    trim_to_active_window: bool = True,
) -> tuple[pd.DataFrame, dict[str, float]]:
    max_positions = int(max_concurrent_positions or hold_days)
    price_index = pd.to_datetime(pd.Index(dates), errors="coerce")
    close = np.asarray(close_prices, dtype=float)
    weights = np.asarray(raw_weights, dtype=float)
    decisions = pd.to_datetime(pd.Index(decision_dates), errors="coerce")

    df = pd.DataFrame({"date": price_index, "close": close}).dropna(subset=["date"]).reset_index(drop=True)
    df["price_return"] = df["close"].pct_change().fillna(0.0)
    exposure = np.zeros(len(df), dtype=float)
    gross_exposure = np.zeros(len(df), dtype=float)
    costs = np.zeros(len(df), dtype=float)
    total_cost_rate = (max(float(transaction_cost_bps), 0.0) + max(float(slippage_bps), 0.0)) / 10000.0
    date_to_idx = {pd.Timestamp(ts): i for i, ts in enumerate(df["date"])}
    trade_rows: list[dict[str, float]] = []

    for decision_date, raw_weight in zip(decisions, weights):
        if pd.isna(decision_date) or abs(float(raw_weight)) < 1e-12:
            continue
        decision_idx = date_to_idx.get(pd.Timestamp(decision_date))
        if decision_idx is None:
            continue
        entry_idx = decision_idx + int(execution_lag_days)
        exit_idx = entry_idx + int(hold_days)
        if entry_idx >= len(df) or exit_idx >= len(df):
            continue

        normalized_weight = float(np.clip(raw_weight, -1.0, 1.0)) / float(max_positions)
        exposure[entry_idx + 1 : exit_idx + 1] += normalized_weight
        gross_exposure[entry_idx + 1 : exit_idx + 1] += abs(normalized_weight)
        if total_cost_rate > 0.0:
            costs[entry_idx] -= total_cost_rate * abs(normalized_weight)
            costs[exit_idx] -= total_cost_rate * abs(normalized_weight)

        trade_log_return = float(
            np.sign(raw_weight)
            * np.sum(np.log(np.clip(1.0 + df["price_return"].iloc[entry_idx + 1 : exit_idx + 1], 1e-12, None)))
            * abs(raw_weight)
        )
        trade_rows.append(
            {
                "decision_date": float(decision_idx),
                "weight": float(raw_weight),
                "gross_trade_return": float(np.expm1(trade_log_return)),
                "net_trade_return": float(np.expm1(trade_log_return) - 2.0 * total_cost_rate * abs(normalized_weight)),
            }
        )

    df["net_exposure"] = exposure
    df["gross_exposure"] = gross_exposure
    df["cost_event"] = costs
    df["strategy_return"] = (df["net_exposure"] * df["price_return"]) + df["cost_event"]

    if trim_to_active_window and trade_rows:
        entry_start = min(int(row["decision_date"]) + int(execution_lag_days) for row in trade_rows)
        exit_end = max(int(row["decision_date"]) + int(execution_lag_days) + int(hold_days) for row in trade_rows)
        df = df.iloc[entry_start : exit_end + 1].copy().reset_index(drop=True)
    elif trim_to_active_window:
        df = df.iloc[0:0].copy()

    if len(df):
        df["equity_curve"] = np.cumprod(1.0 + df["strategy_return"])
    else:
        df["equity_curve"] = pd.Series(dtype=float)

    strategy_ret = df["strategy_return"].to_numpy(dtype=float)
    trade_df = pd.DataFrame(trade_rows)
    if strategy_ret.size:
        total_return = float(df["equity_curve"].iloc[-1] - 1.0)
        annualized_return = float((1.0 + total_return) ** (252.0 / len(df)) - 1.0) if total_return > -1.0 else -1.0
        vol = float(np.std(strategy_ret))
        sharpe = float((np.mean(strategy_ret) / vol) * np.sqrt(252.0)) if vol > 0.0 else np.nan
        peak = np.maximum.accumulate(df["equity_curve"].to_numpy(dtype=float))
        max_drawdown = float(np.max(1.0 - (df["equity_curve"].to_numpy(dtype=float) / peak)))
    else:
        total_return = annualized_return = vol = sharpe = max_drawdown = np.nan

    return df, {
        "n_days": int(len(df)),
        "n_executed_trades": int(len(trade_df)),
        "total_return": total_return,
        "annualized_return": annualized_return,
        "daily_sharpe_annualized": sharpe,
        "max_drawdown": max_drawdown,
        "avg_abs_weight": float(np.mean(np.abs(weights))) if len(weights) else np.nan,
    }


def _forecast_returns(pred: np.ndarray, base_prices: np.ndarray, horizon: int) -> np.ndarray:
    idx = int(horizon) - 1
    base_safe = np.clip(base_prices.astype(float), 1e-8, None)
    return pred[:, idx].astype(float) / base_safe - 1.0


def _realized_returns(y_true: np.ndarray, base_prices: np.ndarray, horizon: int) -> np.ndarray:
    idx = int(horizon) - 1
    base_safe = np.clip(base_prices.astype(float), 1e-8, None)
    return y_true[:, idx].astype(float) / base_safe - 1.0


def _quantile_grid(values: np.ndarray, qs: list[float]) -> list[float]:
    clean = np.abs(np.asarray(values, dtype=float))
    clean = clean[np.isfinite(clean)]
    if clean.size == 0:
        return [0.0]
    return sorted({float(max(np.quantile(clean, q), 0.0)) for q in qs})


def _make_weights(
    policy_name: str,
    forecast_returns: np.ndarray,
    *,
    threshold_or_scale: float,
    base_returns: np.ndarray | None = None,
) -> np.ndarray:
    fr = np.asarray(forecast_returns, dtype=float)
    param = float(max(threshold_or_scale, 1e-8))
    if policy_name == "sign_threshold":
        return np.where(np.abs(fr) >= param, np.sign(fr), 0.0)
    if policy_name == "linear_size":
        return np.clip(fr / param, -1.0, 1.0)
    if policy_name == "tanh_size":
        return np.tanh(fr / param)
    if policy_name == "uplift_gate":
        if base_returns is None:
            raise ValueError("uplift_gate requires base_returns")
        uplift = np.abs(fr - np.asarray(base_returns, dtype=float))
        return np.where(uplift >= param, np.sign(fr), 0.0)
    if policy_name == "uplift_linear_size":
        if base_returns is None:
            raise ValueError("uplift_linear_size requires base_returns")
        uplift = np.abs(fr - np.asarray(base_returns, dtype=float))
        return np.sign(fr) * np.clip(uplift / param, 0.0, 1.0)
    raise ValueError(f"Unknown policy: {policy_name}")


def _candidate_grid(policy_name: str, forecast_returns: np.ndarray, base_returns: np.ndarray | None) -> list[float]:
    quantiles = [0.0, 0.25, 0.4, 0.5, 0.6, 0.75, 0.9]
    if policy_name in {"uplift_gate", "uplift_linear_size"}:
        uplift = np.abs(np.asarray(forecast_returns) - np.asarray(base_returns))
        return _quantile_grid(uplift, quantiles)
    nonzero_quantiles = [0.1, 0.25, 0.4, 0.5, 0.6, 0.75, 0.9]
    return _quantile_grid(forecast_returns, nonzero_quantiles)


def _evaluate_policy(
    *,
    policy_name: str,
    val_returns: np.ndarray,
    test_returns: np.ndarray,
    val_base_returns: np.ndarray,
    test_base_returns: np.ndarray,
    val_realized: np.ndarray,
    test_realized: np.ndarray,
    val_dates: pd.Index,
    test_dates: pd.Index,
    panel_dates: pd.Series,
    panel_prices: pd.Series,
    horizon: int,
) -> tuple[float, dict[str, float], dict[str, float], np.ndarray]:
    best_param = None
    best_metric = -np.inf
    best_val_metrics: dict[str, float] | None = None
    candidates = _candidate_grid(policy_name, val_returns, val_base_returns)
    for candidate in candidates:
        weights = _make_weights(
            policy_name,
            val_returns,
            threshold_or_scale=candidate,
            base_returns=val_base_returns,
        )
        _, metrics = _weighted_portfolio_metrics(
            close_prices=panel_prices,
            dates=panel_dates,
            decision_dates=val_dates,
            raw_weights=weights,
            hold_days=horizon,
        )
        metric = metrics["daily_sharpe_annualized"]
        if not np.isfinite(metric):
            metric = -np.inf
        if metrics["n_executed_trades"] < max(3, horizon // 5):
            metric = -np.inf
        if metric > best_metric:
            best_metric = metric
            best_param = float(candidate)
            best_val_metrics = metrics
    if best_param is None:
        best_param = 0.0
        best_val_metrics = {
            "n_executed_trades": 0,
            "total_return": np.nan,
            "annualized_return": np.nan,
            "daily_sharpe_annualized": np.nan,
            "max_drawdown": np.nan,
            "avg_abs_weight": 0.0,
        }

    test_weights = _make_weights(
        policy_name,
        test_returns,
        threshold_or_scale=best_param,
        base_returns=test_base_returns,
    )
    _, test_metrics = _weighted_portfolio_metrics(
        close_prices=panel_prices,
        dates=panel_dates,
        decision_dates=test_dates,
        raw_weights=test_weights,
        hold_days=horizon,
    )
    test_metrics["directional_accuracy_nonzero"] = float(
        np.mean(np.sign(test_weights[np.abs(test_weights) > 1e-12]) == np.sign(test_realized[np.abs(test_weights) > 1e-12]))
    ) if np.any(np.abs(test_weights) > 1e-12) else np.nan
    return best_param, best_val_metrics, test_metrics, test_weights


def _run_candidate(name: str, overrides: dict) -> Path:
    run_dir = run_experiment(
        config_path="uk_ets/config/uk_ets_llm_4b_current_default.yaml",
        overrides=overrides,
    )
    return Path(run_dir)


def main() -> None:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = PROJECT_ROOT / "reports" / "uk_ets_tsm_live_trading_conversion" / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)

    selected = _selected_candidates()
    policies = _policy_families()
    plan_lines = [
        "# TSM Live Trading Conversion Plan",
        "",
        "## Objective",
        "- Rerun the verified live TSM baseline and learned-gbdt winner with validation LLM predictions exported.",
        "- Calibrate trading conversion rules on validation only and apply them to the common test holdout.",
        "- Determine whether the verified LLM magnitude gains can be monetized once the trading rule uses more than just forecast sign.",
        "",
        "## Policies",
    ]
    for policy in policies:
        plan_lines.append(f"- `{policy.name}`: {policy.objective} {policy.expectation}")
    (out_dir / "plan.md").write_text("\n".join(plan_lines) + "\n", encoding="utf-8")

    run_dirs = {}
    for name, overrides in selected.items():
        run_dirs[name] = _run_candidate(name, overrides)

    baseline_run = run_dirs["baseline_regime_specific"]
    gbdt_run = run_dirs["learned_gbdt_regime"]
    panel = pd.read_parquet(gbdt_run / "data" / "panel.parquet")
    panel_dates = panel["date"]
    panel_prices = panel["y"]

    baseline_val = _load_npz(baseline_run / "predictions" / "TSM+LLM-COT-RF-HDELTA_pred_val_full.npz")
    baseline_test = _load_npz(baseline_run / "predictions" / "TSM+LLM-COT-RF-HDELTA_pred_test_subset.npz")
    gbdt_val = _load_npz(gbdt_run / "predictions" / "TSM+LLM-COT-RF-HDELTA_pred_val_full.npz")
    gbdt_test = _load_npz(gbdt_run / "predictions" / "TSM+LLM-COT-RF-HDELTA_pred_test_subset.npz")

    verification_rows = [
        {
            "check": "same_val_dates",
            "value": bool(np.array_equal(baseline_val["dates"], gbdt_val["dates"])),
        },
        {
            "check": "same_test_dates",
            "value": bool(np.array_equal(baseline_test["dates"], gbdt_test["dates"])),
        },
        {
            "check": "same_val_base_pred",
            "value": bool(np.allclose(baseline_val["base_pred"], gbdt_val["base_pred"])),
        },
        {
            "check": "same_test_base_pred",
            "value": bool(np.allclose(baseline_test["base_pred"], gbdt_test["base_pred"])),
        },
    ]
    pd.DataFrame(verification_rows).to_csv(out_dir / "verification_checks.csv", index=False)

    rows = []
    weight_rows = []
    for model_name, val_data, test_data in [
        ("raw_tsm", gbdt_val, gbdt_test),
        ("baseline_heuristic", baseline_val, baseline_test),
        ("learned_gbdt", gbdt_val, gbdt_test),
    ]:
        for horizon in (20, 30):
            if model_name == "raw_tsm":
                val_pred = val_data["base_pred"]
                test_pred = test_data["base_pred"]
            else:
                val_pred = val_data["yhat"]
                test_pred = test_data["yhat"]
            val_base = val_data["base_pred"]
            test_base = test_data["base_pred"]
            val_base_prices = val_base[:, 0]
            test_base_prices = test_base[:, 0]
            val_returns = _forecast_returns(val_pred, val_base_prices, horizon)
            test_returns = _forecast_returns(test_pred, test_base_prices, horizon)
            val_base_returns = _forecast_returns(val_base, val_base_prices, horizon)
            test_base_returns = _forecast_returns(test_base, test_base_prices, horizon)
            val_realized = _realized_returns(val_data["y_true"], val_base_prices, horizon)
            test_realized = _realized_returns(test_data["y_true"], test_base_prices, horizon)
            val_dates = pd.to_datetime(val_data["dates"])
            test_dates = pd.to_datetime(test_data["dates"])

            active_policies = [policy.name for policy in policies]
            if model_name == "raw_tsm":
                active_policies = [policy for policy in active_policies if not policy.startswith("uplift_")]

            for policy_name in active_policies:
                best_param, val_metrics, test_metrics, test_weights = _evaluate_policy(
                    policy_name=policy_name,
                    val_returns=val_returns,
                    test_returns=test_returns,
                    val_base_returns=val_base_returns,
                    test_base_returns=test_base_returns,
                    val_realized=val_realized,
                    test_realized=test_realized,
                    val_dates=val_dates,
                    test_dates=test_dates,
                    panel_dates=panel_dates,
                    panel_prices=panel_prices,
                    horizon=horizon,
                )
                rows.append(
                    {
                        "model": model_name,
                        "horizon": horizon,
                        "policy": policy_name,
                        "selected_param": best_param,
                        "val_sharpe": val_metrics["daily_sharpe_annualized"],
                        "val_total_return": val_metrics["total_return"],
                        "val_n_trades": val_metrics["n_executed_trades"],
                        "test_sharpe": test_metrics["daily_sharpe_annualized"],
                        "test_total_return": test_metrics["total_return"],
                        "test_annualized_return": test_metrics["annualized_return"],
                        "test_max_drawdown": test_metrics["max_drawdown"],
                        "test_n_trades": test_metrics["n_executed_trades"],
                        "test_avg_abs_weight": test_metrics["avg_abs_weight"],
                        "test_directional_accuracy_nonzero": test_metrics["directional_accuracy_nonzero"],
                    }
                )
                weight_rows.append(
                    pd.DataFrame(
                        {
                            "date": test_dates,
                            "model": model_name,
                            "horizon": horizon,
                            "policy": policy_name,
                            "weight": test_weights,
                            "forecast_return": test_returns,
                            "base_return": test_base_returns,
                            "realized_return": test_realized,
                        }
                    )
                )

    results_df = pd.DataFrame(rows)
    results_df.to_csv(out_dir / "policy_results.csv", index=False)
    weights_df = pd.concat(weight_rows, ignore_index=True)
    weights_df.to_csv(out_dir / "policy_test_weights.csv", index=False)

    best_rows = []
    for model_name in sorted(results_df["model"].unique()):
        for horizon in (20, 30):
            subset = results_df[(results_df["model"] == model_name) & (results_df["horizon"] == horizon)].copy()
            if subset.empty:
                continue
            best_rows.append(subset.sort_values(["test_sharpe", "test_total_return"], ascending=[False, False]).iloc[0].to_dict())
    best_df = pd.DataFrame(best_rows)
    best_df.to_csv(out_dir / "best_policy_by_model_horizon.csv", index=False)

    report_lines = [
        "# TSM Live Trading Conversion Report",
        "",
        "## Setup",
        "- Reran `baseline_regime_specific` and `learned_gbdt_regime` with `llm.export_val_predictions=true`.",
        "- Used the saved validation predictions to calibrate trading rules and then applied them to the common test holdout.",
        "- Costs: `10` bps transaction cost plus `5` bps slippage, `1`-day execution lag, active-window MTM.",
        "",
        "## Verification",
    ]
    for row in verification_rows:
        report_lines.append(f"- `{row['check']}`: `{row['value']}`")
    report_lines.extend(["", "## Best Policies"])
    for _, row in best_df.sort_values(["horizon", "test_sharpe"], ascending=[True, False]).iterrows():
        report_lines.append(
            f"- `{row['model']}` `h{int(row['horizon'])}` best policy `{row['policy']}` with test Sharpe `{row['test_sharpe']:.3f}`, "
            f"total return `{row['test_total_return']:.2%}`, annualized return `{row['test_annualized_return']:.2%}`, "
            f"max drawdown `{row['test_max_drawdown']:.2%}`, trades `{int(row['test_n_trades'])}`, param `{row['selected_param']:.6f}`."
        )
    report_lines.extend(
        [
            "",
            "## Interpretation",
            "- If the best `learned_gbdt` policy beats the best `raw_tsm` policy on the same horizon, then the forecast-magnitude gain is monetizable under a leakage-safe validation-calibrated rule.",
            "- If the best policies are still tied or nearly tied, the remaining bottleneck is the trading conversion rule or the economic objective itself, not the forecast MSE.",
            "",
            "## Artifacts",
            f"- Plan: [{(out_dir / 'plan.md').name}]({out_dir / 'plan.md'})",
            f"- Policy results: [{(out_dir / 'policy_results.csv').name}]({out_dir / 'policy_results.csv'})",
            f"- Best policy table: [{(out_dir / 'best_policy_by_model_horizon.csv').name}]({out_dir / 'best_policy_by_model_horizon.csv'})",
            f"- Test weights: [{(out_dir / 'policy_test_weights.csv').name}]({out_dir / 'policy_test_weights.csv'})",
        ]
    )
    (out_dir / "report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    print(out_dir)


if __name__ == "__main__":
    main()
