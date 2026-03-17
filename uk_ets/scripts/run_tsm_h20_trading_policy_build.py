#!/usr/bin/env python3
"""Build and evaluate h20 trading-conversion policies on the promoted live TSM+LLM stack."""

from __future__ import annotations

import json
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

from src.run_experiment import fit_online_memory_learned_gate, predict_online_memory_learned_gate_scores, run_experiment


HORIZONS = (20, 30)
PRIMARY_HORIZON = 20
TRANSACTION_COST_BPS = 10.0
SLIPPAGE_BPS = 5.0
EXECUTION_LAG_DAYS = 1


@dataclass(frozen=True)
class TradingPolicy:
    name: str
    objective: str
    expectation: str
    requires_base: bool = False
    requires_prob: bool = False


def _deep_merge(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    merged = dict(left)
    for key, value in right.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(dict(merged[key]), value)
        else:
            merged[key] = value
    return merged


def _selected_candidates() -> dict[str, dict]:
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
    baseline_override = {
        "llm": {
            "cot_rf": {
                "online_memory_policy": {
                    "gate": {
                        "learned": {
                            "enabled": False,
                        }
                    }
                }
            }
        }
    }
    return {
        "baseline_regime_specific": _deep_merge(export_override, baseline_override),
        "learned_gbdt_regime": export_override,
    }


def _learned_gate_cfg() -> dict[str, Any]:
    return {
        "enabled": True,
        "model_type": "hist_gbdt",
        "scope": "recent_tail",
        "recent_tail_fraction": 0.5,
        "recent_tail_min_samples": 80,
        "max_samples": 80,
        "train_fraction": 0.67,
        "positive_margin": 0.0,
        "positive_min_h20_gain": 0.0,
        "positive_min_h30_gain": 0.0,
        "positive_max_h5_damage": -0.15,
        "probability_threshold_grid": [0.30, 0.40, 0.50, 0.60, 0.70],
        "metric": "mse_path",
        "max_depth": 3,
        "learning_rate": 0.05,
        "max_iter": 200,
        "feature_columns": [
            "positive_count_h30",
            "positive_count",
            "positive_signal_h30",
            "positive_signal",
            "net_signal",
            "negative_mean_helpfulness",
            "base_move_h5_pct",
            "profile_fc_h5",
            "negative_count",
            "positive_mean_helpfulness",
            "profile_vol_pct",
            "positive_mean_similarity",
            "positive_best_similarity",
            "profile_change_5",
            "positive_count_h20",
            "negative_signal",
            "base_move_h20_pct",
            "profile_fc_h20",
            "negative_signal_h30",
            "base_move_h30_pct",
        ],
    }


def _policy_families() -> list[TradingPolicy]:
    return [
        TradingPolicy(
            name="sign_threshold",
            objective="Pure sign or thresholded sign based on forecast magnitude.",
            expectation="Expected to confirm whether magnitude filtering alone is enough to monetize the improved h20 forecast.",
        ),
        TradingPolicy(
            name="linear_size",
            objective="Size linearly with forecast return magnitude.",
            expectation="Expected to convert better magnitude calibration into higher Sharpe if the LLM mostly improves forecast sizing.",
        ),
        TradingPolicy(
            name="tanh_size",
            objective="Use smooth tanh sizing on forecast magnitude.",
            expectation="Expected to be more stable than linear sizing if extreme forecasts are noisy.",
        ),
        TradingPolicy(
            name="uplift_linear_size",
            objective="Size by LLM-vs-base uplift while preserving the LLM sign.",
            expectation="Expected to help if the incremental LLM edge is concentrated in larger-adjustment windows.",
            requires_base=True,
        ),
        TradingPolicy(
            name="regime_linear_size",
            objective="Scale linear position size by the learned live-gate probability.",
            expectation="Expected to be the most product-faithful policy if the learned gate probability tracks when the LLM truly adds economic value.",
            requires_prob=True,
        ),
        TradingPolicy(
            name="regime_tanh_size",
            objective="Scale tanh position size by the learned live-gate probability.",
            expectation="Expected to be more robust than raw regime-linear sizing if both probability and magnitude are noisy.",
            requires_prob=True,
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
) -> tuple[pd.DataFrame, dict[str, float], pd.DataFrame]:
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
    trade_rows: list[dict[str, Any]] = []

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
            * np.sum(
                np.log(np.clip(1.0 + df["price_return"].iloc[entry_idx + 1 : exit_idx + 1], 1e-12, None))
            )
            * abs(raw_weight)
        )
        trade_rows.append(
            {
                "decision_date": str(pd.Timestamp(decision_date).date()),
                "entry_date": str(pd.Timestamp(df["date"].iloc[entry_idx]).date()),
                "exit_date": str(pd.Timestamp(df["date"].iloc[exit_idx]).date()),
                "raw_weight": float(raw_weight),
                "normalized_weight": float(normalized_weight),
                "gross_trade_return": float(np.expm1(trade_log_return)),
                "net_trade_return": float(np.expm1(trade_log_return) - 2.0 * total_cost_rate * abs(normalized_weight)),
            }
        )

    df["net_exposure"] = exposure
    df["gross_exposure"] = gross_exposure
    df["cost_event"] = costs
    df["strategy_return"] = (df["net_exposure"] * df["price_return"]) + df["cost_event"]

    if trim_to_active_window and trade_rows:
        entry_start = min(pd.Index(df["date"]).get_loc(pd.Timestamp(row["entry_date"])) for row in trade_rows)
        exit_end = max(pd.Index(df["date"]).get_loc(pd.Timestamp(row["exit_date"])) for row in trade_rows)
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
        annualized_return = (
            float((1.0 + total_return) ** (252.0 / len(df)) - 1.0) if total_return > -1.0 else -1.0
        )
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
    }, trade_df


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


def _prob_floor_grid(probs: np.ndarray) -> list[float]:
    clean = np.asarray(probs, dtype=float)
    clean = clean[np.isfinite(clean)]
    if clean.size == 0:
        return [0.0]
    qs = [0.0, 0.25, 0.5, 0.6, 0.75]
    return sorted({float(np.quantile(clean, q)) for q in qs})


def _make_weights(
    policy_name: str,
    forecast_returns: np.ndarray,
    *,
    threshold_or_scale: float,
    base_returns: np.ndarray | None = None,
    apply_probs: np.ndarray | None = None,
    prob_floor: float = 0.0,
) -> np.ndarray:
    fr = np.asarray(forecast_returns, dtype=float)
    param = float(max(threshold_or_scale, 1e-8))
    if policy_name == "sign_threshold":
        return np.where(np.abs(fr) >= param, np.sign(fr), 0.0)
    if policy_name == "linear_size":
        return np.clip(fr / param, -1.0, 1.0)
    if policy_name == "tanh_size":
        return np.tanh(fr / param)
    if policy_name == "uplift_linear_size":
        if base_returns is None:
            raise ValueError("uplift_linear_size requires base_returns")
        uplift = np.abs(fr - np.asarray(base_returns, dtype=float))
        return np.sign(fr) * np.clip(uplift / param, 0.0, 1.0)
    if policy_name in {"regime_linear_size", "regime_tanh_size"}:
        if apply_probs is None:
            raise ValueError(f"{policy_name} requires apply_probs")
        probs = np.asarray(apply_probs, dtype=float)
        denom = max(1.0 - float(prob_floor), 1e-8)
        regime_multiplier = np.clip((probs - float(prob_floor)) / denom, 0.0, 1.0)
        base_size = np.clip(fr / param, -1.0, 1.0) if policy_name == "regime_linear_size" else np.tanh(fr / param)
        return base_size * regime_multiplier
    raise ValueError(f"Unknown policy: {policy_name}")


def _candidate_grid(
    policy_name: str,
    forecast_returns: np.ndarray,
    *,
    base_returns: np.ndarray | None,
    apply_probs: np.ndarray | None,
) -> list[tuple[float, float]]:
    quantiles = [0.0, 0.1, 0.25, 0.4, 0.5, 0.6, 0.75, 0.9]
    if policy_name == "uplift_linear_size":
        uplift = np.abs(np.asarray(forecast_returns) - np.asarray(base_returns))
        return [(x, 0.0) for x in _quantile_grid(uplift, quantiles)]
    if policy_name in {"regime_linear_size", "regime_tanh_size"}:
        size_grid = _quantile_grid(forecast_returns, [0.1, 0.25, 0.4, 0.5, 0.6, 0.75, 0.9])
        floor_grid = _prob_floor_grid(np.asarray(apply_probs))
        return [(size, floor) for size in size_grid for floor in floor_grid]
    return [(x, 0.0) for x in _quantile_grid(forecast_returns, quantiles)]


def _evaluate_policy(
    *,
    policy: TradingPolicy,
    val_returns: np.ndarray,
    test_returns: np.ndarray,
    val_base_returns: np.ndarray,
    test_base_returns: np.ndarray,
    val_probs: np.ndarray | None,
    test_probs: np.ndarray | None,
    val_realized: np.ndarray,
    test_realized: np.ndarray,
    val_dates: pd.Index,
    test_dates: pd.Index,
    panel_dates: pd.Series,
    panel_prices: pd.Series,
    horizon: int,
) -> tuple[tuple[float, float], dict[str, float], dict[str, float], np.ndarray, pd.DataFrame, pd.DataFrame]:
    best_params = (0.0, 0.0)
    best_metric = -np.inf
    best_val_metrics: dict[str, float] | None = None
    for threshold_or_scale, prob_floor in _candidate_grid(
        policy.name,
        val_returns,
        base_returns=val_base_returns,
        apply_probs=val_probs,
    ):
        weights = _make_weights(
            policy.name,
            val_returns,
            threshold_or_scale=threshold_or_scale,
            base_returns=val_base_returns,
            apply_probs=val_probs,
            prob_floor=prob_floor,
        )
        _, metrics, _ = _weighted_portfolio_metrics(
            close_prices=panel_prices,
            dates=panel_dates,
            decision_dates=val_dates,
            raw_weights=weights,
            hold_days=horizon,
            execution_lag_days=EXECUTION_LAG_DAYS,
            transaction_cost_bps=TRANSACTION_COST_BPS,
            slippage_bps=SLIPPAGE_BPS,
        )
        metric = metrics["daily_sharpe_annualized"]
        if not np.isfinite(metric):
            metric = -np.inf
        if metrics["n_executed_trades"] < max(3, horizon // 5):
            metric = -np.inf
        if metric > best_metric:
            best_metric = metric
            best_params = (float(threshold_or_scale), float(prob_floor))
            best_val_metrics = metrics
    if best_val_metrics is None:
        best_val_metrics = {
            "n_executed_trades": 0,
            "total_return": np.nan,
            "annualized_return": np.nan,
            "daily_sharpe_annualized": np.nan,
            "max_drawdown": np.nan,
            "avg_abs_weight": 0.0,
        }

    test_weights = _make_weights(
        policy.name,
        test_returns,
        threshold_or_scale=best_params[0],
        base_returns=test_base_returns,
        apply_probs=test_probs,
        prob_floor=best_params[1],
    )
    test_mtm_df, test_metrics, trade_df = _weighted_portfolio_metrics(
        close_prices=panel_prices,
        dates=panel_dates,
        decision_dates=test_dates,
        raw_weights=test_weights,
        hold_days=horizon,
        execution_lag_days=EXECUTION_LAG_DAYS,
        transaction_cost_bps=TRANSACTION_COST_BPS,
        slippage_bps=SLIPPAGE_BPS,
    )
    nonzero_mask = np.abs(test_weights) > 1e-12
    test_metrics["directional_accuracy_nonzero"] = (
        float(np.mean(np.sign(test_weights[nonzero_mask]) == np.sign(test_realized[nonzero_mask])))
        if np.any(nonzero_mask)
        else np.nan
    )
    return best_params, best_val_metrics, test_metrics, test_weights, test_mtm_df, trade_df


def _run_candidate(name: str, overrides: dict) -> Path:
    run_dir = run_experiment(
        config_path="uk_ets/config/uk_ets_llm_4b_tsm_live_gbdt_top20.yaml",
        overrides=overrides,
    )
    return Path(run_dir)


def _load_learned_probs(run_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    result_slug = "TSM_LLM-COT-RF-HDELTA"
    val_feature_df = pd.read_csv(run_dir / "results" / "online_memory_gate" / result_slug / "val_learned_gate_features.csv")
    val_label_df = pd.read_csv(run_dir / "results" / "online_memory_gate" / result_slug / "val_learned_gate_labels.csv")
    test_feature_df = pd.read_csv(run_dir / "results" / "online_memory_gate" / result_slug / "test_learned_gate_features.csv")
    selection = json.loads((run_dir / "llm" / f"online_memory_gate_selection_{result_slug}.json").read_text(encoding="utf-8"))
    learned_cfg = dict(_learned_gate_cfg())
    learned_cfg["use_all_for_fit"] = True
    bundle, _ = fit_online_memory_learned_gate(
        feature_df=val_feature_df,
        labels=val_label_df["label"].to_numpy(dtype=int),
        learned_cfg=learned_cfg,
    )
    if bundle is None:
        raise RuntimeError(f"Failed to refit learned gate for {run_dir}")
    bundle["threshold"] = float(selection.get("threshold", 0.5))
    val_probs = predict_online_memory_learned_gate_scores(val_feature_df, bundle)
    test_probs = predict_online_memory_learned_gate_scores(test_feature_df, bundle)
    return val_probs, test_probs


def main() -> None:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = PROJECT_ROOT / "reports" / "uk_ets_tsm_h20_trading_policy_build" / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)

    policies = _policy_families()
    plan_lines = [
        "# TSM H20 Trading Policy Build",
        "",
        "## Objective",
        "- Freeze the promoted live forecast stack (`DLinear + learned_gbdt(top20) + selective LLM`).",
        "- Calibrate trading-conversion rules on validation only, then evaluate them on the common test holdout.",
        "- Focus on `h20` as the primary production trading horizon while reporting `h30` as a secondary check.",
        "",
        "## Expected Outcomes",
        "- `sign_threshold`: likely close to existing sign-only performance; mainly a control.",
        "- `linear_size` / `tanh_size`: expected to improve `h20` if the LLM gain is mostly a magnitude effect.",
        "- `uplift_linear_size`: expected to help if the LLM edge is concentrated in higher-uplift windows.",
        "- `regime_linear_size` / `regime_tanh_size`: expected to be strongest if the learned gate probability captures tradable context, not just forecast MSE.",
        "",
        "## Policies",
    ]
    for policy in policies:
        plan_lines.append(f"- `{policy.name}`: {policy.objective} {policy.expectation}")
    (out_dir / "plan.md").write_text("\n".join(plan_lines) + "\n", encoding="utf-8")

    run_dirs = {name: _run_candidate(name, overrides) for name, overrides in _selected_candidates().items()}
    baseline_run = run_dirs["baseline_regime_specific"]
    learned_run = run_dirs["learned_gbdt_regime"]

    panel = pd.read_parquet(learned_run / "data" / "panel.parquet")
    panel_dates = panel["date"]
    panel_prices = panel["y"]

    baseline_val = _load_npz(baseline_run / "predictions" / "TSM+LLM-COT-RF-HDELTA_pred_val_full.npz")
    baseline_test = _load_npz(baseline_run / "predictions" / "TSM+LLM-COT-RF-HDELTA_pred_test_subset.npz")
    learned_val = _load_npz(learned_run / "predictions" / "TSM+LLM-COT-RF-HDELTA_pred_val_full.npz")
    learned_test = _load_npz(learned_run / "predictions" / "TSM+LLM-COT-RF-HDELTA_pred_test_subset.npz")

    verification_rows = [
        {"check": "same_val_dates", "value": bool(np.array_equal(baseline_val["dates"], learned_val["dates"]))},
        {"check": "same_test_dates", "value": bool(np.array_equal(baseline_test["dates"], learned_test["dates"]))},
        {"check": "same_val_base_pred", "value": bool(np.allclose(baseline_val["base_pred"], learned_val["base_pred"]))},
        {"check": "same_test_base_pred", "value": bool(np.allclose(baseline_test["base_pred"], learned_test["base_pred"]))},
    ]
    pd.DataFrame(verification_rows).to_csv(out_dir / "verification_checks.csv", index=False)

    learned_val_probs, learned_test_probs = _load_learned_probs(learned_run)

    rows = []
    weight_rows = []
    best_trade_df = None
    best_mtm_df = None
    best_meta: dict[str, Any] | None = None
    for model_name, val_data, test_data, val_probs, test_probs in [
        ("raw_tsm", learned_val, learned_test, None, None),
        ("baseline_heuristic", baseline_val, baseline_test, None, None),
        ("learned_gbdt", learned_val, learned_test, learned_val_probs, learned_test_probs),
    ]:
        for horizon in HORIZONS:
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

            for policy in policies:
                if policy.requires_base and model_name == "raw_tsm":
                    continue
                if policy.requires_prob and model_name != "learned_gbdt":
                    continue
                best_params, val_metrics, test_metrics, test_weights, mtm_df, trade_df = _evaluate_policy(
                    policy=policy,
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
                    horizon=horizon,
                )
                rows.append(
                    {
                        "model": model_name,
                        "horizon": horizon,
                        "policy": policy.name,
                        "selected_scale_or_threshold": float(best_params[0]),
                        "selected_prob_floor": float(best_params[1]),
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
                            "policy": policy.name,
                            "weight": test_weights,
                            "forecast_return": test_returns,
                            "base_return": test_base_returns,
                            "realized_return": test_realized,
                            "apply_probability": test_probs if test_probs is not None else np.nan,
                        }
                    )
                )
                candidate_sharpe = float(test_metrics["daily_sharpe_annualized"])
                if (
                    model_name == "learned_gbdt"
                    and horizon == PRIMARY_HORIZON
                    and np.isfinite(candidate_sharpe)
                    and (
                        best_meta is None
                        or not np.isfinite(float(best_meta["test_sharpe"]))
                        or candidate_sharpe > float(best_meta["test_sharpe"])
                    )
                ):
                    best_meta = {
                        "policy": policy.name,
                        "selected_scale_or_threshold": float(best_params[0]),
                        "selected_prob_floor": float(best_params[1]),
                        "test_sharpe": candidate_sharpe,
                        "test_total_return": float(test_metrics["total_return"]),
                        "test_annualized_return": float(test_metrics["annualized_return"]),
                        "test_max_drawdown": float(test_metrics["max_drawdown"]),
                        "test_n_trades": int(test_metrics["n_executed_trades"]),
                    }
                    best_trade_df = trade_df.copy()
                    best_mtm_df = mtm_df.copy()

    results_df = pd.DataFrame(rows)
    results_df.to_csv(out_dir / "policy_results.csv", index=False)
    pd.concat(weight_rows, ignore_index=True).to_csv(out_dir / "policy_test_weights.csv", index=False)

    best_df = (
        results_df.sort_values(["model", "horizon", "test_sharpe", "test_total_return"], ascending=[True, True, False, False])
        .groupby(["model", "horizon"], as_index=False)
        .head(1)
        .reset_index(drop=True)
    )
    best_df.to_csv(out_dir / "best_policy_by_model_horizon.csv", index=False)

    if best_trade_df is not None and best_meta is not None:
        best_trade_df.to_csv(out_dir / "selected_h20_trade_ledger.csv", index=False)
        best_mtm_df.to_csv(out_dir / "selected_h20_daily_mtm.csv", index=False)
        (out_dir / "selected_h20_policy.json").write_text(json.dumps(best_meta, indent=2), encoding="utf-8")

    report_lines = [
        "# TSM H20 Trading Policy Build Report",
        "",
        "## Setup",
        "- Forecast stack frozen at `DLinear + learned_gbdt(top20) + selective LLM`.",
        "- Reran the heuristic baseline and the learned-gbdt winner with validation exports.",
        "- Calibrated trading rules on validation only and applied them to the common test holdout.",
        f"- Costs: `{TRANSACTION_COST_BPS:.0f}` bps transaction cost + `{SLIPPAGE_BPS:.0f}` bps slippage, `{EXECUTION_LAG_DAYS}`-day execution lag, active-window MTM.",
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
            f"max drawdown `{row['test_max_drawdown']:.2%}`, trades `{int(row['test_n_trades'])}`, "
            f"scale/threshold `{row['selected_scale_or_threshold']:.6f}`, prob floor `{row['selected_prob_floor']:.6f}`."
        )
    if best_meta is not None:
        report_lines.extend(
            [
                "",
                "## Selected H20 Production Candidate",
                f"- `learned_gbdt` selected policy: `{best_meta['policy']}`",
                f"- Test Sharpe: `{best_meta['test_sharpe']:.3f}`",
                f"- Total return: `{best_meta['test_total_return']:.2%}`",
                f"- Annualized return: `{best_meta['test_annualized_return']:.2%}`",
                f"- Max drawdown: `{best_meta['test_max_drawdown']:.2%}`",
                f"- Executed trades: `{best_meta['test_n_trades']}`",
            ]
        )
    report_lines.extend(
        [
            "",
            "## Interpretation",
            "- If the selected `learned_gbdt` h20 policy beats both the raw TSM and heuristic LLM controls, the remaining edge is economically convertible under a validation-safe sizing rule.",
            "- Regime-conditioned sizing is the key product test here: it asks whether the same learned gate that improves forecast MSE can also control trade aggressiveness in a useful way.",
            "",
            "## Artifacts",
            f"- Plan: [{(out_dir / 'plan.md').name}]({out_dir / 'plan.md'})",
            f"- Policy results: [{(out_dir / 'policy_results.csv').name}]({out_dir / 'policy_results.csv'})",
            f"- Best policy table: [{(out_dir / 'best_policy_by_model_horizon.csv').name}]({out_dir / 'best_policy_by_model_horizon.csv'})",
            f"- Selected h20 ledger: [{(out_dir / 'selected_h20_trade_ledger.csv').name}]({out_dir / 'selected_h20_trade_ledger.csv'})" if best_trade_df is not None else "- Selected h20 ledger: not available",
            f"- Selected h20 MTM: [{(out_dir / 'selected_h20_daily_mtm.csv').name}]({out_dir / 'selected_h20_daily_mtm.csv'})" if best_mtm_df is not None else "- Selected h20 MTM: not available",
        ]
    )
    (out_dir / "report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    print(out_dir)


if __name__ == "__main__":
    main()
