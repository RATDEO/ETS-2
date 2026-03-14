"""Return and portfolio metrics for forecast evaluation."""

from typing import Any, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .metrics import compute_metrics_by_horizon


def compute_log_return_paths(
    prices: np.ndarray,
    base_prices: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute step and cumulative log returns from price paths.

    Args:
        prices: Forecast or actual prices (batch, pred_len)
        base_prices: Last observed price for each sample (batch,)

    Returns:
        Tuple of (step_log_returns, cumulative_log_returns)
    """
    if prices.ndim != 2:
        raise ValueError("prices must be a 2D array")
    if base_prices.ndim != 1 or base_prices.shape[0] != prices.shape[0]:
        raise ValueError("base_prices must be 1D with same batch size as prices")

    eps = 1e-8
    prices_safe = np.clip(prices, eps, None)
    base_safe = np.clip(base_prices, eps, None)

    step_returns = np.zeros_like(prices_safe)
    step_returns[:, 0] = np.log(prices_safe[:, 0] / base_safe)
    if prices_safe.shape[1] > 1:
        step_returns[:, 1:] = np.log(prices_safe[:, 1:] / prices_safe[:, :-1])

    cum_returns = np.cumsum(step_returns, axis=1)
    return step_returns, cum_returns


def compute_drift_metrics_by_horizon(
    y_true_prices: np.ndarray,
    y_pred_prices: np.ndarray,
    base_prices: np.ndarray,
    horizons: List[int],
    metrics: Optional[List[str]] = None
) -> pd.DataFrame:
    """Compute error metrics on cumulative log returns at each horizon."""
    _, true_cum = compute_log_return_paths(y_true_prices, base_prices)
    _, pred_cum = compute_log_return_paths(y_pred_prices, base_prices)
    return compute_metrics_by_horizon(true_cum, pred_cum, horizons, metrics)


def _compute_non_overlap_backtest_metrics(
    signal: np.ndarray,
    true_ret: np.ndarray,
    horizon: int,
    start_offset: int = 0,
    transaction_cost_bps: float = 0.0,
) -> dict[str, Any]:
    """Compute one non-overlapping backtest path for a given start offset."""
    schedule_idx = np.arange(start_offset, len(signal), horizon, dtype=int)
    scheduled_signal = signal[schedule_idx]
    scheduled_true_ret = true_ret[schedule_idx]
    scheduled_strategy_log_ret = scheduled_signal * scheduled_true_ret
    scheduled_strategy_simple_ret = np.expm1(scheduled_strategy_log_ret)

    cost_rate = max(float(transaction_cost_bps), 0.0) / 10000.0
    if cost_rate > 0.0:
        trade_multiplier = np.where(
            scheduled_signal != 0,
            (1.0 + scheduled_strategy_simple_ret) * (1.0 - cost_rate) ** 2,
            1.0,
        )
        scheduled_strategy_simple_ret_net = trade_multiplier - 1.0
    else:
        scheduled_strategy_simple_ret_net = scheduled_strategy_simple_ret.copy()

    non_overlap_trade_mask = scheduled_signal != 0
    n_trades_non_overlap = int(non_overlap_trade_mask.sum())
    trade_rate_non_overlap = (
        float(n_trades_non_overlap / len(schedule_idx)) if len(schedule_idx) else np.nan
    )

    if scheduled_strategy_simple_ret.size:
        equity_curve = np.cumprod(1.0 + scheduled_strategy_simple_ret)
        total_return_non_overlap = float(equity_curve[-1] - 1.0)
        running_peak = np.maximum.accumulate(equity_curve)
        max_drawdown_non_overlap = float(np.max(1.0 - (equity_curve / running_peak)))
    else:
        total_return_non_overlap = np.nan
        max_drawdown_non_overlap = np.nan

    if scheduled_strategy_simple_ret_net.size:
        equity_curve_net = np.cumprod(1.0 + scheduled_strategy_simple_ret_net)
        total_return_non_overlap_cost_adj = float(equity_curve_net[-1] - 1.0)
        running_peak_net = np.maximum.accumulate(equity_curve_net)
        max_drawdown_non_overlap_cost_adj = float(
            np.max(1.0 - (equity_curve_net / running_peak_net))
        )
    else:
        total_return_non_overlap_cost_adj = np.nan
        max_drawdown_non_overlap_cost_adj = np.nan

    if len(schedule_idx):
        total_days = int(len(schedule_idx) * horizon)
        if total_return_non_overlap <= -1.0:
            annualized_return_non_overlap = -1.0
        else:
            annualized_return_non_overlap = float(
                (1.0 + total_return_non_overlap) ** (252.0 / max(total_days, 1)) - 1.0
            )
        if total_return_non_overlap_cost_adj <= -1.0:
            annualized_return_non_overlap_cost_adj = -1.0
        else:
            annualized_return_non_overlap_cost_adj = float(
                (1.0 + total_return_non_overlap_cost_adj) ** (252.0 / max(total_days, 1)) - 1.0
            )
    else:
        annualized_return_non_overlap = np.nan
        annualized_return_non_overlap_cost_adj = np.nan

    vol_non_overlap = float(np.std(scheduled_strategy_simple_ret))
    if vol_non_overlap > 0.0:
        sharpe_non_overlap = float(
            (np.mean(scheduled_strategy_simple_ret) / vol_non_overlap) * np.sqrt(252.0 / float(horizon))
        )
    else:
        sharpe_non_overlap = np.nan

    vol_non_overlap_cost_adj = float(np.std(scheduled_strategy_simple_ret_net))
    if vol_non_overlap_cost_adj > 0.0:
        sharpe_non_overlap_cost_adj = float(
            (np.mean(scheduled_strategy_simple_ret_net) / vol_non_overlap_cost_adj)
            * np.sqrt(252.0 / float(horizon))
        )
    else:
        sharpe_non_overlap_cost_adj = np.nan

    hit_rate_non_overlap = (
        float(
            np.mean(
                np.sign(scheduled_true_ret[non_overlap_trade_mask])
                == scheduled_signal[non_overlap_trade_mask]
            )
        )
        if n_trades_non_overlap > 0
        else np.nan
    )

    return {
        "n_periods": int(len(schedule_idx)),
        "n_trades_non_overlap": n_trades_non_overlap,
        "trade_rate_non_overlap": trade_rate_non_overlap,
        "mean_return_non_overlap": float(np.mean(scheduled_strategy_simple_ret))
        if scheduled_strategy_simple_ret.size
        else np.nan,
        "volatility_non_overlap": vol_non_overlap,
        "sharpe_non_overlap": sharpe_non_overlap,
        "hit_rate_non_overlap": hit_rate_non_overlap,
        "total_return_non_overlap": total_return_non_overlap,
        "annualized_return_non_overlap": annualized_return_non_overlap,
        "max_drawdown_non_overlap": max_drawdown_non_overlap,
        "mean_return_non_overlap_cost_adj": float(np.mean(scheduled_strategy_simple_ret_net))
        if scheduled_strategy_simple_ret_net.size
        else np.nan,
        "volatility_non_overlap_cost_adj": vol_non_overlap_cost_adj,
        "sharpe_non_overlap_cost_adj": sharpe_non_overlap_cost_adj,
        "total_return_non_overlap_cost_adj": total_return_non_overlap_cost_adj,
        "annualized_return_non_overlap_cost_adj": annualized_return_non_overlap_cost_adj,
        "max_drawdown_non_overlap_cost_adj": max_drawdown_non_overlap_cost_adj,
    }


def _mean_or_nan(values: list[float]) -> float:
    """Average a list of floats, returning NaN for an empty list."""
    return float(np.mean(values)) if values else np.nan


def compute_long_short_portfolio_metrics(
    y_true_prices: np.ndarray,
    y_pred_prices: np.ndarray,
    base_prices: np.ndarray,
    horizons: List[int],
    signal_threshold: float = 0.0,
    average_non_overlap_offsets: bool = True,
    transaction_cost_bps: float = 0.0,
) -> pd.DataFrame:
    """
    Compute long-short portfolio metrics based on predicted return direction.

    Strategy: go long if predicted return > threshold, short if < -threshold,
    otherwise stay flat.
    """
    _, true_cum = compute_log_return_paths(y_true_prices, base_prices)
    _, pred_cum = compute_log_return_paths(y_pred_prices, base_prices)

    rows = []
    pred_len = y_true_prices.shape[1]
    for h in horizons:
        h_idx = h - 1
        if h_idx >= pred_len:
            continue
        pred_ret = pred_cum[:, h_idx]
        true_ret = true_cum[:, h_idx]

        signal = np.where(
            pred_ret > signal_threshold, 1,
            np.where(pred_ret < -signal_threshold, -1, 0)
        )
        strategy_ret = signal * true_ret
        trade_mask = signal != 0
        n_trades = int(trade_mask.sum())

        mean_ret = float(np.mean(strategy_ret))
        vol = float(np.std(strategy_ret))
        sharpe = float(mean_ret / vol) if vol > 0 else np.nan
        hit_rate = (
            float(np.mean(np.sign(true_ret[trade_mask]) == signal[trade_mask]))
            if n_trades > 0
            else np.nan
        )
        trade_rate = float(n_trades / len(signal)) if len(signal) else np.nan

        # Add non-overlapping horizon-h backtests. The legacy metrics keep the
        # offset-0 schedule for continuity; the new offset-average metrics reduce
        # start-date sensitivity by averaging across all valid offsets.
        offset0_metrics = _compute_non_overlap_backtest_metrics(
            signal=signal,
            true_ret=true_ret,
            horizon=h,
            start_offset=0,
            transaction_cost_bps=transaction_cost_bps,
        )

        offset_metrics = [offset0_metrics]
        if average_non_overlap_offsets and h > 1:
            offset_metrics = [
                _compute_non_overlap_backtest_metrics(
                    signal=signal,
                    true_ret=true_ret,
                    horizon=h,
                    start_offset=start_offset,
                    transaction_cost_bps=transaction_cost_bps,
                )
                for start_offset in range(h)
            ]

        rows.append(
            {
                "horizon": h,
                "mean_return": mean_ret,
                "volatility": vol,
                "sharpe": sharpe,
                "hit_rate": hit_rate,
                "trade_rate": trade_rate,
                "n_trades": n_trades,
                "transaction_cost_bps": float(transaction_cost_bps),
                "offset_count_non_overlap": int(len(offset_metrics)),
                **offset0_metrics,
                "mean_return_non_overlap_offset_avg": _mean_or_nan(
                    [float(m["mean_return_non_overlap"]) for m in offset_metrics]
                ),
                "volatility_non_overlap_offset_avg": _mean_or_nan(
                    [float(m["volatility_non_overlap"]) for m in offset_metrics]
                ),
                "sharpe_non_overlap_offset_avg": _mean_or_nan(
                    [float(m["sharpe_non_overlap"]) for m in offset_metrics]
                ),
                "hit_rate_non_overlap_offset_avg": _mean_or_nan(
                    [float(m["hit_rate_non_overlap"]) for m in offset_metrics]
                ),
                "trade_rate_non_overlap_offset_avg": _mean_or_nan(
                    [float(m["trade_rate_non_overlap"]) for m in offset_metrics]
                ),
                "n_trades_non_overlap_offset_avg": _mean_or_nan(
                    [float(m["n_trades_non_overlap"]) for m in offset_metrics]
                ),
                "total_return_non_overlap_offset_avg": _mean_or_nan(
                    [float(m["total_return_non_overlap"]) for m in offset_metrics]
                ),
                "annualized_return_non_overlap_offset_avg": _mean_or_nan(
                    [float(m["annualized_return_non_overlap"]) for m in offset_metrics]
                ),
                "max_drawdown_non_overlap_offset_avg": _mean_or_nan(
                    [float(m["max_drawdown_non_overlap"]) for m in offset_metrics]
                ),
                "mean_return_non_overlap_cost_adj_offset_avg": _mean_or_nan(
                    [float(m["mean_return_non_overlap_cost_adj"]) for m in offset_metrics]
                ),
                "volatility_non_overlap_cost_adj_offset_avg": _mean_or_nan(
                    [float(m["volatility_non_overlap_cost_adj"]) for m in offset_metrics]
                ),
                "sharpe_non_overlap_cost_adj_offset_avg": _mean_or_nan(
                    [float(m["sharpe_non_overlap_cost_adj"]) for m in offset_metrics]
                ),
                "total_return_non_overlap_cost_adj_offset_avg": _mean_or_nan(
                    [float(m["total_return_non_overlap_cost_adj"]) for m in offset_metrics]
                ),
                "annualized_return_non_overlap_cost_adj_offset_avg": _mean_or_nan(
                    [float(m["annualized_return_non_overlap_cost_adj"]) for m in offset_metrics]
                ),
                "max_drawdown_non_overlap_cost_adj_offset_avg": _mean_or_nan(
                    [float(m["max_drawdown_non_overlap_cost_adj"]) for m in offset_metrics]
                ),
                "total_return_non_overlap_offset_min": _mean_or_nan(
                    [float(np.min([m["total_return_non_overlap"] for m in offset_metrics]))]
                ),
                "total_return_non_overlap_offset_max": _mean_or_nan(
                    [float(np.max([m["total_return_non_overlap"] for m in offset_metrics]))]
                ),
                "total_return_non_overlap_cost_adj_offset_min": _mean_or_nan(
                    [float(np.min([m["total_return_non_overlap_cost_adj"] for m in offset_metrics]))]
                ),
                "total_return_non_overlap_cost_adj_offset_max": _mean_or_nan(
                    [float(np.max([m["total_return_non_overlap_cost_adj"] for m in offset_metrics]))]
                ),
            }
        )

    return pd.DataFrame(rows).set_index("horizon")


def compute_primary_horizon_signals(
    y_pred_prices: np.ndarray,
    base_prices: np.ndarray,
    horizon: int,
    signal_threshold: float = 0.0,
) -> np.ndarray:
    """Convert horizon-level forecast returns into {-1, 0, +1} trade signals."""
    if y_pred_prices.ndim != 2:
        raise ValueError("y_pred_prices must be a 2D array")
    if base_prices.ndim != 1 or base_prices.shape[0] != y_pred_prices.shape[0]:
        raise ValueError("base_prices must be 1D with same batch size as y_pred_prices")
    h_idx = int(horizon) - 1
    if h_idx < 0 or h_idx >= y_pred_prices.shape[1]:
        raise ValueError(f"horizon {horizon} is out of range for prediction length {y_pred_prices.shape[1]}")

    base_safe = np.clip(base_prices.astype(float), 1e-8, None)
    forecast_returns = y_pred_prices[:, h_idx].astype(float) / base_safe - 1.0
    threshold = float(signal_threshold)
    return np.where(
        forecast_returns > threshold,
        1,
        np.where(forecast_returns < -threshold, -1, 0),
    ).astype(int)


def assess_llm_training_cutoff(
    prediction_dates: Sequence[Any],
    cutoff_date: str = "2025-01-01",
) -> dict[str, Any]:
    """Assess whether a prediction set sits before or after a presumed LLM training cutoff."""
    dates = pd.to_datetime(pd.Index(prediction_dates), errors="coerce")
    dates = dates[dates.notna()].sort_values()
    cutoff = pd.Timestamp(cutoff_date)

    if len(dates) == 0:
        return {
            "cutoff_date": str(cutoff.date()),
            "n_predictions": 0,
            "n_pre_cutoff": 0,
            "n_post_cutoff": 0,
            "status": "unknown",
            "note": (
                "No valid prediction dates were available, so the LLM training-cutoff "
                "contamination note could not be assessed."
            ),
        }

    n_pre = int(np.sum(dates < cutoff))
    n_post = int(np.sum(dates >= cutoff))
    first_date = pd.Timestamp(dates.min())
    last_date = pd.Timestamp(dates.max())

    if n_pre == 0:
        status = "post_cutoff_only"
        note = (
            f"All prediction-set dates are on or after {cutoff.date()}. Relative to the assumed "
            "LLM training cutoff around early 2025, this segment is the cleaner regime for "
            "assessing whether results could reflect memorized pretraining exposure."
        )
    elif n_post == 0:
        status = "pre_cutoff_only"
        note = (
            f"All prediction-set dates fall before {cutoff.date()}. For this segment, LLM training "
            "contamination cannot be ruled out confidently because the underlying market period may "
            "overlap with the model's approximate pretraining cutoff."
        )
    else:
        status = "crosses_cutoff"
        note = (
            f"This prediction set crosses the assumed LLM training cutoff at {cutoff.date()}: "
            f"{n_pre} dates are earlier and {n_post} are later. Interpret pre-2025 results as "
            "potentially training-contaminated, and post-2025 results as materially cleaner."
        )

    return {
        "cutoff_date": str(cutoff.date()),
        "first_prediction_date": str(first_date.date()),
        "last_prediction_date": str(last_date.date()),
        "n_predictions": int(len(dates)),
        "n_pre_cutoff": n_pre,
        "n_post_cutoff": n_post,
        "status": status,
        "note": note,
    }


def compute_daily_mtm_portfolio_metrics(
    close_prices: Sequence[float],
    dates: Sequence[Any],
    decision_dates: Sequence[Any],
    signals: Sequence[int],
    hold_days: int,
    execution_lag_days: int = 1,
    transaction_cost_bps: float = 0.0,
    slippage_bps: float = 0.0,
    max_concurrent_positions: Optional[int] = None,
    trim_to_active_window: bool = False,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Compute a simple daily marked-to-market portfolio from horizon signals.

    The convention is intentionally conservative for daily-close data:
    - signal is observed on `decision_date`
    - trade enters at the close `execution_lag_days` sessions later
    - trade then holds for `hold_days` close-to-close sessions
    - each active position uses 1 / `max_concurrent_positions` of portfolio capital
    """
    price_index = pd.to_datetime(pd.Index(dates), errors="coerce")
    close = np.asarray(close_prices, dtype=float)
    signal_arr = np.asarray(signals, dtype=int)
    decision_index = pd.to_datetime(pd.Index(decision_dates), errors="coerce")

    if close.ndim != 1 or len(close) != len(price_index):
        raise ValueError("close_prices and dates must be aligned 1D sequences")
    if len(signal_arr) != len(decision_index):
        raise ValueError("decision_dates and signals must have the same length")
    if int(hold_days) <= 0:
        raise ValueError("hold_days must be positive")
    if int(execution_lag_days) < 0:
        raise ValueError("execution_lag_days cannot be negative")

    max_positions = int(max_concurrent_positions or hold_days)
    if max_positions <= 0:
        raise ValueError("max_concurrent_positions must be positive")

    df = pd.DataFrame(
        {
            "date": price_index,
            "close": close,
        }
    ).dropna(subset=["date"]).reset_index(drop=True)
    if df.empty:
        raise ValueError("No valid dated prices were provided")

    df["price_return"] = df["close"].pct_change().fillna(0.0)
    exposure = np.zeros(len(df), dtype=float)
    cost_events = np.zeros(len(df), dtype=float)
    gross_exposure = np.zeros(len(df), dtype=float)
    trade_rows: list[dict[str, Any]] = []

    date_to_idx = {pd.Timestamp(ts): i for i, ts in enumerate(df["date"])}
    total_cost_rate = max(float(transaction_cost_bps), 0.0) / 10000.0
    total_cost_rate += max(float(slippage_bps), 0.0) / 10000.0
    slot_weight = 1.0 / float(max_positions)

    for raw_decision_date, raw_signal in zip(decision_index, signal_arr):
        if pd.isna(raw_decision_date) or int(raw_signal) == 0:
            continue
        decision_date = pd.Timestamp(raw_decision_date)
        decision_idx = date_to_idx.get(decision_date)
        if decision_idx is None:
            continue

        entry_idx = decision_idx + int(execution_lag_days)
        if entry_idx >= len(df):
            continue
        exit_idx = entry_idx + int(hold_days)
        if exit_idx >= len(df):
            continue

        signed_weight = float(raw_signal) * slot_weight
        exposure[entry_idx + 1 : exit_idx + 1] += signed_weight
        gross_exposure[entry_idx + 1 : exit_idx + 1] += slot_weight
        if total_cost_rate > 0.0:
            cost_events[entry_idx] -= total_cost_rate * slot_weight
            cost_events[exit_idx] -= total_cost_rate * slot_weight

        trade_log_return = float(
            raw_signal
            * np.sum(np.log(np.clip(1.0 + df["price_return"].iloc[entry_idx + 1 : exit_idx + 1], 1e-12, None)))
        )
        trade_rows.append(
            {
                "decision_date": str(decision_date.date()),
                "entry_date": str(pd.Timestamp(df["date"].iloc[entry_idx]).date()),
                "exit_date": str(pd.Timestamp(df["date"].iloc[exit_idx]).date()),
                "signal": int(raw_signal),
                "entry_idx": int(entry_idx),
                "exit_idx": int(exit_idx),
                "gross_trade_return": float(np.expm1(trade_log_return)),
                "net_trade_return": float(np.expm1(trade_log_return) - 2.0 * total_cost_rate * slot_weight),
            }
        )

    df["net_exposure"] = exposure
    df["gross_exposure"] = gross_exposure
    df["cost_event"] = cost_events
    df["strategy_return"] = (df["net_exposure"] * df["price_return"]) + df["cost_event"]

    if trim_to_active_window and trade_rows:
        active_start = min(int(row["entry_idx"]) for row in trade_rows)
        active_end = max(int(row["exit_idx"]) for row in trade_rows)
        df = df.iloc[active_start : active_end + 1].copy().reset_index(drop=True)
    elif trim_to_active_window:
        df = df.iloc[0:0].copy()

    if len(df):
        df["equity_curve"] = np.cumprod(1.0 + df["strategy_return"])
    else:
        df["equity_curve"] = pd.Series(dtype=float)

    strategy_ret = df["strategy_return"].to_numpy(dtype=float)
    if strategy_ret.size:
        total_return = float(df["equity_curve"].iloc[-1] - 1.0)
        annualized_return = float((1.0 + total_return) ** (252.0 / max(len(df), 1)) - 1.0) if total_return > -1.0 else -1.0
        vol = float(np.std(strategy_ret))
        sharpe = float((np.mean(strategy_ret) / vol) * np.sqrt(252.0)) if vol > 0.0 else np.nan
        running_peak = np.maximum.accumulate(df["equity_curve"].to_numpy(dtype=float))
        max_drawdown = float(np.max(1.0 - (df["equity_curve"].to_numpy(dtype=float) / running_peak)))
    else:
        total_return = np.nan
        annualized_return = np.nan
        vol = np.nan
        sharpe = np.nan
        max_drawdown = np.nan

    trade_df = pd.DataFrame(trade_rows)
    trade_hit_rate = (
        float(np.mean(trade_df["net_trade_return"] > 0.0))
        if not trade_df.empty
        else np.nan
    )

    metrics = {
        "n_days": int(len(df)),
        "n_trade_decisions": int(np.sum(signal_arr != 0)),
        "n_executed_trades": int(len(trade_df)),
        "execution_lag_days": int(execution_lag_days),
        "hold_days": int(hold_days),
        "max_concurrent_positions": int(max_positions),
        "transaction_cost_bps": float(transaction_cost_bps),
        "slippage_bps": float(slippage_bps),
        "mean_daily_return": float(np.mean(strategy_ret)) if strategy_ret.size else np.nan,
        "daily_volatility": vol,
        "daily_sharpe_annualized": sharpe,
        "total_return": total_return,
        "annualized_return": annualized_return,
        "max_drawdown": max_drawdown,
        "avg_net_exposure": float(np.mean(np.abs(df["net_exposure"]))) if len(df) else np.nan,
        "avg_gross_exposure": float(np.mean(df["gross_exposure"])) if len(df) else np.nan,
        "trade_hit_rate": trade_hit_rate,
        "trim_to_active_window": bool(trim_to_active_window),
        "active_start_date": str(pd.Timestamp(df["date"].iloc[0]).date()) if len(df) else None,
        "active_end_date": str(pd.Timestamp(df["date"].iloc[-1]).date()) if len(df) else None,
    }
    return df, metrics


def evaluate_signal_threshold_candidates(
    *,
    y_pred_prices: np.ndarray,
    base_prices: np.ndarray,
    decision_dates: Sequence[Any],
    panel_dates: Sequence[Any],
    panel_prices: Sequence[float],
    horizon: int,
    threshold_candidates: Sequence[float],
    execution_lag_days: int = 1,
    hold_days: Optional[int] = None,
    transaction_cost_bps: float = 0.0,
    slippage_bps: float = 0.0,
    max_concurrent_positions: Optional[int] = None,
    objective: str = "daily_sharpe_annualized",
    min_executed_trades: int = 0,
    trim_to_active_window: bool = True,
) -> tuple[pd.DataFrame, float]:
    """Evaluate and select a signal threshold on a fixed validation slice."""
    unique_thresholds = sorted({max(float(x), 0.0) for x in threshold_candidates})
    if not unique_thresholds:
        raise ValueError("threshold_candidates must contain at least one value")

    target_hold_days = int(hold_days or horizon)
    rows: list[dict[str, Any]] = []
    for threshold in unique_thresholds:
        signals = compute_primary_horizon_signals(
            y_pred_prices=y_pred_prices,
            base_prices=base_prices,
            horizon=horizon,
            signal_threshold=threshold,
        )
        _, metrics = compute_daily_mtm_portfolio_metrics(
            close_prices=panel_prices,
            dates=panel_dates,
            decision_dates=decision_dates,
            signals=signals,
            hold_days=target_hold_days,
            execution_lag_days=execution_lag_days,
            transaction_cost_bps=transaction_cost_bps,
            slippage_bps=slippage_bps,
            max_concurrent_positions=max_concurrent_positions or target_hold_days,
            trim_to_active_window=trim_to_active_window,
        )
        objective_value = metrics.get(objective)
        rows.append(
            {
                "signal_threshold": float(threshold),
                **metrics,
                "selection_metric": float(objective_value)
                if objective_value is not None and np.isfinite(objective_value)
                else np.nan,
                "valid_for_selection": bool(
                    metrics.get("n_executed_trades", 0) >= int(min_executed_trades)
                    and objective_value is not None
                    and np.isfinite(objective_value)
                ),
            }
        )

    summary = pd.DataFrame(rows).sort_values("signal_threshold").reset_index(drop=True)
    valid = summary[summary["valid_for_selection"]].copy()
    if valid.empty:
        selected_threshold = float(summary.iloc[0]["signal_threshold"])
    else:
        valid = valid.sort_values(
            by=[
                "selection_metric",
                "total_return",
                "n_executed_trades",
                "signal_threshold",
            ],
            ascending=[False, False, False, False],
        )
        selected_threshold = float(valid.iloc[0]["signal_threshold"])
    return summary, selected_threshold
