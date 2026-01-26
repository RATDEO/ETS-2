"""Return and portfolio metrics for forecast evaluation."""

from typing import List, Optional, Tuple

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


def compute_long_short_portfolio_metrics(
    y_true_prices: np.ndarray,
    y_pred_prices: np.ndarray,
    base_prices: np.ndarray,
    horizons: List[int],
    signal_threshold: float = 0.0
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

        rows.append(
            {
                "horizon": h,
                "mean_return": mean_ret,
                "volatility": vol,
                "sharpe": sharpe,
                "hit_rate": hit_rate,
                "trade_rate": trade_rate,
                "n_trades": n_trades,
            }
        )

    return pd.DataFrame(rows).set_index("horizon")
