"""
Evaluation metrics for time series forecasting.

Implements MSE, RMSE, MAE, MAPE with horizon-specific evaluation.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union
import logging

logger = logging.getLogger(__name__)


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Squared Error."""
    return np.mean((y_true - y_pred) ** 2)


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error."""
    return np.sqrt(mse(y_true, y_pred))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error."""
    return np.mean(np.abs(y_true - y_pred))


def mape(y_true: np.ndarray, y_pred: np.ndarray, epsilon: float = 1e-8) -> float:
    """Mean Absolute Percentage Error."""
    return np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + epsilon))) * 100


def smape(y_true: np.ndarray, y_pred: np.ndarray, epsilon: float = 1e-8) -> float:
    """Symmetric Mean Absolute Percentage Error."""
    return np.mean(
        2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred) + epsilon)
    ) * 100


def compute_all_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metrics: Optional[List[str]] = None
) -> Dict[str, float]:
    """
    Compute all specified metrics.
    
    Args:
        y_true: Ground truth values
        y_pred: Predicted values
        metrics: List of metric names. Defaults to ["mse", "rmse", "mae", "mape"]
        
    Returns:
        Dictionary of metric name -> value
    """
    if metrics is None:
        metrics = ["mse", "rmse", "mae", "mape"]
    
    metric_funcs = {
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "mape": mape,
        "smape": smape
    }
    
    result = {}
    for metric_name in metrics:
        if metric_name in metric_funcs:
            result[metric_name] = metric_funcs[metric_name](y_true, y_pred)
        else:
            logger.warning(f"Unknown metric: {metric_name}")
    
    return result


def compute_metrics_by_horizon(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    horizons: List[int] = [1, 5, 20, 30],
    metrics: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Compute metrics at specific forecast horizons.
    
    Args:
        y_true: Ground truth (batch, pred_len)
        y_pred: Predictions (batch, pred_len)
        horizons: Horizons to evaluate (1-indexed)
        metrics: List of metric names
        
    Returns:
        DataFrame with metrics by horizon
    """
    if metrics is None:
        metrics = ["mse", "rmse", "mae", "mape"]
    
    # Validate shapes
    if y_true.shape != y_pred.shape:
        raise ValueError(f"Shape mismatch: y_true={y_true.shape}, y_pred={y_pred.shape}")
    
    pred_len = y_true.shape[1]
    
    results = []
    for h in horizons:
        if h > pred_len:
            logger.warning(f"Horizon {h} exceeds pred_len {pred_len}")
            continue
        
        # Extract predictions at horizon h (1-indexed)
        h_idx = h - 1
        y_true_h = y_true[:, h_idx]
        y_pred_h = y_pred[:, h_idx]
        
        # Compute metrics
        horizon_metrics = compute_all_metrics(y_true_h, y_pred_h, metrics)
        horizon_metrics["horizon"] = h
        results.append(horizon_metrics)
    
    df = pd.DataFrame(results)
    df = df.set_index("horizon")
    
    return df


def compute_global_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metrics: Optional[List[str]] = None
) -> Dict[str, float]:
    """
    Compute metrics averaged across all horizons.
    
    Args:
        y_true: Ground truth (batch, pred_len)
        y_pred: Predictions (batch, pred_len)
        metrics: List of metric names
        
    Returns:
        Dictionary of global metrics
    """
    if metrics is None:
        metrics = ["mse", "rmse", "mae", "mape"]
    
    # Flatten predictions
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()
    
    return compute_all_metrics(y_true_flat, y_pred_flat, metrics)


def compute_path_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metrics: Optional[List[str]] = None
) -> Dict[str, float]:
    """
    Compute metrics over the full multi-step forecast path.

    This is the paper-style evaluation for multi-step regression where MSE is
    averaged across all predicted steps (e.g., the next 30 days) and all samples.

    Returned keys are suffixed with `_path` to distinguish them from horizon-specific
    metrics.
    """
    global_metrics = compute_global_metrics(y_true, y_pred, metrics=metrics)
    return {f"{name}_path": value for name, value in global_metrics.items()}


def get_per_sample_errors(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    horizons: List[int] = [1, 5, 20, 30]
) -> Dict[int, np.ndarray]:
    """
    Get per-sample squared errors for each horizon.
    
    Useful for significance testing.
    
    Args:
        y_true: Ground truth (batch, pred_len)
        y_pred: Predictions (batch, pred_len)
        horizons: Horizons to evaluate
        
    Returns:
        Dictionary mapping horizon -> array of squared errors
    """
    result = {}
    
    for h in horizons:
        h_idx = h - 1
        if h_idx < y_true.shape[1]:
            errors = (y_true[:, h_idx] - y_pred[:, h_idx]) ** 2
            result[h] = errors
    
    return result
