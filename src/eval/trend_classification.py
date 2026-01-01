"""
Trend classification for carbon price forecasting.

Classifies price movements into up/flat/down categories
for accuracy-based evaluation (as per the original paper).
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
import logging

logger = logging.getLogger(__name__)


def compute_threshold(
    prices: np.ndarray,
    method: str = "volatility",
    multiplier: float = 0.25,
    fixed_value: float = 0.5,
    window: int = 20
) -> float:
    """
    Compute the threshold for flat classification.
    
    Args:
        prices: Historical prices for threshold computation
        method: 'volatility' or 'fixed'
        multiplier: Multiplier for volatility method
        fixed_value: Fixed threshold value
        window: Rolling window for volatility calculation
        
    Returns:
        Threshold value
    """
    if method == "fixed":
        return fixed_value
    
    elif method == "volatility":
        # Compute log returns
        returns = np.log(prices[1:] / prices[:-1])
        
        # Use rolling standard deviation
        if len(returns) >= window:
            rolling_std = pd.Series(returns).rolling(window=window).std().iloc[-1]
        else:
            rolling_std = np.std(returns)
        
        # Threshold = multiplier * std * price level
        threshold = multiplier * rolling_std * np.mean(prices[-window:])
        
        return threshold
    
    else:
        raise ValueError(f"Unknown threshold method: {method}")


def classify_trend(
    price_t: float,
    price_t_plus_h: float,
    threshold: float = 0.0
) -> int:
    """
    Classify price movement into trend category.
    
    Args:
        price_t: Price at time t
        price_t_plus_h: Price at time t+h
        threshold: Threshold for flat classification
        
    Returns:
        -1 (down), 0 (flat), or 1 (up)
    """
    delta = price_t_plus_h - price_t
    
    if abs(delta) < threshold:
        return 0  # flat
    elif delta > 0:
        return 1  # up
    else:
        return -1  # down


def classify_trends_batch(
    y_start: np.ndarray,
    y_end: np.ndarray,
    threshold: float = 0.0
) -> np.ndarray:
    """
    Classify trends for a batch of samples.
    
    Args:
        y_start: Starting prices (batch,)
        y_end: Ending prices (batch,)
        threshold: Threshold for flat classification
        
    Returns:
        Array of trend labels (-1, 0, 1)
    """
    delta = y_end - y_start
    
    labels = np.zeros_like(delta, dtype=int)
    labels[delta > threshold] = 1
    labels[delta < -threshold] = -1
    
    return labels


def compute_trend_accuracy(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_base: np.ndarray,
    threshold: float = 0.0,
    horizons: List[int] = [1, 5, 20, 30]
) -> pd.DataFrame:
    """
    Compute trend classification accuracy at each horizon.
    
    Args:
        y_true: True future values (batch, pred_len)
        y_pred: Predicted future values (batch, pred_len)
        y_base: Base/current prices (batch,) - the starting point
        threshold: Threshold for flat classification
        horizons: Horizons to evaluate
        
    Returns:
        DataFrame with accuracy by horizon
    """
    results = []
    
    for h in horizons:
        h_idx = h - 1
        if h_idx >= y_true.shape[1]:
            continue
        
        # True trends
        true_trends = classify_trends_batch(y_base, y_true[:, h_idx], threshold)
        
        # Predicted trends
        pred_trends = classify_trends_batch(y_base, y_pred[:, h_idx], threshold)
        
        # Compute accuracy
        accuracy = np.mean(true_trends == pred_trends)
        
        # Per-class accuracy
        up_mask = true_trends == 1
        down_mask = true_trends == -1
        flat_mask = true_trends == 0
        
        up_acc = np.mean(pred_trends[up_mask] == 1) if up_mask.sum() > 0 else np.nan
        down_acc = np.mean(pred_trends[down_mask] == -1) if down_mask.sum() > 0 else np.nan
        flat_acc = np.mean(pred_trends[flat_mask] == 0) if flat_mask.sum() > 0 else np.nan
        
        results.append({
            "horizon": h,
            "accuracy": accuracy,
            "accuracy_up": up_acc,
            "accuracy_down": down_acc,
            "accuracy_flat": flat_acc,
            "n_up": up_mask.sum(),
            "n_down": down_mask.sum(),
            "n_flat": flat_mask.sum()
        })
    
    return pd.DataFrame(results).set_index("horizon")


def compute_directional_accuracy(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_base: np.ndarray,
    horizons: List[int] = [1, 5, 20, 30]
) -> pd.DataFrame:
    """
    Compute simple directional accuracy (up/down only, no flat).
    
    Args:
        y_true: True future values (batch, pred_len)
        y_pred: Predicted future values (batch, pred_len)
        y_base: Base prices (batch,)
        horizons: Horizons to evaluate
        
    Returns:
        DataFrame with directional accuracy by horizon
    """
    results = []
    
    for h in horizons:
        h_idx = h - 1
        if h_idx >= y_true.shape[1]:
            continue
        
        # True direction (up = 1, down = 0)
        true_direction = (y_true[:, h_idx] > y_base).astype(int)
        pred_direction = (y_pred[:, h_idx] > y_base).astype(int)
        
        accuracy = np.mean(true_direction == pred_direction)
        
        results.append({
            "horizon": h,
            "directional_accuracy": accuracy
        })
    
    return pd.DataFrame(results).set_index("horizon")


def get_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_base: np.ndarray,
    horizon: int,
    threshold: float = 0.0
) -> pd.DataFrame:
    """
    Compute confusion matrix for trend classification at a specific horizon.
    
    Args:
        y_true: True future values (batch, pred_len)
        y_pred: Predicted future values (batch, pred_len)
        y_base: Base prices (batch,)
        horizon: Specific horizon (1-indexed)
        threshold: Threshold for flat classification
        
    Returns:
        Confusion matrix as DataFrame
    """
    h_idx = horizon - 1
    
    true_trends = classify_trends_batch(y_base, y_true[:, h_idx], threshold)
    pred_trends = classify_trends_batch(y_base, y_pred[:, h_idx], threshold)
    
    # Create confusion matrix
    labels = [-1, 0, 1]
    label_names = ["down", "flat", "up"]
    
    matrix = np.zeros((3, 3), dtype=int)
    for i, true_label in enumerate(labels):
        for j, pred_label in enumerate(labels):
            matrix[i, j] = ((true_trends == true_label) & (pred_trends == pred_label)).sum()
    
    return pd.DataFrame(
        matrix,
        index=[f"true_{name}" for name in label_names],
        columns=[f"pred_{name}" for name in label_names]
    )
