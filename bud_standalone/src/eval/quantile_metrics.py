"""
Quantile regression evaluation metrics.
"""

from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd


def pinball_loss(y_true: np.ndarray, y_pred: np.ndarray, quantile: float) -> float:
    """
    Compute mean pinball loss for a given quantile.
    """
    diff = y_true - y_pred
    loss = np.maximum(quantile * diff, (quantile - 1) * diff)
    return float(np.mean(loss))


def compute_quantile_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    quantiles: List[float],
    horizons: List[int],
) -> pd.DataFrame:
    """
    Compute pinball loss by horizon for each quantile.

    Args:
        y_true: (n_samples, pred_len)
        y_pred: (n_samples, pred_len, n_quantiles)
    """
    rows = []
    for q_idx, q in enumerate(quantiles):
        for h_idx, h in enumerate(horizons):
            if h_idx >= y_true.shape[1]:
                continue
            loss = pinball_loss(y_true[:, h_idx], y_pred[:, h_idx, q_idx], q)
            rows.append(
                {
                    "quantile": q,
                    "horizon": h,
                    "pinball_loss": loss,
                }
            )
    return pd.DataFrame(rows)
