"""
Quantile regression with L1 (LASSO) feature selection.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

try:
    from sklearn.linear_model import QuantileRegressor
    SKLEARN_AVAILABLE = True
except ImportError:  # pragma: no cover
    SKLEARN_AVAILABLE = False


@dataclass
class QuantileLassoConfig:
    quantiles: List[float]
    alpha: float = 0.01
    lags: List[int] = None

    def __post_init__(self) -> None:
        if self.lags is None:
            self.lags = [1, 5, 10, 20]


def _create_feature_matrix(
    X_enc: np.ndarray,
    feature_names: List[str],
    target_idx: int,
    lags: List[int],
) -> Tuple[np.ndarray, List[str]]:
    """
    Build a compact feature matrix:
    - last value of each feature
    - target lags for selected horizons
    - target rolling stats (20-day mean/std when available)
    """
    if X_enc.ndim != 3:
        raise ValueError("X_enc must be (n_samples, seq_len, n_features)")

    n_samples, seq_len, n_features = X_enc.shape
    features = []
    names = []

    last_values = X_enc[:, -1, :]
    features.append(last_values)
    names.extend([f"{name}_last" for name in feature_names])

    target_series = X_enc[:, :, target_idx]
    for lag in lags:
        if lag <= seq_len:
            features.append(target_series[:, -lag].reshape(n_samples, 1))
            names.append(f"y_lag_{lag}")

    window = min(20, seq_len)
    if window > 1:
        features.append(np.mean(target_series[:, -window:], axis=1).reshape(n_samples, 1))
        features.append(np.std(target_series[:, -window:], axis=1).reshape(n_samples, 1))
        names.extend([f"y_mean_{window}", f"y_std_{window}"])

    X = np.column_stack(features).astype(np.float32)
    return X, names


class QuantileLasso:
    """
    Fit a separate L1-penalized quantile model per horizon and quantile.
    """

    def __init__(self, config: QuantileLassoConfig):
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn is required for QuantileLasso")
        self.config = config
        self.models: Dict[float, Dict[int, QuantileRegressor]] = {}
        self.feature_names_: List[str] = []
        self.target_idx_: int = 0

    def fit(
        self,
        X_enc: np.ndarray,
        feature_names: List[str],
        target_idx: int,
        y_future: np.ndarray,
    ) -> "QuantileLasso":
        if y_future.ndim != 2:
            raise ValueError("y_future must be (n_samples, pred_len)")

        self.target_idx_ = target_idx
        X, self.feature_names_ = _create_feature_matrix(
            X_enc, feature_names, target_idx, self.config.lags
        )

        pred_len = y_future.shape[1]
        for q in self.config.quantiles:
            self.models[q] = {}
            for h in range(pred_len):
                model = QuantileRegressor(
                    quantile=q,
                    alpha=self.config.alpha,
                    solver="highs",
                )
                model.fit(X, y_future[:, h])
                self.models[q][h] = model

        return self

    def predict(self, X_enc: np.ndarray) -> np.ndarray:
        if not self.models:
            raise ValueError("QuantileLasso model is not fitted")

        X, _ = _create_feature_matrix(
            X_enc, self.feature_names_, self.target_idx_, self.config.lags
        )

        quantiles = self.config.quantiles
        pred_len = len(next(iter(self.models.values())))
        preds = np.zeros((X.shape[0], pred_len, len(quantiles)), dtype=np.float32)

        for q_idx, q in enumerate(quantiles):
            for h, model in self.models[q].items():
                preds[:, h, q_idx] = model.predict(X)

        return preds

    def coefficients(self) -> Dict[float, np.ndarray]:
        coef_map = {}
        for q, models in self.models.items():
            coef_map[q] = np.vstack([models[h].coef_ for h in sorted(models.keys())])
        return coef_map

    def selected_features(self, threshold: float = 1e-6) -> Dict[float, List[str]]:
        selected = {}
        for q, coef_matrix in self.coefficients().items():
            mean_abs = np.mean(np.abs(coef_matrix), axis=0)
            mask = mean_abs > threshold
            selected[q] = [name for name, keep in zip(self.feature_names_, mask) if keep]
        return selected
