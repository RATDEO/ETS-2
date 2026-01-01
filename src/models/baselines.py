"""
Baseline models for time series forecasting.

Implements:
- Naive persistence: y_hat[t+h] = y[t]
- Seasonal naive: y_hat[t+h] = y[t-5] (weekly)
- Linear regression on lagged features
- ARIMA/SARIMAX
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path
import logging
import pickle

try:
    from sklearn.linear_model import Ridge, Lasso
    from sklearn.preprocessing import StandardScaler as SklearnScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False

logger = logging.getLogger(__name__)


class BaseForecaster:
    """Base class for all forecasting models."""
    
    def __init__(self, pred_len: int = 30):
        self.pred_len = pred_len
        self.is_fitted = False
    
    def fit(self, y: np.ndarray, X: Optional[np.ndarray] = None):
        """Fit the model."""
        raise NotImplementedError
    
    def predict(self, y: np.ndarray, X: Optional[np.ndarray] = None) -> np.ndarray:
        """Generate predictions."""
        raise NotImplementedError
    
    def save(self, path: Union[str, Path]):
        """Save model to file."""
        with open(path, "wb") as f:
            pickle.dump(self, f)
    
    @classmethod
    def load(cls, path: Union[str, Path]) -> "BaseForecaster":
        """Load model from file."""
        with open(path, "rb") as f:
            return pickle.load(f)


class NaivePersistence(BaseForecaster):
    """
    Naive persistence forecast: y_hat[t+h] = y[t] for all h.
    
    Simply repeats the last known value for all forecast horizons.
    """
    
    def __init__(self, pred_len: int = 30):
        super().__init__(pred_len)
        self.name = "naive_persistence"
    
    def fit(self, y: np.ndarray, X: Optional[np.ndarray] = None):
        """No fitting required for naive persistence."""
        self.is_fitted = True
        return self
    
    def predict(self, y: np.ndarray, X: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Generate persistence forecasts.
        
        Args:
            y: Historical values. Can be 1D (single series) or 2D (batch, seq_len)
            X: Ignored
            
        Returns:
            Predictions of shape (batch, pred_len)
        """
        if y.ndim == 1:
            y = y.reshape(1, -1)
        
        # Last value repeated for all horizons
        last_value = y[:, -1:]  # (batch, 1)
        predictions = np.repeat(last_value, self.pred_len, axis=1)
        
        return predictions
    
    def predict_batch(
        self,
        y_history: np.ndarray,
        X: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Generate predictions for a batch of windows.
        
        Args:
            y_history: Shape (batch, seq_len) - historical target values
            X: Ignored
            
        Returns:
            Predictions of shape (batch, pred_len)
        """
        return self.predict(y_history)


class SeasonalNaive(BaseForecaster):
    """
    Seasonal naive forecast: y_hat[t+h] = y[t+h-season_period].
    
    Uses weekly seasonality (5 trading days) by default.
    """
    
    def __init__(self, pred_len: int = 30, season_period: int = 5):
        super().__init__(pred_len)
        self.season_period = season_period
        self.name = f"seasonal_naive_{season_period}"
    
    def fit(self, y: np.ndarray, X: Optional[np.ndarray] = None):
        """No fitting required for seasonal naive."""
        self.is_fitted = True
        return self
    
    def predict(self, y: np.ndarray, X: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Generate seasonal naive forecasts.
        
        Args:
            y: Historical values (batch, seq_len)
            
        Returns:
            Predictions (batch, pred_len)
        """
        if y.ndim == 1:
            y = y.reshape(1, -1)
        
        batch_size = y.shape[0]
        predictions = np.zeros((batch_size, self.pred_len))
        
        for h in range(self.pred_len):
            # Index in history that corresponds to same seasonal position
            seasonal_idx = -(self.season_period - (h % self.season_period))
            if seasonal_idx < -y.shape[1]:
                seasonal_idx = -1  # Fallback to last value
            predictions[:, h] = y[:, seasonal_idx]
        
        return predictions


class LinearBaseline(BaseForecaster):
    """
    Linear regression baseline using lagged features.
    
    Creates a separate model for each forecast horizon.
    """
    
    def __init__(
        self,
        pred_len: int = 30,
        lags: List[int] = [1, 5, 10, 20],
        alpha: float = 1.0,
        model_type: str = "ridge"
    ):
        super().__init__(pred_len)
        
        if not SKLEARN_AVAILABLE:
            raise ImportError("sklearn required for LinearBaseline")
        
        self.lags = lags
        self.alpha = alpha
        self.model_type = model_type
        self.name = f"linear_{model_type}"
        
        # One model per horizon
        self.models = {}
        self.scalers = {}
    
    def _create_features(self, y: np.ndarray) -> np.ndarray:
        """
        Create lagged features from target series.
        
        Args:
            y: Shape (batch, seq_len)
            
        Returns:
            Feature matrix (batch, n_features)
        """
        batch_size, seq_len = y.shape
        
        features = []
        for lag in self.lags:
            if lag <= seq_len:
                features.append(y[:, -lag])
        
        # Add rolling statistics
        features.append(np.mean(y[:, -20:], axis=1))  # 20-day mean
        features.append(np.std(y[:, -20:], axis=1))   # 20-day std
        features.append(y[:, -1] - y[:, -5])          # 5-day momentum
        
        return np.column_stack(features)
    
    def fit(
        self,
        y_history: np.ndarray,
        y_future: np.ndarray,
        X: Optional[np.ndarray] = None
    ):
        """
        Fit linear models for each horizon.
        
        Args:
            y_history: Shape (batch, seq_len)
            y_future: Shape (batch, pred_len)
            X: Optional exogenous features (ignored for now)
        """
        features = self._create_features(y_history)
        
        for h in range(self.pred_len):
            # Create and fit scaler
            scaler = SklearnScaler()
            X_scaled = scaler.fit_transform(features)
            self.scalers[h] = scaler
            
            # Create and fit model
            if self.model_type == "ridge":
                model = Ridge(alpha=self.alpha)
            else:
                model = Lasso(alpha=self.alpha)
            
            model.fit(X_scaled, y_future[:, h])
            self.models[h] = model
        
        self.is_fitted = True
        logger.info(f"Fitted {self.pred_len} linear models")
        
        return self
    
    def predict(
        self,
        y_history: np.ndarray,
        X: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Generate predictions for all horizons.
        
        Args:
            y_history: Shape (batch, seq_len)
            
        Returns:
            Predictions (batch, pred_len)
        """
        if not self.is_fitted:
            raise RuntimeError("Model not fitted")
        
        if y_history.ndim == 1:
            y_history = y_history.reshape(1, -1)
        
        batch_size = y_history.shape[0]
        predictions = np.zeros((batch_size, self.pred_len))
        
        features = self._create_features(y_history)
        
        for h in range(self.pred_len):
            X_scaled = self.scalers[h].transform(features)
            predictions[:, h] = self.models[h].predict(X_scaled)
        
        return predictions


class ARIMABaseline(BaseForecaster):
    """
    ARIMA baseline for univariate forecasting.
    
    Uses auto ARIMA or fixed order.
    """
    
    def __init__(
        self,
        pred_len: int = 30,
        order: Tuple[int, int, int] = (5, 1, 2),
        seasonal_order: Optional[Tuple[int, int, int, int]] = None
    ):
        super().__init__(pred_len)
        
        if not STATSMODELS_AVAILABLE:
            raise ImportError("statsmodels required for ARIMABaseline")
        
        self.order = order
        self.seasonal_order = seasonal_order
        self.name = f"arima_{order}"
    
    def fit(self, y: np.ndarray, X: Optional[np.ndarray] = None):
        """
        ARIMA is fitted per prediction, so this just validates.
        """
        self.is_fitted = True
        return self
    
    def predict_single(self, y: np.ndarray) -> np.ndarray:
        """
        Generate ARIMA forecast for a single series.
        
        Args:
            y: 1D historical series
            
        Returns:
            Predictions (pred_len,)
        """
        try:
            if self.seasonal_order:
                model = SARIMAX(
                    y,
                    order=self.order,
                    seasonal_order=self.seasonal_order,
                    enforce_stationarity=False,
                    enforce_invertibility=False
                )
            else:
                model = ARIMA(
                    y,
                    order=self.order,
                    enforce_stationarity=False,
                    enforce_invertibility=False
                )
            
            fitted = model.fit(disp=False)
            forecast = fitted.forecast(steps=self.pred_len)
            
            return forecast
        
        except Exception as e:
            logger.warning(f"ARIMA failed, using persistence: {e}")
            return np.full(self.pred_len, y[-1])
    
    def predict(
        self,
        y_history: np.ndarray,
        X: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Generate ARIMA predictions for batch.
        
        Args:
            y_history: Shape (batch, seq_len)
            
        Returns:
            Predictions (batch, pred_len)
        """
        if y_history.ndim == 1:
            y_history = y_history.reshape(1, -1)
        
        batch_size = y_history.shape[0]
        predictions = np.zeros((batch_size, self.pred_len))
        
        for i in range(batch_size):
            predictions[i] = self.predict_single(y_history[i])
        
        return predictions


def evaluate_baselines(
    y_train: np.ndarray,
    y_test: np.ndarray,
    y_test_future: np.ndarray,
    pred_len: int = 30,
    horizons: List[int] = [1, 5, 20, 30]
) -> pd.DataFrame:
    """
    Evaluate all baseline models on test data.
    
    Args:
        y_train: Training target for fitting
        y_test: Test input sequences (batch, seq_len)
        y_test_future: Test targets (batch, pred_len)
        pred_len: Prediction length
        horizons: Horizons to evaluate
        
    Returns:
        DataFrame with metrics by baseline and horizon
    """
    from ..eval.metrics import compute_metrics_by_horizon
    
    baselines = {
        "naive_persistence": NaivePersistence(pred_len),
        "seasonal_naive_5": SeasonalNaive(pred_len, season_period=5),
    }
    
    # Add linear baseline if sklearn available
    if SKLEARN_AVAILABLE:
        linear = LinearBaseline(pred_len)
        # Create training data for linear baseline
        n_train = len(y_train) - pred_len - 120  # seq_len assumed 120
        if n_train > 100:
            train_history = np.array([y_train[i:i+120] for i in range(n_train)])
            train_future = np.array([y_train[i+120:i+120+pred_len] for i in range(n_train)])
            linear.fit(train_history, train_future)
            baselines["linear_ridge"] = linear
    
    results = []
    
    for name, model in baselines.items():
        if not model.is_fitted:
            model.fit(y_train)
        
        predictions = model.predict(y_test)
        
        metrics_df = compute_metrics_by_horizon(
            y_test_future, predictions, horizons=horizons
        )
        metrics_df["model"] = name
        results.append(metrics_df.reset_index())
    
    return pd.concat(results, ignore_index=True)
