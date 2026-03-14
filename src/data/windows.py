"""
Window creation for time series forecasting.

Creates supervised learning datasets with:
- Input sequences of length seq_len
- Label context of length label_len
- Prediction targets of length pred_len
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Optional, Tuple, Union
import logging
import pickle
from dataclasses import dataclass

try:
    import torch
    from torch.utils.data import Dataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class WindowConfig:
    """Configuration for window creation."""
    seq_len: int = 120  # Input sequence length
    label_len: int = 30  # Label context length (for decoder)
    pred_len: int = 30  # Prediction horizon
    target_col: str = "y"
    feature_cols: Optional[List[str]] = None
    date_col: str = "date"


@dataclass
class WindowMetadata:
    """Per-window date metadata used for contamination-safe splitting."""
    pred_start_dates: np.ndarray
    pred_end_dates: np.ndarray
    last_input_dates: np.ndarray


def _fill_window_values(window: np.ndarray) -> np.ndarray:
    """Forward-fill within a window and fall back to window means for leading NaNs."""
    frame = pd.DataFrame(window).ffill()
    if frame.isna().any().any():
        col_mean = frame.mean(skipna=True).fillna(0.0)
        frame = frame.fillna(col_mean)
    return frame.values.astype(np.float32)


def _future_fill_value(name: str, last_value: float, target_col: str) -> float:
    """Choose a non-leaking placeholder for unknown future decoder features."""
    lname = str(name).lower()
    if name == target_col and lname.endswith("_return"):
        return 0.0
    if lname.endswith("_return") or lname.endswith("_pct"):
        return 0.0
    if lname.startswith("is_") or lname.endswith("_flag"):
        return 0.0
    return float(last_value)


def make_windows(
    panel: pd.DataFrame,
    config: WindowConfig,
    mode: str = "MS",
    return_metadata: bool = False,
) -> Union[
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, WindowMetadata],
]:
    """
    Create sliding window datasets for time series forecasting.
    
    Args:
        panel: Panel DataFrame with target and features
        config: Window configuration
        mode: 'S' for univariate (target only) or 'MS' for multivariate
        
    Returns:
        Tuple of (X_enc, X_dec, y, dates) or
        (X_enc, X_dec, y, dates, metadata) where:
            - X_enc: Encoder input (batch, seq_len, features)
            - X_dec: Decoder input (batch, label_len + pred_len, features)
            - y: Target (batch, pred_len)
            - dates: Prediction start dates
    """
    # Validate inputs
    if config.target_col not in panel.columns:
        raise ValueError(f"Target column '{config.target_col}' not found")
    
    # Determine feature columns
    if mode == "S":
        # Univariate: only target
        feature_cols = [config.target_col]
    else:
        # Multivariate: target + exogenous features
        if config.feature_cols:
            feature_cols = config.feature_cols.copy()
        else:
            # Use all numeric columns except date
            feature_cols = [c for c in panel.select_dtypes(include=[np.number]).columns
                          if c != config.date_col]
        
        # Ensure target is first
        if config.target_col in feature_cols:
            feature_cols.remove(config.target_col)
        feature_cols = [config.target_col] + feature_cols
    
    logger.info(f"Creating windows with {len(feature_cols)} features, mode={mode}")
    
    # Extract data
    data = panel[feature_cols].values.astype(np.float32)
    target = panel[config.target_col].values.astype(np.float32)
    dates = panel[config.date_col].values
    
    # Handle NaN by forward-filling globally (avoid backfill leakage)
    data = pd.DataFrame(data).ffill().values.astype(np.float32)
    
    # Calculate number of valid windows
    total_len = config.seq_len + config.pred_len
    n_windows = len(panel) - total_len + 1
    
    if n_windows <= 0:
        raise ValueError(f"Not enough data for windows: need {total_len}, have {len(panel)}")
    
    logger.info(f"Creating {n_windows} windows")
    
    # Pre-allocate arrays
    X_enc = np.zeros((n_windows, config.seq_len, len(feature_cols)), dtype=np.float32)
    X_dec = np.zeros((n_windows, config.label_len + config.pred_len, len(feature_cols)), dtype=np.float32)
    y = np.zeros((n_windows, config.pred_len), dtype=np.float32)
    pred_dates = []
    pred_end_dates = []
    last_input_dates = []
    
    # Create windows
    for i in range(n_windows):
        # Encoder input: [i, i + seq_len)
        enc_start = i
        enc_end = i + config.seq_len
        enc_window = _fill_window_values(data[enc_start:enc_end].copy())
        X_enc[i] = enc_window
        
        # Decoder input is contamination-safe:
        # - observed label context from the past
        # - placeholder future slots instead of realized future rows
        context_start = i + config.seq_len - config.label_len
        context_end = i + config.seq_len
        context_window = _fill_window_values(data[context_start:context_end].copy())
        last_row = context_window[-1].copy()
        future_block = np.repeat(last_row[None, :], config.pred_len, axis=0).astype(np.float32)
        for j, name in enumerate(feature_cols):
            future_block[:, j] = _future_fill_value(name, last_row[j], config.target_col)
        X_dec[i] = np.concatenate([context_window, future_block], axis=0)
        
        # Target: [i + seq_len, i + seq_len + pred_len)
        y_start = i + config.seq_len
        y_end = i + config.seq_len + config.pred_len
        y[i] = target[y_start:y_end]
        
        # Prediction start date
        pred_dates.append(dates[y_start])
        pred_end_dates.append(dates[y_end - 1])
        last_input_dates.append(dates[enc_end - 1])
    
    pred_dates = np.array(pred_dates)
    pred_end_dates = np.array(pred_end_dates)
    last_input_dates = np.array(last_input_dates)
    
    logger.info(f"Window shapes: X_enc={X_enc.shape}, X_dec={X_dec.shape}, y={y.shape}")

    if not return_metadata:
        return X_enc, X_dec, y, pred_dates

    metadata = WindowMetadata(
        pred_start_dates=pred_dates,
        pred_end_dates=pred_end_dates,
        last_input_dates=last_input_dates,
    )
    return X_enc, X_dec, y, pred_dates, metadata


def split_windows(
    X_enc: np.ndarray,
    X_dec: np.ndarray,
    y: np.ndarray,
    dates: np.ndarray,
    train_end: str,
    val_end: str,
    window_meta: Optional[WindowMetadata] = None,
) -> dict:
    """
    Split windowed data into train/val/test by date.
    
    Args:
        X_enc, X_dec, y, dates: From make_windows()
        train_end: Last date for training
        val_end: Last date for validation
        
    Returns:
        Dictionary with train/val/test splits
    """
    train_end = pd.Timestamp(train_end)
    val_end = pd.Timestamp(val_end)
    dates = pd.to_datetime(dates)
    end_dates = (
        pd.to_datetime(window_meta.pred_end_dates)
        if window_meta is not None
        else dates
    )
    last_input_dates = (
        pd.to_datetime(window_meta.last_input_dates)
        if window_meta is not None
        else pd.to_datetime(dates) - pd.Timedelta(days=1)
    )

    # Contamination-safe split:
    # - train labels must end within train period
    # - val labels must be fully inside validation period
    # - test labels start after validation period
    train_mask = end_dates <= train_end
    val_mask = (dates > train_end) & (end_dates <= val_end)
    test_mask = dates > val_end
    dropped_mask = ~(train_mask | val_mask | test_mask)

    if window_meta is None:
        logger.warning(
            "split_windows called without window metadata; using legacy boundary logic."
        )
    else:
        train_violations = int(np.sum(end_dates[train_mask] > train_end))
        val_start_violations = int(np.sum(dates[val_mask] <= train_end))
        val_end_violations = int(np.sum(end_dates[val_mask] > val_end))
        test_violations = int(np.sum(dates[test_mask] <= val_end))
        if any((train_violations, val_start_violations, val_end_violations, test_violations)):
            raise ValueError(
                "Split leakage detected after applying contamination-safe boundary rules"
            )
    
    splits = {
        "train": {
            "X_enc": X_enc[train_mask],
            "X_dec": X_dec[train_mask],
            "y": y[train_mask],
            "dates": dates[train_mask],
            "end_dates": end_dates[train_mask],
            "last_input_dates": last_input_dates[train_mask],
        },
        "val": {
            "X_enc": X_enc[val_mask],
            "X_dec": X_dec[val_mask],
            "y": y[val_mask],
            "dates": dates[val_mask],
            "end_dates": end_dates[val_mask],
            "last_input_dates": last_input_dates[val_mask],
        },
        "test": {
            "X_enc": X_enc[test_mask],
            "X_dec": X_dec[test_mask],
            "y": y[test_mask],
            "dates": dates[test_mask],
            "end_dates": end_dates[test_mask],
            "last_input_dates": last_input_dates[test_mask],
        }
    }
    
    for split_name, split_data in splits.items():
        logger.info(f"{split_name}: {len(split_data['y'])} windows")
    if window_meta is not None and np.any(dropped_mask):
        logger.info(
            "Dropped %d boundary-crossing windows to prevent split leakage",
            int(np.sum(dropped_mask)),
        )
    
    return splits


class StandardScaler:
    """Simple standard scaler that stores mean and std."""
    
    def __init__(self):
        self.mean_ = None
        self.std_ = None
        self.feature_names_ = None
    
    def fit(self, X: np.ndarray, feature_names: Optional[List[str]] = None):
        """Fit scaler on training data."""
        # X shape: (n_samples, seq_len, n_features)
        # Flatten time dimension for statistics
        X_flat = X.reshape(-1, X.shape[-1])
        
        self.mean_ = np.nanmean(X_flat, axis=0)
        self.std_ = np.nanstd(X_flat, axis=0)
        
        # Prevent division by zero
        self.std_[self.std_ == 0] = 1.0
        
        self.feature_names_ = feature_names
        
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform data."""
        return (X - self.mean_) / self.std_
    
    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """Inverse transform data."""
        return X * self.std_ + self.mean_
    
    def inverse_transform_target(self, y: np.ndarray, target_idx: int = 0) -> np.ndarray:
        """Inverse transform target variable only."""
        return y * self.std_[target_idx] + self.mean_[target_idx]
    
    def save(self, path: Union[str, Path]):
        """Save scaler to file."""
        with open(path, "wb") as f:
            pickle.dump({
                "mean_": self.mean_,
                "std_": self.std_,
                "feature_names_": self.feature_names_
            }, f)
    
    @classmethod
    def load(cls, path: Union[str, Path]) -> "StandardScaler":
        """Load scaler from file."""
        scaler = cls()
        with open(path, "rb") as f:
            data = pickle.load(f)
        scaler.mean_ = data["mean_"]
        scaler.std_ = data["std_"]
        scaler.feature_names_ = data["feature_names_"]
        return scaler


def save_datasets(
    splits: dict,
    scaler: StandardScaler,
    save_dir: Union[str, Path],
    scaled: bool = True
) -> None:
    """
    Save split datasets and scaler.
    
    Args:
        splits: Dictionary from split_windows()
        scaler: Fitted StandardScaler
        save_dir: Directory to save to
        scaled: Whether to save scaled or raw data
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    for split_name, split_data in splits.items():
        X_enc = split_data["X_enc"]
        X_dec = split_data["X_dec"]
        y = split_data["y"]
        
        if scaled:
            X_enc = scaler.transform(X_enc)
            X_dec = scaler.transform(X_dec)
            # Scale target using first feature (target)
            y_scaled = (y - scaler.mean_[0]) / scaler.std_[0]
        else:
            y_scaled = y
        
        np.savez(
            save_dir / f"{split_name}.npz",
            X_enc=X_enc,
            X_dec=X_dec,
            y=y_scaled,
            dates=split_data["dates"].astype(str)
        )
    
    scaler.save(save_dir / "scaler.pkl")
    
    logger.info(f"Saved datasets to {save_dir}")


if TORCH_AVAILABLE:
    class TimeSeriesDataset(Dataset):
        """PyTorch Dataset for time series forecasting."""
        
        def __init__(
            self,
            X_enc: np.ndarray,
            X_dec: np.ndarray,
            y: np.ndarray,
            dates: Optional[np.ndarray] = None
        ):
            self.X_enc = torch.from_numpy(X_enc)
            self.X_dec = torch.from_numpy(X_dec)
            self.y = torch.from_numpy(y)
            self.dates = dates
        
        def __len__(self) -> int:
            return len(self.y)
        
        def __getitem__(self, idx: int):
            return {
                "X_enc": self.X_enc[idx],
                "X_dec": self.X_dec[idx],
                "y": self.y[idx]
            }
        
        @classmethod
        def from_npz(cls, path: Union[str, Path]) -> "TimeSeriesDataset":
            """Load dataset from .npz file."""
            data = np.load(path)
            return cls(
                X_enc=data["X_enc"],
                X_dec=data["X_dec"],
                y=data["y"],
                dates=data.get("dates")
            )
else:
    class TimeSeriesDataset:
        """Placeholder when PyTorch is not available."""
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch is required for TimeSeriesDataset")
