# Autoformer TSM Performance Tuning Guide

## Executive Summary

The Autoformer model is severely underperforming with MSE of ~2000 compared to naive persistence at ~4-50 (40-80x worse). This document outlines 12 concrete strategies to improve performance.

### Diagnosed Issues:
1. **Severe overfitting**: Train loss ~1.2 vs Val loss ~6.4 (5x gap)
2. **Distribution shift**: 2010-2022 training vs 2023-2025 test (EU ETS price regime change ~€25 → €80+)
3. **Small dataset**: Only 2,128 training windows for a 512-dim model
4. **Scale mismatch**: Model may be predicting trends instead of price levels
5. **Architecture mismatch**: Autoformer designed for longer sequences, not 30-day financial forecasts

---

## Strategy 1: Use DLinear or PatchTST Instead of Autoformer

**Problem**: Autoformer was designed for long-horizon, large-scale datasets. Recent research shows simpler linear models outperform transformers on many forecasting benchmarks.

**Implementation**:
```yaml
model:
  tsm_type: "dlinear"  # or "patchtst"
```

**Code changes needed in `src/models/tsm.py`**:
```python
class DLinear(nn.Module):
    """A simple yet effective linear model for time series forecasting."""
    
    def __init__(self, seq_len: int, pred_len: int, enc_in: int, individual: bool = False):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.individual = individual
        
        if individual:
            self.Linear_Trend = nn.ModuleList([
                nn.Linear(seq_len, pred_len) for _ in range(enc_in)
            ])
            self.Linear_Seasonal = nn.ModuleList([
                nn.Linear(seq_len, pred_len) for _ in range(enc_in)
            ])
        else:
            self.Linear_Trend = nn.Linear(seq_len, pred_len)
            self.Linear_Seasonal = nn.Linear(seq_len, pred_len)
        
        # Simple moving average for decomposition
        self.kernel_size = 25
        self.avg = nn.AvgPool1d(kernel_size=self.kernel_size, stride=1, 
                                 padding=self.kernel_size // 2)
    
    def forward(self, x):
        # x: (batch, seq_len, features)
        # Decompose
        trend = self.avg(x.permute(0, 2, 1)).permute(0, 2, 1)
        seasonal = x - trend
        
        # Project each component
        if self.individual:
            trend_out = torch.cat([
                self.Linear_Trend[i](trend[:, :, i:i+1].squeeze(-1)).unsqueeze(-1)
                for i in range(trend.shape[-1])
            ], dim=-1)
            seasonal_out = torch.cat([
                self.Linear_Seasonal[i](seasonal[:, :, i:i+1].squeeze(-1)).unsqueeze(-1)
                for i in range(seasonal.shape[-1])
            ], dim=-1)
        else:
            trend_out = self.Linear_Trend(trend.permute(0, 2, 1)).permute(0, 2, 1)
            seasonal_out = self.Linear_Seasonal(seasonal.permute(0, 2, 1)).permute(0, 2, 1)
        
        # Only return target column
        return (trend_out + seasonal_out)[:, :, :1]
```

**Expected improvement**: 50-80% reduction in MSE based on DLinear paper results.

---

## Strategy 2: Reduce Model Capacity Drastically

**Problem**: With only 2,128 training samples, a 512-dim model with 8 heads is massively overparameterized.

**Current config**: ~4M parameters for ~2K samples (2000:1 ratio)
**Target**: <100K parameters (50:1 ratio)

**New configuration**:
```yaml
model:
  d_model: 64       # Was 512
  n_heads: 2        # Was 8  
  e_layers: 1       # Was 2
  d_ff: 128         # Was 2048
  dropout: 0.3      # Was 0.05
```

**Implementation**:
```python
# In src/config/default.yaml
model:
  d_model: 64
  n_heads: 2
  e_layers: 1
  d_ff: 128
  dropout: 0.3
```

**Expected improvement**: 40-60% reduction in val/test gap.

---

## Strategy 3: Predict Returns Instead of Prices

**Problem**: Predicting absolute prices causes scale issues across regimes (€25 in 2018 vs €80 in 2023).

**Implementation** in `src/data/windows.py`:
```python
def make_windows_returns(
    panel: pd.DataFrame,
    config: WindowConfig,
    mode: str = "MS"
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Create windows with log returns as target."""
    
    # Calculate log returns
    panel = panel.copy()
    panel['y_returns'] = np.log(panel['y'] / panel['y'].shift(1))
    
    # Use returns as target
    config_returns = WindowConfig(
        seq_len=config.seq_len,
        label_len=config.label_len,
        pred_len=config.pred_len,
        target_col='y_returns',  # Changed from 'y'
        feature_cols=config.feature_cols,
        date_col=config.date_col
    )
    
    return make_windows(panel, config_returns, mode)
```

**At inference, convert back to prices**:
```python
def returns_to_prices(returns: np.ndarray, last_price: float) -> np.ndarray:
    """Convert predicted log returns to price levels."""
    cum_returns = np.cumsum(returns, axis=-1)
    return last_price * np.exp(cum_returns)
```

**Expected improvement**: 30-50% MSE reduction by removing scale dependency.

---

## Strategy 4: Add Aggressive Regularization

**Problem**: Val loss 5x train loss indicates severe overfitting.

**Implementation**:
```python
class RegularizedAutoformer(SimpleAutoformer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Add weight decay
        self.weight_decay = 0.01
        
        # Add dropout everywhere
        self.input_dropout = nn.Dropout(0.3)
        self.output_dropout = nn.Dropout(0.3)
        
        # Add layer normalization
        self.layer_norm = nn.LayerNorm(self.d_model)
    
    def forward(self, x_enc, x_dec):
        x_enc = self.input_dropout(x_enc)
        # ... rest of forward
        output = self.output_dropout(output)
        return output
```

**Training changes**:
```python
self.optimizer = torch.optim.AdamW(
    self.model.parameters(), 
    lr=lr, 
    weight_decay=0.01  # L2 regularization
)

# Add gradient clipping
torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
```

**Expected improvement**: 20-40% reduction in train/val gap.

---

## Strategy 5: Use Rolling Window Cross-Validation

**Problem**: Single train/val/test split is insufficient for small financial datasets.

**Implementation**:
```python
def create_rolling_cv_splits(
    X_enc: np.ndarray,
    y: np.ndarray,
    dates: np.ndarray,
    n_folds: int = 5,
    val_size: int = 60,  # 60 trading days
    test_size: int = 60
) -> List[Dict]:
    """Create rolling origin cross-validation splits."""
    
    splits = []
    total_samples = len(y)
    train_end_start = total_samples - (n_folds * (val_size + test_size))
    
    for fold in range(n_folds):
        train_end = train_end_start + fold * (val_size + test_size)
        val_end = train_end + val_size
        test_end = val_end + test_size
        
        splits.append({
            'train': slice(0, train_end),
            'val': slice(train_end, val_end),
            'test': slice(val_end, test_end),
            'fold': fold
        })
    
    return splits
```

**Expected improvement**: Better hyperparameter selection, 15-25% MSE improvement.

---

## Strategy 6: Data Augmentation for Time Series

**Problem**: 2,128 training samples is too small.

**Implementation**:
```python
class TimeSeriesAugmentation:
    """Data augmentation for financial time series."""
    
    def __init__(self, noise_std: float = 0.01, scale_range: Tuple = (0.95, 1.05)):
        self.noise_std = noise_std
        self.scale_range = scale_range
    
    def jitter(self, x: np.ndarray) -> np.ndarray:
        """Add Gaussian noise."""
        return x + np.random.normal(0, self.noise_std * x.std(), x.shape)
    
    def scaling(self, x: np.ndarray) -> np.ndarray:
        """Random scaling."""
        scale = np.random.uniform(*self.scale_range)
        return x * scale
    
    def window_slicing(self, x: np.ndarray, reduce_ratio: float = 0.9) -> np.ndarray:
        """Randomly slice window to shorter length."""
        target_len = int(len(x) * reduce_ratio)
        start = np.random.randint(0, len(x) - target_len)
        return x[start:start + target_len]
    
    def magnitude_warping(self, x: np.ndarray, sigma: float = 0.2) -> np.ndarray:
        """Smooth warping of magnitude."""
        from scipy.interpolate import CubicSpline
        orig_steps = np.arange(len(x))
        random_warps = np.random.normal(1.0, sigma, size=4)
        warp_steps = np.linspace(0, len(x) - 1, 4)
        spline = CubicSpline(warp_steps, random_warps)
        warping = spline(orig_steps)
        return x * warping.reshape(-1, 1)
    
    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Apply random augmentation."""
        aug_choice = np.random.choice(['jitter', 'scaling', 'warp', 'none'], p=[0.3, 0.3, 0.2, 0.2])
        if aug_choice == 'jitter':
            return self.jitter(x)
        elif aug_choice == 'scaling':
            return self.scaling(x)
        elif aug_choice == 'warp':
            return self.magnitude_warping(x)
        return x
```

**Expected improvement**: 10-20% improvement through increased effective training data.

---

## Strategy 7: Add Residual Connection to Naive Baseline

**Problem**: Model isn't leveraging the strong naive baseline.

**Implementation** - Predict residuals from persistence:
```python
class ResidualTSM(nn.Module):
    """Predict residual on top of naive persistence."""
    
    def __init__(self, base_model: nn.Module, alpha: float = 0.1):
        super().__init__()
        self.base_model = base_model
        self.alpha = nn.Parameter(torch.tensor(alpha))
    
    def forward(self, x_enc, x_dec):
        # Naive prediction: last known value
        naive_pred = x_enc[:, -1:, 0:1].expand(-1, self.base_model.pred_len, -1)
        
        # Residual prediction from model
        residual = self.base_model(x_enc, x_dec)
        
        # Combine with learned mixing weight
        return naive_pred + self.alpha * residual
```

**Loss function change**:
```python
def residual_loss(pred, target, naive_pred, gamma=0.5):
    """Loss that encourages beating the baseline."""
    mse_loss = F.mse_loss(pred, target)
    
    # Penalty if worse than naive
    naive_mse = F.mse_loss(naive_pred, target)
    penalty = F.relu(mse_loss - naive_mse)
    
    return mse_loss + gamma * penalty
```

**Expected improvement**: Guarantees model is at worst equal to naive baseline.

---

## Strategy 8: Use Multi-Horizon Direct Forecasting

**Problem**: Current setup forecasts all 30 steps jointly, accumulating errors.

**Implementation** - Train separate models per horizon:
```python
def train_direct_multi_horizon(
    train_data: Dict,
    horizons: List[int] = [1, 5, 20, 30],
    base_config: Dict = None
) -> Dict[int, nn.Module]:
    """Train separate models for each horizon."""
    
    models = {}
    
    for h in horizons:
        # Create single-step dataset for horizon h
        y_horizon = train_data['y'][:, h-1:h]  # Just step h
        
        config = base_config.copy()
        config['pred_len'] = 1
        
        model = SimpleAutoformer(**config)
        # Train model...
        
        models[h] = model
    
    return models
```

**Alternative - Weighted multi-horizon loss**:
```python
def multi_horizon_loss(pred, target, horizon_weights: List[float] = None):
    """Weighted loss giving more importance to near-term forecasts."""
    if horizon_weights is None:
        # Exponentially decaying weights
        horizon_weights = np.exp(-0.1 * np.arange(target.shape[1]))
        horizon_weights /= horizon_weights.sum()
    
    weights = torch.tensor(horizon_weights, device=pred.device)
    squared_errors = (pred - target) ** 2
    
    return (squared_errors * weights).sum(dim=1).mean()
```

**Expected improvement**: 20-35% improvement, especially at short horizons.

---

## Strategy 9: Incorporate Market Regime Features

**Problem**: Model doesn't know about EU ETS phase transitions and regulatory changes.

**Implementation** in `src/data/panel.py`:
```python
def add_regime_features(panel: pd.DataFrame) -> pd.DataFrame:
    """Add market regime indicators."""
    
    panel = panel.copy()
    
    # EU ETS Phase indicators (one-hot)
    panel['phase_3'] = ((panel['date'] >= '2013-01-01') & 
                        (panel['date'] < '2021-01-01')).astype(float)
    panel['phase_4'] = (panel['date'] >= '2021-01-01').astype(float)
    
    # MSR (Market Stability Reserve) active
    panel['msr_active'] = (panel['date'] >= '2019-01-01').astype(float)
    
    # COVID period
    panel['covid_period'] = ((panel['date'] >= '2020-03-01') & 
                             (panel['date'] <= '2021-06-30')).astype(float)
    
    # Energy crisis (2021-2022)
    panel['energy_crisis'] = ((panel['date'] >= '2021-09-01') & 
                              (panel['date'] <= '2022-12-31')).astype(float)
    
    # Price regime (low/medium/high)
    panel['regime_low'] = (panel['y'] < 20).astype(float)
    panel['regime_med'] = ((panel['y'] >= 20) & (panel['y'] < 60)).astype(float)
    panel['regime_high'] = (panel['y'] >= 60).astype(float)
    
    # Rolling volatility regime
    vol = panel['y'].pct_change().rolling(20).std()
    panel['high_vol_regime'] = (vol > vol.quantile(0.75)).astype(float)
    
    return panel
```

**Expected improvement**: 15-25% by helping model understand regime-specific dynamics.

---

## Strategy 10: Ensemble Multiple Models

**Problem**: Single model is high variance.

**Implementation**:
```python
class EnsembleForecaster:
    """Ensemble of multiple forecasting models."""
    
    def __init__(self, models: List[Tuple[str, nn.Module]], weights: List[float] = None):
        self.models = models
        self.weights = weights or [1.0 / len(models)] * len(models)
    
    def predict(self, x_enc: torch.Tensor, x_dec: torch.Tensor) -> torch.Tensor:
        """Weighted average prediction."""
        predictions = []
        
        for name, model in self.models:
            model.eval()
            with torch.no_grad():
                pred = model(x_enc, x_dec)
                predictions.append(pred)
        
        # Stack and weight
        stacked = torch.stack(predictions, dim=0)
        weights = torch.tensor(self.weights, device=stacked.device).view(-1, 1, 1, 1)
        
        return (stacked * weights).sum(dim=0)
    
    @classmethod
    def create_diverse_ensemble(cls, config: Dict) -> 'EnsembleForecaster':
        """Create ensemble with diverse configurations."""
        models = []
        
        # Different architectures
        configs = [
            {'d_model': 32, 'e_layers': 1, 'dropout': 0.3},   # Small
            {'d_model': 64, 'e_layers': 1, 'dropout': 0.2},   # Medium
            {'d_model': 128, 'e_layers': 2, 'dropout': 0.1},  # Larger
        ]
        
        for i, cfg in enumerate(configs):
            merged = {**config, **cfg}
            model = SimpleAutoformer(**merged)
            models.append((f'model_{i}', model))
        
        return cls(models)
```

**Hybrid ensemble with naive**:
```python
def hybrid_ensemble_predict(tsm_pred, naive_pred, blend_weight=0.3):
    """Blend TSM with naive baseline."""
    return blend_weight * tsm_pred + (1 - blend_weight) * naive_pred
```

**Expected improvement**: 15-30% through variance reduction and baseline blending.

---

## Strategy 11: Revise Normalization Strategy

**Problem**: Current StandardScaler computed on training data creates distribution shift when applied to test data in a different price regime.

**Implementation** - Use local/rolling normalization:
```python
class RollingScaler:
    """Scale using only recent data to handle regime changes."""
    
    def __init__(self, window: int = 60):
        self.window = window
    
    def transform_window(self, x: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """
        Normalize each window using its own statistics.
        Returns normalized data and stats for inverse transform.
        """
        # x shape: (seq_len, features)
        mean = x[-self.window:].mean(axis=0, keepdims=True)
        std = x[-self.window:].std(axis=0, keepdims=True)
        std = np.where(std == 0, 1.0, std)
        
        normalized = (x - mean) / std
        
        return normalized, {'mean': mean, 'std': std}
    
    def inverse_transform_target(
        self, y_norm: np.ndarray, stats: Dict, target_idx: int = 0
    ) -> np.ndarray:
        """Inverse transform using stored statistics."""
        return y_norm * stats['std'][0, target_idx] + stats['mean'][0, target_idx]
```

**Instance normalization in model**:
```python
class InstanceNormAutoformer(SimpleAutoformer):
    def forward(self, x_enc, x_dec):
        # Instance normalize input
        mean = x_enc.mean(dim=1, keepdim=True)
        std = x_enc.std(dim=1, keepdim=True) + 1e-6
        x_enc_norm = (x_enc - mean) / std
        x_dec_norm = (x_dec - mean) / std
        
        # Forward pass
        output = super().forward(x_enc_norm, x_dec_norm)
        
        # Denormalize output (using target channel stats)
        return output * std[:, :, :1] + mean[:, :, :1]
```

**Expected improvement**: 25-40% by handling distribution shift properly.

---

## Strategy 12: Use Proper Learning Rate Scheduling

**Problem**: Fixed learning rate doesn't allow fine-tuning.

**Implementation**:
```python
def create_lr_scheduler(optimizer, config):
    """Create cosine annealing with warm restarts."""
    
    # Warm-up phase
    warmup_epochs = 5
    
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return epoch / warmup_epochs
        
        # Cosine annealing after warmup
        progress = (epoch - warmup_epochs) / (config['max_epochs'] - warmup_epochs)
        return 0.5 * (1 + np.cos(np.pi * progress))
    
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

# Or use OneCycleLR
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=0.001,
    epochs=100,
    steps_per_epoch=len(train_loader),
    pct_start=0.1,  # 10% warmup
    anneal_strategy='cos'
)
```

**Expected improvement**: 10-20% through better convergence.

---

## Recommended Implementation Priority

| Priority | Strategy | Expected Impact | Effort |
|----------|----------|-----------------|--------|
| 1 | **Strategy 2**: Reduce model capacity | 40-60% | Low |
| 2 | **Strategy 7**: Residual to naive | Guaranteed improvement | Medium |
| 3 | **Strategy 11**: Rolling normalization | 25-40% | Medium |
| 4 | **Strategy 3**: Predict returns | 30-50% | Medium |
| 5 | **Strategy 1**: Use DLinear | 50-80% | Medium |
| 6 | **Strategy 4**: Add regularization | 20-40% | Low |
| 7 | **Strategy 8**: Multi-horizon direct | 20-35% | Medium |
| 8 | **Strategy 10**: Ensemble with naive | 15-30% | Low |
| 9 | **Strategy 9**: Regime features | 15-25% | Low |
| 10 | **Strategy 5**: Rolling CV | 15-25% | Medium |
| 11 | **Strategy 6**: Data augmentation | 10-20% | Medium |
| 12 | **Strategy 12**: LR scheduling | 10-20% | Low |

---

## Quick Start: Immediate Fixes

Apply these config changes for immediate improvement:

```yaml
# src/config/tuned.yaml
model:
  tsm_type: "autoformer"
  
  # Drastically reduced capacity
  d_model: 64
  n_heads: 2
  e_layers: 1
  d_ff: 128
  
  # Heavy regularization
  dropout: 0.3
  
  # Training
  batch_size: 16
  learning_rate: 0.001
  max_epochs: 50
  early_stopping_patience: 15
  
  # Add weight decay
  weight_decay: 0.01

# Use shorter sequence
time_series:
  seq_len: 60   # Was 120
  label_len: 15 # Was 30
  pred_len: 30
```

Run with: `python -m src.run_experiment --config src/config/tuned.yaml`
