"""
Time Series Model (TSM) implementation.

Implements Autoformer-style architecture for 30-step daily forecasting.
Also includes DLinear as a simpler alternative that often outperforms
complex transformer models on time series benchmarks.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path
import logging
import pickle

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

logger = logging.getLogger(__name__)


if TORCH_AVAILABLE:
    
    class DLinear(nn.Module):
        """
        DLinear: A simple yet effective baseline for time series forecasting.
        
        Paper: "Are Transformers Effective for Time Series Forecasting?"
        Often outperforms complex transformer models with fewer parameters.
        """
        
        def __init__(
            self,
            seq_len: int,
            pred_len: int,
            enc_in: int,
            individual: bool = True,
            kernel_size: int = 25
        ):
            super().__init__()
            self.seq_len = seq_len
            self.pred_len = pred_len
            self.individual = individual
            self.channels = enc_in
            
            # Moving average for decomposition
            self.kernel_size = kernel_size
            padding = (kernel_size - 1) // 2
            self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=1, padding=padding)
            
            if individual:
                # Separate linear layers per channel
                self.Linear_Trend = nn.ModuleList([
                    nn.Linear(seq_len, pred_len) for _ in range(enc_in)
                ])
                self.Linear_Seasonal = nn.ModuleList([
                    nn.Linear(seq_len, pred_len) for _ in range(enc_in)
                ])
            else:
                # Shared linear layers
                self.Linear_Trend = nn.Linear(seq_len, pred_len)
                self.Linear_Seasonal = nn.Linear(seq_len, pred_len)
        
        def forward(self, x_enc, x_dec=None):
            """
            Forward pass.
            
            Args:
                x_enc: (batch, seq_len, channels)
                x_dec: ignored (for API compatibility)
                
            Returns:
                (batch, pred_len, 1) - only target channel
            """
            # Decompose: trend and seasonal
            x = x_enc.permute(0, 2, 1)  # (batch, channels, seq_len)
            trend = self.avg(x)
            trend = trend.permute(0, 2, 1)  # (batch, seq_len, channels)
            seasonal = x_enc - trend
            
            if self.individual:
                trend_out = torch.zeros(
                    x_enc.shape[0], self.pred_len, self.channels, 
                    device=x_enc.device
                )
                seasonal_out = torch.zeros_like(trend_out)
                
                for i in range(self.channels):
                    trend_out[:, :, i] = self.Linear_Trend[i](trend[:, :, i])
                    seasonal_out[:, :, i] = self.Linear_Seasonal[i](seasonal[:, :, i])
            else:
                trend_out = self.Linear_Trend(trend.permute(0, 2, 1)).permute(0, 2, 1)
                seasonal_out = self.Linear_Seasonal(seasonal.permute(0, 2, 1)).permute(0, 2, 1)
            
            output = trend_out + seasonal_out
            
            # Return only target channel (first channel)
            return output[:, :, :1]
    
    
    class ResidualWrapper(nn.Module):
        """
        Wrapper that adds residual connection to naive persistence.
        
        Ensures model can at worst match naive baseline.
        """
        
        def __init__(self, base_model: nn.Module, pred_len: int, alpha_init: float = 0.1):
            super().__init__()
            self.base_model = base_model
            self.pred_len = pred_len
            # Learnable blending weight (starts conservative)
            self.alpha = nn.Parameter(torch.tensor(alpha_init))
        
        def forward(self, x_enc, x_dec):
            # Naive prediction: repeat last known value
            last_value = x_enc[:, -1:, 0:1]  # (batch, 1, 1)
            naive_pred = last_value.expand(-1, self.pred_len, -1)  # (batch, pred_len, 1)
            
            # Model residual prediction
            residual = self.base_model(x_enc, x_dec)
            
            # Blend with constrained alpha (sigmoid to keep in [0, 1])
            blend = torch.sigmoid(self.alpha)
            
            return naive_pred + blend * residual
    
    
    class MovingAvg(nn.Module):
        """Moving average block for trend extraction."""
        
        def __init__(self, kernel_size: int, stride: int = 1):
            super().__init__()
            self.kernel_size = kernel_size
            self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)
        
        def forward(self, x):
            # x: (batch, seq_len, features)
            # Pad front to maintain sequence length
            front = x[:, :1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
            end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
            x = torch.cat([front, x, end], dim=1)
            
            # Pool requires (batch, features, seq_len)
            x = x.permute(0, 2, 1)
            x = self.avg(x)
            x = x.permute(0, 2, 1)
            return x
    
    
    class SeriesDecomp(nn.Module):
        """Series decomposition block."""
        
        def __init__(self, kernel_size: int):
            super().__init__()
            self.moving_avg = MovingAvg(kernel_size)
        
        def forward(self, x):
            trend = self.moving_avg(x)
            seasonal = x - trend
            return seasonal, trend
    
    
    class AutoCorrelation(nn.Module):
        """Auto-correlation mechanism for Autoformer."""
        
        def __init__(self, d_model: int, n_heads: int, d_keys: int = None, dropout: float = 0.1):
            super().__init__()
            
            d_keys = d_keys or (d_model // n_heads)
            
            self.n_heads = n_heads
            self.d_keys = d_keys
            
            self.query_projection = nn.Linear(d_model, d_keys * n_heads)
            self.key_projection = nn.Linear(d_model, d_keys * n_heads)
            self.value_projection = nn.Linear(d_model, d_keys * n_heads)
            self.out_projection = nn.Linear(d_keys * n_heads, d_model)
            
            self.dropout = nn.Dropout(dropout)
        
        def forward(self, queries, keys, values):
            B, L, _ = queries.shape
            _, S, _ = keys.shape
            H = self.n_heads
            
            queries = self.query_projection(queries).view(B, L, H, -1)
            keys = self.key_projection(keys).view(B, S, H, -1)
            values = self.value_projection(values).view(B, S, H, -1)
            
            # Simple attention (simplified from full autocorrelation)
            scale = 1. / np.sqrt(self.d_keys)
            scores = torch.einsum("blhe,bshe->bhls", queries, keys) * scale
            
            attn = self.dropout(F.softmax(scores, dim=-1))
            out = torch.einsum("bhls,bshd->blhd", attn, values)
            
            out = out.reshape(B, L, -1)
            return self.out_projection(out)
    
    
    class EncoderLayer(nn.Module):
        """Encoder layer with auto-correlation."""
        
        def __init__(
            self,
            d_model: int,
            n_heads: int,
            d_ff: int,
            kernel_size: int = 25,
            dropout: float = 0.1
        ):
            super().__init__()
            
            self.attention = AutoCorrelation(d_model, n_heads, dropout=dropout)
            self.decomp1 = SeriesDecomp(kernel_size)
            self.decomp2 = SeriesDecomp(kernel_size)
            
            self.ff = nn.Sequential(
                nn.Linear(d_model, d_ff),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_ff, d_model)
            )
            
            self.dropout = nn.Dropout(dropout)
        
        def forward(self, x):
            # Auto-correlation
            attn_out = self.attention(x, x, x)
            x = x + self.dropout(attn_out)
            x, _ = self.decomp1(x)
            
            # Feed-forward
            ff_out = self.ff(x)
            x = x + self.dropout(ff_out)
            x, _ = self.decomp2(x)
            
            return x
    
    
    class SimpleAutoformer(nn.Module):
        """
        Simplified Autoformer for time series forecasting.
        
        This is a streamlined version for 30-step prediction.
        """
        
        def __init__(
            self,
            enc_in: int,
            dec_in: int,
            c_out: int,
            seq_len: int,
            label_len: int,
            pred_len: int,
            d_model: int = 512,
            n_heads: int = 8,
            e_layers: int = 2,
            d_ff: int = 2048,
            dropout: float = 0.05,
            kernel_size: int = 25
        ):
            super().__init__()
            
            self.seq_len = seq_len
            self.label_len = label_len
            self.pred_len = pred_len
            
            # Embedding
            self.enc_embedding = nn.Linear(enc_in, d_model)
            self.dec_embedding = nn.Linear(dec_in, d_model)
            
            # Encoder
            self.encoder = nn.ModuleList([
                EncoderLayer(d_model, n_heads, d_ff, kernel_size, dropout)
                for _ in range(e_layers)
            ])
            
            # Decoder (simplified)
            self.decoder = nn.ModuleList([
                EncoderLayer(d_model, n_heads, d_ff, kernel_size, dropout)
                for _ in range(1)
            ])
            
            # Output projection
            self.projection = nn.Linear(d_model, c_out)
            
            # Decomposition
            self.decomp = SeriesDecomp(kernel_size)
        
        def forward(self, x_enc, x_dec):
            """
            Forward pass.
            
            Args:
                x_enc: Encoder input (batch, seq_len, enc_in)
                x_dec: Decoder input (batch, label_len + pred_len, dec_in)
                
            Returns:
                Predictions (batch, pred_len, c_out)
            """
            # Embed
            enc_out = self.enc_embedding(x_enc)
            dec_out = self.dec_embedding(x_dec)
            
            # Encode
            for layer in self.encoder:
                enc_out = layer(enc_out)
            
            # Decode (with cross-attention simplified to self-attention)
            for layer in self.decoder:
                dec_out = layer(dec_out)
            
            # Project to output
            output = self.projection(dec_out)
            
            # Return only prediction part
            return output[:, -self.pred_len:, :]
    
    
    class TSMForecaster:
        """
        Wrapper class for training and inference with TSM model.
        """
        
        def __init__(
            self,
            config: Dict,
            device: str = "auto"
        ):
            self.config = config
            
            # Determine device
            if device == "auto":
                if torch.cuda.is_available():
                    self.device = torch.device("cuda")
                elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                    self.device = torch.device("mps")
                else:
                    self.device = torch.device("cpu")
            else:
                self.device = torch.device(device)
            
            # Build model
            self.model = self._build_model()
            self.model.to(self.device)
            
            self.optimizer = None
            self.scheduler = None
            self.best_val_loss = float("inf")
        
        def _build_model(self) -> nn.Module:
            """Build the model from config."""
            model_config = self.config.get("model", {})
            ts_config = self.config.get("time_series", {})
            
            tsm_type = model_config.get("tsm_type", "autoformer").lower()
            seq_len = ts_config.get("seq_len", 120)
            pred_len = ts_config.get("pred_len", 30)
            enc_in = model_config.get("enc_in", 10)
            use_residual = model_config.get("use_residual_wrapper", False)
            
            logger.info(f"Building TSM model: type={tsm_type}, seq_len={seq_len}, pred_len={pred_len}")
            
            if tsm_type == "dlinear":
                # Simple but effective linear model
                base_model = DLinear(
                    seq_len=seq_len,
                    pred_len=pred_len,
                    enc_in=enc_in,
                    individual=model_config.get("dlinear_individual", True),
                    kernel_size=model_config.get("kernel_size", 25)
                )
                logger.info(f"Built DLinear model with {sum(p.numel() for p in base_model.parameters())} parameters")
                
            else:  # autoformer or default
                base_model = SimpleAutoformer(
                    enc_in=enc_in,
                    dec_in=model_config.get("dec_in", enc_in),
                    c_out=1,
                    seq_len=seq_len,
                    label_len=ts_config.get("label_len", 30),
                    pred_len=pred_len,
                    d_model=model_config.get("d_model", 512),
                    n_heads=model_config.get("n_heads", 8),
                    e_layers=model_config.get("e_layers", 2),
                    d_ff=model_config.get("d_ff", 2048),
                    dropout=model_config.get("dropout", 0.05)
                )
                logger.info(f"Built Autoformer model with {sum(p.numel() for p in base_model.parameters())} parameters")
            
            # Optionally wrap with residual connection to naive baseline
            if use_residual:
                base_model = ResidualWrapper(
                    base_model, 
                    pred_len=pred_len,
                    alpha_init=model_config.get("residual_alpha", 0.1)
                )
                logger.info("Wrapped model with residual connection to naive baseline")
            
            return base_model
        
        def fit(
            self,
            train_loader: DataLoader,
            val_loader: DataLoader,
            epochs: int = 100,
            patience: int = 10,
            save_path: Optional[Path] = None
        ) -> Dict:
            """
            Train the model.
            
            Args:
                train_loader: Training DataLoader
                val_loader: Validation DataLoader
                epochs: Maximum epochs
                patience: Early stopping patience
                save_path: Path to save best model
                
            Returns:
                Training history
            """
            model_config = self.config.get("model", {})
            lr = model_config.get("learning_rate", 0.0001)
            weight_decay = model_config.get("weight_decay", 0.0)
            grad_clip = model_config.get("grad_clip", 1.0)
            
            # Use AdamW for proper weight decay (L2 regularization)
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(), 
                lr=lr,
                weight_decay=weight_decay
            )
            
            # Cosine annealing with warm restarts
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self.optimizer, T_0=10, T_mult=2, eta_min=lr * 0.01
            )
            
            self.grad_clip = grad_clip
            criterion = nn.MSELoss()
            
            history = {"train_loss": [], "val_loss": []}
            no_improve = 0
            
            for epoch in range(epochs):
                # Training
                self.model.train()
                train_losses = []
                
                for batch in train_loader:
                    x_enc = batch["X_enc"].to(self.device)
                    x_dec = batch["X_dec"].to(self.device)
                    y = batch["y"].to(self.device)
                    
                    self.optimizer.zero_grad()
                    
                    output = self.model(x_enc, x_dec)
                    loss = criterion(output.squeeze(-1), y)
                    
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                    self.optimizer.step()
                    self.scheduler.step()
                    
                    train_losses.append(loss.item())
                
                train_loss = np.mean(train_losses)
                
                # Validation
                val_loss = self.evaluate(val_loader)
                
                self.scheduler.step(val_loss)
                
                history["train_loss"].append(train_loss)
                history["val_loss"].append(val_loss)
                
                logger.info(f"Epoch {epoch+1}/{epochs} - Train: {train_loss:.6f}, Val: {val_loss:.6f}")
                
                # Early stopping
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    no_improve = 0
                    if save_path:
                        self.save(save_path)
                else:
                    no_improve += 1
                    if no_improve >= patience:
                        logger.info(f"Early stopping at epoch {epoch+1}")
                        break
            
            return history
        
        def evaluate(self, loader: DataLoader) -> float:
            """Evaluate model on a DataLoader."""
            self.model.eval()
            losses = []
            criterion = nn.MSELoss()
            
            with torch.no_grad():
                for batch in loader:
                    x_enc = batch["X_enc"].to(self.device)
                    x_dec = batch["X_dec"].to(self.device)
                    y = batch["y"].to(self.device)
                    
                    output = self.model(x_enc, x_dec)
                    loss = criterion(output.squeeze(-1), y)
                    losses.append(loss.item())
            
            return np.mean(losses)
        
        def predict(
            self,
            loader: DataLoader
        ) -> Tuple[np.ndarray, np.ndarray]:
            """
            Generate predictions.
            
            Returns:
                Tuple of (predictions, targets)
            """
            self.model.eval()
            predictions = []
            targets = []
            
            with torch.no_grad():
                for batch in loader:
                    x_enc = batch["X_enc"].to(self.device)
                    x_dec = batch["X_dec"].to(self.device)
                    y = batch["y"]
                    
                    output = self.model(x_enc, x_dec)
                    
                    predictions.append(output.squeeze(-1).cpu().numpy())
                    targets.append(y.numpy())
            
            return np.concatenate(predictions), np.concatenate(targets)
        
        def save(self, path: Union[str, Path]):
            """Save model checkpoint."""
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            
            torch.save({
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict() if self.optimizer else None,
                "best_val_loss": self.best_val_loss,
                "config": self.config
            }, path)
        
        def load(self, path: Union[str, Path]):
            """Load model checkpoint."""
            checkpoint = torch.load(path, map_location=self.device)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.best_val_loss = checkpoint.get("best_val_loss", float("inf"))

else:
    class SimpleAutoformer:
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch is required for TSM models")
    
    class TSMForecaster:
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch is required for TSM models")
