"""Models for EU ETS forecasting."""
from .baselines import (
    NaivePersistence,
    SeasonalNaive,
    LinearBaseline,
    ARIMABaseline
)

try:
    from .tsm import SimpleAttentionForecaster, SimpleAutoformer, TSMForecaster
    TSM_AVAILABLE = True
except ImportError:
    TSM_AVAILABLE = False
    SimpleAttentionForecaster = None
    SimpleAutoformer = None
    TSMForecaster = None

__all__ = [
    "NaivePersistence",
    "SeasonalNaive",
    "LinearBaseline",
    "ARIMABaseline",
    "SimpleAttentionForecaster",
    "SimpleAutoformer",
    "TSMForecaster",
    "TSM_AVAILABLE",
]
