"""Models for EU ETS forecasting."""
from .baselines import (
    NaivePersistence,
    SeasonalNaive,
    LinearBaseline,
    ARIMABaseline
)

try:
    from .tsm import SimpleAutoformer, TSMForecaster
    TSM_AVAILABLE = True
except ImportError:
    TSM_AVAILABLE = False
    SimpleAutoformer = None
    TSMForecaster = None

__all__ = [
    "NaivePersistence",
    "SeasonalNaive",
    "LinearBaseline",
    "ARIMABaseline",
    "SimpleAutoformer",
    "TSMForecaster",
    "TSM_AVAILABLE",
]
