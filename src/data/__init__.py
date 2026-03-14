"""Data loading and processing modules."""
from .load_indices import load_carbon_indices
from .load_auctions import load_auction_data
from .load_energy import load_energy_benchmarks
from .load_icap import load_icap_data, get_icap_target_series
from .load_eua_futures import load_eua_futures_data, get_eua_futures_target_series
from .load_vstoxx import load_vstoxx
from .fx import load_eurusd_fx, convert_usd_to_eur
from .panel import build_panel, select_feature_columns, add_sentiment_features, load_daily_sentiment_features
from .windows import make_windows, TimeSeriesDataset

__all__ = [
    "load_carbon_indices",
    "load_auction_data",
    "load_energy_benchmarks",
    "load_icap_data",
    "get_icap_target_series",
    "load_eua_futures_data",
    "get_eua_futures_target_series",
    "load_vstoxx",
    "load_eurusd_fx",
    "convert_usd_to_eur",
    "build_panel",
    "select_feature_columns",
    "add_sentiment_features",
    "load_daily_sentiment_features",
    "make_windows",
    "TimeSeriesDataset",
]
