"""Evaluation modules for EU ETS forecasting."""
from .metrics import (
    mse, rmse, mae, mape,
    compute_all_metrics,
    compute_metrics_by_horizon
)
from .trend_classification import (
    classify_trend,
    compute_trend_accuracy
)
from .significance import (
    paired_t_test,
    wilcoxon_test,
    run_significance_tests
)

__all__ = [
    "mse", "rmse", "mae", "mape",
    "compute_all_metrics",
    "compute_metrics_by_horizon",
    "classify_trend",
    "compute_trend_accuracy",
    "paired_t_test",
    "wilcoxon_test",
    "run_significance_tests",
]
