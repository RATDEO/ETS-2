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
from .paper_replication import (
    PaperReplicationCandidate,
    aggregate_fold_rows,
    build_outer_folds,
    score_paper_candidate,
)
from .residual_event_model import (
    ResidualCandidate,
    block_columns,
    build_origin_feature_frame,
    extract_anchor_residual_targets,
    interpolate_anchor_residuals,
    make_candidate_predictions,
    summarize_prediction,
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
    "PaperReplicationCandidate",
    "build_outer_folds",
    "score_paper_candidate",
    "aggregate_fold_rows",
    "ResidualCandidate",
    "block_columns",
    "build_origin_feature_frame",
    "extract_anchor_residual_targets",
    "interpolate_anchor_residuals",
    "make_candidate_predictions",
    "summarize_prediction",
    "paired_t_test",
    "wilcoxon_test",
    "run_significance_tests",
]
