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
    ResidualModelBundle,
    ScalarRampCandidate,
    ScalarRampModelBundle,
    WalkForwardFold,
    block_columns,
    build_origin_feature_frame,
    apply_scalar_ramp,
    extract_optimal_ramp_targets,
    extract_anchor_residual_targets,
    fit_residual_candidate,
    fit_scalar_ramp_candidate,
    interpolate_anchor_residuals,
    make_candidate_predictions,
    predict_residual_candidate,
    predict_scalar_ramp_candidate,
    ramp_weights,
    summarize_prediction,
    walk_forward_candidate_predictions,
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
    "ResidualModelBundle",
    "ScalarRampCandidate",
    "ScalarRampModelBundle",
    "WalkForwardFold",
    "block_columns",
    "build_origin_feature_frame",
    "apply_scalar_ramp",
    "extract_optimal_ramp_targets",
    "extract_anchor_residual_targets",
    "fit_residual_candidate",
    "fit_scalar_ramp_candidate",
    "interpolate_anchor_residuals",
    "make_candidate_predictions",
    "predict_residual_candidate",
    "predict_scalar_ramp_candidate",
    "ramp_weights",
    "summarize_prediction",
    "walk_forward_candidate_predictions",
    "paired_t_test",
    "wilcoxon_test",
    "run_significance_tests",
]
