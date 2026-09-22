from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.eval.residual_event_model import (
    ResidualCandidate,
    ScalarRampCandidate,
    apply_scalar_ramp,
    block_columns,
    build_origin_feature_frame,
    interpolate_anchor_residuals,
    fit_residual_candidate,
    extract_optimal_ramp_targets,
    fit_scalar_ramp_candidate,
    make_candidate_predictions,
    predict_residual_candidate,
    predict_scalar_ramp_candidate,
    walk_forward_candidate_predictions,
)


def test_interpolate_anchor_residuals_freezes_h1() -> None:
    anchors = np.array([[1.0, 2.0, 4.0, 6.0]], dtype=float)
    path = interpolate_anchor_residuals(anchors, pred_len=30, anchor_horizons=(5, 10, 20, 30), freeze_h1=True)
    assert path.shape == (1, 30)
    assert path[0, 0] == 0.0
    assert np.isclose(path[0, 4], 1.0)
    assert np.isclose(path[0, 9], 2.0)
    assert np.isclose(path[0, 19], 4.0)
    assert np.isclose(path[0, 29], 6.0)


def test_identity_candidate_returns_base_prediction() -> None:
    panel = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=2, freq="D"),
            "y": [70.0, 71.0],
            "y_return": [0.0, 0.01],
            "y_ma_5d": [70.0, 70.5],
            "y_ma_20d": [70.0, 70.5],
            "y_vol_20d": [1.0, 1.1],
            "y_momentum_5d": [0.0, 0.5],
            "y_momentum_20d": [0.0, 0.5],
        }
    )
    event_panel = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=2, freq="D"),
            "evt_news_count": [1.0, 0.0],
            "evt_sent_mean": [0.5, 0.0],
        }
    )
    base_pred = np.full((2, 30), 72.0)
    feature_frame = build_origin_feature_frame(panel, event_panel, panel["date"], base_pred)
    blocks = block_columns(feature_frame)
    X = feature_frame[blocks["base"]].to_numpy(dtype=float)

    candidate = ResidualCandidate(name="identity", estimator="identity", feature_blocks=("base",))
    pred = make_candidate_predictions(candidate, X_fit=X, y_fit_anchor=np.zeros((2, 4)), X_apply=X, base_apply_pred=base_pred)
    assert np.allclose(pred, base_pred)


def test_pct_candidate_respects_adjustment_cap() -> None:
    base_pred = np.full((2, 30), 100.0)
    X = np.ones((2, 1), dtype=float)
    y_fit = np.full((2, 4), 10.0, dtype=float)
    candidate = ResidualCandidate(
        name="ridge_pct",
        estimator="ridge",
        feature_blocks=("base",),
        target_kind="pct",
        alpha=1.0,
        max_adjustment_pct=5.0,
    )
    pred = make_candidate_predictions(candidate, X_fit=X, y_fit_anchor=y_fit, X_apply=X, base_apply_pred=base_pred)
    assert np.max(np.abs((pred - base_pred) / base_pred)) <= 0.050001


def test_fitted_bundle_round_trip_matches_one_shot_prediction() -> None:
    X = np.arange(24, dtype=float).reshape(8, 3)
    y_fit = np.column_stack([X[:, 0] * scale for scale in (0.001, 0.002, 0.003, 0.004)])
    base_pred = np.full((8, 30), 80.0)
    candidate = ResidualCandidate(
        name="ridge_pct_shrunk",
        estimator="ridge",
        feature_blocks=("base",),
        target_kind="pct",
        alpha=10.0,
        max_adjustment_pct=4.0,
        shrinkage=0.5,
    )
    expected = make_candidate_predictions(candidate, X, y_fit, X, base_pred)
    bundle = fit_residual_candidate(candidate, X, y_fit)
    actual = predict_residual_candidate(bundle, X, base_pred)
    assert np.allclose(actual, expected)


def test_walk_forward_predictions_are_strictly_out_of_fold() -> None:
    X = np.arange(12, dtype=float).reshape(-1, 1)
    y_anchor = np.column_stack([X[:, 0] * scale for scale in (0.01, 0.02, 0.03, 0.04)])
    base_pred = np.full((12, 30), 100.0)
    candidate = ResidualCandidate(
        name="ridge_pct",
        estimator="ridge",
        feature_blocks=("base",),
        target_kind="pct",
        max_adjustment_pct=5.0,
    )
    indices, predictions, folds = walk_forward_candidate_predictions(
        candidate,
        X,
        y_anchor,
        base_pred,
        initial_train_size=6,
        fold_size=2,
        purge_size=2,
    )
    assert indices.tolist() == list(range(6, 12))
    assert predictions.shape == (6, 30)
    assert [(fold.fit_end, fold.apply_start, fold.apply_end) for fold in folds] == [
        (4, 6, 8),
        (6, 8, 10),
        (8, 10, 12),
    ]
    assert all(fold.fit_end + 2 <= fold.apply_start for fold in folds)


def test_optimal_scalar_ramp_recovers_known_adjustment() -> None:
    base = np.full((3, 30), 100.0)
    known = np.array([-0.04, 0.0, 0.03])
    y_true = apply_scalar_ramp(base, known, max_adjustment_pct=5.0)
    recovered = extract_optimal_ramp_targets(y_true, base, max_adjustment_pct=5.0)
    assert np.allclose(recovered, known)
    assert np.allclose(y_true[:, 0], base[:, 0])


def test_scalar_ramp_bundle_respects_lookback_and_cap() -> None:
    X = np.arange(30, dtype=float).reshape(10, 3)
    y = np.linspace(-0.1, 0.1, 10)
    base = np.full((2, 30), 80.0)
    candidate = ScalarRampCandidate(
        name="ridge_qwen",
        estimator="ridge",
        feature_blocks=("base", "event_core"),
        lookback=6,
        alpha=100.0,
        max_adjustment_pct=2.0,
        shrinkage=1.5,
    )
    bundle = fit_scalar_ramp_candidate(candidate, X, y)
    pred = predict_scalar_ramp_candidate(bundle, X[-2:], base)
    assert pred.shape == base.shape
    assert np.allclose(pred[:, 0], base[:, 0])
    assert np.max(np.abs((pred - base) / base)) <= 0.020001
