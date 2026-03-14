from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.eval.residual_event_model import (
    ResidualCandidate,
    block_columns,
    build_origin_feature_frame,
    interpolate_anchor_residuals,
    make_candidate_predictions,
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
