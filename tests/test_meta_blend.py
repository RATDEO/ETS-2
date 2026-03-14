from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.eval.meta_blend import MetaBlender, StatefulMetaBlender, rolling_meta_blend_cv


def test_meta_blender_fits_and_predicts_same_shape():
    rng = np.random.default_rng(42)
    tsm = rng.normal(size=(40, 3))
    llm = tsm + 0.1 * rng.normal(size=(40, 3))
    y = 0.7 * tsm + 0.3 * llm + 0.01 * rng.normal(size=(40, 3))

    blender = MetaBlender(alphas=[1e-4, 1e-2, 1.0], n_folds=3)
    blender.fit(tsm, llm, y)
    pred = blender.predict(tsm, llm)

    assert pred.shape == y.shape
    assert len(blender.summary_rows()) == 3


def test_stateful_meta_blender_fits_and_predicts_same_shape():
    rng = np.random.default_rng(123)
    tsm = rng.normal(size=(40, 3))
    llm = tsm + 0.1 * rng.normal(size=(40, 3))
    state = rng.normal(size=(40, 2))
    y = 0.6 * tsm + 0.4 * llm + 0.02 * rng.normal(size=(40, 3))

    blender = StatefulMetaBlender(alphas=[1e-4, 1e-2, 1.0], n_folds=3)
    blender.fit(tsm, llm, y, state)
    pred = blender.predict(tsm, llm, state)

    assert pred.shape == y.shape
    assert len(blender.summary_rows()) == 3


def test_rolling_meta_blend_cv_only_predicts_outer_validation_rows():
    rng = np.random.default_rng(7)
    tsm = rng.normal(size=(24, 2))
    llm = tsm + 0.1 * rng.normal(size=(24, 2))
    y = 0.8 * tsm + 0.2 * llm + 0.01 * rng.normal(size=(24, 2))

    pred, folds = rolling_meta_blend_cv(tsm, llm, y, n_outer_folds=4, n_inner_folds=3)

    assert pred.shape == y.shape
    assert len(folds) == 3
    assert np.isnan(pred[:6]).all()
    assert np.isfinite(pred[6:]).all()
