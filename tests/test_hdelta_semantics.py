from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.llm.refine import interpolate_horizon_adjustments, parse_horizon_deltas


def test_parse_horizon_deltas_reads_expected_keys():
    payload = '{"adjustments": {"h1": 0.3, "h5": -0.5, "h20": 0.8, "h30": -0.2}}'
    parsed = parse_horizon_deltas(payload)
    assert parsed == {1: 0.3, 5: -0.5, 20: 0.8, 30: -0.2}


def test_interpolate_horizon_adjustments_hits_key_points():
    path = interpolate_horizon_adjustments(
        30,
        {1: 0.3, 5: -0.5, 20: 0.8, 30: -0.2},
    )
    assert len(path) == 30
    assert np.isclose(path[0], 0.3)
    assert np.isclose(path[4], -0.5)
    assert np.isclose(path[19], 0.8)
    assert np.isclose(path[29], -0.2)
    assert -0.5 < path[10] < 0.8


def test_frozen_horizon_can_be_zeroed_before_interpolation():
    adjusted = {1: 0.3, 5: -0.5, 20: 0.8, 30: -0.2}
    adjusted[1] = 0.0
    path = interpolate_horizon_adjustments(30, adjusted)
    assert np.isclose(path[0], 0.0)
    assert np.isclose(path[4], -0.5)
