from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.llm.refine import interpolate_horizon_anchor_prices, parse_horizon_anchor_prices


def test_parse_horizon_anchor_prices_reads_expected_keys():
    payload = '{"anchors": {"h1": 70.5, "h10": 71.2, "h20": 70.8, "h30": 70.4}}'
    parsed = parse_horizon_anchor_prices(payload)
    assert parsed == {1: 70.5, 10: 71.2, 20: 70.8, 30: 70.4}


def test_interpolate_horizon_anchor_prices_hits_key_points():
    path = interpolate_horizon_anchor_prices(
        30,
        {1: 70.5, 10: 71.2, 20: 70.8, 30: 70.4},
    )
    assert len(path) == 30
    assert np.isclose(path[0], 70.5)
    assert np.isclose(path[9], 71.2)
    assert np.isclose(path[19], 70.8)
    assert np.isclose(path[29], 70.4)
    assert 70.8 < path[14] < 71.2
