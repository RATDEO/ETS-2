from __future__ import annotations

from pathlib import Path

import numpy as np

from tools.benchmark_frozen_compact import load_split


def test_load_split_accepts_trusted_historical_object_dates(tmp_path: Path) -> None:
    path = tmp_path / "split.npz"
    np.savez(
        path,
        X_enc=np.zeros((2, 3, 1), dtype=np.float32),
        y=np.zeros((2, 4), dtype=np.float32),
        dates=np.array(["2024-01-02", "2024-01-03"], dtype=object),
    )
    split = load_split(path, mean=0.0, std=1.0)
    assert split.dates.tolist() == ["2024-01-02", "2024-01-03"]
    assert split.x.shape == (2, 3, 1)
    assert split.y.shape == (2, 4)
