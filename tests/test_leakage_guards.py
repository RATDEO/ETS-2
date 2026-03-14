from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.windows import WindowConfig, make_windows, split_windows
from src.config.config import enforce_split_test_end


def test_decoder_future_slots_do_not_use_realized_future_rows():
    n = 12
    panel = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=n, freq="D"),
            "target_return": np.arange(n, dtype=np.float32) / 10.0,
            "price_level": np.arange(100, 100 + n, dtype=np.float32),
            "flow_pct": np.arange(n, dtype=np.float32) / 100.0,
            "is_event": np.ones(n, dtype=np.float32),
        }
    )
    cfg = WindowConfig(
        seq_len=4,
        label_len=2,
        pred_len=2,
        target_col="target_return",
        feature_cols=["target_return", "price_level", "flow_pct", "is_event"],
        date_col="date",
    )

    X_enc, X_dec, y, dates, meta = make_windows(panel, cfg, return_metadata=True)

    target_idx = cfg.feature_cols.index("target_return")
    level_idx = cfg.feature_cols.index("price_level")
    pct_idx = cfg.feature_cols.index("flow_pct")
    flag_idx = cfg.feature_cols.index("is_event")

    # First window: observed context uses rows 2:4, future target rows would be 4:6.
    future_decoder = X_dec[0, cfg.label_len :, :]
    realized_future = panel.loc[4:5, cfg.feature_cols].to_numpy(dtype=np.float32)
    last_observed = panel.loc[3, cfg.feature_cols].to_numpy(dtype=np.float32)

    np.testing.assert_allclose(y[0], panel.loc[4:5, "target_return"].to_numpy(dtype=np.float32))

    # Unknown future returns/flags must not equal realized future values.
    np.testing.assert_allclose(future_decoder[:, target_idx], 0.0)
    np.testing.assert_allclose(future_decoder[:, pct_idx], 0.0)
    np.testing.assert_allclose(future_decoder[:, flag_idx], 0.0)
    assert not np.allclose(future_decoder[:, target_idx], realized_future[:, target_idx])
    assert not np.allclose(future_decoder[:, pct_idx], realized_future[:, pct_idx])

    # Level-like features use last observed carry-forward, not realized future rows.
    np.testing.assert_allclose(future_decoder[:, level_idx], last_observed[level_idx])
    assert not np.allclose(future_decoder[:, level_idx], realized_future[:, level_idx])

    # Metadata should reflect the forecast window correctly.
    assert str(pd.Timestamp(dates[0]).date()) == "2024-01-05"
    assert str(pd.Timestamp(meta.pred_end_dates[0]).date()) == "2024-01-06"
    assert str(pd.Timestamp(meta.last_input_dates[0]).date()) == "2024-01-04"


def test_split_windows_drops_boundary_crossing_windows():
    n = 16
    panel = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=n, freq="D"),
            "y": np.arange(n, dtype=np.float32),
            "x": np.arange(100, 100 + n, dtype=np.float32),
        }
    )
    cfg = WindowConfig(
        seq_len=4,
        label_len=2,
        pred_len=3,
        target_col="y",
        feature_cols=["y", "x"],
        date_col="date",
    )

    X_enc, X_dec, y, dates, meta = make_windows(panel, cfg, return_metadata=True)
    splits = split_windows(
        X_enc,
        X_dec,
        y,
        dates,
        train_end="2024-01-08",
        val_end="2024-01-11",
        window_meta=meta,
    )

    total_assigned = sum(len(s["y"]) for s in splits.values())
    assert total_assigned < len(y)
    assert len(y) - total_assigned == 4

    assert (pd.to_datetime(splits["train"]["end_dates"]) <= pd.Timestamp("2024-01-08")).all()
    assert (pd.to_datetime(splits["val"]["dates"]) > pd.Timestamp("2024-01-08")).all()
    assert (pd.to_datetime(splits["val"]["end_dates"]) <= pd.Timestamp("2024-01-11")).all()
    assert (pd.to_datetime(splits["test"]["dates"]) > pd.Timestamp("2024-01-11")).all()


def test_enforce_split_test_end_sets_target_max_date_when_missing():
    raw = {
        "target": {"mode": "returns"},
        "split": {"test_end": "2026-03-04"},
    }

    normalized = enforce_split_test_end(raw)

    assert normalized["target"]["max_date"] == "2026-03-04"


def test_enforce_split_test_end_preserves_earlier_target_cap():
    raw = {
        "target": {"mode": "returns", "max_date": "2026-02-28"},
        "split": {"test_end": "2026-03-04"},
    }

    normalized = enforce_split_test_end(raw)

    assert normalized["target"]["max_date"] == "2026-02-28"
