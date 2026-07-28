from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from tools import benchmark_frozen_compact as bench


def _write_split(
    run_dir: Path,
    name: str,
    panel: pd.DataFrame,
    date_indices: list[int],
    pred_len: int,
    seed: int,
) -> None:
    rng = np.random.default_rng(seed)
    prices = panel["y"].to_numpy(dtype=float)
    dates = pd.to_datetime(panel["date"])
    histories = []
    targets = []
    target_dates = []
    for idx in date_indices:
        base_and_future = prices[idx - 1 : idx + pred_len]
        targets.append(np.diff(np.log(base_and_future)))
        target_dates.append(str(dates.iloc[idx].date()))
        target_history = np.diff(np.log(prices[idx - 6 : idx]))
        exogenous = rng.normal(size=(5, 2))
        histories.append(np.column_stack([target_history, exogenous]))

    np.savez(
        run_dir / "data/datasets" / f"{name}.npz",
        X_enc=np.asarray(histories, dtype=np.float64),
        X_dec=np.zeros((len(histories), 2, 3), dtype=np.float64),
        y=np.asarray(targets, dtype=np.float64),
        dates=np.asarray(target_dates),
    )


def _make_synthetic_run(tmp_path: Path) -> Path:
    run_dir = tmp_path / "synthetic_run"
    (run_dir / "data/datasets").mkdir(parents=True)
    (run_dir / "results").mkdir(parents=True)

    dates = pd.date_range("2020-01-01", periods=150, freq="D")
    trend = np.linspace(0.0, 0.24, len(dates))
    cycle = 0.018 * np.sin(np.arange(len(dates)) / 7.0)
    panel = pd.DataFrame({"date": dates, "y": 40.0 * np.exp(trend + cycle)})
    panel.to_csv(run_dir / "data/panel.csv", index=False)

    with (run_dir / "data/datasets/scaler.pkl").open("wb") as handle:
        pickle.dump(
            {
                "mean_": np.array([0.0, 0.0, 0.0]),
                "std_": np.array([1.0, 1.0, 1.0]),
                "feature_names_": ["y_return", "feature_a", "feature_b"],
            },
            handle,
        )

    _write_split(run_dir, "train", panel, list(range(20, 45)), pred_len=4, seed=1)
    _write_split(run_dir, "val", panel, list(range(65, 78)), pred_len=4, seed=2)
    _write_split(run_dir, "test", panel, list(range(105, 118)), pred_len=4, seed=3)

    with (run_dir / "config_resolved.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump({"target": {"mode": "returns"}}, handle)

    pd.DataFrame(
        [
            {"model": "tsm", "subset": "full", "mse_path": 2.0},
            {"model": "linear_ridge", "subset": "full", "mse_path": 1.5},
            {"model": "naive_persistence", "subset": "full", "mse_path": 3.0},
        ]
    ).to_csv(run_dir / "results/path_metrics.csv", index=False)
    return run_dir


def test_price_reconstruction_supports_log_and_simple_returns() -> None:
    base = np.array([100.0])
    daily_log = np.log(np.array([[1.01, 0.99, 1.02]]))
    expected_log = 100.0 * np.cumprod(np.array([[1.01, 0.99, 1.02]]), axis=1)
    np.testing.assert_allclose(bench.prices_from_daily(daily_log, base, "log"), expected_log)

    daily_simple = np.array([[0.01, -0.01, 0.02]])
    expected_simple = 100.0 * np.cumprod(1.0 + daily_simple, axis=1)
    np.testing.assert_allclose(
        bench.prices_from_daily(daily_simple, base, "simple"), expected_simple
    )


def test_synthetic_benchmark_is_aligned_and_validation_selected(tmp_path: Path) -> None:
    run_dir = _make_synthetic_run(tmp_path)
    result, validation = bench.benchmark_run(run_dir, seed=42)

    assert result["transform"] == "log"
    assert result["alignment_mae"] < 1e-10
    assert np.isfinite(result["mse_path"])
    assert 0.0 <= result["shrinkage"] <= 1.0
    assert result["reported_tsm_mse"] == 2.0
    assert result["reported_linear_ridge_mse"] == 1.5
    assert len(validation) == 36
    assert {row["run_id"] for row in validation} == {"synthetic_run"}
