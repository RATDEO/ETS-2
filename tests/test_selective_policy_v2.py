from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


bench = _load("benchmark_selective_policy_v2", ROOT / "tools/benchmark_selective_policy_v2.py")


def synthetic_frame(seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    windows = ["W4", "W3", "W2", "W1"]
    starts = pd.to_datetime(["2022-01-01", "2023-01-01", "2024-01-01", "2025-01-01"])
    rows = []
    for window, start in zip(windows, starts):
        for i in range(60):
            signal = rng.normal()
            adjustment = signal + rng.normal(scale=0.15)
            base = 4.0 + rng.gamma(2.0, 1.0)
            # LLM helps when the safe signal is positive and harms otherwise.
            uplift = 0.8 * signal + rng.normal(scale=0.15)
            llm = max(0.001, base - uplift)
            rows.append(
                {
                    "window": window,
                    "date": start + pd.Timedelta(days=i),
                    "base_path_mse": base,
                    "llm_path_mse": llm,
                    "path_uplift_abs": base - llm,
                    "llm_applied": 1,
                    "asof_y": 50 + signal,
                    "asof_y_return": signal,
                    "base_move_h20_pct": signal,
                    "base_move_h30_pct": signal * 1.1,
                    "adjust_h1": 0.0,
                    "adjust_h5": adjustment * 0.2,
                    "adjust_h20": adjustment,
                    "adjust_h30": adjustment * 1.2,
                    "apply_prompt_length": 6000,
                    "teaching_examples": 8,
                    "matched_teaching_count": 4,
                    "support_example_count": 2,
                    "positive_memory_count": 1,
                    "negative_memory_count": 1,
                    "dynamic_frozen_horizons": 0,
                }
            )
    return pd.DataFrame(rows)


def test_causal_policy_uses_only_older_windows() -> None:
    frame = synthetic_frame()
    results, decisions, selections, features = bench.run_policy(frame, "precall", seed=42)
    assert results["test_window"].tolist() == ["W2", "W1"]
    assert (pd.to_datetime(results["train_max_date"]) < pd.to_datetime(results["test_min_date"])).all()
    assert results["selected_mse"].mean() < results["base_mse"].mean()
    assert "asof_y_return" in features
    assert not decisions.empty
    assert not selections.empty


def test_postcall_policy_excludes_outcomes_and_future_biases() -> None:
    frame = synthetic_frame()
    features = bench.resolve_features(frame, "postcall")
    assert not set(features).intersection(bench.OUTCOME_COLUMNS)
    assert not any("bias" in name for name in features)
    results, *_ = bench.run_policy(frame, "postcall", seed=42)
    assert results["selected_mse"].mean() < results["base_mse"].mean()


def test_block_bootstrap_is_deterministic() -> None:
    frame = synthetic_frame()
    _, decisions, _, _ = bench.run_policy(frame, "postcall", seed=42)
    a = bench.moving_block_bootstrap(decisions, n_bootstrap=100, block_size=10, seed=123)
    b = bench.moving_block_bootstrap(decisions, n_bootstrap=100, block_size=10, seed=123)
    assert a == b
