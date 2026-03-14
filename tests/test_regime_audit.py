from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.eval.regime_audit import (
    apply_fixed_gate,
    build_origin_regime_frame,
    gate_candidate_summary,
    overall_metrics,
    regime_metric_tables,
)


def test_overall_metrics_orders_models_by_path_mse():
    y_true = np.array([[1.0, 2.0], [2.0, 3.0]])
    preds = {
        "worse": np.array([[2.0, 3.0], [3.0, 4.0]]),
        "better": np.array([[1.1, 2.1], [2.1, 3.1]]),
    }
    out = overall_metrics(y_true, preds, horizons=[1, 2])
    assert list(out["model"]) == ["better", "worse"]


def test_build_origin_regime_frame_labels_expected_columns(tmp_path: Path):
    data_dir = tmp_path / "Data"
    futures_dir = data_dir / "eua-futures"
    futures_dir.mkdir(parents=True)
    futures_csv = futures_dir / "European Union Allowance (EUA) Yearly Futures Historical Data AUTO.csv"
    futures_csv.write_text(
        "\n".join(
            [
                '"Date","Price","Open","High","Low","Vol.","Change %"',
                '"01/01/2024","10.0","10.0","10.5","9.5","1K","0.0%"',
                '"02/01/2024","10.2","10.2","10.7","10.0","1K","2.0%"',
                '"03/01/2024","10.4","10.4","10.8","10.1","1K","2.0%"',
                '"04/01/2024","10.1","10.1","10.5","9.9","1K","-2.9%"',
                '"05/01/2024","10.3","10.3","10.7","10.0","1K","2.0%"',
                '"08/01/2024","10.5","10.5","10.9","10.3","1K","1.9%"',
                '"09/01/2024","10.8","10.8","11.0","10.5","1K","2.9%"',
                '"10/01/2024","10.6","10.6","10.9","10.4","1K","-1.9%"',
            ]
        )
    )
    sentiment_path = tmp_path / "sentiment.csv"
    sentiment_df = pd.DataFrame(
        {
            "seendate": pd.date_range("2024-01-01", periods=8, freq="D"),
            "sent_score": [0.1, 0.2, 0.25, -0.2, -0.3, 0.4, 0.5, -0.1],
        }
    )
    sentiment_df.to_csv(sentiment_path, index=False)
    official_path = tmp_path / "official_events.csv"
    official_df = pd.DataFrame(
        {
            "event_date": ["2024-01-04", "2024-01-08"],
            "title": ["Verified emissions update", "Auction result heavy supply"],
            "event_type": ["verified_emissions", "auction_supply"],
            "affected_channel": ["compliance_demand", "allowance_supply"],
            "direction": ["bullish", "bearish"],
            "intensity": [2.0, 3.0],
            "confidence": [1.0, 1.0],
        }
    )
    official_df.to_csv(official_path, index=False)

    origins = pd.to_datetime(["2024-01-05", "2024-01-08", "2024-01-10"])
    frame, thresholds = build_origin_regime_frame(
        origins,
        data_dir=data_dir,
        sentiment_path=sentiment_path,
        official_event_path=official_path,
        volatility_window=3,
        trend_window=3,
        return_window=2,
        sentiment_window=2,
    )
    assert list(frame["date"]) == list(origins)
    assert {
        "volatility_regime",
        "trend_regime_20d",
        "sentiment_alignment_5d",
        "trend_vol_regime",
        "official_event_alignment_5d",
        "official_supply_regime_30d",
    } <= set(frame.columns)
    assert thresholds.volatility_median >= 0.0


def test_gate_candidate_summary_requires_multiple_distinct_winners():
    regime_path_df = pd.DataFrame(
        [
            {"regime_family": "volatility_regime", "regime_value": "high_vol", "model": "a", "n_samples": 40, "path_mse": 1.0},
            {"regime_family": "volatility_regime", "regime_value": "high_vol", "model": "b", "n_samples": 40, "path_mse": 1.3},
            {"regime_family": "volatility_regime", "regime_value": "low_vol", "model": "b", "n_samples": 42, "path_mse": 0.9},
            {"regime_family": "volatility_regime", "regime_value": "low_vol", "model": "a", "n_samples": 42, "path_mse": 1.2},
        ]
    )
    summary = gate_candidate_summary(regime_path_df, min_samples_per_bucket=25, min_margin=0.1)
    assert summary["recommended_families"] == ["volatility_regime"]


def test_apply_fixed_gate_selects_model_by_regime_with_fallback():
    origin = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
            "regime": ["a", "b", "missing"],
        }
    )
    preds = {
        "m1": np.array([[1.0, 1.0], [1.1, 1.1], [1.2, 1.2]]),
        "m2": np.array([[2.0, 2.0], [2.1, 2.1], [2.2, 2.2]]),
    }
    selected, selection_df = apply_fixed_gate(
        origin,
        preds,
        regime_column="regime",
        model_by_regime={"a": "m2", "b": "m1"},
        fallback_model="m1",
        strategy_name="demo",
    )
    assert selected.shape == (3, 2)
    assert np.allclose(selected[0], preds["m2"][0])
    assert np.allclose(selected[1], preds["m1"][1])
    assert np.allclose(selected[2], preds["m1"][2])
    assert selection_df["selected_model"].tolist() == ["m2", "m1", "m1"]
