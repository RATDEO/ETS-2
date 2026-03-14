from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.news.official_event_features import (
    build_official_event_summary_map,
    build_official_origin_feature_frame,
)


def test_build_official_event_summary_map_returns_recent_context(tmp_path: Path) -> None:
    events = pd.DataFrame(
        [
            {
                "event_date": "2024-01-05",
                "title": "Commission publishes verified emissions data",
                "event_type": "verified_emissions",
                "affected_channel": "compliance_demand",
                "direction": "bullish",
                "intensity": 2.0,
                "confidence": 1.0,
            },
            {
                "event_date": "2024-01-08",
                "title": "Auction result shows heavy supply",
                "event_type": "auction_supply",
                "affected_channel": "allowance_supply",
                "direction": "bearish",
                "intensity": 3.0,
                "confidence": 1.0,
            },
        ]
    )
    path = tmp_path / "official.csv"
    events.to_csv(path, index=False)
    summary_map = build_official_event_summary_map(path, pd.to_datetime(["2024-01-08"]), lookback_days=10, max_titles=2)
    summary = summary_map[pd.Timestamp("2024-01-08")]
    assert "official_event_balance" in summary
    assert "policy_count=1" in summary["official_event_balance"]
    assert "Auction result shows heavy supply" in summary["official_recent_events"]


def test_build_official_origin_feature_frame_emits_expected_columns(tmp_path: Path) -> None:
    events = pd.DataFrame(
        [
            {
                "event_date": "2024-01-05",
                "title": "Commission publishes verified emissions data",
                "event_type": "verified_emissions",
                "affected_channel": "compliance_demand",
                "direction": "bullish",
                "intensity": 2.0,
                "confidence": 1.0,
            },
            {
                "event_date": "2024-01-08",
                "title": "Auction result shows heavy supply",
                "event_type": "auction_supply",
                "affected_channel": "allowance_supply",
                "direction": "bearish",
                "intensity": 3.0,
                "confidence": 1.0,
            },
        ]
    )
    path = tmp_path / "official.csv"
    events.to_csv(path, index=False)
    frame, thresholds = build_official_origin_feature_frame(pd.to_datetime(["2024-01-08", "2024-01-10"]), path)
    assert {"official_event_net_30d", "official_supply_net_30d", "official_supply_regime_30d"} <= set(frame.columns)
    assert thresholds.official_event_abs_median >= 0.0
