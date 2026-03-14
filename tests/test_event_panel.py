from __future__ import annotations

import pandas as pd
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.news.event_panel import build_event_panel, parse_relevance_hits


def test_parse_relevance_hits_counts_tokens() -> None:
    result = parse_relevance_hits("eu_ets_strong;eu_context;carbon_terms=3;energy_driver")
    assert result["evt_eu_ets_strong_count"] == 1.0
    assert result["evt_eu_context_count"] == 1.0
    assert result["evt_energy_driver_count"] == 1.0
    assert result["evt_carbon_terms_sum"] == 3.0


def test_build_event_panel_adds_rolling_features(tmp_path) -> None:
    headlines = pd.DataFrame(
        [
            {
                "seendate": "2024-01-01",
                "relevance_hits": "eu_ets_strong;carbon_terms=2",
                "llm_score": 1.0,
                "llm_importance": 6.0,
                "llm_score_importance": 0.6,
            },
            {
                "seendate": "2024-01-02",
                "relevance_hits": "energy_driver",
                "llm_score": -1.0,
                "llm_importance": 8.0,
                "llm_score_importance": -0.8,
            },
        ]
    )
    proxy = pd.DataFrame(
        [
            {"seendate": "2024-01-01", "sent_score": 0.5, "news_volume": 2, "sent_change": 0.5},
            {"seendate": "2024-01-02", "sent_score": -0.5, "news_volume": 1, "sent_change": -1.0},
        ]
    )

    headlines_path = tmp_path / "headlines.csv"
    proxy_path = tmp_path / "proxy.csv"
    headlines.to_csv(headlines_path, index=False)
    proxy.to_csv(proxy_path, index=False)

    calendar = pd.date_range("2024-01-01", periods=3, freq="D")
    panel = build_event_panel(headlines_path, calendar=calendar, proxy_daily_path=proxy_path)

    assert list(panel["date"]) == list(calendar)
    assert "evt_news_count_3d_sum" in panel.columns
    assert "evt_sent_mean_3d_ma" in panel.columns
    assert "proxy_news_volume_3d_sum" in panel.columns
    assert panel.loc[0, "evt_news_count"] == 1.0
    assert panel.loc[1, "evt_energy_driver_count"] == 1.0
    assert panel.loc[2, "evt_news_count"] == 0.0


def test_build_event_panel_v2_adds_taxonomy_features(tmp_path) -> None:
    headlines = pd.DataFrame(
        [
            {
                "title": "European Commission proposes CBAM reform after energy shock",
                "seendate": "2024-01-01",
                "relevance_hits": "eu_ets_strong;energy_driver;carbon_terms=2",
                "llm_score": 1.0,
                "llm_importance": 8.0,
                "llm_score_importance": 0.8,
            },
            {
                "title": "Shipping freight groups warn of auction costs in EU ETS",
                "seendate": "2024-01-02",
                "relevance_hits": "eu_ets_strong;eu_context;carbon_terms=2",
                "llm_score": -1.0,
                "llm_importance": 6.0,
                "llm_score_importance": -0.6,
            },
        ]
    )
    headlines_path = tmp_path / "headlines_v2.csv"
    headlines.to_csv(headlines_path, index=False)

    panel = build_event_panel(headlines_path, taxonomy_version="v2")
    assert "evt2_policy_count" in panel.columns
    assert "evt2_energy_count" in panel.columns
    assert "evt2_shipping_count" in panel.columns
    assert "evt2_cbam_count" in panel.columns
    assert panel.loc[0, "evt2_policy_count"] == 1.0
    assert panel.loc[0, "evt2_cbam_count"] == 1.0
    assert panel.loc[1, "evt2_shipping_count"] == 1.0
    assert panel.loc[1, "evt2_auction_count"] == 1.0


def test_build_event_panel_official_v1_adds_structured_features(tmp_path) -> None:
    events = pd.DataFrame(
        [
            {
                "source": "dg_clima",
                "published_at": "2024-01-01",
                "event_date": "2024-01-01",
                "title": "Commission publishes verified emissions data",
                "body_text": "Official update",
                "event_type": "verified_emissions",
                "affected_channel": "compliance_demand",
                "direction": "bullish",
                "intensity": 2.0,
                "expected_horizon": "5_20d",
                "novelty": "new",
                "policy_stage": "implementation",
                "confidence": 0.9,
            },
            {
                "source": "eex_official",
                "published_at": "2024-01-02",
                "event_date": "2024-01-02",
                "title": "Auction 4. Period CAP3 EU",
                "body_text": "Official EEX auction result",
                "event_type": "auction_supply",
                "affected_channel": "allowance_supply",
                "direction": "bearish",
                "intensity": 3.0,
                "expected_horizon": "1_5d",
                "novelty": "scheduled",
                "policy_stage": "implementation",
                "confidence": 1.0,
            },
        ]
    )
    events_path = tmp_path / "official_events.csv"
    events.to_csv(events_path, index=False)

    panel = build_event_panel(events_path, taxonomy_version="official_v1")
    assert "evt_official_type_verified_emissions_count" in panel.columns
    assert "evt_official_type_auction_supply_count" in panel.columns
    assert "evt_official_channel_allowance_supply_count" in panel.columns
    assert "evt_official_dir_bearish_count" in panel.columns
    assert panel.loc[0, "evt_official_type_verified_emissions_count"] == 1.0
    assert panel.loc[1, "evt_official_type_auction_supply_count"] == 1.0
    assert panel.loc[1, "evt_official_channel_allowance_supply_count"] == 1.0
    assert panel.loc[1, "evt_official_dir_bearish_count"] == 1.0
