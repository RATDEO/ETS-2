from pathlib import Path
import sys

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.panel import (
    _add_target_feature_aliases,
    _add_uk_energy_weather_interactions,
    _is_carbon_futures_target,
    _target_display_name,
)
from src.data.panel import _extend_proxy_event_features
from src.data.load_eua_futures import infer_futures_metadata


ROOT = Path(__file__).resolve().parents[1]


def test_uk_target_aliases_are_supported():
    assert _is_carbon_futures_target("uka_futures", "UKA_FUTURES")
    assert _is_carbon_futures_target("carbon_futures", "UKA_FUTURES")
    assert _target_display_name("UKA_FUTURES") == "UKA Futures"


def test_uk_target_feature_aliases_are_added():
    panel = pd.DataFrame(
        {
            "eua_volume": [1.0, 2.0],
            "eua_range": [0.3, 0.4],
            "eua_range_pct": [0.01, 0.02],
        }
    )

    out = _add_target_feature_aliases(panel.copy(), "UKA_FUTURES")

    assert out["target_volume"].tolist() == [1.0, 2.0]
    assert out["uka_volume"].tolist() == [1.0, 2.0]
    assert out["target_range_pct"].tolist() == [0.01, 0.02]
    assert out["uka_range_pct"].tolist() == [0.01, 0.02]


def test_uk_futures_filename_infers_uka_metadata():
    meta = infer_futures_metadata("UK Emissions Allowances (UKA) Futures Historical Data AUTO.csv")

    assert meta["instrument"] == "UKA_FUTURES"
    assert meta["source"] == "uka_futures"
    assert meta["currency"] == "GBP"


def test_proxy_event_features_extend_beyond_last_print():
    panel = pd.DataFrame(
        {
            "date": pd.to_datetime(["2025-12-09", "2025-12-10", "2025-12-11", "2025-12-12"]),
            "uk_icap_primary_print_day": [0.0, 1.0, None, None],
            "uk_icap_secondary_print_day": [1.0, 0.0, None, None],
        }
    )

    out = _extend_proxy_event_features(panel.copy())

    assert out["uk_icap_primary_print_day"].tolist() == [0.0, 1.0, 0.0, 0.0]
    assert out["uk_icap_secondary_print_day"].tolist() == [1.0, 0.0, 0.0, 0.0]
    assert pd.isna(out["uk_icap_primary_days_since_print"].iloc[0])
    assert out["uk_icap_primary_days_since_print"].iloc[1:].tolist() == [0.0, 1.0, 2.0]
    assert out["uk_icap_secondary_days_since_print"].tolist() == [0.0, 1.0, 2.0, 3.0]
    assert "uk_icap_dow_sin" in out.columns
    assert "uk_icap_dow_cos" in out.columns


def test_proxy_event_features_support_days_since_cap():
    panel = pd.DataFrame(
        {
            "date": pd.to_datetime(["2025-12-10", "2026-01-10", "2026-02-15"]),
            "uk_icap_secondary_print_day": [1.0, 0.0, 0.0],
        }
    )

    out = _extend_proxy_event_features(panel.copy(), days_since_cap=30)

    assert out["uk_icap_secondary_days_since_print"].tolist() == [0.0, 30.0, 30.0]


def test_uk_energy_weather_interactions_are_added():
    panel = pd.DataFrame(
        {
            "y": [50.0, 52.0, 54.0, 53.0, 55.0, 57.0],
            "y_return": [None, 0.01, 0.02, -0.01, 0.015, 0.012],
            "uk_gas_price": [1.0, 1.1, 1.2, 1.3, 1.25, 1.22],
            "uk_power_price": [3.0, 3.1, 3.2, 3.3, 3.4, 3.5],
            "brent_price": [80.0, 81.0, 82.0, 83.0, 84.0, 85.0],
            "coal_price": [100.0, 102.0, 103.0, 104.0, 106.0, 108.0],
            "uk_gas_return": [None, 0.02, 0.01, -0.02, 0.01, 0.02],
            "uk_power_return": [None, 0.01, 0.015, 0.01, 0.005, 0.02],
            "brent_return": [None, 0.01, 0.01, 0.01, 0.01, 0.01],
            "coal_return": [None, 0.02, 0.015, 0.01, 0.015, 0.02],
            "uk_hdd18": [12.0, 11.5, 10.5, 9.5, 10.0, 11.0],
            "uk_hdd18_7d_ma": [12.0, 11.8, 11.0, 10.0, 10.2, 10.5],
            "uk_temp_mean_c": [4.0, 5.0, 6.0, 7.0, 6.5, 5.5],
            "uk_temp_mean_7d_ma": [4.0, 4.5, 5.3, 6.0, 6.1, 5.8],
        }
    )

    out = _add_uk_energy_weather_interactions(panel.copy())

    assert "uk_hdd18_surprise" in out.columns
    assert "uk_gas_hdd18_interaction" in out.columns
    assert "uk_power_hdd18_surprise_interaction" in out.columns
    assert "uka_gas_ratio" in out.columns
    assert "uka_gas_ratio_z20" in out.columns
    assert "uka_gas_return_spread" in out.columns
    assert out["uk_hdd18_surprise"].iloc[3] == -0.5
    assert out["uk_gas_hdd18_interaction"].iloc[1] == 0.02 * 11.5


def test_uk_configs_default_to_market_specific_features():
    config_paths = [
        ROOT / "uk_ets" / "config" / "uk_ets_default.yaml",
        ROOT / "uk_ets" / "config" / "uk_ets_tuned.yaml",
    ]

    for path in config_paths:
        cfg = yaml.safe_load(path.read_text())
        assert cfg["target"]["instrument"] == "UKA_FUTURES"
        assert cfg["target"]["source"] == "uka_futures"
        assert cfg["target"]["currency"] == "GBP"
        assert cfg["model"].get("dlinear_channel_mixer") == "residual_linear"

        features = cfg["features"]
        assert features["include_brent"] is True
        assert features["include_coal"] is True
        assert features["include_uk_gas"] is True
        assert features["include_uk_power"] is True
        assert features["include_uk_weather"] is True
        assert features["include_indices"] is False
        assert features["include_vstoxx"] is False
        assert features["include_icap_secondary"] is False
        assert features["auction_proxy_days_since_cap"] == 30

        preferred = features["preferred_feature_order"]
        assert "target_range_pct" in preferred
        assert "target_volume" in preferred
        assert "uk_icap_primary_days_since_print" in preferred
        assert "uk_icap_secondary_days_since_print" in preferred
        assert "uk_power_return" in preferred
        assert "uk_gas_return" in preferred
        assert "brent_return" in preferred
        assert "coal_return" in preferred
        assert "uk_hdd18" in preferred
        assert "uk_hdd18_7d_ma" in preferred
        assert "uk_temp_mean_c" in preferred
        assert all(not name.startswith("idx_") for name in preferred)
        assert all("vstoxx" not in name for name in preferred)
        assert "uk_power_gas_spread" in preferred


def test_recovered_uk_tsm_config_uses_compact_feature_slate():
    path = ROOT / "uk_ets" / "config" / "uk_ets_tsm_recovered.yaml"
    cfg = yaml.safe_load(path.read_text())

    assert cfg["target"]["instrument"] == "UKA_FUTURES"
    assert cfg["model"]["tsm_type"] == "dlinear"
    assert cfg["model"]["dlinear_channel_mixer"] == "residual_linear"
    assert cfg["model"]["dlinear_individual"] is False
    assert cfg["model"]["kernel_size"] == 3
    assert cfg["model"]["learning_rate"] == 0.003

    features = cfg["features"]
    assert features["max_exogenous_features_model"] == 9
    preferred = features["preferred_feature_order"]
    assert "uk_icap_primary_days_since_print" not in preferred
    assert "uk_icap_secondary_days_since_print" not in preferred
    assert preferred[:5] == [
        "target_range_pct",
        "target_volume",
        "is_auction_day",
        "uk_icap_primary_print_day",
        "uk_icap_secondary_print_day",
    ]
