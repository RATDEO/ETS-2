from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.load_energy import (
    create_energy_features,
    load_energy_benchmarks,
    load_uk_sap_gas,
    load_uk_system_power,
)


def test_load_uk_energy_csvs_and_create_features(tmp_path):
    energy_dir = tmp_path / "energy-benchmarks"
    energy_dir.mkdir(parents=True, exist_ok=True)

    gas_path = energy_dir / "uk-sap-gas-pence-per-kwh.csv"
    power_path = energy_dir / "uk-system-electricity-price-pence-per-kwh.csv"

    pd.DataFrame(
        {
            "date": ["2025-01-01", "2025-01-02", "2025-01-03"],
            "price_native": [1.70, 1.80, 1.90],
            "rolling_avg_7d": [1.70, 1.75, 1.80],
        }
    ).to_csv(gas_path, index=False)
    pd.DataFrame(
        {
            "date": ["2025-01-01", "2025-01-02", "2025-01-03"],
            "price_native": [4.0, 4.1, 4.4],
            "rolling_avg_7d": [4.0, 4.05, 4.17],
        }
    ).to_csv(power_path, index=False)

    gas_df = load_uk_sap_gas(gas_path)
    power_df = load_uk_system_power(power_path)

    assert gas_df["currency"].iloc[0] == "GBp/kWh"
    assert power_df["currency"].iloc[0] == "GBp/kWh"
    assert gas_df["source"].iloc[0] == "ons_national_gas"
    assert power_df["source"].iloc[0] == "ons_elexon"

    benchmarks = load_energy_benchmarks(tmp_path)
    assert set(benchmarks) == {"uk_gas", "uk_power"}

    calendar = pd.DatetimeIndex(pd.to_datetime(["2025-01-01", "2025-01-02", "2025-01-03"]))
    features = create_energy_features(benchmarks, calendar, compute_returns=True)

    assert "uk_gas_return" in features.columns
    assert "uk_power_return" in features.columns
    assert "uk_power_gas_spread" in features.columns
    assert "uk_power_gas_ratio" in features.columns
    assert "uk_power_gas_return_spread" in features.columns
    assert "uk_power_gas_ratio_z20" in features.columns
    assert np.isclose(features.loc[2, "uk_power_gas_spread"], 4.4 - 1.9, atol=1e-12)


def test_create_energy_features_adds_relative_spreads_and_ratios():
    calendar = pd.DatetimeIndex(pd.date_range("2025-01-01", periods=6, freq="D"))
    benchmarks = {
        "brent": pd.DataFrame({"date": calendar, "price_eur": [80, 81, 82, 83, 84, 85]}),
        "coal": pd.DataFrame({"date": calendar, "price_eur": [100, 102, 103, 104, 106, 108]}),
        "uk_gas": pd.DataFrame({"date": calendar, "price_native": [1.0, 1.1, 1.2, 1.15, 1.18, 1.22]}),
        "uk_power": pd.DataFrame({"date": calendar, "price_native": [3.0, 3.2, 3.3, 3.4, 3.5, 3.7]}),
    }

    features = create_energy_features(benchmarks, calendar, compute_returns=True)

    assert "coal_brent_ratio" in features.columns
    assert "coal_brent_return_spread" in features.columns
    assert "coal_brent_ratio_z20" in features.columns
    assert "uk_power_gas_spread_z20" in features.columns
    assert "uk_power_gas_vol_ratio_20d" in features.columns
    assert np.isclose(features.loc[0, "coal_brent_ratio"], 100 / 80, atol=1e-12)
