from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.load_weather import load_weather_feature_pack


def test_load_weather_feature_pack(tmp_path):
    weather_dir = tmp_path / "weather-proxy"
    weather_dir.mkdir(parents=True, exist_ok=True)
    path = weather_dir / "uk_weather_daily_feature_pack.csv"

    pd.DataFrame(
        {
            "date": ["2025-01-01", "2025-01-02"],
            "uk_temp_mean_c": [4.0, 5.0],
            "uk_temp_min_c": [1.0, 2.0],
            "uk_temp_max_c": [7.0, 8.0],
            "uk_hdd18": [14.0, 13.0],
            "uk_hdd18_7d_ma": [14.0, 13.5],
            "uk_temp_mean_7d_ma": [4.0, 4.5],
        }
    ).to_csv(path, index=False)

    df = load_weather_feature_pack(tmp_path)

    assert df.columns.tolist() == [
        "date",
        "uk_temp_mean_c",
        "uk_temp_min_c",
        "uk_temp_max_c",
        "uk_hdd18",
        "uk_hdd18_7d_ma",
        "uk_temp_mean_7d_ma",
    ]
    assert len(df) == 2
