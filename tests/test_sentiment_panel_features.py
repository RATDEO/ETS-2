from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.panel import add_sentiment_features, select_feature_columns


def test_add_sentiment_features_creates_expected_columns(tmp_path: Path):
    csv_path = tmp_path / "daily_sentiment.csv"
    pd.DataFrame(
        {
            "seendate": ["2024-01-01", "2024-01-02", "2024-01-03"],
            "sent_score": [0.1, 0.2, -0.1],
        }
    ).to_csv(csv_path, index=False)

    calendar = pd.date_range("2024-01-01", periods=4, freq="D")
    panel = pd.DataFrame({"date": calendar, "y": [10.0, 11.0, 12.0, 13.0]})
    out = add_sentiment_features(
        panel,
        calendar,
        {
            "include_sentiment_features": True,
            "sentiment_path": str(csv_path),
            "sentiment_date_col": "seendate",
            "sentiment_score_col": "sent_score",
            "sentiment_windows": [3, 7],
            "include_sentiment_change": True,
        },
    )

    assert "sent_score" in out.columns
    assert "sent_score_3d_ma" in out.columns
    assert "sent_score_7d_ma" in out.columns
    assert "sent_score_change_1d" in out.columns
    assert float(out.loc[out["date"] == pd.Timestamp("2024-01-04"), "sent_score"].iloc[0]) == -0.1


def test_select_feature_columns_respects_preferred_order():
    panel = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=3, freq="D"),
            "y_return": [0.0, 0.1, 0.2],
            "y": [10.0, 11.0, 12.0],
            "coal_return": [0.1, 0.2, 0.3],
            "sent_score": [0.0, 0.1, 0.2],
            "sent_score_3d_ma": [0.0, 0.05, 0.1],
        }
    )

    cols = select_feature_columns(
        panel,
        target_col="y_return",
        max_exogenous_features=3,
        preferred_feature_order=["sent_score", "sent_score_3d_ma"],
    )

    assert cols == ["y_return", "y", "sent_score", "sent_score_3d_ma"]


def test_add_sentiment_features_uses_structured_news_columns(tmp_path: Path):
    csv_path = tmp_path / "feat4.csv"
    pd.DataFrame(
        {
            "seendate": ["2024-01-01", "2024-01-02", "2024-01-03"],
            "sent_score": [0.1, 0.2, -0.1],
            "news_volume": [2, 4, 3],
            "sent_change": [0.0, 0.1, -0.3],
        }
    ).to_csv(csv_path, index=False)

    calendar = pd.date_range("2024-01-01", periods=4, freq="D")
    panel = pd.DataFrame({"date": calendar, "y": [10.0, 11.0, 12.0, 13.0]})
    out = add_sentiment_features(
        panel,
        calendar,
        {
            "include_sentiment_features": True,
            "sentiment_path": str(csv_path),
            "sentiment_date_col": "seendate",
            "sentiment_score_col": "sent_score",
            "sentiment_windows": [3],
            "include_source_sent_change": True,
            "include_sentiment_volume": True,
            "sentiment_volume_windows": [3],
            "include_sentiment_interaction": True,
        },
    )

    assert "sent_change" in out.columns
    assert "sent_news_volume" in out.columns
    assert "sent_news_volume_3d_ma" in out.columns
    assert "sent_score_x_volume" in out.columns
