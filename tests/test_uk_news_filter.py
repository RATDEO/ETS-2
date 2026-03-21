from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from uk_ets.scripts.filter_uk_news_headlines import filter_uk_headlines, score_uk_ets_relevance


def test_score_uk_ets_relevance_prefers_uka_headline():
    uk_score, uk_hits = score_uk_ets_relevance("UK ETS carbon auction drives UKA futures higher")
    eu_score, eu_hits = score_uk_ets_relevance("EU ETS carbon allowance prices fall after EUA auction")

    assert uk_score > eu_score
    assert "uk_ets_strong" in uk_hits
    assert "non_uk_market" in eu_hits


def test_filter_uk_headlines_keeps_relevant_and_dedupes():
    df = pd.DataFrame(
        {
            "seendate": [
                "2025-01-01T08:00:00Z",
                "2025-01-01T09:00:00Z",
                "2025-01-02T08:00:00Z",
            ],
            "title": [
                "UK ETS carbon auction drives UKA futures higher",
                "UK ETS carbon auction drives UKA futures higher",
                "EU ETS carbon allowance prices fall after EUA auction",
            ],
            "language": ["English", "English", "English"],
            "url": ["a", "b", "c"],
        }
    )

    out = filter_uk_headlines(df, min_relevance_score=6, dedupe_method="sequence")

    assert len(out) == 1
    assert out.iloc[0]["title"] == "UK ETS carbon auction drives UKA futures higher"
