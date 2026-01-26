"""News ingestion and sentiment utilities."""

from .gdelt import fetch_gdelt_headlines, default_gdelt_query
from .sentiment import label_headlines_with_llm, aggregate_daily_sentiment

__all__ = [
    "fetch_gdelt_headlines",
    "default_gdelt_query",
    "label_headlines_with_llm",
    "aggregate_daily_sentiment",
]
