"""News ingestion and sentiment utilities."""

from .event_extraction import build_event_extraction_prompt, extract_event_batch
from .gdelt import fetch_gdelt_headlines, default_gdelt_query
from .event_panel import build_event_panel, parse_relevance_hits
from .official_sources import (
    DG_CLIMA_NEWS_URL,
    DG_CLIMA_RSS_URL,
    EEX_AUCTION_PAGE_URL,
    EEX_AUCTION_REPORT_INDEX_URL,
    build_deterministic_eex_event_records,
    fetch_dg_clima_articles,
    fetch_eex_sources,
)
from .sentiment import label_headlines_with_llm, aggregate_daily_sentiment

__all__ = [
    "build_event_extraction_prompt",
    "extract_event_batch",
    "fetch_gdelt_headlines",
    "default_gdelt_query",
    "build_event_panel",
    "parse_relevance_hits",
    "DG_CLIMA_NEWS_URL",
    "DG_CLIMA_RSS_URL",
    "EEX_AUCTION_PAGE_URL",
    "EEX_AUCTION_REPORT_INDEX_URL",
    "build_deterministic_eex_event_records",
    "fetch_dg_clima_articles",
    "fetch_eex_sources",
    "label_headlines_with_llm",
    "aggregate_daily_sentiment",
]
