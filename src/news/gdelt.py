"""GDELT doc API ingestion for news headlines."""

from __future__ import annotations

import json
import time
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Iterable, List, Dict, Optional
from urllib.parse import urlencode
from urllib.request import urlopen, Request
from urllib.error import HTTPError


GDELT_DOC_ENDPOINT = "https://api.gdeltproject.org/api/v2/doc/doc"
logger = logging.getLogger(__name__)


DEFAULT_KEYWORDS = [
    "emissions trading system",
    "carbon allowance",
    "carbon permit",
    "carbon price",
    "carbon market",
    "emissions allowance",
    "European emissions trading system",
    "European carbon market",
    "EUA futures",
]

STRICT_KEYWORDS = [
    "EU ETS",
    "European Union Allowance",
    "EUA futures",
    "emissions trading",
    "emissions trading system",
    "carbon allowance",
    "emissions allowance",
    "carbon permit",
    "carbon trading",
    "cap and trade",
]

EXPANDED_KEYWORDS = [
    "EU ETS",
    "European Union Allowance",
    "EUA",
    "EUA futures",
    "emissions trading",
    "carbon allowance",
    "carbon price",
    "carbon market",
    "carbon trading",
    "carbon credit",
    "allowance auction",
    "cap and trade",
]

DEFAULT_THEME_TERMS = [
    "EU",
    "Europe",
    "European",
    "European Union",
]

DEFAULT_DOMAINS = [
    "europa.eu",
    "ec.europa.eu",
    "eex.com",
    "theice.com",
    "ice.com",
    "reuters.com",
    "bloomberg.com",
    "ft.com",
    "euractiv.com",
    "montelnews.com",
    "argusmedia.com",
    "icis.com",
    "carbon-pulse.com",
    "ieta.org",
    "iea.org",
]


def default_gdelt_query(keywords: Optional[List[str]] = None) -> str:
    keys = keywords or DEFAULT_KEYWORDS
    quoted = [f"\"{k}\"" if " " in k else k for k in keys]
    return "(" + " OR ".join(quoted) + ")"


def build_gdelt_query(
    keywords: Optional[List[str]] = None,
    themes: Optional[List[str]] = None,
    domains: Optional[List[str]] = None
) -> str:
    parts = []
    if keywords:
        parts.append(default_gdelt_query(keywords))
    if themes:
        parts.append(default_gdelt_query(themes))
    if domains:
        domain_terms = [f"domain:{d}" for d in domains if d]
        if domain_terms:
            parts.append("(" + " OR ".join(domain_terms) + ")")
    return " AND ".join(parts)


def _parse_gdelt_datetime(value: str) -> datetime:
    try:
        return datetime.strptime(value, "%Y%m%d%H%M%S")
    except ValueError:
        return datetime.strptime(value, "%Y%m%dT%H%M%SZ")


def _format_gdelt_datetime(value: datetime) -> str:
    return value.strftime("%Y%m%d%H%M%S")


def _request_gdelt(
    params: Dict[str, str],
    timeout: int = 30,
    max_retries: int = 3,
    sleep_seconds: float = 1.0
) -> Dict:
    url = f"{GDELT_DOC_ENDPOINT}?{urlencode(params)}"
    req = Request(url, headers={"User-Agent": "ETS-News-Ingest/1.0"})

    for attempt in range(max_retries):
        try:
            with urlopen(req, timeout=timeout) as resp:
                data = resp.read().decode("utf-8")
            return json.loads(data)
        except json.JSONDecodeError:
            logger.warning("GDELT returned non-JSON response (attempt %d).", attempt + 1)
        except HTTPError as exc:
            if exc.code == 429:
                wait = max(sleep_seconds, 1.0) * (attempt + 1) * 5
                logger.warning(
                    "GDELT rate limited (429). Sleeping %.1fs before retry %d.",
                    wait,
                    attempt + 1,
                )
                time.sleep(wait)
                continue
            logger.warning("GDELT request failed (attempt %d): %s", attempt + 1, exc)
        except Exception as exc:
            logger.warning("GDELT request failed (attempt %d): %s", attempt + 1, exc)
        time.sleep(sleep_seconds * (attempt + 1))

    return {}


def _iter_time_windows(
    start: datetime, end: datetime, window_days: int
) -> Iterable[tuple[datetime, datetime]]:
    cursor = start
    delta = timedelta(days=window_days)
    while cursor <= end:
        window_end = min(end, cursor + delta - timedelta(seconds=1))
        yield cursor, window_end
        cursor = window_end + timedelta(seconds=1)


def fetch_gdelt_headlines(
    start_date: str,
    end_date: str,
    query: str,
    window_days: int = 1,
    max_records: int = 250,
    sleep_seconds: float = 0.2,
    language_filter: Optional[str] = "English",
) -> List[Dict]:
    start = datetime.fromisoformat(start_date)
    end = datetime.fromisoformat(end_date)
    articles: List[Dict] = []
    seen_urls = set()

    for window_start, window_end in _iter_time_windows(start, end, window_days):
        page_start = window_start
        while page_start <= window_end:
            params = {
                "query": query,
                "mode": "ArtList",
                "format": "json",
                "maxrecords": str(max_records),
                "sort": "DateAsc",
                "startdatetime": _format_gdelt_datetime(page_start),
                "enddatetime": _format_gdelt_datetime(window_end),
            }
            payload = _request_gdelt(params)
            batch = payload.get("articles", [])
            if not batch:
                break

            last_seen = None
            for item in batch:
                url = item.get("url")
                if not url or url in seen_urls:
                    continue
                lang = item.get("language")
                if language_filter and lang and lang.lower() != language_filter.lower():
                    continue
                seen_urls.add(url)
                articles.append(item)
                last_seen = item.get("seendate")

            if len(batch) < max_records or last_seen is None:
                break

            page_start = _parse_gdelt_datetime(last_seen) + timedelta(seconds=1)
            time.sleep(sleep_seconds)

        time.sleep(sleep_seconds)

    return articles
