"""UK official-source ingestion for UK ETS sentiment workflows."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterable
from urllib.parse import urljoin

from lxml import html
import pandas as pd
import requests
import xml.etree.ElementTree as ET

GOVUK_BASE_URL = "https://www.gov.uk"
GOVUK_SEARCH_URL = "https://www.gov.uk/search/all"
ELEXON_RSS_URL = "https://www.elexon.co.uk/feed/"
USER_AGENT = "UK-ETS-SENT/1.0"

UK_ETS_GOVUK_QUERIES = (
    '"UK ETS"',
    '"UK emissions trading scheme"',
    '"UK emissions allowance"',
    '"UK carbon auction"',
    '"UK carbon price"',
)


@dataclass(frozen=True)
class GovUkArticle:
    title: str
    url: str
    published_at: str | None
    summary: str
    body_text: str
    source: str = "gov_uk"


def _session_get(session: requests.Session, url: str, *, params: dict | None = None, timeout: int = 30) -> requests.Response:
    response = session.get(
        url,
        params=params,
        timeout=timeout,
        headers={"User-Agent": USER_AGENT},
    )
    response.raise_for_status()
    return response


def _clean_text(parts: Iterable[str]) -> str:
    return " ".join(str(part).strip() for part in parts if str(part).strip())


def _parse_govuk_search_results(html_text: str) -> list[dict[str, str | None]]:
    doc = html.fromstring(html_text)
    rows: list[dict[str, str | None]] = []
    for node in doc.xpath("//li[contains(@class,'gem-c-document-list__item')]"):
        href = node.xpath("string(.//a[1]/@href)").strip()
        title = _clean_text(node.xpath(".//a[1]//text()"))
        summary = _clean_text(node.xpath(".//p[contains(@class,'description')]//text()"))
        meta = _clean_text(node.xpath(".//ul[contains(@class,'metadata')]//text()"))
        if not href or not title:
            continue
        rows.append(
            {
                "title": title,
                "url": urljoin(GOVUK_BASE_URL, href),
                "search_meta": meta or None,
                "summary": summary,
            }
        )
    return rows


def _extract_govuk_json_ld(doc: html.HtmlElement) -> dict | None:
    for raw in doc.xpath("//script[@type='application/ld+json']/text()"):
        raw = (raw or "").strip()
        if not raw:
            continue
        try:
            payload = json.loads(raw)
        except Exception:
            continue
        if isinstance(payload, list):
            candidates = payload
        else:
            candidates = [payload]
        for obj in candidates:
            if not isinstance(obj, dict):
                continue
            obj_type = str(obj.get("@type", ""))
            if obj_type in {"Article", "NewsArticle", "Report", "WebPage"}:
                return obj
    return None


def _extract_govuk_article(html_text: str, *, url: str, fallback_title: str, fallback_summary: str) -> GovUkArticle:
    doc = html.fromstring(html_text)
    obj = _extract_govuk_json_ld(doc) or {}

    title = str(obj.get("name") or "").strip()
    if not title:
        title = _clean_text(doc.xpath("//h1//text()")) or fallback_title

    summary = str(obj.get("description") or "").strip()
    if not summary:
        summary = fallback_summary

    published_at = str(obj.get("datePublished") or "").strip() or None
    if not published_at:
        published_at = doc.xpath("string(//meta[@property='article:published_time']/@content)").strip() or None
    if not published_at:
        times = [str(v).strip() for v in doc.xpath("//time/@datetime") if str(v).strip()]
        published_at = times[-1] if times else None

    main_parts = doc.xpath("//main//text()")
    body_text = _clean_text(main_parts)
    if not body_text:
        body_text = summary

    return GovUkArticle(
        title=title,
        url=url,
        published_at=published_at,
        summary=summary,
        body_text=body_text,
    )


def fetch_govuk_search_articles(
    session: requests.Session,
    *,
    queries: Iterable[str] = UK_ETS_GOVUK_QUERIES,
    max_pages_per_query: int = 10,
    max_records: int = 250,
    start_date: str | None = None,
    end_date: str | None = None,
) -> pd.DataFrame:
    merged: dict[str, dict[str, str | None]] = {}
    for query in queries:
        for page in range(1, max_pages_per_query + 1):
            rows = _parse_govuk_search_results(
                _session_get(
                    session,
                    GOVUK_SEARCH_URL,
                    params={"keywords": query, "order": "updated-newest", "page": page},
                ).text
            )
            if not rows:
                break
            for row in rows:
                merged[str(row["url"])] = row
                if len(merged) >= max_records:
                    break
            if len(merged) >= max_records:
                break
        if len(merged) >= max_records:
            break

    articles: list[dict[str, str | None]] = []
    for row in merged.values():
        try:
            article = _extract_govuk_article(
                _session_get(session, str(row["url"])).text,
                url=str(row["url"]),
                fallback_title=str(row.get("title", "")),
                fallback_summary=str(row.get("summary", "")),
            )
        except requests.RequestException:
            article = GovUkArticle(
                title=str(row.get("title", "")),
                url=str(row["url"]),
                published_at=None,
                summary=str(row.get("summary", "")),
                body_text=str(row.get("summary", "")),
            )
        articles.append(
            {
                "source": article.source,
                "title": article.title,
                "url": article.url,
                "published_at": article.published_at,
                "summary": article.summary,
                "body_text": article.body_text,
                "domain": "gov.uk",
            }
        )

    df = pd.DataFrame(articles)
    if df.empty:
        return df
    df["published_at"] = pd.to_datetime(df["published_at"], errors="coerce", utc=True)
    if start_date:
        df = df[df["published_at"] >= pd.Timestamp(start_date, tz="UTC")]
    if end_date:
        df = df[df["published_at"] <= pd.Timestamp(end_date, tz="UTC") + pd.Timedelta(days=1)]
    df["seendate"] = df["published_at"].dt.normalize()
    return df.dropna(subset=["published_at"]).drop_duplicates(subset=["url"]).sort_values(
        "published_at",
        ascending=False,
        kind="stable",
    )


def fetch_elexon_rss_articles(session: requests.Session, *, max_records: int = 100) -> pd.DataFrame:
    xml_text = _session_get(session, ELEXON_RSS_URL).text
    root = ET.fromstring(xml_text)
    rows: list[dict[str, str | None]] = []
    for item in root.findall(".//item")[:max_records]:
        title = (item.findtext("title") or "").strip()
        url = (item.findtext("link") or "").strip()
        published_at = (item.findtext("pubDate") or "").strip() or None
        summary = (item.findtext("description") or "").strip()
        if not title or not url:
            continue
        rows.append(
            {
                "source": "elexon_rss",
                "title": title,
                "url": url,
                "published_at": published_at,
                "summary": summary,
                "body_text": summary,
                "domain": "elexon.co.uk",
            }
        )
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["published_at"] = pd.to_datetime(df["published_at"], errors="coerce", utc=True)
    df["seendate"] = df["published_at"].dt.normalize()
    return df.dropna(subset=["published_at"]).drop_duplicates(subset=["url"]).sort_values(
        "published_at",
        ascending=False,
        kind="stable",
    )
