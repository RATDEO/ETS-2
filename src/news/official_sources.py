"""Automated official-source ingestion for structured carbon-market events."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import io
import re
from urllib.parse import urljoin
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
import requests
from lxml import html

DG_CLIMA_NEWS_URL = "https://climate.ec.europa.eu/news-other-reads/news_en"
DG_CLIMA_RSS_URL = "https://climate.ec.europa.eu/node/2/rss_en"
ENTSOE_BASE_URL = "https://www.entsoe.eu"
ENTSOE_NEWS_RSS_URL = "https://www.entsoe.eu/rss/news.xml"
EEX_AUCTION_PAGE_URL = "https://www.eex.com/en/markets/environmental-markets/eu-ets-auctions"
EEX_AUCTION_REPORT_INDEX_URL = "https://public.eex-group.com/eex/eua-auction-report/"
USER_AGENT = "ETS-Official-Events/1.0"

DG_CLIMA_RELEVANCE_TERMS = (
    "ets",
    "emissions trading",
    "allowance",
    "auction",
    "verified emissions",
    "union registry",
    "maritime",
    "aviation",
    "cbam",
    "market stability reserve",
)

ENTSOE_RELEVANCE_TERMS = (
    "power system",
    "outage",
    "system incident",
    "generation adequacy",
    "adequacy",
    "load",
    "demand",
    "interconnector",
    "cross-border",
    "renewable",
    "coal",
    "gas",
    "nuclear",
    "electricity market",
    "balancing",
    "congestion",
)


@dataclass(frozen=True)
class DGClimaArticle:
    title: str
    url: str
    published_at: str | None
    summary: str
    body_text: str
    source: str = "dg_clima"


@dataclass(frozen=True)
class EntsoeArticle:
    title: str
    url: str
    published_at: str | None
    summary: str
    body_text: str
    source: str = "entsoe"


def _session_get(session: requests.Session, url: str, timeout: int = 30) -> requests.Response:
    response = session.get(url, timeout=timeout, headers={"User-Agent": USER_AGENT})
    response.raise_for_status()
    return response


def parse_dg_clima_rss(xml_text: str) -> list[dict[str, str | None]]:
    root = ET.fromstring(xml_text)
    items: list[dict[str, str | None]] = []
    for item in root.findall(".//item"):
        title = (item.findtext("title") or "").strip()
        link = (item.findtext("link") or "").strip()
        pub_date = (item.findtext("pubDate") or "").strip() or None
        description = (item.findtext("description") or "").strip()
        if not title or not link:
            continue
        items.append(
            {
                "title": title,
                "url": link,
                "published_at": pub_date,
                "summary": description,
            }
        )
    return items


def fetch_dg_clima_rss(session: requests.Session) -> list[dict[str, str | None]]:
    xml_text = _session_get(session, DG_CLIMA_RSS_URL).text
    return parse_dg_clima_rss(xml_text)


def parse_entsoe_rss(xml_text: str, base_url: str = ENTSOE_BASE_URL) -> list[dict[str, str | None]]:
    root = ET.fromstring(xml_text)
    items: list[dict[str, str | None]] = []
    for item in root.findall(".//item"):
        title = (item.findtext("title") or "").strip()
        link = (item.findtext("link") or "").strip()
        pub_date = (item.findtext("pubDate") or "").strip() or None
        description = (item.findtext("description") or "").strip()
        if not title or not link:
            continue
        items.append(
            {
                "title": title,
                "url": urljoin(base_url, link),
                "published_at": pub_date,
                "summary": description,
            }
        )
    return items


def fetch_entsoe_rss(session: requests.Session) -> list[dict[str, str | None]]:
    xml_text = _session_get(session, ENTSOE_NEWS_RSS_URL).text
    return parse_entsoe_rss(xml_text)


def parse_dg_clima_listing(html_text: str, base_url: str = DG_CLIMA_NEWS_URL) -> list[dict[str, str | None]]:
    doc = html.fromstring(html_text)
    results: list[dict[str, str | None]] = []
    for node in doc.xpath("//article"):
        href = node.xpath("string(.//a[@href][1]/@href)").strip()
        if "/news-other-reads/news/" not in href:
            continue
        title = " ".join(t.strip() for t in node.xpath(".//a[@href][1]//text()") if t.strip())
        summary = " ".join(t.strip() for t in node.xpath(".//p//text()") if t.strip())
        date_text = " ".join(t.strip() for t in node.xpath(".//*[contains(@class,'date') or self::time]//text()") if t.strip())
        if not title:
            continue
        results.append(
            {
                "title": title,
                "url": urljoin(base_url, href),
                "published_at": date_text or None,
                "summary": summary,
            }
        )
    return results


def fetch_dg_clima_archive(session: requests.Session, pages: int = 1) -> list[dict[str, str | None]]:
    pages = max(1, int(pages))
    rows: list[dict[str, str | None]] = []
    seen: set[str] = set()
    for page_idx in range(pages):
        url = DG_CLIMA_NEWS_URL if page_idx == 0 else f"{DG_CLIMA_NEWS_URL}?page={page_idx}"
        html_text = _session_get(session, url).text
        for row in parse_dg_clima_listing(html_text, base_url=url):
            if row["url"] in seen:
                continue
            seen.add(str(row["url"]))
            rows.append(row)
    return rows


def extract_dg_clima_article(html_text: str, url: str, published_at: str | None = None) -> DGClimaArticle:
    doc = html.fromstring(html_text)
    title = doc.xpath("string(//meta[@property='og:title']/@content)").strip()
    if not title:
        title = " ".join(t.strip() for t in doc.xpath("//h1//text()") if t.strip())
    summary = doc.xpath("string(//meta[@name='description']/@content)").strip()
    main_text = doc.xpath("string(//main)").strip()
    body_text = " ".join(main_text.split())
    if summary and body_text.startswith(summary):
        body_text = body_text
    return DGClimaArticle(
        title=title,
        url=url,
        published_at=published_at,
        summary=summary,
        body_text=body_text,
    )


def fetch_dg_clima_articles(session: requests.Session, pages: int = 1) -> pd.DataFrame:
    items = fetch_dg_clima_rss(session)
    archive_items = fetch_dg_clima_archive(session, pages=pages)
    merged: dict[str, dict[str, str | None]] = {}
    for item in archive_items + items:
        merged[str(item["url"])] = item

    rows: list[dict[str, str | None]] = []
    for item in merged.values():
        try:
            article = extract_dg_clima_article(
                _session_get(session, str(item["url"])).text,
                url=str(item["url"]),
                published_at=item.get("published_at"),
            )
        except requests.RequestException:
            article = DGClimaArticle(
                title=str(item.get("title", "")),
                url=str(item["url"]),
                published_at=item.get("published_at"),
                summary=str(item.get("summary", "")),
                body_text=str(item.get("summary", "")),
            )
        rows.append(
            {
                "source": article.source,
                "title": article.title,
                "url": article.url,
                "published_at": article.published_at,
                "summary": article.summary,
                "body_text": article.body_text,
                "cheap_relevance": float(is_dg_clima_relevant(article.title, article.summary, article.body_text)),
            }
        )
    return pd.DataFrame(rows).drop_duplicates(subset=["url"]).sort_values("published_at", ascending=False, kind="stable")


def extract_entsoe_article(html_text: str, url: str, published_at: str | None = None) -> EntsoeArticle:
    doc = html.fromstring(html_text)
    title = doc.xpath("string(//meta[@property='og:title']/@content)").strip()
    if not title:
        title = " ".join(t.strip() for t in doc.xpath("//h1//text()") if t.strip())
    summary = doc.xpath("string(//meta[@name='description']/@content)").strip()
    main_text = doc.xpath("string(//main)").strip()
    body_text = " ".join(main_text.split())
    return EntsoeArticle(
        title=title,
        url=url,
        published_at=published_at,
        summary=summary,
        body_text=body_text,
    )


def fetch_entsoe_articles(session: requests.Session) -> pd.DataFrame:
    items = fetch_entsoe_rss(session)
    rows: list[dict[str, str | None]] = []
    for item in items:
        try:
            article = extract_entsoe_article(
                _session_get(session, str(item["url"])).text,
                url=str(item["url"]),
                published_at=item.get("published_at"),
            )
        except requests.RequestException:
            article = EntsoeArticle(
                title=str(item.get("title", "")),
                url=str(item["url"]),
                published_at=item.get("published_at"),
                summary=str(item.get("summary", "")),
                body_text=str(item.get("summary", "")),
            )
        rows.append(
            {
                "source": article.source,
                "title": article.title,
                "url": article.url,
                "published_at": article.published_at,
                "summary": article.summary,
                "body_text": article.body_text,
                "cheap_relevance": float(is_entsoe_relevant(article.title, article.summary, article.body_text)),
            }
        )
    return pd.DataFrame(rows).drop_duplicates(subset=["url"]).sort_values("published_at", ascending=False, kind="stable")


def is_dg_clima_relevant(title: str, summary: str, body_text: str) -> bool:
    text = " ".join([title or "", summary or "", body_text or ""]).lower()
    return any(term in text for term in DG_CLIMA_RELEVANCE_TERMS)


def is_entsoe_relevant(title: str, summary: str, body_text: str) -> bool:
    text = " ".join([title or "", summary or "", body_text or ""]).lower()
    return any(term in text for term in ENTSOE_RELEVANCE_TERMS)


def discover_eex_calendar_url(session: requests.Session) -> str:
    page_text = _session_get(session, EEX_AUCTION_PAGE_URL).text
    match = re.search(
        r'(/fileadmin/EEX/Downloads/Trading/Calendar/Auction_Calendar/[^"\']+\.xlsx)',
        page_text,
        re.IGNORECASE,
    )
    if not match:
        raise ValueError("Could not discover EEX auction calendar xlsx URL.")
    return urljoin(EEX_AUCTION_PAGE_URL, match.group(1))


def discover_eex_auction_report_urls(session: requests.Session) -> list[str]:
    html_text = _session_get(session, EEX_AUCTION_REPORT_INDEX_URL).text
    rel_paths = sorted(
        set(
            re.findall(
                r'href="(emission-spot-primary-market-auction-report-\d{4}-data\.(?:xls|xlsx))"',
                html_text,
                re.IGNORECASE,
            )
        )
    )
    return [urljoin(EEX_AUCTION_REPORT_INDEX_URL, rel) for rel in rel_paths]


def _extract_report_year(url: str) -> int | None:
    match = re.search(r"-(20\d{2})-data\.(?:xls|xlsx)$", str(url))
    if not match:
        return None
    return int(match.group(1))


def parse_eex_auction_report(content: bytes, source_url: str) -> pd.DataFrame:
    workbook = pd.ExcelFile(io.BytesIO(content), engine="openpyxl")
    frame = workbook.parse(workbook.sheet_names[0], header=5)
    frame.columns = [str(col).strip() for col in frame.columns]
    frame = frame[frame["Date"].notna()].copy()
    frame["Date"] = pd.to_datetime(frame["Date"], errors="coerce").dt.normalize()
    frame = frame[frame["Date"].notna()].copy()
    frame = frame.rename(
        columns={
            "Auction Name": "auction_name",
            "Contract": "contract",
            "Auction Price €/tCO2": "auction_price_eur",
            "Auction Volume tCO2": "auction_volume_tco2",
            "Total Revenue €": "total_revenue_eur",
            "Cover Ratio": "cover_ratio",
            "Country": "country",
        }
    )
    keep_cols = [
        "Date",
        "auction_name",
        "contract",
        "auction_price_eur",
        "auction_volume_tco2",
        "total_revenue_eur",
        "cover_ratio",
        "country",
    ]
    out = frame[keep_cols].copy()
    out["source"] = "eex_auction_report"
    out["source_url"] = source_url
    return out.sort_values("Date", kind="stable").reset_index(drop=True)


def parse_eex_auction_calendar(content: bytes, source_url: str) -> pd.DataFrame:
    from openpyxl import load_workbook

    workbook = load_workbook(io.BytesIO(content), data_only=True)
    sheet = workbook["Calendar"]
    rows: list[dict[str, object]] = []
    current_date: datetime | None = None
    for row_idx in range(1, sheet.max_row + 1):
        values = [sheet.cell(row_idx, col_idx).value for col_idx in range(1, 9)]
        first = values[0]
        if isinstance(first, datetime):
            current_date = first
            continue
        if current_date is None:
            continue
        trading_time = values[1]
        product_code = values[2]
        trading_period = values[3]
        volume = values[4]
        auction_name = values[6]
        if not trading_time or not auction_name:
            continue
        try:
            volume_value = float(volume) if volume is not None else None
        except (TypeError, ValueError):
            continue
        rows.append(
            {
                "date": pd.Timestamp(current_date).normalize(),
                "call_trading_period": str(trading_time).strip(),
                "product_code": str(product_code).strip() if product_code else None,
                "trading_period": str(trading_period).strip() if trading_period else None,
                "auction_volume_tco2": volume_value,
                "auction_name": str(auction_name).strip(),
                "source": "eex_auction_calendar",
                "source_url": source_url,
            }
        )
    return pd.DataFrame(rows).sort_values("date", kind="stable").reset_index(drop=True)


def fetch_eex_sources(
    session: requests.Session,
    *,
    report_year_min: int = 2021,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    calendar_url = discover_eex_calendar_url(session)
    calendar_content = _session_get(session, calendar_url).content
    calendar_df = parse_eex_auction_calendar(calendar_content, source_url=calendar_url)

    report_urls = discover_eex_auction_report_urls(session)
    if not report_urls:
        raise ValueError("No EEX auction report files discovered.")
    selected_report_urls = [url for url in report_urls if (_extract_report_year(url) or 0) >= int(report_year_min)]
    if not selected_report_urls:
        selected_report_urls = report_urls[-1:]
    report_frames: list[pd.DataFrame] = []
    for report_url in selected_report_urls:
        report_content = _session_get(session, report_url).content
        report_frames.append(parse_eex_auction_report(report_content, source_url=report_url))
    report_df = (
        pd.concat(report_frames, ignore_index=True)
        .sort_values(["Date", "auction_name"], kind="stable")
        .drop_duplicates(subset=["Date", "auction_name", "auction_volume_tco2"], keep="last")
        .reset_index(drop=True)
    )
    return calendar_df, report_df, {"calendar_url": calendar_url, "report_urls": selected_report_urls}


def build_deterministic_eex_event_records(calendar_df: pd.DataFrame, report_df: pd.DataFrame) -> pd.DataFrame:
    """Convert official EEX auction sources into deterministic structured event records."""
    report = report_df.copy()
    report = report.sort_values("Date", kind="stable").reset_index(drop=True)
    report["auction_volume_20d_ma"] = report["auction_volume_tco2"].rolling(window=20, min_periods=5).mean()
    # Before five auctions have printed there is no stable rolling baseline yet, so fall back to neutral.
    report["auction_volume_ratio_20d"] = (
        report["auction_volume_tco2"] / report["auction_volume_20d_ma"]
    ).replace([np.inf, -np.inf], np.nan)
    report["auction_volume_ratio_20d"] = report["auction_volume_ratio_20d"].fillna(1.0)
    report["auction_price_change_1d"] = report["auction_price_eur"].diff()
    report["supply_direction"] = np.where(
        report["auction_volume_ratio_20d"] > 1.05,
        "bearish",
        np.where(report["auction_volume_ratio_20d"] < 0.95, "bullish", "neutral"),
    )
    report["event_type"] = "auction_supply"
    report["affected_channel"] = "allowance_supply"
    report["expected_horizon"] = "1_5d"
    report["intensity"] = np.clip(np.abs(report["auction_volume_ratio_20d"] - 1.0) / 0.05, 0.0, 3.0)
    report["confidence"] = 1.0
    report["novelty"] = "scheduled"
    report["policy_stage"] = "implementation"
    report["title"] = report["auction_name"].fillna("EEX Auction")
    report["source"] = "eex_official"
    report["published_at"] = report["Date"].dt.strftime("%Y-%m-%d")
    report["event_date"] = report["Date"].dt.strftime("%Y-%m-%d")
    report["body_text"] = (
        "Official EEX auction result. "
        + "Price EUR/tCO2="
        + report["auction_price_eur"].astype(str)
        + ", volume tCO2="
        + report["auction_volume_tco2"].astype(str)
        + ", volume_ratio_20d="
        + report["auction_volume_ratio_20d"].round(4).astype(str)
    )
    result_records = report[
        [
            "source",
            "published_at",
            "event_date",
            "title",
            "body_text",
            "event_type",
            "affected_channel",
            "supply_direction",
            "intensity",
            "expected_horizon",
            "novelty",
            "policy_stage",
            "confidence",
        ]
    ].rename(columns={"supply_direction": "direction"})

    future_sched = calendar_df.copy()
    future_sched["event_type"] = "auction_schedule"
    future_sched["affected_channel"] = "allowance_supply"
    future_sched["direction"] = "scheduled"
    future_sched["intensity"] = 1.0
    future_sched["expected_horizon"] = "5_20d"
    future_sched["novelty"] = "scheduled"
    future_sched["policy_stage"] = "implementation"
    future_sched["confidence"] = 1.0
    future_sched["published_at"] = datetime.now(timezone.utc).date().isoformat()
    future_sched["event_date"] = pd.to_datetime(future_sched["date"]).dt.strftime("%Y-%m-%d")
    future_sched["title"] = future_sched["auction_name"]
    future_sched["body_text"] = (
        "Official EEX auction calendar entry. "
        + "Volume tCO2="
        + future_sched["auction_volume_tco2"].fillna(0).astype(int).astype(str)
        + ", product="
        + future_sched["product_code"].fillna("")
    )
    future_sched["source"] = "eex_official"
    sched_records = future_sched[
        [
            "source",
            "published_at",
            "event_date",
            "title",
            "body_text",
            "event_type",
            "affected_channel",
            "direction",
            "intensity",
            "expected_horizon",
            "novelty",
            "policy_stage",
            "confidence",
        ]
    ]

    combined = pd.concat([result_records, sched_records], ignore_index=True)
    return combined.sort_values(["event_date", "title"], kind="stable").reset_index(drop=True)
