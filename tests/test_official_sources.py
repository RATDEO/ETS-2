from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.news.official_sources import (
    build_deterministic_eex_event_records,
    parse_dg_clima_listing,
    parse_dg_clima_rss,
    parse_entsoe_rss,
)


def test_parse_dg_clima_rss_extracts_items() -> None:
    xml_text = """
    <rss><channel>
      <item>
        <title>Verified Emissions Report</title>
        <link>https://example.com/a</link>
        <pubDate>Tue, 18 Feb 2026 16:07:33 +0100</pubDate>
        <description>Published soon.</description>
      </item>
    </channel></rss>
    """
    rows = parse_dg_clima_rss(xml_text)
    assert len(rows) == 1
    assert rows[0]["title"] == "Verified Emissions Report"
    assert rows[0]["url"] == "https://example.com/a"


def test_parse_dg_clima_listing_extracts_news_links() -> None:
    html_text = """
    <html><body>
      <article>
        <a href="/news-other-reads/news/example-2026-01-01_en">Example title</a>
        <p>Short summary</p>
      </article>
    </body></html>
    """
    rows = parse_dg_clima_listing(html_text, base_url="https://climate.ec.europa.eu/news-other-reads/news_en")
    assert len(rows) == 1
    assert rows[0]["title"] == "Example title"
    assert rows[0]["url"].endswith("/news-other-reads/news/example-2026-01-01_en")


def test_parse_entsoe_rss_expands_relative_links() -> None:
    xml_text = """
    <rss><channel>
      <item>
        <title>Grid incident update</title>
        <link>/news/2026/02/05/power-system-incident-in-ukraine-and-moldova/</link>
        <pubDate>Thu, 05 Feb 2026 11:08:57 +0100</pubDate>
        <description>Incident affected the continental power system.</description>
      </item>
    </channel></rss>
    """
    rows = parse_entsoe_rss(xml_text)
    assert len(rows) == 1
    assert rows[0]["title"] == "Grid incident update"
    assert rows[0]["url"] == "https://www.entsoe.eu/news/2026/02/05/power-system-incident-in-ukraine-and-moldova/"


def test_build_deterministic_eex_event_records_emits_supply_and_schedule() -> None:
    calendar_df = pd.DataFrame(
        [
            {
                "date": "2026-03-10",
                "call_trading_period": "09.00 am - 11.00 am",
                "product_code": "T3PA",
                "trading_period": "4th Period",
                "auction_volume_tco2": 2712500,
                "auction_name": "Spot Market - EU Primary Auction CAP3 - EUA",
                "source": "eex_auction_calendar",
                "source_url": "https://example.com/calendar.xlsx",
            }
        ]
    )
    report_df = pd.DataFrame(
        [
            {"Date": "2026-03-01", "auction_name": "A1", "contract": "T3PA", "auction_price_eur": 70.0, "auction_volume_tco2": 200.0, "total_revenue_eur": 14000.0, "cover_ratio": 1.2, "country": "EU"},
            {"Date": "2026-03-02", "auction_name": "A2", "contract": "T3PA", "auction_price_eur": 71.0, "auction_volume_tco2": 220.0, "total_revenue_eur": 15620.0, "cover_ratio": 1.3, "country": "EU"},
            {"Date": "2026-03-03", "auction_name": "A3", "contract": "T3PA", "auction_price_eur": 72.0, "auction_volume_tco2": 210.0, "total_revenue_eur": 15120.0, "cover_ratio": 1.1, "country": "EU"},
            {"Date": "2026-03-04", "auction_name": "A4", "contract": "T3PA", "auction_price_eur": 73.0, "auction_volume_tco2": 205.0, "total_revenue_eur": 14965.0, "cover_ratio": 1.1, "country": "EU"},
            {"Date": "2026-03-05", "auction_name": "A5", "contract": "T3PA", "auction_price_eur": 74.0, "auction_volume_tco2": 300.0, "total_revenue_eur": 22200.0, "cover_ratio": 1.4, "country": "EU"},
        ]
    )
    report_df["Date"] = pd.to_datetime(report_df["Date"])
    events = build_deterministic_eex_event_records(calendar_df, report_df)
    assert {"auction_supply", "auction_schedule"} <= set(events["event_type"])
    supply_events = events.loc[events["event_type"] == "auction_supply"]
    assert supply_events["intensity"].notna().all()
