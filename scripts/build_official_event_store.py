#!/usr/bin/env python3
"""Build an automated official-source event store for EU ETS."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from news.event_extraction import extract_event_batch
from news.official_sources import (
    build_deterministic_eex_event_records,
    fetch_dg_clima_articles,
    fetch_entsoe_articles,
    fetch_eex_sources,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build automated official event store.")
    parser.add_argument("--output-dir", type=Path, default=Path("data/news/official"))
    parser.add_argument("--dg-pages", type=int, default=25, help="Number of DG CLIMA archive pages to crawl in addition to RSS.")
    parser.add_argument("--eex-report-year-min", type=int, default=2021, help="Minimum EEX auction-report year to ingest.")
    parser.add_argument("--extract-limit", type=int, default=20, help="Maximum DG CLIMA articles to send to the LLM extractor.")
    parser.add_argument("--entsoe-extract-limit", type=int, default=20, help="Maximum ENTSO-E articles to send to the LLM extractor.")
    parser.add_argument("--include-nonrelevant", action="store_true", default=False, help="Keep non-relevant extracted official article records in final outputs.")
    parser.add_argument("--llm-base-url", default="http://192.168.1.140:9877/v1")
    parser.add_argument("--llm-model", default=None, help="Model id. If omitted, discover first model from /models.")
    parser.add_argument("--llm-api-key", default="deo")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--timeout-seconds", type=float, default=60.0)
    return parser.parse_args()


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _discover_model_id(base_url: str, api_key: str) -> str:
    response = requests.get(
        base_url.rstrip("/") + "/models",
        timeout=20,
        headers={"Authorization": f"Bearer {api_key}"},
    )
    response.raise_for_status()
    payload = response.json()
    data = payload.get("data") or payload.get("models") or []
    if not data:
        raise ValueError(f"No models returned by {base_url}/models")
    first = data[0]
    return str(first.get("id") or first.get("model") or first.get("name"))


def _write_jsonl(path: Path, frame: pd.DataFrame) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in frame.to_dict(orient="records"):
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _extract_or_empty(
    rows: pd.DataFrame,
    *,
    model: str,
    base_url: str,
    api_key: str,
    cache_dir: Path,
    log_dir: Path,
    temperature: float,
    timeout_seconds: float,
) -> pd.DataFrame:
    if rows.empty:
        return pd.DataFrame(
            columns=[
                *rows.columns.tolist(),
                "is_relevant",
                "event_type",
                "affected_channel",
                "direction",
                "intensity",
                "expected_horizon",
                "novelty",
                "policy_stage",
                "confidence",
                "summary",
            ]
        )
    return extract_event_batch(
        rows,
        model=model,
        base_url=base_url,
        api_key=api_key,
        cache_dir=cache_dir,
        log_dir=log_dir,
        temperature=temperature,
        timeout_seconds=timeout_seconds,
    )


def main() -> int:
    args = parse_args()
    out_dir = args.output_dir if args.output_dir.is_absolute() else (ROOT / args.output_dir)
    _ensure_dir(out_dir)
    _ensure_dir(out_dir / "cache")
    _ensure_dir(out_dir / "logs")
    _ensure_dir(out_dir / "raw")

    llm_model = args.llm_model or _discover_model_id(args.llm_base_url, args.llm_api_key)

    session = requests.Session()
    dg_df = fetch_dg_clima_articles(session, pages=args.dg_pages)
    dg_df.to_csv(out_dir / "dg_clima_articles.csv", index=False)
    entsoe_df = fetch_entsoe_articles(session)
    entsoe_df.to_csv(out_dir / "entsoe_articles.csv", index=False)

    calendar_df, report_df, eex_meta = fetch_eex_sources(session, report_year_min=int(args.eex_report_year_min))
    calendar_df.to_csv(out_dir / "eex_auction_calendar.csv", index=False)
    report_df.to_csv(out_dir / "eex_auction_report.csv", index=False)

    dg_extract_input = dg_df[dg_df["cheap_relevance"] > 0].head(int(args.extract_limit)).copy()
    dg_events = _extract_or_empty(
        dg_extract_input,
        model=llm_model,
        base_url=args.llm_base_url,
        api_key=args.llm_api_key,
        cache_dir=out_dir / "cache" / "official_event_extract",
        log_dir=out_dir / "logs" / "official_event_extract",
        temperature=float(args.temperature),
        timeout_seconds=float(args.timeout_seconds),
    )
    dg_events["source"] = "dg_clima_official"
    dg_events["event_date"] = pd.to_datetime(dg_events["published_at"], errors="coerce").dt.date.astype("string")
    if not args.include_nonrelevant:
        dg_events = dg_events[dg_events["is_relevant"]].copy()
    dg_events.to_csv(out_dir / "dg_clima_event_records.csv", index=False)

    entsoe_extract_input = entsoe_df[entsoe_df["cheap_relevance"] > 0].head(int(args.entsoe_extract_limit)).copy()
    entsoe_events = _extract_or_empty(
        entsoe_extract_input,
        model=llm_model,
        base_url=args.llm_base_url,
        api_key=args.llm_api_key,
        cache_dir=out_dir / "cache" / "official_event_extract_entsoe",
        log_dir=out_dir / "logs" / "official_event_extract_entsoe",
        temperature=float(args.temperature),
        timeout_seconds=float(args.timeout_seconds),
    )
    entsoe_events["source"] = "entsoe_official"
    entsoe_events["event_date"] = pd.to_datetime(entsoe_events["published_at"], errors="coerce").dt.date.astype("string")
    if not args.include_nonrelevant:
        entsoe_events = entsoe_events[entsoe_events["is_relevant"]].copy()
    entsoe_events.to_csv(out_dir / "entsoe_event_records.csv", index=False)

    eex_events = build_deterministic_eex_event_records(calendar_df, report_df)
    eex_events.to_csv(out_dir / "eex_official_event_records.csv", index=False)

    common_cols = [
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
    dg_final = dg_events.reindex(columns=common_cols)
    entsoe_final = entsoe_events.reindex(columns=common_cols)
    eex_final = eex_events.reindex(columns=common_cols)
    combined = pd.concat([dg_final, entsoe_final, eex_final], ignore_index=True).sort_values(
        ["event_date", "source", "title"], ascending=[False, True, True], kind="stable"
    )
    combined.to_csv(out_dir / "official_event_records_v1.csv", index=False)
    _write_jsonl(out_dir / "official_event_records_v1.jsonl", combined)

    manifest = {
        "built_at_utc": datetime.now(timezone.utc).isoformat(),
        "llm_base_url": args.llm_base_url,
        "llm_model": llm_model,
        "dg_pages": int(args.dg_pages),
        "eex_report_year_min": int(args.eex_report_year_min),
        "dg_articles_total": int(len(dg_df)),
        "dg_articles_extracted": int(len(dg_extract_input)),
        "dg_events_kept": int(len(dg_final)),
        "entsoe_articles_total": int(len(entsoe_df)),
        "entsoe_articles_extracted": int(len(entsoe_extract_input)),
        "entsoe_events_kept": int(len(entsoe_final)),
        "eex_calendar_rows": int(len(calendar_df)),
        "eex_report_rows": int(len(report_df)),
        "eex_event_rows": int(len(eex_final)),
        "combined_event_rows": int(len(combined)),
        "sources": {
            "dg_clima_rss": "https://climate.ec.europa.eu/node/2/rss_en",
            "dg_clima_news": "https://climate.ec.europa.eu/news-other-reads/news_en",
            "entsoe_news_rss": "https://www.entsoe.eu/rss/news.xml",
            "eex_calendar_url": eex_meta["calendar_url"],
            "eex_report_urls": eex_meta["report_urls"],
        },
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"output_dir": str(out_dir), **manifest}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
