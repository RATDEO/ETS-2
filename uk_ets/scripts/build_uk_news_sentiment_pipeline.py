#!/usr/bin/env python3
"""Build a UK ETS news sentiment corpus and daily sentiment series."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Iterable

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]

import sys

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.news.gdelt import build_gdelt_query, fetch_gdelt_headlines
from src.news.sentiment import label_headlines_with_llm
from src.news.uk_official_sources import fetch_elexon_rss_articles, fetch_govuk_search_articles
from uk_ets.scripts.filter_uk_news_headlines import filter_uk_headlines


UK_NEWS_KEYWORDS = [
    "UK ETS",
    "UK emissions trading scheme",
    "UK emissions allowance",
    "UKA futures",
    "UK carbon market",
    "UK carbon price",
    "UK carbon auction",
]

UK_NEWS_DOMAINS = [
    "gov.uk",
    "reuters.com",
    "ft.com",
    "bloomberg.com",
    "carbon-pulse.com",
    "argusmedia.com",
    "icis.com",
    "montelnews.com",
    "theice.com",
    "ice.com",
    "icapcarbonaction.com",
    "businessgreen.com",
]


def _write_daily_series(df: pd.DataFrame, score_col: str, out_path: Path) -> pd.DataFrame:
    work = df.copy()
    seendate = pd.to_datetime(work["seendate"], errors="coerce")
    if getattr(seendate.dt, "tz", None) is not None:
        seendate = seendate.dt.tz_convert(None)
    work["seendate"] = seendate.dt.normalize()
    work[score_col] = pd.to_numeric(work[score_col], errors="coerce").fillna(0.0)
    daily = (
        work.groupby("seendate", as_index=False)
        .agg(
            sent_score=(score_col, "mean"),
            news_volume=("title", "size"),
        )
        .sort_values("seendate")
        .reset_index(drop=True)
    )
    daily["sent_change"] = daily["sent_score"].diff().fillna(0.0)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    daily.to_csv(out_path, index=False)
    return daily


def _label_in_batches(
    headlines: pd.DataFrame,
    *,
    output_path: Path,
    cache_dir: Path,
    headline_col: str,
    model: str,
    base_url: str,
    api_key: str,
    timeout_seconds: float,
    max_tokens: int,
    votes: int,
    batch_size: int,
    sleep_seconds: float,
) -> pd.DataFrame:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    frames: list[pd.DataFrame] = []
    total = len(headlines)
    for start_idx in range(0, total, batch_size):
        batch = headlines.iloc[start_idx:start_idx + batch_size]
        labeled = label_headlines_with_llm(
            batch,
            headline_col=headline_col,
            model=model,
            temperature=0.0,
            max_tokens=max_tokens,
            votes=votes,
            cache_dir=str(cache_dir),
            api_key=api_key,
            base_url=base_url,
            timeout_seconds=timeout_seconds,
            sleep_seconds=sleep_seconds,
            scoring_mode="effect_importance",
            vote_cache_mode="independent",
            importance_confidence_power=1.0,
            vote_method="per_call",
        )
        frames.append(labeled)
        print(f"Labeled {min(start_idx + batch_size, total)}/{total}", flush=True)

    labeled_all = pd.concat(frames, ignore_index=True) if frames else headlines.head(0).copy()
    labeled_all.to_csv(output_path, index=False)
    return labeled_all


def main() -> int:
    parser = argparse.ArgumentParser(description="Build UK ETS daily news sentiment using the existing EU SENT workflow.")
    parser.add_argument("--data-root", default="uk_ets/Data_auto_uk")
    parser.add_argument("--source-mode", choices=["govuk", "govuk_plus_elexon", "gdelt"], default="govuk_plus_elexon")
    parser.add_argument("--news-start", default="2021-05-19")
    parser.add_argument("--news-end", default="2026-03-13")
    parser.add_argument("--window-days", type=int, default=30)
    parser.add_argument("--max-records", type=int, default=100)
    parser.add_argument("--govuk-pages-per-query", type=int, default=8)
    parser.add_argument("--use-domain-filter", action="store_true", default=False)
    parser.add_argument("--min-relevance-score", type=int, default=6)
    parser.add_argument("--model", default="qwen3-vl-4b-gpu")
    parser.add_argument("--base-url", default="http://192.168.1.140:9877/v1")
    parser.add_argument("--api-key", default="deo")
    parser.add_argument("--timeout-seconds", type=float, default=120.0)
    parser.add_argument("--votes", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=80)
    parser.add_argument("--max-per-day", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--sleep-seconds", type=float, default=0.05)
    args = parser.parse_args()

    data_root = (PROJECT_ROOT / args.data_root).resolve()
    news_dir = data_root / "news"
    news_dir.mkdir(parents=True, exist_ok=True)

    raw_path = news_dir / "headlines_raw.csv"
    filtered_path = news_dir / "headlines_filtered.csv"
    labeled_path = news_dir / "headlines_labeled_qwen_votes3_effect_importance.csv"
    baseline_daily_path = news_dir / "daily_sentiment_uk_qwen_votes3.csv"
    importance_daily_path = news_dir / "daily_sentiment_uk_qwen_votes3_importance.csv"

    query = None
    print(f"Fetching UK headlines: {args.news_start} -> {args.news_end}", flush=True)
    if args.source_mode == "gdelt":
        if args.use_domain_filter:
            query = build_gdelt_query(
                keywords=UK_NEWS_KEYWORDS,
                domains=UK_NEWS_DOMAINS,
            )
        else:
            query = build_gdelt_query(keywords=UK_NEWS_KEYWORDS)
        articles = fetch_gdelt_headlines(
            start_date=args.news_start,
            end_date=args.news_end,
            query=query,
            window_days=int(args.window_days),
            max_records=int(args.max_records),
            sleep_seconds=float(args.sleep_seconds),
            language_filter="English",
        )
        raw_df = pd.DataFrame(articles)
    else:
        import requests

        session = requests.Session()
        raw_df = fetch_govuk_search_articles(
            session,
            max_pages_per_query=int(args.govuk_pages_per_query),
            max_records=int(args.max_records),
            start_date=args.news_start,
            end_date=args.news_end,
        )
        if args.source_mode == "govuk_plus_elexon":
            elexon_df = fetch_elexon_rss_articles(session)
            if not elexon_df.empty:
                elexon_df = elexon_df[
                    (elexon_df["published_at"] >= pd.Timestamp(args.news_start, tz="UTC"))
                    & (elexon_df["published_at"] <= pd.Timestamp(args.news_end, tz="UTC") + pd.Timedelta(days=1))
                ].copy()
                raw_df = pd.concat([raw_df, elexon_df], ignore_index=True, sort=False)
        if raw_df.empty:
            raw_df = pd.DataFrame(columns=["source", "title", "url", "published_at", "summary", "body_text", "domain", "seendate"])
        else:
            raw_df = raw_df.copy()
            seendate = pd.to_datetime(raw_df["seendate"], errors="coerce")
            if getattr(seendate.dt, "tz", None) is not None:
                seendate = seendate.dt.tz_convert(None)
            raw_df["seendate"] = seendate.dt.normalize()
            raw_df["domain"] = raw_df.get("domain", pd.Series(index=raw_df.index, dtype=object)).fillna("")
    print(f"Fetched {len(raw_df)} raw UK headlines", flush=True)
    raw_df.to_csv(raw_path, index=False)

    filtered_df = filter_uk_headlines(
        raw_df,
        min_relevance_score=int(args.min_relevance_score),
        similarity_threshold=0.6,
        event_similarity_days=3,
        event_similarity_threshold=0.75,
        dedupe_method="simhash",
        simhash_max_hamming=3,
    )
    print(f"Filtered down to {len(filtered_df)} UK headlines", flush=True)
    if args.max_per_day and args.max_per_day > 0 and "seendate" in filtered_df.columns:
        filtered_df["seendate"] = pd.to_datetime(filtered_df["seendate"], errors="coerce")
        filtered_df["__day__"] = filtered_df["seendate"].dt.date
        if "relevance_score" in filtered_df.columns:
            filtered_df = (
                filtered_df.sort_values(
                    ["__day__", "relevance_score"],
                    ascending=[True, False],
                    kind="mergesort",
                )
                .groupby("__day__", group_keys=False)
                .head(int(args.max_per_day))
                .reset_index(drop=True)
            )
        filtered_df = filtered_df.drop(columns=["__day__"])
    print(f"After max-per-day cap: {len(filtered_df)} UK headlines", flush=True)
    filtered_df.to_csv(filtered_path, index=False)

    print("Starting LLM sentiment labeling", flush=True)
    labeled_df = _label_in_batches(
        filtered_df,
        output_path=labeled_path,
        cache_dir=news_dir / "cache_qwen_votes3",
        headline_col="title",
        model=args.model,
        base_url=args.base_url,
        api_key=args.api_key,
        timeout_seconds=float(args.timeout_seconds),
        max_tokens=int(args.max_tokens),
        votes=int(args.votes),
        batch_size=int(args.batch_size),
        sleep_seconds=float(args.sleep_seconds),
    )
    print(f"Labeled {len(labeled_df)} UK headlines", flush=True)

    baseline_daily = _write_daily_series(labeled_df, "llm_score", baseline_daily_path)
    importance_daily = _write_daily_series(labeled_df, "llm_score_importance", importance_daily_path)
    print(
        f"Wrote daily sentiment series: baseline={len(baseline_daily)} rows, importance={len(importance_daily)} rows",
        flush=True,
    )

    summary = {
        "generated_at": datetime.now().isoformat(),
        "source_mode": args.source_mode,
        "query": query,
        "raw_headlines": int(len(raw_df)),
        "filtered_headlines": int(len(filtered_df)),
        "labeled_headlines": int(len(labeled_df)),
        "baseline_daily_rows": int(len(baseline_daily)),
        "importance_daily_rows": int(len(importance_daily)),
        "raw_path": str(raw_path),
        "filtered_path": str(filtered_path),
        "labeled_path": str(labeled_path),
        "baseline_daily_path": str(baseline_daily_path),
        "importance_daily_path": str(importance_daily_path),
    }
    (news_dir / "uk_sentiment_build_summary.json").write_text(
        pd.Series(summary).to_json(indent=2),
        encoding="utf-8",
    )
    print(news_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
