#!/usr/bin/env python3
"""Fetch EU ETS-related headlines from GDELT and save to CSV."""

import argparse
from pathlib import Path
import sys

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from src.news.gdelt import (
    fetch_gdelt_headlines,
    default_gdelt_query,
    build_gdelt_query,
    STRICT_KEYWORDS,
    EXPANDED_KEYWORDS,
    DEFAULT_THEME_TERMS,
    DEFAULT_DOMAINS,
)


def _parse_list(value: str | None) -> list[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def _load_domains_file(path: str | None) -> list[str]:
    if not path:
        return []
    domains = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            domains.append(line)
    return domains


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch GDELT headlines for EU ETS.")
    parser.add_argument("--start", required=True, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", required=True, help="End date (YYYY-MM-DD)")
    parser.add_argument("--output", default="data/news/headlines_raw.csv")
    parser.add_argument("--window-days", type=int, default=1)
    parser.add_argument("--max-records", type=int, default=250)
    parser.add_argument("--sleep-seconds", type=float, default=1.0)
    parser.add_argument("--query", default=None)
    parser.add_argument("--strict", action="store_true", default=False)
    parser.add_argument("--expanded", action="store_true", default=False)
    parser.add_argument("--themes", default=None)
    parser.add_argument("--use-default-themes", action="store_true", default=False)
    parser.add_argument("--domains", default=None)
    parser.add_argument("--domains-file", default=None)
    parser.add_argument("--use-default-domains", action="store_true", default=False)
    args = parser.parse_args()

    domains = _parse_list(args.domains)
    domains.extend(_load_domains_file(args.domains_file))
    themes = _parse_list(args.themes)

    if not args.query:
        if args.use_default_themes and not themes:
            themes = DEFAULT_THEME_TERMS
        if args.use_default_domains and not domains:
            domains = DEFAULT_DOMAINS

    if args.strict and not args.query:
        query = build_gdelt_query(
            keywords=STRICT_KEYWORDS,
            themes=themes,
            domains=domains,
        )
    elif args.expanded and not args.query:
        query = build_gdelt_query(
            keywords=EXPANDED_KEYWORDS,
            themes=themes,
            domains=domains,
        )
    else:
        query = args.query or default_gdelt_query()
    articles = fetch_gdelt_headlines(
        start_date=args.start,
        end_date=args.end,
        query=query,
        window_days=args.window_days,
        max_records=args.max_records,
        sleep_seconds=args.sleep_seconds,
        language_filter="English",
    )

    if not articles:
        print("No articles found.")
        return

    df = pd.DataFrame(articles)
    if domains and "domain" in df.columns:
        df = df[df["domain"].isin(domains)]
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Saved {len(df)} headlines to {output_path}")


if __name__ == "__main__":
    main()
