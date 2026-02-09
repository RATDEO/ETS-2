#!/usr/bin/env python3
"""Backfill GDELT headlines with checkpointing and resume support."""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timedelta
from pathlib import Path
import sys

import pandas as pd

# Ensure repo root is on sys.path for src imports
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from src.news.gdelt import (  # noqa: E402
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


def _iter_windows(start: datetime, end: datetime, window_days: int):
    cursor = start
    delta = timedelta(days=window_days)
    while cursor <= end:
        window_end = min(end, cursor + delta - timedelta(seconds=1))
        yield cursor, window_end
        cursor = window_end + timedelta(seconds=1)


def _load_checkpoint(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _save_checkpoint(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill GDELT headlines with resume support.")
    parser.add_argument("--start", required=True, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", required=True, help="End date (YYYY-MM-DD)")
    parser.add_argument("--output", default="data/news/headlines_raw_backfill.csv")
    parser.add_argument("--checkpoint", default="data/news/backfill_checkpoint.json")
    parser.add_argument("--window-days", type=int, default=7)
    parser.add_argument("--window-retries", type=int, default=5)
    parser.add_argument("--sleep-seconds", type=float, default=2.0)
    parser.add_argument("--max-records", type=int, default=250)
    parser.add_argument("--query", default=None)
    parser.add_argument("--strict", action="store_true", default=False)
    parser.add_argument("--expanded", action="store_true", default=False)
    parser.add_argument("--themes", default=None)
    parser.add_argument("--use-default-themes", action="store_true", default=False)
    parser.add_argument("--domains", default=None)
    parser.add_argument("--domains-file", default=None)
    parser.add_argument("--use-default-domains", action="store_true", default=False)
    parser.add_argument("--language", default="English")
    parser.add_argument("--resume", action="store_true", default=False)
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

    output_path = Path(args.output)
    checkpoint_path = Path(args.checkpoint)

    seen_urls = set()
    if output_path.exists():
        try:
            existing = pd.read_csv(output_path, usecols=["url"])
            seen_urls.update(existing["url"].dropna().tolist())
        except Exception:
            pass

    start = datetime.fromisoformat(args.start)
    end = datetime.fromisoformat(args.end)

    if args.resume:
        checkpoint = _load_checkpoint(checkpoint_path)
        last_end = checkpoint.get("last_end")
        if last_end:
            start = datetime.fromisoformat(last_end) + timedelta(seconds=1)
            print(f"Resuming from {start.isoformat()}")
    else:
        checkpoint = {}

    if start > end:
        print("Start date is after end date; nothing to do.")
        return

    first_write = not output_path.exists()
    window_index = int(checkpoint.get("windows_completed", 0))

    for window_start, window_end in _iter_windows(start, end, args.window_days):
        window_index += 1
        window_ok = False
        for attempt in range(args.window_retries):
            articles = fetch_gdelt_headlines(
                start_date=window_start.date().isoformat(),
                end_date=window_end.date().isoformat(),
                query=query,
                window_days=args.window_days,
                max_records=args.max_records,
                sleep_seconds=args.sleep_seconds,
                language_filter=args.language,
            )
            if articles:
                df = pd.DataFrame(articles)
                if "url" in df.columns:
                    df = df[~df["url"].isin(seen_urls)]
                    seen_urls.update(df["url"].dropna().tolist())
                if not df.empty:
                    df.to_csv(output_path, mode="a", header=first_write, index=False)
                    first_write = False
                window_ok = True
                break

            wait = max(args.sleep_seconds, 1.0) * (attempt + 1) * 5
            print(
                f"Empty response for {window_start.date()}–{window_end.date()} "
                f"(attempt {attempt + 1}/{args.window_retries}); sleeping {wait:.1f}s"
            )
            time.sleep(wait)

        if not window_ok:
            print(
                f"Window {window_start.date()}–{window_end.date()} still empty after retries."
            )

        checkpoint = {
            "last_start": window_start.isoformat(),
            "last_end": window_end.isoformat(),
            "windows_completed": window_index,
            "output": str(output_path),
        }
        _save_checkpoint(checkpoint_path, checkpoint)
        time.sleep(args.sleep_seconds)

    print(f"Backfill complete. Output: {output_path}")


if __name__ == "__main__":
    main()
