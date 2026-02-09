#!/usr/bin/env python3
"""Backfill GDELT Doc API headlines for one or more queries with resume support.

Why this exists:
- We want systematic, query-driven headline collection (not "download huge then filter").
- We need stable checkpointing for long date ranges.
- We avoid nested windowing by fetching one GDELT window per loop iteration.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
import sys

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from src.news.gdelt import fetch_gdelt_headlines  # noqa: E402


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


def _read_queries(path: str | None) -> list[str]:
    if not path:
        return []
    out = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            out.append(line)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill GDELT headlines for multiple queries.")
    parser.add_argument("--start", required=True, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", required=True, help="End date (YYYY-MM-DD)")
    parser.add_argument("--output", default="data/news/headlines_raw_multiquery.csv")
    parser.add_argument("--checkpoint", default="data/news/multiquery_checkpoint.json")
    parser.add_argument("--window-days", type=int, default=30)
    parser.add_argument("--sleep-seconds", type=float, default=1.0)
    parser.add_argument("--max-records", type=int, default=250)
    parser.add_argument("--language", default="English")
    parser.add_argument("--resume", action="store_true", default=False)
    parser.add_argument("--query", action="append", default=[], help="Repeatable query string.")
    parser.add_argument("--query-file", default=None, help="File with one query per line.")
    args = parser.parse_args()

    queries = list(args.query or [])
    queries.extend(_read_queries(args.query_file))
    queries = [q.strip() for q in queries if q and q.strip()]
    if not queries:
        raise SystemExit("Provide at least one --query or --query-file.")

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
            print(f"Resuming from {start.date().isoformat()}")

    if start > end:
        print("Start date is after end date; nothing to do.")
        return

    first_write = not output_path.exists()
    window_index = 0

    for window_start, window_end in _iter_windows(start, end, int(args.window_days)):
        window_index += 1
        all_articles = []
        for qi, query in enumerate(queries, start=1):
            # Fetch one window in one go (avoid nested sub-windowing).
            articles = fetch_gdelt_headlines(
                start_date=window_start.date().isoformat(),
                end_date=window_end.date().isoformat(),
                query=query,
                window_days=10_000,
                max_records=args.max_records,
                sleep_seconds=args.sleep_seconds,
                language_filter=args.language,
            )
            print(
                f"[{window_index}] {window_start.date()}–{window_end.date()} "
                f"query {qi}/{len(queries)} -> {len(articles)} articles"
            )
            if articles:
                for a in articles:
                    a["_query_id"] = qi
                all_articles.extend(articles)
            time.sleep(args.sleep_seconds)

        if all_articles:
            df = pd.DataFrame(all_articles)
            if "url" in df.columns:
                df = df[~df["url"].isin(seen_urls)]
                seen_urls.update(df["url"].dropna().tolist())
            if not df.empty:
                output_path.parent.mkdir(parents=True, exist_ok=True)
                df.to_csv(output_path, mode="a", header=first_write, index=False)
                first_write = False

        checkpoint = {
            "last_start": window_start.isoformat(),
            "last_end": window_end.isoformat(),
            "windows_completed": window_index,
            "output": str(output_path),
            "queries": queries,
        }
        _save_checkpoint(checkpoint_path, checkpoint)
        time.sleep(args.sleep_seconds)

    print(f"Backfill complete. Output: {output_path}")


if __name__ == "__main__":
    main()

