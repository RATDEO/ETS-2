#!/usr/bin/env python3
"""Aggregate labeled headlines into a daily sentiment series."""

import argparse
from datetime import datetime
from zoneinfo import ZoneInfo
from pathlib import Path

import pandas as pd

from src.news.sentiment import aggregate_daily_sentiment


def main() -> None:
    parser = argparse.ArgumentParser(description="Build daily sentiment series.")
    parser.add_argument("--input", default="data/news/headlines_labeled.csv")
    parser.add_argument("--output", default="data/news/daily_sentiment.csv")
    parser.add_argument("--date-col", default="seendate")
    parser.add_argument("--score-col", default="llm_score")
    parser.add_argument(
        "--timing",
        choices=["all", "overnight", "intraday"],
        default="all",
        help="Filter headlines by overnight/intraday timing."
    )
    parser.add_argument("--timezone", default="Europe/Brussels")
    parser.add_argument("--overnight-start", default="09:00")
    parser.add_argument("--overnight-end", default="16:00")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    if args.timing != "all":
        if args.date_col not in df.columns:
            raise ValueError(f"Missing date column: {args.date_col}")
        start_time = datetime.strptime(args.overnight_start, "%H:%M").time()
        end_time = datetime.strptime(args.overnight_end, "%H:%M").time()
        ts = pd.to_datetime(df[args.date_col], errors="coerce", utc=True)
        local_ts = ts.dt.tz_convert(ZoneInfo(args.timezone))
        local_time = local_ts.dt.time
        valid_mask = local_ts.notna()
        overnight_mask = (local_time < start_time) | (local_time >= end_time)
        if args.timing == "overnight":
            df = df[valid_mask & overnight_mask]
        else:
            df = df[valid_mask & ~overnight_mask]
        print(f"Filtered to {len(df)} {args.timing} headlines")

    daily = aggregate_daily_sentiment(
        df, date_col=args.date_col, score_col=args.score_col
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    daily.to_csv(output_path, index=False)
    print(f"Saved daily sentiment series to {output_path}")


if __name__ == "__main__":
    main()
