#!/usr/bin/env python3
"""Label headlines with LLM sentiment and save to CSV."""

import argparse
from pathlib import Path

import pandas as pd

from src.news.sentiment import label_headlines_with_llm


def main() -> None:
    parser = argparse.ArgumentParser(description="Label news headlines with LLM sentiment.")
    parser.add_argument("--input", default="data/news/headlines_filtered.csv")
    parser.add_argument("--output", default="data/news/headlines_labeled.csv")
    parser.add_argument("--headline-col", default="title")
    parser.add_argument("--date-col", default="seendate")
    parser.add_argument("--max-per-day", type=int, default=0)
    parser.add_argument("--model", default="gpt-5.2")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=80)
    parser.add_argument("--votes", type=int, default=3)
    parser.add_argument("--cache-dir", default="data/news/cache")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    if args.max_per_day and args.max_per_day > 0 and args.date_col in df.columns:
        df[args.date_col] = pd.to_datetime(df[args.date_col]).dt.date
        df = (
            df.groupby(args.date_col, group_keys=False)
            .apply(lambda x: x.sample(n=min(len(x), args.max_per_day), random_state=42))
            .reset_index(drop=True)
        )
    labeled = label_headlines_with_llm(
        df,
        headline_col=args.headline_col,
        model=args.model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        votes=args.votes,
        cache_dir=args.cache_dir,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    labeled.to_csv(output_path, index=False)
    print(f"Saved {len(labeled)} labeled headlines to {output_path}")


if __name__ == "__main__":
    main()
