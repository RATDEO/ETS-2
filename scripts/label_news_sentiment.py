#!/usr/bin/env python3
"""Label headlines with LLM sentiment and save to CSV."""

import argparse
from pathlib import Path
import sys

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

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
    parser.add_argument("--base-url", default=None, help="OpenAI-compatible base URL (e.g. http://host:port/v1)")
    parser.add_argument("--api-key", default=None, help="API key for OpenAI-compatible endpoint")
    parser.add_argument("--timeout-seconds", type=float, default=None)
    parser.add_argument("--cache-dir", default="data/news/cache")
    parser.add_argument("--batch-size", type=int, default=200)
    parser.add_argument("--sleep-seconds", type=float, default=0.2)
    parser.add_argument(
        "--scoring-mode",
        choices=["effect", "effect_importance"],
        default="effect",
        help="Label direction only, or direction + importance (0-10).",
    )
    parser.add_argument(
        "--vote-cache-mode",
        choices=["shared", "independent"],
        default="independent",
        help="Whether vote calls share cache entries or use independent cache keys.",
    )
    parser.add_argument(
        "--importance-confidence-power",
        type=float,
        default=1.0,
        help="Confidence penalty exponent for importance-weighted score (0 disables penalty).",
    )
    parser.add_argument(
        "--vote-method",
        choices=["per_call", "single_call_multi_vote"],
        default="per_call",
        help="Voting execution mode for effect_importance scoring.",
    )
    parser.add_argument("--resume", action="store_true", default=False)
    parser.add_argument("--id-col", default="url")
    parser.add_argument(
        "--max-per-day-mode",
        choices=["random", "top_score"],
        default="random",
        help="If max-per-day is set, sample randomly or take top by relevance_score.",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    if args.max_per_day and args.max_per_day > 0 and args.date_col in df.columns:
        df[args.date_col] = pd.to_datetime(df[args.date_col]).dt.date
        if args.max_per_day_mode == "top_score" and "relevance_score" in df.columns:
            df = (
                df.sort_values(
                    [args.date_col, "relevance_score"],
                    ascending=[True, False],
                    kind="mergesort",
                )
                .groupby(args.date_col, group_keys=False)
                .head(args.max_per_day)
                .reset_index(drop=True)
            )
        else:
            df = (
                df.groupby(args.date_col, group_keys=False)
                .apply(lambda x: x.sample(n=min(len(x), args.max_per_day), random_state=42))
                .reset_index(drop=True)
            )
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    seen_ids = set()
    if args.resume and output_path.exists():
        try:
            existing = pd.read_csv(output_path, usecols=[args.id_col])
            seen_ids = set(existing[args.id_col].dropna().astype(str).tolist())
            print(f"Resuming: found {len(seen_ids)} existing labeled rows.")
        except Exception as exc:
            print(f"Resume failed to read existing output: {exc}")

    if seen_ids and args.id_col in df.columns:
        df = df[~df[args.id_col].astype(str).isin(seen_ids)]

    total = len(df)
    if total == 0:
        print("No new headlines to label.")
        return

    header_written = output_path.exists()
    for start_idx in range(0, total, args.batch_size):
        batch = df.iloc[start_idx:start_idx + args.batch_size]
        labeled = label_headlines_with_llm(
            batch,
            headline_col=args.headline_col,
            model=args.model,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            votes=args.votes,
            cache_dir=args.cache_dir,
            api_key=args.api_key,
            base_url=args.base_url,
            timeout_seconds=args.timeout_seconds,
            sleep_seconds=args.sleep_seconds,
            scoring_mode=args.scoring_mode,
            vote_cache_mode=args.vote_cache_mode,
            importance_confidence_power=args.importance_confidence_power,
            vote_method=args.vote_method,
        )
        labeled.to_csv(output_path, index=False, mode="a", header=not header_written)
        header_written = True
        print(f"Labeled {min(start_idx + args.batch_size, total)}/{total}")

    print(f"Saved labeled headlines to {output_path}")


if __name__ == "__main__":
    main()
