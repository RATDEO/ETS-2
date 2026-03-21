#!/usr/bin/env python3
"""Filter UK ETS news headlines and remove near-duplicates."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.filter_news_headlines import (
    _dedupe_across_days,
    _dedupe_across_days_simhash,
    _dedupe_group,
    _dedupe_group_simhash,
    _normalize_title,
)


UK_ETS_STRONG_TERMS = [
    "uk ets",
    "uk emissions trading scheme",
    "uka",
    "uka futures",
    "uk allowance",
    "uk allowances",
    "uk carbon auction",
    "uk emissions allowance",
]

UK_CONTEXT_TERMS = [
    "united kingdom",
    "britain",
    "british",
    "uk government",
    "desnz",
    "beis",
    "westminster",
    "london",
]

CARBON_MARKET_TERMS = [
    "carbon",
    "emission",
    "emissions",
    "allowance",
    "allowances",
    "permit",
    "permits",
    "auction",
    "auctions",
    "futures",
    "trading",
    "market",
    "price",
]

ENERGY_DRIVER_TERMS = [
    "gas",
    "lng",
    "power",
    "electricity",
    "coal",
    "brent",
    "oil",
]

UK_POLICY_TERMS = [
    "cap and trade",
    "carbon budget",
    "net zero",
    "uk auction",
    "free allocation",
    "consultation",
    "scheme review",
]

NON_UK_MARKET_TERMS = [
    "eu ets",
    "eua",
    "euas",
    "european union allowance",
    "california",
    "rggi",
    "china ets",
    "korea ets",
    "nz ets",
]


def score_uk_ets_relevance(title: str) -> tuple[int, list[str]]:
    """Heuristic relevance score for UK ETS / UKA price drivers."""
    norm = _normalize_title(title)
    if not norm:
        return 0, []

    hits: list[str] = []
    score = 0

    if any(term in norm for term in UK_ETS_STRONG_TERMS):
        score += 7
        hits.append("uk_ets_strong")

    if any(term in norm for term in UK_CONTEXT_TERMS):
        score += 2
        hits.append("uk_context")

    carbon_hits = sum(1 for term in CARBON_MARKET_TERMS if term in norm)
    if carbon_hits:
        score += min(4, carbon_hits)
        hits.append(f"carbon_terms={carbon_hits}")

    if any(term in norm for term in ENERGY_DRIVER_TERMS):
        score += 1
        hits.append("energy_driver")

    if any(term in norm for term in UK_POLICY_TERMS):
        score += 2
        hits.append("uk_policy")

    if any(term in norm for term in NON_UK_MARKET_TERMS) and "uk ets" not in norm and "uka" not in norm:
        score -= 6
        hits.append("non_uk_market")

    return score, hits


def filter_uk_headlines(
    df: pd.DataFrame,
    *,
    min_relevance_score: int = 6,
    similarity_threshold: float = 0.6,
    event_similarity_days: int = 0,
    event_similarity_threshold: float = 0.6,
    dedupe_method: str = "sequence",
    simhash_max_hamming: int = 3,
) -> pd.DataFrame:
    if "title" not in df.columns:
        raise ValueError("Expected a 'title' column in the input file.")

    work = df.copy()
    if "language" in work.columns:
        work = work[work["language"].astype(str).str.lower() == "english"]

    scores = work["title"].astype(str).apply(score_uk_ets_relevance)
    work["relevance_score"] = [s for s, _ in scores]
    work["relevance_hits"] = [";".join(h) for _, h in scores]
    work = work[work["relevance_score"] >= int(min_relevance_score)].copy()

    subset_cols = [col for col in ["url", "title"] if col in work.columns]
    if subset_cols:
        work = work.drop_duplicates(subset=subset_cols).reset_index(drop=True)

    if "seendate" in work.columns:
        work["seendate"] = pd.to_datetime(work["seendate"], errors="coerce")
        work["__day__"] = work["seendate"].dt.date
        deduped = []
        for _, group in work.groupby("__day__", dropna=False):
            if dedupe_method == "simhash":
                deduped.append(_dedupe_group_simhash(group, simhash_max_hamming))
            else:
                deduped.append(_dedupe_group(group, similarity_threshold))
        work = pd.concat(deduped, ignore_index=True)
        work = work.drop(columns=["__day__"])
        if event_similarity_days and event_similarity_days > 0:
            work = work.sort_values("seendate")
            if dedupe_method == "simhash":
                work = _dedupe_across_days_simhash(
                    work,
                    window_days=event_similarity_days,
                    max_hamming=simhash_max_hamming,
                )
            else:
                work = _dedupe_across_days(
                    work,
                    threshold=event_similarity_threshold,
                    window_days=event_similarity_days,
                )

    return work.reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Filter UK ETS headlines and remove near-duplicates.")
    parser.add_argument("--input", default="uk_ets/Data_auto_uk/news/headlines_raw.csv")
    parser.add_argument("--output", default="uk_ets/Data_auto_uk/news/headlines_filtered.csv")
    parser.add_argument("--min-relevance-score", type=int, default=6)
    parser.add_argument("--similarity-threshold", type=float, default=0.6)
    parser.add_argument("--event-similarity-days", type=int, default=0)
    parser.add_argument("--event-similarity-threshold", type=float, default=0.6)
    parser.add_argument("--dedupe-method", choices=["sequence", "simhash"], default="sequence")
    parser.add_argument("--simhash-max-hamming", type=int, default=3)
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    filtered = filter_uk_headlines(
        df,
        min_relevance_score=args.min_relevance_score,
        similarity_threshold=args.similarity_threshold,
        event_similarity_days=args.event_similarity_days,
        event_similarity_threshold=args.event_similarity_threshold,
        dedupe_method=args.dedupe_method,
        simhash_max_hamming=args.simhash_max_hamming,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    filtered.to_csv(output_path, index=False)
    print(f"Saved {len(filtered)} filtered headlines to {output_path}")


if __name__ == "__main__":
    main()
