#!/usr/bin/env python3
"""Filter EU/English headlines and remove near-duplicates."""

from __future__ import annotations

import argparse
import difflib
import re
from pathlib import Path
from typing import Iterable, List

import pandas as pd

DEFAULT_KEYWORDS = [
    "emissions trading system",
    "carbon allowance",
    "carbon permit",
    "carbon price",
    "carbon market",
    "emissions allowance",
    "European emissions trading system",
    "European carbon market",
    "EUA futures",
]

STRICT_PRIMARY_KEYWORDS = [
    "eu ets",
    "european emissions trading system",
    "emissions trading system",
    "eua",
    "euas",
    "eua futures",
    "eu allowance",
    "european union allowance",
    "carbon allowance",
    "carbon permit",
    "emissions allowance",
]

STRICT_SECONDARY_KEYWORDS = [
    "carbon price",
    "carbon market",
    "carbon trading",
    "allowance price",
    "emissions trading",
    "cap and trade",
    "carbon credit",
    "carbon credits",
    "carbon auction",
    "emissions market",
]


EU_COUNTRIES = {
    "Austria",
    "Belgium",
    "Bulgaria",
    "Croatia",
    "Cyprus",
    "Czech Republic",
    "Denmark",
    "Estonia",
    "Finland",
    "France",
    "Germany",
    "Greece",
    "Hungary",
    "Ireland",
    "Italy",
    "Latvia",
    "Lithuania",
    "Luxembourg",
    "Malta",
    "Netherlands",
    "Poland",
    "Portugal",
    "Romania",
    "Slovakia",
    "Slovenia",
    "Spain",
    "Sweden",
}


def _normalize_title(title: str) -> str:
    title = title.lower()
    title = re.sub(r"[^a-z0-9\\s]", " ", title)
    title = re.sub(r"\\s+", " ", title).strip()
    return title


def _has_keyword(title: str, keywords: Iterable[str]) -> bool:
    text = title.lower()
    return any(k.lower() in text for k in keywords)

def _keyword_hits(title: str, keywords: Iterable[str]) -> int:
    norm = _normalize_title(title)
    tokens = set(norm.split())
    hits = 0
    for key in keywords:
        key = key.lower().strip()
        if not key:
            continue
        if " " in key:
            if key in norm:
                hits += 1
        else:
            if key in tokens:
                hits += 1
    return hits


def _passes_strict_relevance(
    title: str,
    primary: Iterable[str],
    secondary: Iterable[str],
    min_secondary: int
) -> bool:
    if _keyword_hits(title, primary) >= 1:
        return True
    return _keyword_hits(title, secondary) >= min_secondary


def _dedupe_group(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    kept_rows = []
    kept_titles: List[str] = []
    for _, row in df.iterrows():
        title = str(row.get("title", "")).strip()
        if not title:
            continue
        norm = _normalize_title(title)
        is_dup = False
        for existing in kept_titles:
            if difflib.SequenceMatcher(None, norm, existing).ratio() >= threshold:
                is_dup = True
                break
        if not is_dup:
            kept_rows.append(row)
            kept_titles.append(norm)
    if not kept_rows:
        return df.head(0)
    return pd.DataFrame(kept_rows)


def _dedupe_across_days(
    df: pd.DataFrame,
    threshold: float,
    window_days: int
) -> pd.DataFrame:
    kept_rows = []
    recent: List[tuple[pd.Timestamp, str]] = []

    for _, row in df.iterrows():
        title = str(row.get("title", "")).strip()
        if not title:
            continue
        date = row.get("seendate")
        if pd.isna(date):
            kept_rows.append(row)
            continue
        date = pd.to_datetime(date)
        cutoff = date - pd.Timedelta(days=window_days)
        recent = [(d, t) for d, t in recent if d >= cutoff]
        norm = _normalize_title(title)

        is_dup = False
        for _, existing in recent:
            if difflib.SequenceMatcher(None, norm, existing).ratio() >= threshold:
                is_dup = True
                break
        if not is_dup:
            kept_rows.append(row)
            recent.append((date, norm))

    if not kept_rows:
        return df.head(0)
    return pd.DataFrame(kept_rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Filter EU/English headlines and remove near-duplicates.")
    parser.add_argument("--input", default="data/news/headlines_raw.csv")
    parser.add_argument("--output", default="data/news/headlines_filtered.csv")
    parser.add_argument("--language", default="English")
    parser.add_argument("--all-sources", action="store_true", default=False)
    parser.add_argument("--include-uk", action="store_true", default=False)
    parser.add_argument("--similarity-threshold", type=float, default=0.6)
    parser.add_argument("--require-keywords", action="store_true", default=False)
    parser.add_argument("--strict-relevance", action="store_true", default=False)
    parser.add_argument("--min-secondary-hits", type=int, default=2)
    parser.add_argument("--event-similarity-days", type=int, default=0)
    parser.add_argument("--event-similarity-threshold", type=float, default=0.6)
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    if "title" not in df.columns:
        raise ValueError("Expected a 'title' column in the input file.")

    if args.language and "language" in df.columns:
        df = df[df["language"].str.lower() == args.language.lower()]

    if not args.all_sources and "sourcecountry" in df.columns:
        allowed = set(EU_COUNTRIES)
        if args.include_uk:
            allowed.add("United Kingdom")
        df = df[df["sourcecountry"].isin(allowed)]

    if args.strict_relevance:
        df = df[
            df["title"].apply(
                lambda t: _passes_strict_relevance(
                    str(t),
                    STRICT_PRIMARY_KEYWORDS,
                    STRICT_SECONDARY_KEYWORDS,
                    args.min_secondary_hits
                )
            )
        ]
    elif args.require_keywords:
        df = df[df["title"].apply(lambda t: _has_keyword(str(t), DEFAULT_KEYWORDS))]

    df = df.drop_duplicates(subset=["url", "title"]).reset_index(drop=True)

    if "seendate" in df.columns:
        df["seendate"] = pd.to_datetime(df["seendate"], errors="coerce")
        df["__day__"] = df["seendate"].dt.date
        deduped = []
        for _, group in df.groupby("__day__"):
            deduped.append(_dedupe_group(group, args.similarity_threshold))
        df = pd.concat(deduped, ignore_index=True)
        df = df.drop(columns=["__day__"])
        if args.event_similarity_days and args.event_similarity_days > 0:
            df = df.sort_values("seendate")
            df = _dedupe_across_days(
                df,
                threshold=args.event_similarity_threshold,
                window_days=args.event_similarity_days
            )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Saved {len(df)} filtered headlines to {output_path}")


if __name__ == "__main__":
    main()
