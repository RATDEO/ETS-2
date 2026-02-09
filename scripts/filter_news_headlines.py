#!/usr/bin/env python3
"""Filter EU/English headlines and remove near-duplicates."""

from __future__ import annotations

import argparse
import difflib
import re
import hashlib
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

EU_ETS_STRONG_TERMS = [
    "eu ets",
    "european union emissions trading",
    "european emissions trading",
    "emissions trading system",
    "eua",
    "euas",
    "eua futures",
    "eua auction",
    "ice eua",
    "eex",
    "market stability reserve",
    "msr",
    "fit for 55",
    "cbam",
    "carbon border adjustment",
]

EU_CONTEXT_TERMS = [
    "eu ",
    " eu",
    "european",
    "brussels",
    "europe",
    "eurozone",
    "european commission",
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

NON_EU_MARKET_TERMS = [
    "california",
    "rggi",
    "regional greenhouse gas initiative",
    "quebec",
    "ontario",
    "china ets",
    "chinese ets",
    "korea ets",
    "korean ets",
    "uk ets",
    "australia",
    "new zealand ets",
    "nz ets",
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


def _tokenize(norm: str) -> list[str]:
    if not norm:
        return []
    return [t for t in norm.split() if len(t) > 2]


def _simhash64(tokens: list[str]) -> int:
    """Compute a simple 64-bit simhash for a list of tokens."""
    if not tokens:
        return 0
    vec = [0] * 64
    for tok in tokens:
        h = hashlib.md5(tok.encode("utf-8")).digest()
        x = int.from_bytes(h[:8], byteorder="big", signed=False)
        for i in range(64):
            bit = 1 if (x >> i) & 1 else 0
            vec[i] += 1 if bit else -1
    out = 0
    for i, v in enumerate(vec):
        if v >= 0:
            out |= (1 << i)
    return out


def _hamming(a: int, b: int) -> int:
    return int((a ^ b).bit_count())


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


def _score_eu_ets_relevance(title: str) -> tuple[int, list[str]]:
    """
    Heuristic relevance score for *EU ETS / EUA price drivers*.

    Goal: keep enough daily coverage while avoiding obvious non-EU cap-and-trade markets.
    """
    norm = _normalize_title(title)
    if not norm:
        return 0, []

    hits: list[str] = []
    score = 0

    if any(term in norm for term in EU_ETS_STRONG_TERMS):
        score += 6
        hits.append("eu_ets_strong")

    if any(term in norm for term in EU_CONTEXT_TERMS):
        score += 2
        hits.append("eu_context")

    carbon_hits = sum(1 for term in CARBON_MARKET_TERMS if term in norm)
    if carbon_hits:
        score += min(4, carbon_hits)  # cap contribution
        hits.append(f"carbon_terms={carbon_hits}")

    if any(term in norm for term in ENERGY_DRIVER_TERMS):
        score += 1
        hits.append("energy_driver")

    # Penalize obvious non-EU markets unless EU ETS is explicitly present.
    if any(term in norm for term in NON_EU_MARKET_TERMS) and "eu ets" not in norm and "eua" not in norm:
        score -= 6
        hits.append("non_eu_market")

    # "cap and trade" is often non-EU; penalize unless EU context exists.
    if "cap and trade" in norm and not any(term in norm for term in EU_CONTEXT_TERMS):
        score -= 3
        hits.append("cap_and_trade_penalty")

    return score, hits


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


def _dedupe_group_simhash(df: pd.DataFrame, max_hamming: int) -> pd.DataFrame:
    kept_rows = []
    kept_hashes: list[int] = []
    for _, row in df.iterrows():
        title = str(row.get("title", "")).strip()
        if not title:
            continue
        norm = _normalize_title(title)
        h = _simhash64(_tokenize(norm))
        is_dup = False
        for existing in kept_hashes:
            if _hamming(h, existing) <= max_hamming:
                is_dup = True
                break
        if not is_dup:
            kept_rows.append(row)
            kept_hashes.append(h)
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


def _dedupe_across_days_simhash(
    df: pd.DataFrame,
    window_days: int,
    max_hamming: int,
) -> pd.DataFrame:
    kept_rows = []
    recent: list[tuple[pd.Timestamp, int]] = []

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
        recent = [(d, h) for d, h in recent if d >= cutoff]
        norm = _normalize_title(title)
        h = _simhash64(_tokenize(norm))

        is_dup = False
        for _, existing in recent:
            if _hamming(h, existing) <= max_hamming:
                is_dup = True
                break
        if not is_dup:
            kept_rows.append(row)
            recent.append((date, h))

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
    parser.add_argument(
        "--relevance-scheme",
        choices=["legacy", "eu_ets_score"],
        default="legacy",
        help="Legacy keyword filtering or EU-ETS-focused relevance scoring.",
    )
    parser.add_argument("--min-relevance-score", type=int, default=6)
    parser.add_argument("--event-similarity-days", type=int, default=0)
    parser.add_argument("--event-similarity-threshold", type=float, default=0.6)
    parser.add_argument(
        "--dedupe-method",
        choices=["sequence", "simhash"],
        default="sequence",
        help="Within-day/rolling dedupe method (SequenceMatcher or simhash).",
    )
    parser.add_argument("--simhash-max-hamming", type=int, default=3)
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

    if args.relevance_scheme == "eu_ets_score":
        scores = df["title"].astype(str).apply(_score_eu_ets_relevance)
        df["relevance_score"] = [s for s, _ in scores]
        df["relevance_hits"] = [";".join(h) for _, h in scores]
        df = df[df["relevance_score"] >= args.min_relevance_score]
    else:
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
            if args.dedupe_method == "simhash":
                deduped.append(_dedupe_group_simhash(group, args.simhash_max_hamming))
            else:
                deduped.append(_dedupe_group(group, args.similarity_threshold))
        df = pd.concat(deduped, ignore_index=True)
        df = df.drop(columns=["__day__"])
        if args.event_similarity_days and args.event_similarity_days > 0:
            df = df.sort_values("seendate")
            if args.dedupe_method == "simhash":
                df = _dedupe_across_days_simhash(
                    df,
                    window_days=args.event_similarity_days,
                    max_hamming=args.simhash_max_hamming,
                )
            else:
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
