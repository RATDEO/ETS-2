#!/usr/bin/env python3
"""Build a dense daily UK news sentiment series with one selected headline per day."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]

import sys

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.news.sentiment import label_headlines_with_llm


ARCHIVE_PATH = PROJECT_ROOT / "data" / "news" / "headlines_raw_2017_2026.csv"
DEFAULT_CACHE_DIR = PROJECT_ROOT / "data" / "news" / "cache"
DEFAULT_PREFILLED_LABELS = (
    PROJECT_ROOT / "data" / "news" / "headlines_labeled_qwen_votes3_daily3_importance_v3_indep_t035_percall.csv"
)

STRONG_UK_ETS_TERMS = [
    "uk ets",
    "uk emissions trading scheme",
    "uk emissions allowance",
    "uk allowance",
    "uk carbon auction",
    "uka",
    "uka futures",
]

UK_TERMS = [
    "uk",
    "united kingdom",
    "britain",
    "british",
    "london",
    "westminster",
    "desnz",
    "beis",
    "ofgem",
    "elexon",
]

CARBON_TERMS = [
    "carbon",
    "emission",
    "emissions",
    "allowance",
    "allowances",
    "auction",
    "auctions",
    "permit",
    "permits",
    "cap and trade",
    "ets",
]

ENERGY_TERMS = [
    "energy",
    "power",
    "electricity",
    "gas",
    "lng",
    "coal",
    "oil",
    "brent",
    "renewable",
    "wind",
    "solar",
    "nuclear",
]

MACRO_TERMS = [
    "inflation",
    "interest rate",
    "interest rates",
    "sterling",
    "pound",
    "budget",
    "tax",
    "economy",
    "economic",
    "industry",
    "industrial",
    "manufacturing",
    "steel",
    "cement",
    "factory",
    "growth",
    "recession",
]

TRUSTED_DOMAIN_PATTERNS = [
    "reuters",
    "bloomberg",
    "ft",
    "financialtimes",
    "theguardian",
    "businessgreen",
    "edie",
    "energylivenews",
    "argus",
    "icis",
    "montel",
    "spglobal",
    "platts",
    "world-nuclear-news",
    "gov.uk",
    "ice",
    "eex",
]


@dataclass(frozen=True)
class DenseBuildPaths:
    news_dir: Path
    exact_selected: Path
    dense_selected: Path
    labeled_exact: Path
    baseline_daily: Path
    importance_daily: Path
    summary: Path


def _normalize_text(series: pd.Series) -> pd.Series:
    return series.astype(str).str.lower().fillna("")


def _contains_any(series: pd.Series, terms: Iterable[str]) -> pd.Series:
    escaped = [term.replace(".", r"\.") for term in terms]
    pattern = "|".join(escaped)
    return series.str.contains(pattern, regex=True, na=False)


def _resolve_paths(data_root: Path, prefix: str) -> DenseBuildPaths:
    news_dir = data_root / "news"
    news_dir.mkdir(parents=True, exist_ok=True)
    return DenseBuildPaths(
        news_dir=news_dir,
        exact_selected=news_dir / f"{prefix}_headlines_exact_selected.csv",
        dense_selected=news_dir / f"{prefix}_headlines_daily_selected.csv",
        labeled_exact=news_dir / f"{prefix}_headlines_labeled_qwen_votes3_effect_importance.csv",
        baseline_daily=news_dir / f"{prefix}_daily_sentiment_uk_qwen_votes3.csv",
        importance_daily=news_dir / f"{prefix}_daily_sentiment_uk_qwen_votes3_importance.csv",
        summary=news_dir / f"{prefix}_build_summary.json",
    )


def _load_archive(start_date: str, end_date: str) -> pd.DataFrame:
    df = pd.read_csv(ARCHIVE_PATH, usecols=["url", "title", "seendate", "domain", "language", "sourcecountry"])
    df = df[df["language"].astype(str).str.lower().eq("english")].copy()
    df["source_seendate"] = pd.to_datetime(df["seendate"], format="%Y%m%dT%H%M%SZ", errors="coerce")
    df["source_date"] = df["source_seendate"].dt.normalize()
    df = df[(df["source_date"] >= pd.Timestamp(start_date)) & (df["source_date"] <= pd.Timestamp(end_date))].copy()
    df = df.dropna(subset=["title", "source_date"]).reset_index(drop=True)
    return df


def _load_prefilled_labels(start_date: str, end_date: str, path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    df["source_seendate"] = pd.to_datetime(df["seendate"], errors="coerce")
    if getattr(df["source_seendate"].dt, "tz", None) is not None:
        df["source_seendate"] = df["source_seendate"].dt.tz_convert(None)
    df["source_date"] = df["source_seendate"].dt.normalize()
    df = df[(df["source_date"] >= pd.Timestamp(start_date)) & (df["source_date"] <= pd.Timestamp(end_date))].copy()
    return df.dropna(subset=["title", "source_date"]).reset_index(drop=True)


def _score_candidates(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    title = _normalize_text(work["title"])
    domain = _normalize_text(work["domain"])
    country = _normalize_text(work["sourcecountry"])

    strong = _contains_any(title, STRONG_UK_ETS_TERMS)
    ukish = _contains_any(title, UK_TERMS) | country.eq("united kingdom") | domain.str.contains(
        r"gov\.uk|theguardian|ft|telegraph|independent|standard|cityam|businessgreen|edie|energylivenews",
        regex=True,
        na=False,
    )
    carbon = _contains_any(title, CARBON_TERMS)
    energy = _contains_any(title, ENERGY_TERMS)
    macro = _contains_any(title, MACRO_TERMS)
    trusted = _contains_any(domain, TRUSTED_DOMAIN_PATTERNS)
    business = carbon | energy | macro | title.str.contains(
        r"market|markets|price|prices|policy|regulation|climate|net zero|economy|economic",
        regex=True,
        na=False,
    )

    carbon_count = sum(title.str.contains(term, regex=False, na=False) for term in CARBON_TERMS)
    energy_count = sum(title.str.contains(term, regex=False, na=False) for term in ENERGY_TERMS)
    macro_count = sum(title.str.contains(term, regex=False, na=False) for term in MACRO_TERMS)

    tier = pd.Series(4, index=work.index, dtype=int)
    tier = tier.mask(business, 3)
    tier = tier.mask(trusted & (carbon | energy), 2)
    tier = tier.mask(ukish & (carbon | energy | macro), 1)
    tier = tier.mask(strong, 0)

    score = (
        strong.astype(int) * 100
        + ukish.astype(int) * 20
        + trusted.astype(int) * 12
        + carbon.astype(int) * 12
        + energy.astype(int) * 8
        + macro.astype(int) * 5
        + carbon_count.clip(upper=5) * 3
        + energy_count.clip(upper=5) * 2
        + macro_count.clip(upper=5) * 1
    )
    work["selection_tier"] = tier
    work["selection_score"] = score
    work["is_strong_uk_ets"] = strong.astype(int)
    work["is_ukish"] = ukish.astype(int)
    work["is_trusted_domain"] = trusted.astype(int)
    work["is_business_relevant"] = business.astype(int)
    return work


def _select_exact_daily(candidates: pd.DataFrame) -> pd.DataFrame:
    ranked = candidates.sort_values(
        ["source_date", "selection_tier", "selection_score", "is_trusted_domain"],
        ascending=[True, True, False, False],
        kind="mergesort",
    ).copy()
    exact = ranked.groupby("source_date", group_keys=False).head(1).reset_index(drop=True)
    exact["selected_id"] = range(len(exact))
    return exact


def _merge_prefilled_exact(prefilled: pd.DataFrame, archive_scored: pd.DataFrame) -> pd.DataFrame:
    if prefilled.empty:
        return _select_exact_daily(archive_scored)

    prefilled_scored = _score_candidates(prefilled)
    prefilled_exact = _select_exact_daily(prefilled_scored)
    covered_dates = set(prefilled_exact["source_date"].tolist())
    archive_missing = archive_scored[~archive_scored["source_date"].isin(covered_dates)].copy()
    fallback_exact = _select_exact_daily(archive_missing) if not archive_missing.empty else archive_missing.copy()
    exact = pd.concat([prefilled_exact, fallback_exact], ignore_index=True, sort=False)
    exact = exact.sort_values("source_date").reset_index(drop=True)
    exact["selected_id"] = range(len(exact))
    exact["prefilled_label"] = exact["llm_score_importance"].notna().astype(int) if "llm_score_importance" in exact.columns else 0
    return exact


def _expand_to_dense_daily(exact_daily: pd.DataFrame, start_date: str, end_date: str) -> pd.DataFrame:
    calendar = pd.DataFrame({"target_date": pd.date_range(start_date, end_date, freq="D")})
    exact_dates = exact_daily["source_date"].tolist()
    exact_by_date = {pd.Timestamp(row["source_date"]): row for _, row in exact_daily.iterrows()}
    selected_rows = []

    for target_date in calendar["target_date"]:
        if target_date in exact_by_date:
            row = dict(exact_by_date[target_date])
            row["target_date"] = target_date
            row["selection_mode"] = "exact"
            row["gap_days"] = 0
            selected_rows.append(row)
            continue

        nearest = min(
            exact_dates,
            key=lambda d: (abs((d - target_date).days), 0 if d <= target_date else 1, d),
        )
        row = dict(exact_by_date[pd.Timestamp(nearest)])
        row["target_date"] = target_date
        row["selection_mode"] = "carry_previous" if nearest < target_date else "carry_next"
        row["gap_days"] = abs((nearest - target_date).days)
        selected_rows.append(row)

    dense = pd.DataFrame(selected_rows)
    dense["seendate"] = pd.to_datetime(dense["target_date"]).dt.normalize()
    return dense


def _label_exact_daily(
    exact_daily: pd.DataFrame,
    *,
    output_path: Path,
    cache_dir: Path,
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

    if "llm_score_importance" in exact_daily.columns:
        ready_mask = exact_daily["llm_score_importance"].notna() & exact_daily["llm_score"].notna()
    else:
        ready_mask = pd.Series(False, index=exact_daily.index)

    prefilled = exact_daily[ready_mask].copy()
    unlabeled = exact_daily[~ready_mask].copy()

    if unlabeled.empty:
        labeled_all = exact_daily.copy()
        labeled_all.to_csv(output_path, index=False)
        return labeled_all

    frames: list[pd.DataFrame] = []
    total = len(unlabeled)
    for start_idx in range(0, total, batch_size):
        batch = unlabeled.iloc[start_idx:start_idx + batch_size]
        labeled = label_headlines_with_llm(
            batch,
            headline_col="title",
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
        print(f"Labeled {min(start_idx + batch_size, total)}/{total} exact daily headlines", flush=True)

    labeled_new = pd.concat(frames, ignore_index=True) if frames else unlabeled.head(0).copy()
    labeled_all = pd.concat([prefilled, labeled_new], ignore_index=True, sort=False)
    labeled_all = labeled_all.sort_values("source_date").reset_index(drop=True)
    labeled_all.to_csv(output_path, index=False)
    return labeled_all


def _project_dense_labels(dense_daily: pd.DataFrame, labeled_exact: pd.DataFrame) -> pd.DataFrame:
    label_cols = [
        "selected_id",
        "llm_vote",
        "llm_score",
        "llm_importance",
        "llm_vote_agreement",
        "llm_score_importance_nopen",
        "llm_score_importance",
        "llm_votes",
        "llm_importance_votes",
        "llm_rationales",
    ]
    base = dense_daily.drop(columns=[c for c in dense_daily.columns if c in label_cols and c != "selected_id"], errors="ignore")
    merged = base.merge(labeled_exact[label_cols], on="selected_id", how="left", validate="many_to_one")
    return merged


def _write_daily_series(df: pd.DataFrame, score_col: str, out_path: Path) -> pd.DataFrame:
    work = df.copy()
    work["seendate"] = pd.to_datetime(work["seendate"], errors="coerce").dt.normalize()
    work[score_col] = pd.to_numeric(work[score_col], errors="coerce").fillna(0.0)
    daily = work[["seendate", score_col]].copy()
    daily = daily.rename(columns={score_col: "sent_score"})
    daily["news_volume"] = 1
    daily["sent_change"] = daily["sent_score"].diff().fillna(0.0)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    daily.to_csv(out_path, index=False)
    return daily


def _write_summary(
    paths: DenseBuildPaths,
    *,
    start_date: str,
    end_date: str,
    exact_daily: pd.DataFrame,
    dense_daily: pd.DataFrame,
    baseline_daily: pd.DataFrame,
    importance_daily: pd.DataFrame,
) -> None:
    summary = {
        "generated_at": datetime.now().isoformat(),
        "archive_path": str(ARCHIVE_PATH),
        "start_date": start_date,
        "end_date": end_date,
        "exact_selected_days": int(len(exact_daily)),
        "dense_selected_days": int(len(dense_daily)),
        "exact_coverage_pct": float(len(exact_daily) / len(dense_daily)) if len(dense_daily) else 0.0,
        "carry_days": int((dense_daily["selection_mode"] != "exact").sum()),
        "max_gap_days": int(dense_daily["gap_days"].max()) if len(dense_daily) else 0,
        "selection_mode_counts": {str(k): int(v) for k, v in dense_daily["selection_mode"].value_counts().to_dict().items()},
        "selection_tier_counts": {str(k): int(v) for k, v in exact_daily["selection_tier"].value_counts().sort_index().to_dict().items()},
        "baseline_daily_rows": int(len(baseline_daily)),
        "importance_daily_rows": int(len(importance_daily)),
        "exact_selected_path": str(paths.exact_selected),
        "dense_selected_path": str(paths.dense_selected),
        "labeled_exact_path": str(paths.labeled_exact),
        "baseline_daily_path": str(paths.baseline_daily),
        "importance_daily_path": str(paths.importance_daily),
    }
    paths.summary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a dense daily UK ETS sentiment series from the archived news pool.")
    parser.add_argument("--data-root", default="uk_ets/Data_auto_uk")
    parser.add_argument("--prefix", default="dense_daily1")
    parser.add_argument("--news-start", default="2021-05-19")
    parser.add_argument("--news-end", default="2026-03-13")
    parser.add_argument("--model", default="qwen3-vl-4b-gpu")
    parser.add_argument("--base-url", default="http://192.168.1.140:9877/v1")
    parser.add_argument("--api-key", default="deo")
    parser.add_argument("--timeout-seconds", type=float, default=120.0)
    parser.add_argument("--votes", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=200)
    parser.add_argument("--sleep-seconds", type=float, default=0.05)
    parser.add_argument("--cache-dir", default=str(DEFAULT_CACHE_DIR))
    parser.add_argument("--prefilled-labels-path", default=str(DEFAULT_PREFILLED_LABELS))
    args = parser.parse_args()

    data_root = (PROJECT_ROOT / args.data_root).resolve()
    paths = _resolve_paths(data_root, args.prefix)

    print("Loading archived headline pool", flush=True)
    archive = _load_archive(args.news_start, args.news_end)
    scored = _score_candidates(archive)
    prefilled = _load_prefilled_labels(args.news_start, args.news_end, Path(args.prefilled_labels_path).expanduser().resolve())
    exact_daily = _merge_prefilled_exact(prefilled, scored)
    exact_daily.to_csv(paths.exact_selected, index=False)
    prefilled_count = int(exact_daily.get("prefilled_label", pd.Series(dtype=int)).sum()) if len(exact_daily) else 0
    print(f"Exact same-day selections: {len(exact_daily)} (prefilled labels={prefilled_count})", flush=True)

    dense_daily = _expand_to_dense_daily(exact_daily, args.news_start, args.news_end)
    dense_daily.to_csv(paths.dense_selected, index=False)
    print(
        f"Dense daily selections: {len(dense_daily)} rows, "
        f"carry days={(dense_daily['selection_mode'] != 'exact').sum()}",
        flush=True,
    )

    labeled_exact = _label_exact_daily(
        exact_daily,
        output_path=paths.labeled_exact,
        cache_dir=Path(args.cache_dir).expanduser().resolve(),
        model=args.model,
        base_url=args.base_url,
        api_key=args.api_key,
        timeout_seconds=float(args.timeout_seconds),
        max_tokens=int(args.max_tokens),
        votes=int(args.votes),
        batch_size=int(args.batch_size),
        sleep_seconds=float(args.sleep_seconds),
    )
    dense_labeled = _project_dense_labels(dense_daily, labeled_exact)

    baseline_daily = _write_daily_series(dense_labeled, "llm_score", paths.baseline_daily)
    importance_daily = _write_daily_series(dense_labeled, "llm_score_importance", paths.importance_daily)
    _write_summary(
        paths,
        start_date=args.news_start,
        end_date=args.news_end,
        exact_daily=exact_daily,
        dense_daily=dense_daily,
        baseline_daily=baseline_daily,
        importance_daily=importance_daily,
    )
    print(paths.summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
