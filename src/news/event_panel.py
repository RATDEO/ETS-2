"""Structured daily news-event feature builder."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable
import re

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

EVENT_TAXONOMY_PATTERNS = {
    "policy": re.compile(
        r"\b(policy|commission|parliament|directive|regulation|reform|fit for 55|market stability reserve|msr|allowance|ets reform|aviation emissions trading)\b"
    ),
    "auction": re.compile(r"\b(auction|eex)\b"),
    "energy": re.compile(r"\b(gas|power|electricity|coal|oil|brent|lng|renewable|wind|solar|nuclear)\b"),
    "shipping": re.compile(r"\b(shipping|maritime|ship|ships|vessel|freight|port)\b"),
    "industry": re.compile(r"\b(steel|cement|industrial|manufactur|factory|production|demand)\b"),
    "macro": re.compile(r"\b(inflation|recession|gdp|economy|central bank|interest rate|tariff|trade war)\b"),
    "geopolitics": re.compile(r"\b(russia|ukraine|sanction|war|opec|middle east)\b"),
    "weather": re.compile(r"\b(weather|heatwave|cold spell|drought|storm|winter)\b"),
    "cbam": re.compile(r"\b(cbam|carbon border)\b"),
}

HIT_FEATURE_KEYS = (
    "evt_eu_ets_strong_count",
    "evt_energy_driver_count",
    "evt_eu_context_count",
    "evt_carbon_terms_sum",
)

STRUCTURED_EVENT_REQUIRED_COLS = {
    "event_date",
    "event_type",
    "affected_channel",
    "direction",
    "intensity",
    "expected_horizon",
    "novelty",
    "policy_stage",
    "confidence",
}

ROLLING_SUM_FEATURES = (
    "evt_news_count",
    "evt_sent_sum",
    "evt_abs_sent_sum",
    "evt_importance_sum",
    "evt_weighted_sent_sum",
    "evt_positive_count",
    "evt_negative_count",
    "evt_unknown_count",
    "evt_high_importance_count",
    "evt_eu_ets_strong_count",
    "evt_energy_driver_count",
    "evt_eu_context_count",
    "evt_carbon_terms_sum",
    "proxy_news_volume",
)

ROLLING_MEAN_FEATURES = (
    "evt_sent_mean",
    "evt_importance_mean",
    "evt_weighted_sent_mean",
    "evt_positive_share",
    "evt_negative_share",
    "proxy_sent_score",
    "proxy_sent_change",
    "topical_sent_score",
    "ds_sent_score",
    "ds_p_yes",
    "ds_p_no",
    "ds_p_unknown",
)


def parse_relevance_hits(raw: object) -> dict[str, float]:
    """Parse semicolon-delimited hit tags into stable numeric event counts."""
    result = {key: 0.0 for key in HIT_FEATURE_KEYS}
    if raw is None or (isinstance(raw, float) and np.isnan(raw)):
        return result

    text = str(raw).strip()
    if not text:
        return result

    for token in text.split(";"):
        token = token.strip()
        if not token:
            continue
        if token == "eu_ets_strong":
            result["evt_eu_ets_strong_count"] += 1.0
        elif token == "energy_driver":
            result["evt_energy_driver_count"] += 1.0
        elif token == "eu_context":
            result["evt_eu_context_count"] += 1.0
        elif token.startswith("carbon_terms="):
            try:
                result["evt_carbon_terms_sum"] += float(token.split("=", 1)[1])
            except ValueError:
                continue
    return result


def _resolve_path(path: str | Path) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    return ROOT / p


def _event_taxonomy_flags(title: object) -> dict[str, float]:
    text = str(title or "").lower()
    return {f"evt2_{name}_flag": float(bool(pattern.search(text))) for name, pattern in EVENT_TAXONOMY_PATTERNS.items()}


def _slug(text: object) -> str:
    value = re.sub(r"[^a-z0-9]+", "_", str(text or "").strip().lower()).strip("_")
    return value or "unknown"


def _is_structured_event_frame(df: pd.DataFrame) -> bool:
    return STRUCTURED_EVENT_REQUIRED_COLS.issubset(df.columns)


def _build_headline_event_panel(
    headlines: pd.DataFrame,
    taxonomy_version: str,
) -> tuple[pd.DataFrame, list[str]]:
    usecols = [
        "seendate",
        "relevance_hits",
        "llm_score",
        "llm_importance",
        "llm_score_importance",
    ]
    missing_required = [col for col in usecols if col not in headlines.columns]
    if missing_required:
        raise ValueError(f"Missing required columns in headline source: {missing_required}")
    if "title" not in headlines.columns:
        headlines["title"] = ""
    headlines = headlines[["title"] + usecols].copy()
    headlines["date"] = pd.to_datetime(headlines["seendate"]).dt.normalize()
    headlines["llm_score"] = pd.to_numeric(headlines["llm_score"], errors="coerce").fillna(0.0)
    headlines["llm_importance"] = pd.to_numeric(headlines["llm_importance"], errors="coerce").fillna(0.0)
    headlines["llm_score_importance"] = pd.to_numeric(
        headlines["llm_score_importance"], errors="coerce"
    ).fillna(0.0)

    hit_frame = pd.DataFrame.from_records(headlines["relevance_hits"].map(parse_relevance_hits))
    work = pd.concat([headlines[["date", "llm_score", "llm_importance", "llm_score_importance"]], hit_frame], axis=1)

    work["evt_news_count"] = 1.0
    work["evt_sent_sum"] = work["llm_score"]
    work["evt_abs_sent_sum"] = work["llm_score"].abs()
    work["evt_importance_sum"] = work["llm_importance"]
    work["evt_weighted_sent_sum"] = work["llm_score_importance"]
    work["evt_positive_count"] = (work["llm_score"] > 0).astype(float)
    work["evt_negative_count"] = (work["llm_score"] < 0).astype(float)
    work["evt_unknown_count"] = (work["llm_score"] == 0).astype(float)
    work["evt_high_importance_count"] = (work["llm_importance"] >= 7.0).astype(float)

    taxonomy_roll_sum_cols: list[str] = []
    if taxonomy_version == "v2":
        taxonomy_flags = pd.DataFrame.from_records(headlines["title"].map(_event_taxonomy_flags))
        taxonomy_work = pd.concat(
            [
                headlines[["date"]].reset_index(drop=True),
                taxonomy_flags,
                headlines[["llm_importance", "llm_score_importance"]].reset_index(drop=True),
            ],
            axis=1,
        )
        for name in EVENT_TAXONOMY_PATTERNS:
            flag_col = f"evt2_{name}_flag"
            count_col = f"evt2_{name}_count"
            importance_col = f"evt2_{name}_importance_sum"
            weighted_col = f"evt2_{name}_weighted_sent_sum"
            taxonomy_work[count_col] = taxonomy_work[flag_col]
            taxonomy_work[importance_col] = taxonomy_work[flag_col] * taxonomy_work["llm_importance"]
            taxonomy_work[weighted_col] = taxonomy_work[flag_col] * taxonomy_work["llm_score_importance"]
            taxonomy_roll_sum_cols.extend([count_col, importance_col, weighted_col])
        taxonomy_daily = (
            taxonomy_work.groupby("date", as_index=False)[taxonomy_roll_sum_cols].sum().sort_values("date").reset_index(drop=True)
        )
    elif taxonomy_version != "v1":
        raise ValueError(f"Unknown taxonomy_version for headline source: {taxonomy_version}")

    daily = (
        work.groupby("date", as_index=False)
        .agg(
            {
                "evt_news_count": "sum",
                "evt_sent_sum": "sum",
                "evt_abs_sent_sum": "sum",
                "evt_importance_sum": "sum",
                "evt_weighted_sent_sum": "sum",
                "evt_positive_count": "sum",
                "evt_negative_count": "sum",
                "evt_unknown_count": "sum",
                "evt_high_importance_count": "sum",
                "evt_eu_ets_strong_count": "sum",
                "evt_energy_driver_count": "sum",
                "evt_eu_context_count": "sum",
                "evt_carbon_terms_sum": "sum",
                "llm_score": "mean",
                "llm_importance": "mean",
                "llm_score_importance": "mean",
            }
        )
        .rename(
            columns={
                "llm_score": "evt_sent_mean",
                "llm_importance": "evt_importance_mean",
                "llm_score_importance": "evt_weighted_sent_mean",
            }
        )
        .sort_values("date")
        .reset_index(drop=True)
    )

    news_count = daily["evt_news_count"].replace(0.0, np.nan)
    daily["evt_positive_share"] = (daily["evt_positive_count"] / news_count).fillna(0.0)
    daily["evt_negative_share"] = (daily["evt_negative_count"] / news_count).fillna(0.0)

    if taxonomy_version == "v2":
        daily = daily.merge(taxonomy_daily, on="date", how="left")
    return daily, taxonomy_roll_sum_cols


def _build_structured_event_panel(events: pd.DataFrame, taxonomy_version: str) -> tuple[pd.DataFrame, list[str]]:
    if taxonomy_version not in {"official_v1"}:
        raise ValueError(f"Unknown taxonomy_version for structured event source: {taxonomy_version}")

    work = events.copy()
    work["date"] = pd.to_datetime(work["event_date"]).dt.normalize()
    work["intensity"] = pd.to_numeric(work["intensity"], errors="coerce").fillna(0.0)
    work["confidence"] = pd.to_numeric(work["confidence"], errors="coerce").fillna(1.0).clip(0.0, 1.0)
    work["direction"] = work["direction"].map(_slug)
    work["event_type"] = work["event_type"].map(_slug)
    work["affected_channel"] = work["affected_channel"].map(_slug)
    work["expected_horizon"] = work["expected_horizon"].map(_slug)
    work["novelty"] = work["novelty"].map(_slug)
    work["policy_stage"] = work["policy_stage"].map(_slug)

    dir_score_map = {
        "bullish": 1.0,
        "bearish": -1.0,
        "positive": 1.0,
        "negative": -1.0,
        "neutral": 0.0,
        "mixed": 0.0,
        "scheduled": 0.0,
        "unknown": 0.0,
    }
    work["dir_score"] = work["direction"].map(dir_score_map).fillna(0.0)
    work["evt_news_count"] = 1.0
    work["evt_sent_sum"] = work["dir_score"] * work["intensity"]
    work["evt_abs_sent_sum"] = work["evt_sent_sum"].abs()
    work["evt_importance_sum"] = work["intensity"]
    work["evt_weighted_sent_sum"] = work["evt_sent_sum"] * work["confidence"]
    work["evt_positive_count"] = (work["dir_score"] > 0).astype(float)
    work["evt_negative_count"] = (work["dir_score"] < 0).astype(float)
    work["evt_unknown_count"] = (work["dir_score"] == 0).astype(float)
    work["evt_high_importance_count"] = (work["intensity"] >= 2.0).astype(float)
    work["evt_eu_ets_strong_count"] = 0.0
    work["evt_energy_driver_count"] = work["affected_channel"].isin({"power_generation", "fuel_switching"}).astype(float)
    work["evt_eu_context_count"] = work["source"].astype(str).str.contains("dg_clima|eex_official", case=False, na=False).astype(float)
    work["evt_carbon_terms_sum"] = (
        work["title"].astype(str).str.contains(r"\b(?:carbon|ets|allowance|emission)\b", case=False, regex=True).astype(float)
    )

    roll_sum_cols: list[str] = []

    def _add_bucket_features(field: str, prefix: str, include_weighted: bool = False) -> pd.DataFrame:
        values = sorted(work[field].dropna().astype(str).unique().tolist())
        if not values:
            return pd.DataFrame({"date": work["date"]})
        frames: list[pd.DataFrame] = [work[["date"]].copy()]
        for value in values:
            flag = (work[field] == value).astype(float)
            count_col = f"{prefix}_{value}_count"
            frames.append(pd.DataFrame({count_col: flag}))
            roll_sum_cols.append(count_col)
            if include_weighted:
                weight_col = f"{prefix}_{value}_weighted_sum"
                frames.append(pd.DataFrame({weight_col: flag * work["evt_weighted_sent_sum"]}))
                roll_sum_cols.append(weight_col)
        merged = pd.concat(frames, axis=1)
        cols = [c for c in merged.columns if c != "date"]
        return merged.groupby("date", as_index=False)[cols].sum()

    type_daily = _add_bucket_features("event_type", "evt_official_type", include_weighted=True)
    channel_daily = _add_bucket_features("affected_channel", "evt_official_channel", include_weighted=False)
    horizon_daily = _add_bucket_features("expected_horizon", "evt_official_horizon", include_weighted=False)
    stage_daily = _add_bucket_features("policy_stage", "evt_official_stage", include_weighted=False)
    novelty_daily = _add_bucket_features("novelty", "evt_official_novelty", include_weighted=False)
    dir_daily = _add_bucket_features("direction", "evt_official_dir", include_weighted=False)

    sum_daily = (
        work.groupby("date", as_index=False)
        .agg(
            {
                "evt_news_count": "sum",
                "evt_sent_sum": "sum",
                "evt_abs_sent_sum": "sum",
                "evt_importance_sum": "sum",
                "evt_weighted_sent_sum": "sum",
                "evt_positive_count": "sum",
                "evt_negative_count": "sum",
                "evt_unknown_count": "sum",
                "evt_high_importance_count": "sum",
                "evt_eu_ets_strong_count": "sum",
                "evt_energy_driver_count": "sum",
                "evt_eu_context_count": "sum",
                "evt_carbon_terms_sum": "sum",
            }
        )
        .sort_values("date")
        .reset_index(drop=True)
    )
    mean_daily = (
        work.groupby("date", as_index=False)[["evt_sent_sum", "intensity", "evt_weighted_sent_sum"]]
        .mean()
        .rename(
            columns={
                "evt_sent_sum": "evt_sent_mean",
                "intensity": "evt_importance_mean",
                "evt_weighted_sent_sum": "evt_weighted_sent_mean",
            }
        )
        .sort_values("date")
        .reset_index(drop=True)
    )
    daily = sum_daily.merge(mean_daily, on="date", how="left")

    news_count = daily["evt_news_count"].replace(0.0, np.nan)
    daily["evt_positive_share"] = (daily["evt_positive_count"] / news_count).fillna(0.0)
    daily["evt_negative_share"] = (daily["evt_negative_count"] / news_count).fillna(0.0)

    for extra_daily in (type_daily, channel_daily, horizon_daily, stage_daily, novelty_daily, dir_daily):
        if len(extra_daily.columns) > 1:
            daily = daily.merge(extra_daily, on="date", how="left")

    return daily, roll_sum_cols


def _load_prefixed_daily_numeric(
    path: str | Path,
    prefix: str,
    date_col: str,
) -> pd.DataFrame:
    csv_path = _resolve_path(path)
    df = pd.read_csv(csv_path)
    if date_col not in df.columns:
        fallback = "seendate" if "seendate" in df.columns else ("date" if "date" in df.columns else None)
        if fallback is None:
            raise ValueError(f"Date column not found in {csv_path}")
        date_col = fallback

    work = df.copy()
    work["date"] = pd.to_datetime(work[date_col]).dt.normalize()

    numeric_cols: list[str] = []
    for col in work.columns:
        if col in {date_col, "date"}:
            continue
        converted = pd.to_numeric(work[col], errors="coerce")
        if converted.notna().any():
            work[col] = converted
            numeric_cols.append(col)

    if not numeric_cols:
        return pd.DataFrame({"date": work["date"].drop_duplicates().sort_values()})

    daily = work.groupby("date", as_index=False)[numeric_cols].mean()
    rename_map = {col: f"{prefix}{col}" for col in numeric_cols}
    return daily.rename(columns=rename_map).sort_values("date").reset_index(drop=True)


def build_event_panel(
    headlines_path: str | Path,
    calendar: Iterable[pd.Timestamp] | None = None,
    rolling_windows: Iterable[int] = (3, 7),
    proxy_daily_path: str | Path | None = None,
    novelty_daily_path: str | Path | None = None,
    ds_daily_path: str | Path | None = None,
    taxonomy_version: str = "v1",
) -> pd.DataFrame:
    """Aggregate headline labels into a structured daily event panel."""
    csv_path = _resolve_path(headlines_path)
    source_df = pd.read_csv(csv_path)
    if _is_structured_event_frame(source_df):
        daily, extra_roll_sum_cols = _build_structured_event_panel(source_df, taxonomy_version=taxonomy_version)
    else:
        daily, extra_roll_sum_cols = _build_headline_event_panel(source_df, taxonomy_version=taxonomy_version)

    if calendar is None:
        calendar_index = pd.DatetimeIndex(daily["date"])
    else:
        calendar_index = pd.DatetimeIndex(pd.to_datetime(list(calendar))).normalize().sort_values().unique()
    panel = pd.DataFrame({"date": calendar_index}).merge(daily, on="date", how="left")

    for aux_path, prefix in (
        (proxy_daily_path, "proxy_"),
        (novelty_daily_path, "topical_"),
        (ds_daily_path, "ds_"),
    ):
        if aux_path is None:
            continue
        aux_daily = _load_prefixed_daily_numeric(aux_path, prefix=prefix, date_col="seendate")
        panel = panel.merge(aux_daily, on="date", how="left")

    numeric_cols = [col for col in panel.columns if col != "date"]
    panel[numeric_cols] = panel[numeric_cols].fillna(0.0)

    derived_cols: dict[str, pd.Series] = {}
    for window in sorted({int(w) for w in rolling_windows if int(w) > 1}):
        for col in ROLLING_SUM_FEATURES:
            if col in panel.columns:
                derived_cols[f"{col}_{window}d_sum"] = panel[col].rolling(window=window, min_periods=1).sum()
        for col in extra_roll_sum_cols:
            if col in panel.columns:
                derived_cols[f"{col}_{window}d_sum"] = panel[col].rolling(window=window, min_periods=1).sum()
        for col in ROLLING_MEAN_FEATURES:
            if col in panel.columns:
                derived_cols[f"{col}_{window}d_ma"] = panel[col].rolling(window=window, min_periods=1).mean()

    if derived_cols:
        panel = pd.concat([panel, pd.DataFrame(derived_cols)], axis=1)

    return panel.sort_values("date").reset_index(drop=True)
