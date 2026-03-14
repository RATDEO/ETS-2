"""Utilities for summarising structured official event records."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

POLICY_EVENT_TYPES = {
    "policy_rule_change",
    "verified_emissions",
    "maritime_ets",
    "aviation_ets",
    "cbam",
}

CHANNELS_SUPPLY = {"allowance_supply"}


@dataclass(frozen=True)
class OfficialEventThresholds:
    official_event_abs_median: float
    official_supply_abs_median: float


def _resolve_path(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else (ROOT / p)


def _direction_score(value: object) -> float:
    mapping = {
        "bullish": 1.0,
        "positive": 1.0,
        "bearish": -1.0,
        "negative": -1.0,
        "neutral": 0.0,
        "mixed": 0.0,
        "scheduled": 0.0,
        "unknown": 0.0,
    }
    return float(mapping.get(str(value or "").strip().lower(), 0.0))


def load_official_event_records(path: str | Path) -> pd.DataFrame:
    csv_path = _resolve_path(path)
    df = pd.read_csv(csv_path)
    required = {
        "event_date",
        "title",
        "event_type",
        "affected_channel",
        "direction",
        "intensity",
        "confidence",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing required official event columns in {csv_path}: {missing}")

    out = df.copy()
    out["event_date"] = pd.to_datetime(out["event_date"], errors="coerce").dt.normalize()
    out = out[out["event_date"].notna()].copy()
    out["intensity"] = pd.to_numeric(out["intensity"], errors="coerce").fillna(0.0)
    out["confidence"] = pd.to_numeric(out["confidence"], errors="coerce").fillna(1.0).clip(0.0, 1.0)
    out["direction_score"] = out["direction"].map(_direction_score)
    out["weighted_score"] = out["direction_score"] * out["intensity"] * out["confidence"]
    out["event_type"] = out["event_type"].astype(str)
    out["affected_channel"] = out["affected_channel"].astype(str)
    out["title"] = out["title"].astype(str)
    return out.sort_values(["event_date", "title"], kind="stable").reset_index(drop=True)


def build_official_event_summary_map(
    event_path: str | Path,
    origin_dates: Iterable[pd.Timestamp],
    *,
    lookback_days: int = 30,
    max_titles: int = 2,
) -> dict[pd.Timestamp, dict[str, str]]:
    events = load_official_event_records(event_path)
    summary_map: dict[pd.Timestamp, dict[str, str]] = {}
    origin_index = pd.DatetimeIndex(pd.to_datetime(list(origin_dates))).normalize()
    policy_mask = events["event_type"].isin(POLICY_EVENT_TYPES)
    supply_mask = events["affected_channel"].isin(CHANNELS_SUPPLY)

    for origin_date in origin_index:
        cutoff = origin_date - pd.Timedelta(days=int(lookback_days))
        window = events[(events["event_date"] <= origin_date) & (events["event_date"] >= cutoff)].copy()
        if window.empty:
            summary_map[pd.Timestamp(origin_date)] = {
                "official_event_state": "No official EEX/DG CLIMA events in lookback window.",
            }
            continue

        policy_count = int(policy_mask.loc[window.index].sum())
        supply_net = float(window.loc[supply_mask.loc[window.index], "weighted_score"].sum())
        total_net = float(window["weighted_score"].sum())
        bullish = float(window.loc[window["direction_score"] > 0, "weighted_score"].sum())
        bearish = float((-window.loc[window["direction_score"] < 0, "weighted_score"]).sum())
        recent = (
            window.sort_values(["intensity", "event_date"], ascending=[False, False])
            .head(max(1, int(max_titles)))[["event_date", "event_type", "title"]]
        )
        recent_text = "; ".join(
            f"{pd.Timestamp(row.event_date).date()} {row.event_type}: {row.title[:96]}"
            for row in recent.itertuples(index=False)
        )
        summary_map[pd.Timestamp(origin_date)] = {
            "official_event_balance": (
                f"{len(window)} events/{int(lookback_days)}d, net={total_net:.2f}, "
                f"bullish={bullish:.2f}, bearish={bearish:.2f}, policy_count={policy_count}, supply_net={supply_net:.2f}"
            ),
            "official_recent_events": recent_text,
        }
    return summary_map


def build_official_origin_feature_frame(
    origin_dates: Iterable[pd.Timestamp],
    event_path: str | Path,
    *,
    short_window: int = 7,
    long_window: int = 30,
) -> tuple[pd.DataFrame, OfficialEventThresholds]:
    events = load_official_event_records(event_path)
    origin_index = pd.DatetimeIndex(pd.to_datetime(list(origin_dates))).normalize()
    policy_mask = events["event_type"].isin(POLICY_EVENT_TYPES)
    supply_mask = events["affected_channel"].isin(CHANNELS_SUPPLY)

    rows: list[dict[str, float | pd.Timestamp | str]] = []
    for origin_date in origin_index:
        short_cutoff = origin_date - pd.Timedelta(days=int(short_window))
        long_cutoff = origin_date - pd.Timedelta(days=int(long_window))
        win_short = events[(events["event_date"] <= origin_date) & (events["event_date"] >= short_cutoff)]
        win_long = events[(events["event_date"] <= origin_date) & (events["event_date"] >= long_cutoff)]
        supply_long = win_long.loc[supply_mask.loc[win_long.index]]
        policy_long = win_long.loc[policy_mask.loc[win_long.index]]
        rows.append(
            {
                "date": pd.Timestamp(origin_date),
                "official_event_count_7d": float(len(win_short)),
                "official_event_count_30d": float(len(win_long)),
                "official_event_net_7d": float(win_short["weighted_score"].sum()),
                "official_event_net_30d": float(win_long["weighted_score"].sum()),
                "official_supply_net_30d": float(supply_long["weighted_score"].sum()),
                "official_supply_count_30d": float(len(supply_long)),
                "official_policy_count_30d": float(len(policy_long)),
            }
        )

    frame = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    event_threshold = float(max(0.25, frame["official_event_net_30d"].abs().median()))
    supply_threshold = float(max(0.25, frame["official_supply_net_30d"].abs().median()))

    def _alignment(total_net: float, realized_ret: float | None = None) -> str:
        if abs(float(total_net)) < event_threshold:
            return "neutral"
        if realized_ret is None:
            return "aligned" if total_net > 0 else "divergent"
        if abs(float(realized_ret)) < 1e-12:
            return "neutral"
        return "aligned" if np.sign(total_net) == np.sign(realized_ret) else "divergent"

    def _supply_regime(supply_net: float) -> str:
        if abs(float(supply_net)) < supply_threshold:
            return "neutral_supply"
        return "bullish_supply" if supply_net > 0 else "bearish_supply"

    frame["official_supply_regime_30d"] = frame["official_supply_net_30d"].map(_supply_regime)
    thresholds = OfficialEventThresholds(
        official_event_abs_median=event_threshold,
        official_supply_abs_median=supply_threshold,
    )
    return frame, thresholds
