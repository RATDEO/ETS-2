#!/usr/bin/env python3
"""Bootstrap automatable UK ETS data into a separate data root.

This keeps the existing EU pipeline untouched while producing UK ETS-ready
inputs in the same file formats expected by the current loaders.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import logging
import os
import re
from datetime import datetime, timedelta, timezone
from io import BytesIO, StringIO
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import urljoin

import numpy as np
import pandas as pd
import requests
import yfinance as yf

import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import scripts.bootstrap_data_sources as eu_bootstrap


LOG = logging.getLogger("bootstrap_uk_ets_data_sources")

INVESTING_UKA_HISTORICAL_URL = (
    "https://www.investing.com/commodities/"
    "uk-emissions-allowances-energy-c1-futures-historical-data"
)
INVESTING_HISTORICAL_AJAX_URL = "https://www.investing.com/instruments/HistoricalDataAjax"

ICAP_BASE = "https://allowancepriceexplorer.icapcarbonaction.com"
ICAP_SYSTEMS_URL = f"{ICAP_BASE}/api/systems"
ICAP_PRICE_DOWNLOAD_URL = f"{ICAP_BASE}/systems/reports/price/download"

ICE_BASE = "https://www.ice.com"
ICE_REPORT_API_BASE = f"{ICE_BASE}/marketdata/api/reports"
ICE_UK_AUCTION_REPORT_ID = 278

ONS_BASE = "https://www.ons.gov.uk"
ONS_GAS_DATASET_PAGE = (
    f"{ONS_BASE}/economy/economicoutputandproductivity/output/datasets/"
    "systemaveragepricesapofgas"
)
ONS_POWER_DATASET_PAGE = (
    f"{ONS_BASE}/economy/economicoutputandproductivity/output/datasets/"
    "systempriceofelectricity"
)
OPEN_METEO_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"

UK_WEATHER_LOCATIONS = [
    {"name": "london", "latitude": 51.5074, "longitude": -0.1278},
    {"name": "birmingham", "latitude": 52.4862, "longitude": -1.8904},
    {"name": "manchester", "latitude": 53.4808, "longitude": -2.2426},
    {"name": "leeds", "latitude": 53.8008, "longitude": -1.5491},
    {"name": "glasgow", "latitude": 55.8642, "longitude": -4.2518},
]

# Fallback volatility symbols: UK FTSE volatility first, then generic VIX.
VOL_SYMBOL_CANDIDATES = ["^VFTSE", "^VIX"]

UK_NEWS_KEYWORDS = [
    "UK ETS",
    "UK emissions trading scheme",
    "UK emissions allowance",
    "UKA futures",
    "UK carbon market",
    "UK carbon price",
    "UK carbon auction",
]

CARBON_INDEX_SYMBOLS = ["GRN", "KCCA", "KEUA", "KRBN"]

SourceStatus = eu_bootstrap.SourceStatus


def _to_date_str(value: Any) -> str | None:
    return eu_bootstrap._to_date_str(value)


def _df_date_stats(df: pd.DataFrame, date_col: str) -> tuple[int, str | None, str | None]:
    return eu_bootstrap._df_date_stats(df, date_col)


def _request_with_retries(
    session: requests.Session,
    url: str,
    *,
    params: dict[str, str] | None = None,
    timeout: int = 60,
    attempts: int = 4,
) -> requests.Response:
    return eu_bootstrap._request_with_retries(
        session=session,
        url=url,
        params=params,
        timeout=timeout,
        attempts=attempts,
    )


def _download_text(session: requests.Session, url: str) -> str:
    return eu_bootstrap._download_text(session, url)


def _download_bytes(
    session: requests.Session,
    url: str,
    params: dict[str, str] | None = None,
) -> bytes:
    return eu_bootstrap._download_bytes(session, url, params=params)


def _discover_ons_workbook_urls(session: requests.Session, dataset_page_url: str) -> list[str]:
    page_html = _download_text(session, dataset_page_url)
    candidates: list[str] = []
    for pattern in [
        r'href="([^"]+\.xlsx)"',
        r'"contentUrl"\s*:\s*"([^"]+\.xlsx)"',
    ]:
        candidates.extend(html.unescape(x) for x in re.findall(pattern, page_html))

    normalized: list[str] = []
    seen: set[str] = set()
    for href in candidates:
        full_url = urljoin(ONS_BASE, href)
        if full_url in seen:
            continue
        seen.add(full_url)
        normalized.append(full_url)

    # In practice the dated workbook links are more reliable than the advertised
    # /current/ link, which sometimes 404s on ONS.
    ordered = [x for x in normalized if "/current/" not in x] + [x for x in normalized if "/current/" in x]
    if not ordered:
        raise ValueError(f"Could not discover ONS workbook URL from {dataset_page_url}")
    return ordered


def _load_ons_daily_sheet(
    workbook_bytes: bytes,
    *,
    sheet_name: str,
    value_col: str,
    rolling_col: str,
) -> pd.DataFrame:
    raw = pd.read_excel(BytesIO(workbook_bytes), sheet_name=sheet_name, header=None, engine="openpyxl")
    header_idx: int | None = None
    rolling_expected = rolling_col.strip().lower()
    for idx in range(min(len(raw), 20)):
        values = {str(x).strip().lower() for x in raw.iloc[idx].tolist() if pd.notna(x)}
        if "date" in values and value_col.strip().lower() in values:
            header_idx = idx
            if rolling_expected in values:
                break
    if header_idx is None:
        raise ValueError(f"Unexpected ONS sheet schema for {sheet_name}: {raw.head(8).to_dict(orient='records')}")

    df = pd.read_excel(BytesIO(workbook_bytes), sheet_name=sheet_name, header=header_idx, engine="openpyxl")
    columns = {str(c).strip(): str(c).strip() for c in df.columns}
    df = df.rename(columns=columns)
    if "Date" not in df.columns or value_col not in df.columns:
        raise ValueError(f"Unexpected ONS sheet schema for {sheet_name}: {list(df.columns)}")

    out = pd.DataFrame(
        {
            "date": pd.to_datetime(df["Date"], errors="coerce"),
            "price_native": pd.to_numeric(df[value_col], errors="coerce"),
            "rolling_avg_7d": pd.to_numeric(df.get(rolling_col), errors="coerce"),
        }
    )
    out = out.dropna(subset=["date", "price_native"]).sort_values("date").drop_duplicates(subset=["date"], keep="last")
    return out.reset_index(drop=True)


def _download_ons_daily_series(
    session: requests.Session,
    *,
    dataset_page_url: str,
    out_root: Path,
    out_filename: str,
    sheet_name: str,
    value_col: str,
    rolling_col: str,
    status_name: str,
    note: str,
) -> SourceStatus:
    out_dir = out_root / "energy-benchmarks"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / out_filename

    candidate_urls = _discover_ons_workbook_urls(session, dataset_page_url)
    last_exc: Exception | None = None
    workbook_url: str | None = None
    parsed: pd.DataFrame | None = None
    for candidate_url in candidate_urls:
        try:
            workbook = _download_bytes(session, candidate_url)
            parsed = _load_ons_daily_sheet(
                workbook,
                sheet_name=sheet_name,
                value_col=value_col,
                rolling_col=rolling_col,
            )
            workbook_url = candidate_url
            break
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            LOG.warning("ONS workbook candidate failed for %s: %s", candidate_url, exc)
            continue

    if parsed is None or workbook_url is None:
        raise RuntimeError(f"Failed to download any ONS workbook candidate for {dataset_page_url}") from last_exc

    parsed.to_csv(out_path, index=False)

    rows, dmin, dmax = _df_date_stats(parsed, "date")
    return SourceStatus(
        name=status_name,
        ok=True,
        path=str(out_path),
        source_url=workbook_url,
        rows=rows,
        min_date=dmin,
        max_date=dmax,
        note=note,
    )


def _normalize_price_rows(df: pd.DataFrame) -> pd.DataFrame:
    return eu_bootstrap._normalize_eua_columns(df)


def _download_uk_gas_ons(session: requests.Session, out_root: Path) -> SourceStatus:
    return _download_ons_daily_series(
        session,
        dataset_page_url=ONS_GAS_DATASET_PAGE,
        out_root=out_root,
        out_filename="uk-sap-gas-pence-per-kwh.csv",
        sheet_name="1.Daily SAP Gas",
        value_col="SAP actual day",
        rolling_col="SAP seven-day rolling average",
        status_name="uk_sap_gas",
        note="ONS daily SAP gas dataset sourced from National Gas Transmission.",
    )


def _download_uk_power_ons(session: requests.Session, out_root: Path) -> SourceStatus:
    return _download_ons_daily_series(
        session,
        dataset_page_url=ONS_POWER_DATASET_PAGE,
        out_root=out_root,
        out_filename="uk-system-electricity-price-pence-per-kwh.csv",
        sheet_name="1.Daily SP Electricity",
        value_col="Daily average",
        rolling_col="Seven-day rolling average",
        status_name="uk_system_power",
        note="ONS daily system electricity price dataset sourced from Elexon BMRS.",
    )


def _build_weather_feature_pack(payload: list[dict[str, Any]]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for idx, item in enumerate(payload):
        daily = item.get("daily", {})
        loc = UK_WEATHER_LOCATIONS[idx]["name"] if idx < len(UK_WEATHER_LOCATIONS) else f"loc_{idx}"
        frame = pd.DataFrame(
            {
                "date": pd.to_datetime(daily.get("time", []), errors="coerce"),
                f"{loc}_temp_mean_c": pd.to_numeric(daily.get("temperature_2m_mean", []), errors="coerce"),
                f"{loc}_temp_min_c": pd.to_numeric(daily.get("temperature_2m_min", []), errors="coerce"),
                f"{loc}_temp_max_c": pd.to_numeric(daily.get("temperature_2m_max", []), errors="coerce"),
            }
        )
        frame = frame.dropna(subset=["date"]).sort_values("date")
        frames.append(frame)

    if not frames:
        raise ValueError("Open-Meteo returned no daily weather frames")

    merged = frames[0]
    for frame in frames[1:]:
        merged = merged.merge(frame, on="date", how="outer")

    mean_cols = [c for c in merged.columns if c.endswith("_temp_mean_c")]
    min_cols = [c for c in merged.columns if c.endswith("_temp_min_c")]
    max_cols = [c for c in merged.columns if c.endswith("_temp_max_c")]
    merged["uk_temp_mean_c"] = merged[mean_cols].mean(axis=1)
    merged["uk_temp_min_c"] = merged[min_cols].mean(axis=1)
    merged["uk_temp_max_c"] = merged[max_cols].mean(axis=1)

    hdd_city_cols: list[str] = []
    for col in mean_cols:
        hdd_col = col.replace("_temp_mean_c", "_hdd18")
        merged[hdd_col] = np.maximum(18.0 - merged[col], 0.0)
        hdd_city_cols.append(hdd_col)
    merged["uk_hdd18"] = merged[hdd_city_cols].mean(axis=1)
    merged["uk_hdd18_7d_ma"] = merged["uk_hdd18"].rolling(window=7, min_periods=1).mean()
    merged["uk_temp_mean_7d_ma"] = merged["uk_temp_mean_c"].rolling(window=7, min_periods=1).mean()

    keep_cols = [
        "date",
        "uk_temp_mean_c",
        "uk_temp_min_c",
        "uk_temp_max_c",
        "uk_hdd18",
        "uk_hdd18_7d_ma",
        "uk_temp_mean_7d_ma",
    ]
    return merged[keep_cols].dropna(subset=["date"]).sort_values("date").drop_duplicates(subset=["date"], keep="last")


def _download_uk_weather_open_meteo(
    session: requests.Session,
    out_root: Path,
    *,
    start_date: datetime,
    end_date: datetime,
) -> SourceStatus:
    out_dir = out_root / "weather-proxy"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "uk_weather_daily_feature_pack.csv"

    params = {
        "latitude": ",".join(str(x["latitude"]) for x in UK_WEATHER_LOCATIONS),
        "longitude": ",".join(str(x["longitude"]) for x in UK_WEATHER_LOCATIONS),
        "start_date": start_date.date().isoformat(),
        "end_date": end_date.date().isoformat(),
        "daily": "temperature_2m_mean,temperature_2m_min,temperature_2m_max",
        "timezone": "Europe/London",
        "models": "era5_land",
    }
    response = _request_with_retries(session, OPEN_METEO_ARCHIVE_URL, params=params, timeout=60)
    payload = response.json()
    if not isinstance(payload, list):
        raise ValueError("Open-Meteo weather archive response did not return a location list")

    weather_df = _build_weather_feature_pack(payload)
    weather_df.to_csv(out_path, index=False)

    rows, dmin, dmax = _df_date_stats(weather_df, "date")
    return SourceStatus(
        name="uk_weather_open_meteo",
        ok=True,
        path=str(out_path),
        source_url=response.url,
        rows=rows,
        min_date=dmin,
        max_date=dmax,
        note="Equal-weight GB metro weather proxy (London, Birmingham, Manchester, Leeds, Glasgow) from Open-Meteo ERA5-Land.",
    )


def _load_existing_target_snapshot(out_path: Path) -> pd.DataFrame:
    if not out_path.exists():
        return pd.DataFrame(columns=["Date", "Price", "Open", "High", "Low", "Vol.", "Change %"])
    try:
        existing = pd.read_csv(out_path, dtype=str).fillna("")
        return _normalize_price_rows(existing)
    except Exception as exc:  # noqa: BLE001
        LOG.warning("Failed reading existing UKA snapshot %s: %s", out_path, exc)
        return pd.DataFrame(columns=["Date", "Price", "Open", "High", "Low", "Vol.", "Change %"])


def _find_uk_icap_systems(systems: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for item in systems:
        name = str(item.get("name", "")).lower()
        sid = item.get("id")
        if not isinstance(sid, int):
            continue
        if (
            "united kingdom emissions trading system" in name
            or "united kingdom emissions trading scheme" in name
            or "uk ets" in name
        ):
            selected.append(item)

    # Prefer endpoints labeled as downloadable when present.
    preferred = [x for x in selected if "download" in str(x.get("name", "")).lower()]
    if preferred:
        return preferred
    return selected


def _save_icap_system_csv(
    session: requests.Session,
    out_dir: Path,
    system: dict[str, Any],
    start_date: datetime,
    end_date: datetime,
) -> tuple[SourceStatus, Path | None]:
    system_id = int(system["id"])
    payload = {
        "systemIds": str(system_id),
        "startDate": str(int(start_date.timestamp() * 1000)),
        "endDate": str(int(end_date.timestamp() * 1000)),
    }

    raw = _download_bytes(session, ICAP_PRICE_DOWNLOAD_URL, params=payload)
    tmp_path = out_dir / f"icap-uk-system-{system_id}.csv"
    tmp_path.write_bytes(raw)

    parsed = pd.read_csv(tmp_path, skiprows=1)
    parsed.columns = [str(c).strip() for c in parsed.columns]
    if "Date" not in parsed.columns:
        status = SourceStatus(
            name=f"icap_uk_system_{system_id}",
            ok=False,
            path=str(tmp_path),
            source_url=ICAP_PRICE_DOWNLOAD_URL,
            note=f"Unexpected schema for ICAP UK system {system_id}",
        )
        return status, None

    parsed["Date"] = pd.to_datetime(parsed["Date"], errors="coerce")
    parsed = parsed.dropna(subset=["Date"]).sort_values("Date")
    if parsed.empty:
        status = SourceStatus(
            name=f"icap_uk_system_{system_id}",
            ok=False,
            path=str(tmp_path),
            source_url=ICAP_PRICE_DOWNLOAD_URL,
            note=f"No valid rows for ICAP UK system {system_id}",
        )
        return status, None

    min_date = parsed["Date"].min().date().isoformat()
    max_date = parsed["Date"].max().date().isoformat()
    final_name = f"icap-graph-price-data-{min_date}-{max_date}.csv"
    final_path = out_dir / final_name
    tmp_path.replace(final_path)

    rows, dmin, dmax = _df_date_stats(parsed, "Date")
    status = SourceStatus(
        name=f"icap_uk_system_{system_id}",
        ok=True,
        path=str(final_path),
        source_url=ICAP_PRICE_DOWNLOAD_URL,
        rows=rows,
        min_date=dmin,
        max_date=dmax,
        note=str(system.get("name", "")),
    )
    return status, final_path


def _download_icap_uk(
    session: requests.Session,
    out_root: Path,
    start_date: datetime,
    end_date: datetime,
) -> tuple[list[SourceStatus], list[Path]]:
    out_dir = out_root / "icap-allowance-price-explorer-secondary-market"
    out_dir.mkdir(parents=True, exist_ok=True)

    response = _request_with_retries(session, ICAP_SYSTEMS_URL, timeout=60)
    systems = response.json()
    uk_systems = _find_uk_icap_systems(systems if isinstance(systems, list) else [])

    if not uk_systems:
        status = SourceStatus(
            name="icap_uk",
            ok=False,
            path=str(out_dir),
            source_url=ICAP_SYSTEMS_URL,
            note="No UK ETS systems found in ICAP systems list.",
        )
        return [status], []

    statuses: list[SourceStatus] = []
    csv_paths: list[Path] = []
    for system in uk_systems:
        try:
            status, final_path = _save_icap_system_csv(
                session=session,
                out_dir=out_dir,
                system=system,
                start_date=start_date,
                end_date=end_date,
            )
            statuses.append(status)
            if final_path is not None:
                csv_paths.append(final_path)
        except Exception as exc:  # noqa: BLE001
            statuses.append(
                SourceStatus(
                    name=f"icap_uk_system_{system.get('id')}",
                    ok=False,
                    path=str(out_dir),
                    source_url=ICAP_PRICE_DOWNLOAD_URL,
                    note=str(exc),
                )
            )
    return statuses, csv_paths


def _rows_from_icap_secondary(icap_paths: list[Path]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path in icap_paths:
        try:
            df = pd.read_csv(path, skiprows=1)
        except Exception as exc:  # noqa: BLE001
            LOG.warning("Unable to read ICAP file %s: %s", path, exc)
            continue

        df.columns = [str(c).strip() for c in df.columns]
        if "Date" not in df.columns:
            continue
        if "Secondary Market" not in df.columns and "Primary Market" not in df.columns:
            continue

        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        for col in ["Secondary Market", "Primary Market"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        secondary = df.get("Secondary Market")
        primary = df.get("Primary Market")
        close = secondary.combine_first(primary) if secondary is not None else primary
        if close is None:
            continue
        out = pd.DataFrame({"date": df["Date"], "close": close})
        out = out.dropna(subset=["date", "close"]).copy()
        if out.empty:
            continue
        frames.append(out)

    if not frames:
        return pd.DataFrame(columns=["Date", "Price", "Open", "High", "Low", "Vol.", "Change %"])

    merged = pd.concat(frames, ignore_index=True)
    merged = merged.sort_values("date").drop_duplicates(subset=["date"], keep="last")

    rows = pd.DataFrame(
        {
            "Date": merged["date"].dt.strftime("%d/%m/%Y"),
            "Price": merged["close"].map(lambda x: f"{float(x):.2f}"),
            "Open": merged["close"].map(lambda x: f"{float(x):.2f}"),
            "High": merged["close"].map(lambda x: f"{float(x):.2f}"),
            "Low": merged["close"].map(lambda x: f"{float(x):.2f}"),
            "Vol.": "",
            "Change %": "",
        }
    )
    return rows.reset_index(drop=True)


def _download_uka_futures(
    session: requests.Session,
    out_root: Path,
    icap_paths: list[Path],
    start_date: datetime,
    end_date: datetime,
) -> SourceStatus:
    out_dir = out_root / "eua-futures"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "UK Emissions Allowances (UKA) Futures Historical Data AUTO.csv"

    existing_df = _load_existing_target_snapshot(out_path)
    icap_seed_df = _rows_from_icap_secondary(icap_paths)

    investing_note = None
    investing_source = INVESTING_UKA_HISTORICAL_URL
    try:
        html = _download_text(session, INVESTING_UKA_HISTORICAL_URL)
        match = re.search(r'<script id="__NEXT_DATA__" type="application/json">(.*?)</script>', html, re.S)
        if not match:
            raise ValueError("Could not locate __NEXT_DATA__ payload for UKA historical page")
        payload = json.loads(match.group(1))
        state = payload.get("props", {}).get("pageProps", {}).get("state", {})
        identifiers = state.get("pageInfoStore", {}).get("identifiers", {}) or {}
        instrument = state.get("commodityStore", {}).get("instrument", {}) or {}
        instrument_name = (
            instrument.get("name", {}).get("fullName")
            or instrument.get("name", {}).get("shortName")
            or "UK Emissions Allowances Energy c1"
        )
        instrument_id = str(identifiers.get("instrument_id") or instrument.get("base", {}).get("id") or "").strip()
        sml_id = str(identifiers.get("sml") or "").strip()
        if not instrument_id:
            raise ValueError("Could not resolve UKA instrument_id from Investing page state")

        ajax_payload = {
            "curr_id": instrument_id,
            "smlID": sml_id,
            "header": instrument_name,
            "st_date": start_date.strftime("%m/%d/%Y"),
            "end_date": end_date.strftime("%m/%d/%Y"),
            "interval_sec": "Daily",
            "sort_col": "date",
            "sort_ord": "DESC",
            "action": "historical_data",
        }
        ajax_headers = {
            "User-Agent": session.headers.get("User-Agent", "UK-ETS-Data-Bootstrap/1.0"),
            "X-Requested-With": "XMLHttpRequest",
            "Referer": INVESTING_UKA_HISTORICAL_URL,
            "Origin": "https://www.investing.com",
        }
        ajax_resp = session.post(
            INVESTING_HISTORICAL_AJAX_URL,
            data=ajax_payload,
            headers=ajax_headers,
            timeout=60,
        )
        ajax_resp.raise_for_status()

        parsed_tables = pd.read_html(StringIO(ajax_resp.text))
        if not parsed_tables:
            raise ValueError("Investing HistoricalDataAjax returned no tables")
        investing_df = parsed_tables[0].copy()
        expected_cols = ["Date", "Price", "Open", "High", "Low", "Vol.", "Change %"]
        missing_cols = [c for c in expected_cols if c not in investing_df.columns]
        if missing_cols:
            raise ValueError(f"Investing HistoricalDataAjax schema missing columns: {missing_cols}")

        investing_df = investing_df[expected_cols].copy()
        investing_df["Date"] = pd.to_datetime(investing_df["Date"], errors="coerce").dt.strftime("%d/%m/%Y")
        investing_df = investing_df.dropna(subset=["Date"]).reset_index(drop=True)
        investing_df = _normalize_price_rows(investing_df)
        investing_df = investing_df[
            ["Date", "Price", "Open", "High", "Low", "Vol.", "Change %"]
        ]
        investing_source = INVESTING_HISTORICAL_AJAX_URL
        investing_note = (
            "Investing HistoricalDataAjax full pull succeeded "
            f"({start_date.date().isoformat()} -> {end_date.date().isoformat()})."
        )
    except Exception as exc:  # noqa: BLE001
        investing_df = pd.DataFrame(columns=["Date", "Price", "Open", "High", "Low", "Vol.", "Change %"])
        investing_note = f"Investing full pull failed; falling back to page payload parser: {exc}"
        try:
            html = _download_text(session, INVESTING_UKA_HISTORICAL_URL)
            investing_df, fallback_note = eu_bootstrap._extract_investing_eua_rows(html)
            if fallback_note:
                investing_note = f"{investing_note} {fallback_note}"
        except Exception as fallback_exc:  # noqa: BLE001
            investing_note = f"{investing_note} Fallback parser also failed: {fallback_exc}"

    frames = [df for df in (existing_df, icap_seed_df, investing_df) if not df.empty]
    if not frames:
        return SourceStatus(
            name="uka_futures",
            ok=False,
            path=str(out_path),
            source_url=INVESTING_UKA_HISTORICAL_URL,
            note="No rows from ICAP backfill, existing snapshot, or Investing pull.",
        )

    merged = pd.concat(frames, ignore_index=True)
    merged = _normalize_price_rows(merged).fillna("")
    merged["_date"] = pd.to_datetime(merged["Date"], format="%d/%m/%Y", errors="coerce")
    merged = merged.dropna(subset=["_date"]).sort_values("_date").drop_duplicates(subset=["Date"], keep="last")
    merged = merged.sort_values("_date", ascending=False).drop(columns=["_date"]).reset_index(drop=True)
    merged.to_csv(out_path, index=False, quoting=csv.QUOTE_ALL)

    stats_df = merged.copy()
    stats_df["date"] = pd.to_datetime(stats_df["Date"], format="%d/%m/%Y", errors="coerce")
    rows, dmin, dmax = _df_date_stats(stats_df, "date")

    notes = [
        f"Rows from existing output snapshot: {len(existing_df)}.",
        f"Rows from ICAP secondary backfill: {len(icap_seed_df)}.",
        f"Rows from Investing payload: {len(investing_df)}.",
    ]
    if investing_note:
        notes.append(investing_note)

    return SourceStatus(
        name="uka_futures",
        ok=True,
        path=str(out_path),
        source_url=investing_source,
        rows=rows,
        min_date=dmin,
        max_date=dmax,
        note=" ".join(notes),
    )


def _build_uk_auction_proxy_from_icap(
    out_root: Path,
    icap_paths: list[Path],
) -> SourceStatus:
    out_dir = out_root / "emission-spot-primary-market-auction"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "emission-spot-primary-market-auction-report-uk-icap-primary-auto.csv"

    frames: list[pd.DataFrame] = []
    for path in icap_paths:
        try:
            df = pd.read_csv(path, skiprows=1)
        except Exception as exc:  # noqa: BLE001
            LOG.warning("Could not read ICAP file for auction proxy %s: %s", path, exc)
            continue
        df.columns = [str(c).strip() for c in df.columns]
        if "Date" not in df.columns or "Primary Market" not in df.columns:
            continue

        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df["Primary Market"] = pd.to_numeric(df["Primary Market"], errors="coerce")
        out = df[["Date", "Primary Market"]].dropna(subset=["Date", "Primary Market"]).copy()
        if out.empty:
            continue
        frames.append(out)

    if not frames:
        return SourceStatus(
            name="uk_auction_proxy",
            ok=False,
            path=str(out_path),
            source_url=ICAP_PRICE_DOWNLOAD_URL,
            note="No ICAP primary-market rows available to build auction proxy.",
        )

    merged = pd.concat(frames, ignore_index=True)
    merged = merged.sort_values("Date").drop_duplicates(subset=["Date"], keep="last")

    proxy = pd.DataFrame(
        {
            "Date": merged["Date"].dt.strftime("%d.%m.%Y"),
            "Auction": "UK ETS Primary Market (ICAP proxy)",
            "Auction Price EUR/tCO2": merged["Primary Market"].map(lambda x: f"{float(x):.2f}"),
            "Auction Volume tCO2": "",
        }
    )
    proxy.to_csv(out_path, index=False, quoting=csv.QUOTE_MINIMAL)

    stats = merged.rename(columns={"Date": "date"}).copy()
    rows, dmin, dmax = _df_date_stats(stats, "date")
    return SourceStatus(
        name="uk_auction_proxy",
        ok=True,
        path=str(out_path),
        source_url=ICAP_PRICE_DOWNLOAD_URL,
        rows=rows,
        min_date=dmin,
        max_date=dmax,
        note="Derived from ICAP UK Primary Market. Auction volume unavailable in ICAP export.",
    )


def _build_uk_auction_feature_pack_from_icap(
    out_root: Path,
    icap_paths: list[Path],
) -> SourceStatus:
    out_dir = out_root / "auction-proxy-features"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "uk_icap_primary_secondary_feature_pack.csv"

    frames: list[pd.DataFrame] = []
    for path in icap_paths:
        try:
            df = pd.read_csv(path, skiprows=1)
        except Exception as exc:  # noqa: BLE001
            LOG.warning("Could not read ICAP file for feature pack %s: %s", path, exc)
            continue
        df.columns = [str(c).strip() for c in df.columns]
        if "Date" not in df.columns:
            continue

        working = pd.DataFrame({"date": pd.to_datetime(df["Date"], errors="coerce")})
        for source_col, out_col in [
            ("Primary Market", "uk_icap_primary"),
            ("Secondary Market", "uk_icap_secondary"),
        ]:
            if source_col in df.columns:
                working[out_col] = pd.to_numeric(df[source_col], errors="coerce")
            else:
                working[out_col] = pd.NA

        working = working.dropna(subset=["date"])
        if working.empty:
            continue
        frames.append(working)

    if not frames:
        return SourceStatus(
            name="uk_auction_proxy_feature_pack",
            ok=False,
            path=str(out_path),
            source_url=ICAP_PRICE_DOWNLOAD_URL,
            note="No ICAP rows available to build UK auction proxy feature pack.",
        )

    merged = (
        pd.concat(frames, ignore_index=True)
        .sort_values("date")
        .drop_duplicates(subset=["date"], keep="last")
        .reset_index(drop=True)
    )

    calendar = pd.DataFrame(
        {
            "date": pd.date_range(
                start=merged["date"].min(),
                end=merged["date"].max(),
                freq="D",
            )
        }
    )
    daily = calendar.merge(merged, on="date", how="left")

    daily["uk_icap_primary_print_day"] = daily["uk_icap_primary"].notna().astype(int)
    daily["uk_icap_secondary_print_day"] = daily["uk_icap_secondary"].notna().astype(int)

    daily["uk_icap_primary_ffill"] = daily["uk_icap_primary"].ffill()
    daily["uk_icap_secondary_ffill"] = daily["uk_icap_secondary"].ffill(limit=10)

    daily["uk_icap_primary_secondary_spread"] = (
        daily["uk_icap_primary_ffill"] - daily["uk_icap_secondary_ffill"]
    )
    denom = daily["uk_icap_secondary_ffill"].replace(0.0, pd.NA)
    daily["uk_icap_primary_secondary_spread_pct"] = (
        daily["uk_icap_primary_secondary_spread"] / denom
    )

    last_primary_date = daily["date"].where(daily["uk_icap_primary_print_day"] == 1).ffill()
    daily["uk_icap_primary_days_since_print"] = (
        (daily["date"] - last_primary_date).dt.days
    )

    for lag in (1, 5, 20):
        daily[f"uk_icap_primary_lag_{lag}"] = daily["uk_icap_primary_ffill"].shift(lag)
        daily[f"uk_icap_spread_lag_{lag}"] = daily["uk_icap_primary_secondary_spread"].shift(lag)

    spread_roll = daily["uk_icap_primary_secondary_spread"].rolling(window=20, min_periods=5)
    spread_mean = spread_roll.mean()
    spread_std = spread_roll.std().replace(0.0, pd.NA)
    daily["uk_icap_spread_z20"] = (
        (daily["uk_icap_primary_secondary_spread"] - spread_mean) / spread_std
    )

    dow = daily["date"].dt.dayofweek.astype(float)
    daily["uk_icap_dow_sin"] = np.sin(2.0 * np.pi * dow / 7.0)
    daily["uk_icap_dow_cos"] = np.cos(2.0 * np.pi * dow / 7.0)

    daily.to_csv(out_path, index=False)

    rows, dmin, dmax = _df_date_stats(daily, "date")
    return SourceStatus(
        name="uk_auction_proxy_feature_pack",
        ok=True,
        path=str(out_path),
        source_url=ICAP_PRICE_DOWNLOAD_URL,
        rows=rows,
        min_date=dmin,
        max_date=dmax,
        note=(
            "Derived from ICAP UK primary/secondary columns. "
            "No ICE reCAPTCHA dependency."
        ),
    )


def _ice_report_json_or_none(resp: requests.Response) -> dict | None:
    if resp.status_code != 200:
        return None
    content_type = (resp.headers.get("content-type") or "").lower()
    if "application/json" not in content_type:
        return None
    try:
        return resp.json()
    except Exception:  # noqa: BLE001
        return None


def _build_report_payload(criteria: list[dict[str, Any]], start_date: datetime, end_date: datetime) -> list[tuple[str, str]]:
    payload: list[tuple[str, str]] = []
    for criterion in criteria:
        name = str(criterion.get("name") or "").strip()
        ctype = str(criterion.get("type") or "").strip().lower()
        if not name:
            continue

        if ctype == "date-range":
            from_name = str(criterion.get("fromName") or f"{name}.from")
            to_name = str(criterion.get("toName") or f"{name}.to")
            payload.append((from_name, start_date.date().isoformat()))
            payload.append((to_name, end_date.date().isoformat()))
            continue

        if ctype == "multiselect":
            defaults = criterion.get("defaultValues")
            values = defaults if isinstance(defaults, list) else []
            if not values:
                options = criterion.get("values")
                if isinstance(options, list) and options:
                    first = options[0]
                    if isinstance(first, dict) and "value" in first:
                        values = [str(first["value"])]
            for value in values:
                payload.append((name, str(value)))
            continue

        default_value = criterion.get("defaultValue")
        if default_value is None:
            options = criterion.get("values")
            if isinstance(options, list) and options:
                first = options[0]
                if isinstance(first, dict) and "value" in first:
                    default_value = first["value"]
        if default_value is not None:
            payload.append((name, str(default_value)))
    return payload


def _extract_report_rows(results_json: dict[str, Any]) -> list[dict[str, Any]]:
    datasets = results_json.get("datasets")
    if not isinstance(datasets, dict):
        return []
    rows: list[dict[str, Any]] = []
    for dataset in datasets.values():
        if not isinstance(dataset, dict):
            continue
        maybe_rows = dataset.get("rows")
        if isinstance(maybe_rows, list):
            rows.extend([r for r in maybe_rows if isinstance(r, dict)])
    return rows


def _flatten_row(row: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in row.items():
        if isinstance(value, dict):
            # Keep download url where provided by ICE report center rows.
            if "url" in value:
                out[key] = value.get("url")
            elif "label" in value:
                out[key] = value.get("label")
            else:
                out[key] = json.dumps(value)
        else:
            out[key] = value
    return out


def _pick_column(columns: list[str], preferred: list[str], contains_any: list[str]) -> str | None:
    lowered = {c.lower(): c for c in columns}
    for exact in preferred:
        if exact.lower() in lowered:
            return lowered[exact.lower()]
    for col in columns:
        lcol = col.lower()
        if all(token in lcol for token in contains_any):
            return col
    return None


def _to_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(
        series.astype(str).str.replace(",", "", regex=False).str.replace(" ", "", regex=False),
        errors="coerce",
    )


def _parse_ice_auction_rows(rows: list[dict[str, Any]]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()
    flat = pd.DataFrame([_flatten_row(r) for r in rows])
    if flat.empty:
        return flat

    cols = list(flat.columns)
    date_col = _pick_column(
        cols,
        preferred=["Date", "Auction Date", "auctionDate", "reportDate"],
        contains_any=["date"],
    )
    price_col = _pick_column(
        cols,
        preferred=["Auction Clearing Price", "Auction Price", "clearingPrice", "auctionPrice"],
        contains_any=["price"],
    )
    volume_col = _pick_column(
        cols,
        preferred=["Auction Volume", "auctionVolume", "volume"],
        contains_any=["volume"],
    )

    if date_col is None or price_col is None:
        return pd.DataFrame()

    df = pd.DataFrame(
        {
            "date": pd.to_datetime(flat[date_col], errors="coerce"),
            "auction_price_eur": _to_numeric(flat[price_col]),
        }
    )
    if volume_col is not None:
        df["auction_volume"] = _to_numeric(flat[volume_col])
    else:
        df["auction_volume"] = pd.NA

    df = df.dropna(subset=["date", "auction_price_eur"]).sort_values("date")
    df = df.drop_duplicates(subset=["date"], keep="last")
    return df


def _write_auction_csv(df: pd.DataFrame, out_path: Path, auction_label: str) -> None:
    out_df = pd.DataFrame(
        {
            "Date": df["date"].dt.strftime("%d.%m.%Y"),
            "Auction": auction_label,
            "Auction Price EUR/tCO2": df["auction_price_eur"].map(lambda x: f"{float(x):.2f}" if pd.notna(x) else ""),
            "Auction Volume tCO2": df["auction_volume"].map(lambda x: "" if pd.isna(x) else f"{float(x):.0f}"),
        }
    )
    out_df.to_csv(out_path, index=False, quoting=csv.QUOTE_MINIMAL)


def _download_uk_auction_actual_ice(
    session: requests.Session,
    out_root: Path,
    start_date: datetime,
    end_date: datetime,
    recaptcha_token: str | None,
) -> SourceStatus:
    out_dir = out_root / "emission-spot-primary-market-auction"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "emission-spot-primary-market-auction-report-uk-ice-actual-auto.csv"

    meta_url = f"{ICE_REPORT_API_BASE}/metadata/{ICE_UK_AUCTION_REPORT_ID}"
    meta_resp = _request_with_retries(session, meta_url, timeout=60)
    meta_json = _ice_report_json_or_none(meta_resp)
    if not meta_json:
        return SourceStatus(
            name="uk_auction_actual_ice",
            ok=False,
            path=str(out_path),
            source_url=meta_url,
            note=f"ICE metadata request failed: status={meta_resp.status_code}",
        )

    requires_recaptcha = bool(meta_json.get("recaptchaRequired", True))
    if requires_recaptcha:
        if not recaptcha_token:
            return SourceStatus(
                name="uk_auction_actual_ice",
                ok=False,
                path=str(out_path),
                source_url=meta_url,
                note=(
                    "ICE report requires reCAPTCHA. Set ICE_REPORT_RECAPTCHA_TOKEN "
                    "(or --ice-report-recaptcha-token) to fetch actual auction data."
                ),
            )

        validate_url = f"{ICE_REPORT_API_BASE}/recaptcha-validation"
        validate_resp = session.post(
            validate_url,
            headers={"Content-Type": "application/x-www-form-urlencoded", "Accept": "application/json"},
            data={"g-recaptcha-response": recaptcha_token},
            timeout=60,
        )
        if validate_resp.status_code != 200:
            return SourceStatus(
                name="uk_auction_actual_ice",
                ok=False,
                path=str(out_path),
                source_url=validate_url,
                note=f"ICE reCAPTCHA validation failed (status={validate_resp.status_code}).",
            )

    criteria_url = f"{ICE_REPORT_API_BASE}/{ICE_UK_AUCTION_REPORT_ID}/criteria"
    criteria_resp = session.get(criteria_url, headers={"Accept": "application/json"}, timeout=60)
    criteria_json = _ice_report_json_or_none(criteria_resp)
    if not isinstance(criteria_json, list):
        return SourceStatus(
            name="uk_auction_actual_ice",
            ok=False,
            path=str(out_path),
            source_url=criteria_url,
            note=(
                f"ICE criteria request failed (status={criteria_resp.status_code}). "
                "This usually means report-center access is still gated."
            ),
        )

    payload = _build_report_payload(criteria_json, start_date=start_date, end_date=end_date)
    results_url = f"{ICE_REPORT_API_BASE}/{ICE_UK_AUCTION_REPORT_ID}/results"
    results_resp = session.post(
        results_url,
        headers={"Content-Type": "application/x-www-form-urlencoded", "Accept": "application/json"},
        data=payload,
        timeout=60,
    )
    results_json = _ice_report_json_or_none(results_resp)
    if not results_json:
        return SourceStatus(
            name="uk_auction_actual_ice",
            ok=False,
            path=str(out_path),
            source_url=results_url,
            note=f"ICE results request failed (status={results_resp.status_code}).",
        )

    raw_path = out_dir / "uk-ice-auction-report-raw.json"
    raw_path.write_text(json.dumps(results_json, indent=2), encoding="utf-8")

    rows = _extract_report_rows(results_json)
    parsed = _parse_ice_auction_rows(rows)
    if parsed.empty:
        return SourceStatus(
            name="uk_auction_actual_ice",
            ok=False,
            path=str(out_path),
            source_url=results_url,
            note=(
                "ICE results parsed but no recognizable auction date/price columns found. "
                f"Saved raw payload to {raw_path} for schema inspection."
            ),
        )

    _write_auction_csv(parsed, out_path, "UK ETS Primary Auction (ICE actual)")
    stats = parsed.rename(columns={"date": "date"}).copy()
    rows_count, dmin, dmax = _df_date_stats(stats, "date")
    return SourceStatus(
        name="uk_auction_actual_ice",
        ok=True,
        path=str(out_path),
        source_url=results_url,
        rows=rows_count,
        min_date=dmin,
        max_date=dmax,
        note=f"Actual ICE report {ICE_UK_AUCTION_REPORT_ID} feed used.",
    )


def _download_uk_vol_proxy(session: requests.Session, out_root: Path) -> SourceStatus:
    out_dir = out_root / "volatility-proxy"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "vstoxx-index.txt"

    for symbol in VOL_SYMBOL_CANDIDATES:
        try:
            df = yf.download(symbol, period="max", interval="1d", progress=False, auto_adjust=False)
        except Exception as exc:  # noqa: BLE001
            LOG.warning("Vol proxy download failed for %s: %s", symbol, exc)
            continue

        if df is None or df.empty:
            continue
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        if "Close" not in df.columns:
            continue

        frame = df.reset_index().rename(columns={"Date": "date", "Close": "value"})
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
        frame["value"] = pd.to_numeric(frame["value"], errors="coerce")
        frame = frame.dropna(subset=["date", "value"]).sort_values("date")
        frame = frame.drop_duplicates(subset=["date"], keep="last")
        if frame.empty:
            continue

        with out_path.open("w", encoding="utf-8", newline="") as handle:
            handle.write("Date;Symbol;Indexvalue\n")
            for row in frame.itertuples(index=False):
                handle.write(f"{pd.Timestamp(row.date).strftime('%d.%m.%Y')};{symbol};{float(row.value):.4f}\n")

        rows, dmin, dmax = _df_date_stats(frame, "date")
        return SourceStatus(
            name="uk_volatility_proxy",
            ok=True,
            path=str(out_path),
            source_url=f"https://finance.yahoo.com/quote/{symbol}",
            rows=rows,
            min_date=dmin,
            max_date=dmax,
            note=f"Using Yahoo symbol {symbol}.",
        )

    # Fallback to existing EU VSTOXX pull if UK-specific symbol is unavailable.
    fallback = eu_bootstrap._download_vstoxx(session, out_root)
    fallback.name = "uk_volatility_proxy_fallback_vstoxx"
    fallback.note = ((fallback.note or "") + " Fallback used because UK volatility symbols were unavailable.").strip()
    return fallback


def _download_carbon_indices_uk(out_root: Path) -> list[SourceStatus]:
    out_dir = out_root / "carbon-market-indices"
    out_dir.mkdir(parents=True, exist_ok=True)

    statuses: list[SourceStatus] = []
    for ticker in CARBON_INDEX_SYMBOLS:
        file_path = out_dir / f"{ticker}_history.csv"
        df = eu_bootstrap._download_yf_history(ticker)
        if df.empty:
            statuses.append(
                SourceStatus(
                    name=f"index_{ticker}",
                    ok=False,
                    path=str(file_path),
                    source_url=f"https://finance.yahoo.com/quote/{ticker}",
                    note=f"No data returned for {ticker}",
                )
            )
            continue
        keep = [c for c in ["date", "Open", "High", "Low", "Close", "Volume"] if c in df.columns]
        df = df[keep].dropna(subset=["date"]).sort_values("date")
        eu_bootstrap._write_index_history_csv(file_path, ticker, df)
        rows, dmin, dmax = _df_date_stats(df, "date")
        statuses.append(
            SourceStatus(
                name=f"index_{ticker}",
                ok=True,
                path=str(file_path),
                source_url=f"https://finance.yahoo.com/quote/{ticker}",
                rows=rows,
                min_date=dmin,
                max_date=dmax,
            )
        )
    return statuses


def _download_uk_news(
    out_root: Path,
    news_start: str,
    news_end: str,
    window_days: int,
    max_records: int,
) -> SourceStatus:
    from src.news.gdelt import fetch_gdelt_headlines, default_gdelt_query

    out_dir = out_root / "news"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "headlines_raw.csv"

    query = default_gdelt_query(UK_NEWS_KEYWORDS)
    articles = fetch_gdelt_headlines(
        start_date=news_start,
        end_date=news_end,
        query=query,
        window_days=window_days,
        max_records=max_records,
        sleep_seconds=0.25,
        language_filter="English",
    )
    if not articles:
        return SourceStatus(
            name="gdelt_news_uk",
            ok=False,
            path=str(out_path),
            source_url="https://api.gdeltproject.org/api/v2/doc/doc",
            note=f"No UK ETS headlines returned between {news_start} and {news_end}",
        )

    df = pd.DataFrame(articles)
    if "seendate" in df.columns:
        df["seendate"] = pd.to_datetime(df["seendate"], errors="coerce")
    df.to_csv(out_path, index=False)
    rows, dmin, dmax = _df_date_stats(df, "seendate")
    return SourceStatus(
        name="gdelt_news_uk",
        ok=True,
        path=str(out_path),
        source_url="https://api.gdeltproject.org/api/v2/doc/doc",
        rows=rows,
        min_date=dmin,
        max_date=dmax,
        note=f"Query={query}",
    )


def _write_manifest(out_dir: Path, statuses: list[SourceStatus]) -> None:
    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "data_root": str(out_dir),
        "region": "UK_ETS",
        "sources": [s.__dict__ for s in statuses],
    }
    (out_dir / "automation_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )


def _parse_args() -> argparse.Namespace:
    today = datetime.now().date().isoformat()
    default_news_start = (datetime.now() - timedelta(days=90)).date().isoformat()

    parser = argparse.ArgumentParser(
        description="Download automatable UK ETS project data into a separate directory.",
    )
    parser.add_argument(
        "--output-dir",
        default="uk_ets/Data_auto_uk",
        help="Destination data root (default: uk_ets/Data_auto_uk)",
    )
    parser.add_argument(
        "--start-date",
        default="2021-05-19",
        help="Start date (YYYY-MM-DD) for UK ETS pulls (default: UK ETS launch period)",
    )
    parser.add_argument(
        "--end-date",
        default=today,
        help="End date (YYYY-MM-DD) for UK ETS pulls",
    )
    parser.add_argument(
        "--include-news",
        action="store_true",
        help="Also fetch UK ETS-focused GDELT headlines into <output-dir>/news",
    )
    parser.add_argument(
        "--ice-report-recaptcha-token",
        default=os.environ.get("ICE_REPORT_RECAPTCHA_TOKEN", ""),
        help=(
            "Optional reCAPTCHA token for ICE report-center auction feeds. "
            "If omitted, actual ICE auction pull may be gated and fallback is used."
        ),
    )
    parser.add_argument(
        "--require-actual-auctions",
        action="store_true",
        help="Fail bootstrap if actual ICE auction feed cannot be fetched (disable proxy fallback).",
    )
    parser.add_argument("--news-start", default=default_news_start, help="News start date (YYYY-MM-DD)")
    parser.add_argument("--news-end", default=today, help="News end date (YYYY-MM-DD)")
    parser.add_argument("--news-window-days", type=int, default=1)
    parser.add_argument("--news-max-records", type=int, default=250)
    return parser.parse_args()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    args = _parse_args()

    out_root = (PROJECT_ROOT / args.output_dir).resolve() if not Path(args.output_dir).is_absolute() else Path(args.output_dir).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    start_date = datetime.fromisoformat(args.start_date)
    end_date = datetime.fromisoformat(args.end_date)

    statuses: list[SourceStatus] = []
    icap_paths: list[Path] = []

    session = requests.Session()
    session.headers.update({"User-Agent": "UK-ETS-Data-Bootstrap/1.0"})

    def run_step(name: str, fn):
        LOG.info("Running step: %s", name)
        try:
            result = fn()
            if isinstance(result, Iterable) and not isinstance(result, (str, bytes, dict, SourceStatus, Path)):
                statuses.extend(list(result))
            elif isinstance(result, SourceStatus):
                statuses.append(result)
            else:
                statuses.append(
                    SourceStatus(
                        name=name,
                        ok=False,
                        path=str(out_root),
                        source_url="",
                        note=f"Unexpected step return type: {type(result)}",
                    )
                )
        except Exception as exc:  # noqa: BLE001
            LOG.exception("Step failed: %s", name)
            statuses.append(
                SourceStatus(
                    name=name,
                    ok=False,
                    path=str(out_root),
                    source_url="",
                    note=str(exc),
                )
            )

    def run_icap():
        nonlocal icap_paths
        icap_statuses, icap_paths_local = _download_icap_uk(
            session=session,
            out_root=out_root,
            start_date=start_date,
            end_date=end_date,
        )
        icap_paths = icap_paths_local
        return icap_statuses

    run_step("icap_uk", run_icap)
    run_step(
        "uka_futures",
        lambda: _download_uka_futures(
            session,
            out_root,
            icap_paths,
            start_date=start_date,
            end_date=end_date,
        ),
    )
    actual_auction_status = _download_uk_auction_actual_ice(
        session=session,
        out_root=out_root,
        start_date=start_date,
        end_date=end_date,
        recaptcha_token=(args.ice_report_recaptcha_token or "").strip() or None,
    )
    statuses.append(actual_auction_status)
    if not actual_auction_status.ok:
        if args.require_actual_auctions:
            LOG.error("Actual ICE auction feed required but unavailable; failing without proxy fallback.")
        else:
            LOG.warning("Actual ICE auction feed unavailable; using ICAP primary-market proxy fallback.")
            proxy_status = _build_uk_auction_proxy_from_icap(out_root, icap_paths)
            statuses.append(proxy_status)
            if proxy_status.ok:
                actual_auction_status.ok = True
                prefix = (actual_auction_status.note or "").strip()
                actual_auction_status.note = (
                    f"{prefix} Non-fatal in default mode: ICAP proxy fallback was generated."
                ).strip()
    run_step(
        "uk_auction_proxy_feature_pack",
        lambda: _build_uk_auction_feature_pack_from_icap(out_root, icap_paths),
    )
    run_step("ecb_fx", lambda: eu_bootstrap._download_fx(session, out_root))
    run_step("uk_sap_gas", lambda: _download_uk_gas_ons(session, out_root))
    run_step("uk_system_power", lambda: _download_uk_power_ons(session, out_root))
    run_step(
        "uk_weather_open_meteo",
        lambda: _download_uk_weather_open_meteo(
            session,
            out_root,
            start_date=start_date,
            end_date=end_date,
        ),
    )
    run_step("brent", lambda: eu_bootstrap._download_brent(session, out_root))
    run_step("coal_api2_ara", lambda: eu_bootstrap._download_coal(out_root))
    run_step("carbon_indices", lambda: _download_carbon_indices_uk(out_root))
    run_step("uk_volatility_proxy", lambda: _download_uk_vol_proxy(session, out_root))

    if args.include_news:
        run_step(
            "gdelt_news_uk",
            lambda: _download_uk_news(
                out_root=out_root,
                news_start=args.news_start,
                news_end=args.news_end,
                window_days=args.news_window_days,
                max_records=args.news_max_records,
            ),
        )

    _write_manifest(out_root, statuses)

    ok = sum(1 for x in statuses if x.ok)
    fail = sum(1 for x in statuses if not x.ok)
    actual_auction_ok = any(s.name == "uk_auction_actual_ice" and s.ok for s in statuses)
    if args.require_actual_auctions and not actual_auction_ok:
        LOG.error("Bootstrapping marked failed because --require-actual-auctions was set.")
        fail = max(fail, 1)
    LOG.info("UK bootstrap complete: %s ok / %s failed", ok, fail)
    LOG.info("Manifest: %s", out_root / "automation_manifest.json")
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
