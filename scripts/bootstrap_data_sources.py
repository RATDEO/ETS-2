#!/usr/bin/env python3
"""Bootstrap automatable market data into a separate data root.

This script is intentionally separated from the baseline `Data/` directory so
you can keep:
1) Original training data (frozen)
2) Freshly automated updates (this output)
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import re
import shutil
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import urljoin
import xml.etree.ElementTree as ET

import pandas as pd
import requests
import yfinance as yf


LOG = logging.getLogger("bootstrap_data_sources")

ICAP_BASE = "https://allowancepriceexplorer.icapcarbonaction.com"
ICAP_SYSTEMS_URL = f"{ICAP_BASE}/api/systems"
ICAP_PRICE_DOWNLOAD_URL = f"{ICAP_BASE}/systems/reports/price/download"

EEX_AUCTION_INDEX_URL = "https://public.eex-group.com/eex/eua-auction-report/"
ECB_EURUSD_XML_URL = "https://www.ecb.europa.eu/stats/eurofxref/eurofxref-hist.xml"
FRED_BRENT_CSV_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=DCOILBRENTEU"
STOXX_V2TX_URL = "https://stoxx.com/index/v2tx/"
INVESTING_EUA_HISTORICAL_URL = (
    "https://www.investing.com/commodities/"
    "european-union-allowance-eua-year-futures-historical-data"
)

COAL_SYMBOL = "MTF=F"  # Coal (API2) CIF ARA (ARGUS-McCloskey)
CARBON_INDEX_SYMBOLS = ["GRN", "KCCA", "KEUA", "KRBN", "KSET"]


@dataclass
class SourceStatus:
    name: str
    ok: bool
    path: str
    source_url: str
    rows: int | None = None
    min_date: str | None = None
    max_date: str | None = None
    note: str | None = None


def _to_date_str(value: Any) -> str | None:
    if value is None:
        return None
    try:
        ts = pd.Timestamp(value)
        if pd.isna(ts):
            return None
        return ts.date().isoformat()
    except Exception:
        return None


def _df_date_stats(df: pd.DataFrame, date_col: str) -> tuple[int, str | None, str | None]:
    if df.empty or date_col not in df.columns:
        return 0, None, None
    dates = pd.to_datetime(df[date_col], errors="coerce").dropna()
    if dates.empty:
        return 0, None, None
    return len(df), _to_date_str(dates.min()), _to_date_str(dates.max())


def _write_manifest(out_dir: Path, statuses: list[SourceStatus]) -> None:
    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "data_root": str(out_dir),
        "sources": [asdict(x) for x in statuses],
    }
    (out_dir / "automation_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )


def _request_with_retries(
    session: requests.Session,
    url: str,
    *,
    params: dict[str, str] | None = None,
    timeout: int = 60,
    attempts: int = 4,
) -> requests.Response:
    last_exc: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            resp = session.get(url, params=params, timeout=timeout)
            resp.raise_for_status()
            return resp
        except requests.RequestException as exc:
            last_exc = exc
            if attempt == attempts:
                raise
            sleep_s = min(2 ** (attempt - 1), 8)
            LOG.warning(
                "Request failed (%s/%s): %s | retrying in %ss | %s",
                attempt,
                attempts,
                exc,
                sleep_s,
                url,
            )
            time.sleep(sleep_s)
    if last_exc is not None:
        raise last_exc
    raise RuntimeError(f"Unexpected retry state for URL: {url}")


def _download_text(session: requests.Session, url: str) -> str:
    resp = _request_with_retries(session, url=url)
    return resp.text


def _download_bytes(session: requests.Session, url: str, params: dict[str, str] | None = None) -> bytes:
    resp = _request_with_retries(session, url=url, params=params)
    return resp.content


def _normalize_eua_columns(df: pd.DataFrame) -> pd.DataFrame:
    expected = ["Date", "Price", "Open", "High", "Low", "Vol.", "Change %"]
    rename_map = {col: str(col).strip().strip('"') for col in df.columns}
    df = df.rename(columns=rename_map)

    # Accept minor header variants (e.g. "Change%").
    for col in expected:
        if col in df.columns:
            continue
        target = col.replace(" ", "").lower()
        for candidate in df.columns:
            if str(candidate).replace(" ", "").lower() == target:
                df = df.rename(columns={candidate: col})
                break

    for col in expected:
        if col not in df.columns:
            df[col] = ""

    return df[expected].copy()


def _load_seed_eua_data(seed_data_dir: Path) -> tuple[pd.DataFrame, list[Path]]:
    seed_dir = seed_data_dir / "eua-futures"
    files = sorted(seed_dir.glob("*.csv"))
    if not files:
        return pd.DataFrame(columns=["Date", "Price", "Open", "High", "Low", "Vol.", "Change %"]), []

    frames = []
    for path in files:
        try:
            df = pd.read_csv(path, dtype=str).fillna("")
            df = _normalize_eua_columns(df)
            frames.append(df)
        except Exception as exc:  # noqa: BLE001
            LOG.warning("Failed to read seed EUA file %s: %s", path, exc)

    if not frames:
        return pd.DataFrame(columns=["Date", "Price", "Open", "High", "Low", "Vol.", "Change %"]), files

    merged = pd.concat(frames, ignore_index=True)
    merged["_date"] = pd.to_datetime(merged["Date"], format="%d/%m/%Y", errors="coerce")
    merged = merged.dropna(subset=["_date"]).sort_values("_date").drop_duplicates(subset=["Date"], keep="last")
    merged = merged.drop(columns=["_date"])
    return merged.reset_index(drop=True), files


def _load_existing_eua_snapshot(out_path: Path) -> pd.DataFrame:
    """Load existing automated EUA snapshot so Data_auto can self-seed updates."""
    if not out_path.exists():
        return pd.DataFrame(columns=["Date", "Price", "Open", "High", "Low", "Vol.", "Change %"])
    try:
        df = pd.read_csv(out_path, dtype=str).fillna("")
        return _normalize_eua_columns(df)
    except Exception as exc:  # noqa: BLE001
        LOG.warning("Failed to read existing EUA snapshot %s: %s", out_path, exc)
        return pd.DataFrame(columns=["Date", "Price", "Open", "High", "Low", "Vol.", "Change %"])


def _extract_investing_eua_rows(html: str) -> tuple[pd.DataFrame, str | None]:
    match = re.search(
        r'<script id="__NEXT_DATA__" type="application/json">(.*?)</script>',
        html,
        re.S,
    )
    if not match:
        return pd.DataFrame(), "Could not find __NEXT_DATA__ script on Investing page"

    payload = json.loads(match.group(1))
    state = payload.get("props", {}).get("pageProps", {}).get("state", {})
    store = state.get("historicalDataStore", {})
    rows = store.get("historicalData", {}).get("data", [])
    if not rows:
        return pd.DataFrame(), "Investing page returned no historical rows"

    records = []
    for row in rows:
        dt = pd.to_datetime(
            row.get("rowDateTimestamp") or row.get("rowDate"),
            errors="coerce",
            utc=True,
        )
        if pd.isna(dt):
            continue
        ts = pd.Timestamp(dt).tz_localize(None)

        def fmt_price(raw: Any) -> str:
            try:
                return f"{float(raw):.2f}"
            except Exception:  # noqa: BLE001
                return ""

        volume = str(row.get("volume") or "").strip()
        if not volume:
            try:
                vraw = float(row.get("volumeRaw"))
                if vraw >= 1_000_000:
                    volume = f"{vraw / 1_000_000:.2f}M"
                elif vraw >= 1_000:
                    volume = f"{vraw / 1_000:.2f}K"
                else:
                    volume = f"{int(vraw)}"
            except Exception:  # noqa: BLE001
                volume = ""

        change_pct = ""
        raw_change = row.get("change_precentRaw", row.get("change_precent"))
        try:
            change_pct = f"{float(raw_change):+.2f}%"
        except Exception:  # noqa: BLE001
            pass

        records.append(
            {
                "Date": ts.strftime("%d/%m/%Y"),
                "Price": fmt_price(row.get("last_closeRaw", row.get("last_close"))),
                "Open": fmt_price(row.get("last_openRaw", row.get("last_open"))),
                "High": fmt_price(row.get("last_maxRaw", row.get("last_max"))),
                "Low": fmt_price(row.get("last_minRaw", row.get("last_min"))),
                "Vol.": volume,
                "Change %": change_pct,
            }
        )

    df = pd.DataFrame(records)
    if df.empty:
        return df, "No parseable rows in Investing historicalDataStore payload"

    df["_date"] = pd.to_datetime(df["Date"], format="%d/%m/%Y", errors="coerce")
    df = df.dropna(subset=["_date"]).sort_values("_date").drop_duplicates(subset=["Date"], keep="last")
    df = df.drop(columns=["_date"]).reset_index(drop=True)

    date_range = store.get("dateRange", {})
    note = None
    if isinstance(date_range, dict) and date_range.get("startDate") and date_range.get("endDate"):
        note = (
            "Incremental window from Investing page payload: "
            f"{date_range.get('startDate')} -> {date_range.get('endDate')}"
        )
    return df, note


def _download_eua_futures(session: requests.Session, out_root: Path, seed_data_dir: Path) -> SourceStatus:
    out_dir = out_root / "eua-futures"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "European Union Allowance (EUA) Yearly Futures Historical Data AUTO.csv"

    seed_df, seed_files = _load_seed_eua_data(seed_data_dir)
    existing_df = _load_existing_eua_snapshot(out_path)

    seed_frames = [df for df in (seed_df, existing_df) if not df.empty]
    if seed_frames:
        seed_df = pd.concat(seed_frames, ignore_index=True)
        seed_df["_date"] = pd.to_datetime(seed_df["Date"], format="%d/%m/%Y", errors="coerce")
        seed_df = (
            seed_df.dropna(subset=["_date"])
            .sort_values("_date")
            .drop_duplicates(subset=["Date"], keep="last")
            .drop(columns=["_date"])
            .reset_index(drop=True)
        )

    investing_note = None
    try:
        html = _download_text(session, INVESTING_EUA_HISTORICAL_URL)
        update_df, investing_note = _extract_investing_eua_rows(html)
    except Exception as exc:  # noqa: BLE001
        update_df = pd.DataFrame(columns=["Date", "Price", "Open", "High", "Low", "Vol.", "Change %"])
        investing_note = f"Incremental pull failed: {exc}"

    if seed_df.empty and update_df.empty:
        return SourceStatus(
            name="eua_futures",
            ok=False,
            path=str(out_path),
            source_url=INVESTING_EUA_HISTORICAL_URL,
            note=(
                f"No seed data found in {seed_data_dir / 'eua-futures'} and no "
                "incremental data from Investing."
            ),
        )

    merged = pd.concat([seed_df, update_df], ignore_index=True)
    merged = _normalize_eua_columns(merged).fillna("")
    merged["_date"] = pd.to_datetime(merged["Date"], format="%d/%m/%Y", errors="coerce")
    merged = merged.dropna(subset=["_date"]).sort_values("_date").drop_duplicates(subset=["Date"], keep="last")
    merged = merged.sort_values("_date", ascending=False).drop(columns=["_date"]).reset_index(drop=True)

    merged.to_csv(out_path, index=False, quoting=csv.QUOTE_ALL)

    stats_df = merged.copy()
    stats_df["date"] = pd.to_datetime(stats_df["Date"], format="%d/%m/%Y", errors="coerce")
    rows, dmin, dmax = _df_date_stats(stats_df, "date")

    notes = []
    if seed_files:
        notes.append(f"Seeded from {len(seed_files)} baseline file(s).")
    if not existing_df.empty:
        notes.append(f"Seeded from existing output snapshot ({len(existing_df)} rows).")
    if not seed_files and existing_df.empty:
        notes.append("No baseline seed files found.")
    notes.append(f"Incremental rows from Investing: {len(update_df)}.")
    if investing_note:
        notes.append(investing_note)

    return SourceStatus(
        name="eua_futures",
        ok=True,
        path=str(out_path),
        source_url=INVESTING_EUA_HISTORICAL_URL,
        rows=rows,
        min_date=dmin,
        max_date=dmax,
        note=" ".join(notes),
    )


def _icap_pick_system_ids(systems: list[dict[str, Any]]) -> tuple[int | None, int | None]:
    legacy_id = None
    modern_id = None
    for system in systems:
        name = str(system.get("name", ""))
        sid = system.get("id")
        if not isinstance(sid, int):
            continue
        lname = name.lower()
        if "european union emissions trading system (until 2018)" in lname:
            legacy_id = sid
        if "european union emissions trading system (from 2019, download)" in lname:
            modern_id = sid
    # Fallback in case "download" label changes.
    if modern_id is None:
        for system in systems:
            name = str(system.get("name", "")).lower()
            sid = system.get("id")
            if isinstance(sid, int) and "european union emissions trading system (from 2019)" in name:
                modern_id = sid
                break
    return legacy_id, modern_id


def _save_icap_csv(
    session: requests.Session,
    out_dir: Path,
    system_id: int,
    start_date: datetime,
    end_date: datetime,
) -> SourceStatus:
    payload = {
        "systemIds": str(system_id),
        "startDate": str(int(start_date.timestamp() * 1000)),
        "endDate": str(int(end_date.timestamp() * 1000)),
    }
    raw = _download_bytes(session, ICAP_PRICE_DOWNLOAD_URL, params=payload)

    tmp_path = out_dir / f"icap-system-{system_id}.csv"
    tmp_path.write_bytes(raw)

    parsed = pd.read_csv(tmp_path, skiprows=1)
    parsed.columns = [str(c).strip() for c in parsed.columns]
    if "Date" not in parsed.columns:
        raise ValueError(f"Unexpected ICAP CSV schema for system {system_id}")

    parsed["Date"] = pd.to_datetime(parsed["Date"], errors="coerce")
    parsed = parsed.dropna(subset=["Date"]).sort_values("Date")
    if parsed.empty:
        raise ValueError(f"ICAP dataset {system_id} returned no valid rows")

    min_date = parsed["Date"].min().date().isoformat()
    max_date = parsed["Date"].max().date().isoformat()
    final_name = f"icap-graph-price-data-{min_date}-{max_date}.csv"
    final_path = out_dir / final_name
    tmp_path.replace(final_path)

    rows, dmin, dmax = _df_date_stats(parsed, "Date")
    return SourceStatus(
        name=f"icap_system_{system_id}",
        ok=True,
        path=str(final_path),
        source_url=ICAP_PRICE_DOWNLOAD_URL,
        rows=rows,
        min_date=dmin,
        max_date=dmax,
    )


def _download_icap(session: requests.Session, out_root: Path, end_date: datetime) -> list[SourceStatus]:
    icap_dir = out_root / "icap-allowance-price-explorer-secondary-market"
    icap_dir.mkdir(parents=True, exist_ok=True)

    systems = session.get(ICAP_SYSTEMS_URL, timeout=60).json()
    legacy_id, modern_id = _icap_pick_system_ids(systems)

    statuses: list[SourceStatus] = []
    if legacy_id is None or modern_id is None:
        statuses.append(
            SourceStatus(
                name="icap",
                ok=False,
                path=str(icap_dir),
                source_url=ICAP_SYSTEMS_URL,
                note=f"Could not find expected EU ETS system ids (legacy={legacy_id}, modern={modern_id})",
            )
        )
        return statuses

    statuses.append(
        _save_icap_csv(
            session=session,
            out_dir=icap_dir,
            system_id=legacy_id,
            start_date=datetime(2005, 1, 1),
            end_date=datetime(2018, 12, 31),
        )
    )
    statuses.append(
        _save_icap_csv(
            session=session,
            out_dir=icap_dir,
            system_id=modern_id,
            start_date=datetime(2019, 1, 1),
            end_date=end_date,
        )
    )
    return statuses


def _download_auctions(
    session: requests.Session,
    out_root: Path,
    seed_data_dir: Path,
) -> list[SourceStatus]:
    out_dir = out_root / "emission-spot-primary-market-auction"
    out_dir.mkdir(parents=True, exist_ok=True)

    statuses: list[SourceStatus] = []

    # Seed legacy CSV snapshots so parsing works without optional Excel engines.
    seed_dir = seed_data_dir / "emission-spot-primary-market-auction"
    seed_files = sorted(seed_dir.glob("emission-spot-primary-market-auction-report-*.csv"))
    for seed_file in seed_files:
        out_path = out_dir / seed_file.name
        shutil.copy2(seed_file, out_path)
        statuses.append(
            SourceStatus(
                name=f"eex_auction_seed_csv_{seed_file.name}",
                ok=True,
                path=str(out_path),
                source_url=str(seed_file),
                note="Copied from baseline seed directory.",
            )
        )

    html = _download_text(session, EEX_AUCTION_INDEX_URL)
    matches = sorted(
        set(
            re.findall(
                r'href="(emission-spot-primary-market-auction-report-\d{4}-data\.(?:xls|xlsx))"',
                html,
            )
        )
    )

    if not matches:
        statuses.append(
            SourceStatus(
                name="eex_auctions",
                ok=False,
                path=str(out_dir),
                source_url=EEX_AUCTION_INDEX_URL,
                note="No auction report files discovered in EEX index",
            )
        )
        return statuses

    for rel in matches:
        url = urljoin(EEX_AUCTION_INDEX_URL, rel)
        out_path = out_dir / rel
        out_path.write_bytes(_download_bytes(session, url))
        statuses.append(
            SourceStatus(
                name=f"eex_auction_file_{rel}",
                ok=True,
                path=str(out_path),
                source_url=url,
                note="Raw source file (Excel).",
            )
        )
    return statuses


def _download_fx(session: requests.Session, out_root: Path) -> SourceStatus:
    fx_path = out_root / "usd.xml"
    fx_path.write_bytes(_download_bytes(session, ECB_EURUSD_XML_URL))

    # Parse coverage quickly for manifest.
    tree = ET.parse(fx_path)
    root = tree.getroot()
    rows = []
    for cube_day in root.iter():
        if not cube_day.tag.endswith("Cube"):
            continue
        day = cube_day.attrib.get("time")
        if not day:
            continue
        for item in list(cube_day):
            if item.attrib.get("currency") == "USD":
                rows.append(day)
                break
    dates = pd.to_datetime(pd.Series(rows), errors="coerce").dropna()
    return SourceStatus(
        name="ecb_eurusd",
        ok=True,
        path=str(fx_path),
        source_url=ECB_EURUSD_XML_URL,
        rows=len(dates),
        min_date=_to_date_str(dates.min() if not dates.empty else None),
        max_date=_to_date_str(dates.max() if not dates.empty else None),
    )


def _download_brent(session: requests.Session, out_root: Path) -> SourceStatus:
    out_dir = out_root / "energy-benchmarks"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "eu-brent-spot-usd.csv"

    text = _download_text(session, FRED_BRENT_CSV_URL)
    df = pd.read_csv(pd.io.common.StringIO(text))
    df = df.rename(columns={"observation_date": "date", "DCOILBRENTEU": "price_usd"})
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["price_usd"] = pd.to_numeric(df["price_usd"], errors="coerce")
    df = df.dropna(subset=["date", "price_usd"]).sort_values("date")

    with out_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["Back to Contents", "Data 1: Europe Brent Spot Price FOB (Dollars per Barrel)"])
        writer.writerow(["Sourcekey", "RBRTE"])
        writer.writerow(["Date", "Europe Brent Spot Price FOB (Dollars per Barrel)"])
        for row in df.itertuples(index=False):
            writer.writerow([pd.Timestamp(row.date).strftime("%b %d, %Y"), f"{float(row.price_usd):.2f}"])

    rows, dmin, dmax = _df_date_stats(df, "date")
    return SourceStatus(
        name="fred_brent",
        ok=True,
        path=str(out_path),
        source_url=FRED_BRENT_CSV_URL,
        rows=rows,
        min_date=dmin,
        max_date=dmax,
    )


def _flatten_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        # yfinance may return columns like ("Close", "TICKER")
        df.columns = df.columns.get_level_values(0)
    return df


def _download_yf_history(symbol: str, period: str = "max") -> pd.DataFrame:
    df = yf.download(symbol, period=period, interval="1d", progress=False, auto_adjust=False)
    if df is None or df.empty:
        return pd.DataFrame()
    df = _flatten_ohlcv(df)
    df = df.reset_index().rename(columns={"Date": "date"})
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df.dropna(subset=["date"]).sort_values("date")


def _download_coal(out_root: Path) -> SourceStatus:
    out_dir = out_root / "energy-benchmarks"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "rotterdam-coal-futures-usd.csv"

    df = _download_yf_history(COAL_SYMBOL)
    if df.empty:
        return SourceStatus(
            name="coal_api2_ara",
            ok=False,
            path=str(out_path),
            source_url=f"https://finance.yahoo.com/quote/{COAL_SYMBOL}",
            note="No data returned by yfinance for MTF=F",
        )

    df = df.rename(
        columns={
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
        }
    )
    df["change_pct"] = df["close"].pct_change() * 100.0
    df = df.sort_values("date", ascending=False)

    out_df = pd.DataFrame(
        {
            "Date": df["date"].dt.strftime("%d/%m/%Y"),
            "Price": df["close"].map(lambda x: f"{float(x):.2f}" if pd.notna(x) else ""),
            "Open": df["open"].map(lambda x: f"{float(x):.2f}" if pd.notna(x) else ""),
            "High": df["high"].map(lambda x: f"{float(x):.2f}" if pd.notna(x) else ""),
            "Low": df["low"].map(lambda x: f"{float(x):.2f}" if pd.notna(x) else ""),
            "Vol.": df["volume"].map(
                lambda x: "" if pd.isna(x) else f"{float(x) / 1000.0:.2f}K"
            ),
            "Change %": df["change_pct"].map(
                lambda x: "" if pd.isna(x) else f"{float(x):+.2f}%"
            ),
        }
    )
    out_df.to_csv(out_path, index=False, quoting=csv.QUOTE_ALL)

    rows, dmin, dmax = _df_date_stats(df, "date")
    return SourceStatus(
        name="coal_api2_ara",
        ok=True,
        path=str(out_path),
        source_url=f"https://finance.yahoo.com/quote/{COAL_SYMBOL}",
        rows=rows,
        min_date=dmin,
        max_date=dmax,
        note="Symbol MTF=F (Coal API2 CIF ARA).",
    )


def _write_index_history_csv(path: Path, ticker: str, df: pd.DataFrame) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["Price", "Close", "High", "Low", "Open", "Volume"])
        writer.writerow(["Ticker", ticker, ticker, ticker, ticker, ticker])
        writer.writerow(["Date", "", "", "", "", ""])
        for row in df.itertuples(index=False):
            writer.writerow(
                [
                    pd.Timestamp(row.date).strftime("%Y-%m-%d"),
                    f"{float(row.Close):.10f}" if pd.notna(row.Close) else "",
                    f"{float(row.High):.10f}" if pd.notna(row.High) else "",
                    f"{float(row.Low):.10f}" if pd.notna(row.Low) else "",
                    f"{float(row.Open):.10f}" if pd.notna(row.Open) else "",
                    int(row.Volume) if pd.notna(row.Volume) else "",
                ]
            )


def _download_carbon_indices(out_root: Path) -> list[SourceStatus]:
    out_dir = out_root / "carbon-market-indices"
    out_dir.mkdir(parents=True, exist_ok=True)

    statuses: list[SourceStatus] = []
    for ticker in CARBON_INDEX_SYMBOLS:
        file_path = out_dir / f"{ticker}_history.csv"
        df = _download_yf_history(ticker)
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
        _write_index_history_csv(file_path, ticker, df)
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


def _extract_stoxx_chart_data(html: str) -> list[list[float]]:
    match = re.search(r"window\.chart_data\s*=\s*(\[\[.*?\]\]);", html, re.S)
    if not match:
        return []
    raw = match.group(1)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return []


def _download_vstoxx(session: requests.Session, out_root: Path) -> SourceStatus:
    out_dir = out_root / "volatility-proxy"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "vstoxx-index.txt"

    html = _download_text(session, STOXX_V2TX_URL)
    points = _extract_stoxx_chart_data(html)
    if not points:
        return SourceStatus(
            name="vstoxx",
            ok=False,
            path=str(out_path),
            source_url=STOXX_V2TX_URL,
            note="Could not parse window.chart_data from STOXX page",
        )

    df = pd.DataFrame(points, columns=["ts_ms", "vstoxx"])
    df["date"] = pd.to_datetime(df["ts_ms"], unit="ms", errors="coerce").dt.normalize()
    df["vstoxx"] = pd.to_numeric(df["vstoxx"], errors="coerce")
    df = df.dropna(subset=["date", "vstoxx"]).sort_values("date").drop_duplicates(subset=["date"], keep="last")

    with out_path.open("w", encoding="utf-8", newline="") as handle:
        handle.write("Date;Symbol;Indexvalue\n")
        for row in df.itertuples(index=False):
            handle.write(f"{pd.Timestamp(row.date).strftime('%d.%m.%Y')};V2TX;{float(row.vstoxx):.4f}\n")

    rows, dmin, dmax = _df_date_stats(df, "date")
    return SourceStatus(
        name="vstoxx",
        ok=True,
        path=str(out_path),
        source_url=STOXX_V2TX_URL,
        rows=rows,
        min_date=dmin,
        max_date=dmax,
        note="Parsed from embedded window.chart_data on STOXX index page.",
    )


def _download_news(
    out_root: Path,
    news_start: str,
    news_end: str,
    window_days: int,
    max_records: int,
) -> SourceStatus:
    # Lazy import to avoid forcing news dependencies if not needed.
    from src.news.gdelt import fetch_gdelt_headlines, default_gdelt_query

    out_dir = out_root / "news"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "headlines_raw.csv"

    articles = fetch_gdelt_headlines(
        start_date=news_start,
        end_date=news_end,
        query=default_gdelt_query(),
        window_days=window_days,
        max_records=max_records,
        sleep_seconds=0.25,
        language_filter="English",
    )
    if not articles:
        return SourceStatus(
            name="gdelt_news",
            ok=False,
            path=str(out_path),
            source_url="https://api.gdeltproject.org/api/v2/doc/doc",
            note=f"No headlines returned between {news_start} and {news_end}",
        )

    df = pd.DataFrame(articles)
    if "seendate" in df.columns:
        df["seendate"] = pd.to_datetime(df["seendate"], errors="coerce")
    df.to_csv(out_path, index=False)
    rows, dmin, dmax = _df_date_stats(df, "seendate")
    return SourceStatus(
        name="gdelt_news",
        ok=True,
        path=str(out_path),
        source_url="https://api.gdeltproject.org/api/v2/doc/doc",
        rows=rows,
        min_date=dmin,
        max_date=dmax,
    )


def _parse_args() -> argparse.Namespace:
    today = datetime.now().date().isoformat()
    default_news_start = (datetime.now() - timedelta(days=90)).date().isoformat()

    parser = argparse.ArgumentParser(
        description="Download automatable EU ETS project data into a separate directory."
    )
    parser.add_argument(
        "--output-dir",
        default="Data_auto",
        help="Destination data root (default: Data_auto)",
    )
    parser.add_argument(
        "--seed-data-dir",
        default="Data",
        help=(
            "Baseline data directory used to seed datasets that only expose "
            "incremental updates (default: Data)"
        ),
    )
    parser.add_argument(
        "--end-date",
        default=today,
        help="End date (YYYY-MM-DD) for ICAP and related pulls",
    )
    parser.add_argument(
        "--include-news",
        action="store_true",
        help="Also fetch GDELT headlines into <output-dir>/news",
    )
    parser.add_argument("--news-start", default=default_news_start, help="News start date (YYYY-MM-DD)")
    parser.add_argument("--news-end", default=today, help="News end date (YYYY-MM-DD)")
    parser.add_argument("--news-window-days", type=int, default=1)
    parser.add_argument("--news-max-records", type=int, default=250)
    return parser.parse_args()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    args = _parse_args()

    out_root = Path(args.output_dir).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    seed_data_dir = Path(args.seed_data_dir).resolve()

    end_date = datetime.fromisoformat(args.end_date)
    statuses: list[SourceStatus] = []

    session = requests.Session()
    session.headers.update({"User-Agent": "ETS-Data-Bootstrap/1.0"})

    def run_step(name: str, fn):
        LOG.info("Running step: %s", name)
        try:
            result = fn()
            if isinstance(result, Iterable) and not isinstance(result, (str, bytes, dict)):
                statuses.extend(list(result))
            else:
                statuses.append(result)
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

    run_step("eua_futures", lambda: _download_eua_futures(session, out_root, seed_data_dir))
    run_step("icap", lambda: _download_icap(session, out_root, end_date))
    run_step("eex_auctions", lambda: _download_auctions(session, out_root, seed_data_dir))
    run_step("ecb_fx", lambda: _download_fx(session, out_root))
    run_step("brent", lambda: _download_brent(session, out_root))
    run_step("coal_api2_ara", lambda: _download_coal(out_root))
    run_step("carbon_indices", lambda: _download_carbon_indices(out_root))
    run_step("vstoxx", lambda: _download_vstoxx(session, out_root))

    if args.include_news:
        run_step(
            "gdelt_news",
            lambda: _download_news(
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
    LOG.info("Bootstrap complete: %s ok / %s failed", ok, fail)
    LOG.info("Manifest: %s", out_root / "automation_manifest.json")
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
