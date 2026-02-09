#!/usr/bin/env python3
"""Backfill GDELT GKG (v2) headlines by extracting URLs and fetching titles.

This is intended to extend coverage into earlier years where the Doc API is limited.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import re
import sys
import time
import zipfile
from dataclasses import dataclass
from datetime import datetime
from html.parser import HTMLParser
from urllib.parse import urlparse
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError

MASTERFILELIST_URL = "http://data.gdeltproject.org/gdeltv2/masterfilelist.txt"
USER_AGENT = "ETS-News-GKG/1.0"

THEME_TOKENS = [
    "ENV_CARBON",
    "ENV_CLIMATECHANGE",
    "ENV_GREENHOUSEGAS",
    "ENV_EMISSIONS",
    "ENV_POLLUTION",
    "ENV_CLIMATE",
    "ENV_CLEANENERGY",
    "ENV_RENEWABLEENERGY",
    "ECON_CARBON",
    "ECON_ENERGY",
]

LOCATION_TOKENS = [
    "EUROPE",
    "EUROPEAN UNION",
    "EU",
]

URL_KEYWORDS = [
    "eu-ets",
    "euets",
    "eua",
    "eua-futures",
    "emissions",
    "carbon",
    "allowance",
    "cap-and-trade",
]


class TitleParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self._title: list[str] = []
        self._in_title = False
        self._og_title = None

    @property
    def title(self) -> str | None:
        if self._og_title:
            return self._og_title.strip()
        if self._title:
            return "".join(self._title).strip()
        return None

    def handle_starttag(self, tag, attrs):
        if tag == "title":
            self._in_title = True
        if tag == "meta":
            attrs_dict = dict(attrs)
            if attrs_dict.get("property") == "og:title":
                content = attrs_dict.get("content")
                if content:
                    self._og_title = content

    def handle_endtag(self, tag):
        if tag == "title":
            self._in_title = False

    def handle_data(self, data):
        if self._in_title:
            self._title.append(data)


@dataclass
class GkgRecord:
    url: str
    title: str
    seendate: str
    domain: str
    language: str
    sourcecountry: str


def _fetch_url(url: str, timeout: int = 15) -> bytes:
    req = Request(url, headers={"User-Agent": USER_AGENT})
    with urlopen(req, timeout=timeout) as resp:
        return resp.read(200_000)


def _fetch_title(url: str, retries: int = 2) -> str | None:
    for attempt in range(retries + 1):
        try:
            data = _fetch_url(url)
            parser = TitleParser()
            parser.feed(data.decode("utf-8", errors="ignore"))
            title = parser.title
            if title:
                return title
        except (HTTPError, URLError, TimeoutError):
            time.sleep(0.5 * (attempt + 1))
        except Exception:
            time.sleep(0.5 * (attempt + 1))
    return None


def _title_from_url(url: str) -> str | None:
    parsed = urlparse(url)
    if not parsed.path:
        return None
    slug = parsed.path.rstrip("/").split("/")[-1]
    if not slug or len(slug) < 4:
        return None
    slug = re.sub(r"\\.(html|htm|php|aspx|jsp)$", "", slug, flags=re.IGNORECASE)
    slug = slug.replace("-", " ").replace("_", " ")
    slug = re.sub(r"\\s+", " ", slug).strip()
    if not slug or len(slug) < 6:
        return None
    return slug


def _load_masterfilelist() -> list[tuple[str, str]]:
    req = Request(MASTERFILELIST_URL, headers={"User-Agent": USER_AGENT})
    with urlopen(req, timeout=30) as resp:
        data = resp.read().decode("utf-8")
    entries = []
    for line in data.splitlines():
        parts = line.split()
        if len(parts) < 3:
            continue
        url = parts[-1]
        if not url.endswith(".gkg.csv.zip"):
            continue
        filename = url.rsplit("/", 1)[-1]
        timestamp = filename.split(".")[0]
        if len(timestamp) != 14 or not timestamp.isdigit():
            continue
        entries.append((timestamp, url))
    return entries


def _timestamp_to_datetime(ts: str) -> datetime:
    return datetime.strptime(ts, "%Y%m%d%H%M%S")


def _match_any(value: str, tokens: list[str]) -> bool:
    value = value.upper()
    return any(token in value for token in tokens)


def _url_has_keywords(url: str) -> bool:
    lower = url.lower()
    return any(token in lower for token in URL_KEYWORDS)


def _iter_gkg_rows(zip_bytes: bytes):
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        name = zf.namelist()[0]
        with zf.open(name) as f:
            reader = csv.reader((line.decode("utf-8", errors="ignore") for line in f), delimiter="\t")
            for row in reader:
                yield row


def _parse_gkg_row(row: list[str]) -> tuple[str, str, str, str] | None:
    if len(row) < 11:
        return None
    url = row[4]
    if "http" not in url:
        return None
    themes = row[8] if len(row) > 8 else ""
    locations = row[10] if len(row) > 10 else ""
    seendate = row[1]
    source = row[3] if len(row) > 3 else ""
    return url, themes, locations, seendate


def _write_rows(path: str, rows: list[GkgRecord]):
    fieldnames = ["url", "url_mobile", "title", "seendate", "socialimage", "domain", "language", "sourcecountry"]
    write_header = False
    try:
        with open(path, "r", newline="", encoding="utf-8") as f:
            existing = f.readline()
            if not existing:
                write_header = True
    except FileNotFoundError:
        write_header = True

    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        for rec in rows:
            writer.writerow({
                "url": rec.url,
                "url_mobile": "",
                "title": rec.title,
                "seendate": rec.seendate,
                "socialimage": "",
                "domain": rec.domain,
                "language": rec.language,
                "sourcecountry": rec.sourcecountry,
            })


def main() -> int:
    parser = argparse.ArgumentParser(description="Backfill GDELT GKG headlines with title fetching.")
    parser.add_argument("--start", required=True, help="Start date YYYY-MM-DD")
    parser.add_argument("--end", required=True, help="End date YYYY-MM-DD")
    parser.add_argument("--output", required=True, help="Output CSV path")
    parser.add_argument("--checkpoint", default="data/news/gkg_backfill_checkpoint.json")
    parser.add_argument("--step-hours", type=int, default=6, help="Sample files every N hours")
    parser.add_argument("--max-files", type=int, default=0, help="Optional cap on number of files to process")
    parser.add_argument("--sleep", type=float, default=0.2, help="Sleep between files")
    parser.add_argument("--no-location-filter", action="store_true")
    parser.add_argument("--no-title-fetch", action="store_true")
    parser.add_argument("--fallback-url-title", action="store_true", help="Use URL slug as title when fetch fails")
    parser.add_argument("--allow-url-only", action="store_true", help="Allow URL keyword match without theme match")
    args = parser.parse_args()

    start_dt = datetime.fromisoformat(args.start)
    end_dt = datetime.fromisoformat(args.end)

    entries = _load_masterfilelist()
    filtered = []
    for ts, url in entries:
        dt = _timestamp_to_datetime(ts)
        if not (start_dt <= dt <= end_dt):
            continue
        if args.step_hours > 0:
            if dt.minute != 0 or dt.second != 0:
                continue
            if dt.hour % args.step_hours != 0:
                continue
        filtered.append((ts, url))

    checkpoint = {"index": 0}
    try:
        with open(args.checkpoint, "r", encoding="utf-8") as f:
            checkpoint = json.load(f)
    except FileNotFoundError:
        pass

    start_index = int(checkpoint.get("index", 0))
    processed_files = 0
    seen_urls: set[str] = set()

    for idx in range(start_index, len(filtered)):
        ts, url = filtered[idx]
        if args.max_files and processed_files >= args.max_files:
            break

        try:
            req = Request(url, headers={"User-Agent": USER_AGENT})
            with urlopen(req, timeout=60) as resp:
                zip_bytes = resp.read()
        except Exception:
            checkpoint["index"] = idx + 1
            with open(args.checkpoint, "w", encoding="utf-8") as f:
                json.dump(checkpoint, f)
            time.sleep(args.sleep)
            continue

        rows_out: list[GkgRecord] = []
        for row in _iter_gkg_rows(zip_bytes):
            parsed = _parse_gkg_row(row)
            if not parsed:
                continue
            url_value, themes, locations, seendate = parsed
            if url_value in seen_urls:
                continue

            theme_match = _match_any(themes, THEME_TOKENS)
            location_match = True if args.no_location_filter else _match_any(locations, LOCATION_TOKENS)
            url_match = _url_has_keywords(url_value)

            if args.allow_url_only:
                if not (theme_match or url_match):
                    continue
            else:
                if not (theme_match and (location_match or url_match)):
                    continue

            title = ""
            if not args.no_title_fetch:
                title = _fetch_title(url_value) or ""
            if not title and args.fallback_url_title:
                title = _title_from_url(url_value) or ""
            if not title:
                continue

            parsed_url = urlparse(url_value)
            domain = parsed_url.netloc.lower()

            rows_out.append(GkgRecord(
                url=url_value,
                title=title,
                seendate=seendate,
                domain=domain,
                language="",
                sourcecountry="",
            ))
            seen_urls.add(url_value)

        if rows_out:
            _write_rows(args.output, rows_out)

        processed_files += 1
        checkpoint["index"] = idx + 1
        with open(args.checkpoint, "w", encoding="utf-8") as f:
            json.dump(checkpoint, f)
        time.sleep(args.sleep)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
