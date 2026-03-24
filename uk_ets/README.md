# UK ETS Automated Pipeline (Isolated Subdirectory)

This `uk_ets/` directory is a UK ETS adaptation of the existing project flow.
It does not modify the main EU pipeline.

## What it does

1. Downloads/updates UK ETS-compatible source data into `uk_ets/Data_auto_uk`.
2. Writes files in the same schemas your existing loaders expect.
3. Runs the existing experiment stack against this UK data root.

## Data sources (automated)

- UKA futures target:
  - Investing UKA historical page (`uk-emissions-allowances-energy-c1-futures-historical-data`)
  - Backfilled from ICAP UK ETS secondary market when needed
- UK ETS market series (primary/secondary):
  - ICAP Allowance Price Explorer API
- UK auctions (actual feed path):
  - ICE Report Center API (`report 278`, UK ETS auctions)
  - Note: ICE marks this feed as `recaptchaRequired`, so unattended pulls may be gated
- Auction proxy fallback:
  - Derived from ICAP UK Primary Market column when actual ICE auction feed is unavailable
- Auction-substitute feature pack (fully automatable, no reCAPTCHA):
  - Built from ICAP UK Primary + Secondary columns
  - Includes daily spread, spread %, lagged primary/spread, print-day flags, days-since-print
- FX (EUR/USD):
  - ECB historical XML
- Brent:
  - FRED `DCOILBRENTEU`
- Coal:
  - Yahoo Finance `MTF=F`
- Carbon indices:
  - Yahoo Finance `KEUA`, `KRBN`, `GRN`, `KCCA`
- Volatility proxy:
  - Yahoo `^VFTSE` (fallback `^VIX`)
  - written to `volatility-proxy/vstoxx-index.txt` format for compatibility
- Optional news:
  - GDELT DOC 2.0 (UK ETS-focused keyword query)

## Paths

- UK data root: `uk_ets/Data_auto_uk`
- UK config: `uk_ets/config/uk_ets_default.yaml`
- Bootstrap script: `uk_ets/scripts/bootstrap_uk_ets_data_sources.py`
- Runner script: `uk_ets/scripts/run_uk_automated_pipeline_once.py`
- Auction proxy feature pack: `uk_ets/Data_auto_uk/auction-proxy-features/uk_icap_primary_secondary_feature_pack.csv`

## Quick start

1. Bootstrap UK data only:

```bash
python uk_ets/scripts/bootstrap_uk_ets_data_sources.py \
  --output-dir uk_ets/Data_auto_uk \
  --start-date 2021-05-19 \
  --end-date 2026-03-04
```

2. Bootstrap + run one experiment:

```bash
python uk_ets/scripts/run_uk_automated_pipeline_once.py \
  --config uk_ets/config/uk_ets_default.yaml \
  --data-dir uk_ets/Data_auto_uk \
  --target-mode returns
```

3. Require actual ICE auction feed (fail if proxy would be used):

```bash
python uk_ets/scripts/run_uk_automated_pipeline_once.py \
  --config uk_ets/config/uk_ets_default.yaml \
  --data-dir uk_ets/Data_auto_uk \
  --target-mode returns \
  --require-actual-auctions
```

4. Attempt ICE actual auction fetch with a reCAPTCHA token:

```bash
ICE_REPORT_RECAPTCHA_TOKEN=\"<token>\" \
python uk_ets/scripts/bootstrap_uk_ets_data_sources.py \
  --output-dir uk_ets/Data_auto_uk \
  --start-date 2021-05-19 \
  --end-date 2026-03-04 \
  --require-actual-auctions
```

3. Include news fetch in the same run:

```bash
python uk_ets/scripts/run_uk_automated_pipeline_once.py \
  --include-news \
  --news-start 2025-01-01 \
  --news-end 2026-03-04
```

## Live Forecast Backend

The portfolio/backend deployment path described in
`docs/uk_ets_live_forecast_deployment_foundation.md` now has a dedicated runner:

```bash
python uk_ets/scripts/run_uk_live_forecast_backend.py \
  --mode main \
  --config uk_ets/config/uk_ets_llm_4b_canonical_default.yaml \
  --data-dir uk_ets/Data_auto_uk \
  --archive-path uk_ets/live_forecast/archive/forecast_history.csv \
  --export-dir uk_ets/live_forecast/exports
```

What it does:

1. refreshes UK ETS source data unless `--skip-bootstrap` is passed
2. runs the canonical UK ETS backtest unless `--run-dir` is provided
3. seeds/backfills a CSV archive from the canonical run outputs
4. builds a current live forward forecast snapshot
5. backfills realized actuals when target dates are now known
6. exports website-ready files:
   - `latest.json`
   - `horizon_1d.json`
   - `horizon_5d.json`
   - `horizon_20d.json`
   - `horizon_30d.json`

For a fast local portfolio refresh on a development machine, reuse a completed
canonical seed run and export directly into the portfolio app's `public/`
directory:

```bash
python uk_ets/scripts/run_uk_live_forecast_backend.py \
  --mode main \
  --skip-bootstrap \
  --skip-experiment \
  --config uk_ets/config/uk_ets_llm_4b_canonical_default.yaml \
  --run-dir runs/20260320_183637_a4d80c \
  --data-dir uk_ets/Data_auto_uk \
  --archive-path uk_ets/live_forecast/archive/forecast_history.csv \
  --export-dir /Users/davidwilkinson/Desktop/portfolio/app/public/data/ukets
```

When `--run-dir` and `--skip-experiment` are used together, the runner now uses
that run directory only as the historical seed artifact and still builds the
current live snapshot from the config passed via `--config`.

Optional VPS deployment is available with:

```bash
python uk_ets/scripts/run_uk_live_forecast_backend.py \
  --remote-host your-vps-host \
  --remote-staging-dir /srv/ukets-staging \
  --remote-live-dir /var/www/davidwilkinson.space/data/ukets
```

## Requirements

- Python env with project dependencies in `requirements.txt`
- Network access to upstream data endpoints
- Existing project modules available from repo root (`src/`, `scripts/`)

## Notes and caveats

- Forecast target in this UK branch is still the daily UKA futures close (stored
  in `eua-futures/` for loader compatibility), not the auction-clearing series.
- Auction data is used as an exogenous feature only.
- ICE report-center auction APIs can require reCAPTCHA; when token/access is missing,
  default mode falls back to ICAP proxy and records this in `automation_manifest.json`.
- UK config now enables `include_auction_proxy_pack: true` and uses this ICAP-derived
  feature set as the production-safe auction substitute path.
- UK configs now declare the target explicitly as `UKA_FUTURES` in `GBP`.
- Default UK model features are limited to UKA target microstructure plus UK ICAP /
  auction-derived signals; inherited carbon ETFs and VSTOXX/VIX-style proxies are
  no longer enabled by default.
- `automation_manifest.json` is written each run with per-source status and coverage.
