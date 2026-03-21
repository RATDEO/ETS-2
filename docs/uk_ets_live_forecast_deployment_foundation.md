# UK ETS Live Forecast Deployment Foundation

## Purpose

This document defines the deployment architecture, data flow, storage model, automation pattern, and website integration plan for publishing the live UK ETS forecasting output from the `ETS-2` repository to `davidwilkinson.space`.

The goal of this stage is to turn the UK ETS forecasting work into a reliable, autonomous public-facing research feature that:

- updates every day with fresh data
- runs on self-hosted infrastructure
- preserves a trustworthy research archive
- exposes clean chart-ready outputs to the portfolio website
- demonstrates both live forecasting and historical performance

---

## Core Objective

The system should:

1. run the UK ETS forecasting pipeline automatically every day
2. fetch fresh source data
3. generate updated forecasts
4. preserve historical forecast records in an internal archive
5. regenerate public website JSON outputs from that archive
6. deploy those JSON outputs to the VPS hosting `davidwilkinson.space`
7. only replace the live website data if the full run succeeds

This creates a public research display that shows:

- the latest live UK ETS forecast
- horizon-specific forecast comparisons
- historical performance of base TSM vs TSM+LLM against reality

---

## Infrastructure Decision

### Chosen production environment

A **dedicated LXC container on Proxmox** will run the forecasting pipeline.

### Why LXC was chosen

- better suited for autonomous headless operation
- lighter and more efficient than a full VM
- easier to schedule and monitor
- easier to isolate as a single-purpose production worker
- better fit for a daily batch forecasting service

### Not chosen

A macOS VM was considered but rejected for production use. It may still remain useful for development/debugging, but it is not part of the live deployment architecture.

---

## Role of the LXC

The LXC is a **daily forecast worker**, not a website server.

Its responsibilities are:

1. update the repo
2. refresh data sources
3. run the UK ETS forecast pipeline
4. append forecast outputs to the internal archive
5. backfill realized outcomes when they become available
6. regenerate website-facing JSON files
7. sync those files to the VPS
8. log success/failure

The website itself remains hosted separately on the VPS.

---

## High-Level Architecture

```text
Proxmox LXC
  -> pulls ETS-2 repo updates
  -> fetches fresh UK ETS data
  -> runs forecast pipeline
  -> appends archive rows
  -> backfills realized actuals
  -> regenerates public JSON files
  -> rsyncs JSON to VPS staging
  -> promotes staging to live only on full success

VPS (davidwilkinson.space)
  -> serves static JSON files from /data/ukets/
  -> frontend fetches those JSON files
  -> charts and summary cards render from the latest exported data
```

---

## Forecast Presentation Model

The website will present two distinct classes of information:

### 1. Current live forecast

This answers: **what does the model say now?**

It will show:

- latest observed UK ETS price
- latest generated timestamp
- current forecast values for 1, 5, 20, and 30 steps ahead
- both base TSM and LLM+TSM values
- headline emphasis on the LLM+TSM output

### 2. Historical forecast-performance view

This answers: **how did earlier forecasts compare to reality?**

The site will allow the user to toggle between:

- 1 day
- 5 day
- 20 day
- 30 day

For the selected horizon, the graph will show three lines:

- actual realized price
- base TSM forecast made that many days earlier
- LLM+TSM forecast made that many days earlier

This is the core public demonstration of model usefulness.

---

## Forecast Horizon Logic

The primary public projection focus will be:

- **20-day horizon**

Additional toggle views will also be available for:

- 1-day horizon
- 5-day horizon
- 30-day horizon

Each horizon view must compare forecasts made at that horizon against the realized future value.

Example:

If the user selects **20d**, then for each target date on the chart:

- `actual` = the real UK ETS value on that target date
- `base_tsm_forecast` = the 20-day-ahead TSM prediction made 20 days earlier
- `llm_tsm_forecast` = the 20-day-ahead LLM+TSM prediction made 20 days earlier

This ensures the comparison is fair and interpretable.

---

## Internal Storage Strategy

### Source of truth

The internal research archive will begin as a **CSV archive** stored inside the LXC.

This archive is the authoritative historical record.

### Why CSV first

- simple to inspect
- easy to backfill and regenerate from
- good enough for current scale
- low operational complexity

A database can be introduced later if required, but is not needed for phase 1.

### Important architectural principle

The website does **not** accumulate daily JSON files to build history.

Instead:

- the archive CSV grows over time
- each daily run regenerates complete horizon JSON files from the archive
- the frontend simply fetches a complete chart-ready file per horizon

This means the public website gets a fresh full-history snapshot every day.

---

## Archive Record Model

The archive should preserve **forecast rows as immutable predictions**, then later fill realized values when they become known.

### Chosen policy

- append new forecast rows on each run
- later update those rows only to fill `actual` and `is_realized`

This preserves research credibility while remaining practical.

### Preferred archive schema

The archive will store **all forecast steps**, not only the four headline website horizons.

Recommended columns:

```text
run_id
forecast_made_on
latest_observed_date
latest_observed_price
target_date
step_index
actual
base_tsm_forecast
llm_tsm_forecast
model_version
model_commit
data_version
is_realized
generated_at
```

### Meaning of fields

- `run_id`: unique identifier for a specific run
- `forecast_made_on`: date the forecast was produced
- `latest_observed_date`: last real market date visible to the model when forecast was made
- `latest_observed_price`: last real price visible to the model when forecast was made
- `target_date`: the future date being predicted
- `step_index`: horizon step number, e.g. 1 through 30
- `actual`: realized value for the target date, filled later once known
- `base_tsm_forecast`: base model forecast for that target date
- `llm_tsm_forecast`: LLM-refined forecast for that target date
- `model_version`: human-readable model/pipeline label
- `model_commit`: git commit hash for reproducibility
- `data_version`: optional input data snapshot/version marker
- `is_realized`: whether the row now has a realized outcome
- `generated_at`: full timestamp for the run

### Why store all steps

This makes the archive more useful for:

- future website redesigns
- deeper diagnostics
- later research writeups
- alternative horizon views
- more flexible analytics later on

The public JSON exporter will simply select `step_index` values 1, 5, 20, and 30.

---

## Public JSON Files

The worker will regenerate the following website-facing files on every successful run:

- `latest.json`
- `horizon_1d.json`
- `horizon_5d.json`
- `horizon_20d.json`
- `horizon_30d.json`

### 1. `latest.json`

Purpose: current live forecast snapshot.

Example structure:

```json
{
  "market": "UK ETS",
  "generated_at": "2026-03-17T15:30:00Z",
  "latest_observed_date": "2026-03-17",
  "latest_observed_price": 49.82,
  "model_version": "uk_ets_llm_pipeline_v1",
  "forecasts": {
    "1d": {
      "target_date": "2026-03-18",
      "base_tsm_value": 49.95,
      "llm_tsm_value": 50.08,
      "predicted_change_abs": 0.26,
      "predicted_change_pct": 0.52
    },
    "5d": {
      "target_date": "2026-03-22",
      "base_tsm_value": 50.34,
      "llm_tsm_value": 50.71,
      "predicted_change_abs": 0.89,
      "predicted_change_pct": 1.79
    },
    "20d": {
      "target_date": "2026-04-06",
      "base_tsm_value": 51.10,
      "llm_tsm_value": 52.03,
      "predicted_change_abs": 2.21,
      "predicted_change_pct": 4.44
    },
    "30d": {
      "target_date": "2026-04-16",
      "base_tsm_value": 50.88,
      "llm_tsm_value": 51.76,
      "predicted_change_abs": 1.94,
      "predicted_change_pct": 3.89
    }
  }
}
```

### 2. Horizon files

Each horizon file provides a complete chart-ready historical series.

Example: `horizon_20d.json`

```json
{
  "market": "UK ETS",
  "horizon_days": 20,
  "generated_at": "2026-03-17T15:30:00Z",
  "model_version": "uk_ets_llm_pipeline_v1",
  "series": [
    {
      "target_date": "2026-01-15",
      "forecast_made_on": "2025-12-26",
      "actual": 52.41,
      "base_tsm_forecast": 51.02,
      "llm_tsm_forecast": 52.08
    }
  ]
}
```

The same schema applies to:

- `horizon_1d.json`
- `horizon_5d.json`
- `horizon_20d.json`
- `horizon_30d.json`

### Full-history rule

Each horizon JSON will contain **all available realized history**, not just the last year.

That makes the public chart logic simple and ensures the full archive remains usable.

---

## Automation Model

The whole system is intended to be **autonomous**.

### Chosen scheduling pattern

Two automated runs will exist:

- **19:30 Europe/London** — main production run
- **06:30 Europe/London** — repair run

### Main run responsibilities

1. pull repo updates
2. refresh all required data
3. run the UK ETS forecast
4. append new archive rows
5. backfill newly realized outcomes
6. regenerate all website JSON files
7. sync exported files to VPS staging
8. promote staging to live only if full run succeeds
9. log result

### Repair run responsibilities

1. pull repo updates
2. refresh data again
3. backfill late-arriving actuals or revised values
4. regenerate website JSON
5. sync only if outputs changed
6. promote only if the repair run completes successfully

### Purpose of repair run

The repair pass protects against delayed source availability and minor overnight revisions, improving public data quality without changing the core architecture.

---

## Scheduling Technology

### Chosen scheduler

Use:

- **systemd service + systemd timer**

### Why systemd timer

- cleaner production setup than cron
- better service control
- better observability and status handling
- better logging integration
- better fit for a single-purpose autonomous worker

---

## Runner Structure

The runner design will be:

- one shared Python runner layer
- one **main** mode
- one **repair** mode

This avoids duplication while keeping the operational behavior clear.

### Design principle

The repo remains the source of forecasting code.

The deployment layer should be **thin** and wrap existing UK ETS commands rather than rewriting project logic.

### Chosen approach for tasks 1–4

1. **Runner scripts**: shared runner with `main` and `repair` modes
2. **Repo integration**: thin deployment wrapper around existing UK ETS commands
3. **Archive logic**: append forecast rows, backfill only realized values later
4. **JSON export**: regenerate all website JSON files from the archive on each run

---

## Directory Layout in the LXC

Proposed root:

```text
/opt/ukets-forecast/
  repo/              # ETS-2 git checkout
  data/              # pulled/raw/intermediate data
  archive/           # forecast_history.csv
  exports/           # latest.json, horizon_*.json
  logs/              # daily run logs
  scripts/           # runner + deploy scripts
  config/            # env/config files
```

### Intent of each directory

- `repo/`: checked-out forecasting code
- `data/`: fetched source data and intermediate artifacts
- `archive/`: internal historical forecast store
- `exports/`: public JSON files to be deployed to website
- `logs/`: run logs and diagnostics
- `scripts/`: operational scripts and wrappers
- `config/`: environment/configuration material

---

## Website Deployment Strategy

### Chosen sync model

Use **push-based deployment from the LXC to the VPS** using:

- `rsync` over SSH

### Why this was chosen

- simple and robust
- avoids exposing home infrastructure publicly
- efficient for changed files only
- easy to automate
- fits static JSON delivery well

### Not chosen

- VPS pull model
- direct live API served from the LXC

Both were rejected as less clean for this architecture.

---

## Public Website Data Location

### Chosen public path

The JSON files will live under the main site path:

- `/data/ukets/`

### Final public URLs

- `/data/ukets/latest.json`
- `/data/ukets/horizon_1d.json`
- `/data/ukets/horizon_5d.json`
- `/data/ukets/horizon_20d.json`
- `/data/ukets/horizon_30d.json`

### Why this path was chosen

- these outputs are static website data, not a true API product
- simplest frontend integration
- easiest deployment and cache handling
- avoids needless subdomain/API complexity

---

## Safe Deployment Policy

### Chosen policy

**Only replace live website files if the entire run succeeds.**

This means the system keeps a **last-known-good** live dataset on the website.

### Deployment flow

1. generate fresh JSON in the LXC
2. sync to a staging directory on the VPS
3. validate that the expected files exist
4. only then promote the staging set to the live `/data/ukets/` directory
5. if anything fails, leave the existing live files untouched

### Benefit

The public website never shows half-written or partially updated data.

---

## Frontend Contract

The frontend will consume only the exported JSON files.

### Frontend responsibilities

- fetch `latest.json` for the current live projection view
- fetch the selected horizon file for the historical chart
- render toggle behavior between 1d / 5d / 20d / 30d
- display three-line comparisons:
  - actual
  - base TSM
  - LLM+TSM

### Frontend does not need to

- accumulate daily deltas
- reconstruct history
- know anything about archive internals
- query a database

This keeps the website layer intentionally simple.

---

## Historical Launch Backfill

The public experience should begin with a meaningful amount of existing history.

### Launch principle

The CSV archive should be pre-populated from past forecast/history data where possible before public rollout.

### Result

The website can launch with:

- substantial past realized forecast-performance history
- full charts from day one
- immediate evidence of research continuity

---

## Reliability Principles

This stage of the project is built around the following principles:

1. **Autonomy**
   - the system should run without manual intervention

2. **Separation of concerns**
   - forecasting happens in the LXC
   - public serving happens on the VPS

3. **Research credibility**
   - forecasts are preserved as historical records
   - realized outcomes are filled later, not rewritten wholesale

4. **Last-known-good deployment**
   - public data updates only on complete success

5. **Simple frontend contract**
   - website consumes static JSON snapshots only

6. **Future flexibility**
   - archive stores more detail than the initial website needs

---

## Final Agreed System

The agreed phase-1 system is:

- a dedicated Proxmox LXC runs the UK ETS forecasting worker
- the `ETS-2` repo is checked out there and updated automatically
- the worker runs twice daily:
  - 19:30 Europe/London main run
  - 06:30 Europe/London repair run
- forecasts are archived internally in a CSV-based historical store
- the archive stores all forecast steps, not only website headline horizons
- the worker regenerates full-history website JSON files every successful run
- the worker pushes those files to the VPS with `rsync` over SSH
- the VPS serves them under `/data/ukets/`
- live site files are only replaced if the full run succeeds
- the website shows both current forecast output and historical forecast-vs-reality performance

---

## Immediate Next Implementation Stage

The next execution phase should turn this foundation into concrete operational assets:

1. map exact UK ETS repo commands into the main/repair runner
2. define the archive writer and backfill updater in code
3. implement JSON exporters from the archive
4. implement the `rsync` + staging-to-live deployment step
5. create the LXC
6. install dependencies and clone the repo
7. add systemd service and timer units
8. test with a dry-run export to the website
9. backfill launch history into the CSV archive
10. connect the frontend to the published JSON files

---

## Status

This document is the agreed architectural foundation for the live UK ETS deployment stage of the project.

It defines the operating model clearly enough to begin implementation.

