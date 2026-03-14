# Greenfield Build Spec: Option 4 EUA Forecaster (From Scratch)

## 1. Objective

Build a complete EUA forecasting system in a brand-new repository that:

1. Acquires raw market and news data from upstream sources automatically.
2. Trains a base Autoformer model and two LLM refinement variants.
3. Reproduces the Option 4 comparison framework.
4. Runs daily in production to generate new forecasts.

This document is written as an execution contract for a coding LLM.

---

## 2. Benchmark Targets to Reproduce

Use these as hard acceptance targets for the first full replication run.

1. Option 1: Base Autoformer
2. Option 2: Autoformer + CoT
3. Option 3: Autoformer + CoT + Sentiment
4. Option 4: Autoformer + CoT + RAMP

Target metrics:

- Option 1:
  - `h1_mse=0.0974587401`
  - `h5_mse=1.8901053336`
  - `h20_mse=19.2321477727`
  - `h30_mse=40.7649746402`
  - `path_mse=14.5157740858`
- Option 2:
  - `h1_mse=0.8592593710`
  - `h5_mse=2.7859793521`
  - `h20_mse=16.4193515483`
  - `h30_mse=34.9366153779`
  - `path_mse=13.0727497192`
- Option 3:
  - `h1_mse=0.2649888961`
  - `h5_mse=1.8328393011`
  - `h20_mse=18.3856659581`
  - `h30_mse=38.5202010711`
  - `path_mse=13.8941514319`
- Option 4:
  - `h1_mse=0.1132312413`
  - `h5_mse=1.8322721766`
  - `h20_mse=16.8039407144`
  - `h30_mse=34.5106707168`
  - `path_mse=12.7200138910`

Option 4 blending definition:

- Blend base TSM forecast with Option 2 (CoT) forecast only.
- Use ramp schedule with:
  - `strength=0.50`
  - `min_weight=0.15`
  - `power=1.0`
  - `pred_len=30`

Blend formula:

```text
w_h = (linspace(min_weight, strength, pred_len)[h-1]) ^ power
y_blend[h] = (1 - w_h) * y_tsm[h] + w_h * y_cot[h]
```

Acceptance tolerance:

- Same hardware/software stack: absolute error <= `1e-6`.
- Different hardware stack: absolute error <= `1e-4`.

---

## 3. Build Order (Mandatory)

Execute phases in this exact order:

1. Repository bootstrap and package layout.
2. Data connectors and raw data contracts.
3. Feature panel builder and split logic.
4. Base Autoformer training/inference.
5. LLM refinement layer.
6. Evaluation and comparison table generator.
7. Production orchestration and monitoring.

Do not skip phase order.

---

## 4. New Repository Layout

Create this structure:

```text
repo/
  pyproject.toml
  requirements.txt
  .env.example
  src/
    config/
      default.yaml
      paper_like.yaml
    data/
      sources/
      transforms/
      panel_builder.py
      split.py
    models/
      autoformer.py
      trainer.py
      inference.py
    llm/
      prompts.py
      client.py
      refine.py
      parse.py
      cache.py
    eval/
      metrics.py
      comparison.py
    pipeline/
      run_experiment.py
      run_daily.py
      publish.py
    utils/
      io.py
      logging.py
      reproducibility.py
  scripts/
    bootstrap_data_sources.py
    build_daily_sentiment.py
    label_news_sentiment.py
    compare_options.py
    validate_reproduction.py
  data/
    raw/
    processed/
    snapshots/
  runs/
  docs/
    OPTION4_REPRODUCTION_AND_PRODUCTION_RUNBOOK.md
```

---

## 5. Data Acquisition Contracts (From Scratch)

Implement source adapters with retries, backoff, and schema validation.

Required datasets:

1. EUA futures prices (target series).
2. Energy benchmarks:
  - Brent spot (USD).
  - Rotterdam coal futures (USD).
3. Carbon ETFs/indices:
  - `KEUA`, `KRBN`, `GRN`, `KCCA`, `KSET`.
4. EU auction data (price and volume).
5. ICAP secondary EUA market series.
6. VSTOXX volatility index.
7. EURUSD FX rates.
8. News headline feed for sentiment.

Recommended sources:

1. ECB SDW for FX (`EUR/USD`) in XML/CSV form.
2. GDELT DOC 2.0 API for headline retrieval.
3. Official exchange/operator CSV endpoints where available for EUA/auction/volatility.
4. Backup market data providers with legal access when primary source unavailable.

For each source adapter, enforce:

1. Idempotent fetch.
2. Timestamped raw snapshot.
3. Parsed normalized table with canonical columns.
4. Checksum and row-count metadata.
5. Date continuity and duplicate-date handling.

If source is unavailable:

1. Keep last known valid snapshot.
2. Flag run as degraded.
3. Continue only if policy allows stale data.

---

## 6. Canonical Modeling Setup

Use these fixed settings for first reproduction:

1. Forecast horizons: `1,5,20,30`.
2. Prediction length: `30`.
3. Sequence length: `60`.
4. Label length: `15`.
5. Splits:
  - `train_end=2023-06-30`
  - `val_end=2024-06-30`
  - `test_end=2026-01-05`
6. Target mode: returns-based training, price-path evaluation.
7. Model family: Autoformer baseline.

Feature set must include:

1. Target lag/rolling features.
2. Brent and coal returns.
3. Selected carbon index features.
4. Auction features (price, volume, auction-day indicator, lags).
5. ICAP secondary market feature.
6. VSTOXX feature.

---

## 7. LLM Refinement Methods

Implement two methods:

1. `TSM+LLM-COT-RF`
2. `TSM+LLM-COT-SENT-RF`

Common requirements:

1. `temperature=0.0` for benchmark runs.
2. Strict response parsing from final assistant output only.
3. Ignore reasoning traces and internal chain-of-thought fields.
4. Save per-call metadata including parse success and fallback flags.
5. Use response cache keyed by:
  - provider
  - model
  - base_url
  - prompt payload
  - system instruction
  - response format

CoT-RF setup:

1. history window: `18` points in prompt context.
2. teaching examples: `k=5`.
3. similarity-based selection with lookback window.
4. two-stage reflect-then-apply flow.

CoT-SENT-RF setup:

1. Same as CoT-RF plus sentiment history in prompt.
2. sentiment history length: `18`.

Fallback policy:

1. If parse fails after retries, use TSM forecast for that sample.
2. Record fallback in metadata.

---

## 8. Sentiment Pipeline (Daily)

Build this pipeline:

1. Fetch EU ETS-relevant headlines daily.
2. Filter and deduplicate headlines.
3. Label sentiment using 3-vote majority.
4. Aggregate to daily score.

Baseline sentiment scoring:

1. Direction labels only: `YES/NO/UNKNOWN`.
2. Map to `+1/-1/0`.
3. Daily score = mean of same-day labeled scores.

Store artifacts:

1. `headlines_raw.csv`
2. `headlines_filtered.csv`
3. `headlines_labeled.csv`
4. `daily_sentiment.csv`

---

## 9. Evaluation Outputs (Required)

Every full experiment run must emit:

1. `metrics_by_horizon.csv`
2. `path_metrics.csv`
3. `four_option_comparison_with_cot_ramp.csv`
4. `predictions` artifacts for:
  - TSM
  - CoT
  - CoT+Sent
  - Option 4 blended path
5. `config_resolved.yaml`
6. `llm_metadata.jsonl`

Definitions:

1. `h*_mse`: MSE at exact horizon.
2. `path_mse`: mean MSE over all prediction steps and all samples.

---

## 10. Reproduction Protocol in New Repo

Run this sequence:

1. `scripts/bootstrap_data_sources.py` to build all required raw and processed datasets.
2. `pipeline/run_experiment.py --config src/config/paper_like.yaml`.
3. `scripts/compare_options.py` to build Option 1-4 table.
4. `scripts/validate_reproduction.py` against benchmark targets from Section 2.

Validation script must fail if:

1. any required option row missing.
2. any required metric missing.
3. tolerance breached.

---

## 11. Production Daily Forecaster Design

Implement a single daily orchestrator:

1. Ingest market/fundamental data.
2. Ingest and label news.
3. Build latest daily sentiment file.
4. Run forecast inference.
5. Compute Option 4 output.
6. Publish forecast package.
7. Emit monitoring payload.

Daily publish package must include:

1. forecast date.
2. model/version identifiers.
3. TSM forecast and Option 4 forecast.
4. data freshness report.
5. fallback rate and parse success rate.
6. runtime and failure status.

---

## 12. Operations and Monitoring

Track these production KPIs:

1. Data freshness age by source.
2. LLM parse success rate.
3. LLM fallback rate.
4. Daily run latency.
5. Rolling error drift vs baseline.

Alert thresholds:

1. parse success < 99%.
2. fallback rate > 5%.
3. missing critical source.
4. stale critical source beyond SLA.

Graceful degradation:

1. If LLM unavailable, publish TSM-only forecast and set status `DEGRADED`.
2. If critical market data unavailable, abort publish and set status `FAILED`.

---

## 13. CI Requirements

Create CI workflows for:

1. unit tests for parsers, blend logic, metrics.
2. contract tests for each source adapter.
3. deterministic smoke test on small fixture dataset.
4. schema regression checks on produced artifacts.

Mandatory tests:

1. Blend math exactness for Option 4 parameters.
2. Path-metric calculations.
3. LLM response parser rejecting reasoning-only output.
4. Fallback behavior correctness.

---

## 14. Definition of Done

Done means all are true:

1. A brand-new repo can be initialized and run end-to-end with no manual dataset placement.
2. Option 1-4 table is generated automatically.
3. Reproduction validator passes benchmark tolerances.
4. Daily scheduler runs unattended and publishes forecast package.
5. Monitoring and alerting are active.

---

## 15. Implementation Notes for the Coding LLM

1. Do not hardcode secrets.
2. Read API keys and endpoints from environment.
3. Version every run with immutable run IDs.
4. Never overwrite previous run artifacts.
5. Keep code and config separated so experiments are reproducible.

---

## 16. Data Automation Deep Dive (Verified on 2026-02-26)

### 16.1 Source-by-Source Automation Feasibility

1. EUA futures target series:
  - Source tested: `https://www.investing.com/commodities/european-union-allowance-eua-year-futures-historical-data`
  - Status: `PARTIAL`
  - What works: page HTML contains a rolling incremental window in `historicalDataStore` (observed window: 2026-01-26 to 2026-02-25).
  - What does not work robustly: full-history API endpoints are protected by anti-bot/challenge flows, so unattended full backfill from scratch is not reliable.
  - Current strategy: seed from frozen baseline in `Data/eua-futures`, then append incremental rows into `Data_auto/eua-futures`.

2. ICAP secondary market:
  - Sources tested:
    - `https://allowancepriceexplorer.icapcarbonaction.com/api/systems`
    - `https://allowancepriceexplorer.icapcarbonaction.com/systems/reports/price/download`
  - Status: `FULLY AUTOMATABLE`
  - Notes: system IDs can be discovered dynamically, then downloaded by date range.

3. EEX primary auctions:
  - Source tested: `https://public.eex-group.com/eex/eua-auction-report/`
  - Status: `FULLY AUTOMATABLE`
  - Notes: yearly files are published as `.xls/.xlsx`; parser supports both. Runtime parsing of Excel files requires `xlrd`/`openpyxl` (now listed in `requirements.txt`).

4. EURUSD FX:
  - Sources tested:
    - `https://www.ecb.europa.eu/stats/eurofxref/eurofxref-hist.xml`
    - `https://data-api.ecb.europa.eu/service/data/EXR/D.USD.EUR.SP00.A?format=csvdata`
  - Status: `FULLY AUTOMATABLE`

5. Brent spot:
  - Source tested: `https://fred.stlouisfed.org/graph/fredgraph.csv?id=DCOILBRENTEU`
  - Status: `FULLY AUTOMATABLE`

6. Rotterdam coal proxy:
  - Source tested: Yahoo Finance via `MTF=F` (through `yfinance`)
  - Status: `AUTOMATABLE WITH DEPENDENCY RISK`
  - Notes: unofficial wrapper/provider risk exists; no guaranteed SLA.

7. Carbon ETF/index proxies (`GRN`, `KCCA`, `KEUA`, `KRBN`, `KSET`):
  - Source tested: Yahoo Finance via `yfinance`
  - Status: `AUTOMATABLE WITH CAVEATS`
  - Notes: `KSET` is delisted (history ends on 2024-03-28), so no new rows after delisting.

8. VSTOXX:
  - Source tested: `https://stoxx.com/index/v2tx/` (embedded `window.chart_data`)
  - Status: `PARTIAL`
  - What works: scraping embedded chart payload from the public page.
  - Limitation: dedicated official API endpoints require authentication/licensing.

9. News headlines:
  - Source tested: `https://api.gdeltproject.org/api/v2/doc/doc`
  - Status: `FULLY AUTOMATABLE` (with rate/coverage caveats)

### 16.2 What Cannot Be Reliably Fully Automated (Current Constraints)

1. A fully official, unrestricted, free EUA futures settlement history feed suitable for unattended full historical rebuilds was not identified.
2. A fully official, unrestricted VSTOXX historical feed was not identified without authentication/licensing.
3. Any provider behind anti-bot/challenge workflows is fragile for production scraping unless you operate a browser-based collector and accept compliance risk.

### 16.3 Separation of Frozen vs Automated Data

1. Frozen training baseline remains in `Data/`.
2. Automated refreshes are written to `Data_auto/`.
3. `Data_auto/automation_manifest.json` records per-source success, row counts, and date coverage.

### 16.4 Operational Commands

1. Bootstrap automatable sources into separate data root:

```bash
python scripts/bootstrap_data_sources.py --output-dir Data_auto --seed-data-dir Data
```

2. Run the full experiment against automated data root:

```bash
python -m src.run_experiment --config src/config/default.yaml --data-dir Data_auto
```

## 17. Separate Automated Pipeline + CoT End-to-End (Implemented 2026-02-26)

The separate runner now supports a full CoT workflow on top of `Data_auto` without touching the frozen `Data/` baseline and without writing to shared `paper/` outputs by default.

### 17.1 What the runner now does

1. Uses automated data root (`Data_auto`) and keeps frozen base data separate.
2. Uses returns-mode target by default (`target.mode=returns`) to prevent price-level collapse.
3. Can run CoT end-to-end using:
  - `TSM+LLM-COT-RF`
  - `TSM+LLM-COT-SENT-RF`
4. Generates `four_option_comparison_with_cot_ramp.csv` in each run.
5. Writes paper snapshot only to `runs/<run_id>/paper_snapshot` and skips shared `paper/` unless explicitly enabled.

### 17.2 CoT end-to-end command (recommended)

```bash
python scripts/run_automated_pipeline_once.py \
  --config src/config/default.yaml \
  --data-dir Data_auto \
  --seed-data-dir Data \
  --skip-bootstrap \
  --cot-end-to-end \
  --cot-method cot_rf \
  --llm-api-key deo
```

Notes:

1. `--llm-max-samples` defaults to `12` for runtime/cost control.
2. For a full split CoT run, set `--llm-max-samples 100000`.
3. If `--llm-base-url` is omitted, runner resolves in this order:
  - `OPENAI_BASE_URL`
  - fallback: `http://192.168.1.140:9877/v1`

### 17.3 CoT + Sentiment variant

If daily sentiment exists, run:

```bash
python scripts/run_automated_pipeline_once.py \
  --config src/config/default.yaml \
  --data-dir Data_auto \
  --seed-data-dir Data \
  --skip-bootstrap \
  --cot-end-to-end \
  --cot-method cot_sent_rf \
  --sentiment-path data/news/daily_sentiment.csv \
  --llm-api-key deo
```

If sentiment file is missing, build it first (raw -> filtered -> labeled -> daily aggregate) and then rerun.

### 17.4 Required outputs from a CoT run

Under `runs/<run_id>/results/`:

1. `metrics_by_horizon.csv`
2. `path_metrics.csv`
3. `four_option_comparison_with_cot_ramp.csv`

Under `runs/<run_id>/llm/`:

1. Per-method metadata/logs and cached responses.
2. Blend-grid selection metadata (when CoT blend is enabled).

### 17.5 Option mapping in generated comparison table

1. `Option1_Autoformer` -> `tsm`
2. `Option2_CoT_RF` -> `TSM+LLM-COT-RF`
3. `Option3_CoT_SENT_RF` -> `TSM+LLM-COT-SENT-RF` (when enabled)
4. `Option4_CoT_RAMP` -> `TSM+LLM-COT-RF_blend_ramp_bestval_path`

Option 4 uses ramp blend settings aligned to this runbook:

1. `strength=0.50` (via blend weight grid fixed at `0.5`)
2. `min_weight=0.15`
3. `power=1.0`

### 17.6 Validation Run (2026-02-26)

Executed:

```bash
python scripts/run_automated_pipeline_once.py \
  --config src/config/default.yaml \
  --data-dir Data_auto \
  --seed-data-dir Data \
  --skip-bootstrap \
  --cot-end-to-end \
  --cot-method cot_rf \
  --llm-max-samples 4 \
  --llm-api-key deo
```

Outputs:

1. Run directory: `runs/20260226_161113_b3d7f6`
2. Option table: `runs/20260226_161113_b3d7f6/results/four_option_comparison_with_cot_ramp.csv`
3. LLM logs: `runs/20260226_161113_b3d7f6/llm/logs/llm_calls.jsonl`
4. Blend selection: `runs/20260226_161113_b3d7f6/llm/blend_grid_selection_TSM_LLM-COT-RF.json`

Observed CoT metrics from this validation run:

1. `Option1_Autoformer (tsm)`: `path_mse=15.294339`
2. `Option2_CoT_RF`: `path_mse=11.900138` (`llm_subset`, `n_samples=4`)
3. `Option4_CoT_RAMP`: `path_mse=15.730363` (`llm_subset`, `n_samples=4`)

Interpretation:

1. CoT path MSE looked better than base TSM on this small subset sample.
2. Option 4 blend did not beat CoT-only in this small-sample validation.
3. Use a larger `--llm-max-samples` for stable comparison decisions.

### 17.7 Option 1 Full-Split Run (2026-02-26)

Executed Option 1 follow-up using full test split (`n=382`) with cached CoT responses and blend-grid disabled (Option 1 focus):

```bash
python - <<'PY'
from src.run_experiment import run_experiment
run_experiment(
  config_path='runs/20260226_162339_f91555/config_resolved.yaml',
  overrides={
    'llm': {'blend_grid': {'enabled': False}, 'calibrate_blend': {'enabled': False}},
    'output': {'write_project_paper': False},
  },
  data_dir='Data_auto',
)
PY
```

Observed horizon MSE:

1. `Option1_Autoformer (tsm)`:
  - `h1=0.090856`
  - `h5=1.772453`
  - `h20=20.012070`
  - `h30=45.426651`
  - `path_mse=15.294339`
2. `Option2_CoT_RF (TSM+LLM-COT-RF)`:
  - `h1=0.919657`
  - `h5=2.549415`
  - `h20=16.594425`
  - `h30=38.882804`
  - `path_mse=13.377793`

### 17.8 Option 1 Rerun (Old Hyperparams on Current Data, 2026-02-27)

To avoid fixed `run_id` reuse from resolved configs, metadata-stripped temp configs were created:

```bash
python - <<'PY'
import yaml
from pathlib import Path
for src, dst in [
    ('runs/20260205_231634_6a6539/config_resolved.yaml', 'tmp/config_old_hparams.yaml'),
    ('runs/20260226_162339_f91555/config_resolved.yaml', 'tmp/config_current_hparams.yaml'),
]:
    data = yaml.safe_load(Path(src).read_text())
    for k in ('run_id', 'git_hash', 'timestamp'):
        data.pop(k, None)
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    Path(dst).write_text(yaml.safe_dump(data, sort_keys=False))
PY
```

Executed Option 1 rerun (old hyperparameters, current `Data_auto`, blend disabled, capped CoT subset):

```bash
python - <<'PY'
from src.run_experiment import run_experiment
run_experiment(
  config_path='tmp/config_old_hparams.yaml',
  overrides={
    'llm': {
      'max_samples': 40,
      'blend_grid': {'enabled': False, 'max_samples': 40},
      'calibrate_blend': {'enabled': False},
    },
    'output': {'generate_paper': False, 'write_project_paper': False},
  },
  data_dir='Data_auto',
)
PY
```

Output run:

1. `runs/20260227_005459_14addd`

Observed MSE:

1. `Option1_Autoformer (tsm, full, n=382)`:
  - `h1=0.096391`
  - `h5=1.856243`
  - `h20=20.157988`
  - `h30=46.037006`
  - `path_mse=15.415826`
2. `Option2_CoT_RF (TSM+LLM-COT-RF, llm_subset, n=40)`:
  - `h1=1.016525`
  - `h5=2.629148`
  - `h20=13.406530`
  - `h30=22.778991`
  - `path_mse=9.994070`

### 17.9 Option 2 Follow-up (Current Hyperparams + Blend, 2026-02-27)

Executed current-hyperparameter run with blend enabled and capped CoT subsets for both test and blend-val tuning:

```bash
python - <<'PY'
from src.run_experiment import run_experiment
run_experiment(
  config_path='tmp/config_current_hparams.yaml',
  overrides={
    'llm': {
      'max_samples': 40,
      'blend_grid': {'enabled': True, 'schedule': 'ramp', 'max_samples': 40},
      'calibrate_blend': {'enabled': False},
    },
    'output': {'generate_paper': False, 'write_project_paper': False},
  },
  data_dir='Data_auto',
)
PY
```

Output run:

1. `runs/20260227_010125_e37a5b`
2. Blend selection: best `mse_path` at `w=0.50`

Observed MSE:

1. `Option1_Autoformer (tsm, full, n=382)`:
  - `h1=0.090856`
  - `h5=1.772453`
  - `h20=20.012070`
  - `h30=45.426651`
  - `path_mse=15.294339`
2. `Option2_CoT_RF (TSM+LLM-COT-RF, llm_subset, n=40)`:
  - `h1=1.167408`
  - `h5=2.186126`
  - `h20=15.703725`
  - `h30=22.034274`
  - `path_mse=11.157004`
3. `Option4_CoT_RAMP (TSM+LLM-COT-RF_blend_ramp_bestval_path, llm_subset, n=40)`:
  - `h1=0.127775`
  - `h5=1.852575`
  - `h20=19.159607`
  - `h30=21.750362`
  - `path_mse=11.973283`

### 17.10 Uncapped reruns for both options (2026-02-27)

Ran both uncapped reruns using existing run configs and setting `llm.max_samples` / `blend_grid.max_samples` to `100000` (effective full split).

Step 1 (old hyperparams on current data, blend disabled):

```bash
python - <<'PY'
from src.run_experiment import run_experiment
run_experiment(
  config_path='runs/20260227_005459_14addd/config_resolved.yaml',
  overrides={
    'llm': {
      'max_samples': 100000,
      'blend_grid': {'enabled': False, 'max_samples': 100000},
      'calibrate_blend': {'enabled': False},
    },
    'output': {'generate_paper': False, 'write_project_paper': False},
  },
  data_dir='Data_auto',
)
PY
```

Step 2 (current hyperparams with blend enabled):

```bash
python - <<'PY'
from src.run_experiment import run_experiment
run_experiment(
  config_path='runs/20260227_010125_e37a5b/config_resolved.yaml',
  overrides={
    'llm': {
      'max_samples': 100000,
      'blend_grid': {'enabled': True, 'schedule': 'ramp', 'max_samples': 100000},
      'calibrate_blend': {'enabled': False},
    },
    'output': {'generate_paper': False, 'write_project_paper': False},
  },
  data_dir='Data_auto',
)
PY
```

Note:

1. Compatible historic cache directories were merged into the active run caches to accelerate runtime.
2. Metrics below are from completed uncapped reruns with `llm_subset n=382` (full test split).

Uncapped observed MSE:

1. Step 1 run: `runs/20260227_005459_14addd`
  - `Option1_Autoformer (tsm, full, n=382)`:
    - `h1=0.096391`
    - `h5=1.856243`
    - `h20=20.157988`
    - `h30=46.037006`
    - `path_mse=15.415826`
  - `Option2_CoT_RF (TSM+LLM-COT-RF, llm_subset, n=382)`:
    - `h1=0.929356`
    - `h5=2.591029`
    - `h20=17.114400`
    - `h30=38.213522`
    - `path_mse=13.961019`
  - Path delta (`Option2 - Option1`): `-1.454807` (`-9.44%`)
2. Step 2 run: `runs/20260227_010125_e37a5b`
  - `Option1_Autoformer (tsm, full, n=382)`:
    - `h1=0.090856`
    - `h5=1.772453`
    - `h20=20.012070`
    - `h30=45.426651`
    - `path_mse=15.294339`
  - `Option2_CoT_RF (TSM+LLM-COT-RF, llm_subset, n=382)`:
    - `h1=0.966062`
    - `h5=2.555232`
    - `h20=16.602918`
    - `h30=38.886850`
    - `path_mse=13.387671`
  - `Option4_CoT_RAMP (TSM+LLM-COT-RF_blend_ramp_bestval_path, llm_subset, n=382)`:
    - `h1=0.112020`
    - `h5=1.719195`
    - `h20=17.121616`
    - `h30=38.389888`
    - `path_mse=13.158546`
  - Path deltas:
    - `Option2 - Option1`: `-1.906669` (`-12.47%`)
    - `Option4 - Option1`: `-2.135794` (`-13.96%`)
    - `Option4 - Option2`: `-0.229125` (`-1.71%`)

### 17.11 Hybrid Option-4 (`h1` forced to base) (2026-02-27)

Implemented hybrid registration in the experiment pipeline:

1. Base blend model still produced as before:
  - `TSM+LLM-COT-RF_blend_ramp_bestval_path`
2. New hybrid model:
  - `TSM+LLM-COT-RF_blend_ramp_bestval_path_h1base`
  - Logic: use ramp blend for all horizons, then overwrite `h1` with `tsm` prediction.

Also updated automated option table generation to prefer `*_h1base` when available.

Validation using existing uncapped run predictions from `runs/20260227_010125_e37a5b`:

1. `Option4_CoT_RAMP`:
  - `h1=0.112020`
  - `h5=1.719195`
  - `h20=17.121616`
  - `h30=38.389888`
  - `path_mse=13.158546`
2. `Option4_CoT_RAMP_H1_BASE`:
  - `h1=0.090856` (same as `Option1_Autoformer`)
  - `h5=1.719195`
  - `h20=17.121616`
  - `h30=38.389888`
  - `path_mse=13.157840`

Net effect: this satisfies the requirement to keep `h1` pure Autoformer while preserving ramp-blend gains on longer horizons.

### 17.12 Fresh uncapped run with `h1` base override model (2026-02-27)

Executed a fresh full-split run on `Data_auto` with current hyperparameters and blend enabled, using the new hybrid registration:

1. Run directory: `runs/20260227_192949_7d0e17`
2. Data root: `Data_auto` (separate automated pipeline data)
3. LLM subset size: uncapped (`n=382` effective)

Observed results (`llm_subset`, `n=382`):

1. `Option1_Autoformer (tsm)`:
  - `h1=0.090856`
  - `path_mse=15.294339`
2. `Option4_CoT_RAMP (TSM+LLM-COT-RF_blend_ramp_bestval_path)`:
  - `h1=0.112049`
  - `path_mse=13.158230`
3. `Option4_CoT_RAMP_H1_BASE (TSM+LLM-COT-RF_blend_ramp_bestval_path_h1base)`:
  - `h1=0.090856` (forced to base Autoformer)
  - `path_mse=13.157524`

Result: the separate automated pipeline now produces the hybrid `h1`-base model as a first-class output in run artifacts, with path MSE effectively matching prior expectation (`~13.1578`).

### 17.13 Production hardening for portable bud (2026-02-27)

Implemented additional hardening to make the automated bud portable across projects while keeping the same data sources and LLM access pattern:

1. Runner seed fallback:
  - `scripts/run_automated_pipeline_once.py` now falls back to `--data-dir` as seed source when `--seed-data-dir` is missing.
2. EUA self-seeding:
  - `scripts/bootstrap_data_sources.py` now merges incremental EUA rows with existing `Data_auto` snapshot when available, so copied `Data_auto` can continue updating without original `Data/`.
3. LLM key resolution:
  - Runner now resolves API key in order: CLI arg -> `OPENAI_API_KEY` -> local default `deo`.
4. Portable bud docs and wrapper:
  - Added `bud/README.md` with dependencies/env/run commands.
  - Added executable `bud/run_production_once.sh` wrapper for production runs.

Recommended production invocation from bud docs:

```bash
bash bud/run_production_once.sh
```

### 17.14 True standalone bud package (single-folder portability) (2026-02-27)

Created `bud_standalone/` so the pipeline can be copied as one folder into another project and executed there.

Contents:

1. `bud_standalone/src/` (full pipeline code)
2. `bud_standalone/scripts/` (automated runner + data bootstrap)
3. `bud_standalone/requirements.txt`
4. `bud_standalone/Data_auto/` (seed snapshot for self-seeding updates)
5. `bud_standalone/init.sh`
6. `bud_standalone/run_once.sh`
7. `bud_standalone/init_and_run.sh`
8. `bud_standalone/README.md`

Usage in a different repo:

```bash
cd bud_standalone
./init.sh
export OPENAI_API_KEY=deo
export OPENAI_BASE_URL=http://192.168.1.140:9877/v1
./run_once.sh
```

Or:

```bash
./init_and_run.sh
```

## 18. Frontend Export Flow (2026-02-27)

To keep training/inference in this backend repo and keep the portfolio frontend lightweight, frontend-ready JSON export is now supported.

### 18.1 Added components

1. Export module:
  - `src/frontend_export.py`
2. Export CLI:
  - `scripts/export_frontend_artifacts.py`
3. Automated runner integration:
  - `scripts/run_automated_pipeline_once.py`
  - Supports:
    - `--export-frontend`
    - `--frontend-output-dir`
    - `--frontend-publish-dir`
    - `--frontend-model-name`
    - `--frontend-history-days`

### 18.2 Exported files

For a completed run, exporter writes:

1. `latest_forecast.json`
2. `model_comparison.json`
3. `recent_history.json`
4. `manifest.json`

Default output location:

1. `runs/<run_id>/frontend_export/`

### 18.3 Manual export command

```bash
python scripts/export_frontend_artifacts.py \
  --run-dir runs/20260227_192949_7d0e17
```

### 18.4 Publish directly into frontend static assets

Local development pattern:

1. Backend repo runs exporter.
2. Exporter copies JSON into frontend repo `public/` subdirectory.
3. Frontend fetches static JSON with normal HTTP requests.

Example:

```bash
python scripts/export_frontend_artifacts.py \
  --run-dir runs/20260227_192949_7d0e17 \
  --publish-dir /path/to/portfolio/public/ets
```

Then frontend can fetch:

1. `/ets/latest_forecast.json`
2. `/ets/model_comparison.json`
3. `/ets/recent_history.json`

### 18.5 End-to-end automated run + export

```bash
python scripts/run_automated_pipeline_once.py \
  --config src/config/default.yaml \
  --data-dir Data_auto \
  --seed-data-dir Data_auto \
  --cot-end-to-end \
  --cot-method cot_rf \
  --llm-max-samples 100000 \
  --no-paper-snapshot \
  --export-frontend \
  --frontend-publish-dir /path/to/portfolio/public/ets
```

### 18.6 Current scope

Exporter currently publishes the latest available backtest window from the selected model, not a separate live production forecast from the most recent panel row.

Current exported `forecast_kind`:

1. `backtest_latest_window`

This is sufficient for frontend integration and local development. A dedicated production inference path can be added later if live post-cutoff forecasts are required.

### 18.7 Publish flow independent of automated refresh

Frontend publishing does not require using the automated data-refresh flow. It only requires a completed backend run directory with standard run outputs.

Added script:

1. `scripts/publish_frontend_artifacts.py`

Behavior:

1. Accepts an explicit `--run-dir`, or
2. If omitted, finds the latest completed run under `runs/`
3. Exports frontend JSON bundle
4. Copies the bundle into a fixed frontend static/public directory

Example local-laptop flow:

```bash
python scripts/publish_frontend_artifacts.py \
  --publish-dir /path/to/portfolio/app/public/ets
```

Or publish a specific backend run:

```bash
python scripts/publish_frontend_artifacts.py \
  --run-dir runs/20260227_192949_7d0e17 \
  --publish-dir /path/to/portfolio/app/public/ets
```

Frontend then fetches static files from:

1. `/ets/latest_forecast.json`
2. `/ets/model_comparison.json`
3. `/ets/recent_history.json`

Recommended local integration:

1. Run backend experiment in this repo.
2. Run publish script pointing at frontend repo `public/ets`.
3. Refresh frontend page; it reads updated JSON automatically.

## 19. Contamination-Safe Live Bundle And 5-Year Archive

### 19.1 Goal

Add a separate frontend data contract for:

1. A daily-updating EUA futures ticker using the same daily data cadence as the backend.
2. A live forward forecast from the most recent data cutoff.
3. A historical archive of forecast vintages for the last 5 years.
4. A contamination-safe comparison between:
   - `tsm`
   - `TSM+LLM-COT-RF_blend_ramp_bestval_path_h1base`

### 19.2 Leakage issue in the old evaluation path

The original `make_windows()` path feeds the decoder actual future rows:

1. Encoder: past observed rows
2. Decoder context: trailing observed rows
3. Decoder future section: realized future rows from the dataset

That is acceptable for a research-style teacher-forced training setup, but not acceptable for a production-style historical display where we want to claim the model only knew information available at forecast time.

For the live/frontend bundle, a separate path was added instead of modifying the main experiment pipeline.

### 19.3 New implementation

Added:

1. `src/frontend_live_bundle.py`
2. `scripts/build_frontend_live_bundle.py`

This path:

1. Builds the same panel from `Data_auto`.
2. Uses `target.mode=returns`.
3. Builds clean windows where future decoder slots are placeholders instead of realized future values.
4. Re-trains the TSM model on expanding history for each archive origin.
5. Runs `TSM+LLM-COT-RF` on top of the clean TSM forecast.
6. Applies the fixed winning blend:
   - schedule: `ramp`
   - strength: `0.5`
   - `min_weight=0.15`
   - `power=1.0`
   - `h1` forced back to base TSM

### 19.4 Placeholder policy for clean decoder future slots

For the contamination-safe path:

1. Price-like levels are carried forward from the last observed row.
2. Return-like fields (`*_return`, `*_pct`) are set to `0.0`.
3. Binary flag-like fields (`is_*`, `*_flag`) are set to `0.0`.
4. Target future values are never filled with realized outcomes.

This is a production approximation, not a teacher-forced research setup.

### 19.5 Archive cadence choice

Daily retraining over 5 years would be far too expensive for the backend role this repo now plays.

Default archive cadence was set to quarterly-equivalent trading steps:

1. `origin_step=63`

That produced 21 rolling origins from:

1. `2021-02-25`
2. through `2025-12-22`

The daily ticker remains daily. Only the expensive historical retraining/archive uses quarterly spacing.

### 19.6 Output files

Bundle output:

1. `current_price.json`
2. `actuals_recent.json`
3. `current_forecast.json`
4. `forecast_archive.csv`
5. `forecast_archive.json`
6. `model_summary.json`
7. `status.json`
8. `manifest.json`

Run-level supporting files:

1. `results/archive_origin_metrics.csv`
2. `results/archive_model_summary.csv`
3. `predictions/forecast_archive.csv`

### 19.7 Command used

```bash
python scripts/build_frontend_live_bundle.py \
  --data-dir Data_auto \
  --target-mode returns \
  --origin-step 63 \
  --llm-api-key deo
```

### 19.8 Completed run

Completed run:

1. `runs/20260228_024712_30d3e6`

Frontend bundle:

1. `runs/20260228_024712_30d3e6/frontend_live_bundle/`

Current daily ticker:

1. Instrument: `EUA_FUTURES`
2. As of: `2026-02-25`
3. Daily close: `72.58`

Current live forecast origin:

1. `2026-02-25`

First 5 forecast points:

1. Base TSM: `72.4472, 72.4827, 72.5840, 72.7476, 72.9074`
2. CoT Ramp h1base: `72.4472, 72.5179, 72.6129, 72.7574, 72.8960`

### 19.9 Historical archive result

Clean 5-year rolling archive result with 21 origins:

1. `tsm`: `path_mse = 62.491778`
2. `TSM+LLM-COT-RF_blend_ramp_bestval_path_h1base`: `path_mse = 67.153091`

Interpretation:

1. Once future information is removed from decoder inputs, the base TSM now outperforms the CoT+ramp blend on the honest rolling archive.
2. The blend is not useless; it still gives an alternate path worth showing on the frontend.
3. But the old “blend is clearly best” conclusion does not survive this contamination-safe historical test.

### 19.10 Frontend implication

The portfolio page should show both:

1. Base TSM
2. CoT Ramp final blend

Reason:

1. This better reflects the actual research story.
2. Users can see where the blend helps and where it does not.
3. The historical archive now supports that claim with pre-made forecast vintages and realized outcomes.

## 20. Main Pipeline Leakage Audit And Fix

### 20.1 Why this was necessary

The main experiment pipeline had been claiming that the CoT-enhanced path beat base `tsm`.

That claim was no longer acceptable after the contamination-safe archive showed the opposite. The main pipeline therefore had to be audited and corrected directly, not just bypassed with a separate clean path.

### 20.2 Leakage sources identified

Three concrete leakage paths were found.

1. Decoder future leakage in `src/data/windows.py`
   - `make_windows()` was building `X_dec` using actual future rows from the dataset for the decoder future segment.
   - That exposed realized future feature values to both training and evaluation windows.

2. Split-boundary leakage in `src/data/windows.py`
   - `split_windows()` was assigning windows by prediction start date only.
   - A training window could still have labels extending into validation.
   - A validation window could still have labels extending into test.

3. CoT teaching-example leakage in `src/run_experiment.py`
   - The CoT example pool was effectively assembled across train, validation, and test windows.
   - That allowed later test-time prompts to use truths from earlier test windows.

### 20.3 Fixes applied

Applied fixes:

1. `make_windows()` now supports `return_metadata=True` and returns `WindowMetadata` with:
   - prediction start dates
   - prediction end dates
   - last observed input dates

2. Decoder future slots are now filled contamination-safely:
   - return-like fields: `0.0`
   - binary flag-like fields: `0.0`
   - level-like fields: carry forward last observed value
   - target future values are never taken from realized future rows

3. `split_windows()` now uses window end dates for safe assignment:
   - train: prediction end must be `<= train_end`
   - validation: prediction start must be `> train_end` and prediction end `<= val_end`
   - test: prediction start must be `> val_end`

4. Boundary-crossing windows are dropped instead of silently leaking across splits.

5. CoT teaching pools are now non-leaking:
   - validation CoT examples can only use training truths
   - test CoT examples can only use training + validation truths

### 20.4 Files changed

Primary fixes:

1. `src/data/windows.py`
2. `src/run_experiment.py`

Callers updated to consume `WindowMetadata`:

1. `scripts/ablate_autoformer_features.py`
2. `scripts/plot_model_predictions.py`
3. `scripts/compare_sentiment_series_mse.py`

### 20.5 Post-fix split audit

Using the fixed pipeline on `Data_auto` with `target.mode=returns`:

1. Train windows: `3519`
2. Validation windows: `226`
3. Test windows: `382`
4. Dropped boundary-crossing windows: `58`

No split integrity violations were found after the patch:

1. train end violations: `0`
2. validation start violations: `0`
3. validation end violations: `0`
4. test start violations: `0`

### 20.6 Clean rerun command

```bash
python scripts/run_automated_pipeline_once.py \
  --config src/config/default.yaml \
  --data-dir Data_auto \
  --seed-data-dir Data_auto \
  --skip-bootstrap \
  --target-mode returns \
  --cot-end-to-end \
  --cot-method cot_rf \
  --llm-max-samples 100000 \
  --llm-api-key deo \
  --no-paper-snapshot
```

### 20.7 Clean rerun result

Completed run:

1. `runs/20260228_034518_1810b8`

Corrected four-option comparison:

1. `Option1_Autoformer`
   - model: `tsm`
   - path MSE: `21.197815`
   - `h1=1.394445`, `h5=6.723088`, `h20=26.516499`, `h30=42.091316`

2. `Option2_CoT_RF`
   - model: `TSM+LLM-COT-RF`
   - path MSE: `25.312672`
   - `h1=1.558107`, `h5=6.995776`, `h20=32.611543`, `h30=54.045942`

3. `Option4_CoT_RAMP`
   - model: `TSM+LLM-COT-RF_blend_ramp_bestval_path_h1base`
   - path MSE: `21.450275`
   - `h1=1.394445`, `h5=6.679678`, `h20=27.101953`, `h30=43.547563`

Relevant files:

1. `runs/20260228_034518_1810b8/results/path_metrics.csv`
2. `runs/20260228_034518_1810b8/results/metrics_by_horizon.csv`
3. `runs/20260228_034518_1810b8/results/four_option_comparison_with_cot_ramp.csv`

### 20.8 Interpretation

After removing leakage from the main pipeline:

1. Base `tsm` beats raw `TSM+LLM-COT-RF` by a large margin.
2. Base `tsm` also beats the old “working” CoT ramp blend.
3. The blend still slightly improves horizon 5 MSE versus base `tsm`, but it loses on path MSE and on the longer horizons that dominate the full-path score.

Path-MSE deltas versus base `tsm`:

1. `TSM+LLM-COT-RF`: `+4.114857` (`+19.41%`)
2. `TSM+LLM-COT-RF_blend_ramp_bestval_path_h1base`: `+0.252460` (`+1.19%`)

Conclusion:

1. The previous “CoT blend is best overall” result was materially inflated by leakage.
2. With contamination removed, the honest winner in the main pipeline is base `tsm`.
3. CoT/ramp is now an exploratory alternative, not the production default.

### 20.9 Before-vs-after comparison

Last contaminated uncapped run:

1. `runs/20260227_192949_7d0e17`

Path MSE comparison:

1. Old contaminated `tsm`: `15.294339`
2. Old contaminated `TSM+LLM-COT-RF`: `13.388702`
3. Old contaminated `TSM+LLM-COT-RF_blend_ramp_bestval_path_h1base`: `13.157524`

Versus the corrected run:

1. Clean `tsm`: `21.197815`
2. Clean `TSM+LLM-COT-RF`: `25.312672`
3. Clean `TSM+LLM-COT-RF_blend_ramp_bestval_path_h1base`: `21.450275`

This is the key result:

1. The old pipeline made the CoT variants look substantially better than base `tsm`.
2. After leakage was removed, that ranking flipped.
3. The apparent CoT advantage was therefore not robust and cannot be cited as a valid main-pipeline result.

### 20.10 Regression guards

Added:

1. `tests/test_leakage_guards.py`

The tests cover:

1. Decoder future slots do not reuse realized future rows.
2. Split assignment drops boundary-crossing windows instead of leaking them across train/validation/test.

Validation command:

```bash
pytest -q tests/test_leakage_guards.py
```

Result:

1. `2 passed`

## 21. Post-Leakage Retuning

### 21.1 Goal

After fixing leakage, the old hyperparameters were no longer trustworthy.

The next step was therefore:

1. Re-tune the base `tsm` on validation only
2. Re-tune the `TSM+LLM-COT-RF` stack on validation only
3. Use the test split only once at the end for the final comparison

### 21.2 Additional fix discovered during retuning

The main experiment runner was not actually honoring `model.batch_size`.

`src/run_experiment.py` used hard-coded `batch_size=32` inside the TSM training loaders. This was corrected so the retune and final run both use the configured batch size.

### 21.3 Retuning script

Added:

1. `scripts/tune_clean_tsm_cot_rf.py`

This script:

1. Searches a small clean TSM hyperparameter grid on the validation split only
2. Refits the best validation TSM candidate
3. Searches CoT-RF prompt/example settings on a spaced validation subset
4. Writes the selected clean config for a final test-only run

### 21.4 Validation tuning outputs

Output root:

1. `reports/clean_retune/20260228_125350/`

Key files:

1. `reports/clean_retune/20260228_125350/tsm_search_results.csv`
2. `reports/clean_retune/20260228_125350/cot_search_results.csv`
3. `reports/clean_retune/20260228_125350/selection_summary.json`
4. `reports/clean_retune/20260228_125350/best_clean_tuned_config.yaml`
5. `reports/clean_retune/20260228_125350/final_test_comparison.csv`

### 21.5 TSM validation search result

Candidates were selected by validation `path_mse`, not by test.

Best validation candidate:

1. `af_s90_l30_m64_ff128_do030_lr5e4_wd1e2`
2. Validation `path_mse = 38.563675`
3. Test snapshot during tuning: `20.990627`

Important discipline point:

1. Another candidate had a slightly lower test snapshot (`20.722113`)
2. It was not selected
3. Selection stayed locked to validation only

### 21.6 CoT validation tuning result

CoT candidate search used:

1. A spaced validation subset of `40` samples
2. Similarity-based and high-error example selection variants

Result:

1. All tested CoT variants were worse than the best validation TSM
2. Best CoT validation result: `path_mse = 39.225579`
3. Best blend strength on validation: `w = 0.0`

Meaning:

1. The tuned Option-4 blend selected no LLM contribution at all
2. The honest tuned CoT stack collapses to base `tsm`

### 21.7 Final clean comparison

Selected config promoted to:

1. `src/config/retuned_clean_cot_rf.yaml`

Final test run:

1. `runs/20260228_133323_5e482d`

The run was intentionally stopped after the raw test CoT metric was produced. We did not need to spend additional time rerunning validation blend selection inside the main pipeline because the separate clean tuning stage had already selected `w = 0.0`.

Final held-out comparison:

1. Retuned base `tsm`
   - `path_mse = 20.990627`
   - `h1 = 1.388325`
   - `h5 = 6.432295`
   - `h20 = 26.410645`
   - `h30 = 41.794662`

2. Retuned raw `TSM+LLM-COT-RF`
   - `path_mse = 23.441910`
   - `h1 = 1.514101`
   - `h5 = 6.631292`
   - `h20 = 29.401695`
   - `h30 = 49.218285`

3. Retuned CoT ramp blend
   - Validation-selected `w = 0.0`
   - Therefore identical to retuned base `tsm`
   - `path_mse = 20.990627`

### 21.8 Interpretation

After leakage removal and clean retuning:

1. Base `tsm` improved from `21.197815` to `20.990627`
2. Raw `TSM+LLM-COT-RF` remained clearly worse than base `tsm`
3. The tuned CoT blend chose zero LLM weight on validation
4. So the honest post-retune production choice is base `tsm`, not CoT

### 21.9 Practical conclusion

The project is still valid after leakage correction, but the claim changes:

1. The clean Autoformer-style TSM remains competitive
2. The leakage-era claim that CoT was the best model does not survive either clean evaluation or clean retuning
3. If CoT is kept, it should be presented as an exploratory overlay rather than the production default

## 22. Paper-Style Leakage-Free Sentiment Re-Run

After the clean `TSM+LLM-COT-RF` retune still failed to beat base `tsm`, we re-ran the closest paper-aligned sentiment-refinement variant on the leakage-fixed pipeline.

### 22.1 Config used

Config:

1. `src/config/paper_llm_cot_sent_rf_qwen_k5_h18_similarity_ctx_full.yaml`

Key paper-style settings:

1. `history_points = 18`
2. `k_examples = 5`
3. `example_selection = similarity`
4. `lookback_days = 365`
5. `retain_context = true`
6. Sentiment enabled from `data/news/daily_sentiment.csv`

The run used the same non-leaking windows and example-pool restrictions introduced in Section 20.

### 22.2 Clean run

Run:

1. `runs/20260228_145058_40d91f`

Command family:

1. `scripts/run_automated_pipeline_once.py`
2. `--config src/config/paper_llm_cot_sent_rf_qwen_k5_h18_similarity_ctx_full.yaml`
3. `--data-dir Data_auto`
4. `--seed-data-dir Data_auto`
5. `--skip-bootstrap`
6. `--target-mode returns`
7. `--cot-end-to-end`
8. `--cot-method cot_sent_rf`

### 22.3 Result

Clean same-run comparison:

1. `tsm`
   - `path_mse = 20.942024`
   - `h1 = 1.383144`
   - `h5 = 6.415215`
   - `h20 = 26.299204`
   - `h30 = 41.988529`

2. `TSM+LLM-COT-SENT-RF`
   - `path_mse = 20.604430`
   - `h1 = 1.433835`
   - `h5 = 6.455291`
   - `h20 = 25.792649`
   - `h30 = 41.087699`

Improvement versus same-run `tsm`:

1. Absolute path MSE improvement: `0.337594`
2. Relative path MSE improvement: `1.61%`

Comparison to the prior clean retuned base `tsm` from Section 21:

1. Retuned clean `tsm`: `20.990627`
2. Paper-style `TSM+LLM-COT-SENT-RF`: `20.604430`
3. Relative improvement: `1.84%`

This is also below the same-run `linear_ridge` baseline:

1. `linear_ridge path_mse = 20.738794`

### 22.4 Artifact status

The decisive raw test artifact was saved successfully:

1. `runs/20260228_145058_40d91f/predictions/TSM+LLM-COT-SENT-RF_pred_test_subset.npz`

The optional outer validation blend-grid phase was still pending when the run stopped responding on the first validation-side LLM request. Since the raw leakage-free test result had already been written, the process was interrupted intentionally rather than waiting indefinitely.

Important interpretation:

1. The raw paper-style `TSM+LLM-COT-SENT-RF` result is valid
2. It does not depend on any test-selected outer blend weight
3. What remains incomplete is only the extra validation-selected ramp-blend registration

### 22.5 Updated conclusion

This changes the post-leakage conclusion:

1. Clean `TSM+LLM-COT-RF` still does not work well
2. But the paper-style sentiment refinement variant does recover a real, leakage-free edge
3. The edge is modest, not dramatic
4. The current best tested leakage-free TSM-family result is now raw `TSM+LLM-COT-SENT-RF`, not base `tsm`

### 22.6 Dedicated validation-only outer-blend tuning

To finish the paper-style method cleanly, we added a dedicated checkpoint-backed validator:

1. `scripts/tune_paper_cot_sent_validation.py`

This script:

1. Loads the exact checkpoint and datasets from `runs/20260228_145058_40d91f`
2. Recomputes train/val/test TSM forecasts from the saved checkpoint
3. Runs `TSM+LLM-COT-SENT-RF` on the full validation split only
4. Uses a strict train-only teaching pool for validation examples
5. Selects the outer ramp blend on validation
6. Applies that validation-selected choice to the saved raw test artifact from the same run

Tuning run:

1. `reports/paper_sent_validation_tune/20260228_155831`

Validation result:

1. Best outer blend weight: `w = 0.50`
2. Best validation variant: `cot_sent_rf_blend_ramp_w0.50_h1base`

Validation path MSEs:

1. `tsm`: `36.930645`
2. `cot_sent_rf_raw`: `36.898245`
3. `cot_sent_rf_blend_ramp_w0.50`: `36.867807`
4. `cot_sent_rf_blend_ramp_w0.50_h1base`: `36.867783`

Held-out test result after applying the validation-selected choice to the saved raw test artifact:

1. `tsm`: `20.942024`
2. `cot_sent_rf_raw`: `20.604430`
3. `cot_sent_rf_blend_ramp_w0.50`: `20.739130`
4. `cot_sent_rf_blend_ramp_w0.50_h1base`: `20.739025`

Interpretation:

1. The dedicated validation-only selection completed cleanly
2. Validation does prefer a modest outer blend
3. But that validation-selected outer blend is still worse on held-out test than raw `TSM+LLM-COT-SENT-RF`
4. Therefore the honest production recommendation remains raw `TSM+LLM-COT-SENT-RF`, not the extra outer blend
5. In hindsight, the earlier main-runner phase was more likely an observability problem than a hard deadlock; the equivalent validation-side work does complete when run through a dedicated checkpoint-backed script

### 22.7 Blend semantics cleanup

We then audited the blend paths directly in code.

Important correction:

1. `TSM+LLM-COT-SENT-RF` does **not** use the refiner's internal `llm.blend` path
2. The internal `llm.blend` path is only used by delta-style methods such as:
   - `TSM+LLM-COT-SENT-RF-DELTA`
   - `TSM+LLM-NORM-DELTA`
3. So the earlier ambiguity was not that raw `TSM+LLM-COT-SENT-RF` was being blended twice
4. The real ambiguity was that the config surface did not clearly distinguish:
   - methods that actually honor `llm.blend`
   - methods that ignore it
   - methods that can still receive an outer validation-selected blend grid

Code changes:

1. Added explicit blend-semantic helpers in `src/llm/refine.py`
2. Exported them via `src/llm/__init__.py`
3. `src/run_experiment.py` now:
   - logs when a method ignores `llm.blend`
   - logs when a method really does use internal blending
   - skips the outer blend-grid stage for methods that already use internal blending
4. Added regression coverage in `tests/test_blend_semantics.py`

Practical effect:

1. `TSM+LLM-COT-SENT-RF` remains a raw paper-style LLM forecast unless an outer blend is explicitly applied later
2. True double-blending is now blocked for methods that already use internal blend logic
3. Future experiment logs should be much clearer about which blend path is actually active

### 22.8 Automated runner integration

The automated runner now uses the dedicated CoT-SENT validation selector by default.

Implementation:

1. `scripts/run_automated_pipeline_once.py`

Behavior:

1. If `--cot-end-to-end --cot-method cot_sent_rf` is used, the runner now disables the in-run outer blend-grid stage for `TSM+LLM-COT-SENT-RF`
2. After the raw run completes, it launches `scripts/tune_paper_cot_sent_validation.py`
3. The selector writes under:
   - `runs/<run_id>/results/postrun_cot_sent_validation/`
4. The automated option-comparison builder can now read that post-run selection output directly

New CLI controls:

1. `--skip-postrun-cot-sent-validation`
2. `--force-postrun-cot-sent-validation`
3. `--postrun-cot-sent-output-dir`

Practical effect:

1. CoT-SENT selection is now reproducible from saved run artifacts
2. The main run no longer hides the expensive validation-side selection inside a long opaque stage
3. Raw `TSM+LLM-COT-SENT-RF` remains the honest recommended forecast unless later validation evidence changes that

### 22.9 CoT-SENT prompt hyperparameter screen

We then ran a focused stage-1 validation-subset screen for CoT-SENT prompt/example hyperparameters on the leakage-free setup, rather than spending full validation/test budget on every candidate.

Script:

1. `scripts/tune_paper_cot_sent_hparams.py`

Run:

1. `python scripts/tune_paper_cot_sent_hparams.py --base-run runs/20260228_145058_40d91f --llm-api-key deo`

Output root:

1. `reports/cot_sent_hparam_tune/20260228_164828/`

The run was intentionally stopped during the fourth candidate after the stage-1 screen had already produced enough signal to decide whether prompt retuning was promising.

Completed stage-1 subset results (`n=60`, validation only):

1. `paper_sim_k5_h18_lb365_ctx`: `37.044909`
2. `sim_k10_h18_lb365_ctx`: `38.301441`
3. `sim_k5_h24_lb365_ctx`: `51.625450`

Reference on the same subset:

1. raw `tsm`: `37.343967`

Interpretation:

1. The paper-style CoT-SENT setup remained the best of the completed prompt hyperparameter variants
2. Increasing teaching examples from `k=5` to `k=10` hurt
3. Increasing history/prompt length from `18` to `24` hurt badly
4. This is evidence that "more prompt context" is not the route to improvement here
5. Because the paper baseline remained on top and the remaining candidates were lower-priority variants, we stopped the sweep and moved budget to sentiment-signal improvement instead of continuing to full validation/test reruns

Conclusion from the screen:

1. Option 2 did not reveal a better prompt/example configuration than the paper baseline
2. The better next target was option 3: improving the sentiment input itself

### 22.10 CoT-SENT signal screen

We then ran a focused stage-1 validation-subset screen over sentiment-series candidates while keeping the CoT-SENT method fixed.

Script:

1. `scripts/tune_paper_cot_sent_signals.py`

Additional implementation work:

1. Added support for selecting a subset of signal candidates via `--signal-names`
2. Added `--stop-after-stage1` so we can run an honest screen without automatically paying full validation/test cost
3. Added engineered signal candidates derived from:
   - `sent_score`
   - `news_volume`
   - `sent_change`
4. Added a train-only engineered proxy candidate (`feat4_train_linear_proxy`) that fits a regularized linear projection on training-window next-day move targets only, then compresses it back to a bounded scalar signal

Run:

1. `python scripts/tune_paper_cot_sent_signals.py --base-run runs/20260228_145058_40d91f --llm-api-key deo --val-subset-samples 40 --signal-names baseline_daily_sentiment,qwen_votes3_daily3,qwen_votes3_daily3_event90_proxy,qwen_votes3_daily3_calib_regime_trainret,feat4_combo,feat4_train_linear_proxy --stop-after-stage1`

Output root:

1. `reports/cot_sent_signal_tune/20260228_171604/`

Reference on the same `n=40` validation subset:

1. raw `tsm`: `38.965942`

Completed screen results before the broader sweep terminated:

1. `baseline_daily_sentiment`: `39.233032`
2. `qwen_votes3_daily3`: `38.769094`

Interpretation:

1. The current paper sentiment file (`daily_sentiment.csv`) was slightly worse than raw `tsm` on this subset
2. Replacing it with `qwen_votes3_daily3` improved path MSE by `0.463938` versus the current baseline signal
3. `qwen_votes3_daily3` also beat raw `tsm` on the same subset by `0.196848`
4. So the sentiment input does matter, and the current baseline sentiment series is likely part of the performance ceiling

Operational note:

1. The broader signal sweep terminated after beginning `qwen_votes3_daily3_event90_proxy`
2. At termination, the run had already established the key finding above: `qwen_votes3_daily3` is a materially better sentiment-series candidate than `daily_sentiment.csv` on the leakage-free validation subset

Conclusion from the screen:

1. Option 3 is more promising than option 2
2. The next strict step should be a single deeper rerun using `qwen_votes3_daily3` as the CoT-SENT sentiment input on full validation/test, rather than more broad search

### 22.11 Deeper follow-up on top signal candidates

We then followed up on the top signal candidates in a stronger `n=60` validation-subset setting and completed the exact full validation/test continuations for both alternatives.

Additional tuning-script hardening:

1. Added `--timeout-seconds` to `scripts/tune_paper_cot_sent_signals.py`
2. Fixed an engineered-signal export bug for numpy-backed derived signals

#### 22.11.1 `qwen_votes3_daily3` follow-up

Run root:

1. `reports/cot_sent_signal_tune/qwen_votes3_daily3_full_20260228/`

Completed result:

1. Stage-1 validation subset (`n=60`): `35.765596`

Reference on the same `n=60` subset:

1. raw `tsm`: `37.343967`

Interpretation:

1. `qwen_votes3_daily3` improved on raw `tsm` by `1.578371` path MSE on the stronger subset
2. This is materially stronger evidence than the earlier `n=40` screen
3. It justified continuing into full validation and held-out test

Artifact:

1. `reports/cot_sent_signal_tune/qwen_votes3_daily3_full_20260228/stage1_subset_results.csv`

Exact full continuation results:

1. Validation (`n=226`): `36.986159`
2. Held-out test (`n=382`): `20.769153`

Additional artifacts:

1. `reports/cot_sent_signal_tune/qwen_votes3_daily3_full_20260228/stage2_full_val_results.csv`
2. `reports/cot_sent_signal_tune/qwen_votes3_daily3_full_20260228/stage3_test_result.csv`
3. `reports/cot_sent_signal_tune/qwen_votes3_daily3_full_20260228/final_test_comparison.csv`

Full-run interpretation:

1. `qwen_votes3_daily3` looked better on subset screens
2. It did not hold up on the exact held-out test
3. It finished worse than the baseline paper signal by `0.164723` path MSE on test

#### 22.11.2 `feat4_train_linear_proxy` follow-up

Run:

1. `python scripts/tune_paper_cot_sent_signals.py --base-run runs/20260228_145058_40d91f --llm-api-key deo --signal-names feat4_train_linear_proxy --val-subset-samples 60 --stop-after-stage1 --output-dir reports/cot_sent_signal_tune/feat4_train_linear_proxy_stage1_20260228 --timeout-seconds 30`

Run root:

1. `reports/cot_sent_signal_tune/feat4_train_linear_proxy_stage1_20260228/`

Completed result:

1. Stage-1 validation subset (`n=60`): `36.383526`

Reference on the same `n=60` subset:

1. raw `tsm`: `37.343967`
2. `qwen_votes3_daily3`: `35.765596`

Interpretation:

1. The train-only engineered proxy is better than raw `tsm`
2. But it is still worse than `qwen_votes3_daily3` by `0.617929` path MSE on the same subset
3. So among the stronger subset follow-up runs, `qwen_votes3_daily3` remained the best signal candidate

Artifact:

1. `reports/cot_sent_signal_tune/feat4_train_linear_proxy_stage1_20260228/stage1_subset_results.csv`

Exact full continuation results:

1. Validation (`n=226`): `37.328854`
2. Held-out test (`n=382`): `20.794578`

Additional artifacts:

1. `reports/cot_sent_signal_tune/feat4_train_linear_proxy_full_20260228/stage2_full_val_results.csv`
2. `reports/cot_sent_signal_tune/feat4_train_linear_proxy_full_20260228/stage3_test_result.csv`
3. `reports/cot_sent_signal_tune/feat4_train_linear_proxy_full_20260228/final_test_comparison.csv`

Full-run interpretation:

1. `feat4_train_linear_proxy` also failed to hold up on the exact held-out test
2. It finished worse than the baseline paper signal by `0.190148` path MSE on test
3. It also finished slightly worse than `qwen_votes3_daily3` by `0.025425` path MSE on test

Final conclusion:

1. The most promising CoT-SENT improvement path is still signal choice, not prompt retuning
2. But neither alternative signal generalized better than the baseline paper signal once evaluated on the exact held-out test
3. Final held-out test ranking is:
   - baseline paper signal: `20.604430`
   - `qwen_votes3_daily3`: `20.769153`
   - `feat4_train_linear_proxy`: `20.794578`
4. So the original baseline paper signal remains the best tested CoT-SENT signal in this project

### 22.12 Strategy 1 larger-model LLM endpoint test

Goal:

1. Test Strategy 1 from `docs/COT_SENT_IMPROVEMENT_STRATEGIES.md`
2. Replace the existing local CoT-SENT LLM with the larger endpoint at `http://192.168.1.140:9881/v1`
3. Keep the baseline paper signal fixed so the only material change is the LLM

Endpoint:

1. Base URL: `http://192.168.1.140:9881/v1`
2. API key: `deo`
3. Model reported by `/models`: `qwen3.5-35b-a3b-ud-q4-k-xl`

Run:

1. `python -u scripts/tune_paper_cot_sent_signals.py --base-run runs/20260228_145058_40d91f --llm-api-key deo --llm-base-url http://192.168.1.140:9881/v1 --llm-model qwen3.5-35b-a3b-ud-q4-k-xl --signal-names baseline_daily_sentiment --output-dir reports/cot_sent_signal_tune/strategy1_large_model_baseline_signal_20260301 --timeout-seconds 60`

Artifacts:

1. `reports/cot_sent_signal_tune/strategy1_large_model_baseline_signal_20260301/stage1_subset_results.csv`
2. `reports/cot_sent_signal_tune/strategy1_large_model_baseline_signal_20260301/stage2_full_val_results.csv`
3. `reports/cot_sent_signal_tune/strategy1_large_model_baseline_signal_20260301/stage3_test_result.csv`
4. `reports/cot_sent_signal_tune/strategy1_large_model_baseline_signal_20260301/final_test_comparison.csv`
5. `reports/cot_sent_signal_tune/strategy1_large_model_baseline_signal_20260301/selection_summary.json`

Results:

1. Stage-1 validation subset (`n=60`): `35.796020`
2. Full validation (`n=226`): `36.930645`
3. Held-out test (`n=382`): `20.725666`

Comparison to the current clean baseline CoT-SENT:

1. Baseline paper signal on prior endpoint: `20.604430`
2. Strategy 1 larger-model result: `20.725666`
3. Difference: `+0.121237` path MSE worse

Observed failure mode:

1. The larger endpoint frequently returned empty reflection content
2. Logged metadata showed `success=false`, `reflect_attempts=3`, `apply_attempts=0`, `fallback=true`
3. Error text: `Empty rules_text from reflection stage`
4. This means the pipeline often fell back to the base TSM forecast instead of applying a usable CoT-SENT refinement

Conclusion:

1. Strategy 1 did not materially improve performance
2. On the exact held-out test it was worse than the current clean baseline
3. The larger model is not production-worthy in the current prompt/endpoint configuration because it fails too often at the reflection stage

#### 22.12.1 Strategy 1 rerun after switching the large model to instruct mode

Goal:

1. Re-test the same larger endpoint after switching the served model into instruct mode
2. Check whether the previous failure was caused by response formatting rather than model capability

Run:

1. `python -u scripts/tune_paper_cot_sent_signals.py --base-run runs/20260228_145058_40d91f --llm-api-key deo --llm-base-url http://192.168.1.140:9881/v1 --llm-model qwen3.5-35b-a3b-ud-q4-k-xl --signal-names baseline_daily_sentiment --output-dir reports/cot_sent_signal_tune/strategy1_large_model_instruct_baseline_signal_20260301 --timeout-seconds 60`

Observed behavior:

1. Instruct mode did fix the previous empty-reflection failure
2. Reflection calls returned concrete rules with `success=true`
3. Apply calls also returned structured `{"yhat": [...]}` outputs with `success=true`
4. So this rerun did test the model's actual CoT-SENT behavior rather than fallback-to-TSM behavior

Stage-1 result:

1. Validation subset (`n=60`): `185.288195`

Comparison:

1. Current clean baseline CoT-SENT stage-1 reference: `37.044909`
2. Prior larger-model non-instruct stage-1 result: `35.796020`
3. Instruct-mode larger-model stage-1 result: `185.288195`

Interpretation:

1. Instruct mode fixed the response-format issue
2. But once the model was actually allowed to apply its reasoning, the resulting forecasts became catastrophically worse
3. This indicates the problem is not just parsing or endpoint compatibility; the model's generated corrections are too aggressive and unstable under the current prompt design

Run disposition:

1. The full validation/test continuation was intentionally stopped after the stage-1 result
2. Continuing a candidate that is ~5x worse than the clean subset baseline would waste compute without changing the decision

### 22.13 Strategy 2 structured horizon-delta rerun

Goal:

1. Replace the free-form 30-step rewrite with bounded percentage adjustments at key horizons only
2. Test whether the larger instruct model becomes usable once its output space is constrained

Implementation:

1. Added a separate method: `TSM+LLM-COT-SENT-RF-HDELTA`
2. Reflection stage remains the same
3. Apply stage now returns structured JSON:
   - `{"adjustments": {"h1": ..., "h5": ..., "h20": ..., "h30": ...}}`
4. The four horizon adjustments are linearly interpolated across the 30-day path
5. Adjustments are clamped to a configurable max absolute percentage before being applied to the TSM path

Code changes:

1. `src/llm/prompts.py`
2. `src/llm/refine.py`
3. `src/run_experiment.py`
4. `scripts/tune_paper_cot_sent_hparams.py`
5. `scripts/tune_paper_cot_sent_signals.py`
6. `tests/test_hdelta_semantics.py`

Validation:

1. `python -m py_compile src/llm/prompts.py src/llm/refine.py src/run_experiment.py scripts/tune_paper_cot_sent_hparams.py scripts/tune_paper_cot_sent_signals.py`
2. `pytest -q tests/test_hdelta_semantics.py tests/test_blend_semantics.py tests/test_leakage_guards.py`
3. Result: `6 passed`

Primary run:

1. `python -u scripts/tune_paper_cot_sent_signals.py --base-run runs/20260228_145058_40d91f --llm-api-key deo --llm-base-url http://192.168.1.140:9881/v1 --llm-model qwen3.5-35b-a3b-ud-q4-k-xl --signal-names baseline_daily_sentiment --method TSM+LLM-COT-SENT-RF-HDELTA --hdelta-max-adjustment-pct 3.0 --output-dir reports/cot_sent_signal_tune/strategy2_hdelta_large_model_stage1_20260301 --timeout-seconds 60 --stop-after-stage1`

Artifacts:

1. `reports/cot_sent_signal_tune/strategy2_hdelta_large_model_stage1_20260301/stage1_subset_results.csv`
2. `reports/cot_sent_signal_tune/strategy2_hdelta_large_model_stage1_20260301/selection_summary.json`

Result:

1. Stage-1 validation subset (`n=60`): `41.385346`

Comparison:

1. Clean baseline CoT-SENT stage-1 reference: `37.044909`
2. Strategy 2 HDELTA on large instruct model: `41.385346`
3. Difference: `+4.340437` path MSE worse

Interpretation:

1. Strategy 2 fixed the architectural failure mode from Strategy 1 instruct-mode rerun
2. The large model now returned valid structured horizon adjustments rather than unstable full-path rewrites
3. The result is no longer catastrophic (`41.39` instead of `185.29`)
4. But it still does not beat the clean baseline, so it was not carried into full validation/test

Additional note:

1. A same-day 4B control rerun for HDELTA was attempted against `http://192.168.1.140:9877/v1`
2. That endpoint was down (`Connection refused`)
3. So no fair small-model HDELTA comparison was available during this run

#### 22.13.1 Strategy 2 HDELTA rerun on restored 4B endpoint

Goal:

1. Run the same Strategy 2 HDELTA stage-1 evaluation on the restored 4B endpoint
2. Compare 4B vs 35B on the same structured-output method

Endpoint:

1. Base URL: `http://192.168.1.140:9877/v1`
2. Model: `qwen3-vl-4b-gpu`
3. API key: `deo`

Run:

1. `python -u scripts/tune_paper_cot_sent_signals.py --base-run runs/20260228_145058_40d91f --llm-api-key deo --llm-base-url http://192.168.1.140:9877/v1 --llm-model qwen3-vl-4b-gpu --signal-names baseline_daily_sentiment --method TSM+LLM-COT-SENT-RF-HDELTA --hdelta-max-adjustment-pct 3.0 --output-dir reports/cot_sent_signal_tune/strategy2_hdelta_small_model_stage1_20260301 --timeout-seconds 60 --stop-after-stage1`

Artifacts:

1. `reports/cot_sent_signal_tune/strategy2_hdelta_small_model_stage1_20260301/stage1_subset_results.csv`
2. `reports/cot_sent_signal_tune/strategy2_hdelta_small_model_stage1_20260301/selection_summary.json`

Result:

1. Stage-1 validation subset (`n=60`): `35.322736`

Comparison:

1. Clean baseline CoT-SENT stage-1 reference: `37.044909`
2. Strategy 2 HDELTA on 4B model: `35.322736`
3. Strategy 2 HDELTA on 35B instruct model: `41.385346`

Interpretation:

1. The same structured-output Strategy 2 method is materially better on the 4B model than on the 35B instruct model
2. The 4B HDELTA run also beats the clean baseline stage-1 reference by `1.722173` path MSE
3. This strengthens the diagnosis that the larger model is a poor fit for the current prompt/config semantics, not that Strategy 2 is intrinsically wrong

Full continuation:

1. `python -u scripts/tune_paper_cot_sent_signals.py --base-run runs/20260228_145058_40d91f --llm-api-key deo --llm-base-url http://192.168.1.140:9877/v1 --llm-model qwen3-vl-4b-gpu --signal-names baseline_daily_sentiment --method TSM+LLM-COT-SENT-RF-HDELTA --hdelta-max-adjustment-pct 3.0 --output-dir reports/cot_sent_signal_tune/strategy2_hdelta_small_model_full_20260301 --timeout-seconds 60`

Additional artifacts:

1. `reports/cot_sent_signal_tune/strategy2_hdelta_small_model_full_20260301/stage2_full_val_results.csv`
2. `reports/cot_sent_signal_tune/strategy2_hdelta_small_model_full_20260301/stage3_test_result.csv`
3. `reports/cot_sent_signal_tune/strategy2_hdelta_small_model_full_20260301/final_test_comparison.csv`
4. `reports/cot_sent_signal_tune/strategy2_hdelta_small_model_full_20260301/selection_summary.json`

Full-run results:

1. Stage-1 validation subset (`n=60`): `35.313688`
2. Full validation (`n=226`): `36.522435`
3. Held-out test (`n=382`): `20.618083`

Comparison to current clean baseline CoT-SENT:

1. Baseline paper signal held-out test: `20.604430`
2. Strategy 2 HDELTA on 4B model held-out test: `20.618083`
3. Difference: `+0.013654` path MSE worse

Interpretation:

1. Strategy 2 HDELTA on the 4B model survives full validation/test as a near-tie with the current clean baseline
2. It is not the new winner on held-out test
3. But it is the first structured-output variant that remains credible all the way through the full run
4. The full-path rewrite variant remains architecturally inferior, while HDELTA is now a defensible base for further config work

### 22.14 HDELTA config-improvement sweep

Goal:

1. Implement the first five HDELTA config improvements directly in the tuning path
2. Test whether any of them improve the 4B HDELTA stage-1 result before spending full validation/test budget

Implemented knobs:

1. Strict JSON reflection for HDELTA
2. `retain_context` override
3. Lower HDELTA percentage caps
4. Frozen horizons (used for `h1=0.0`)
5. Prompt mode where sentiment is explicitly secondary to price action

Code changes:

1. `src/llm/prompts.py`
2. `src/llm/refine.py`
3. `scripts/tune_paper_cot_sent_signals.py`
4. `tests/test_hdelta_semantics.py`

Validation:

1. `python -m py_compile src/llm/prompts.py src/llm/refine.py scripts/tune_paper_cot_sent_signals.py`
2. `pytest -q tests/test_hdelta_semantics.py tests/test_blend_semantics.py tests/test_leakage_guards.py`
3. Result: `7 passed`

Reference:

1. Current 4B HDELTA stage-1 baseline: `35.313688`

Sweep runs:

1. `strict_only`
   - `--cot-strict-json-prompt --hdelta-max-adjustment-pct 3.0`
2. `full_guard_1p5`
   - `--cot-strict-json-prompt --retain-context-override false --hdelta-max-adjustment-pct 1.5 --hdelta-freeze-horizons 1 --hdelta-sentiment-secondary`
3. `full_guard_1p0`
   - `--cot-strict-json-prompt --retain-context-override false --hdelta-max-adjustment-pct 1.0 --hdelta-freeze-horizons 1 --hdelta-sentiment-secondary`
4. `noctx_guard_1p5`
   - `--retain-context-override false --hdelta-max-adjustment-pct 1.5 --hdelta-freeze-horizons 1 --hdelta-sentiment-secondary`

Results (`n=60` stage-1 subset):

1. `strict_only`: `35.461333`
2. `full_guard_1p0`: `35.624509`
3. `noctx_guard_1p5`: `35.670058`
4. `full_guard_1p5`: `35.702872`
5. Baseline 4B HDELTA: `35.313688`

Interpretation:

1. None of the first-pass config changes improved on the current 4B HDELTA baseline
2. Strict JSON reflection alone was the least harmful change, but still slightly worse than baseline
3. Turning off context and tightening the cap did not help in this first sweep
4. Freezing `h1` and making sentiment secondary remain architecturally sensible, but they did not improve the subset metric under the current prompt semantics

Decision:

1. Do not promote any of the new config variants yet
2. Keep the current 4B HDELTA baseline as the best structured-output configuration tested so far

### 22.15 Strategy 4 error-stratified teaching examples

Goal:

1. Replace the regime-similarity example selector with a diversity-on-error selector
2. Test whether more varied forecast-error archetypes improve HDELTA refinement quality

Implementation:

1. Added `error_stratified` as a supported `example_selection` mode in `scripts/tune_paper_cot_sent_hparams.py`
2. Each candidate example window now has a 4D error profile:
   - signed errors at `h1`, `h5`, `h20`, `h30`
3. Selection logic:
   - include one near-perfect example (minimum normalized error norm)
   - choose remaining examples by farthest-first traversal in normalized error-profile space
   - keep the final set sorted by date
4. Added `--example-selection-override` to `scripts/tune_paper_cot_sent_signals.py`

Validation:

1. `python -m py_compile scripts/tune_paper_cot_sent_hparams.py scripts/tune_paper_cot_sent_signals.py`
2. `pytest -q tests/test_hdelta_semantics.py tests/test_blend_semantics.py tests/test_leakage_guards.py`
3. Result: `7 passed`

Run:

1. `python -u scripts/tune_paper_cot_sent_signals.py --base-run runs/20260228_145058_40d91f --llm-api-key deo --llm-base-url http://192.168.1.140:9877/v1 --llm-model qwen3-vl-4b-gpu --signal-names baseline_daily_sentiment --method TSM+LLM-COT-SENT-RF-HDELTA --hdelta-max-adjustment-pct 3.0 --example-selection-override error_stratified --output-dir reports/cot_sent_signal_tune/error_stratified_hdelta_stage1_20260301 --timeout-seconds 60 --stop-after-stage1`

Artifacts:

1. `reports/cot_sent_signal_tune/error_stratified_hdelta_stage1_20260301/stage1_subset_results.csv`
2. `reports/cot_sent_signal_tune/error_stratified_hdelta_stage1_20260301/selection_summary.json`

Result:

1. Stage-1 validation subset (`n=60`): `35.529845`

Comparison:

1. Current 4B HDELTA stage-1 baseline: `35.313688`
2. Error-stratified HDELTA stage-1: `35.529845`
3. Difference: `+0.216157` path MSE worse

Interpretation:

1. The selector did change the teaching set substantially and increased example diversity
2. But it did not improve the stage-1 metric
3. So error-stratified selection is not promoted past subset screening yet

### 22.16 Strategy 5 lightweight learned post-hoc calibration layer

Goal:

1. Fit a simple per-horizon ridge combiner on validation predictions
2. Test whether a learned calibration layer can extract more value from the 4B HDELTA forecast than the raw HDELTA path

Implementation:

1. Added `src/eval/meta_blend.py`
2. Added `scripts/evaluate_meta_blend.py`
3. The blender fits, for each horizon, a ridge regression on:
   - `tsm_pred_h`
   - `llm_pred_h`
   - intercept
4. Regularization is selected via blocked validation CV

Validation:

1. `python -m py_compile src/eval/meta_blend.py scripts/evaluate_meta_blend.py`
2. `pytest -q tests/test_meta_blend.py tests/test_hdelta_semantics.py tests/test_blend_semantics.py tests/test_leakage_guards.py`
3. Result: `8 passed`

Run:

1. `python scripts/evaluate_meta_blend.py --run-dir reports/cot_sent_signal_tune/strategy2_hdelta_small_model_full_20260301 --output-dir reports/cot_sent_signal_tune/strategy2_hdelta_small_model_full_20260301/meta_blend_eval`

Artifacts:

1. `reports/cot_sent_signal_tune/strategy2_hdelta_small_model_full_20260301/meta_blend_eval/summary.json`
2. `reports/cot_sent_signal_tune/strategy2_hdelta_small_model_full_20260301/meta_blend_eval/val_comparison.csv`
3. `reports/cot_sent_signal_tune/strategy2_hdelta_small_model_full_20260301/meta_blend_eval/test_comparison.csv`
4. `reports/cot_sent_signal_tune/strategy2_hdelta_small_model_full_20260301/meta_blend_eval/blend_coefficients.csv`

Results:

Validation:

1. `tsm`: `36.930645`
2. `llm_hdelta`: `36.522435`
3. `meta_blend`: `26.333310`

Held-out test:

1. `tsm`: `20.725666`
2. `llm_hdelta`: `20.618083`
3. `meta_blend`: `20.688997`

Interpretation:

1. The learned calibration layer strongly overfit validation
2. On held-out test it improved over raw `tsm`
3. But it did not beat raw `llm_hdelta`
4. So the current simple ridge blender is not the new best model

#### 22.16.1 Stateful meta-blend with volatility/sentiment/divergence features

Goal:

1. Upgrade the simple per-horizon ridge blender into a state-aware combiner
2. Condition the blend on recent market state rather than only raw `tsm` and `llm_hdelta`

Implementation:

1. Extended `src/eval/meta_blend.py` with `StatefulMetaBlender`
2. Added state features derived from the clean base run:
   - `vol20`: 20-day rolling volatility of log returns
   - `sent3`: 3-day moving average of daily sentiment
   - `div`: absolute TSM-vs-LLM divergence (per horizon)
3. The stateful design matrix includes:
   - `tsm`
   - `llm`
   - `div`
   - `vol20`
   - `sent3`
   - interaction terms between `tsm/llm` and `vol20/sent3/div`
   - intercept
4. Regularization is still selected with blocked CV on validation

Validation:

1. `python -m py_compile src/eval/meta_blend.py scripts/evaluate_meta_blend.py`
2. `pytest -q tests/test_meta_blend.py tests/test_hdelta_semantics.py tests/test_blend_semantics.py tests/test_leakage_guards.py`
3. Result: `9 passed`

Run:

1. `python scripts/evaluate_meta_blend.py --run-dir reports/cot_sent_signal_tune/strategy2_hdelta_small_model_full_20260301 --output-dir reports/cot_sent_signal_tune/strategy2_hdelta_small_model_full_20260301/meta_blend_eval`

Additional artifacts:

1. `reports/cot_sent_signal_tune/strategy2_hdelta_small_model_full_20260301/meta_blend_eval/stateful_blend_coefficients.csv`
2. `reports/cot_sent_signal_tune/strategy2_hdelta_small_model_full_20260301/meta_blend_eval/test_meta_blend_predictions.npz`

Results:

Validation:

1. `stateful_meta_blend`: `24.184899`

Held-out test:

1. `stateful_meta_blend`: `23.265203`

Interpretation:

1. The state-aware blender overfit validation even more severely than the simple ridge combiner
2. It materially degraded held-out test performance
3. So the current stateful calibration layer should be rejected

#### 22.16.2 35B rerun of the HDELTA improvement stack

Goal:

1. Check whether the later HDELTA improvements were unfairly optimized around the 4B model
2. Re-run the same improvement path on the 35B instruct endpoint to see if the larger model becomes competitive under safer constraints

Endpoint:

1. Base URL: `http://192.168.1.140:9881/v1`
2. Model: `qwen3.5-35b-a3b-ud-q4-k-xl`
3. API key: `deo`

Stage-1 config sweep on 35B (`n=60`):

1. `strict_only`: `41.540398`
2. `full_guard_1p0`: `36.881143`
3. `noctx_guard_1p5`: `37.875709`
4. `full_guard_1p5`: `37.649152`
5. `error_stratified`: `37.620271`

Interpretation:

1. The naive 35B strict-only variant remained bad
2. The same guardrail package that helped conceptually on 4B helped much more on 35B:
   - strict structured reflection
   - `retain_context=false`
   - `h1` frozen
   - `1.0%` cap
   - sentiment treated as secondary
3. `full_guard_1p0` was the only 35B variant worth carrying to full validation/test

Runs:

1. `python -u scripts/tune_paper_cot_sent_signals.py --base-run runs/20260228_145058_40d91f --llm-api-key deo --llm-base-url http://192.168.1.140:9881/v1 --llm-model qwen3.5-35b-a3b-ud-q4-k-xl --signal-names baseline_daily_sentiment --method TSM+LLM-COT-SENT-RF-HDELTA --hdelta-max-adjustment-pct 3.0 --cot-strict-json-prompt --output-dir reports/cot_sent_signal_tune/config_sweep_35b_20260301/strict_only --timeout-seconds 60 --stop-after-stage1`
2. `python -u scripts/tune_paper_cot_sent_signals.py --base-run runs/20260228_145058_40d91f --llm-api-key deo --llm-base-url http://192.168.1.140:9881/v1 --llm-model qwen3.5-35b-a3b-ud-q4-k-xl --signal-names baseline_daily_sentiment --method TSM+LLM-COT-SENT-RF-HDELTA --hdelta-max-adjustment-pct 1.0 --cot-strict-json-prompt --retain-context-override false --hdelta-freeze-horizons 1 --hdelta-sentiment-secondary --output-dir reports/cot_sent_signal_tune/config_sweep_35b_20260301/full_guard_1p0 --timeout-seconds 60 --stop-after-stage1`
3. `python -u scripts/tune_paper_cot_sent_signals.py --base-run runs/20260228_145058_40d91f --llm-api-key deo --llm-base-url http://192.168.1.140:9881/v1 --llm-model qwen3.5-35b-a3b-ud-q4-k-xl --signal-names baseline_daily_sentiment --method TSM+LLM-COT-SENT-RF-HDELTA --hdelta-max-adjustment-pct 1.5 --retain-context-override false --hdelta-freeze-horizons 1 --hdelta-sentiment-secondary --output-dir reports/cot_sent_signal_tune/config_sweep_35b_20260301/noctx_guard_1p5 --timeout-seconds 60 --stop-after-stage1`
4. `python -u scripts/tune_paper_cot_sent_signals.py --base-run runs/20260228_145058_40d91f --llm-api-key deo --llm-base-url http://192.168.1.140:9881/v1 --llm-model qwen3.5-35b-a3b-ud-q4-k-xl --signal-names baseline_daily_sentiment --method TSM+LLM-COT-SENT-RF-HDELTA --hdelta-max-adjustment-pct 1.5 --cot-strict-json-prompt --retain-context-override false --hdelta-freeze-horizons 1 --hdelta-sentiment-secondary --output-dir reports/cot_sent_signal_tune/config_sweep_35b_20260301/full_guard_1p5 --timeout-seconds 60 --stop-after-stage1`
5. `python -u scripts/tune_paper_cot_sent_signals.py --base-run runs/20260228_145058_40d91f --llm-api-key deo --llm-base-url http://192.168.1.140:9881/v1 --llm-model qwen3.5-35b-a3b-ud-q4-k-xl --signal-names baseline_daily_sentiment --method TSM+LLM-COT-SENT-RF-HDELTA --hdelta-max-adjustment-pct 3.0 --example-selection-override error_stratified --output-dir reports/cot_sent_signal_tune/error_stratified_hdelta_35b_stage1_20260301 --timeout-seconds 60 --stop-after-stage1`

Artifacts:

1. `reports/cot_sent_signal_tune/config_sweep_35b_20260301/strict_only/stage1_subset_results.csv`
2. `reports/cot_sent_signal_tune/config_sweep_35b_20260301/full_guard_1p0/stage1_subset_results.csv`
3. `reports/cot_sent_signal_tune/config_sweep_35b_20260301/noctx_guard_1p5/stage1_subset_results.csv`
4. `reports/cot_sent_signal_tune/config_sweep_35b_20260301/full_guard_1p5/stage1_subset_results.csv`
5. `reports/cot_sent_signal_tune/error_stratified_hdelta_35b_stage1_20260301/stage1_subset_results.csv`

Best 35B full guarded run:

1. Run:
   `python -u scripts/tune_paper_cot_sent_signals.py --base-run runs/20260228_145058_40d91f --llm-api-key deo --llm-base-url http://192.168.1.140:9881/v1 --llm-model qwen3.5-35b-a3b-ud-q4-k-xl --signal-names baseline_daily_sentiment --method TSM+LLM-COT-SENT-RF-HDELTA --hdelta-max-adjustment-pct 1.0 --cot-strict-json-prompt --retain-context-override false --hdelta-freeze-horizons 1 --hdelta-sentiment-secondary --output-dir reports/cot_sent_signal_tune/strategy2_hdelta_large_model_full_guard_1p0_20260301 --timeout-seconds 60`
2. Validation path MSE: `37.742088`
3. Held-out test path MSE: `20.872258`

Comparison:

1. Baseline paper CoT-SENT: `20.604430`
2. 4B raw HDELTA: `20.618083`
3. 35B raw guarded HDELTA: `20.872258`

Interpretation:

1. The 35B model can be made operationally sane under HDELTA guardrails
2. Those guardrails materially improved the 35B path relative to the earlier 35B HDELTA screen (`41.385346` on stage 1)
3. Even after that fix, the raw 35B HDELTA path still lost to both the baseline paper CoT-SENT and the 4B raw HDELTA on held-out test

35B meta-blend follow-up:

1. Run:
   `python scripts/evaluate_meta_blend.py --run-dir reports/cot_sent_signal_tune/strategy2_hdelta_large_model_full_guard_1p0_20260301 --output-dir reports/cot_sent_signal_tune/strategy2_hdelta_large_model_full_guard_1p0_20260301/meta_blend_eval_35b`
2. Validation:
   - `tsm`: `36.930645`
   - `llm_hdelta`: `37.742088`
   - `meta_blend`: `27.218403`
   - `stateful_meta_blend`: `23.262615`
3. Held-out test:
   - `tsm`: `20.725666`
   - `llm_hdelta`: `20.872258`
   - `meta_blend`: `19.521435`
   - `stateful_meta_blend`: `30.517486`

Interpretation:

1. The simple ridge meta-blend on top of the guarded 35B run is the first 35B-based result that materially beats the current baseline stack
2. It improves over the paper CoT-SENT baseline by `1.082995` path MSE on held-out test
3. It also beats the 4B raw HDELTA by `1.096649`
4. The stateful blender remains unstable and should still be rejected

Current ranking after the 35B rerun:

1. 35B guarded HDELTA + simple meta-blend: `19.521435`
2. Baseline paper CoT-SENT: `20.604430`
3. 4B raw HDELTA: `20.618083`
4. Raw `tsm`: `20.725666`
5. 35B raw guarded HDELTA: `20.872258`
6. 35B stateful meta-blend: `30.517486`

Conclusion:

1. The earlier blanket conclusion that 35B was simply a bad fit was too crude
2. Raw 35B prompting still underperforms
3. But once the prompt is tightly constrained and the final combination is learned post hoc, the 35B path becomes the best held-out result seen so far
4. That makes the next technical question narrower: whether this 35B simple meta-blend result survives a stricter validation protocol such as rolling validation or nested blend fitting

#### 22.16.3 Strict outer rolling validation for meta-blend

Goal:

1. Re-test the promising meta-blend results under a stricter protocol
2. Avoid treating a one-shot full-validation blend fit as production-worthy if it cannot generalize across outer validation folds

Method:

1. Added `scripts/evaluate_meta_blend_strict.py`
2. Added `rolling_meta_blend_cv(...)` to `src/eval/meta_blend.py`
3. Outer protocol:
   - split validation sequentially into rolling outer folds
   - for each outer fold, fit the simple ridge meta-blend on prior validation rows only
   - choose alpha inside that fold using the existing blocked inner CV logic
   - score only on the held-out outer fold
4. After that outer-CV check, still report the old full-validation-fit test number for reference

Validation:

1. `python -m py_compile src/eval/meta_blend.py scripts/evaluate_meta_blend_strict.py`
2. `pytest -q tests/test_meta_blend.py tests/test_hdelta_semantics.py tests/test_blend_semantics.py tests/test_leakage_guards.py`
3. Result: `10 passed`

Runs:

1. `python scripts/evaluate_meta_blend_strict.py --run-dir reports/cot_sent_signal_tune/strategy2_hdelta_large_model_full_guard_1p0_20260301 --output-dir reports/cot_sent_signal_tune/strategy2_hdelta_large_model_full_guard_1p0_20260301/meta_blend_strict_eval_35b`
2. `python scripts/evaluate_meta_blend_strict.py --run-dir reports/cot_sent_signal_tune/strategy2_hdelta_small_model_full_20260301 --output-dir reports/cot_sent_signal_tune/strategy2_hdelta_small_model_full_20260301/meta_blend_strict_eval_4b`

Artifacts:

1. `reports/cot_sent_signal_tune/strategy2_hdelta_large_model_full_guard_1p0_20260301/meta_blend_strict_eval_35b/summary.json`
2. `reports/cot_sent_signal_tune/strategy2_hdelta_large_model_full_guard_1p0_20260301/meta_blend_strict_eval_35b/outer_cv_comparison.csv`
3. `reports/cot_sent_signal_tune/strategy2_hdelta_large_model_full_guard_1p0_20260301/meta_blend_strict_eval_35b/outer_fold_results.csv`
4. `reports/cot_sent_signal_tune/strategy2_hdelta_small_model_full_20260301/meta_blend_strict_eval_4b/summary.json`
5. `reports/cot_sent_signal_tune/strategy2_hdelta_small_model_full_20260301/meta_blend_strict_eval_4b/outer_cv_comparison.csv`
6. `reports/cot_sent_signal_tune/strategy2_hdelta_small_model_full_20260301/meta_blend_strict_eval_4b/outer_fold_results.csv`

Results: 35B guarded HDELTA

Outer rolling validation (`n=169` evaluable rows):

1. `tsm`: `44.363195`
2. `llm_hdelta`: `45.383946`
3. `meta_blend_outer_cv`: `59.073209`

Reference held-out test from the old full-validation-fit blend:

1. `tsm`: `20.725666`
2. `llm_hdelta`: `20.872258`
3. `meta_blend_fullval_fit`: `19.521435`

Results: 4B HDELTA

Outer rolling validation (`n=169` evaluable rows):

1. `tsm`: `44.363195`
2. `llm_hdelta`: `44.004717`
3. `meta_blend_outer_cv`: `57.966146`

Reference held-out test from the old full-validation-fit blend:

1. `tsm`: `20.725666`
2. `llm_hdelta`: `20.618083`
3. `meta_blend_fullval_fit`: `20.688997`

Interpretation:

1. The simple meta-blend fails the stricter outer rolling validation check on both 35B and 4B
2. It is not just a weak win; it is materially worse than the raw baselines when forced to generalize across outer validation folds
3. That means the attractive held-out test result for the 35B full-validation-fit blend is not stable enough to promote
4. The correct decision is to reject the current meta-blend as a production candidate despite the good one-shot test score

Corrected ranking after strict validation:

1. Baseline paper CoT-SENT: `20.604430`
2. 4B raw HDELTA: `20.618083`
3. Raw `tsm`: `20.725666`
4. 35B raw guarded HDELTA: `20.872258`
5. 35B full-validation-fit simple meta-blend: `19.521435` (`do not promote; rejected by strict outer validation`)
6. Stateful meta-blends: rejected

Conclusion:

1. The 35B guarded raw path is still useful as an exploratory branch, but not the best honest model
2. The simple meta-blend does not survive stricter validation discipline
3. So the best honest results remain the baseline paper CoT-SENT and the 4B raw HDELTA near-tie
4. The next improvement should not be more blending; it should move upstream into the forecasting inputs or architecture

#### 22.17 Strategy 3 sentiment-aware TSM input panel (partial run)

Goal:

1. Inject daily sentiment directly into the TSM input panel rather than relying on prompt-only sentiment interpretation
2. Refit a clean sentiment-aware TSM base and compare the tracked model options on that new base

Implementation:

1. Added sentiment feature merge utilities in `src/data/panel.py`
2. Added explicit preferred feature ordering so sentiment columns are guaranteed into the TSM feature set
3. Added `scripts/run_strategy3_sentiment_tsm.py` to:
   - build a sentiment-aware panel
   - retune/refit the clean TSM
   - save a reusable base run with checkpoint + panel
   - evaluate the tracked overlay methods on top of that base
4. Added tests in `tests/test_sentiment_panel_features.py`

Validation:

1. `python -m py_compile src/data/panel.py src/run_experiment.py src/frontend_live_bundle.py scripts/tune_clean_tsm_cot_rf.py scripts/tune_paper_cot_sent_hparams.py scripts/run_strategy3_sentiment_tsm.py`
2. `pytest -q tests/test_sentiment_panel_features.py tests/test_meta_blend.py tests/test_hdelta_semantics.py tests/test_blend_semantics.py tests/test_leakage_guards.py`
3. Result: `12 passed`

Feature verification:

The sentiment-aware TSM feature list was explicitly verified to contain:

1. `y_return`
2. `y`
3. `sent_score`
4. `sent_score_3d_ma`
5. `sent_score_7d_ma`
6. `sent_score_change_1d`

Run:

1. `python -u scripts/run_strategy3_sentiment_tsm.py --llm-api-key deo`
2. Output dir: `reports/strategy3_sentiment_tsm/20260301_201329`

TSM retune results:

1. Best candidate remained `af_s90_l30_m64_ff128_do030_lr5e4_wd1e2`
2. Best validation path MSE: `38.563675`
3. Best held-out test path MSE: `20.990627`

Interpretation:

1. Adding baseline daily sentiment features to the TSM input panel did not improve the clean TSM retune over the existing baseline
2. This was not a feature-selection bug; the sentiment features were present in the model input set

Completed overlay result:

1. `paper_cot_sent_4b_on_tsm_sent` held-out test path MSE: `21.148469`

Interpretation:

1. The strongest existing 4B paper CoT-SENT path became worse on top of the sentiment-aware TSM base
2. So baseline daily sentiment injection, in this first Strategy 3 form, did not help either the TSM or the paper CoT-SENT overlay

Runtime decision:

1. The remaining two full overlays (`4B raw HDELTA`, `35B raw guarded HDELTA`) were started but intentionally stopped before completion
2. Reason: the LLM overlay runtime on the new base was too high for a single turn and the completed results already showed no positive signal from the baseline daily sentiment feature injection

Partial artifact:

1. `reports/strategy3_sentiment_tsm/20260301_201329/partial_four_option_status.csv`
2. `reports/strategy3_sentiment_tsm/20260301_201329/partial_summary.json`

Partial Strategy 3 conclusion:

1. Baseline daily sentiment as direct TSM features did not improve the clean TSM benchmark
2. It also worsened the completed 4B paper CoT-SENT overlay
3. So Strategy 3 is not validated yet under the baseline daily sentiment series
4. If Strategy 3 is continued, the next rational variant is to inject a stronger sentiment series into the TSM panel rather than re-running the same baseline sentiment through more expensive overlays

#### 22.17.1 Strategy 3 variant: inject `qwen_votes3_daily3` into TSM inputs

Goal:

1. Retry Strategy 3 using the stronger LLM-derived daily sentiment series instead of the baseline `daily_sentiment.csv`
2. Keep the same four-option comparison set for fair comparison:
   - `base_tsm_sent`
   - `paper_cot_sent_4b_on_tsm_sent`
   - `hdelta_4b_on_tsm_sent`
   - `hdelta_35b_guarded_on_tsm_sent`

Run:

1. `python -u scripts/run_strategy3_sentiment_tsm.py --llm-api-key deo --sentiment-path data/news/daily_sentiment_qwen_votes3_daily3.csv`
2. Output dir: `reports/strategy3_sentiment_tsm/20260301_211719`

Verification:

1. Saved base run config points to `data/news/daily_sentiment_qwen_votes3_daily3.csv`
2. The injected sentiment columns were present in the actual TSM feature set:
   - `sent_score`
   - `sent_score_3d_ma`
   - `sent_score_7d_ma`
   - `sent_score_change_1d`

TSM retune result:

1. Best candidate remained `af_s90_l30_m64_ff128_do030_lr5e4_wd1e2`
2. Validation path MSE: `38.563675`
3. Held-out test path MSE: `21.264900`

Four-option held-out test results:

1. `hdelta_35b_guarded_on_tsm_sent`: `20.810390`
2. `paper_cot_sent_4b_on_tsm_sent`: `20.862739`
3. `hdelta_4b_on_tsm_sent`: `21.084306`
4. `base_tsm_sent`: `21.264900`

Artifacts:

1. `reports/strategy3_sentiment_tsm/20260301_211719/four_option_comparison.csv`
2. `reports/strategy3_sentiment_tsm/20260301_211719/selection_summary.json`
3. `reports/strategy3_sentiment_tsm/20260301_211719/paper_cot_sent_4b/paper_cot_sent_4b_on_tsm_sent/test/summary.json`
4. `reports/strategy3_sentiment_tsm/20260301_211719/hdelta_4b/hdelta_4b_on_tsm_sent/test/summary.json`
5. `reports/strategy3_sentiment_tsm/20260301_211719/hdelta_35b_guarded/hdelta_35b_guarded_on_tsm_sent/test/summary.json`

Interpretation:

1. The stronger `qwen_votes3_daily3` sentiment series did improve this Strategy 3 branch relative to the earlier baseline-sentiment injection run
2. But it still did not beat the best existing non-Strategy-3 results
3. Compared with the current global benchmark set:
   - baseline paper CoT-SENT: `20.604430`
   - 4B raw HDELTA: `20.618083`
   - raw `tsm`: `20.725666`
   - 35B raw guarded HDELTA: `20.872258`
4. The best Strategy 3 variant here (`20.810390`) remains worse than the top three existing clean results

Conclusion:

1. Strategy 3 with a stronger LLM-derived sentiment series is directionally better than Strategy 3 with baseline daily sentiment
2. However, it is still not a winning branch yet
3. The best honest models remain outside Strategy 3 for now

#### 22.17.2 Strategy 3 stronger structured-news feature design (`feat4`) 

Goal:

1. Move beyond scalar daily sentiment and inject a richer structured-news feature set into the TSM input panel
2. Use the strongest existing structured daily news file:
   `data/news/daily_sentiment_qwen_votes3_daily3_gate_policy_shipping_event90_proxy_feat4.csv`
3. Only continue to full 4-option overlay evaluation if the TSM stage remains competitive

Implementation:

1. Extended `src/data/panel.py` to ingest multi-column structured news files
2. Added support for structured derived features such as:
   - `sent_score`
   - `sent_score_3d_ma`
   - `sent_score_7d_ma`
   - `sent_change`
   - `sent_news_volume`
   - `sent_news_volume_3d_ma`
   - `sent_score_x_volume`
3. Made model feature capacity configurable via `features.max_exogenous_features_model`
4. Reused `scripts/run_strategy3_sentiment_tsm.py` with:
   - `--structured-news`
   - `--max-exogenous-features 12`

Validation:

1. `python -m py_compile src/data/panel.py src/run_experiment.py scripts/tune_clean_tsm_cot_rf.py scripts/tune_paper_cot_sent_hparams.py scripts/run_strategy3_sentiment_tsm.py`
2. `pytest -q tests/test_sentiment_panel_features.py tests/test_meta_blend.py tests/test_hdelta_semantics.py tests/test_blend_semantics.py tests/test_leakage_guards.py`
3. Result: `13 passed`

Run:

1. `python -u scripts/run_strategy3_sentiment_tsm.py --llm-api-key deo --sentiment-path data/news/daily_sentiment_qwen_votes3_daily3_gate_policy_shipping_event90_proxy_feat4.csv --structured-news --max-exogenous-features 12`
2. Output dir: `reports/strategy3_sentiment_tsm/20260301_234846`

Feature verification:

1. The panel widened from `44` to `50` columns
2. The structured-news columns were present in the data pipeline
3. The TSM search therefore tested a real richer-news branch rather than a wiring bug

TSM gate results:

1. Best candidate: `af_s60_l15_m32_ff64_do030_lr1e3_wd1e2`
2. Best validation path MSE: `39.787758`
3. Best held-out test path MSE: `21.190252`

Artifacts:

1. `reports/strategy3_sentiment_tsm/20260301_234846/tsm_search_results.csv`
2. `reports/strategy3_sentiment_tsm/20260301_234846/partial_summary.json`

Interpretation:

1. The stronger structured-news feature design was materially worse than the current honest benchmarks already at the TSM stage
2. Specifically, its best held-out test path MSE `21.190252` was worse than:
   - baseline paper CoT-SENT: `20.604430`
   - 4B raw HDELTA: `20.618083`
   - raw `tsm`: `20.725666`
   - best prior Strategy 3 branch (`qwen_votes3_daily3` + 35B guarded): `20.810390`
3. That made the branch dominated before any 4B or 35B overlay was applied

Decision:

1. Stop the branch after the TSM gate
2. Do not spend extra compute on 4B/35B overlays for this structured-news design
3. Record it as rejected

Conclusion:

1. Richer structured-news features are not automatically better
2. This `feat4` design was too invasive and degraded the base forecaster materially
3. If Strategy 3 continues later, it should use a narrower structured-news design rather than this wider `feat4` feature set

#### 22.18 Regime audit on the clean benchmark set

Goal:

1. Stop expanding Strategy 3 and instead inspect whether the current honest benchmark models specialize by regime
2. Compare only the agreed clean benchmark set:
   - baseline paper CoT-SENT
   - 4B raw HDELTA
   - raw `tsm`
   - 35B raw guarded HDELTA
3. Determine whether a regime gate is worth validating next

Implementation:

1. Added reusable regime-audit utilities in `src/eval/regime_audit.py`
2. Added runner script `scripts/run_regime_audit.py`
3. Added regression coverage in `tests/test_regime_audit.py`

Validation:

1. `python -m py_compile src/eval/regime_audit.py scripts/run_regime_audit.py`
2. `pytest -q tests/test_regime_audit.py tests/test_meta_blend.py tests/test_hdelta_semantics.py tests/test_blend_semantics.py tests/test_leakage_guards.py tests/test_sentiment_panel_features.py`
3. Result: `16 passed`

Run:

1. `python scripts/run_regime_audit.py`
2. Output dir: `reports/regime_audit/20260302_005326`

Overall benchmark table:

1. Baseline paper CoT-SENT: `20.604430`
2. 4B raw HDELTA: `20.618083`
3. Raw `tsm`: `20.725666`
4. 35B raw guarded HDELTA: `20.872258`

Regime definitions:

1. `volatility_regime`
   - high/low split at the median 20-day realized log-return volatility
   - threshold: `0.016589`
2. `trend_regime_20d`
   - up/flat/down from the prior 20-day price move using the same volatility-scaled threshold logic as the repo trend evaluator
3. `sentiment_alignment_5d`
   - compares 3-day average baseline sentiment with prior 5-day price return
   - buckets: `aligned`, `divergent`, `neutral`
   - magnitude gates:
     - sentiment abs median: `0.050000`
     - return abs median: `0.023617`
4. `trend_vol_regime`
   - cross of `trend_regime_20d` and `volatility_regime`

Key results by path MSE:

1. `volatility_regime`
   - high vol: baseline paper CoT-SENT wins (`26.054460`)
   - low vol: 35B raw guarded HDELTA wins (`14.896868`)
2. `trend_regime_20d`
   - down: baseline paper CoT-SENT wins (`19.689594`)
   - up: 4B raw HDELTA wins (`20.266885`)
   - flat: 35B raw guarded HDELTA wins (`29.639063`), but this bucket is only `18` samples
3. `sentiment_alignment_5d`
   - neutral: baseline paper CoT-SENT wins (`19.372391`, `330` samples)
   - divergent: 4B raw HDELTA wins (`36.000963`, `28` samples)
   - aligned: 35B raw guarded HDELTA wins (`16.393058`, `24` samples)
4. `trend_vol_regime`
   - `down__high_vol`: baseline paper CoT-SENT wins (`28.872135`)
   - `down__low_vol`: baseline paper CoT-SENT wins (`7.705599`)
   - `up__high_vol`: baseline paper CoT-SENT wins (`22.731910`)
   - `up__low_vol`: 35B raw guarded HDELTA wins (`17.944181`)
   - `flat__high_vol`: 4B raw HDELTA wins (`34.440593`) on only `12` samples
   - `flat__low_vol`: 35B raw guarded HDELTA wins (`18.962565`) on only `6` samples

Interpretation:

1. The winners do move by regime, so there is real specialization in the benchmark set
2. The safest and largest bucket is still `neutral` sentiment alignment, where baseline paper CoT-SENT remains best
3. 4B raw HDELTA is strongest in the `divergent` sentiment bucket and in `up` trend conditions
4. 35B raw guarded HDELTA benefits in calmer / aligned regimes, especially `low_vol`, but several of its strongest buckets are small
5. Winner-share and mean-path-MSE do not always point to the same model inside a bucket, which means some apparent regime wins are concentrated rather than broadly stable

Oracle upper bounds:

1. These are ex post only and are not deployable
2. They answer a narrower question: how much headroom exists if a regime gate were perfectly specified on this same test period
3. Results:
   - `trend_vol_regime`: `20.378191`
   - `sentiment_alignment_5d`: `20.404056`
   - `volatility_regime`: `20.475664`
   - `trend_regime_20d`: `20.502978`

Conclusion:

1. A regime gate is worth validating because the oracle ceilings are meaningfully below the current best single-model result (`20.604430`)
2. But it is not honest to promote a gate yet from this test-only audit
3. The next correct step is to validate one or two simple predefined gates on rolling archival splits rather than introducing more prompt or blend complexity

Artifacts:

1. `reports/regime_audit/20260302_005326/overall_metrics.csv`
2. `reports/regime_audit/20260302_005326/origin_regimes.csv`
3. `reports/regime_audit/20260302_005326/regime_path_mse.csv`
4. `reports/regime_audit/20260302_005326/regime_horizon_mse.csv`
5. `reports/regime_audit/20260302_005326/regime_winner_counts.csv`
6. `reports/regime_audit/20260302_005326/oracle_gate_upper_bounds.csv`
7. `reports/regime_audit/20260302_005326/summary.md`

#### 22.19 Fixed regime-gate validation on the rolling archive

Goal:

1. Take the two simplest gate families suggested by the clean benchmark audit
2. Freeze their mappings ahead of time
3. Validate them on the contamination-safe rolling archive rather than on the single held-out split

Implementation:

1. Added `apply_fixed_gate(...)` to `src/eval/regime_audit.py`
2. Added `scripts/validate_regime_gates_archive.py`
3. Reused the contamination-safe rolling-prefix training path from `src/frontend_live_bundle.py`
4. At each origin, generated forecasts for:
   - baseline paper CoT-SENT
   - 4B raw HDELTA
   - raw `tsm`
   - 35B raw guarded HDELTA
5. Applied two fixed gates:
   - `gate_sentiment_alignment_5d`
   - `gate_trend_vol_regime`

Validation:

1. `python -m py_compile scripts/validate_regime_gates_archive.py src/eval/regime_audit.py`
2. `pytest -q tests/test_regime_audit.py tests/test_meta_blend.py tests/test_hdelta_semantics.py tests/test_blend_semantics.py tests/test_leakage_guards.py tests/test_sentiment_panel_features.py`
3. Result: `17 passed`

Smoke test:

1. `python scripts/validate_regime_gates_archive.py --max-origins 1 --output-dir reports/regime_gate_validation/smoke_1origin`
2. Completed successfully before the full run

Full run:

1. `python scripts/validate_regime_gates_archive.py`
2. Output dir: `reports/regime_gate_validation/20260302_025134`
3. Archive origins: `21`
4. Origin dates:
   - `2021-02-25`
   - `2021-05-26`
   - `2021-08-23`
   - `2021-11-18`
   - `2022-02-15`
   - `2022-05-17`
   - `2022-08-12`
   - `2022-11-09`
   - `2023-02-07`
   - `2023-05-09`
   - `2023-08-04`
   - `2023-11-01`
   - `2024-02-01`
   - `2024-05-02`
   - `2024-07-30`
   - `2024-10-25`
   - `2025-01-27`
   - `2025-04-28`
   - `2025-07-24`
   - `2025-10-21`
   - `2025-12-22`

Operational check:

1. No LLM model fell back to base `tsm` on any archive origin
2. Fallback count:
   - baseline paper CoT-SENT: `0`
   - 4B raw HDELTA: `0`
   - 35B raw guarded HDELTA: `0`

Fixed gate definitions used:

1. `gate_sentiment_alignment_5d`
   - `neutral -> Baseline paper CoT-SENT`
   - `aligned -> 35B raw guarded HDELTA`
   - `divergent -> 4B raw HDELTA`
2. `gate_trend_vol_regime`
   - `down__high_vol -> Baseline paper CoT-SENT`
   - `down__low_vol -> Baseline paper CoT-SENT`
   - `up__high_vol -> Baseline paper CoT-SENT`
   - `up__low_vol -> 35B raw guarded HDELTA`
   - `flat__high_vol -> 4B raw HDELTA`
   - `flat__low_vol -> 35B raw guarded HDELTA`

Archive results:

1. `gate_sentiment_alignment_5d`: `51.937463`
2. baseline paper CoT-SENT: `53.189472`
3. `gate_trend_vol_regime`: `53.242641`
4. 35B raw guarded HDELTA: `53.774054`
5. 4B raw HDELTA: `54.326750`
6. raw `tsm`: `54.565439`

Interpretation:

1. `gate_sentiment_alignment_5d` is the first gate that actually beats the single-model baselines on a contamination-safe rolling archive
2. It improves over the best single model (baseline paper CoT-SENT) by:
   - absolute path MSE: `1.252009`
   - relative: `2.35%`
3. The gain is coming mostly from medium and longer horizons:
   - `h1`: slightly worse than baseline (`3.893813` vs `3.871905`)
   - `h5`: better (`12.721002` vs `13.363095`)
   - `h20`: better (`79.559137` vs `81.668476`)
   - `h30`: better (`83.553512` vs `85.594605`)
4. The gate is simple rather than hyperactive:
   - selected baseline paper CoT-SENT on `17/21` origins
   - selected 35B raw guarded HDELTA on `3/21` origins
   - selected 4B raw HDELTA on `1/21` origins
5. So the improvement came from a few targeted switches, not from broadly replacing the baseline model

Why the sentiment-alignment gate worked:

1. The `aligned` and `divergent` buckets were rare, but they sometimes contained very large baseline misses
2. In the archive run, the profitable switches were:
   - `2021-11-18`: aligned -> 35B (`68.850014` vs baseline `92.916227`)
   - `2022-05-17`: aligned -> 35B (`63.068828` vs baseline `72.168887`)
3. The gate also had losing switches:
   - `2021-05-26`: aligned -> 35B (`5.251656` vs baseline `4.333537`)
   - `2022-02-15`: divergent -> 4B (`214.009724` vs baseline `208.053770`)
4. But the wins were large enough to outweigh those losses on average

Why the trend-vol gate failed:

1. It defaulted to baseline paper CoT-SENT in most cases, but switched to 35B for all `up__low_vol` origins
2. That helped on some archive origins:
   - `2021-11-18`
   - `2022-08-12`
   - `2025-01-27`
3. But it hurt on others, especially the recent ones:
   - `2025-07-24`
   - `2025-10-21`
   - `2025-12-22`
4. That left it slightly worse than the single-model baseline overall

Conclusion:

1. The regime-audit idea was directionally correct
2. But only the simpler `sentiment_alignment_5d` gate survived rolling archive validation
3. `trend_vol_regime` is not a good production candidate in its current fixed form
4. The best validated archive strategy now is:
   - baseline paper CoT-SENT as default
   - switch to 35B raw guarded HDELTA in `aligned`
   - switch to 4B raw HDELTA in `divergent`
5. This is still not a fully independent final production result, because the mapping itself came from the prior benchmark audit
6. But it is now the strongest gate candidate we have, and much more credible than the earlier test-only oracle analysis

Artifacts:

1. `reports/regime_gate_validation/20260302_025134/archive_gate_summary.csv`
2. `reports/regime_gate_validation/20260302_025134/archive_origin_progress.csv`
3. `reports/regime_gate_validation/20260302_025134/gate_assignments_by_origin.csv`
4. `reports/regime_gate_validation/20260302_025134/benchmark_archive_predictions.csv`
5. `reports/regime_gate_validation/20260302_025134/metadata.json`
6. `reports/regime_gate_validation/20260302_025134/summary.json`

#### 22.19.1 Denser archive cadence (`origin_step=21`)

Goal:

1. Re-run the same fixed-gate validation at denser monthly-equivalent trading cadence
2. Check whether the `gate_sentiment_alignment_5d` result survives a much larger archive sample

Run:

1. `python scripts/validate_regime_gates_archive.py --origin-step 21`
2. Output dir: `reports/regime_gate_validation/20260302_113939`
3. Archive origins: `61`

Operational check:

1. No LLM path fell back to base `tsm`
2. Fallback count remained:
   - baseline paper CoT-SENT: `0`
   - 4B raw HDELTA: `0`
   - 35B raw guarded HDELTA: `0`

Dense-cadence results:

1. `gate_sentiment_alignment_5d`: `44.756433`
2. baseline paper CoT-SENT: `44.979842`
3. `gate_trend_vol_regime`: `45.529749`
4. 4B raw HDELTA: `46.595348`
5. raw `tsm`: `47.160486`
6. 35B raw guarded HDELTA: `47.198194`

Improvement vs best single model:

1. absolute path MSE: `0.223409`
2. relative: `0.50%`

Interpretation:

1. The sentiment-alignment gate still wins on the denser archive
2. But the edge is much smaller than on the quarterly archive (`0.223409` vs `1.252009`)
3. That means the original quarterly result was directionally right but somewhat overstated
4. The stronger conclusion now is:
   - the gate has a small but persistent edge
   - not a large, dramatic edge

Horizon view:

1. `h1`: worse than baseline (`3.330006` vs `3.254072`)
2. `h5`: better (`13.111512` vs `13.264705`)
3. `h20`: better (`56.574411` vs `57.697298`)
4. `h30`: slightly better (`82.335272` vs `82.365225`)

Gate behavior at dense cadence:

1. `gate_sentiment_alignment_5d` selected:
   - baseline paper CoT-SENT on `47/61`
   - 35B raw guarded HDELTA on `8/61`
   - 4B raw HDELTA on `6/61`
2. So even at denser cadence, the gate remains mostly a baseline model with selective overrides

#### 22.19.2 Production/export promotion

Decision:

1. Promote only `gate_sentiment_alignment_5d` into the production/frontend live bundle path
2. Do not promote `gate_trend_vol_regime`
3. Keep the old `tsm + CoT-RF blend` live bundle path only as a legacy comparison mode

Implementation:

1. Added a dedicated benchmark-gate bundle generator:
   - `src/regime_gate_bundle.py`
2. Switched `scripts/build_frontend_live_bundle.py` to default to:
   - `--strategy benchmark_gate`
3. Kept the previous path available behind:
   - `--strategy legacy_cot_rf`

What the promoted bundle now exports:

1. `current_price.json`
2. `actuals_recent.json`
3. `current_forecast.json`
4. `forecast_archive.csv`
5. `forecast_archive.json`
6. `model_summary.json`
7. `status.json`
8. `manifest.json`
9. `gate_assignments_by_origin.csv`

Live bundle benchmark set:

1. `Baseline paper CoT-SENT`
2. `4B raw HDELTA`
3. `Raw tsm`
4. `35B raw guarded HDELTA`
5. synthetic promoted strategy:
   - `gate_sentiment_alignment_5d`

Smoke validation:

1. Command:
   - `python scripts/build_frontend_live_bundle.py --data-dir Data_auto --max-origins 1 --history-years 1`
2. Run:
   - `runs/20260302_130650_0896e7`
3. Strategy emitted:
   - `benchmark_gate`
4. Bundle output directory:
   - `runs/20260302_130650_0896e7/frontend_live_bundle`

Checks:

1. `python -m py_compile src/regime_gate_bundle.py scripts/build_frontend_live_bundle.py src/frontend_live_bundle.py src/eval/regime_audit.py`
2. `pytest -q tests/test_regime_audit.py tests/test_meta_blend.py tests/test_blend_semantics.py tests/test_hdelta_semantics.py tests/test_leakage_guards.py tests/test_sentiment_panel_features.py`
3. Result:
   - `17 passed`

Switch quality:

1. The gate beat the baseline on `10/61` origins
2. It lost on most switched origins, but a subset of wins was large enough to preserve a small average advantage
3. Examples of profitable switches:
   - `2021-11-18`: aligned -> 35B (`80.583515` vs baseline `101.342457`)
   - `2022-05-17`: aligned -> 35B (`70.454770` vs baseline `83.811040`)
   - `2023-01-09`: divergent -> 4B (`66.019162` vs baseline `86.058503`)
4. Examples of bad switches that reduced the edge:
   - `2021-07-23`: aligned -> 35B (`67.095126` vs baseline `21.581927`)
   - `2024-03-01`: aligned -> 35B (`45.171126` vs baseline `39.406127`)
   - `2025-10-21`: aligned -> 35B (`2.433878` vs baseline `1.335237`)

Conclusion:

1. The simple `sentiment_alignment_5d` gate still survives the harder archive test
2. `trend_vol_regime` still does not
3. The gate is now credible enough to keep as the leading conditional strategy
4. But the current evidence supports only a modest production claim:
   - small improvement over baseline paper CoT-SENT
   - not a dominant replacement
5. If promoted, it should be framed as a lightweight overlay on the baseline, not as a wholly new winner

Artifacts:

1. `reports/regime_gate_validation/20260302_113939/archive_gate_summary.csv`
2. `reports/regime_gate_validation/20260302_113939/archive_origin_progress.csv`
3. `reports/regime_gate_validation/20260302_113939/gate_assignments_by_origin.csv`
4. `reports/regime_gate_validation/20260302_113939/benchmark_archive_predictions.csv`
5. `reports/regime_gate_validation/20260302_113939/metadata.json`
6. `reports/regime_gate_validation/20260302_113939/summary.json`

#### 22.19.3 New backend LLM on `:9878`

Goal:

1. Test the new local backend model exposed at `http://192.168.1.140:9878/v1`
2. Determine whether it improves the clean CoT-SENT benchmark set before changing the promoted backend path

Endpoint:

1. Base URL: `http://192.168.1.140:9878/v1`
2. Reported model id: `qwen3.5-4b-ud-q4-k-xl`
3. API key: `deo`

Paper CoT-SENT result:

1. Command:
   - `python -u scripts/tune_paper_cot_sent_signals.py --base-run runs/20260228_145058_40d91f --llm-api-key deo --llm-base-url http://192.168.1.140:9878/v1 --llm-model qwen3.5-4b-ud-q4-k-xl --signal-names baseline_daily_sentiment --output-dir reports/cot_sent_signal_tune/strategy1_qwen35_4b_ud_baseline_signal_20260302 --timeout-seconds 60`
2. Stage-1 subset result:
   - `218.426553`
3. Decision:
   - reject paper-style full-path CoT on this model immediately

HDELTA screen:

1. Raw HDELTA stage-1:
   - `38.480119`
2. Guarded HDELTA stage-1:
   - `37.463427`
3. Guardrails used:
   - `strict_json_prompt=true`
   - `retain_context=false`
   - `freeze_horizons=[1]`
   - `hdelta_max_adjustment_pct=1.0`
   - `hdelta_sentiment_secondary=true`
4. Decision:
   - only the guarded HDELTA variant was worth full validation/test

Full guarded HDELTA run:

1. Command:
   - `python -u scripts/tune_paper_cot_sent_signals.py --base-run runs/20260228_145058_40d91f --llm-api-key deo --llm-base-url http://192.168.1.140:9878/v1 --llm-model qwen3.5-4b-ud-q4-k-xl --signal-names baseline_daily_sentiment --method TSM+LLM-COT-SENT-RF-HDELTA --hdelta-max-adjustment-pct 1.0 --cot-strict-json-prompt --retain-context-override false --hdelta-freeze-horizons 1 --hdelta-sentiment-secondary --output-dir reports/cot_sent_signal_tune/strategy2_hdelta_qwen35_4b_ud_guarded_full_20260302 --timeout-seconds 60`
2. Full validation:
   - `38.714637`
3. Held-out test:
   - `20.519838`
4. Test horizons:
   - `h1=1.384549`
   - `h5=6.421122`
   - `h20=25.866504`
   - `h30=40.655884`

Comparison against the current clean benchmark set:

1. New model guarded HDELTA (`qwen3.5-4b-ud-q4-k-xl @ 9878`): `20.519838`
2. Baseline paper CoT-SENT (`qwen3-vl-4b-gpu @ 9877`): `20.604430`
3. Existing 4B raw HDELTA (`qwen3-vl-4b-gpu @ 9877`): `20.618083`
4. Raw `tsm`: `20.725666`
5. 35B raw guarded HDELTA (`qwen3.5-35b-a3b-ud-q4-k-xl @ 9881`): `20.872258`

Interpretation:

1. The new model is not usable with the paper-style full-path rewrite prompt
2. The new model does work under guarded HDELTA
3. Guarded HDELTA on the new model is now the best single-model held-out test result in the benchmark set
4. The improvement is small:
   - `0.084592` better than baseline paper CoT-SENT
   - `0.098245` better than the existing 4B raw HDELTA
5. This does **not** yet prove that the promoted `gate_sentiment_alignment_5d` production path should change, because the archive gate was not revalidated with the new model substituted for the old 4B branch

Artifacts:

1. `reports/cot_sent_signal_tune/strategy1_qwen35_4b_ud_baseline_signal_20260302/stage1_subset_results.csv`
2. `reports/cot_sent_signal_tune/strategy2_hdelta_qwen35_4b_ud_raw_stage1_20260302/stage1_subset_results.csv`
3. `reports/cot_sent_signal_tune/strategy2_hdelta_qwen35_4b_ud_guarded_stage1_20260302/stage1_subset_results.csv`
4. `reports/cot_sent_signal_tune/strategy2_hdelta_qwen35_4b_ud_guarded_full_20260302/stage2_full_val_results.csv`
5. `reports/cot_sent_signal_tune/strategy2_hdelta_qwen35_4b_ud_guarded_full_20260302/stage3_test_result.csv`
6. `reports/cot_sent_signal_tune/strategy2_hdelta_qwen35_4b_ud_guarded_full_20260302/final_test_comparison.csv`

#### 22.19.4 Dense gate revalidation with `:9878` divergent branch

Goal:

1. Revalidate the promoted dense archive gate after substituting the new `:9878` model into the divergent branch
2. Decide whether the production `gate_sentiment_alignment_5d` path should switch from the old `4B raw HDELTA` branch to the new guarded `qwen3.5-4b-ud-q4-k-xl`

Method:

1. A naive full rerun of all branches was started first
2. That rerun was rejected because the unchanged paper endpoint at `:9877` fell back on every completed origin, making the comparison invalid
3. The clean rerun instead reused the already-validated dense archive predictions for:
   - `Baseline paper CoT-SENT`
   - `35B raw guarded HDELTA`
   - `Raw tsm`
4. It recomputed only the new divergent branch:
   - `qwen3.5-4b-ud-q4-k-xl guarded HDELTA @ :9878`
5. This preserved the original 61-origin archive and changed only the branch we were actually evaluating

Command:

1. `python -u scripts/validate_regime_gates_archive.py --origin-step 21 --llm-api-key deo --paper-base-url http://192.168.1.140:9877/v1 --paper-model qwen3-vl-4b-gpu --divergent-base-url http://192.168.1.140:9878/v1 --divergent-model qwen3.5-4b-ud-q4-k-xl --divergent-mode guarded --llm35b-base-url http://192.168.1.140:9881/v1 --llm35b-model qwen3.5-35b-a3b-ud-q4-k-xl --reuse-predictions-csv reports/regime_gate_validation/20260302_113939/benchmark_archive_predictions.csv --reuse-models 'Baseline paper CoT-SENT,35B raw guarded HDELTA,Raw tsm' --output-dir reports/regime_gate_validation/20260302_9878_divergent_guarded_reuse`

Dense archive results (`n=61`):

1. `Baseline paper CoT-SENT`: `44.979842`
2. `gate_sentiment_alignment_5d`: `45.000285`
3. `qwen3.5-4b-ud-q4-k-xl guarded HDELTA`: `45.355948`
4. `gate_trend_vol_regime`: `45.406847`
5. `Raw tsm`: `47.160486`
6. `35B raw guarded HDELTA`: `47.198194`

Comparison to the previous dense archive:

1. Previous promoted gate (`old divergent = 4B raw HDELTA`): `44.756433`
2. New mixed gate (`divergent = qwen3.5-4b-ud-q4-k-xl guarded HDELTA`): `45.000285`
3. Difference:
   - `+0.243852` worse
4. New mixed gate vs baseline paper CoT-SENT:
   - `+0.020443` worse

Why the gate got worse:

1. The new guarded `:9878` branch is a better **single model** than the old `4B raw HDELTA` on the full dense archive:
   - new guarded branch: `45.355948`
   - old raw branch: `46.595348`
2. But the gate only uses the divergent branch on `6/61` origins
3. On those exact divergent origins, the old branch was actually better:
   - old `4B raw HDELTA` on divergent origins: `76.994874`
   - new guarded `:9878` branch on divergent origins: `79.474038`
4. So the global single-model improvement does not translate into a better fixed gate

Gate behavior:

1. `gate_sentiment_alignment_5d` selections remained:
   - baseline paper CoT-SENT on `47/61`
   - 35B raw guarded HDELTA on `8/61`
   - new guarded `:9878` branch on `6/61`
2. Those `6` changed rows were exactly the `divergent` bucket:
   - `2022-02-15`
   - `2022-06-15`
   - `2023-01-09`
   - `2024-04-03`
   - `2025-02-25`
   - `2025-03-26`

Operational checks:

1. Reused unchanged branches:
   - `Baseline paper CoT-SENT`: `61`
   - `35B raw guarded HDELTA`: `61`
2. New guarded `:9878` branch fallbacks:
   - `0`

Conclusion:

1. Do **not** switch the promoted production gate to the new `:9878` branch
2. The current dense archive evidence still supports the existing promoted gate:
   - baseline paper CoT-SENT
   - 35B raw guarded HDELTA in `aligned`
   - old `4B raw HDELTA` in `divergent`
3. The new `:9878` model remains the best **single-model held-out test** result
4. But it is not the best branch for the current fixed dense archive gate

Artifacts:

1. `reports/regime_gate_validation/20260302_9878_divergent_guarded_reuse/archive_gate_summary.csv`
2. `reports/regime_gate_validation/20260302_9878_divergent_guarded_reuse/archive_origin_progress.csv`
3. `reports/regime_gate_validation/20260302_9878_divergent_guarded_reuse/gate_assignments_by_origin.csv`
4. `reports/regime_gate_validation/20260302_9878_divergent_guarded_reuse/benchmark_archive_predictions.csv`
5. `reports/regime_gate_validation/20260302_9878_divergent_guarded_reuse/metadata.json`
6. `reports/regime_gate_validation/20260302_9878_divergent_guarded_reuse/summary.json`

## 22.20 `paper_replication_v2` architecture path

Goal:

1. Stop comparing ad hoc runs against stale manuscript tables.
2. Create a dedicated, leakage-safe paper-replication branch that:
   - scores candidates with paper-style path MSE and Day-10/20/30 trend accuracy,
   - selects on rolling validation folds,
   - only then spends held-out test budget.
3. Keep the promoted production gate untouched while this branch is exploratory.

Implementation:

1. Added `src/eval/paper_replication.py`
   - candidate dataclass
   - chronological outer-fold builder
   - per-candidate paper-style scoring
   - fold aggregation
2. Added `scripts/run_paper_replication_v2.py`
   - reuses the clean checkpoint from a saved base run
   - rebuilds TSM train/val/test forecasts
   - runs paper-style `TSM+LLM-COT-SENT-RF` candidates on rolling validation folds
   - supports per-candidate endpoint/model overrides
   - can reuse the saved clean raw test artifact when the selected candidate exactly matches the known paper baseline
3. Added `src/config/paper_replication_v2.yaml`
   - includes baseline 9877 paper candidates
   - includes live 9878 paper-style challenger candidates
   - sets conservative replication-timeout controls (`timeout_seconds=60`, `max_retries=1`)
4. Added `tests/test_paper_replication.py`

Validation:

1. `python -m py_compile src/eval/paper_replication.py scripts/run_paper_replication_v2.py`
2. `pytest -q tests/test_paper_replication.py`
3. Result: `2 passed`

### 22.20.1 Initial live pilot on the available `:9878` paper branch

Reason:

1. The old 4B paper endpoint at `:9877` was down during this run (`Connection refused`), so a clean live baseline replication there was not possible.
2. Rather than block the new architecture entirely, we ran a small rolling-validation pilot on the live `:9878` full-path paper branch to check whether it is even directionally promising.

Command:

1. `python -u scripts/run_paper_replication_v2.py --candidate-names paper_event90_proxy_ud4b,paper_event90_proxy_ud4b_strictjson --max-samples-per-fold 2 --skip-selected-test --llm-api-key deo`

Output:

1. `reports/paper_replication_v2/20260302_183730`

Rolling validation result (`n=2` samples per fold, `3` folds):

1. `paper_event90_proxy_ud4b`
   - `mse_path_mean = 47.110761`
   - `paper_d10_acc_mean = 0.166667`
   - `paper_d20_acc_mean = 0.166667`
   - `paper_d30_acc_mean = 0.166667`
2. `paper_event90_proxy_ud4b_strictjson`
   - `mse_path_mean = 57.828295`
   - `paper_d10_acc_mean = 0.000000`
   - `paper_d20_acc_mean = 0.000000`
   - `paper_d30_acc_mean = 0.166667`

Selection:

1. `paper_event90_proxy_ud4b` was selected as the less-bad of the two live `:9878` paper-style candidates.
2. Held-out test was intentionally skipped.

Interpretation:

1. The new `paper_replication_v2` architecture is working as intended.
2. The live `:9878` full-path paper-style branch is still structurally bad.
3. Strict JSON on the same branch made it worse, not better.
4. There is no justification to spend held-out test budget on that `:9878` paper branch.
5. The actual baseline paper comparison still needs the `:9877` endpoint back up so the clean old-4B paper candidate can be re-evaluated under the new nested-validation path.

Artifacts:

1. `reports/paper_replication_v2/20260302_183730/val_fold_metrics.csv`
2. `reports/paper_replication_v2/20260302_183730/val_candidate_summary.csv`
3. `reports/paper_replication_v2/20260302_183730/selected_candidate.json`

### 22.20.2 Old 4B `:9877` paper rerun on `paper_replication_v2`

Goal:

1. Re-run the actual paper-style old 4B branch on the new nested-validation architecture.
2. Check whether a cleaner event-deduped sentiment feed closes any of the remaining gap without reintroducing leakage.

Validation selection run:

1. `python -u scripts/run_paper_replication_v2.py --candidate-names paper_baseline,paper_event90_proxy --output-dir reports/paper_replication_v2/20260302_191842 --max-samples-per-fold 2 --llm-api-key deo`

Validation result (`n=2` samples per fold, `3` folds):

1. `paper_event90_proxy`: `mse_path_mean = 32.918562`
2. `paper_baseline`: `mse_path_mean = 35.332787`

So under the new rolling validation path, the event-deduped paper candidate beat the baseline paper candidate on `:9877`.

Operational note:

1. The uncapped selected-candidate test stage was too slow for practical use in this path because the paper-style full-path refinement is still expensive per sample.
2. We therefore reran the selected candidate only with a capped hold-out slice to get an honest directional result from the new architecture instead of waiting on a large paper-style test.

Capped held-out check:

1. `python scripts/run_paper_replication_v2.py --candidate-names paper_event90_proxy --output-dir reports/paper_replication_v2/20260302_195236_proxy_cap10 --max-samples-per-fold 2 --test-max-samples 10 --llm-api-key deo`

Result (`n=10` held-out test samples):

1. selected candidate: `paper_event90_proxy`
2. `mse_path = 32.396579`
3. `h1_mse = 0.781794`
4. `h5_mse = 8.424711`
5. `h20_mse = 38.161984`
6. `h30_mse = 76.700763`
7. `paper_d10_acc = 0.50`
8. `paper_d20_acc = 0.50`
9. `paper_d30_acc = 0.50`

Gap checks:

1. vs clean current benchmark `20.604430`: `+11.792149`
2. vs target `20.0`: `+12.396579`
3. vs old repo paper-era `13.8262`: `+18.570379`
4. vs old repo paper-era `12.5149`: `+19.881679`

Conclusion:

1. The new `paper_replication_v2` evaluation path is working and it does select a different paper candidate than the old baseline.
2. But even the better `:9877` paper-style candidate is still far from the current clean leaderboard.
3. The remaining architecture gap is therefore in the paper-style full-path refinement itself, not in the new validation harness.

Artifacts:

1. `reports/paper_replication_v2/20260302_191842/val_fold_metrics.csv`
2. `reports/paper_replication_v2/20260302_191842/val_candidate_summary.csv`
3. `reports/paper_replication_v2/20260302_191842/selected_candidate.json`
4. `reports/paper_replication_v2/20260302_195236_proxy_cap10/selected_test_summary.json`
5. `reports/paper_replication_v2/20260302_195236_proxy_cap10/gap_summary.json`

### 22.20.3 Structured paper-branch corrections inside `paper_replication_v2`

Goal:

1. Test whether the paper-branch gap is mainly caused by the unconstrained 30-step rewrite.
2. Re-run structured CoT-SENT candidates through the same nested-validation harness instead of trusting the separate HDELTA experiments alone.

Implementation:

1. Extended `PaperReplicationCandidate` and `run_paper_replication_v2.py` so paper candidates can carry candidate-level HDELTA settings:
   - `hdelta_key_horizons`
   - `hdelta_max_adjustment_pct`
   - `hdelta_freeze_horizons`
   - `hdelta_sentiment_secondary`
2. Added new paper-replication candidates in `src/config/paper_replication_v2.yaml`.

Validation:

1. `python -m py_compile src/eval/paper_replication.py scripts/run_paper_replication_v2.py`
2. `pytest -q tests/test_paper_replication.py`
3. Result: `3 passed`

#### 22.20.3.1 HDELTA with original key horizons `1/5/20/30`

Command:

1. `python -u scripts/run_paper_replication_v2.py --candidate-names paper_event90_proxy_hdelta_raw,paper_event90_proxy_hdelta_guarded --output-dir reports/paper_replication_v2/20260302_200105_hdelta_val --max-samples-per-fold 2 --skip-selected-test --llm-api-key deo`

Validation result:

1. `paper_event90_proxy_hdelta_raw`: `mse_path_mean = 35.691820`
2. `paper_event90_proxy_hdelta_guarded`: `mse_path_mean = 39.074176`

Comparison against the already-measured full-path paper candidate on the same nested-validation cap:

1. `paper_event90_proxy` full-path: `32.918562`
2. best structured `1/5/20/30` HDELTA: `35.691820`
3. gap: `+2.773257`

Decision:

1. No held-out test budget spent.
2. Neither structured `1/5/20/30` candidate beat the existing full-path paper candidate on validation.

Artifacts:

1. `reports/paper_replication_v2/20260302_200105_hdelta_val/val_candidate_summary.csv`
2. `reports/paper_replication_v2/20260302_200105_hdelta_val/selected_candidate.json`
3. `reports/paper_replication_v2/20260302_200105_hdelta_val/val_fold_metrics.csv`

#### 22.20.3.2 Paper-aligned HDELTA with key horizons `1/10/20/30`

Reason:

1. The first HDELTA test was structurally better than the full-path rewrite, but it was still aligned to `h5` rather than the paper-style day-10/day-20/day-30 trend scoreboard.
2. We therefore retried the same architecture with paper-aligned anchor horizons.

Command:

1. `python -u scripts/run_paper_replication_v2.py --candidate-names paper_event90_proxy_hdelta_paperh_raw,paper_event90_proxy_hdelta_paperh_guarded --output-dir reports/paper_replication_v2/20260302_200714_hdelta_paperh_val --max-samples-per-fold 2 --skip-selected-test --llm-api-key deo`

Validation result:

1. `paper_event90_proxy_hdelta_paperh_raw`: `mse_path_mean = 36.281037`
2. `paper_event90_proxy_hdelta_paperh_guarded`: `mse_path_mean = 38.753320`

Comparison against the same full-path paper candidate:

1. `paper_event90_proxy` full-path: `32.918562`
2. best paper-aligned HDELTA: `36.281037`
3. gap: `+3.362475`

Conclusion:

1. Structured horizon-delta corrections are not enough to recover the paper branch.
2. The miss is not just that `h5` was misaligned with the paper metric; even paper-aligned `1/10/20/30` anchors still underperformed.
3. The next architecture change should therefore move beyond percentage horizon deltas, not just retune them.

Artifacts:

1. `reports/paper_replication_v2/20260302_200714_hdelta_paperh_val/val_candidate_summary.csv`
2. `reports/paper_replication_v2/20260302_200714_hdelta_paperh_val/selected_candidate.json`
3. `reports/paper_replication_v2/20260302_200714_hdelta_paperh_val/val_fold_metrics.csv`

### 22.20.4 Absolute anchor-price paper branch (`HPRICE`)

Goal:

1. Try the next architecture step after HDELTA failed:
   - absolute anchor prices instead of percentage deltas
   - paper-relevant anchor horizons
   - deterministic interpolation
   - `h1` frozen to base TSM
   - structured reflection rules only

Implementation:

1. Added `TSM+LLM-COT-SENT-RF-HPRICE` in `src/llm/refine.py`
2. Added `CoT-SENT-RF-HPRICE-APPLY` in `src/llm/prompts.py`
3. Added:
   - anchor-price JSON parser
   - strict response schema
   - deterministic anchor interpolation
4. Wired candidate-level `hprice_*` config into `run_paper_replication_v2.py`
5. Added tests:
   - `tests/test_hprice_semantics.py`
   - extended `tests/test_paper_replication.py`

Validation:

1. `python -m py_compile src/llm/refine.py src/llm/prompts.py src/eval/paper_replication.py scripts/run_paper_replication_v2.py src/run_experiment.py`
2. `pytest -q tests/test_hdelta_semantics.py tests/test_hprice_semantics.py tests/test_paper_replication.py`
3. Result: `9 passed`

#### 22.20.4.1 Attempt on `:9877`

Command:

1. `python -u scripts/run_paper_replication_v2.py --candidate-names paper_event90_proxy_hprice_guarded --output-dir reports/paper_replication_v2/20260302_201301_hprice_guarded_val --max-samples-per-fold 2 --skip-selected-test --llm-api-key deo`

Result:

1. The run completed, but `:9877` was down (`Connection error`, and `curl` to `http://192.168.1.140:9877/v1/models` failed).
2. All reflection calls failed and the path fell back to TSM.
3. The resulting score (`mse_path_mean = 39.041280`) is therefore not a valid architecture result.

Decision:

1. Do not use the `:9877` HPRICE score as evidence for or against the method.

#### 22.20.4.2 Live endpoint validation on `:9878` and `:9881`

Command:

1. `python -u scripts/run_paper_replication_v2.py --candidate-names paper_event90_proxy_hprice_guarded_ud4b,paper_event90_proxy_hprice_guarded_35b --output-dir reports/paper_replication_v2/20260302_201816_hprice_live_val --max-samples-per-fold 2 --skip-selected-test --llm-api-key deo`

Validation result:

1. `paper_event90_proxy_hprice_guarded_35b`: `mse_path_mean = 38.405166`
2. `paper_event90_proxy_hprice_guarded_ud4b`: `mse_path_mean = 40.507742`

Comparison to the earlier live full-path `:9878` pilot:

1. `paper_event90_proxy_ud4b` full-path: `47.110761`
2. `paper_event90_proxy_ud4b_strictjson` full-path: `57.828295`
3. So the anchor-price architecture materially improved the live paper-style branch on the endpoints that were actually available.

#### 22.20.4.3 Capped held-out checks for selected `35B HPRICE`

10-sample cap:

1. `reports/paper_replication_v2/20260302_202108_hprice_35b_cap10/selected_test_summary.json`
2. `mse_path = 31.323950`

20-sample cap:

1. `reports/paper_replication_v2/20260302_202352_hprice_35b_cap20/selected_test_summary.json`
2. `mse_path = 28.219638`

40-sample cap:

1. `reports/paper_replication_v2/20260302_202746_hprice_35b_cap40/selected_test_summary.json`
2. `mse_path = 20.868700`
3. `h1_mse = 1.250939`
4. `h5_mse = 7.913822`
5. `h20_mse = 25.671423`
6. `h30_mse = 38.045835`

Same 40-sample slice comparison against saved clean baselines:

1. `35B HPRICE`: `20.868700`
2. clean `TSM`: `21.504989`
3. clean `:9877` paper full-path`: `21.291889`

So on the same 40-sample hold-out slice:

1. `35B HPRICE` beat `TSM` by `0.636289` path MSE (`2.96%`)
2. `35B HPRICE` beat clean `:9877` paper full-path by `0.423190` path MSE (`1.99%`)
3. It also improved all four reported horizons versus the old `:9877` paper full-path on that slice.

Interpretation:

1. This is the first paper-branch architectural change that is directionally working.
2. The gain does not come from the old `:9877` 4B branch; it comes from the guarded anchor-price architecture on the live `35B` endpoint.
3. The result is still capped (`n=40`), so it is not yet a full held-out promotion.
4. But unlike the earlier HDELTA variants, this branch is now close to the clean `<20` target (`gap = 0.868700` on the 40-sample slice).

Conclusion:

1. The paper branch is not solved, but the architecture is finally moving in the right direction.
2. The strongest next step is to run `paper_event90_proxy_hprice_guarded_35b` on a larger held-out cap or the full held-out split, not to go back to full-path rewrites or HDELTA.

Artifacts:

1. `reports/paper_replication_v2/20260302_201816_hprice_live_val/val_candidate_summary.csv`
2. `reports/paper_replication_v2/20260302_202108_hprice_35b_cap10/selected_test_summary.json`
3. `reports/paper_replication_v2/20260302_202352_hprice_35b_cap20/selected_test_summary.json`
4. `reports/paper_replication_v2/20260302_202746_hprice_35b_cap40/selected_test_summary.json`
5. `reports/paper_replication_v2/20260302_202746_hprice_35b_cap40/gap_summary.json`

#### 22.20.4.4 Old `:9877` 4B `HPRICE` rerun after endpoint recovery

Reason:

1. The first `:9877` `HPRICE` attempt was invalid because the endpoint was down.
2. Once `:9877` came back, we reran the same guarded anchor-price branch on the exact same 40-sample capped hold-out used for the `35B HPRICE` check.

Command:

1. `python -u scripts/run_paper_replication_v2.py --candidate-names paper_event90_proxy_hprice_guarded --output-dir reports/paper_replication_v2/20260303_000512_hprice_4b_cap40 --max-samples-per-fold 2 --test-max-samples 40 --llm-api-key deo`

Result (`n=40` capped hold-out):

1. `mse_path = 20.973162`
2. `h1_mse = 1.250939`
3. `h5_mse = 7.926387`
4. `h20_mse = 25.797628`
5. `h30_mse = 38.402693`
6. `paper_d10_acc = 0.60`
7. `paper_d20_acc = 0.45`
8. `paper_d30_acc = 0.475`

Same 40-sample slice comparison:

1. `35B HPRICE`: `20.868700`
2. `4B HPRICE @ :9877`: `20.973162`
3. clean `:9877` paper full-path: `21.291889`
4. clean `TSM`: `21.504989`

Interpretation:

1. The recovered `:9877` 4B `HPRICE` branch is valid and competitive.
2. It is slightly worse than the `35B HPRICE` branch by `0.104462` path MSE.
3. It still beats the old clean `:9877` paper full-path branch by `0.318727` (`1.50%`).
4. It also beats clean `TSM` by `0.531827` (`2.47%`).

Conclusion:

1. The anchor-price architecture is helping both model scales.
2. `35B HPRICE` remains the best paper-branch contender.
3. But the margin over the recovered `:9877` 4B `HPRICE` branch is small, so model size is no longer the dominant story; the architecture change is.

Artifacts:

1. `reports/paper_replication_v2/20260303_000512_hprice_4b_cap40/selected_test_summary.json`
2. `reports/paper_replication_v2/20260303_000512_hprice_4b_cap40/gap_summary.json`

#### 22.20.4.5 Full held-out test for `35B HPRICE`

Reason:

1. The capped `10/20/40` results showed that `35B HPRICE` was directionally better than both clean `TSM` and the old clean `:9877` paper full-path branch.
2. The next technical bar was the true full held-out split, not another capped slice.

Operational note:

1. The first uncapped command accidentally inherited `paper_replication_v2.yaml`'s default `test_max_samples: 120`.
2. That run finished at `n=120` and was useful as an intermediate check (`mse_path = 20.166073`), but it was not the full held-out split.
3. We then reran with `--test-max-samples 0` to force the full test block and reused the 120-sample cache.

Final command:

1. `python -u scripts/run_paper_replication_v2.py --candidate-names paper_event90_proxy_hprice_guarded_35b --output-dir reports/paper_replication_v2/20260303_003149_hprice_35b_full_uncapped --max-samples-per-fold 2 --test-max-samples 0 --llm-api-key deo`

Final held-out result (`n=382`, full split):

1. `mse_path = 20.657072`
2. `h1_mse = 1.384549`
3. `h5_mse = 6.403430`
4. `h20_mse = 25.972636`
5. `h30_mse = 41.041513`
6. `paper_d10_acc = 0.518325`
7. `paper_d20_acc = 0.486911`
8. `paper_d30_acc = 0.473822`

Same full-split comparison:

1. old clean `:9877` paper full-path: `20.604430`
2. `35B HPRICE`: `20.657072`
3. clean `TSM`: `20.942024`

Interpretation:

1. `35B HPRICE` beat clean `TSM` by `0.284952` (`1.36%`).
2. `35B HPRICE` did **not** beat the old clean `:9877` paper full-path branch on the full held-out split.
3. The gap to the old paper full-path branch is small: `+0.052642` path MSE (`0.26%`).
4. So the architectural change is still real and useful, but the full-split result is not enough to declare a new overall winner.

Conclusion:

1. `35B HPRICE` is now the most credible alternate paper-branch architecture.
2. It clearly improves on clean `TSM`.
3. But it does not displace the old clean `:9877` paper full-path branch on the full held-out split.
4. The full-split story is therefore:
   - best current paper-branch full result: old clean `:9877` full-path (`20.604430`)
   - very close runner-up: `35B HPRICE` (`20.657072`)
   - both beat clean `TSM` (`20.942024`)

Artifacts:

1. `reports/paper_replication_v2/20260303_001420_hprice_35b_full/selected_test_summary.json`
2. `reports/paper_replication_v2/20260303_003149_hprice_35b_full_uncapped/selected_test_summary.json`
3. `reports/paper_replication_v2/20260303_003149_hprice_35b_full_uncapped/gap_summary.json`

#### 22.20.4.6 Full held-out test for recovered `:9877` 4B `HPRICE`

Reason:

1. After the full uncapped `35B HPRICE` run finished, the remaining question was whether the small 35B edge was actually worth the extra model cost.
2. To answer that, we ran the recovered `:9877` 4B `HPRICE` branch on the same full held-out split.

Command:

1. `python -u scripts/run_paper_replication_v2.py --candidate-names paper_event90_proxy_hprice_guarded --output-dir reports/paper_replication_v2/20260303_012030_hprice_4b_full_uncapped --max-samples-per-fold 2 --test-max-samples 0 --llm-api-key deo`

Final held-out result (`n=382`, full split):

1. `mse_path = 20.724603`
2. `h1_mse = 1.384549`
3. `h5_mse = 6.391484`
4. `h20_mse = 26.065643`
5. `h30_mse = 41.276217`
6. `paper_d10_acc = 0.523560`
7. `paper_d20_acc = 0.468586`
8. `paper_d30_acc = 0.458115`

Same full-split ranking:

1. old clean `:9877` paper full-path: `20.604430`
2. `35B HPRICE`: `20.657072`
3. `4B HPRICE @ :9877`: `20.724603`
4. clean `TSM`: `20.942024`

Interpretation:

1. Both `HPRICE` branches beat clean `TSM` on the full held-out split.
2. `35B HPRICE` beats `4B HPRICE` by `0.067531` path MSE.
3. That edge is real but small.
4. Neither `HPRICE` branch displaces the old clean `:9877` paper full-path branch on the full held-out split.

Conclusion:

1. The architecture change matters more than model size.
2. `35B HPRICE` is the best of the new paper-branch architectures.
3. But the cost/performance gap over `4B HPRICE` is small, while both remain slightly behind the old clean paper full-path benchmark.

Artifacts:

1. `reports/paper_replication_v2/20260303_012030_hprice_4b_full_uncapped/selected_test_summary.json`
2. `reports/paper_replication_v2/20260303_012030_hprice_4b_full_uncapped/gap_summary.json`

#### 22.20.4.7 Simplify the paper branch toward 4B

Reason:
1. `35B HPRICE` only beat recovered `:9877` `4B HPRICE` by `0.067531` path MSE on the full held-out split.
2. Both `HPRICE` branches still trailed the old clean `:9877` paper full-path result.
3. That makes the extra 35B complexity hard to justify as the default operating path.

Changes:
1. Set `src/config/paper_replication_v2.yaml` `default_profile: simplified_4b_hprice`
2. Added named profiles:
   - `simplified_4b_hprice`
   - `compare_hprice_4b_vs_35b`
   - `legacy_paper_4b_fullpath`
3. Changed the config default `test_max_samples` from `120` to `0` so the default path uses the full held-out split.
4. Extended `scripts/run_paper_replication_v2.py` with `--profile`
5. Profile resolution rule is now:
   - `--candidate-names` wins
   - else `--profile`
   - else config `default_profile`
6. Added regression coverage in `tests/test_paper_replication.py`

Operational effect:
1. `python scripts/run_paper_replication_v2.py --llm-api-key deo` now runs the simplified `:9877` `4B HPRICE` branch by default.
2. `35B` stays available, but only when explicitly requested via `--profile compare_hprice_4b_vs_35b` or `--candidate-names`.

Validation:
1. `python -m py_compile scripts/run_paper_replication_v2.py`
2. `pytest -q tests/test_paper_replication.py`

#### 22.20.4.8 Dense archive validation for simplified `4B HPRICE`

Goal:
1. Check whether the simplified `:9877` `4B HPRICE` branch is stable beyond the paper-replication lane.
2. Reuse the unchanged dense-archive benchmark predictions so the only new live branch is the simplified `4B HPRICE` paper role.

Code:
1. Extended `scripts/validate_regime_gates_archive.py` with `--paper-mode {fullpath,hprice}`
2. Added candidate-level `hprice_*` support and per-candidate `sentiment_path`
3. Added regression coverage in `tests/test_validate_regime_gates_archive.py`

Validation:
1. `python -m py_compile scripts/validate_regime_gates_archive.py`
2. `pytest -q tests/test_validate_regime_gates_archive.py tests/test_paper_replication.py`
3. Result: `9 passed`

Command:
1. `python -u scripts/validate_regime_gates_archive.py --paper-mode hprice --origin-step 21 --llm-api-key deo --reuse-predictions-csv reports/regime_gate_validation/20260302_113939/benchmark_archive_predictions.csv --reuse-models '4B raw HDELTA,35B raw guarded HDELTA,Raw tsm'`

Output:
1. `reports/regime_gate_validation/20260303_135303`

Dense archive result, `n=61` origins:
1. `gate_trend_vol_regime`: `44.397768`
2. `gate_sentiment_alignment_5d`: `44.588477`
3. `4B guarded HPRICE`: `44.782777`
4. `4B raw HDELTA`: `46.595348`
5. `Raw tsm`: `47.160486`
6. `35B raw guarded HDELTA`: `47.198194`

Comparisons to prior dense archive reference:
1. Prior promoted gate (`gate_sentiment_alignment_5d` with old paper full-path): `44.756433`
2. Old dense-archive paper full-path baseline: `44.979842`
3. New single-model `4B guarded HPRICE`: `44.782777`
4. New `gate_sentiment_alignment_5d` with `4B HPRICE` paper role: `44.588477`

Interpretation:
1. The simplified `4B HPRICE` branch held up on the full dense archive.
2. As a single model, it beat both HDELTA branches and the old dense-archive paper full-path baseline.
3. The existing promoted gate also improved when its neutral paper role was swapped from old paper full-path to `4B HPRICE`.
4. `gate_trend_vol_regime` now won on this run, but promoting it immediately would still be a fresh selection on this archive result.
5. The safer production-facing takeaway is that `4B HPRICE` is now a credible replacement for the old paper full-path branch, and the already-promoted `gate_sentiment_alignment_5d` looks stronger with that swap.

Operational notes:
1. No `4B HPRICE` archive origin fell back to TSM.
2. Reused branches remained unchanged from the prior dense archive benchmark.

Artifacts:
1. `reports/regime_gate_validation/20260303_135303/archive_gate_summary.csv`
2. `reports/regime_gate_validation/20260303_135303/archive_origin_progress.csv`
3. `reports/regime_gate_validation/20260303_135303/gate_assignments_by_origin.csv`
4. `reports/regime_gate_validation/20260303_135303/benchmark_archive_predictions.csv`
5. `reports/regime_gate_validation/20260303_135303/summary.json`

#### 22.20.4.9 `event_panel_v1` + `residual_model_v1`

Goal:
1. Stop asking the LLM to write forecasts directly.
2. Use the existing labeled headline assets as a structured daily event feature store.
3. Train a supervised residual corrector over the frozen clean `TSM` instead.

Architecture:
1. `src/news/event_panel.py`
   - builds a daily event panel from `headlines_labeled_qwen_votes3_daily3_importance_v3_indep_t035_percall.csv`
   - aggregates:
     - daily counts
     - sentiment sums/means
     - importance sums/means
     - weighted sentiment
     - `eu_ets_strong` / `energy_driver` / `eu_context` / `carbon_terms`
   - merges auxiliary daily series:
     - proxy sentiment
     - novelty/topical sentiment
     - Dawid-Skene posteriors
2. `src/eval/residual_event_model.py`
   - builds per-origin supervised features from:
     - base `TSM` forecast anchors/slopes
     - market state at the origin date
     - structured event features at the origin date
   - predicts correction anchors at `d5/d10/d20/d30`
   - interpolates the full path deterministically with `h1` frozen to base `TSM`
3. `scripts/run_event_panel_residual_v1.py`
   - replays the clean `20260228_145058_40d91f` checkpoint
   - selects candidate correctors on late validation only
   - evaluates the selected candidate once on held-out test

Validation:
1. `python -m py_compile src/news/event_panel.py src/eval/residual_event_model.py scripts/run_event_panel_residual_v1.py`
2. `pytest -q tests/test_event_panel.py tests/test_residual_event_model.py`
3. Result: `5 passed`

Run 1:
1. `python scripts/run_event_panel_residual_v1.py`
2. Output: `reports/event_panel_residual_v1/20260303_152528`

Run 1 result:
1. Absolute-price residual candidates were unstable and all lost on late validation.
2. The selector chose `identity`, meaning no residual correction beat base `TSM`.

Run 2:
1. Tightened the formulation into a supervised `HDELTA` analogue:
   - percentage anchor targets instead of absolute residual targets
   - explicit `8%` clip on learned corrections
   - smaller curated `event_core` feature block
2. Re-ran:
   - `python scripts/run_event_panel_residual_v1.py`
   - output: `reports/event_panel_residual_v1/20260303_152717`

Run 2 late-validation selection:
1. `identity`: `33.471913`
2. best learned candidate `ridge_base_pct_a1`: `73.154748`
3. best event-aware candidate `ridge_event_core_pct_a10`: `73.674407`

Run 2 held-out test:
1. Selected candidate: `identity`
2. `residual_event_v1`: `20.942024`
3. `TSM`: `20.942024`
4. old clean paper full-path: `20.604430`
5. `4B HPRICE`: `20.724603`

Interpretation:
1. The current structured event store is not strong enough to support a useful supervised residual corrector.
2. This is not a selection bug; the late-validation selector is explicitly rejecting the learned correction.
3. The problem is upstream information quality, not just the model family.
4. In its current form, `event_panel_v1 -> residual_model_v1` is not product-viable and should not be promoted.

Artifacts:
1. `reports/event_panel_residual_v1/20260303_152717/val_candidate_results.csv`
2. `reports/event_panel_residual_v1/20260303_152717/test_comparison.csv`
3. `reports/event_panel_residual_v1/20260303_152717/selected_candidate.json`
4. `reports/event_panel_residual_v1/20260303_152717/feature_manifest.json`

Recommended next step:
1. If we stay on this architectural track, the next version must improve the event store itself, not just the residual learner.
2. The cleanest next move is `event_panel_v2` with richer event taxonomy and/or rolling out-of-sample train predictions to enlarge residual-model training data.

#### 22.20.4.10 `event_panel_v2` richer taxonomy

Goal:
1. Test whether the failure of `event_panel_v1` was mainly because the event store was still too sentiment-heavy and weakly structured.
2. Add category-specific daily event features before paying the cost of rolling out-of-sample `TSM` retrains.

Changes:
1. Extended `src/news/event_panel.py` with `taxonomy_version="v2"`
2. Added title-driven taxonomy buckets:
   - `policy`
   - `auction`
   - `energy`
   - `shipping`
   - `industry`
   - `macro`
   - `geopolitics`
   - `weather`
   - `cbam`
3. For each category, aggregated:
   - count
   - importance sum
   - weighted sentiment sum
   - rolling `3d` / `7d` sums
4. Updated `scripts/run_event_panel_residual_v1.py` with `--event-version`
5. Expanded `event_core` in `src/eval/residual_event_model.py` to include the strongest new taxonomy features
6. Added coverage in `tests/test_event_panel.py`
7. Cleaned the builder to assemble rolling features via `pd.concat(...)` instead of repeated column insertion

Validation:
1. `python -m py_compile src/news/event_panel.py src/eval/residual_event_model.py scripts/run_event_panel_residual_v1.py`
2. `pytest -q tests/test_event_panel.py tests/test_residual_event_model.py`
3. Result: `6 passed`

Command:
1. `python scripts/run_event_panel_residual_v1.py --event-version v2`

Output:
1. `reports/event_panel_residual_v1/20260303_153356`

Result:
1. Selected candidate: `identity`
2. Held-out test `path_mse`: `20.942024`
3. This is exactly equal to base `TSM`

Interpretation:
1. The richer event taxonomy did not unlock a viable supervised residual correction path.
2. Because late validation still selected `identity`, there is no justification for paying the extra cost of rolling out-of-sample `TSM` retrains yet.
3. The correct reading is that `event_panel_v2` is still not carrying enough predictive signal for this residual architecture.

Artifacts:
1. `reports/event_panel_residual_v1/20260303_153356/val_candidate_results.csv`
2. `reports/event_panel_residual_v1/20260303_153356/test_comparison.csv`
3. `reports/event_panel_residual_v1/20260303_153356/selected_candidate.json`
4. `reports/event_panel_residual_v1/20260303_153356/feature_manifest.json`

Decision:
1. Do not continue to the rolling out-of-sample `TSM` retrain stage on this branch.
2. The next attempt on the structured-news path should change the source information itself, not just the supervised wrapper.

#### 22.20.4.11 Automated official-source event store

Goal:
1. Replace weak headline-level sentiment inputs with fully automated official sources that can be refreshed on the same kind of daily cadence as the rest of the system.
2. Keep the source pipeline operationally simple: deterministic ingestion where possible, LLM extraction only where free text must be structured.

Implemented files:
1. `src/news/official_sources.py`
   - DG CLIMA RSS + newsroom archive ingestion
   - EEX auction calendar discovery and parsing
   - EEX auction report discovery and parsing
   - deterministic conversion of EEX files into structured event records
2. `src/news/event_extraction.py`
   - strict JSON event extraction for official article bodies via the local OpenAI-compatible endpoint
3. `scripts/build_official_event_store.py`
   - end-to-end builder that fetches, parses, extracts, and writes a combined official event store
4. `tests/test_official_sources.py`
5. `tests/test_event_extraction.py`

Automated sources now covered:
1. DG CLIMA RSS:
   - `https://climate.ec.europa.eu/node/2/rss_en`
2. DG CLIMA newsroom archive:
   - `https://climate.ec.europa.eu/news-other-reads/news_en`
3. EEX EU ETS auction calendar workbook:
   - latest file discovered automatically from the public page
4. EEX EU ETS auction result workbook:
   - latest file discovered automatically from the public file index

Design:
1. DG CLIMA articles are fetched automatically from RSS plus archive pages.
2. Article bodies are extracted automatically where the page is reachable.
3. A local LLM then converts the article body into strict structured fields:
   - `event_type`
   - `affected_channel`
   - `direction`
   - `intensity`
   - `expected_horizon`
   - `novelty`
   - `policy_stage`
   - `confidence`
4. EEX files are not sent to the LLM.
5. EEX auction events are derived deterministically from official workbook fields, which keeps that branch fully automated and reproducible.

Validation:
1. `python -m py_compile src/news/official_sources.py src/news/event_extraction.py scripts/build_official_event_store.py`
2. `pytest -q tests/test_official_sources.py tests/test_event_extraction.py`
3. Result: `5 passed`

Live build command:
1. `python scripts/build_official_event_store.py --dg-pages 2 --extract-limit 12 --llm-base-url http://192.168.1.140:9877/v1 --llm-api-key deo`

Live build output:
1. Output dir: `data/news/official`
2. Built at UTC: `2026-03-03T15:58:35.881667+00:00`
3. LLM endpoint: `http://192.168.1.140:9877/v1`
4. LLM model: `qwen3-vl-4b-gpu`
5. DG CLIMA articles fetched: `36`
6. DG CLIMA articles extracted: `12`
7. DG CLIMA events kept: `9`
8. EEX auction calendar rows: `215`
9. EEX auction report rows: `36`
10. Deterministic EEX event rows: `251`
11. Combined official event rows: `260`

Artifacts written automatically:
1. `data/news/official/dg_clima_articles.csv`
2. `data/news/official/dg_clima_event_records.csv`
3. `data/news/official/eex_auction_calendar.csv`
4. `data/news/official/eex_auction_report.csv`
5. `data/news/official/eex_official_event_records.csv`
6. `data/news/official/official_event_records_v1.csv`
7. `data/news/official/official_event_records_v1.jsonl`
8. `data/news/official/manifest.json`
9. `data/news/official/cache/official_event_extract/*.json`
10. `data/news/official/logs/official_event_extract/llm_calls.jsonl`

Important operational notes:
1. The EEX branch is now clean and deterministic:
   - early rows with no stable rolling baseline fall back to neutral `volume_ratio_20d=1.0`
   - `intensity` is no longer allowed to become `NaN`
2. Some DG CLIMA newsroom links resolve to stale or external pages.
   - the ingester handles that automatically
   - those rows fall back to listing metadata instead of crashing the build
   - cheap relevance and strict extraction then filter most of them out
3. This official-source stack is fully rerunnable and does not depend on manual file downloads.
4. ENTSO-E is not included in this version.
   - that should be treated as a separate source integration because it has a different access and parsing surface

Conclusion:
1. We now have a production-usable automated official-source event store for the stronger structured-news direction.
2. The source layer is materially better than the old headline-sentiment path because it uses official policy and auction data directly.
3. The next step on this branch should use `official_event_records_v1` as the new base event source, rather than creating yet another transformation of the older headline sentiment files.

#### 22.20.4.12 Official event store wired into residual path

Goal:
1. Replace the old headline-label source in the structured residual branch with the new fully automated official event store.
2. Test whether better source quality alone is enough to make the supervised residual corrector useful.

Changes:
1. Extended `src/news/event_panel.py` so it now auto-detects two source schemas:
   - legacy headline-label CSVs
   - structured official event records from `data/news/official/official_event_records_v1.csv`
2. Added `official_v1` feature mode in `src/news/event_panel.py`
   - directional aggregation from structured `direction`
   - event-type counts and weighted sums
   - channel counts
   - horizon counts
   - novelty counts
   - policy-stage counts
3. Updated `src/eval/residual_event_model.py` to expose these official structured features through the `event_core` block.
4. Updated `scripts/run_event_panel_residual_v1.py`
   - default event source is now `data/news/official/official_event_records_v1.csv`
   - default event version is now `official_v1`
   - auxiliary proxy/novelty/Dawid-Skene files are off by default on this branch
5. Added test coverage for the official event schema path in `tests/test_event_panel.py`

Validation:
1. `python -m py_compile src/news/event_panel.py src/eval/residual_event_model.py scripts/run_event_panel_residual_v1.py`
2. `pytest -q tests/test_event_panel.py tests/test_residual_event_model.py`
3. Result: `7 passed`

Command:
1. `python scripts/run_event_panel_residual_v1.py`

Output:
1. `reports/event_panel_residual_v1/20260303_161009`

Feature footprint:
1. event source: `data/news/official/official_event_records_v1.csv`
2. event version: `official_v1`
3. event features total: `156`
4. `event_core` features: `51`

Late-validation selection:
1. `identity`: `33.471913`
2. `ridge_base_pct_a1`: `73.154748`
3. `ridge_event_core_pct_a10`: `74.142421`
4. `hgb_market_pct`: `85.716256`
5. `hgb_market_event_core_pct`: `85.716256`

Held-out test:
1. Selected candidate: `identity`
2. `residual_event_v1`: `20.942024`
3. `TSM`: `20.942024`
4. `4B HPRICE`: `20.724603`
5. old clean paper full-path: `20.604430`

Interpretation:
1. The official-source event store is operationally better, but it still does not support a useful supervised residual corrector in this formulation.
2. This is not because the feature block is empty or weakly wired:
   - the branch now uses `156` structured event features from official records
   - the selector still rejects every learned correction
3. The bottleneck is therefore no longer source automation.
4. The bottleneck is that this residual architecture still cannot convert the new event information into a stable forecast improvement.

Artifacts:
1. `reports/event_panel_residual_v1/20260303_161009/val_candidate_results.csv`
2. `reports/event_panel_residual_v1/20260303_161009/test_comparison.csv`
3. `reports/event_panel_residual_v1/20260303_161009/selected_candidate.json`
4. `reports/event_panel_residual_v1/20260303_161009/feature_manifest.json`

Decision:
1. Keep the automated official-source event store.
2. Do not promote the residual-event correction branch.
3. Any further work on this direction must change the modeling architecture, not just the source file or aggregation wrapper.

#### 22.20.4.13 Historical EEX expansion for the official event store

Problem:
1. The first official event store build was operationally correct but too short for honest historical evaluation.
2. It only used the latest EEX annual report workbook, so the EEX branch started effectively in late 2025/2026.
3. That made both prompt-context testing and official-event gate testing weak, because the source did not span the archive period.

Change:
1. Expanded `src/news/official_sources.py` so `fetch_eex_sources(...)` now ingests all EEX annual auction-report workbooks from a configurable minimum year.
2. Added `--eex-report-year-min` to `scripts/build_official_event_store.py`.
3. Rebuilt the official event store with `--eex-report-year-min 2021`.

Validation:
1. `python -m py_compile src/news/official_sources.py src/news/official_event_features.py src/llm/prompts.py src/llm/refine.py src/eval/regime_audit.py scripts/build_official_event_store.py scripts/run_paper_replication_v2.py scripts/validate_regime_gates_archive.py`
2. `pytest -q tests/test_official_sources.py tests/test_event_extraction.py tests/test_event_panel.py tests/test_residual_event_model.py tests/test_official_event_features.py tests/test_regime_audit.py tests/test_validate_regime_gates_archive.py tests/test_paper_replication.py`
3. Result: `27 passed`

Command:
1. `python scripts/build_official_event_store.py --dg-pages 2 --eex-report-year-min 2021 --extract-limit 12 --llm-base-url http://192.168.1.140:9877/v1 --llm-api-key deo`

Result:
1. Output dir: `data/news/official`
2. Event coverage now spans `2021-01-29` to `2026-12-11`
3. EEX report URLs ingested:
   - `2021`
   - `2022`
   - `2023`
   - `2024`
   - `2025`
   - `2026`
4. EEX report rows: `1145`
5. Deterministic EEX event rows: `1360`
6. Combined official event rows: `1369`

Interpretation:
1. The official source layer is now historically usable for the current archive period.
2. This was a necessary correction before any official-event prompt or gate evaluation could be taken seriously.

#### 22.20.4.14 Option 1: official event context inside `4B HPRICE`

Goal:
1. Test whether the stronger surviving `4B HPRICE` branch improves when given a deterministic official-event context block derived from `official_event_records_v1.csv`.
2. Keep the architecture fixed:
   - same `4B HPRICE`
   - same guardrails
   - same base sentiment path
   - only add compact official-event context to reflection/apply prompts

Changes:
1. Added `src/news/official_event_features.py`
   - builds per-origin official event summaries
   - exposes rolling official-event features and thresholds
2. Extended `src/llm/prompts.py`
   - CoT-SENT reflection/apply prompts now render an optional `Official / Exogenous Context` block
3. Updated `src/llm/refine.py`
   - passes `exogenous_summary` into the CoT-SENT/HDELTA/HPRICE prompt templates
4. Updated `scripts/run_paper_replication_v2.py`
   - builds cached official-event summary maps
   - attaches official context to both current samples and teaching examples
5. Added candidate `paper_event90_proxy_hprice_guarded_official_ctx` to `src/config/paper_replication_v2.yaml`

Focused run:
1. `python scripts/run_paper_replication_v2.py --candidate-names paper_event90_proxy_hprice_guarded_official_ctx --max-samples-per-fold 6 --test-max-samples 40 --llm-api-key deo`
2. Output: `reports/paper_replication_v2/20260303_163518`

Validation:
1. `paper_event90_proxy_hprice_guarded_official_ctx` validation `mse_path_mean = 41.444102`

Held-out cap-40 result:
1. official-context `4B HPRICE`: `20.973930`
2. prior matched `4B HPRICE` cap-40 baseline: `20.973162`
3. difference: `+0.000767` worse

Matched cap-40 baseline reference:
1. `reports/paper_replication_v2/20260303_000512_hprice_4b_cap40/selected_test_summary.json`

Interpretation:
1. The added official-event prompt context did not help.
2. On the matched capped test slice it was effectively flat, but slightly worse.
3. Given the extra prompt length and zero measurable gain, this branch should not be promoted.

Artifacts:
1. `reports/paper_replication_v2/20260303_163518/val_candidate_summary.csv`
2. `reports/paper_replication_v2/20260303_163518/selected_test_summary.json`
3. `reports/paper_replication_v2/20260303_163518/gap_summary.json`

Decision:
1. Keep the prompt-context plumbing because it is technically correct and reusable.
2. Do not use official-event context inside `4B HPRICE` as a promoted improvement path.

#### 22.20.4.15 Option 2: deterministic official-event regimes for gating

Goal:
1. Test official-event regimes as gate inputs rather than direct correction features.
2. Use the simplified benchmark set built around:
   - `4B guarded HPRICE`
   - `4B raw HDELTA`
   - `35B raw guarded HDELTA`
   - `Raw tsm`

Changes:
1. Added `official_event_alignment_5d` and `official_supply_regime_30d` to `src/eval/regime_audit.py`
2. Added official-event threshold support to `RegimeThresholds`
3. Extended `scripts/validate_regime_gates_archive.py`
   - accepts `--official-event-path`
   - defines official-event gate families alongside the existing sentiment/trend families
4. Because the existing validator still retrains prefixes even when predictions are reused, the actual metric check for this iteration was run directly from the saved dense-archive prediction file.

Quick dense-archive evaluation:
1. Source predictions:
   - `reports/regime_gate_validation/20260303_135303/benchmark_archive_predictions.csv`
2. Output summary:
   - `reports/regime_gate_validation/20260303_official_gate_quick/archive_gate_summary.csv`
3. Gate assignments:
   - `reports/regime_gate_validation/20260303_official_gate_quick/gate_assignments_by_origin.csv`

Results, `n=61` origins:
1. `gate_official_supply_regime_30d`: `44.725463`
2. `4B guarded HPRICE`: `44.782777`
3. `gate_official_event_alignment_5d`: `45.016746`
4. `4B raw HDELTA`: `46.595348`
5. `Raw tsm`: `47.160486`
6. `35B raw guarded HDELTA`: `47.198194`

Interpretation:
1. `official_supply_regime_30d` did help.
2. It beat the single-model `4B guarded HPRICE` baseline by `0.057314`.
3. That is a real but small gain, about `0.13%`.
4. `official_event_alignment_5d` did not help; it was worse than plain `4B guarded HPRICE`.
5. So the useful official-event regime is supply-specific, not generic event alignment.

Behavior:
1. `official_supply_regime_30d` selections:
   - `4B guarded HPRICE`: `30`
   - `35B raw guarded HDELTA`: `28`
   - `4B raw HDELTA`: `3`
2. regime counts:
   - `neutral_supply`: `30`
   - `bearish_supply`: `28`
   - `bullish_supply`: `3`

Decision:
1. Keep `official_supply_regime_30d` as a valid research gate family.
2. Do not promote `official_event_alignment_5d`.
3. The improvement is too small to justify immediate production promotion, but it is strong enough to keep for further validation.

#### 22.20.4.16 Fresh recent-window check for `official_supply_regime_30d`

Goal:
1. Check whether the small dense-archive gain from `official_supply_regime_30d` survives a fresher tail-window slice before any production promotion.

Method:
1. Reused the saved dense-archive benchmark predictions from `reports/regime_gate_validation/20260303_135303/benchmark_archive_predictions.csv`
2. Evaluated only the most recent `12` origins
3. Applied the same fixed supply gate mapping:
   - `neutral_supply -> 4B guarded HPRICE`
   - `bearish_supply -> 35B raw guarded HDELTA`
   - `bullish_supply -> 4B raw HDELTA`

Outputs:
1. `reports/regime_gate_validation/20260303_official_gate_recent12/recent12_summary.csv`
2. `reports/regime_gate_validation/20260303_official_gate_recent12/recent12_assignments.csv`

Results, `n=12` origins:
1. `35B raw guarded HDELTA`: `26.728109`
2. `4B guarded HPRICE`: `26.868987`
3. `4B raw HDELTA`: `26.933890`
4. `gate_official_supply_regime_30d`: `27.380043`
5. `Raw tsm`: `27.841787`

Interpretation:
1. The supply gate did **not** hold up on the freshest recent slice.
2. It was worse than each of the three component expert models on this `12`-origin tail window.
3. So the gate remains research-only; it should not be promoted into production on the basis of the broader dense-archive gain alone.

Decision:
1. Keep `official_supply_regime_30d` as a candidate gate family.
2. Do not promote it into the production/export path yet.

#### 22.20.4.17 Deeper DG CLIMA history and ENTSO-E automation

Goal:
1. Finish the official-source layer properly:
   - deeper DG CLIMA history
   - automated ENTSO-E official news ingestion
2. Keep the official event store updateable with the same one-command workflow as the current system.

Changes:
1. Extended `src/news/official_sources.py`
   - added ENTSO-E news RSS support
   - added ENTSO-E article extraction
   - added ENTSO-E cheap relevance filtering
2. Updated `scripts/build_official_event_store.py`
   - default `--dg-pages` increased from `2` to `25`
   - added `--entsoe-extract-limit`
   - writes:
     - `entsoe_articles.csv`
     - `entsoe_event_records.csv`
   - includes ENTSO-E metadata in `manifest.json`
3. Added tests:
   - `tests/test_official_sources.py`
   - `tests/test_official_event_features.py`
   - `tests/test_event_extraction.py`

Validation:
1. `python -m py_compile src/news/official_sources.py scripts/build_official_event_store.py`
2. `pytest -q tests/test_official_sources.py tests/test_official_event_features.py tests/test_event_extraction.py`
3. Result: `8 passed`

Live rebuild:
1. `python scripts/build_official_event_store.py --dg-pages 25 --eex-report-year-min 2021 --extract-limit 16 --entsoe-extract-limit 12 --llm-base-url http://192.168.1.140:9877/v1 --llm-api-key deo`

Manifest:
1. `data/news/official/manifest.json`

Live rebuild result:
1. `dg_pages = 25`
2. `dg_articles_total = 338`
3. `dg_events_kept = 13`
4. `entsoe_articles_total = 20`
5. `entsoe_articles_extracted = 12`
6. `entsoe_events_kept = 0`
7. `eex_report_rows = 1145`
8. `combined_event_rows = 1373`

Interpretation:
1. DG CLIMA history is now materially deeper and comfortably covers the current archive period.
2. ENTSO-E is now fully automated via its public RSS/news pages and is included in the same official event store build.
3. In this live rebuild, the strict relevance/extraction stage kept `0` ENTSO-E records, which is acceptable:
   - the source is automated
   - the current feed simply did not yield ETS-relevant events under the current extractor
4. This is better than forcing low-quality ENTSO-E records into the downstream dataset.

Decision:
1. Keep the deeper DG CLIMA crawl.
2. Keep ENTSO-E in the official build as an automated source.
3. Do not claim ENTSO-E is adding predictive value yet; at this stage it is source coverage and infrastructure, not a proven signal.

#### 22.20.4.18 Official store inside the strongest surviving full-path paper branch

Goal:
1. Stop spending effort on weaker branches and test the improved official event store inside the strongest surviving path:
   - `paper_event90_proxy`
   - method `TSM+LLM-COT-SENT-RF`
   - same `:9877` 4B model
   - same sentiment source
   - only add official-event context

Changes:
1. Added candidate profile `compare_fullpath_4b_official_ctx` to `src/config/paper_replication_v2.yaml`
2. Added candidate `paper_event90_proxy_official_ctx`
   - same full-path paper setup as `paper_event90_proxy`
   - plus:
     - `official_event_path = data/news/official/official_event_records_v1.csv`
     - `official_event_lookback_days = 30`
     - `official_event_max_titles = 3`

Bug fix discovered during first run:
1. The prompt plumbing for full-path official context was not actually safe end-to-end.
2. `CoTSentRFReflectionTemplate.format()` referenced `exogenous_summary` without accepting it.
3. Fixed in `src/llm/prompts.py`
4. Added regression test:
   - `tests/test_prompts_exogenous.py`
5. Also hardened `scripts/run_paper_replication_v2.py` so a total validation failure now raises a clear error instead of a follow-on `KeyError`.

Validation:
1. `python -m py_compile src/llm/prompts.py scripts/run_paper_replication_v2.py`
2. `pytest -q tests/test_paper_replication.py tests/test_prompts_exogenous.py`
3. Result: `8 passed`

Matched validation comparison:
1. Baseline command:
   - `python scripts/run_paper_replication_v2.py --candidate-names paper_event90_proxy --max-samples-per-fold 6 --skip-selected-test --llm-api-key deo`
   - output: `reports/paper_replication_v2/20260303_175617`
2. Official-context command:
   - `python scripts/run_paper_replication_v2.py --candidate-names paper_event90_proxy_official_ctx --max-samples-per-fold 6 --skip-selected-test --llm-api-key deo`
   - output: `reports/paper_replication_v2/20260303_174329`

Matched validation result, `3` folds, `6` samples/fold:
1. `paper_event90_proxy`: `38.583099`
2. `paper_event90_proxy_official_ctx`: `38.096425`
3. difference: `-0.486674` better for official context

Interpretation:
1. This is the first sign that the improved official store may help the strongest full-path paper branch.
2. The gain is modest, about `1.26%` on matched validation.
3. So unlike the earlier HPRICE official-context attempt, this branch is not immediately dominated.

Matched cap-40 baseline reference from saved clean full-path predictions:
1. Source:
   - `runs/20260228_145058_40d91f/predictions/TSM+LLM-COT-SENT-RF_pred_test_subset.npz`
2. Reconstructed cap-40 baseline:
   - `path_mse = 21.291889`
   - `h1 = 1.269155`
   - `h5 = 8.124964`
   - `h20 = 25.956260`
   - `h30 = 39.046225`
3. Saved to:
   - `reports/paper_replication_v2/baseline_paper_event90_proxy_cap40_from_saved.csv`

Completed capped held-out test:
1. Command:
   - `python scripts/run_paper_replication_v2.py --candidate-names paper_event90_proxy_official_ctx --max-samples-per-fold 6 --test-max-samples 40 --output-dir reports/paper_replication_v2/20260303_180035 --llm-api-key deo`
2. Output:
   - `reports/paper_replication_v2/20260303_180035/selected_test_summary.json`

Official-context cap-40 result:
1. `path_mse = 20.801385`
2. `h1 = 1.172991`
3. `h5 = 8.003892`
4. `h20 = 25.118548`
5. `h30 = 38.223388`

Matched clean baseline cap-40 reference:
1. `path_mse = 21.291889`
2. `h1 = 1.269155`
3. `h5 = 8.124964`
4. `h20 = 25.956260`
5. `h30 = 39.046225`

Cap-40 comparison:
1. path MSE improvement: `-0.490504`
2. relative improvement: about `2.30%`
3. horizon deltas vs matched baseline:
   - `h1`: `-0.096165`
   - `h5`: `-0.121072`
   - `h20`: `-0.837712`
   - `h30`: `-0.822836`

Interpretation:
1. The improved official event store materially helped the strongest surviving full-path paper branch on the matched capped held-out slice.
2. This is stronger than the earlier validation-only signal and is the first clean end-to-end result where the official store adds value to the strongest branch.
3. The result is still capped at `40` samples, so it is not yet the final promotion metric.
4. But it is now strong enough to justify a full held-out follow-up.

Artifacts:
1. `reports/paper_replication_v2/20260303_175617/val_candidate_summary.csv`
2. `reports/paper_replication_v2/20260303_174329/val_candidate_summary.csv`
3. `reports/paper_replication_v2/baseline_paper_event90_proxy_cap40_from_saved.csv`
4. `reports/paper_replication_v2/20260303_fullpath_official_ctx_compare/validation_comparison.csv`
5. `reports/paper_replication_v2/20260303_fullpath_official_ctx_compare/summary.csv`

Decision:
1. Keep the official-context full-path branch alive.
2. Promote it from “validation-only” to “serious contender”.
3. The next justified run on this branch is the uncapped full held-out test, not more prompt surgery.

Completed uncapped full held-out test:
1. Command:
   - `python scripts/run_paper_replication_v2.py --candidate-names paper_event90_proxy_official_ctx --max-samples-per-fold 6 --test-max-samples 0 --output-dir reports/paper_replication_v2/20260303_180035_full_uncapped --llm-api-key deo`
2. Output:
   - `reports/paper_replication_v2/20260303_180035_full_uncapped/selected_test_summary.json`

Uncapped full-split result, `n=382`:
1. `path_mse = 20.854719`
2. `h1 = 1.486676`
3. `h5 = 6.448772`
4. `h20 = 26.039853`
5. `h30 = 41.648924`

Comparison:
1. clean baseline paper full-path: `20.604430`
2. official-context paper full-path: `20.854719`
3. `35B HPRICE`: `20.657072`
4. `4B HPRICE`: `20.724603`
5. clean `TSM`: `20.942024`

Interpretation:
1. The earlier cap-40 gain did **not** hold on the full held-out split.
2. Official context remained better than base `TSM`, but worse than:
   - the clean baseline paper full-path
   - `35B HPRICE`
   - `4B HPRICE`
3. So this branch does not close the gap enough to justify promotion.
4. The most likely explanation is that the official context signal helped on the small capped slice but did not generalize across the whole held-out period.

Final decision for this branch:
1. Keep the official event store as infrastructure.
2. Reject `paper_event90_proxy_official_ctx` as a promoted forecasting branch.
3. Do not spend more prompt-engineering time on this exact context injection path.

### 22.20.4.19 Event-aware teaching-example retrieval on the strongest paper full-path branch

Objective:
1. Keep the current strongest clean full-path branch fixed:
   - method: `TSM+LLM-COT-SENT-RF`
   - candidate: `paper_event90_proxy`
2. Do **not** inject official events into the prompt.
3. Change only the teaching-example retrieval logic.

Implementation:
1. Added `retrieval_official_event_path` to `PaperReplicationCandidate`.
2. Added a new selection mode `event_similarity` in `scripts/run_paper_replication_v2.py`.
3. Retrieval vector now combines:
   - existing history summary features
   - candidate sentiment/state features derived from the panel
   - official event aggregate features from `data/news/official/official_event_records_v1.csv`
4. Prompt context remained unchanged, so this isolates retrieval quality instead of repeating the failed context-injection branch.

Candidate added:
1. `paper_event90_proxy_event_similarity`

Validation-only run:
1. Command:
   - `python scripts/run_paper_replication_v2.py --candidate-names paper_event90_proxy,paper_event90_proxy_event_similarity --max-samples-per-fold 6 --skip-selected-test --llm-api-key deo --output-dir reports/paper_replication_v2/20260303_event_similarity_val`
2. Output:
   - `reports/paper_replication_v2/20260303_event_similarity_val/val_candidate_summary.csv`

Validation result:
1. `paper_event90_proxy`: `38.583099`
2. `paper_event90_proxy_event_similarity`: `40.369323`

Per-horizon validation means:
1. baseline `paper_event90_proxy`
   - `h1 = 4.598265`
   - `h5 = 24.558621`
   - `h20 = 46.133454`
   - `h30 = 60.526874`
2. `paper_event90_proxy_event_similarity`
   - `h1 = 4.721771`
   - `h5 = 24.940016`
   - `h20 = 47.611876`
   - `h30 = 65.708131`

Interpretation:
1. Event-aware example retrieval made the branch worse on validation.
2. The degradation was broad, not isolated to one horizon.
3. Because this lost at the validation gate, it did **not** earn a capped or uncapped held-out test.

Artifacts:
1. `reports/paper_replication_v2/20260303_event_similarity_val/val_candidate_summary.csv`
2. `reports/paper_replication_v2/20260303_event_similarity_val/val_fold_metrics.csv`
3. `reports/paper_replication_v2/20260303_event_similarity_val/selected_candidate.json`

Decision:
1. Reject `event_similarity` retrieval on the current full-path winner.
2. Keep the official event store, but do not use it for teaching-example retrieval in this branch.
3. Do not spend held-out test budget on this retrieval variant.

### 22.20.4.20 Post-2023 regime-isolation test for the structural-break hypothesis

Objective:
1. Test whether the large EU ETS regime transition into 2023 was materially degrading model performance.
2. Re-run the strongest surviving 4B branches on a post-break dataset only.
3. Keep the evaluation honest and contamination-safe.

Implementation:
1. Added `target.min_date` / `target.max_date` passthrough to `src/data/panel.py`.
2. Added dedicated configs:
   - `src/config/post_2023_paper_event90_proxy.yaml`
   - `src/config/post_2023_hprice_4b.yaml`
3. Both configs use:
   - `target.min_date = 2023-01-01`
   - `train_end = 2024-12-31`
   - `val_end = 2025-06-30`
   - `test_end = 2026-02-25`

Resulting post-2023 dataset:
1. panel rows: `794`
2. date range: `2023-01-02` to `2026-02-25`
3. windows: `705`
4. split counts:
   - train: `424`
   - val: `97`
   - test: `126`

#### Full-path post-2023 run

Command:
1. `python -m src.run_experiment --config src/config/post_2023_paper_event90_proxy.yaml --data-dir Data_auto`

Run:
1. `runs/20260304_174436_c0e1b7`

Path MSE:
1. `naive_persistence`: `12.692016`
2. `seasonal_naive`: `13.369102`
3. `linear_ridge`: `14.800273`
4. `linear_lasso`: `21.047690`
5. `tsm`: `21.589029`
6. `TSM+LLM-COT-SENT-RF`: `20.885818`

Key horizons:
1. `tsm`
   - `h1 = 1.044371`
   - `h5 = 7.287224`
   - `h20 = 23.611553`
   - `h30 = 50.894348`
2. `TSM+LLM-COT-SENT-RF`
   - `h1 = 1.123402`
   - `h5 = 6.806958`
   - `h20 = 23.375420`
   - `h30 = 48.361116`

Interpretation:
1. The paper-style full-path branch still improved on the post-2023 TSM.
2. But both were far worse than the simple baselines.
3. Removing pre-2023 data did **not** make the advanced pipeline competitive.

#### HPRICE post-2023 run

Command:
1. `python -m src.run_experiment --config src/config/post_2023_hprice_4b.yaml --data-dir Data_auto`

Run:
1. `runs/20260304_180038_5de4f4`

Path MSE:
1. `naive_persistence`: `12.692016`
2. `seasonal_naive`: `13.369102`
3. `linear_ridge`: `14.800273`
4. `linear_lasso`: `21.047690`
5. `tsm`: `21.589029`
6. `TSM+LLM-COT-SENT-RF-HPRICE`: `21.352035`

Key horizons:
1. `TSM+LLM-COT-SENT-RF-HPRICE`
   - `h1 = 1.044371`
   - `h5 = 5.316018`
   - `h20 = 23.630203`
   - `h30 = 50.910801`

Interpretation:
1. HPRICE also improved on the post-2023 TSM.
2. HPRICE did better than TSM mainly at `h5`.
3. HPRICE remained much worse than `naive_persistence` and `linear_ridge`.

#### Structural-break conclusion

Observed ranking on the post-2023 regime:
1. `naive_persistence`: `12.692016`
2. `seasonal_naive`: `13.369102`
3. `linear_ridge`: `14.800273`
4. `TSM+LLM-COT-SENT-RF`: `20.885818`
5. `linear_lasso`: `21.047690`
6. `TSM+LLM-COT-SENT-RF-HPRICE`: `21.352035`
7. `tsm`: `21.589029`

Interpretation:
1. The post-2023 regime-isolation test does **not** support the claim that the 2020-2023 step change was the main thing suppressing performance.
2. Removing the pre-break history made the TSM itself worse relative to simple baselines.
3. The likely problem is the nature of the post-2024 market regime:
   - strong persistence
   - low incremental signal
   - limited benefit from deep sequence modeling and LLM correction
4. The advanced models still add some value over the post-2023 TSM, but not enough to compete with trivial or linear baselines.

Artifacts:
1. `runs/20260304_174436_c0e1b7/results/path_metrics.csv`
2. `runs/20260304_174436_c0e1b7/results/metrics_by_horizon.csv`
3. `runs/20260304_180038_5de4f4/results/path_metrics.csv`
4. `runs/20260304_180038_5de4f4/results/metrics_by_horizon.csv`

### 22.20.4.21 Post-2023 short-memory TSM search and CoT follow-up

To test whether the weak post-2023 result was mainly a bad base-model choice rather than the regime itself, I ran a targeted short-memory TSM search that included:
1. `dlinear` and `autoformer`
2. `returns` and `price` targets
3. short sequence lengths (`10-20`) and a few slightly longer controls

Command:
1. `python scripts/tune_post_2023_tsm.py --config src/config/post_2023_paper_event90_proxy.yaml --data-dir Data_auto`

Artifacts:
1. `reports/post_2023_tsm_tune/20260304_181715/tsm_search_results.csv`
2. `reports/post_2023_tsm_tune/20260304_181715/baseline_reference.csv`

Best candidate:
1. `dlinear_ret_s10_l5`
   - `target_mode = returns`
   - `seq_len = 10`
   - `label_len = 5`
   - `test_path_mse = 13.003386`

Interpretation:
1. The post-2023 TSM family was not exhausted.
2. A very short-memory DLinear base closed most of the gap to `naive_persistence`.
3. This was strong enough to justify a direct CoT rerun on the same split.

#### Full-path CoT on the best short-memory base

Config:
1. `src/config/post_2023_paper_event90_proxy_dlinear_s10_l5.yaml`

Command:
1. `python -m src.run_experiment --config src/config/post_2023_paper_event90_proxy_dlinear_s10_l5.yaml --data-dir Data_auto`

Run:
1. `runs/20260304_181827_554e77`

Path MSE:
1. `naive_persistence`: `12.692016`
2. `tsm`: `13.003386`
3. `TSM+LLM-COT-SENT-RF`: `14.106963`

Key horizons:
1. `tsm`
   - `h1 = 0.891673`
   - `h5 = 4.296810`
   - `h20 = 11.929129`
   - `h30 = 35.420010`
2. `TSM+LLM-COT-SENT-RF`
   - `h1 = 0.927895`
   - `h5 = 4.412755`
   - `h20 = 13.600384`
   - `h30 = 37.705655`

Interpretation:
1. On the stronger post-2023 base, unconstrained full-path CoT clearly made the forecast worse.
2. The degradation was broad across horizons, not just at `h1`.
3. This rules out the idea that the earlier failure was only due to a weak post-2023 base model.

Artifacts:
1. `runs/20260304_181827_554e77/results/path_metrics.csv`
2. `runs/20260304_181827_554e77/results/metrics_by_horizon.csv`

#### HPRICE on the best short-memory base

Config:
1. `src/config/post_2023_hprice_4b_dlinear_s10_l5.yaml`

Command:
1. `python -m src.run_experiment --config src/config/post_2023_hprice_4b_dlinear_s10_l5.yaml --data-dir Data_auto`

Run:
1. `runs/20260304_183839_aff261`

Path MSE:
1. `naive_persistence`: `12.692016`
2. `TSM+LLM-COT-SENT-RF-HPRICE`: `12.904609`
3. `tsm`: `13.003386`
4. `seasonal_naive`: `13.369102`

Key horizons:
1. `tsm`
   - `h1 = 0.891673`
   - `h5 = 4.296810`
   - `h20 = 11.929129`
   - `h30 = 35.420010`
2. `TSM+LLM-COT-SENT-RF-HPRICE`
   - `h1 = 0.891673`
   - `h5 = 3.467464`
   - `h20 = 11.929063`
   - `h30 = 35.417011`

Interpretation:
1. The constrained `HPRICE` correction is the first post-2023 CoT-style result that got materially close to the dominant baseline.
2. It improved on the best short-memory TSM by `0.098777` path MSE.
3. It still did **not** beat `naive_persistence`.
4. The residual gap to `naive_persistence` is `0.212593`, about `1.68%`.

Overall conclusion for the post-2023 branch:
1. The structural-break hypothesis was only part of the story.
2. Fixing the base model mattered more than truncating the regime by itself.
3. Unconstrained full-path CoT still fails on the stronger post-2023 base.
4. Constrained `HPRICE` on the stronger base is directionally promising, but it still does not clear the baseline bar.

Artifacts:
1. `runs/20260304_183839_aff261/results/path_metrics.csv`
2. `runs/20260304_183839_aff261/results/metrics_by_horizon.csv`

### 22.20.4.22 Post-2023 HPRICE control sweep on the short-memory base

Goal:
1. Close the remaining gap from `12.904609` to `naive_persistence=12.692016` on the same post-2023 split.
2. Keep base fixed: `dlinear_ret_s10_l5` (`seq_len=10`, `label_len=5`).
3. Tune only HPRICE controls.

Swept configs:
1. `post_2023_hprice_4b_dlinear_s10_l5_k152030_cap1.yaml`
2. `post_2023_hprice_4b_dlinear_s10_l5_k152030_cap2.yaml`
3. `post_2023_hprice_4b_dlinear_s10_l5_k15102030_cap2.yaml`

Runs:
1. `runs/20260304_191208_cf528a` (k=[1,5,20,30], cap=1.0)
2. `runs/20260304_192148_8c8641` (k=[1,5,20,30], cap=2.0)
3. `runs/20260304_193126_343711` (k=[1,5,10,20,30], cap=2.0)

Path MSE summary (same split, `n=126`):
1. `naive_persistence`: `12.692016`
2. previous best HPRICE (`k=[1,10,20,30], cap=1.0`): `12.904609`
3. `k=[1,5,20,30], cap=1.0`: `12.965150`
4. `k=[1,5,20,30], cap=2.0`: `12.964733`
5. `k=[1,5,10,20,30], cap=2.0`: `13.011117`

Interpretation:
1. The original tuned HPRICE setup remains best.
2. Adding direct `h5` anchors and/or increasing correction cap did not improve path MSE.
3. The remaining gap to `naive_persistence` stayed unresolved.
4. This local HPRICE-parameter sweep is now saturated for this branch.

Artifacts:
1. `runs/20260304_191208_cf528a/results/path_metrics.csv`
2. `runs/20260304_192148_8c8641/results/path_metrics.csv`
3. `runs/20260304_193126_343711/results/path_metrics.csv`

### 22.20.4.23 UK ETS pivot (isolated subdirectory)

Goal:
1. Stand up a UK ETS version of the existing pipeline without touching the EU flow.
2. Keep automation-first data refresh behavior.
3. Preserve current loader contracts by writing UK feeds in the same schemas.

Implementation added:
1. `uk_ets/scripts/bootstrap_uk_ets_data_sources.py`
2. `uk_ets/scripts/run_uk_automated_pipeline_once.py`
3. `uk_ets/config/uk_ets_default.yaml`
4. `uk_ets/README.md`

Default UK data root:
1. `uk_ets/Data_auto_uk`

Automated UK source mapping:
1. Target series (`eua-futures/*.csv`): UKA futures (Investing UKA page) with ICAP UK secondary-market backfill.
2. ICAP features (`icap-allowance-price-explorer-secondary-market/*.csv`): ICAP UK ETS systems API/download endpoint.
3. Auctions (`emission-spot-primary-market-auction/*.csv`):
   - Primary path: ICE report-center UK ETS auctions feed (`report 278`, actual source).
   - If unavailable in default mode (reCAPTCHA gating), fallback proxy from ICAP UK Primary Market.
   - `--require-actual-auctions` disables fallback and fails bootstrap if ICE actual feed is not reachable.
4. FX (`usd.xml`): ECB historical XML.
5. Energy (`energy-benchmarks/*`): FRED Brent + Yahoo coal (`MTF=F`).
6. Carbon indices (`carbon-market-indices/*`): Yahoo `GRN`, `KCCA`, `KEUA`, `KRBN`.
7. Volatility proxy (`volatility-proxy/vstoxx-index.txt`): Yahoo `^VFTSE` with `^VIX` fallback, then STOXX fallback.
8. Optional UK news (`news/headlines_raw.csv`): GDELT with UK ETS-focused query terms.

Run commands:
1. Bootstrap only:
   - `python uk_ets/scripts/bootstrap_uk_ets_data_sources.py --output-dir uk_ets/Data_auto_uk --start-date 2021-05-19 --end-date 2026-03-04`
2. Bootstrap (actual auctions required):
   - `python uk_ets/scripts/bootstrap_uk_ets_data_sources.py --output-dir uk_ets/Data_auto_uk --start-date 2021-05-19 --end-date 2026-03-04 --require-actual-auctions`
3. Bootstrap with provided ICE reCAPTCHA token:
   - `ICE_REPORT_RECAPTCHA_TOKEN=\"<token>\" python uk_ets/scripts/bootstrap_uk_ets_data_sources.py --output-dir uk_ets/Data_auto_uk --start-date 2021-05-19 --end-date 2026-03-04 --require-actual-auctions`
4. Bootstrap + one experiment pass:
   - `python uk_ets/scripts/run_uk_automated_pipeline_once.py --config uk_ets/config/uk_ets_default.yaml --data-dir uk_ets/Data_auto_uk --target-mode returns`

Verification status in this environment:
1. Script wiring and CLI are valid (`--help`, import path, py_compile all pass).
2. Full live pull could not be completed here because this sandbox currently cannot resolve external DNS hosts.
3. On a normal networked machine/server, each run writes `uk_ets/Data_auto_uk/automation_manifest.json` with per-source success/failure and date coverage.

### 22.20.4.24 UK ETS first successful end-to-end run

Command:
1. `python uk_ets/scripts/run_uk_automated_pipeline_once.py --config uk_ets/config/uk_ets_default.yaml --data-dir uk_ets/Data_auto_uk --target-mode returns`

Run:
1. `runs/20260305_010646_a4c05f`

Bootstrap outcome:
1. `11 ok / 0 failed` (UK bootstrap completed successfully).
2. UK target now spans `2021-05-19` to `2026-03-04` with `348` rows.
3. UK ICAP system detected and used: `id=31` (`United Kingdom Emissions Trading Scheme (download)`).
4. UK auction proxy generated from ICAP primary market: `116` rows.

Experiment outcome:
1. Completed through baselines + TSM + evaluation artifacts.
2. LLM refinement was skipped because no API key was provided in the run environment.
3. Test sample count is currently very small (`n=4`) under this UK config/data window.

Path MSE (`n=4`):
1. `linear_ridge`: `18.909238`
2. `linear_lasso`: `21.532584`
3. `tsm`: `27.622862`
4. `seasonal_naive`: `28.328791`
5. `naive_persistence`: `30.383854`

TSM horizon MSE:
1. `h1=3.701384`
2. `h5=57.533859`
3. `h20=3.833452`
4. `h30=6.706131`

Artifacts:
1. `uk_ets/Data_auto_uk/automation_manifest.json`
2. `runs/20260305_010646_a4c05f/results/path_metrics.csv`
3. `runs/20260305_010646_a4c05f/results/metrics_by_horizon.csv`

### 22.20.4.25 UK ETS no-reCAPTCHA auction-feature pack (production-safe path)

Goal:
1. Remove hard dependence on ICE reCAPTCHA-gated auction payloads for daily automated operation.
2. Keep auction-like signal strength using only automatable feeds.
3. Keep EU/main pipeline untouched.

Code changes:
1. `uk_ets/scripts/bootstrap_uk_ets_data_sources.py`
   - Added `uk_auction_proxy_feature_pack` step.
   - Generates `auction-proxy-features/uk_icap_primary_secondary_feature_pack.csv` from ICAP UK primary/secondary columns.
   - Feature pack includes: print-day flags, primary/secondary spread, spread %, lagged primary/spread, days-since-primary-print, cyclical weekday encodings.
2. `src/data/panel.py`
   - Added optional loader/merge path for `include_auction_proxy_pack`.
   - Added filtering logic to honor configured `auction_features` (e.g., omit `volume` cleanly).
3. `uk_ets/config/uk_ets_default.yaml`
   - `auction_features` now set to `["price", "is_auction_day"]` (volume removed).
   - Enabled `include_auction_proxy_pack: true` with default path:
     `auction-proxy-features/uk_icap_primary_secondary_feature_pack.csv`.
   - Added `preferred_feature_order` entries for UK ICAP proxy-pack fields.
4. `uk_ets/README.md`
   - Documented the new production-safe ICAP-derived auction feature path.

Bootstrap verification (no ICE token):
1. Command:
   - `python uk_ets/scripts/bootstrap_uk_ets_data_sources.py --output-dir uk_ets/Data_auto_uk --start-date 2021-05-19 --end-date 2026-03-05`
2. Outcome:
   - `13 ok / 0 failed`
   - ICE actual feed still gated; ICAP auction proxy fallback used.
   - New feature pack generated:
     `uk_ets/Data_auto_uk/auction-proxy-features/uk_icap_primary_secondary_feature_pack.csv`
   - Coverage: `1667` daily rows (`2021-05-19` to `2025-12-10`).

End-to-end run on updated UK config:
1. Command:
   - `python uk_ets/scripts/run_uk_automated_pipeline_once.py --config uk_ets/config/uk_ets_default.yaml --data-dir uk_ets/Data_auto_uk --target-mode returns --skip-bootstrap`
2. Run:
   - `runs/20260305_014353_708917`
3. Panel/build confirmation:
   - Panel size: `348 x 60`
   - Logged merge of proxy pack columns into panel.
   - Window split still has very small held-out test set (`n=4` windows), so metrics are noisy.

Path MSE (`n=4`):
1. `linear_ridge`: `18.909238`
2. `linear_lasso`: `21.532584`
3. `naive_persistence`: `30.383854`
4. `seasonal_naive`: `28.328791`
5. `tsm`: `51.165100`

TSM horizon MSE:
1. `h1=4.553787`
2. `h5=85.610771`
3. `h20=8.491344`
4. `h30=5.170843`

Artifacts:
1. `uk_ets/Data_auto_uk/auction-proxy-features/uk_icap_primary_secondary_feature_pack.csv`
2. `uk_ets/Data_auto_uk/automation_manifest.json`
3. `runs/20260305_014353_708917/results/path_metrics.csv`
4. `runs/20260305_014353_708917/results/metrics_by_horizon.csv`

### 22.20.4.26 UK ETS full run with local LLM key (`deo`)

Goal:
1. Execute a full UK run with LLM refinement enabled using the local endpoint credentials used in the EU branch.

Bootstrap command:
1. `python uk_ets/scripts/bootstrap_uk_ets_data_sources.py --output-dir uk_ets/Data_auto_uk --start-date 2021-05-19 --end-date 2026-03-05`

Experiment command (clean run):
1. `python - <<'PY'`
2. `from src.run_experiment import run_experiment`
3. `run_experiment(config_path='uk_ets/config/uk_ets_default.yaml', overrides={'target': {'mode': 'returns'}, 'output': {'write_project_paper': False}, 'compute': {'num_workers': 0}, 'llm': {'api_key': 'deo', 'base_url': 'http://192.168.1.140:9877/v1', 'model': 'qwen3-vl-4b-gpu'}}, data_dir='uk_ets/Data_auto_uk')`
4. `PY`

Run:
1. `runs/20260305_015229_26ddb4`

Notes:
1. `num_workers=0` was required in this shell-launch context to avoid Python multiprocessing worker-spawn issues.
2. LLM stage executed successfully for `DP`, `CoT`, `CoT-RF`, and `TSM+LLM`.
3. Test set remains very small (`n=4`), so metrics are high-variance.

Path MSE:
1. `linear_ridge`: `18.909238` (`full`)
2. `linear_lasso`: `21.532584` (`full`)
3. `seasonal_naive`: `28.328791` (`full`)
4. `naive_persistence`: `30.383854` (`full`)
5. `TSM+LLM`: `41.149824` (`llm_subset`)
6. `tsm`: `51.165100` (`full`)
7. `CoT-RF`: `53.972899` (`llm_subset`)
8. `CoT`: `86.472727` (`llm_subset`)
9. `DP`: `205.791006` (`llm_subset`)

TSM horizon MSE:
1. `h1=4.553787`
2. `h5=85.610771`
3. `h20=8.491344`
4. `h30=5.170843`

TSM+LLM horizon MSE:
1. `h1=3.198451`
2. `h5=56.668112`
3. `h20=22.723254`
4. `h30=35.034223`

Artifacts:
1. `runs/20260305_015229_26ddb4/results/path_metrics.csv`
2. `runs/20260305_015229_26ddb4/results/metrics_by_horizon.csv`
3. `runs/20260305_015229_26ddb4/llm/logs/llm_calls.jsonl`

### 22.20.4.27 UK target-density fix (sparse ICAP seed replaced with full Investing pull)

Issue found:
1. UK target coverage was only `348` rows over ~5 years, which is too sparse for robust modeling.
2. Root cause: UKA target builder mostly inherited ICAP UK cadence (`~Mon/Wed`) and only appended a small Investing page window (`~20` rows).

Fix applied:
1. Updated `uk_ets/scripts/bootstrap_uk_ets_data_sources.py` target step to use Investing legacy endpoint:
   - `https://www.investing.com/instruments/HistoricalDataAjax`
2. It now requests full date-range history using instrument metadata from page `__NEXT_DATA__` and parses the full table payload.
3. Existing fallback parser remains in place if full pull fails.

Validation:
1. Bootstrap rerun:
   - `python uk_ets/scripts/bootstrap_uk_ets_data_sources.py --output-dir uk_ets/Data_auto_uk --start-date 2021-05-19 --end-date 2026-03-05`
2. New UKA target coverage:
   - `rows=1233`
   - `min_date=2021-05-19`
   - `max_date=2026-03-04`
   - source recorded as `https://www.investing.com/instruments/HistoricalDataAjax`
3. Recomputed split/window sizes on refreshed UK data:
   - panel rows: `1233`
   - windows created: `1084`
   - train windows: `654`
   - val windows: `228`
   - test windows: `144`
