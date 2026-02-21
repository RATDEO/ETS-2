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

