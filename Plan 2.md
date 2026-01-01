
# EU ETS Futures Forecasting with TSM + LLM Refinement (Meta-DTS-style) — Implementation Plan

## 0) Objective and deliverables

### Goal
Reproduce (as closely as feasible) the methodology of **“Can Large Language Models forecast carbon price movements?”** but **for EU ETS (EUA futures)** using your own datasets, and optionally add an econometric quantile/feature-selection track inspired by **“Carbon prices forecasting in quantiles”**.

### Primary deliverables
1. **Reproducible codebase** that:
   - ingests your datasets (listed below),
   - builds a daily aligned panel,
   - trains a strong **30-step daily forecasting model** (TSM baseline),
   - applies **LLM-based refinement** to the 30-step forecast path,
   - evaluates at horizons **1, 5, 20, 30** days ahead,
   - runs ablations and robustness checks.
2. A **Jupyter Notebook "paper"** (`paper.ipynb`) that:
   - presents the research as an interactive document with embedded visualizations,
   - documents data sources, preprocessing steps, and methodology,
   - displays key results tables and figures inline with executable code,
   - includes model comparisons and statistical significance tests,
   - can be opened and run in VS Code or Jupyter Lab for full interactivity.

---

## 1) Repository layout and conventions

### Expected repo structure
- `Data/` (provided)
- `src/`
  - `config/`
  - `data/`
  - `models/`
  - `llm/`
  - `eval/`
  - `paper/`
  - `utils/`
- `runs/` (timestamped experiment outputs)
- `paper/` (generated manuscript + assets)
- `requirements.txt` or `pyproject.toml`

### Fixed conventions
- All timestamps stored as `datetime64[ns]` and normalized to **UTC midnight** for daily series.
- All prices stored in **EUR** (see FX conversion section).
- All features joined to a **single daily calendar** with a chosen primary trading calendar.

---

## 2) Data inventory (your inputs)

### Provided raw files
- `Data/carbon-market-indices/`
  - `GRN_history.csv`, `KCCA_history.csv`, `KEUA_history.csv`, `KRBN_history.csv`, `KSET_history.csv`
- `Data/emission-spot-primary-market-auction/`
  - auction reports 2017–2025 (CSV per year)
- `Data/energy-benchmarks/`
  - `eu-brent-spot-usd.csv`
  - `rotterdam-coal-futures-usd.csv`
- `Data/icap-allowance-price-explorer-secondary-market/`
  - `icap-graph-price-data-2005-03-09-2018-12-28.csv`
  - `icap-graph-price-data-2019-01-02-2025-11-27.csv`
  - plus an image (ignored unless needed)

### Target series (must be added / defined)
- EUA futures contract prices (your downloaded futures contract data):
  - Put under `Data/eua-futures/` (e.g., `EUA_frontmonth.csv` or similar)
  - Must include: `date`, `close` (or settlement), and optional volume/open interest.

**Action**: ensure the futures file exists and name it explicitly in `config/data.yaml`.

---

## 3) Step-by-step execution plan

### Step 1 — Project configuration and reproducibility
**Implement**
- `src/config/default.yaml` with:
  - `target.instrument = "EUA_FUTURES"`
  - `target.column = "close_eur"`
  - `freq = "D"`
  - `horizons = [1, 5, 20, 30]`
  - `pred_len = 30` (mandatory)
  - `seq_len`, `label_len` (to be tuned; defaults below)
  - `train/val/test` split dates (rolling origin)
  - random seeds, device settings

**Outputs**
- `runs/<run_id>/config_resolved.yaml`
- `runs/<run_id>/reproducibility.md` (seed, env, git hash)

---

### Step 2 — Data loaders (raw → standardized)
**Implement standardized loaders** returning a DataFrame with columns:
- `date` (daily), `value` (float), `series_name` (string), `currency` (optional), `source`

**Files to create**
- `src/data/load_indices.py`
- `src/data/load_auctions.py`
- `src/data/load_energy.py`
- `src/data/load_icap.py`
- `src/data/load_eua_futures.py`

**Key requirements**
- Robust parsing: date formats, thousands separators, missing rows.
- Deduplicate dates by choosing last/settlement/close consistently.
- Store raw snapshots in `runs/<run_id>/data/raw_cache/`.

**Outputs**
- `runs/<run_id>/data/standardized/*.parquet`

---

### Step 3 — FX conversion (USD → EUR)
All USD-priced series must be converted to EUR prior to modeling.

**Inputs needed**
- EURUSD daily FX rate series (spot). If not provided, add:
  - `Data/fx/EURUSD.csv` (date, EURUSD)
  - or implement a connector to a licensed provider (preferred in production).

**Conversion rule**
- Euro to USD conversion rates are included in usd.xml

**Implement**
- `src/data/fx.py`:
  - `load_eurusd()`
  - `convert_usd_to_eur(df_usd, eurusd_df, price_col, out_col="price_eur")`
  - alignment: forward-fill FX on non-trading days; log and flag gaps > 3 days.

**Applies to**
- `eu-brent-spot-usd.csv`
- `rotterdam-coal-futures-usd.csv`
- any other USD series discovered during profiling

**Outputs**
- standardized EUR series parquet + an audit:
  - `runs/<run_id>/data/fx_audit.csv` (coverage, gaps, ffill counts)

---

### Step 4 — Daily alignment and feature engineering (panel build)
**Objective**
Build a single daily table `panel.csv/parquet` with:
- `date`
- target: `y = EUA_futures_close_eur`
- exogenous predictors: indices, auctions, energy, ICAP secondary

**Implement**
- `src/data/panel.py`:
  - choose a master calendar: dates present in the EUA futures series
  - left-join all features to that calendar
  - missing handling:
    - for continuous market series: forward-fill up to `k` days, else NA
    - for auctions: create event features (see below), do NOT forward-fill price naively

**Auction feature design**
From auction reports, produce:
- `auction_price_eur` (on auction dates)
- `auction_volume` / `total_allowances` (if available)
- `is_auction_day` (binary)
- lagged auction price features: `auction_price_lag_1`, `lag_5`, `lag_20`
- rolling stats (if dense enough): `auction_price_rolling_mean_20`

**Energy feature design**
- Brent EUR spot: level + returns
- Coal EUR futures: level + returns
- spreads: e.g., coal vs brent ratio or difference (optional)

**Index feature design**
- levels + log returns (daily)
- 20d rolling vol

**ICAP feature design**
- align ICAP daily (if sparse, treat as slow-moving proxy); compute returns where possible

**Outputs**
- `runs/<run_id>/data/panel.parquet`
- `runs/<run_id>/data/panel_schema.json`
- `runs/<run_id>/paper/fig_data_coverage.png` (coverage heatmap)

---

### Step 5 — Target framing: 30-step sequence forecasting on daily data
**Objective**
Create supervised learning datasets for:
- input window length `seq_len` (e.g., 120 trading days default)
- label context `label_len` (e.g., 30)
- output horizon `pred_len = 30`

**Implement**
- `src/data/windows.py`:
  - `make_windows(panel, target_col, feature_cols, seq_len, label_len, pred_len)`
  - support both:
    - `S` (univariate) baseline: only target history
    - `MS` (multivariate → univariate): target + exogenous
- standard scaling:
  - fit scaler on train only
  - store scaler object per run

**Outputs**
- `runs/<run_id>/data/datasets/{train,val,test}.npz`
- `runs/<run_id>/data/scaler.pkl`

---

### Step 6 — Baselines (must-have for credibility)
Implement and evaluate these baselines before deep models:
1. **Naive persistence**: `y_hat[t+h] = y[t]`
2. **Seasonal naive** (optional): `y_hat[t+h] = y[t-5]` (weekly)
3. **ARIMA/SARIMAX** (optional): for sanity check
4. **Linear regression / ridge** on lagged features (fast benchmark)

**Implement**
- `src/models/baselines.py`
- `src/eval/metrics.py`

**Outputs**
- `runs/<run_id>/results/baselines_metrics.csv`
- paper table: “Baseline performance”

---

### Step 7 — TSM baseline: Autoformer (or equivalent)
**Objective**
Train a model producing a **30-step forecast path** each day.

**Implement**
Option A (recommended): use `pytorch-forecasting` or a clean modern TS library
- Pros: easier integration, less legacy code friction
Option B (paper-faithful): integrate THUML Autoformer
- Pros: closer to original paper baseline

Regardless of option:
- must support multivariate inputs and 30-step outputs.

**Training protocol**
- rolling-origin evaluation:
  - train on [start, T_train_end]
  - validate on next block
  - test on final block
- store checkpoints, predictions, and training curves

**Outputs**
- `runs/<run_id>/models/tsm_checkpoint.pt`
- `runs/<run_id>/predictions/tsm_pred_test.parquet` with columns:
  - `date_t`, `y_true_t_plus_1..30`, `yhat_t_plus_1..30`

---

### Step 8 — LLM refinement layer (Meta-DTS-style)
**Objective**
Given:
- recent observed target history (and optionally key exogenous summaries),
- a TSM 30-step forecast path,
produce a refined 30-step forecast path.

#### 8.1 Prompt tasks (mirror original paper structure)
Implement the following variants as separate “methods”:
- **DP** (direct prompting): LLM forecasts next 30 days from history alone
- **CoT**: same as DP but with chain-of-thought style reasoning prompt
- **CoT-RF**: CoT + self-refinement pass (LLM critiques and revises)
- **TSM+LLM (main)**: LLM refines the TSM path (the key method)

Optional:
- **CoT-Sent / CoT-Sent-RF** if you include headline sentiment features

#### 8.2 I/O format (strict)
- Input serialization:
  - past `seq_len` days of `y` (and optionally 3–6 engineered summary features)
  - TSM forecast: vector length 30
  - specify currency EUR and date anchors
- Output must be parseable as JSON:
  - `{"yhat":[...30 floats...]}`
- Implement robust parsing and fallback:
  - if parse fails: retry with a stricter format instruction
  - log failures and exclude from metrics only if unavoidable

**Implement**
- `src/llm/prompts.py` (templates)
- `src/llm/refine.py` (calls, parsing, retries)
- `src/llm/cache.py` (hash-based caching of LLM calls)

**Cost control**
- run LLM only on validation/test windows
- cache all completions
- limit context length by:
  - compressing history (e.g., last 120 points)
  - using summary stats for exogenous variables

**Outputs**
- `runs/<run_id>/predictions/llm_refined_test.parquet`
- `runs/<run_id>/llm/logs/*.jsonl` (inputs/outputs hashed)

---

### Step 9 — Evaluation (match original paper as closely as possible)
**Primary regression metric**
- **MSE** for the predicted prices at:
  - h = 1, 5, 20, 30
- Also report:
  - averaged MSE across steps 1..30 (optional “global” metric)

**Trend classification metric**
- Define a 3-way label for each horizon:
  - up / flat / down using a transparent threshold:
    - `flat` if `|Δ| < τ`, else sign of `Δ`
  - choose τ as a small fraction of recent volatility (e.g., 0.25 * rolling 20d std of returns)
- Report **Accuracy** at horizons 1/5/20/30.

**Significance tests (paper-aligned)**
- paired **t-test** on per-window squared errors (method vs baseline)
- **Wilcoxon signed-rank** on per-window squared errors

**Implement**
- `src/eval/metrics.py`
- `src/eval/significance.py`
- `src/eval/report_tables.py`

**Outputs**
- `runs/<run_id>/results/metrics_by_horizon.csv`
- `runs/<run_id>/results/significance_tests.csv`
- figures:
  - error vs horizon
  - sample forecast paths (TSM vs LLM-refined vs truth)

---

### Step 10 — Robustness checks (must include)
1. **Noise-injection test** on TSM forecast input to LLM:
   - add 5%, 10%, 20%, 30% noise to the TSM path
   - rerun LLM refinement
   - show MSE degradation curve
2. **Ablations**
   - TSM only
   - LLM only (DP/CoT)
   - TSM + LLM refine
   - with/without exogenous features
3. **Temporal stability**
   - evaluate across subperiods (e.g., 2017–2019, 2020–2021, 2022–2025)

**Outputs**
- robustness tables + plots under `runs/<run_id>/results/robustness/`

---

### Step 11 — Optional econometric track (quantile + feature selection)
**Objective**
Add interpretability and tail behavior analysis.
- Quantile regression at q ∈ {0.1, 0.5, 0.9} (or a grid)
- Penalized selection (LASSO/group lasso) on exogenous + lags
- Report quantile loss and drivers by quantile

**Implement**
- `src/models/quantile_lasso.py`
- `src/eval/quantile_metrics.py`

**Outputs**
- coefficient paths, selected features by quantile
- quantile-forecast performance vs baselines

---

## 4) Auto-generated paper (Jupyter Notebook format)

### Paper build system
Generate a Jupyter Notebook (`paper.ipynb`) that presents the research interactively:
- Executable code cells that load and visualize data
- Markdown cells with methodology and interpretation
- Inline tables and figures with full interactivity
- Can be opened directly in VS Code or Jupyter Lab

**Implement**
- `src/paper/paper_writer.py`
  - appends sections as pipeline stages complete
  - inserts run metadata, split dates, dataset stats, and metrics tables
  - includes a “Comparison to original paper” section:
    - same metric types (MSE + Accuracy)
    - same ablations
    - similar robustness narrative (noise injection)

### Paper outline (generated)
1. Abstract
2. Introduction (problem + why EU ETS)
3. Data (sources, coverage, FX conversion, preprocessing)
4. Methods
   - Baselines
   - TSM model (30-step)
   - LLM prompting + refinement
   - Evaluation protocol and tests
5. Results
   - MSE by horizon (1/5/20/30)
   - Trend accuracy by horizon
   - Significance tests
   - Robustness (noise injection, ablations)
6. Discussion
   - what improves, where it fails, regime notes
7. Conclusion
8. Reproducibility appendix (config, seeds, compute)
9. References

**Outputs**
- `paper/manuscript.md`
- `paper/manuscript.pdf` (if enabled)
- `runs/<run_id>/paper_snapshot/` (frozen copy)

---

## 5) Experiment orchestration

### Single command runner
Create:
- `python -m src.run_experiment --config src/config/default.yaml`

It should execute:
1. load/standardize data
2. FX convert USD→EUR
3. build panel + windows
4. train baselines + TSM
5. generate TSM predictions (val/test)
6. run LLM refinement methods
7. evaluate + significance tests
8. robustness suite
9. write paper

**Outputs**
- Everything stored under `runs/<run_id>/`

---

## 6) Critical design decisions to hard-code early

1. **Calendar and missingness**
   - master calendar = EUA futures dates
   - forward-fill rules per series type documented in paper

2. **Scaling**
   - fit only on train
   - store scalers and inverse-transform predictions for price-level MSE

3. **Data leakage prevention**
   - strictly ensure exogenous features at date `t` do not incorporate information from `t+1..t+30`

4. **LLM determinism**
   - set temperature low (0–0.2) for refinement
   - log prompts and outputs; cache all calls

5. **Versioning**
   - log git hash, config, package versions
   - store all raw input snapshots in run folder

---

## 7) Acceptance criteria (“Definition of done”)

A run is considered complete when:
- All datasets load successfully and are converted to EUR where needed
- A daily 30-step TSM model trains and produces test forecasts
- LLM refinement runs end-to-end with cached logs and parseable outputs
- Metrics tables include MSE and Accuracy at 1/5/20/30
- t-test and Wilcoxon comparisons are reported
- Noise-injection robustness plot is produced
- `paper/manuscript.md` is generated and references all key artifacts

---

## 8) Immediate next actions for the coding LLM

1. Create repo scaffolding and config system
2. Implement loaders + FX conversion + panel builder
3. Implement baselines and evaluation metrics
4. Implement a TSM model that outputs 30-step paths
5. Implement LLM refinement with strict JSON output and caching
6. Generate the initial manuscript sections from pipeline metadata
7. Iterate on seq_len/label_len and feature sets via config sweeps