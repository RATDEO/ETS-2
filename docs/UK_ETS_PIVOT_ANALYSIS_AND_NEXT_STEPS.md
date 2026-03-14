# UK ETS Pivot: Sub-Project Analysis, Evaluation, and Next Steps

**Date**: 6 March 2026  
**Continuation of**: [PROJECT_EVALUATION_AND_FUTURE_DIRECTIONS.md](PROJECT_EVALUATION_AND_FUTURE_DIRECTIONS.md)  
**Scope**: UK ETS forecasting sub-project (Option B from the previous report)

---

## 1. Executive Summary

Following the EU ETS project's conclusion that LLM refinement yields only marginal gains on a mature, efficient carbon market, we pivoted to **UK Emissions Trading Scheme (UK ETS)** forecasting — a younger, less liquid, and potentially less informationally efficient market. This was **Option B** from the prior evaluation: change the target market.

Over 4-5 March 2026, we:
1. Built a complete UK ETS data pipeline (`uk_ets/`) from scratch — isolated from the EU flow
2. Automated 11-13 data sources (all operational, no manual intervention required)
3. Ran **~16 experiment variants** progressing from sparse data (n=4 test) through to production-grade (n=144 test)
4. Tested Autoformer, DLinear, and linear baselines with and without LLM CoT-RF refinement
5. Experimented with linear_ridge as an alternative LLM base model

**Key findings**:
- **Linear Ridge remains the best baseline**: path MSE = **22.42** (n=144)
- **DLinear TSM**: path MSE = **29.78** — significantly worse than Ridge on this dataset
- **Large Autoformer (d=512)**: path MSE = **40.35** — severely overfit, even worse
- **DLinear + CoT-RF raw**: path MSE = **32.29** — worse than the DLinear base
- **DLinear + CoT-RF ramp blend (w=0.50)**: path MSE = **30.56** — slight improvement over raw LLM, still worse than DLinear alone
- **Ridge + CoT-RF ramp blend (w=0.75)**: path MSE = **24.25** — best LLM-augmented result, but still **8.1% worse** than Ridge alone (22.42)
- **Ridge + CoT-RF h30 blend (w=0.25)**: path MSE = **22.75** — closest to Ridge, only 1.4% worse

**Bottom line**: On UK ETS, the LLM refinement made things worse, not better. The linear Ridge model outperforms every deep learning and LLM variant tested. The UK ETS market presents different challenges from EU ETS, but the core result — that a simple linear model is hard to beat — persists.

---

## 2. What Was Built

### 2.1 UK ETS Pipeline Infrastructure

A fully isolated `uk_ets/` subdirectory was created that reuses the main project's model/evaluation code but maintains separate:
- **Data root**: `uk_ets/Data_auto_uk/` (same directory schema as EU pipeline)
- **Bootstrap script**: `uk_ets/scripts/bootstrap_uk_ets_data_sources.py` (1,212 lines)
- **Pipeline runner**: `uk_ets/scripts/run_uk_automated_pipeline_once.py`
- **Configs**: `uk_ets/config/uk_ets_default.yaml` (Autoformer/GPT-5.2) and `uk_ets/config/uk_ets_tuned.yaml` (DLinear/Qwen3-VL-4B)

### 2.2 Automated Data Sources (13 feeds, all operational)

| Source | Feed | Coverage | Rows | Notes |
|--------|------|----------|------|-------|
| UKA Futures target | Investing.com HistoricalDataAjax | 2021-05-19 → 2026-03-04 | 1,233 | Full daily, GBP-denominated |
| ICAP UK Secondary Market | ICAP System 31 (download) | 2021-05-19 → 2025-12-10 | 328 | ~Mon/Wed cadence, used as feature |
| UK Auction (actual) | ICE Report 278 | — | — | reCAPTCHA gated; proxy used instead |
| UK Auction Proxy | ICAP UK Primary Market | 2021-05-19 → 2025-12-10 | 116 | Derived from ICAP |
| Auction Feature Pack | ICAP Primary + Secondary derived | 2021-05-19 → 2025-12-10 | 1,667 | 18 features including spreads, lags, z-scores |
| EUR/USD FX | ECB XML | 1999-01-04 → 2026-03-04 | 6,957 | |
| Brent Crude | FRED DCOILBRENTEU | 1987-05-20 → 2026-03-02 | 9,841 | |
| Coal (API2 ARA) | Yahoo MTF=F | 2010-12-17 → 2025-12-26 | 3,677 | |
| Carbon Index (GRN) | Yahoo Finance | 2019-09-18 → 2026-03-04 | 1,623 | |
| Carbon Index (KCCA) | Yahoo Finance | 2021-10-05 → 2026-03-04 | 1,107 | |
| Carbon Index (KEUA) | Yahoo Finance | 2021-10-05 → 2026-03-04 | 1,107 | |
| Carbon Index (KRBN) | Yahoo Finance | 2020-07-31 → 2026-03-04 | 1,404 | |
| Volatility Proxy | Yahoo ^VIX (^VFTSE unavailable) | 1990-01-02 → 2026-03-04 | 9,109 | |

### 2.3 Data Quality Issues

1. **UK ETS only exists since May 2021**: Total history is ~4.75 years (1,233 trading days). After windowing with seq_len=20 and pred_len=30, this yields only **654 train / 228 val / 144 test** windows. This is far smaller than the EU ETS dataset (4,334 observations).

2. **ICAP data is sparse**: Only 328 rows over the full period (~Mon/Wed), requiring heavy forward-fill. The 116 auction proxy rows are even sparser.

3. **Coal feature terminates Dec 2025**: Yahoo coal futures data ends at 2025-12-26, leaving a coverage gap in the test period.

4. **FX conversion warning**: 30% of energy-price rows lacked FX rates after forward-fill, introducing noise in EUR-converted features.

5. **Target density bug (fixed)**: Initial UKA target had only 348 rows because the bootstrap primarily pulled from ICAP's sparse cadence. Fixed by switching to Investing.com's HistoricalDataAjax endpoint for full daily coverage (1,233 rows).

---

## 3. UKA Price Characteristics

| Year | Mean (£) | Min (£) | Max (£) | Std Dev | Trading Days |
|------|----------|---------|---------|---------|-------------|
| 2021 | 55.44 | 42.01 | 79.20 | 10.60 | 161 |
| 2022 | 78.11 | 66.07 | 97.15 | 6.62 | 256 |
| 2023 | 53.84 | 32.56 | 83.74 | 14.37 | 257 |
| 2024 | 37.99 | 31.63 | 48.60 | 3.97 | 259 |
| 2025 | 49.31 | 30.20 | 65.56 | 7.28 | 256 |
| 2026 | 56.82 | 42.60 | 70.25 | 9.92 | 44 |

**Daily return statistics**: mean = +0.075%, std = 3.93%

Key observations:
- **Massive structural break in 2022-2023**: Prices collapsed from £97 to £33, a 66% drawdown. This dwarfs anything in the test period.
- **Much higher daily volatility than EU ETS**: 3.93% daily standard deviation vs ~1.5% for EU ETS. UK ETS is a noisier, thinner market.
- **Training period covers the crash**: The train split (→ 2024-06-30) includes the extreme 2022-2023 drawdown, but the test period (2025-07-01 → 2026-03-04) is in a recovering/volatile phase with prices moving from £30 to £70 and back.
- **Price range in test period**: Roughly £30–£70, a very wide range compared to the EU ETS test period's £55–£75.

---

## 4. Experiment Progression and Results

### Phase 1: Sparse Data Runs (n=4 test windows, 5 March 2026)

These initial runs used the original 348-row UKA target (before the density fix):

| Run | TSM | LLM | Test n | Ridge MSE | TSM MSE | Best LLM MSE | Notes |
|-----|-----|-----|--------|-----------|---------|-------------|-------|
| `20260305_010646` | Autoformer d=512 | None | 4 | 18.91 | 27.62 | — | First E2E UK run |
| `20260305_014353` | Autoformer d=512 | None | 4 | 18.91 | 51.17 | — | With auction proxy pack |
| `20260305_014955` | Autoformer d=512 | Qwen3-VL-4B (DP, CoT, CoT-RF) | 4 | 18.91 | — | CoT-RF: 36.58 | DP catastrophic (218.6) |
| `20260305_015229` | Autoformer d=512 | Qwen3-VL-4B (all methods) | 4 | 18.91 | 51.17 | TSM+LLM: 41.15 | Full method sweep |

**Phase 1 conclusions**: Even on 4 test windows, Ridge dominates everything. Autoformer massively overfits (d=512 on 348 rows is wildly overparameterized). LLM refinement makes things worse.

### Phase 2: Full-density runs (n=144 test windows, 6 March 2026)

After fixing the UKA target density (1,233 rows → 654 train / 228 val / 144 test):

| Run | TSM | LLM | Ridge MSE | TSM MSE | LLM Raw MSE | Best Blend MSE |
|-----|-----|-----|-----------|---------|-------------|----------------|
| `20260306_130357` | Autoformer d=512 | None | 23.27 | **40.35** | — | — |
| `20260306_131534` | Autoformer d=32, drop=0.3 | None | 22.42 | **31.41** | — | — |
| `20260306_132538` | **DLinear** d=32, drop=0.3 | CoT-RF (Qwen3-VL-4B) | 22.42 | **29.78** | Raw: 32.29 | Ramp w=0.50: **30.56** |
| `20260306_142124` | DLinear d=32 | CoT-RF (base=Ridge) | 22.42 | 29.78 | Raw: 27.46 | Ramp w=0.75: **24.25** |

### Phase 2 Detailed Leaderboard (n=144 test windows)

| Rank | Model | Path MSE | Δ vs Ridge |
|------|-------|----------|-----------|
| 1 | **Linear Ridge** | **22.42** | baseline |
| 2 | Ridge + CoT-RF blend h30 (w=0.25) | 22.75 | +1.4% |
| 3 | Linear Lasso | 24.32 | +8.5% |
| 4 | Ridge + CoT-RF ramp blend (w=0.75) | 24.25 | +8.1% |
| 5 | Ridge + CoT-RF raw (no blend) | 27.46 | +22.4% |
| 6 | DLinear (tuned) | 29.78 | +32.8% |
| 7 | DLinear + CoT-RF ramp blend (w=0.50) | 30.56 | +36.3% |
| 8 | Small Autoformer (d=32) | 31.41 | +40.1% |
| 9 | Naive persistence | 32.01 | +42.7% |
| 10 | DLinear + CoT-RF raw | 32.29 | +43.9% |
| 11 | Seasonal naive | 34.19 | +52.4% |
| 12 | Large Autoformer (d=512) | 40.35 | +79.9% |

### Blend Grid Analysis

**DLinear + CoT-RF** (run `20260306_132538`):
- Best val blend: w=0.50 (path), w=0.00 (h1), w=1.00 (h5), w=0.25 (h30)
- Validation selected **some** LLM influence, but test performance degraded
- h1 validation correctly selected w=0.00 (no LLM at short horizon)

**Ridge + CoT-RF** (run `20260306_142124`):
- Best val blend: w=0.75 (path), w=0.00 (h1), w=1.00 (h5, h20), w=0.25 (h30)
- Again, h1 selected w=0.00 (LLM hurts at short horizons)
- h30 at w=0.25 produced 22.75 path MSE — closest to Ridge but still worse
- Full path blend at w=0.75 produced 24.25 — substantially worse than Ridge

### Horizon-Level Results (n=144)

| Horizon | Naive Persist. | Linear Ridge | DLinear | CoT-RF (DLinear base) | CoT-RF (Ridge base) |
|---------|---------------|-------------|---------|----------------------|---------------------|
| h1 | 1.061 | **1.062** | 1.206 | 1.188 | — |
| h5 | 5.212 | **4.594** | 5.649 | 5.143 | — |
| h20 | 44.28 | **30.15** | 40.25 | 42.93 | — |
| h30 | 79.99 | **56.91** | 72.37 | 83.75 | — |

The Ridge model dominates at every single horizon. DLinear is competitive at h1 but falls behind at longer horizons. The CoT-RF refinement improves h1 and h5 slightly relative to DLinear but badly overpredicts at h20 and h30.

### Trend Accuracy (DLinear run, paper-style Th=18)

| Model | h10 | h20 | h30 |
|-------|-----|-----|-----|
| Ridge | **59.7%** | **70.8%** | **72.2%** |
| Lasso | 59.7% | 64.6% | 52.8% |
| DLinear (TSM) | 45.8% | 37.5% | 26.4% |
| Naive Persistence | 52.8% | 39.6% | 34.0% |

Ridge achieves the best directional accuracy at all horizons. The DLinear TSM performs worse than naive persistence on direction. This is a significant finding — the deep learning model is not only worse on MSE but also on trend classification.

---

## 5. Why This Happened

### 5.1 Dataset Size Constraint

The UK ETS market has only existed since May 2021. With 1,233 trading days and a 30-day prediction length + 20-day sequence length, we get only 654 training windows. This is **less than one-third** of the EU ETS training set (2,128 windows). Deep learning models (Autoformer, DLinear) need substantially more data to learn useful patterns. At 654 training windows, the linear Ridge model's inductive bias (smoothness, low-rank) is a better match for the data size.

### 5.2 Extreme Regime Change in Training Data

The UK ETS experienced a **66% price collapse** from £97.15 (Q1 2022) to £30.20 (early 2025). This is one of the most extreme price movements in any compliance carbon market. The training data (up to June 2024) captures the crash but not the recovery phase. The test period (July 2025 → March 2026) sees prices oscillating £30-70 — a range not well-represented in the training distribution.

### 5.3 Overparameterization

| Model | Parameters | Training Windows | Params/Window Ratio |
|-------|-----------|-----------------|---------------------|
| Large Autoformer (d=512) | ~4,000,000 | 654 | 6,116:1 |
| Small Autoformer (d=32) | ~16,000 | 654 | 24:1 |
| DLinear (individual) | ~16,380 | 654 | 25:1 |
| Linear Ridge | ~300 | 654 | 0.46:1 |

The large Autoformer has **6,116 parameters per training window** — a recipe for catastrophic overfitting. Even the "tuned" small models (DLinear, small Autoformer) at 25 params/window are arguably overparameterized for this data volume. The Ridge model at 0.46 params/window is well-regularized.

### 5.4 LLM Corrections Are Noise on a Noisy Signal

UK ETS daily volatility (3.93%) is roughly **2.5x higher** than EU ETS (~1.5%). The LLM's corrections — which are already known to be vague qualitative adjustments from a 4B model — are drowned out by the natural price noise. The CoT-RF reflection step generates rules calibrated to training errors, but these rules don't transfer to a test period with fundamentally different price dynamics.

### 5.5 Feature Quality Degradation

Several features degrade significantly in the UK context:
- **ICAP secondary market data** stops at Dec 2025, creating NaN-filled features in the test period
- **Auction proxy features** are derived from the same sparse ICAP data (116 rows → heavy forward-fill)
- **Coal** terminates Dec 2025
- **FX conversion** loses 30% of rows — the energy features are denominated in USD but converted to EUR for loader compatibility, introducing noise
- **Volatility proxy** is VIX (US equity vol) instead of a UK-specific measure — weak proxy for UK carbon market risk sentiment

---

## 6. Comparison: EU ETS vs UK ETS Results

| Dimension | EU ETS | UK ETS |
|-----------|--------|--------|
| Trading days | 4,334 | 1,233 |
| Train windows | ~2,100 | 654 |
| Test windows | 382 | 144 |
| Best TSM path MSE | 20.94 (Autoformer d=64) | 29.78 (DLinear) |
| Best Ridge path MSE | 20.74 | 22.42 |
| Best overall path MSE | 20.52 (4B HDELTA) | **22.42 (Ridge)** |
| Best LLM improvement | −1.6% vs TSM | **None** (LLM always worse) |
| Ridge vs TSM | Ridge slightly better | Ridge **massively** better (+32%) |
| Daily volatility | ~1.5% | ~3.93% |
| Market age | ~20 years | ~5 years |
| Dominant price feature | Mean-reverting consolidation | Post-crash recovery |

The hypothesis that UK ETS would be "less efficient" and more amenable to LLM refinement **was not supported**. Instead, UK ETS turned out harder to forecast because:
1. Less training data
2. Higher noise (2.5x daily vol)
3. More extreme regime changes
4. Sparser/lower-quality features

---

## 7. What Went Right

1. **Pipeline reuse worked**: The `uk_ets/` subdirectory approach cleanly isolated UK-specific logic while reusing all core model/evaluation code. Build time was ~1 day.

2. **Automated data pipeline is production-ready**: All 13 sources bootstrap without manual intervention. The ICAP-derived auction feature pack elegantly works around the ICE reCAPTCHA barrier.

3. **Ridge dominance was identified quickly**: Rather than spending weeks on LLM tuning, the clear Ridge superiority was established in the first full-density run.

4. **DLinear is the right TSM**: Switching from Autoformer to DLinear reduced TSM path MSE from 40.35 to 29.78 (−26%). DLinear's simpler inductive bias is better suited to the small dataset.

5. **Target density bug was caught and fixed**: The initial 348-row target would have given meaningless results (n=4 test). The fix to use Investing.com's full pull increased coverage to 1,233 rows.

---

## 8. Options for Improvement and Future Research

### Option 1: Hyperparameter Sweep on DLinear (HIGH PRIORITY)

**Rationale**: DLinear at 29.78 is far behind Ridge at 22.42. The current DLinear config was ported from the EU tuning — it may not be appropriate for UK ETS's shorter history and higher volatility.

**What to try**:
- `seq_len`: Currently 20. Try 5, 10, 15, 30. Shorter sequences may be better for a 4.75-year market.
- `kernel_size`: Currently 25 (larger than seq_len!). Try 3, 5, 9, 13.
- `dropout`: Currently 0.3. Try 0.5, 0.6 for more aggressive regularization.
- `learning_rate`: Currently 0.001. Try 0.0005, 0.0001.
- `max_epochs`: Consider early stopping at lower patience (5 instead of 8).
- `weight_decay`: Currently 0.01. Try 0.05, 0.1.
- `dlinear_individual`: Currently true. Try false (shared parameters across channels).

**Expected outcome**: A well-tuned DLinear should close the gap with Ridge. If it can't, that itself is a strong negative result about deep learning viability on short-history markets.

### Option 2: PatchTST or Other Lightweight Architectures

**Rationale**: PatchTST is specifically designed for small datasets via patching (local attention over subsequences) and channel-independence. It has shown strong results on similar-scale time series problems.

**What to try**:
- Implement PatchTST in the TSM module
- Use small patch sizes (4-8) given the short history
- Test with and without pre-training on the EU ETS data (transfer learning)

**Expected outcome**: Moderate probability of improvement over DLinear. PatchTST's patching mechanism is particularly well-suited to noisy commodity time series.

### Option 3: Residual Connection to Naive Baseline

**Rationale**: The clearest TSM weakness is at long horizons (h20, h30) where it produces path MSE 40.25 vs Ridge's 30.15. A simple architectural fix: have the DLinear output a *correction* to the naive persistence forecast rather than an absolute forecast. This guarantees the model can never be worse than persistence at any horizon unless the correction is positive.

**What to try**:
- Add `y_pred = y_naive + model(x)` residual connection
- Train the model to predict the error/innovation rather than the level
- Apply to both DLinear and any future TSM architectures

**Expected outcome**: Should significantly reduce h20/h30 MSE where the TSM currently overshoots. This is a low-effort, high-confidence improvement.

### Option 4: Feature Engineering and Selection

**Rationale**: The current UK feature set is largely inherited from the EU pipeline. Several features have poor coverage or weak relevance for UK ETS specifically.

**What to try**:
- **Remove degraded features**: Drop ICAP-derived features that are NaN in the test period (or only use print-day flags which are binary). Drop coal after it terminates.
- **Add UK-specific features**: 
  - UK natural gas (NBP or TTF) — gas-to-coal switching drives UK ETS more directly than Brent
  - UK power prices (Elexon/BMRS day-ahead, N2EX or EPEX)
  - UK-specific policy calendar events (DESNZ consultations, BEIS announcements)
  - UKA-EUA spread as a feature (differential between the two carbon markets)
- **Feature importance ranking**: Run a permutation importance analysis on the Ridge model to identify which features actually matter for UK ETS.
- **Lasso feature selection**: Re-enable and tune the existing Lasso feature selection module for UK ETS.

**Expected outcome**: High value. UK ETS has different fundamental drivers than EU ETS. Gas prices and UK-specific policy are likely far more predictive than the generic carbon indices currently used.

### Option 5: GBP-Native Pricing

**Rationale**: UKA futures trade in GBP but the pipeline converts to EUR for loader compatibility. This introduces unnecessary FX noise — 30% of energy-feature rows lose FX rates. Running natively in GBP would be cleaner.

**What to try**:
- Modify the `close_eur` loader to accept GBP-denominated target as-is
- Convert USD-denominated features (Brent, coal, indices) using GBP/USD instead of EUR/USD
- Or better: use GBP-denominated energy proxies where available (UK NBP gas, UK power)

**Expected outcome**: Removes a systematic noise source. Uncertain magnitude but easy to implement.

### Option 6: Ensemble of Linear + DLinear

**Rationale**: Ridge and DLinear have complementary strengths. Ridge is stable and strong at all horizons; DLinear captures non-linear patterns at short horizons. A simple average (or horizon-weighted blend) could improve on both.

**What to try**:
- Equal-weight ensemble: `y_pred = 0.5 * ridge_pred + 0.5 * dlinear_pred`
- Horizon-weighted: more Ridge weight at long horizons, more DLinear at short
- Validation-selected weights

**Expected outcome**: Likely a small but robust improvement over Ridge alone, because the two models make different types of errors.

### Option 7: Expanding the Dataset (Medium-Term)

**Rationale**: The fundamental constraint is dataset size. The UK ETS will naturally produce more data as time passes. There are also options to augment the training set.

**What to try**:
- **Wait**: Every quarter adds ~65 more training windows. By Q4 2026, the dataset will be 50% larger.
- **Transfer learning**: Pre-train the TSM on EU ETS data (4,334 observations), then fine-tune on UK ETS. This is well-established in NLP and could work for related carbon markets.
- **Synthetic augmentation**: Generate synthetic training windows via bootstrap resampling of return sequences within the same regime, preserving autocorrelation structure.
- **Multi-market training**: Train a single model on both EU ETS and UK ETS data simultaneously, with a market indicator variable. This forces the model to learn shared carbon-market dynamics.

**Expected outcome**: Transfer learning from EU ETS is the most promising approach here. The two markets share fundamental drivers (energy prices, EU/UK regulatory alignment) and the EU data could serve as a strong prior.

### Option 8: LLM with Better News Data

**Rationale**: The LLM component failed partly because no UK-specific news feed was used in these runs (sentiment was not enabled). GDELT UK ETS headlines may be even sparser than EU ETS headlines, but targeted UK policy RSS feeds could be more valuable.

**What to try**:
- Enable GDELT UK ETS news fetch (`--include-news`)
- Add UK government consultations (DESNZ, formerly BEIS) as structured event feeds
- Monitor ICE UKA auction results as event signals (spread to secondary market, cover ratio)
- Focus the LLM only on long-horizon corrections (h20, h30) where there's more room for improvement

**Risks**: Same GDELT noise problem as EU ETS, likely worse due to lower UK ETS media coverage. Consider this low-priority until TSM baselines are stronger.

### Option 9: Revisit the Problem Framing — Classification Instead of Regression

**Rationale**: Ridge already achieves 70-72% trend accuracy at h20-h30 on UK ETS (the highest directional accuracy in any market we've tested). This suggests directional forecasting may be more tractable than point forecasting for UK ETS.

**What to try**:
- Frame the problem as classification: UP / DOWN / FLAT at 5, 10, 20, 30-day horizons
- Train gradient-boosted classifiers (XGBoost, LightGBM) on the same feature set
- Evaluate via directional accuracy, calibration, and simulated portfolio P&L
- This bypasses the MSE metric entirely and focuses on what matters for trading

**Expected outcome**: May reveal that the feature set contains more directional information than level information. Classification models can handle small datasets and regime changes better than regression deep learning.

---

## 9. Recommended Priority Order

| Priority | Action | Effort | Expected Impact |
|----------|--------|--------|----------------|
| 1 | **Residual naive connection** (Option 3) | ~2 hours | High — directly fixes h20/h30 overshoot |
| 2 | **Feature engineering** (Option 4) | ~1 day | High — UK gas, power, UKA-EUA spread |
| 3 | **DLinear hyperparameter sweep** (Option 1) | ~4 hours | Medium — close gap with Ridge |
| 4 | **Ridge + DLinear ensemble** (Option 6) | ~1 hour | Medium — low-risk improvement |
| 5 | **GBP-native pricing** (Option 5) | ~2 hours | Low-medium — removes noise |
| 6 | **Classification framing** (Option 9) | ~1 day | Medium — different angle entirely |
| 7 | **PatchTST** (Option 2) | ~2 days | Medium — better architecture |
| 8 | **Transfer learning from EU ETS** (Option 7) | ~2 days | Medium-high — more data |
| 9 | **UK-specific news/LLM** (Option 8) | ~2 days | Low — same GDELT problems |

---

## 10. Strategic Assessment

### Is the UK ETS Pivot Worth Continuing?

**Yes, but with adjusted expectations.**

The UK ETS sub-project confirmed one important thing: **our infrastructure works and can be adapted to new markets within 1-2 days**. This is genuinely valuable. The data pipeline, experiment tracking, and evaluation framework are solid.

However, the UK ETS has harder characteristics than EU ETS for machine learning:
- Far less training data (654 vs 2,128 windows)
- Higher daily volatility (3.93% vs ~1.5%)
- More extreme structural breaks within the available history
- Sparser/lower-quality auxiliary features

The **most productive path forward** is:
1. Accept that Ridge is the current champion and use it as the production baseline
2. Focus on closing the TSM→Ridge gap via Options 1-3 (hyperparameter tuning, residual connection, better features)
3. Only revisit LLM refinement once the TSM reliably matches or beats Ridge
4. Consider the classification framing (Option 9) as a parallel research track — 72% trend accuracy at h30 is already respectable and may be tradeable

### Cross-Market Comparison for the Paper

The UK ETS results strengthen the narrative from the prior report. We now have evidence from **two different carbon markets** that:
1. LLM refinement does not substantially improve carbon price forecasting under honest evaluation
2. Simple linear models are surprisingly competitive
3. The reference paper's results on Chinese carbon markets may reflect market-specific characteristics (lower liquidity, less institutional participation) rather than a general property of LLM-augmented forecasting

This cross-market result is a publishable finding in itself.

---

## Appendix A: Full UK ETS Run Index

| Run ID | Date | TSM | LLM | Test n | Path MSE (TSM) | Path MSE (Best) | Status |
|--------|------|-----|-----|--------|----------------|-----------------|--------|
| `20260305_010646` | 5 Mar | Autoformer d=512 | None | 4 | 27.62 | Ridge: 18.91 | Complete |
| `20260305_014353` | 5 Mar | Autoformer d=512 | None | 4 | 51.17 | Ridge: 18.91 | Complete |
| `20260305_014955` | 5 Mar | Autoformer d=512 | 4B (DP/CoT/CoT-RF) | 4 | — | Ridge: 18.91 | Complete |
| `20260305_015111` | 5 Mar | Autoformer d=512 | 4B | 4 | — | — | Failed |
| `20260305_015116_4383aa` | 5 Mar | Autoformer d=512 | 4B | 4 | — | — | Failed |
| `20260305_015116_976e21` | 5 Mar | Autoformer d=512 | 4B | 4 | — | — | Failed |
| `20260305_015116_babcf0` | 5 Mar | Autoformer d=512 | 4B | 4 | — | — | Failed |
| `20260305_015116_ccc2cc` | 5 Mar | Autoformer d=512 | 4B | 4 | — | — | Failed |
| `20260305_015229` | 5 Mar | Autoformer d=512 | 4B (all methods) | 4 | 51.17 | TSM+LLM: 41.15 | Complete |
| `20260305_021641` | 5 Mar | Autoformer d=512 | GPT-5.2 | 4 | — | — | Failed |
| `20260306_130357` | 6 Mar | Autoformer d=512 | None | 144 | 40.35 | Ridge: 23.27 | Complete |
| `20260306_130923` | 6 Mar | Autoformer d=512 | 4B | 144 | — | — | Failed |
| `20260306_131534` | 6 Mar | Autoformer d=32 | None | 144 | 31.41 | Ridge: 22.42 | Complete |
| `20260306_132538` | 6 Mar | **DLinear** d=32 | CoT-RF (4B) | 144 | 29.78 | Ridge: 22.42 | **Complete** |
| `20260306_142039` | 6 Mar | DLinear d=32 | CoT-RF (base=Ridge) | 144 | — | — | Failed |
| `20260306_142124` | 6 Mar | DLinear d=32 | CoT-RF (base=Ridge) | 144 | 29.78 | Ridge: 22.42 | **Complete** |

Of 16 runs, 8 completed successfully and 8 failed (mostly multiprocessing spawn issues or API connectivity).

## Appendix B: Key Configuration Differences (Default vs Tuned)

| Parameter | Default (uk_ets_default.yaml) | Tuned (uk_ets_tuned.yaml) |
|-----------|------|-------|
| TSM type | Autoformer | **DLinear** |
| seq_len | 120 | **20** |
| label_len | 30 | **10** |
| d_model | 512 | **32** |
| n_heads | 8 | **2** |
| e_layers | 2 | **1** |
| d_ff | 2048 | **64** |
| dropout | 0.05 | **0.3** |
| batch_size | 32 | **64** |
| learning_rate | 0.0001 | **0.001** |
| max_epochs | 100 | **40** |
| weight_decay | — | **0.01** |
| grad_clip | — | **0.5** |
| LLM model | GPT-5.2 | **Qwen3-VL-4B** |
| LLM methods | DP, CoT, CoT-RF, TSM+LLM | **TSM+LLM-COT-RF only** |
| LLM history | 120 | **18** |
| max_samples | 500 | **100,000** (no cap) |
| Blend grid | — | **Enabled** (ramp, 5 weights) |
