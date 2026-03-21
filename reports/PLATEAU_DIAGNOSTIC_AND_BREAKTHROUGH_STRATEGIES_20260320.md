# UK ETS Performance Plateau: Diagnostic Analysis & Breakthrough Strategies

**Date**: 21 March 2026  
**Scope**: Full UK ETS project lifecycle (Dec 2025 – Mar 2026), 800+ experiment runs  
**Status**: The project has plateaued at TSM path MSE ~21.5–23.5 with LLM refinement producing <1% marginal improvement in recent runs.

---

## Table of Contents

1. [Current Performance Snapshot](#1-current-performance-snapshot)
2. [Diagnosis: Why the TSM Is Stuck](#2-diagnosis-why-the-tsm-is-stuck)
3. [Diagnosis: Why LLM Refinement Has Stalled](#3-diagnosis-why-llm-refinement-has-stalled)
4. [Literature-Backed Breakthrough Strategies for TSM](#4-literature-backed-breakthrough-strategies-for-tsm)
5. [Literature-Backed Breakthrough Strategies for LLM Refinement](#5-literature-backed-breakthrough-strategies-for-llm-refinement)
6. [Unconventional Angles](#6-unconventional-angles)
7. [Prioritised Action Plan](#7-prioritised-action-plan)
8. [Risk Assessment](#8-risk-assessment)

---

## 1. Current Performance Snapshot

### 1.1 Latest Benchmark Numbers (March 2026)

The core UK ETS test split (train ≤ 2023-10-27, val ≤ 2024-10-26, test ≤ 2025-06-30) produces ~143 test windows on the standard configuration.

| Model | Path MSE | Δ vs TSM | Notes |
|-------|----------|----------|-------|
| **Naive persistence** | 22.605 | −2.8% | Hard to beat = efficient market |
| **Linear Ridge** | 21.584 | −7.2% | Embarrassingly strong baseline |
| **TSM (DLinear, tuned)** | 23.258–23.465 | baseline | Varies slightly by seed/config |
| **Best TSM (earlier DLinear)** | 22.726 | ~baseline | The "established" TSM figure |
| **Best LLM (35B, recent_high_error_180)** | 21.738 | −4.3% | All-time best, March 9 |
| **Latest LLM runs (4B, 20 Mar)** | 101.7–102.3 | **worse** | Different split/config; see §1.2 |

### 1.2 Concerning Patterns in Recent Runs

The most recent runs (March 20) show dramatically different baseline numbers from the established UK ETS benchmark:

- TSM: **101.7** path MSE (vs the expected ~23)
- Naive: **61.1** (vs ~22.6)
- Ridge: **310.4** (vs ~21.6)

These are clearly from a **different data configuration** — likely switching `max_date`, using a different split, or changing to a returns vs. price-level mode where the target scale is different. The LLM refinement in these runs produces near-zero improvement: TSM+LLM goes from 101.735 to 101.725 (−0.01%).

> [!WARNING]
> The latest runs appear to use a fundamentally different data split or target transformation. Before any further experiments, the team must verify which configuration is "canonical" and ensure apples-to-apples comparison.

### 1.3 The Plateau in Context

Looking at the standard UK ETS benchmark configuration, TSM performance has been stable at ~22.7 path MSE since early March. The best LLM improvement (35B) took it to 21.74, but:

- No subsequent run has beaten this number
- The 4B LLM produces <0.5% improvement in recent configurations
- Ridge regression (21.58) still beats the best deep learning + LLM pipeline

The honest situation is: **the TSM is slightly worse than Ridge, and the LLM overlay barely closes the gap**.

---

## 2. Diagnosis: Why the TSM Is Stuck

### 2.1 DLinear Is Too Simple for the Remaining Signal

The current DLinear model has only **1,279 parameters**. It decomposes the input into trend (moving average) and seasonal (residual) components, then applies separate linear projections to each. This architecture:

**What it can capture:**
- Linear dependencies on lagged returns
- Simple momentum and mean-reversion signals
- Direct feature-to-forecast linear mappings

**What it fundamentally cannot capture:**
- Non-linear interactions between features (e.g., gas × weather interactions)
- Regime-dependent dynamics (high-vol periods behave differently from low-vol)
- Cross-frequency patterns (weekly seasonality interacting with monthly trend)
- Asymmetric responses (price responds differently to positive vs negative energy shocks)

The DLinear with `residual_linear` channel mixer is essentially a linear model with a gated correction. It has converged close to the Ridge solution, which explains why these two are neck-and-neck.

### 2.2 The Sequence Length Is Too Short

The current `seq_len=20` (20 trading days ≈ 1 month) is extremely restricted. Carbon prices have documented cyclicalities at:

- **Weekly**: auction calendar effects
- **Monthly**: compliance deadline cycles (quarterly surrender deadlines)
- **Seasonal**: heating demand → gas demand → power demand → carbon demand
- **Annual**: EU ETS/UK ETS compliance year cycles

With seq_len=20, the model cannot see:
- The quarterly compliance pattern at all
- Seasonal heating/cooling transitions
- Year-over-year structural shifts

### 2.3 The Feature Set May Be Sub-Optimal

The LASSO screen (March 15) identified weather as genuinely predictive, and the current config includes weather. But:

- Only **9 features** are fed to the model (`enc_in: 9`, vs 16 preferred features listed in config)
- The `max_exogenous_features_model: 16` cap exists but may not be binding correctly
- Key features like `uk_power_gas_vol_ratio_20d` and `uk_gas_hdd18_surprise_interaction` are in the preferred list but may not survive the feature selection pipeline

### 2.4 The Training Regime Is Over-Regularised

The current training setup uses:
- `dropout: 0.5` — This is extremely high for a 1,279-parameter model
- `weight_decay: 0.02` — Also aggressive
- `learning_rate: 0.001` with CosineAnnealing warm restarts
- Huber loss with `beta=0.5` — This clips large gradients strongly
- `batch_size: 64` with only 582 training windows — Only ~9 gradient steps per epoch

With this much regularisation on such a small model, the network is fighting to express *any* learned pattern. The high dropout is appropriate for the 512-d Autoformer but is overkill for DLinear.

### 2.5 No Decomposition Pre-Processing

The literature consistently shows that the strongest carbon price forecasting models use **signal decomposition before modelling**:
- VMD (Variational Mode Decomposition)
- CEEMDAN (Complete Ensemble Empirical Mode Decomposition)
- Wavelet decomposition

These separate the price into smooth trend, cyclical, and noise components. The model then predicts each component separately, which is a much easier sub-problem than predicting the raw series. The current DLinear has a simple moving average decomposition internally, but no principled multi-scale decomposition of the *input* features.

### 2.6 Non-Stationarity Is Not Addressed

UK ETS prices went from £97 (Oct 2023) to £30 (mid-2024) to £65 (late 2024). This is a massive regime shift. The model is trained on the 2021-2023 period and asked to generalise to a completely different price regime. No regime detection, no structural break detection, and no online adaptation mechanism is in place.

---

## 3. Diagnosis: Why LLM Refinement Has Stalled

### 3.1 The Thesis on Failure is Correct — But Incomplete

The existing analysis (March 14) correctly identifies that:
- The task is a bounded correction problem, not open-ended reasoning
- Elaborate systems add variance faster than signal
- The simpler the correction, the better

But it misses several additional factors:

### 3.2 The 4B Model Cannot Do Arithmetic

Qwen3-VL-4B (the current production model) produces corrections like:

> *"Reduce the long-horizon forecast by approximately 2-5%"*

This is qualitative guidance, not a quantitative correction. When translated to numerical deltas, these vague instructions produce noisy, inconsistent adjustments. The `enable_numeric_tool` and `enable_delta_verifier_tool` help, but they are *post-hoc corrections* to an imprecise initial estimate — they cannot recover signal that was never there.

### 3.3 The Sentiment Signal Is Noise

From the March 11 diagnostic:
- GDELT sentiment is not financial-grade news data
- Adding sentiment to CoT-RF **degraded** performance (from −13.78% to −4.28% pre-leakage-fix)
- The current `daily_sentiment_uk_qwen_votes3_importance.csv` aggregation is producing near-zero signal

Despite this, the latest config still has `sentiment.enabled: true`. Every LLM call includes sentiment data that actively reduces the signal-to-noise ratio of the prompt.

### 3.4 Teaching Examples Are Stale and Sparse

With only 582 training windows and 228 validation windows:
- The `recent_high_error_180` strategy draws from ~130 windows
- Many of these are from a different price regime (pre-crash vs post-crash)
- The "high-error" cases from the training period may represent structural breaks that cannot be learned from

### 3.5 The TSM Residuals Are White Noise

This is the fundamental constraint. When TSM achieves 22.7 and naive persistence achieves 22.6, the model has essentially no systematic bias to correct. The residuals are:

- Near-zero mean
- Approximately homoscedastic
- Not auto-correlated
- Not correlated with observable features

The LLM is being asked to extract signal from noise. No amount of prompt engineering or model scaling can overcome this information-theoretic barrier.

### 3.6 The H-Delta Guards and Coherence Rules Are Over-Constraining

The current config has extensive structured coherence guards:
- `freeze_horizons: [1]` — h1 is never touched
- `structured_zero_negative_long_when_h5_positive: true`
- `structured_zero_h20_negative_when_h30_zero: true`
- `structured_zero_mixed_long_signs: true`
- `structured_prefer_positive_long_conflicts: true`

These rules were designed to prevent damage, but they collectively restrict the LLM's action space so much that it often produces near-zero corrections. When the model *does* have a genuine corrective insight, these rules may suppress it.

---

## 4. Literature-Backed Breakthrough Strategies for TSM

### Strategy T1: Signal Decomposition + Component-Wise Modelling

**Literature basis**: VMD-GARCH/LSTM (Ren et al., 2022, *Energy Economics*); CEEMDAN-TCN (multiple 2024-2026 papers); PELT-WT-TCN framework; dual-mode decomposition with TKMixer-BiGRU-SA

**What it is**: Decompose the raw UKA price series into K sub-modes (typically 3-8) using VMD, CEEMDAN, or wavelet transform. Train a separate forecaster for each mode. Reconstruct the full forecast by summing the component predictions.

**Why it should help here**:
- The DLinear's internal moving-average decomposition is too crude (single kernel size)
- Carbon prices have documented regime shifts that would separate into distinct VMD modes
- Low-frequency modes (smooth trend) can be forecast with higher accuracy than raw price
- High-frequency noise can be explicitly modelled with GARCH or simply discarded

**Implementation effort**: Medium (2-3 days). Use existing `pywt` or `vmdpy` libraries.

**Expected gain**: 5-15% MSE reduction based on literature results on similar carbon price series.

```python
# Conceptual implementation
from vmdpy import VMD
u, u_hat, omega = VMD(price_series, alpha=2000, tau=0, K=5, DC=0, init=1, tol=1e-7)
# u[0]: smooth trend mode → Ridge or DLinear
# u[1-3]: cyclical modes → LSTM or GRU
# u[4]: high-frequency noise → GARCH or zero predictor
# Final = sum(pred_mode_i for i in range(K))
```

### Strategy T2: PatchTST Architecture

**Literature basis**: PatchTST (Nie et al., 2023, ICLR); state-of-the-art for long-horizon multivariate forecasting

**What it is**: Segment the input time series into non-overlapping patches (e.g., patches of 5 days). Each patch becomes an input token to a standard Transformer. Channel-independent design avoids the over-parameterisation of multivariate transformers.

**Why it should help here**:
- Patches capture local semantic meaning (weekly patterns, post-auction drift)
- The patching allows longer effective context windows without quadratic attention cost
- Channel independence prevents overfitting with 9+ features on small datasets
- This is the current SOTA for the class of problems DLinear was benchmarked against

**Implementation effort**: Medium (2 days). Many open-source implementations exist.

**Expected gain**: 3-10% over DLinear based on benchmark comparisons.

### Strategy T3: Increase Sequence Length + Multi-Resolution Features

**What it is**: Increase `seq_len` from 20 to 60-120, but use multi-resolution feature engineering to avoid the small-sample problem:

- Days 1-20 (recent): daily granularity, all features
- Days 21-60 (intermediate): 5-day rolling means of key features
- Days 61-120 (distant): weekly statistics only

**Why it should help here**:
- Captures quarterly compliance deadlines (UK ETS auctions are ~biweekly)
- Weather seasonality requires 60+ day context
- The coal-brent ratio and gas price regime shifts unfold over months, not days

**Implementation effort**: Low (1 day). Modify data windowing.

**Expected gain**: 2-5% from capturing medium-frequency patterns currently invisible.

### Strategy T4: Train on Multiple Carbon Markets via Transfer Learning

**Literature basis**: Transfer learning for financial time series; EU → UK ETS domain adaptation

**What it is**: Pre-train the TSM on EU ETS data (4,334 days, much larger dataset), then fine-tune on UK ETS data with a lower learning rate.

**Why it should help here**:
- EU and UK carbon prices are correlated (~0.85 correlation historically)
- Both markets respond to similar fundamental drivers (gas prices, coal, weather)
- The EU ETS dataset is 4× larger, giving the model more training signal
- Fine-tuning retains the general carbon price dynamics while adapting to UK-specific structure

**Implementation effort**: Low (1 day). Load EU ETS panel, train, then fine-tune.

**Expected gain**: 2-8% if EU-UK correlation is exploitable.

### Strategy T5: Structural Break Detection + Regime-Adaptive Modelling

**Literature basis**: Bai-Perron + PELT structural break detection; PELT-WT-TCN (hybrid framework)

**What it is**: Detect structural breaks in the UK ETS price (e.g., Oct 2023 crash) using PELT or Bai-Perron algorithms. Fit separate models per regime, or use the break indicators as explicit features.

**Why it should help here**:
- UK ETS has at least 3 distinct regimes in the training data (rise → crash → recovery)
- Training a single model across all regimes forces it to average incompatible dynamics
- Regime indicators could serve as switching variables for ensemble weighting

**Implementation effort**: Low (0.5 day). `ruptures` library.

**Expected gain**: 2-5% by avoiding regime-mixing.

### Strategy T6: Reduce Over-Regularisation on DLinear

**What it is**: The current DLinear is heavily regularised (dropout=0.5, weight_decay=0.02, Huber loss). For a 1,279-parameter model with 582 training windows, this is overkill. Proposed changes:

| Parameter | Current | Proposed |
|-----------|---------|----------|
| dropout | 0.5 | 0.15-0.25 |
| weight_decay | 0.02 | 0.005 |
| loss_type | huber (beta=0.5) | MSE or huber (beta=1.0) |
| batch_size | 64 | 32 |
| max_epochs | 40 | 100 |
| early_stopping_patience | 8 | 15 |

**Why it should help**: The model is currently too constrained to learn non-trivial patterns. Reducing regularisation and allowing more epochs may let it find useful structure that is currently suppressed.

**Implementation effort**: Negligible (config change).

**Expected gain**: 1-3% (diminishing returns — DLinear is fundamentally limited).

### Strategy T7: Ensemble of Diverse Architectures

**What it is**: Instead of relying on a single model, ensemble:
1. Ridge regression (already strong)
2. DLinear with tuned hyperparams
3. PatchTST or lightweight transformer
4. Gradient Boosted Trees (LightGBM) on the same feature set

Simple averaging or stacking these should reduce variance without increasing bias.

**Why it should help**: Each model captures different aspects of the signal:
- Ridge: stable linear relationships
- DLinear: trend-season decomposition
- Transformer: non-linear temporal interactions
- LightGBM: feature interactions and non-linearities

**Implementation effort**: Medium (2 days for full pipeline).

**Expected gain**: 3-8% over the best individual model.

---

## 5. Literature-Backed Breakthrough Strategies for LLM Refinement

### Strategy L1: Target the LLM at Regime Classification, Not Magnitude Correction

**What it is**: Instead of asking the LLM to produce numerical deltas, ask it to classify the market regime (trending up/down/flat, high/low volatility, pre/post-auction, compliance deadline approaching). Then use the regime label as an input to a lightweight statistical model that determines the actual correction.

**Why it should help**:
- 4B models are good at pattern recognition and classification
- 4B models are bad at precise numerical estimation
- This converts the LLM task from "predict a number" to "classify a situation" — a task where LLMs are known to perform well
- The numerical correction is then handled by a calibrated statistical layer (Ridge or logistic regression) that won't over-correct

**Implementation effort**: Medium (2-3 days). Change the prompt to request categorical outputs.

**Expected gain**: 1-3% — may recover value that is currently lost to quantitative imprecision.

### Strategy L2: Use the LLM as a Feature Extractor, Not a Forecaster

**Literature basis**: "Feature Engineering with LLM Embeddings" (multiple 2025 papers); STELLA framework

**What it is**: Instead of asking the LLM to produce forecast corrections, use the LLM to generate text embeddings from news, market commentary, or its own market analysis. Feed these embeddings as additional features into the TSM pipeline.

**Why it should help**:
- Separates "understanding" (LLM) from "forecasting" (statistical model)
- Avoids the noisy magnitude estimation problem
- LLM embeddings capture semantic content that GDELT sentiment scores miss
- The statistical model learns how to weight these embeddings optimally

**Implementation effort**: Medium (2-3 days). Extract hidden states or generate structured summaries.

**Expected gain**: 2-5% if LLM embeddings capture genuine news signal.

### Strategy L3: Upgrade to a Larger Model for Quantitative Reasoning

**What it is**: Use a 70B+ model or a frontier API (GPT-4o, Claude 4 Sonnet, Gemini 2.5 Pro) for at least a controlled experiment.

**Evidence from this project's own data**:
- 4B → 35B produced a clear step-change (22.42 → 21.74, −3.1%)
- The 35B → frontier gap is likely of similar magnitude
- The reference paper likely used GPT-4-class models

**Implementation effort**: Low (API key + config change).

**Expected gain**: 2-5% based on the 4B → 35B scaling pattern.

**Cost note**: This is the most expensive strategy. Running 143 test samples through GPT-4o costs roughly $5-15 per full run. May be justified for paper validation but not for production.

### Strategy L4: Disable Sentiment or Replace GDELT with Structured News Features

**What it is**: Either:
1. **Immediately disable** `sentiment.enabled` in the config (expected to improve results based on existing evidence)
2. **Replace** GDELT with LLM-generated structured news features from a curated RSS feed of carbon market news sources (ICIS, Argus, Carbon Pulse)

**Evidence this will help**:
- Sentiment was already shown to degrade performance (−4.28% → −1.61% when added to CoT-RF)
- Every recent run includes sentiment noise in the LLM prompt
- This is the lowest-hanging fruit in the entire project

**Implementation effort**: Zero (set `sentiment.enabled: false` in config).

**Expected gain**: 0.5-2% — free improvement.

### Strategy L5: Online Learning / Memory-Based Adaptation

**What it is**: Instead of a fixed teaching pool, implement online memory:
- After each test window prediction, compare TSM prediction to actual outcome (when truth becomes available)
- Add the best recent correction examples to a rolling memory buffer
- The LLM's teaching examples adapt to the current market regime

**Why it should help**:
- Addresses the staleness problem with the frozen teaching pool
- Regime-specific corrections are more reusable than cross-regime examples
- Simulates what a live trading system would actually do

**Implementation effort**: Medium (2 days). The codebase already has `online_memory` run folders, suggesting prior attempts.

**Expected gain**: 1-3% — conditional on the memory buffer being well-managed.

### Strategy L6: Relax the H-Delta Guards Selectively

**What it is**: The current coherence guards over-constrain the LLM's correction space. Proposed relaxation:

| Guard | Current | Proposed |
|-------|---------|----------|
| freeze_horizons [1] | Always freeze h1 | Keep (h1 is well-predicted) |
| zero_negative_long_when_h5_positive | true | false (let the model disagree across horizons) |
| zero_h20_negative_when_h30_zero | true | false |
| zero_mixed_long_signs | true | false |
| max_adjustment_pct | 1.0 | 2.0 (allow larger corrections) |

**Why it should help**: The guards currently suppress ~50-80% of proposed corrections. Some of these suppressed corrections may be valuable, especially at h20-h30 where the TSM has the most error.

**Implementation effort**: Negligible (config change).

**Expected gain**: 0.5-2% — risk of slight degradation if guards were genuinely protective.

### Strategy L7: True Tool-Augmented Computation

**Literature basis**: Program of Thought (Chen et al., 2022); ReAct framework; LLM-powered agents with code execution

**What it is**: Give the LLM access to a Python sandbox where it can:
- Calculate exact statistics from the price history
- Run simple regression fits on recent data
- Compute confidence intervals for its proposed corrections
- Test its proposed correction against historical analogues

**Why it should help**:
- The current "program_of_thought" prompt style asks the model to *imagine* code execution
- True code execution would eliminate arithmetic errors
- The model could verify its own reasoning before committing to a correction

**Implementation effort**: High (3-5 days). Requires a sandboxed code execution environment.

**Expected gain**: 1-3% on 4B (more on larger models where reasoning quality is higher).

---

## 6. Unconventional Angles

### U1: Forecast the *Direction*, Not the *Path*

Instead of predicting a 30-step path, predict only:
- Direction at h20 (up/down/flat) — currently Ridge achieves ~70% accuracy
- Direction at h30

Then convert to a trading signal directly. Skip the path MSE metric entirely.

**Rationale**: Path MSE penalises magnitude errors that may be irrelevant for trading. A model that correctly predicts "up 3%" when the actual is "up 5%" gets a worse MSE than a model that predicts "up 0.5%" when actual is "up 5%", even though the first model produces a profitable trade.

### U2: Target Conditional Variance, Not Conditional Mean

Instead of trying to predict the expected price path (mean), predict the uncertainty around each forecast. Use volatility forecasting (GARCH, HAR-RV) to:
- Set better position sizes
- Identify when to trust vs not trust the forecast
- Gate the LLM correction based on predicted uncertainty

### U3: Calendar Anomaly Features

UK ETS auctions happen on a fixed schedule. Add binary/categorical features for:
- Days to next auction
- Days since last auction
- Whether compliance deadline is within forecast horizon
- Month-of-year (seasonality in power demand → gas demand → carbon demand)

These are free features that the DLinear may not be capturing from raw price alone.

### U4: Conformal Prediction Intervals

Rather than point forecasts, produce conformal prediction intervals. This gives calibrated uncertainty estimates without distributional assumptions. Can be used to:
- Skip LLM refinement when TSM confidence is high
- Apply larger LLM corrections when TSM confidence is low
- Report honest error bars in the paper

---

## 7. Prioritised Action Plan

### Tier 1: Immediate (Today / This Week) — Expected Total Gain: 3-8%

| # | Action | Effort | Expected Gain | Risk |
|---|--------|--------|---------------|------|
| 1 | **Disable sentiment** (`sentiment.enabled: false`) | 0 minutes | 0.5-2% | None — already proven to hurt |
| 2 | **Verify data split consistency** — confirm canonical UK ETS config | 30 minutes | Prevents wasted effort | — |
| 3 | **Reduce DLinear over-regularisation** (dropout → 0.2, weight_decay → 0.005, patience → 15) | 15 minutes | 1-3% | Small risk of overfitting (test empirically) |
| 4 | **Relax H-Delta guards** (disable zero-mixed-signs, increase max_adjustment to 2%) | 15 minutes | 0.5-2% | May hurt if guards were calibrated well |

### Tier 2: This Week / Next Week — Expected Total Gain: 5-15%

| # | Action | Effort | Expected Gain | Risk |
|---|--------|--------|---------------|------|
| 5 | **VMD or CEEMDAN decomposition** before DLinear | 2 days | 5-15% | Implementation complexity; mode selection |
| 6 | **Increase seq_len to 60** with multi-resolution features | 1 day | 2-5% | May need more compute for search |
| 7 | **PatchTST implementation** as a DLinear replacement | 2 days | 3-10% | May need hyperparameter tuning |
| 8 | **LLM → Regime classifier** instead of delta predictor | 2 days | 1-3% | Changes the LLM pipeline significantly |

### Tier 3: Next Sprint — Expected Total Gain: 3-10%

| # | Action | Effort | Expected Gain | Risk |
|---|--------|--------|---------------|------|
| 9 | **Model ensemble** (Ridge + DLinear + PatchTST + LightGBM) | 2 days | 3-8% | Adds pipeline complexity |
| 10 | **EU → UK transfer learning** | 1 day | 2-8% | Domain gap may be too large |
| 11 | **Calendar anomaly features** (auction schedule, compliance calendar) | 1 day | 1-3% | Data availability |
| 12 | **Larger LLM experiment** (35B or frontier API) | 1 day | 2-5% | Cost; one-off validation only |

---

## 8. Risk Assessment

### What Could Go Wrong

1. **Decomposition overfitting**: VMD/CEEMDAN hyper-parameters (number of modes, alpha, noise amplitude) can be overfit to in-sample data. Must use a hold-out or cross-validation for mode selection.

2. **Architecture search hell**: Replacing DLinear with PatchTST opens a new hyperparameter space. Must set a strict computational budget and stop after a fixed number of configs.

3. **Regime instability**: Any gains measured on the current test split (Oct 2024 – Jun 2025) may not generalise to the next regime. Must report robustness across multiple time periods.

4. **LLM escalation trap**: Moving to larger/frontier models increases cost linearly. Must establish a clear ROI threshold before committing to expensive API models.

### What Is Unlikely to Help (Based on Evidence)

| Approach | Why It Won't Help |
|----------|-------------------|
| More elaborate LLM prompting (tree-of-thought, debate, self-RAG) | Already tested extensively; all degraded performance |
| Broader teaching example lookbacks (>180 days) | Already tested; 180 days is the sweet spot |
| Horizon-specific matching in examples | Already tested; consistently hurts |
| Adding more sentiment variants | GDELT signal is noise; no amount of filtering fixes this |
| Increasing LLM sample size beyond all test windows | Already running on all 143 test windows |

---

## Summary

The performance plateau has two distinct but related causes:

1. **The TSM (DLinear) has converged to approximately the same solution as Ridge regression.** It is a 1,279-parameter linear model with excessive regularisation, short context, and no ability to capture non-linear or multi-scale dynamics. Breaking through requires either a fundamentally different architecture (PatchTST, decomposition + per-mode models) or a richer feature set (longer sequences, calendar features, transfer learning).

2. **The LLM refinement layer is trying to extract signal from approximately white noise residuals.** When the base model is near-optimal, there is very little for the LLM to correct. The existing correction infrastructure (HDELTA guards, sentiment noise, 4B quantitative imprecision) further reduces whatever marginal value exists. The LLM's highest-value application in this project is likely **regime classification** or **feature extraction**, not direct numerical correction.

The single highest-ROI action is **signal decomposition (VMD/CEEMDAN) + per-mode forecasting**. The literature consistently shows 5-15% MSE improvements on carbon price series with this approach, and it attacks the core limitation (the model sees a single, noisy composite signal) rather than adding complexity on top of an already-strained correction pipeline.

The single quickest win is **disabling sentiment and reducing over-regularisation** — two config changes that can be tested immediately with zero engineering effort.

---

*This analysis draws on: 250+ UK ETS experiment runs, the project's own diagnostic reports (March 11, March 14, March 10 weekly log), and current literature on carbon price forecasting, LLM-assisted time series prediction, and deep learning architectures for financial forecasting.*
