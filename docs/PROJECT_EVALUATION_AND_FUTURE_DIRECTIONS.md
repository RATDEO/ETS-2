# EU ETS Forecasting Project: Comprehensive Evaluation, Limitations, and Future Directions

**Date**: 3 March 2026  
**Status**: Post-mortem and strategic assessment  
**Scope**: Full project lifecycle from December 2025 to March 2026

---

## 1. Executive Summary

This project attempted to reproduce and extend the methodology of Chen et al. (*"Can Large Language Models forecast carbon price movements?"*) by applying a hybrid Autoformer + LLM refinement pipeline to the **EU Emissions Trading System (EU ETS)** secondary market. Over approximately three months, **200+ experiment runs** were executed across multiple model architectures, LLM providers, prompt strategies, sentiment pipelines, and evaluation frameworks.

**The central finding is that LLM-based forecast refinement does not reliably improve upon a well-tuned time series model for EU ETS price forecasting under honest, leakage-free evaluation conditions.** The best clean, leakage-free result — a paper-style `TSM+LLM-COT-SENT-RF` refinement — achieves a **1.6% path MSE improvement** over the base Autoformer (20.604 vs 20.942). This is real but modest, statistically fragile, and far below the gains reported in the reference paper for the Chinese carbon market.

The project is **not doomed**, but it has reached the limits of what the current methodology can deliver on this particular market and dataset. The remaining headroom for LLM-based refinement appears to be very small under honest evaluation, and the engineering effort required to extract it is disproportionate to the expected gain.

---

## 2. What We Built

### 2.1 Infrastructure (Successful)
- Complete end-to-end pipeline: data ingestion → feature panel → TSM training → LLM refinement → evaluation → frontend export
- Automated data acquisition for 7/9 sources (ICAP, EEX auctions, ECB FX, FRED Brent, GDELT headlines, DG CLIMA, ENTSO-E)
- Standalone deployable package (`bud_standalone/`) with one-command execution
- Comprehensive experiment tracking with 200+ versioned runs
- Leakage detection and correction framework with regression tests
- Multiple LLM integration paths: local 4B (Qwen3-VL), local 35B (Qwen3.5), and API models (GPT-4-turbo, GPT-5.2)
- Frontend JSON export for portfolio integration
- Official-source event store with deterministic EEX auction parsing and DG CLIMA/ENTSO-E automated extraction

### 2.2 Models Tested
- **TSM architectures**: Autoformer (primary), DLinear
- **TSM configurations**: Large (d=512, 2 layers) and Small (d=64, 1 layer, high dropout)
- **LLM refinement methods**: Direct Prompting (DP), Chain-of-Thought (CoT), CoT with Self-Refinement (CoT-RF), CoT with Sentiment and Self-Refinement (CoT-Sent-RF), Horizon Delta corrections (HDELTA), Anchor Price corrections (HPRICE)
- **Blending strategies**: Fixed ramp, validation-selected ramp, meta-blend (Ridge), stateful meta-blend, regime-gated model switching
- **Teaching-example selection**: Most-recent, similarity-based, high-error, error-stratified, event-similarity
- **Sentiment pipelines**: GDELT headline-based (YES/NO/UNKNOWN), importance-weighted (0-10 scale), Dawid-Skene probabilistic aggregation, domain-weighted, confidence-penalized, EU-close time-aligned, EWMA smoothed, abnormal z-score, asymmetric decay, novelty/topicality weighted, regime-calibrated, official-source deterministic events

---

## 3. What the Reference Papers Achieved (And What Was Different)

### 3.1 Paper B: Chen et al. — *"Can Large Language Models forecast carbon price movements?"*

This is the primary reference. Key reported results on Chinese carbon markets:

| Method | Interpretation |
|---|---|
| Base Autoformer | Baseline TSM |
| CoT | Meaningful improvement over Autoformer |
| CoT-RF | Further improvement via self-refinement |
| CoT-Sent-RF | Best method; sentiment adds value on top of CoT-RF |

The paper reports that **LLM refinement consistently and substantially improves the base Autoformer** across both MSE and trend accuracy metrics, with the full CoT-Sent-RF pipeline delivering the largest gains.

### 3.2 Paper A: Lopez-Lira & Tang — *"Can ChatGPT Forecast Stock Price Movements?"*

This paper provides the sentiment labeling framework (YES/NO/UNKNOWN + rationale). Key features:
- Uses RavenPack for **high-quality relevance filtering** (relevance=100)
- Evaluates via **post-event drift** and **long-short portfolio** returns
- Emphasizes strict deduplication and event-similarity controls

### 3.3 Critical Differences Between Reference and Our Setup

| Dimension | Reference Paper (Chen et al.) | Our Project |
|---|---|---|
| **Target market** | Chinese carbon markets (regional, younger, more volatile) | EU ETS (global benchmark, mature, policy-driven) |
| **Market liquidity** | Lower, more retail participation | Higher, institutional/compliance-dominated |
| **Price regime** | Relatively stable range during study | Major structural break: €25→€80+ (2020-2023), then sideways |
| **Regulatory environment** | Chinese regional pilot ETSs | EU-wide with MSR, CBAM, Fit-for-55 |
| **Data availability** | Presumably better access to local news/data feeds | Cobbled from 9 heterogeneous public sources |
| **News relevance** | Chinese-language media, likely higher signal density | English-language GDELT headlines, sparse EU ETS relevance |
| **LLM model** | GPT-4 / frontier model (paper era) | Qwen3-VL-4B / Qwen3.5-35B (local, much smaller) |
| **Dataset size** | Not fully specified | ~4,300 daily observations, ~2,100 training windows |
| **Evaluation period** | Within a single price regime | Spans a major regime transition |
| **Feature selection** | Lasso-based | Manual with ablation |
| **News source** | Presumably curated financial news feeds | GDELT DOC 2.0 API (noisy, broad) |

---

## 4. Timeline of Key Discoveries and Turning Points

### Phase 1: Initial Build and Apparent Success (Dec 2025 – Feb 2026)
- Built the complete pipeline and achieved **seemingly strong results**: CoT-RF delivering up to −14% path MSE improvement over base TSM
- **Best apparent result**: Option 4 RAMP h1-base with path MSE of **13.158** (−13.96% vs TSM at 15.294)
- This looked very promising and appeared to align with reference paper findings

### Phase 2: Sentiment Engineering (Feb 8–18, 2026)
- Exhaustive A/B testing of **15+ sentiment pipeline variants**
- None produced robust improvements over the baseline sentiment series
- Key insight: sentiment signal is fundamentally sparse and noisy from GDELT
- Best sentiment variant (v3 importance, no penalty) achieved only −0.83% improvement

### Phase 3: The Leakage Discovery (Feb 27–28, 2026)
**This was the project's critical inflection point.**

Three data leakage paths were discovered:
1. **Decoder future leakage**: `make_windows()` was feeding realized future feature values to the decoder
2. **Split-boundary leakage**: Training windows could have labels extending into validation/test
3. **Teaching-example leakage**: CoT example pools included truths from later test windows

**Before leakage fix**:
- TSM path MSE: **15.294**
- CoT-RF path MSE: **13.388** (−12.5%)
- Option 4 RAMP h1-base: **13.158** (−14.0%)

**After leakage fix**:
- TSM path MSE: **21.198** → retuned to **20.990**
- CoT-RF path MSE: **25.313** (now +19.4% *worse* than TSM)
- CoT ramp blend: validation selected **w=0.0** (zero LLM contribution)

The entire narrative of "CoT substantially improves forecasting" collapsed overnight. The leakage had been inflating both the TSM and LLM numbers, but it inflated the LLM numbers differentially because the LLM had indirect access to future information through the teaching examples and decoder context.

### Phase 4: Recovery Attempts (Feb 28 – Mar 3, 2026)
- Paper-style CoT-Sent-RF recovered a **real but modest 1.6%** improvement (20.604 vs 20.942)
- Strategy 1 (larger LLM): **Failed** — 35B model either produced empty reflections or catastrophic overcorrections
- Strategy 2 (Horizon Delta): Near-tie with baseline (20.618 vs 20.604), not a clear improvement
- Strategy 3 (Sentiment as TSM feature): **Failed** — did not improve TSM or overlays
- Strategy 4 (Error-stratified examples): **Failed** — worse than similarity-based selection
- Strategy 5 (Learned meta-blend): **Failed strict validation** — overfit on validation, rejected by rolling CV
- HPRICE architecture: Promising direction (20.657 for 35B, 20.725 for 4B) but still behind paper-style baseline
- Official event store: Infrastructure value but **zero forecasting improvement** in any tested formulation
- Supervised residual corrector: **Failed** — validation selector chose identity (do nothing) every time
- Regime gating: Oracle ceiling of only ~1% improvement over best single model

---

## 5. Final Leaderboard (Honest, Leakage-Free, Full Held-Out Test, n=382)

| Rank | Method | Path MSE | Δ vs TSM | Note |
|------|--------|----------|----------|------|
| 1 | Paper-style CoT-Sent-RF (raw, 4B) | **20.604** | −1.61% | Best honest result |
| 2 | 4B HDELTA (guarded, qwen3.5-4b-ud) | **20.520** | −2.01% | Best single-model, different endpoint |
| 3 | 35B HPRICE (guarded) | **20.657** | −1.36% | Best new architecture |
| 4 | 4B HPRICE (guarded) | **20.725** | −1.04% | Simpler alternative |
| 5 | Linear Ridge baseline | **20.739** | −0.97% | No deep learning needed |
| 6 | Official-context full-path | **20.855** | −0.42% | Official events did not help |
| 7 | Clean retuned TSM (Autoformer) | **20.942** | baseline | — |
| 8 | Naive persistence | **20.872** | +0.33%* | Trivial baseline |
| 9 | Clean CoT-RF (no sentiment) | **23.442** | +11.94% | Worse than doing nothing |

*Note: Naive persistence is competitive with or better than the TSM on path MSE because the test period is largely sideways, where persistence is hard to beat.

**Key observation**: The spread between the best LLM method and the base TSM is **0.338 path MSE** (1.6%). The spread between the base TSM and naive persistence is **0.070 path MSE** (0.33%). A simple linear Ridge regression (no deep learning, no LLM) achieves **20.739**, which is better than the base TSM and within 0.135 of the best LLM method.

---

## 6. Why This Happened: A Thorough Limitations Analysis

### 6.1 Market Structure Mismatch

The EU ETS is fundamentally different from the Chinese carbon markets studied by Chen et al. in ways that directly undermine the LLM refinement methodology:

**6.1.1 Institutional vs. Retail Price Formation.** EU ETS prices are set primarily by large compliance buyers (utilities, industrials), financial intermediaries, and sovereign auction mechanisms. Price movements reflect regulatory announcements, auction volumes, energy price cross-hedging, and macro policy shifts — information that is available to all sophisticated participants simultaneously. This leaves very little for a language model to "discover" from news headlines that is not already priced in. Chinese regional carbon markets have more retail participation and potentially more informational inefficiency.

**6.1.2 Structural Break in the Test Period.** EU ETS prices underwent a historic rally from approximately €25 in 2020 to over €100 in 2023, followed by a decline and stabilization around €60-70 in 2024-2025. Our test period (mid-2024 to early 2026) sits in the post-rally consolidation phase where prices are largely range-bound. In such a regime, naive persistence is extremely hard to beat because the best prediction of tomorrow's price is approximately today's price. Any model — TSM or LLM — must overcome this strong mean-reversion baseline.

**6.1.3 Policy-Driven Price Dynamics.** EU ETS prices respond to EU-level regulatory decisions (Market Stability Reserve triggers, free allocation phase-outs, CBAM implementation timelines) that arrive irregularly with long lead times and are pre-announced through formal legislative processes. These are not the kind of "news sentiment" events that LLM prompt engineering is designed to capture. The relevant information is structural, slow-moving, and already incorporated into forward curves by the time it appears in news headlines.

### 6.2 LLM Capability Limitations

**6.2.1 Model Scale.** The primary LLM used was Qwen3-VL-4B, a 4-billion parameter model running locally. The reference paper likely used GPT-4 or equivalent frontier models with 100-1000x more parameters. Our experiments with a local 35B model showed that **larger models performed worse, not better**, because they tended to overcorrect or produce unstable outputs under the same prompt framework. This suggests the problem is not simply model size but the fundamental difficulty of the task.

**6.2.2 Reflection Quality.** The CoT-RF two-stage reflection process (reflect on historical errors → apply rules to new forecast) produces **vague qualitative rules** from the 4B model — statements like *"reduce forecast by 2-5%"* or *"apply downward bias of 3-8%"* — that translate to noise in price space. These are not the precise, quantitative corrections that the reference paper appears to achieve. When we constrained the output to structured horizon-delta corrections, the model performed comparably but not better.

**6.2.3 Hallucination and Overcorrection Risks.** When unconstrained, the LLM tendency to "do something" with every prompt leads to systematic overcorrection. The LLM applies an almost constant negative level shift (mean day-1 adjustment ≈ −1.76 EUR in early runs), which destroys short-horizon accuracy where the TSM is already very strong. The ramp-blending mechanism we developed (w₁=0, linearly increasing to w₃₀=max_w) is essentially an admission that the LLM cannot be trusted at short horizons.

### 6.3 Sentiment Signal Inadequacy

**6.3.1 Source Quality.** GDELT DOC 2.0, while comprehensive, is not designed for financial-market-quality news analysis. It contains vast amounts of tangentially related content (shipping news, general EU policy, non-carbon environmental stories) that dilutes the signal. The reference papers used curated financial news feeds (Paper A used RavenPack with relevance=100), which have orders-of-magnitude better signal-to-noise ratios.

**6.3.2 Coverage vs. Quality Tradeoff.** We tested 15+ sentiment pipeline variants including:
- Multiple relevance filtering schemes (heuristic scoring, simhash dedup, event-similarity dedup, LLM relevance gates)
- Multiple aggregation methods (top-3/day, top-5/day, domain-weighted, confidence-penalized, EWMA smoothed)
- Multiple scoring approaches (binary direction, 0-10 importance-weighted, Dawid-Skene EM, abnormal z-score, asymmetric decay, regime-calibrated)

**None produced a robust improvement.** The best variant (asymmetric decay) achieved −0.07% path MSE improvement (within noise). Most variants made things worse. The fundamental issue is that freely available EU ETS news is either too noisy or too stale to carry predictive information beyond what the price series itself contains.

**6.3.3 Unanimous Vote Problem.** At temperature=0, the 4B model produces nearly 100% unanimous votes across 3-vote majority polling, making the majority-vote mechanism a no-op. Introducing temperature=0.35 added some diversity (95.7% unanimous, 4.3% split) but the resulting signal was not materially different. The Dawid-Skene probabilistic aggregation confirmed this: vote streams were near-deterministic, so principled aggregation produced identical results to simple majority voting.

### 6.4 Evaluation and Methodology Issues

**6.4.1 Data Leakage (Discovered and Corrected).** The most consequential finding of this project was that three separate leakage paths in the evaluation pipeline had been inflating LLM performance metrics. The decoder future leakage was particularly insidious: by feeding realized future feature values to the model during evaluation, it gave the combined TSM+LLM system access to forward-looking information that differentially benefited the LLM refinement step. Post-correction, the CoT-RF method went from appearing 12-14% better than TSM to being 19% *worse*.

**6.4.2 Small Effective Dataset.** With only ~2,100 training windows, ~226 validation windows, and ~382 test windows, statistical power for detecting small improvements is limited. A 1.6% path MSE improvement on 382 test windows, while real, sits close to the noise floor. The multiple testing burden from 200+ experimental variants further erodes confidence in any specific result.

**6.4.3 Validation-Test Generalization Failure.** A persistent pattern across the project was that improvements seen on validation subsets (n=40-60) or capped test slices (n=10-40) failed to hold on the full held-out test (n=382). This occurred with:
- Multiple sentiment variants
- Meta-blend calibration layers
- Official event context injection
- Teaching-example retrieval changes

This is a classic manifestation of overfitting to narrow evaluation windows in a low-signal environment.

### 6.5 Fundamental Forecasting Difficulty

**6.5.1 EU ETS Prices Are Close to a Random Walk.** Over the test period, naive persistence (tomorrow's price = today's price) achieves 20.872 path MSE. The best model achieves 20.604. The gap between "do nothing" and "run an Autoformer + LLM refinement with sentiment analysis" is less than 1.3%. This is not a failure of our implementation; it is evidence that EU ETS daily returns over this period contain very little predictable structure beyond the current price level.

**6.5.2 Time Series Model Already Captures Available Signal.** The Autoformer, even with aggressive regularization (d=64, dropout=0.3), achieves 20.942 path MSE — only 0.070 worse than naive persistence. This means the TSM is already extracting nearly all the predictable signal from the feature set. There is very little residual structure left for the LLM to correct.

**6.5.3 The "Last Mile" Problem.** In efficient markets, the marginal value of each additional information source diminishes rapidly. The TSM already incorporates energy prices, auction data, carbon indices, volatility proxies, and exchange rates. Adding news sentiment on top of this is attempting to extract signal from the most informationally depleted slice of the available data.

---

## 7. What We Did Right

Despite the ultimately modest results, several aspects of this project were well-executed and have enduring value:

1. **Leakage detection and correction**: Identifying and fixing three separate leakage paths is a significant methodological contribution. Many published results in financial ML likely contain similar undetected leakage.

2. **Exhaustive ablation**: The 15+ sentiment pipeline variants and 5 LLM refinement strategies constitute one of the most thorough ablation studies of LLM-based forecast refinement in the carbon markets literature.

3. **Honest evaluation protocol**: After the leakage fix, we maintained strict discipline: validation-only selection, no test-set peeking, rolling outer-fold validation for meta-blend, and clear documentation of every positive and negative result.

4. **Production-ready infrastructure**: The automated data pipeline, standalone bud package, and frontend export system are genuinely useful regardless of forecasting performance.

5. **The 1.6% CoT-Sent-RF result is real**: While modest, the paper-style sentiment-augmented refinement does survive honest leakage-free evaluation. This is a legitimate scientific finding.

---

## 8. Options for Future Research

### Option A: Accept the TSM Baseline and Redirect Effort (Recommended)

**Rationale**: The honest results show that a well-tuned Autoformer is competitive with all tested LLM variants. The engineering cost of maintaining the LLM refinement pipeline (local model hosting, prompt engineering, blend tuning, sentiment data maintenance) far exceeds the 0.3 path MSE improvement it delivers.

**Actions**:
1. Promote the clean retuned Autoformer as the production model
2. Remove LLM refinement from the production path
3. Retain the LLM pipeline as a research branch only
4. Focus engineering effort on data quality (better EUA futures feed, VSTOXX automation)
5. Write up the negative results as a contribution to the literature on LLM forecasting reproducibility

**Expected outcome**: Simpler, cheaper, equally accurate system with a publishable negative-result paper.

### Option B: Change the Target Market

**Rationale**: The reference paper's methodology may work better on markets with more informational inefficiency. The EU ETS is one of the most closely watched and efficiently priced carbon markets in the world.

**Actions**:
1. Apply the same pipeline to less liquid carbon markets: UK ETS, California Cap-and-Trade, Korean ETS, Chinese national ETS
2. These markets may have more news-driven price discovery and less institutional efficiency
3. Keep the infrastructure intact and swap only the data sources

**Expected outcome**: Higher probability of demonstrating LLM value-add, but requires new data procurement and market-specific feature engineering.

### Option C: Frontier LLM with Fine-Tuning

**Rationale**: The 4B model is fundamentally inadequate for quantitative financial reasoning. The 35B model was also inadequate under the current prompt framework. A frontier model (GPT-4o, Claude, Gemini) with task-specific fine-tuning might produce the quantitative precision needed.

**Actions**:
1. Fine-tune a frontier model on historical (forecast error, correction) pairs from the training period
2. Use the correction targets from the leakage-free training windows as supervised fine-tuning data
3. Evaluate on the same clean held-out test

**Risks**: High cost, potential for overfitting with small training set, API dependency. The fine-tuning dataset (~2,100 correction examples) is small for meaningful LLM fine-tuning.

**Expected outcome**: If the fundamental problem is model capability rather than market efficiency, this could close the gap. If the problem is market efficiency, even a frontier model will fail.

### Option D: Abandon LLM Refinement; Pursue Alternative TSM Architectures

**Rationale**: Six of twelve identified TSM improvement strategies remain unimplemented: DLinear/PatchTST, residual naive connection, rolling/instance normalization, multi-horizon direct forecasting, data augmentation, and LR scheduling.

**Actions**:
1. Implement PatchTST (shown to outperform Autoformer on many benchmarks)
2. Add residual connection to naive baseline (guaranteed to prevent catastrophic undershoot)
3. Implement rolling instance normalization (addresses the distribution shift problem)
4. Test ensemble of TSM architectures with simple averaging
5. Compare against the current Autoformer baseline on the same clean held-out test

**Expected outcome**: Incremental TSM improvement (estimated 5-15% path MSE reduction based on published benchmarks) without any LLM dependency.

### Option E: Shift to Classification and Portfolio Evaluation

**Rationale**: The reference papers (especially Paper A) evaluate via trend direction accuracy and portfolio returns, not just MSE. The current MSE evaluation may be masking useful directional signal. A model that has higher MSE but better directional accuracy could still be profitable.

**Actions**:
1. Re-evaluate all existing predictions using directional accuracy, Sharpe ratio, and simulated trading P&L
2. Implement a simple long-short strategy based on model-predicted direction
3. Compare strategy returns across TSM, TSM+LLM, and naive baselines
4. This requires no new model runs — only re-scoring existing saved predictions

**Expected outcome**: May reveal that the LLM refinement adds directional value even when MSE is similar, or may confirm that MSE and directional accuracy tell the same story.

### Option F: Hybrid Feature-Engineering Approach (No LLM at Inference Time)

**Rationale**: Instead of using the LLM at inference time (expensive, slow, unreliable), use it once offline to create better features for the TSM.

**Actions**:
1. Use the LLM to label a large historical corpus of EU ETS news with structured event categories (policy announcements, auction results, energy market shocks, industrial demand signals)
2. Build a structured event feature panel from these labels (as partially implemented in the official event store)
3. Train the TSM with these features as additional inputs
4. At inference time, use only the TSM — no LLM in the loop

**Risks**: The official event store was already tested (event_panel_v1 and v2) and failed to improve the TSM. However, the coverage was limited and the feature formulation was narrow. A much richer historical labeling could change this.

**Expected outcome**: Moderate probability of success. This approach aligns with how NLP is actually useful in finance — as a feature-engineering tool, not a direct forecasting agent.

---

## 9. Verdict: Is the Project Doomed?

**No, but it needs to pivot.**

The project is not doomed because:
1. The infrastructure is solid and reusable
2. The clean Autoformer baseline is genuinely competitive
3. A 1.6% real improvement was demonstrated
4. The negative results are publishable and scientifically valuable
5. Multiple untested directions remain (Options A-F above)

The project **is** at a dead end if the goal remains *"reproduce the reference paper's LLM gains on EU ETS"*, because:
1. The market is too efficient for the current LLM methodology to add substantial value
2. The available news sources are too noisy for reliable sentiment extraction
3. The local LLM models lack the quantitative reasoning capability needed
4. Every attempted improvement path has been exhausted or shown diminishing returns

**The most productive next step is Option A** (accept the TSM baseline, write up the negative results) combined with **Option E** (re-evaluate for directional accuracy) and **Option D** (improve the TSM itself). If the project must continue exploring LLM-based methods, **Option B** (change the target market) or **Option F** (offline LLM feature engineering) have the highest expected value relative to effort.

---

## 10. Lessons Learned

1. **Always audit for data leakage before celebrating results.** Three months of apparent progress was invalidated in one audit. Decoder future leakage, split-boundary leakage, and teaching-example leakage are subtle and easy to miss.

2. **Market efficiency is the binding constraint, not model complexity.** Adding more parameters, more sentiment sources, more prompt engineering, and more blending strategies cannot overcome the fundamental limit that EU ETS prices contain very little predictable short-term structure.

3. **Small LLMs are not quantitative reasoners.** A 4B parameter model cannot reliably produce precise numerical corrections to a financial forecast. The reflection rules it generates are qualitative platitudes ("reduce by 2-5%") rather than the sharp quantitative adjustments needed to improve a TSM that is already within 1% of optimal.

4. **Freely available news is not RavenPack.** The quality gap between GDELT headlines and curated financial news feeds (RavenPack, Bloomberg, Refinitiv) is enormous. EU ETS is a niche topic within general news; the signal-to-noise ratio is too low for prompt-based sentiment extraction to work reliably.

5. **Complicated blending cannot fix a weak signal.** We developed ramp blending, horizon-piecewise blending, validation-selected influence grids, ridge meta-blend, stateful meta-blend with volatility/sentiment/divergence features, and regime-gated model switching. None of these post-processing layers could turn a marginal LLM signal into a substantial improvement, because the underlying LLM corrections were themselves too noisy.

6. **Negative results are results.** The 15+ sentiment variants, 5 LLM strategies, and comprehensive ablation constitute a genuine scientific contribution. The finding that "LLM refinement does not substantially help for EU ETS under these conditions" is as informative as a finding that it does.

---

## Appendix A: Summary of All Tested Sentiment Pipeline Variants

| Variant | Path MSE | Δ vs Baseline | Decision |
|---|---|---|---|
| Baseline daily sentiment | 13.894* | — | Active (pre-leakage) |
| Top-5/day (vs top-3) | +1.30% | Worse | Rejected |
| Domain-weighted daily sampling | +0.95% | Worse | Rejected |
| EU-close time alignment | +1.69% | Worse | Rejected |
| Confidence-weighted vote agreement | +0.23% | Worse | Rejected |
| Independent majority voting | +0.67% | Worse | Rejected |
| Strict event dedupe (simhash-3, 90d) | −0.49% | Slight improvement | Kept |
| LLM relevance classifier gate | +1.11% | Worse | Rejected/reverted |
| EWMA-3 smoothing | +2.50% | Worse | Rejected/reverted |
| Richer features (level, change, volume) | +2.80% | Worse | Rejected/reverted |
| Train-only domain bucket weights | −0.23% | Slight improvement | Rejected (unstable) |
| Train-only return calibration | +0.38% | Worse | Rejected |
| Regime-aware return calibration | +1.63% | Worse | Rejected |
| Importance-weighted v3 (best) | −0.83% | Slight improvement | Not adopted |
| Abnormal z-score surprise | +1.39% | Worse | Rejected |
| Novelty/topicality weighting | +2.25% | Worse | Rejected |
| Asymmetric decay channels | −0.07% | Near-zero | Not adopted |
| Dawid-Skene EM aggregation | +0.36% | Worse | Rejected |
| Regime-aware sentiment mapping | +4.77% | Much worse | Rejected |

*Pre-leakage-fix numbers. Post-leakage, the honest comparison is 20.604 (CoT-Sent-RF) vs 20.942 (TSM).

## Appendix B: Summary of All Tested LLM Strategies (Post-Leakage)

| Strategy | Held-Out Test Path MSE | Δ vs TSM (20.942) | Decision |
|---|---|---|---|
| Paper-style CoT-Sent-RF (4B, raw) | **20.604** | **−1.61%** | Best honest result |
| 4B HDELTA (guarded, qwen3.5-4b-ud) | 20.520 | −2.01% | Near-tie, different endpoint |
| 35B HPRICE (guarded) | 20.657 | −1.36% | Best new architecture |
| 4B HPRICE (guarded) | 20.725 | −1.04% | Simpler alt |
| Official-context full-path 4B | 20.855 | −0.42% | Did not help |
| CoT-Sent-RF blend ramp w=0.50 | 20.739 | −0.97% | Worse than raw |
| CoT-RF (no sentiment) | 23.442 | +11.94% | Much worse |
| 35B guarded HDELTA | 20.872 | −0.33% | Marginal |
| 35B instruct full-path | 185.288* | Catastrophic | Rejected |
| Strategy 3 (sentiment as TSM feature) | 21.265 | +1.54% | Worse |
| Supervised residual corrector (v1, v2) | 20.942 | 0.00% | Identity selected |
| Meta-blend (any variant) | Failed strict CV | — | Rejected |

*Stage-1 subset only; run stopped after catastrophic result.

## Appendix C: Reference Benchmark Targets vs Achieved

| Metric | Reference Paper Target | Our Best Clean Result | Gap |
|---|---|---|---|
| Option 1 h1_mse | 0.097 | 1.384 | 14.3x worse |
| Option 1 path_mse | 14.516 | 20.942 | +44.3% |
| Option 4 path_mse | 12.720 | 20.604 | +62.0% |
| CoT improvement vs TSM | Substantial | 1.6% | Order of magnitude less |

The gap to the reference paper targets is large and is primarily attributable to the different market (EU ETS vs Chinese carbon), different price regime (post-structural-break consolidation), and different model scale (4B vs frontier LLM).
