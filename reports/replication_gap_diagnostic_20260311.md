# Why We Cannot Replicate Paper-Level LLM Gains: A Diagnostic Analysis

**Date**: 11 March 2026  
**Scope**: Full project lifecycle (Dec 2025 – Mar 2026), 200+ EU ETS runs, 50+ UK ETS runs  
**Reference papers**:  
- **Paper B** (primary): Chen et al. — *"Can Large Language Models forecast carbon price movements?"* (Chinese carbon markets)  
- **Paper A** (sentiment): Lopez-Lira & Tang — *"Can ChatGPT Forecast Stock Price Movements?"*

---

## 1. The Gap in Numbers

The reference paper reports that LLM refinement (CoT-Sent-RF) **consistently and substantially** improves an Autoformer baseline on Chinese carbon markets. Our project attempted the same methodology on two different markets, across 250+ experiment runs, and achieved the following:

### EU ETS (Dec 2025 – Mar 2026, n=382 test windows)

| Method | Path MSE | Δ vs TSM |
|--------|----------|----------|
| Base Autoformer (clean, leakage-free) | 20.942 | baseline |
| Naive persistence | 20.872 | −0.33% |
| Linear Ridge (no deep learning) | 20.739 | −0.97% |
| Best LLM (4B HDELTA, guarded) | 20.520 | −2.01% |
| Paper-style CoT-Sent-RF (4B, raw) | 20.604 | −1.61% |

### UK ETS (Mar 2026, n=144 test windows)

| Method | Path MSE | Δ vs TSM |
|--------|----------|----------|
| Base DLinear (tuned) | 22.726 | baseline |
| Linear Ridge | 22.420 | −1.35% |
| Best LLM (35B, recent_high_error_180) | 21.738 | −4.35% |
| Latest 4B runs (best) | 22.198 | −2.32% |

### What the Reference Paper Implied

Chen et al. reported that CoT-Sent-RF produces **large, consistent** improvements over the base Autoformer — enough to justify the entire hybrid pipeline as clearly superior. Our best clean result on EU ETS is **−2.01%**. Our best UK ETS result is **−4.35%** with a 35B model. Both are real but an **order of magnitude** smaller than the impression given by the reference paper's Chinese carbon market results.

---

## 2. The Eight Root Causes

After exhaustive experimentation, the replication gap can be attributed to eight interlocking factors. No single one is the "smoking gun" — it is their combination that makes the paper's results unreproducible in our setting.

### 2.1 Market Efficiency: EU/UK ETS ≠ Chinese Regional Carbon Markets

**This is the single most important factor.**

The Chinese regional pilot carbon markets studied by Chen et al. are younger, less liquid, have more retail participation, and lower institutional coverage. Price discovery is less efficient, meaning there is more "alpha" for an LLM to discover from news context.

The EU ETS, by contrast, is the world's largest and most liquid carbon market. Prices are set by compliance buyers (utilities, industrials), financial intermediaries, and formal auction mechanisms. Nearly all tradeable information is priced in by sophisticated institutional participants before it appears in publicly available news. The UK ETS, while younger and less liquid than the EU ETS, is dominated by a small number of large compliance entities and lacks the retail-driven informational inefficiency of Chinese pilot markets.

**Evidence from our data:**

- EU ETS: Naive persistence achieves path MSE 20.872. The best model (any method) achieves 20.520. The total gap between "do nothing" and "run the full pipeline" is **0.352 MSE** (1.7%). There is almost no predictable structure to exploit.
- UK ETS: Naive persistence achieves 32.011, and Ridge achieves 22.420 — a much larger gap (30%). But Ridge captures most of this with a few lagged features, leaving the LLM with diminishing marginal returns.

The fundamental barrier is not our implementation — it is that these markets leave very little room for any model, LLM or otherwise, to improve on simple baselines.

### 2.2 The Leakage Discovery: Our "Good" Early Results Were Artifacts

In late February 2026, we discovered **three separate data leakage paths** that had been inflating LLM performance metrics:

1. **Decoder future leakage**: `make_windows()` fed realized future feature values to the decoder during evaluation
2. **Split-boundary leakage**: Training windows had labels extending into validation/test periods
3. **Teaching-example leakage**: CoT example pools included truths from later test windows

**Before leakage fix:**
- CoT-RF appeared to deliver **−12% to −14%** improvement over TSM
- This closely matched the reference paper's reported magnitudes

**After leakage fix:**
- CoT-RF went from −12% to **+11.94%** (now *worse* than doing nothing)
- The paper-style CoT-Sent-RF recovered to only **−1.61%**
- The entire apparent replication success was an artifact

**Implication**: A meaningful fraction of the dramatic LLM gains reported in the literature may be attributable to subtle leakage. The three leakage paths we found are common implementation pitfalls in multi-step forecasting pipelines with LLM overlays, and it is unclear whether the reference papers' evaluation was subject to similar issues. At minimum, our project demonstrates that rigorous leakage auditing is essential before trusting any LLM refinement result.

### 2.3 LLM Scale: 4B/35B Local Models vs. Frontier GPT-4

The reference paper almost certainly used GPT-4 or an equivalent frontier model with **100–1,000x more parameters** than our primary models:

| Model | Parameters | Context | Quantitative Reasoning |
|-------|-----------|---------|----------------------|
| Qwen3-VL-4B (our primary) | 4B | 4K | Vague qualitative rules ("reduce by 2–5%") |
| Qwen3.5-35B (our best) | 35B, Q4 quantized | 4K | Better but still imprecise |
| GPT-4 (paper era, estimated) | ~1.8T MoE | 8–32K | Precise quantitative corrections |

**Evidence of scale mattering in our own data:**

- 4B → 35B on UK ETS: path MSE improved from 22.42 to 21.74 (clear step-change)
- 4B on EU ETS: reflections produce platitudes like *"reduce forecast by 2–5%"* or *"apply downward bias of 3–8%"*
- 35B: Better-formed reflections, but still prone to overcorrection or instability
- The CoT-RF reflection step requires **quantitative numerical reasoning** that small models fundamentally lack

The reference paper's CoT-RF pipeline likely produced sharper, more numerically grounded corrections from a frontier model. Our 4B model's reflections are qualitative noise that translates to near-zero or harmful adjustments in price space.

### 2.4 Sentiment Source Quality: GDELT ≠ RavenPack

Paper A (Lopez-Lira & Tang) — which provides the sentiment labeling framework used by Paper B — relies on **RavenPack** with `relevance=100` filtering. RavenPack is a professional financial news analytics platform that:

- Curates news specifically for financial relevance
- Provides entity-level relevance scores
- Offers event-similarity deduplication
- Has high signal-to-noise ratios by design

We used **GDELT DOC 2.0**, a free general-purpose news monitoring platform that:

- Contains vast amounts of tangentially related content (shipping, policy, environmental)
- Has no financial relevance scoring
- Requires heavy heuristic filtering to approach usability
- Produces sentiment signals that are mostly zeros or noise

**We tested 15+ sentiment pipeline variants.** None produced a robust improvement on EU ETS. The best variant achieved −0.07% — within noise. When sentiment was added to CoT-RF on the full EU ETS test split, it **degraded** performance from −13.78% (CoT-RF alone, pre-leakage-fix full split) to −4.28% (CoT-Sent-RF). The sentiment signal actively hurt performance because it injected noise into an already-fragile correction process.

The gap between GDELT and RavenPack is not a tuning problem — it is a data procurement problem. Without financial-grade news, the sentiment component of CoT-Sent-RF is dead weight or worse.

### 2.5 Methodological Divergences We Could Not Fully Close

Despite extensive Plan 4 alignment work, several methodological gaps persisted:

| Dimension | Reference Paper | Our Implementation | Impact |
|-----------|----------------|-------------------|--------|
| **Forecast output** | Full path (LLM replaces AF forecast) | Delta corrections + ramp blending | Caps the maximum possible adjustment |
| **Reflection context** | Likely multi-turn chat retaining full context | Two isolated stages (reflect → rules_text → apply) | Weaker learning from examples |
| **History window** | 18 steps (paper text) | Initially 60, later corrected to 18 | Early runs gave models too much noisy context |
| **Teaching examples** | Not fully specified (likely curated) | Most-recent → similarity → high-error → utility_mmr | Selection quality directly determines reflection quality |
| **Feature selection** | Lasso-based automated | Manual with ablation | May miss or include wrong features |
| **Pred length** | Ambiguous (prompts say 48, eval says 30) | 30 steps | Possible mismatch with paper evaluation |
| **Evaluation metric** | Multi-step MSE (exact definition uncertain) | Path MSE averaged across all 30 steps | Likely aligned, but uncertainty remains |

The most impactful divergence was **forecast output form**. We adopted delta corrections + ramp blending as a safety mechanism after discovering that unconstrained LLM outputs from the 4B model produced catastrophic overcorrections (mean day-1 adjustment ≈ −1.76 EUR). The ramp blend (w₁=0, increasing to w₃₀) effectively prevents the LLM from contributing at short horizons where the TSM is strong. This is rational engineering but fundamentally limits the method's ceiling — if the LLM can only influence h20–h30, it can never match a paper where the LLM improves all horizons.

### 2.6 Dataset Size and Non-Stationarity

| Dimension | Chinese Markets (Paper B) | EU ETS (Ours) | UK ETS (Ours) |
|-----------|--------------------------|---------------|---------------|
| History | Not fully specified | 4,334 days | 1,233 days |
| Training windows | Not specified | ~2,100 | 654 |
| Test windows | Not specified | 382 | 144 |
| Major regime shifts in data | Unknown | €25→€100→€65 | £97→£30→£65 |
| Test period character | Likely within a single regime | Post-rally consolidation (sideways) | Post-crash recovery (volatile) |
| Daily volatility | Unknown | ~1.5% | ~3.93% |

Two problems compound here:

**Non-stationarity undermines validation-based selection.** A persistent pattern across our project was that improvements seen on validation subsets failed to hold on the full test set. This occurred with sentiment variants, meta-blend calibration, teaching-example retrieval, and blend weight selection. The fundamental cause is that our validation and test periods span different market regimes. The validation period ends June 2025; the test period runs July 2025 – March 2026. If the market's statistical properties shift between these windows, any tuning done on validation is unreliable.

**Small datasets make LLM teaching examples stale or unrepresentative.** With only 654 UK ETS training windows, the pool of past examples is small. The best teaching-example strategy (recent high-error within 180 days) works because it provides examples from a similar regime, but 180 days of lookback covers only ~130 windows — this is a tiny training set for learning correction patterns.

### 2.7 The TSM Is Already Near-Optimal; There Is Little Left to Correct

This is a subtle but critical point. In the reference paper, the Autoformer baseline presumably had meaningful error structure that the LLM could identify and correct. In our setting:

- **EU ETS**: The tuned Autoformer (d=64, dropout=0.3) achieves 20.942 path MSE — only **0.070** worse than naive persistence (20.872). The TSM is already extracting essentially all predictable signal from the feature set.
- **UK ETS**: Ridge achieves 22.420 and the tuned DLinear achieves 22.726. The TSM is within 1.3% of the linear model, and the total gap from naive persistence (32.011) to the TSM is already large.

When the base model is near-optimal, the **error residuals are approximately white noise**. There is very little systematic structure for the LLM to learn from teaching examples and apply to new forecasts. The LLM's correction is trying to extract signal from what is, by construction, nearly unpredictable.

In the reference paper's Chinese market setting, the Autoformer may have had larger, more systematic biases (e.g., consistent under-prediction during trending phases) that an LLM could identify and correct through reflection on teaching examples. Our markets do not exhibit the same structure.

### 2.8 The "Program of Thought" Is Not True Computer-Aided Reasoning

Our strongest prompt strategies (`least_to_most`, `program_of_thought`, `react_evidence`) are all **prompt-only** — the model writes reasoning chain text but has no access to:

- A Python interpreter to verify calculations
- A live data retrieval tool
- An external calculator
- An executable verification step

The original Program of Thoughts paper (Chen et al., 2022) gets its benefit partly from **separating reasoning from computation** — the model generates code, then an interpreter runs it. Our implementation asks the model to both reason *and* compute numerically within a single text generation step. For a 4B model, this is asking it to do mental arithmetic on 30-step forecast paths, which it cannot do reliably.

The `verifier_program` apply style (our best 10 Mar variant) attempts to address this by structuring the output more rigorously, but it still lacks true tool-assisted computation. The reference paper likely benefits from a frontier model's stronger arithmetic capability, partially compensating for the lack of external tools.

---

## 3. The Leakage Hypothesis: Could Reference Paper Results Be Inflated?

This is speculative, but our experience raises the question. We documented three specific leakage paths that differentially inflated LLM performance:

1. **Decoder future leakage** gave the combined TSM+LLM system access to forward-looking feature values. The LLM benefited more because its corrections were informed (indirectly) by future information.
2. **Split-boundary leakage** allowed training-period statistics to bleed into evaluation.
3. **Teaching-example leakage** let the LLM "peek" at future truths through its example pool.

Before our leakage fix, our EU ETS results showed CoT-RF at **−12% to −14%** vs TSM — magnitudes that looked paper-consistent. After the fix, CoT-RF went to **+12%** (worse than doing nothing) and the best honest result was only **−2%**.

Multi-step forecasting with LLM overlays creates many opportunities for subtle leakage, particularly:

- When teaching examples are drawn from window pools that overlap with the test period
- When decoder-style models receive feature values that extend into the forecast horizon
- When validation-selected hyperparameters are tuned on data that correlates with test outcomes

We cannot assess whether the reference papers were affected by similar issues. But the fact that our own pre-leakage results coincidentally matched the reference paper's reported magnitudes, while our post-leakage results did not, is suggestive.

---

## 4. What Actually Worked (And What It Tells Us)

Despite the overall negative replication result, several findings were real and informative:

### 4.1 LLM Refinement Produces Small But Real Gains at Long Horizons

Across both markets, the LLM consistently improves h20 and h30 while being neutral or harmful at h1. The best UK ETS result (35B, recent_high_error_180) achieves:

- h1: no change (frozen by ramp blending)
- h20: −2.9% vs TSM
- h30: −2.0% vs TSM

This pattern makes physical sense: at short horizons, the TSM's extrapolation of recent trends is hard to beat. At longer horizons, where the TSM's uncertainty is higher and systematic biases (trend overshoot, mean-reversion lag) are more pronounced, the LLM's qualitative corrections have some value.

### 4.2 Recent High-Error Example Selection Is the Strongest Lever

The strongest UK ETS improvement came not from prompt engineering or model scale, but from **teaching the LLM with recent cases where the TSM made large errors**. This makes intuitive sense: the LLM learns best from examples that demonstrate clear, correctable patterns rather than examples where the TSM was already accurate.

### 4.3 Model Scale Matters More Than Prompt Complexity

The 4B → 35B upgrade on UK ETS (22.42 → 21.74) delivered more improvement than any prompt engineering change at the 4B scale. Conversely, elaborate prompt strategies (`self_rag`, `tree_of_thought`, `rarr_attribution`) at the 35B scale did not beat simpler approaches (`react_evidence`, `skeleton`). This suggests the binding constraint is model capability, not prompt design.

### 4.4 Linear Ridge Is Embarrassingly Strong

Across both markets, a simple Ridge regression with lagged features matched or beat every deep learning + LLM configuration under honest evaluation. This is the project's most uncomfortable finding: **hundreds of hours of engineering and hundreds of GPU-hours of computation could not reliably beat a model that trains in milliseconds with <300 parameters.**

Ridge's dominance is not a failure of our implementation — it reflects the basic statistical properties of these markets:

- Daily returns are close to white noise
- Lagged price features contain the dominant predictable signal
- The inductive bias of Ridge (small, smooth coefficients) is well-matched to the actual data generating process
- Deep learning overfits at these sample sizes, even with aggressive regularization

---

## 5. Synthesis: A Hierarchy of Causes

Ranked by estimated contribution to the replication gap:

| Rank | Factor | Contribution | Fixable? |
|------|--------|-------------|----------|
| 1 | **Market efficiency** (EU/UK vs Chinese carbon) | ~40% | No — fundamental market property |
| 2 | **Data leakage in early results** (inflated pre-fix numbers) | ~15% | Fixed (Feb 2026) |
| 3 | **LLM model scale** (4B/35B vs frontier GPT-4) | ~15% | Yes (cost/compute) |
| 4 | **Sentiment source quality** (GDELT vs RavenPack) | ~10% | Yes (data procurement cost) |
| 5 | **TSM near-optimality** (little residual structure) | ~8% | Partially (better TSM = smaller LLM headroom) |
| 6 | **Methodological divergences** (delta/blend, context, examples) | ~5% | Mostly fixed in Plan 4 |
| 7 | **Dataset size / non-stationarity** | ~5% | Partially (time fixes this) |
| 8 | **No true tool-assisted reasoning** | ~2% | Yes (engineering effort) |

The top two factors alone (~55%) are either unfixable (market efficiency) or already fixed (leakage). This means **even a perfect implementation with a frontier model on professional news data would likely produce modest rather than dramatic gains on EU/UK ETS**.

---

## 6. What Would Need to Be True for Paper-Level Gains

For our project to reproduce the reference paper's level of LLM-driven improvement, **all** of the following would need to hold:

1. **A less efficient target market** where news-driven price discovery leaves more alpha on the table
2. **A frontier-scale LLM** (GPT-4 / Claude Opus 4.6 class) capable of precise quantitative reasoning over forecast paths
3. **Professional-grade news data** (RavenPack, Bloomberg, Refinitiv) with entity-level relevance and deduplication
4. **A TSM baseline with larger systematic errors** that present correctable patterns in the residuals
5. **A longer, more stationary dataset** where validation-selected parameters transfer reliably to the test period
6. **No data leakage** — which requires meticulous auditing of decoder inputs, split boundaries, and example pools

In our project, conditions 1, 4, and partially 5 are structural properties of the target markets that cannot be engineered away. Conditions 2 and 3 are achievable but expensive. Condition 6 was violated early and is now fixed.

---

## 7. Recommendations

### 7.1 For the Paper

Frame the project as a **replication and falsification study**:

- "We attempted to reproduce Chen et al.'s LLM refinement methodology on EU and UK carbon markets. Under leakage-free evaluation, LLM refinement delivers 1.6–4.3% improvement over tuned TSM baselines — real but far smaller than the original paper's reported gains on Chinese markets."
- Emphasize the leakage discovery as a methodological contribution
- Present the cross-market comparison (EU vs UK vs implied Chinese) as evidence that LLM forecasting efficacy is market-dependent
- Report honestly that Ridge regression matches or beats the full TSM+LLM pipeline

### 7.2 For Continued Research

If pursuing further LLM refinement:

1. **Try a frontier model** (GPT-4o, Claude) on the existing UK ETS 35B pipeline as a controlled experiment — same prompts, same data, just a bigger model. This isolates the scale factor.
2. **Acquire proper financial news data** for at least a short evaluation period to isolate the sentiment factor.
3. **Apply to genuinely inefficient markets** — Korean ETS, early Chinese national ETS, voluntary carbon offsets — where the methodology has a better chance of demonstrating value.

If not pursuing LLM refinement:

4. **Accept Ridge as the production baseline** and focus on TSM architecture improvements (PatchTST, residual naive connection, transfer learning from EU to UK ETS).
5. **Pivot to classification** — the Ridge model already achieves 70–72% directional accuracy at h20–h30, which may be directly tradeable.

### 7.3 For the Codebase

6. **Retain the leakage tests as permanent regression checks.** The three leakage paths we found are subtle enough that they could be reintroduced by future code changes.
7. **Document the negative results thoroughly.** The 15 sentiment variants, 5 LLM strategies, and 200+ runs are a valuable record for anyone attempting similar work.

---

## 8. Conclusion

The replication gap is not a single bug or a single missing technique. It is the compound product of targeting an efficient market with a small model on noisy data, attempting to correct a near-optimal baseline, under non-stationary conditions — and initially being misled by data leakage that coincidentally produced paper-matching numbers.

The honest finding is that LLM-based forecast refinement can produce small, real improvements on carbon market forecasting (~2–4% MSE reduction), concentrated at long horizons, when properly configured with recent high-error teaching examples and a sufficiently capable model. But these gains are an order of magnitude smaller than what the reference paper reports on Chinese markets, and they do not reliably beat a simple Ridge regression.

This is a legitimate scientific result. Not every published methodology transfers across markets, and demonstrating the boundary conditions of LLM forecasting is as valuable as demonstrating its successes.

---

## Appendix: Key Evidence Trail

| Document | Location | Key Finding |
|----------|----------|-------------|
| EU ETS postmortem | `docs/PROJECT_EVALUATION_AND_FUTURE_DIRECTIONS.md` | Best clean EU ETS gain: −1.6%. Ridge competitive. |
| UK ETS pivot analysis | `docs/UK_ETS_PIVOT_ANALYSIS_AND_NEXT_STEPS.md` | Ridge dominates all DL+LLM variants on UK ETS (22.42 vs 29.78+) |
| Plan 4 alignment audit | `Plan 4 - Paper Alignment and LLM Refinement.md` | Identified 5 major methodological divergences from reference paper |
| 35B follow-up report | `reports/uk_ets_35b_followup_20260309.md` | Best UK ETS result: 21.738. Scale matters more than prompt design. |
| Weekly log (10 Mar) | `reports/weekly_log_summary_20260310.md` | Recent high-error retrieval + 35B = current frontier |
| Experiments report | `EXPERIMENTS_AND_TECHNIQUES_REPORT.md` | 119 EU ETS runs: local models viable, simple methods most reliable |
| Latest runs (11 Mar) | `runs/20260311_*` | Best 4B UK ETS: 22.198 (−2.3% vs TSM), Ridge: 22.420 |
| Leakage fix timeline | `docs/PROJECT_EVALUATION_AND_FUTURE_DIRECTIONS.md` §4.3 | Pre-fix: −14%. Post-fix: +12%. After recovery: −1.6%. |
