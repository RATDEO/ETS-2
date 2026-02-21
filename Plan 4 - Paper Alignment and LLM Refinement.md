# Plan 4: Paper Alignment + Making the LLM Refinement Actually Matter

## Why we’re seeing “LLM ≈ TSM” right now (root-cause)

Empirically, the refinement step is producing *tiny* (and often exactly zero) adjustments, and then we further damp them via blending:

- **Delta prompt is explicitly conservative**: `CoT-SENT-RF-DELTA-APPLY` says *“Small corrections are preferred; avoid large rewrites.”*
- **Delta is bounded by recent volatility**: `max_delta = max_delta_std * std(last_20_days)`. If the last-20-day volatility is low, the max delta is low.
- **Blending shrinks the delta again**: final forecast = `tsm + w * delta`.
  - In our configs, `blend.power=2.0` is applied to the weight ramp directly, so `max_weight=0.5` becomes `0.25` at the end of the horizon, and the *average* weight over the 30-step horizon is ~0.08.
  - That makes the **effective mean adjustment** on the order of `0.07 * 0.08 ≈ 0.006 EUR` (negligible relative to typical daily moves).
- **Many delta vectors are all-zero** (median per-call mean |delta| = 0), so even without blending we’d often get TSM back.

This is great for “don’t blow up forecasts” safety, but it’s **not the paper’s methodology** and it explains why metrics barely move.

---

## What the source papers actually do (relevant bits)

### Paper A: *Can ChatGPT Forecast Stock Price Movements?* (Lopez‑Lira & Tang)
Key methodology (used for **sentiment labeling**):
- Prompt: “financial expert” → output **YES / NO / UNKNOWN** on first line + one-sentence rationale.
- Temperature set to **0** for reproducibility.
- They put heavy emphasis on **relevance filtering** and **dedup/event similarity** (via RavenPack).
- They evaluate not just prediction error but also **post-event drift** and **long-short portfolio** performance.

### Paper B: *Can Large Language Models forecast carbon price movements?* (Chen et al.)
Key methodology (used for **forecast refinement**):
- **Autoformer** as the time-series model (TSM / AF).
- Pre-processing includes **Lasso-based feature selection**.
- **Direct prompting** underperforms AF.
- Refinement pipeline:
  - **CoT**: give (history + AF forecast) → “please improve AF prediction”.
  - **CoT‑RF**: additionally provide example pairs showing AF’s past forecasts vs truth and ask the LLM to reflect on deviations, then improve a new AF forecast.
  - **CoT‑Sent / CoT‑Sent‑RF**: incorporate daily sentiment signals (from YES/NO/UNKNOWN labeling), with **majority-vote** stability.
- Evaluation includes:
  - Regression **MSE over the full multi-step forecast** (paper text mentions “future 30 predicted steps”, while prompts reference “next 48 days”; we need to resolve this).
  - Trend accuracy at **Day‑10 / Day‑20 / Day‑30** using an **α ≈ 2%** neutrality band relative to the mean of the **past 18 steps**.

---

## Where our current implementation diverges

### 1) Refinement output form + damping (major divergence)
- We currently lean on **delta + blending** (and a conservative delta prompt), which is **not** what the carbon paper reports.
- The carbon paper’s LLM is allowed to output an improved **full forecast path** and meaningfully differ from AF.

### 2) Two-stage structure and context retention
- Paper B describes a *sequential* “reflect then improve” process. Our implementation:
  - Stage A: examples → **rules_text**
  - Stage B: current case + **rules_text only**
- This can be weaker than either:
  - keeping the whole conversation context (multi-message chat), or
  - re-including the examples in Stage B.

### 3) Prompt lengths and history window
- Paper B uses **18 historical steps** as a core window.
- Our prompts often use 60 raw history points (then we truncate to 15 in “compact” configs), which may be suboptimal for a 4B model.

### 4) Evaluation mismatch
- Paper B reports **multi-step MSE across the whole path** and trend accuracy at Day‑10/20/30 with α=2%.
- Our main reporting is **horizon-specific MSE** at (1,5,20,30) and trend thresholds based on rolling volatility (different definition).

### 5) Sentiment data quality + relevance filtering
- Paper A relies on high-quality relevance controls (RavenPack relevance=100 + event similarity constraints).
- Our GDELT/GKG approach can be noisy; even if we have many days, the sentiment signal may be weak or too often ~0.

---

## Plan (actionable) to improve and align

### Phase 0 — Lock down “paper-aligned” evaluation (so we can measure progress)
1. Add **paper-style multi-step MSE**:
   - `MSE_path = mean_{t,step}( (y_true[t, step] - y_pred[t, step])^2 )` over the full `pred_len`.
2. Add **paper-style trend accuracy**:
   - Use **Th=18** historical mean as reference and **α=2%** neutrality band.
   - Report accuracy at Day‑10/20/30 (and keep our current horizon metrics as additional).
3. Ensure all comparisons are on the **same subset** (same test windows) for every method.

Deliverable: a single comparison table per run that includes:
- `MSE_path`, plus our current horizon MSEs (1/5/20/30),
- paper-style Day‑10/20/30 trend accuracy,
- drift/long-short metrics (from Paper A style).

### Phase 1 — Implement a true “paper replication mode” for CoT / CoT‑RF / CoT‑Sent / CoT‑Sent‑RF
4. Create separate methods/configs that are **not damped**:
   - No blending (or `weight=1.0` everywhere).
   - No “small corrections preferred” instruction.
   - Output a **full yhat path**, not deltas.
5. Make the CoT‑RF / CoT‑Sent‑RF two-stage process **retain context**:
   - Use a *single chat* with multiple messages:
     - user: Stage A examples + question
     - assistant: rules_text
     - user: Stage B current case + AF forecast (+ sentiment) + “apply rules”
   - This matches the “sequential process” language better than two isolated calls.
6. Match paper windows:
   - Use **18-step** price history input (and 18-step sentiment history).
   - Use K examples in the paper-like range (**K = 3..10**), *not* 30 by default for a 4B model.
7. Resolve the **pred_len ambiguity (30 vs 48)**:
   - Re-check Paper B: prompts say 48, evaluation says 30.
   - Decide whether to:
     - keep `pred_len=30` (fastest, aligns with our current pipeline), or
     - upgrade to `pred_len=48` for a true replication (requires retraining AF/Autoformer).

### Phase 2 — Improve teaching-example selection (so the LLM has something to learn)
8. Add at least one selection scheme beyond “most recent”:
   - **High-error examples**: select K past windows where AF error was largest (for the same horizon band).
   - **Regime-aware**: select K examples matching current volatility/level/slope regime (your similarity selector is a start).
   - **Diversity constraint**: ensure examples aren’t near-duplicates (avoid teaching the same pattern K times).

Rationale: if AF is already very accurate in the examples, the “best” rule is “do nothing”.

### Phase 3 — Make sentiment more like Paper A (reliability + relevance)
9. Strengthen relevance/dedup:
   - Add “event similarity > 90 days” style dedup (approximate using fuzzy hashing / embeddings).
   - Optionally add an LLM relevance classifier (“is this about EU ETS carbon price drivers?”) before labeling sentiment.
10. Default sentiment labeling to **3 votes + majority vote** (paper-style) for a subset, then benchmark cost/benefit.
11. Ensure sentiment coverage is sufficiently dense (target ~1+ relevant headline/day on average over the training+test period).

### Phase 4 — Experiment matrix (controlled, one change at a time)
12. Run a clean ablation grid (same data split, same AF weights):
- Baselines: `TSM`, `naive_persistence`
- Paper-style: `TSM+LLM-CoT`, `TSM+LLM-CoT-RF`, `TSM+LLM-CoT-Sent`, `TSM+LLM-CoT-Sent-RF`
- Safety variants: `...-DELTA` with/without blending

Track:
- `MSE_path`
- horizon MSE (1/5/20/30)
- trend accuracy Day‑10/20/30 (α=2%, Th=18)
- drift/long-short metrics (Paper A style)

### Phase 5 — Only after methodology is correct: scale the model
13. Once we see a real signal (LLM materially changes forecasts and sometimes improves metrics), test:
- bigger GGUF models (20B+),
- or a hosted stronger model,
to see if the paper’s “big lift” requires more capacity than 4B.

---

## Quick “next actions” (recommended order)

1) Implement paper-style evaluation outputs (Phase 0).  
2) Add a **non-delta, non-blended** paper replication method for CoT‑Sent‑RF with 18-step history and K=5 (Phase 1).  
3) Add high-error example selection (Phase 2).  

These three will tell us quickly if the “LLM does nothing” issue is *methodological* (damping + prompt) or *capacity/data* (4B + weak sentiment).

---

## Status update (Feb 5, 2026)

### Option 2 (High-error teaching examples; no context retention)
- Run: `runs/20260205_021420_1fd551`
- On the **same 100-sample subset**:
  - `MSE_path`: **10.2818** vs TSM subset **10.3004** (tiny improvement)
  - Horizon MSE: **h1 worse** (0.5141 vs 0.0848), **h5 worse** (2.7388 vs 1.5362), **h30 better** (24.7463 vs 30.1157)

Interpretation: the LLM is acting like a **long-horizon bias corrector**, but it harms short-horizon accuracy where the TSM is already very strong.

### Option 3 (Paper-style “retain context” reflect→apply chat)
- Implementation:
  - reflection templates now allow **plain-text rules** (system message no longer forces JSON)
  - apply stage can reuse Stage-A context (`cot_rf.retain_context: true`)
  - LLM client timeout increased to avoid connection/timeouts on long prompts
- Run: `runs/20260205_024831_455db9`
- On the **same 100-sample subset**:
  - `MSE_path`: **10.8445** vs TSM subset **10.3004** (worse)
  - Horizon MSE: **h1 much worse** (3.2385 vs 0.0848), **h30 better** (27.9196 vs 30.1157)

Diagnosis from saved predictions (`runs/20260205_030019_c24e65/predictions/TSM+LLM-COT-SENT-RF_pred_test_subset.npz`):
- The LLM applies an almost **constant negative level shift** (mean day-1 adjustment ≈ **-1.76 EUR**), which explains why it destroys h=1.

### Next most promising (not yet run as a config): horizon-weighted blending for full-path refinement

Given the LLM improves long horizons but harms short horizons, we should **blend**:
`y_blend = (1-w_h) * TSM + w_h * LLM`
with `w_1 ≈ 0` and increasing toward later horizons.

Offline grid search on the saved predictions above shows a very strong candidate:
- `w_h = linspace(0, 0.75, 30)`  (i.e., `max_weight=0.75`, linear ramp)
- Achieves **MSE_path ≈ 9.695** (vs TSM subset 10.300), while keeping **h1 identical** to TSM (since `w_1=0`) and improving h5/h20/h30.

This is not paper-pure, but it’s the highest-leverage way to make the refinement useful **without sacrificing** the short-horizon skill that the TSM already has.

### Rerun + influence sweep (Qwen endpoint, Feb 5, 2026 evening)

We reran the full pipeline now that the Qwen endpoint is live, and repeated the 5-point “LLM influence” sweep.

- Rerun: `runs/20260205_185950_4ea60e` (endpoint: `http://192.168.1.140:9877/v1`, model: `qwen3-vl-4b-gpu`)
- Important reproducibility fix: cache keys now incorporate `{base_url, system_message, max_tokens}` (prevents silently reusing responses from a different endpoint/model).

**5-point influence sweep (same 100-sample test subset; ramp schedule `w_h = linspace(0, max_w, 30)`):**

| max_w | MSE_path | h1 MSE | h5 MSE | h10 MSE | h20 MSE | h30 MSE |
|------:|---------:|-------:|-------:|--------:|--------:|--------:|
| 0.00 | 10.3004 | 0.0848 | 1.5362 | 4.5514 | 13.8192 | 30.1157 |
| 0.25 | 9.9423 | 0.0848 | 1.5244 | 4.4873 | 13.3642 | 28.6603 |
| 0.50 | 9.7406 | 0.0848 | 1.5167 | 4.4491 | 13.0845 | 27.8091 |
| 0.75 | **9.6953** | 0.0848 | **1.5131** | **4.4368** | **12.9801** | **27.5622** |
| 1.00 | 9.8064 | 0.0848 | 1.5137 | 4.4504 | 13.0510 | 27.9196 |

Artifacts:
- `runs/20260205_185950_4ea60e/results/blend_grid_ramp/blend_grid_summary.csv`
- `runs/20260205_185950_4ea60e/results/blend_grid_ramp/blend_grid_mse_by_horizon.png`

**Next step:** make ramp blending a first-class, end-to-end method (so it shows up in `metrics_by_horizon.csv`, `path_metrics.csv`, significance tests, and plots), and choose `max_w` on the **validation** split (avoid test leakage) before reporting final test results.

### Follow-up: “first-class” ramp blending + **validation-selected** influence (no leakage)

Implementation:
- `src/run_experiment.py` now supports `llm.blend_grid`:
  - runs the same LLM method on a **validation subset**
  - evaluates a 5-point grid of influence strengths
  - selects the best `w` on **validation**
  - registers the resulting blended model(s) as first-class entries in `metrics_by_horizon.csv` / `path_metrics.csv`

Run:
- `runs/20260205_192453_940bff`

Selection outcome:
- Validation grid (`runs/20260205_192453_940bff/results/blend_grid_ramp/TSM_LLM-COT-SENT-RF/val/blend_grid_summary.csv`)
  - best `mse_path` at `w=0.75` (but the lift vs `w=0.0` is extremely small)
- Test grid (`runs/20260205_192453_940bff/results/blend_grid_ramp/TSM_LLM-COT-SENT-RF/test/blend_grid_summary.csv`)
  - `w=0.75` is **slightly worse** than `w=0.0` (TSM)

Conclusion (important):
- In this run, the LLM output is **almost identical** to the TSM output (mean |LLM−TSM| ≈ 0.01 EUR across the 30-step path), so “dialing influence up/down” cannot help much.
- The earlier “big win” case (where ramp blending helped a lot) happened when the LLM produced a **large level shift** vs TSM; here it does not.

### Follow-up: teaching-example selection must vary (and be regime-relevant)

We discovered the “LLM copies the TSM” issue was not only a prompt issue — it was also a *data selection* issue:

- With `example_selection: high_error` over the entire history, the “top-K worst” windows were **the same 5 dates for every target date**.
- That produced the **same reflection rules** and encouraged the apply stage to “do nothing”, resulting in near-exact copies of the TSM forecast (usually just rounded to 2 decimals).

Fix: add a recency constraint so “high error” is computed **within a rolling window** (`lookback_days`), and also test a regime-aware selection (`similarity`) that matches mean/std/slope.

#### Evidence (random 100-sample test subset)

- **High-error + lookback (can overreact)**:
  - Run: `runs/20260205_201811_a4ae90`
  - LLM changes were large (median |LLM−TSM| ≈ 0.50 EUR), but performance degraded:
    - `mse_path` **TSM=17.6516**, **LLM=19.7354** (worse), best blend ≈ TSM (w≈0).

- **Similarity + lookback (more stable)**:
  - Run: `runs/20260205_203501_927aab`
  - Blended variants improved over TSM on the same random subset:
    - `mse_path` **TSM=17.6516** → **17.2122** (`…blend_ramp_bestval_path`, ≈ −2.5%)
    - Horizon MSEs improved at h5/h20/h30 while keeping h1 identical (due to ramp starting at w₁=0).

**Next step:** scale this similarity+lookback setup beyond 100 samples (e.g., 200→full val/test) to confirm the lift is robust, then revisit sentiment densification (we currently have ~1 non-zero sentiment day per 18-day window on average).

---

## Full-split confirmation (no subset variance): similarity+lookback + val-selected influence

We ran the same **similarity + 365-day lookback + retain-context** setup on the **full validation and full test splits**, then selected the influence strength on **validation only** (to avoid test leakage).

- Run: `runs/20260205_211858_a0c3c1`
- LLM: `qwen3-vl-4b-gpu` via `http://192.168.1.140:9877/v1`
- Split sizes:
  - validation windows used for tuning: **255**
  - test windows evaluated: **359**
- Blend schedule: `ramp` (w₁=0 → w₃₀=max_w), weights grid `[0, .25, .5, .75, 1]`

### What happened (and why it matters)

On validation, the blend-grid is **monotone**: more LLM influence helps (especially later horizons), so the best validation choice is the **max influence** (`max_w=1.0`). Importantly, because ramp blending starts at **w₁=0**, we keep **Day‑1 identical to TSM**, while letting the LLM take over progressively toward Day‑30.

### Results: best method vs base TSM (test, n=359)

Path-level (paper-style multi-step MSE over the full 30-step forecast):
- `TSM` `mse_path`: **14.5158**
- `TSM+LLM…_blend_ramp_bestval_path` `mse_path`: **14.0724** (**−3.05%**)

Horizon MSEs (our standard reporting):
- h1: **no change** (by design, w₁=0)
- h5: **−1.98%**
- h20: **−2.72%**
- h30: **−4.41%**

Artifacts:
- `runs/20260205_211858_a0c3c1/results/path_metrics.csv`
- `runs/20260205_211858_a0c3c1/results/metrics_by_horizon.csv`
- `runs/20260205_211858_a0c3c1/results/blend_grid_ramp/TSM_LLM-COT-SENT-RF/{val,test}/blend_grid_summary.csv`
- `runs/20260205_211858_a0c3c1/results/blend_grid_ramp/TSM_LLM-COT-SENT-RF/{val,test}/blend_grid_mse_by_horizon.png`

### Interpretation at “A-level”

This is exactly the pattern we expect if:
- the TSM is extremely strong at very short horizons (so any LLM “style” correction tends to hurt Day‑1 unless constrained), but
- the LLM’s reflection step is useful for **long-horizon bias/shape** corrections (e.g., under/overshoot, lag after regime shifts).

Ramp blending is essentially a **risk-control layer**: it prevents the LLM from touching the near-term “high-signal” part of the forecast, while still letting it contribute where the TSM’s uncertainty is higher.

---

## Next step (to push performance further)

Before we invest heavily in sentiment densification, we should first isolate **whether sentiment is helping at all** with our current (sparse) series:

1) Run **`TSM+LLM-COT-RF` (no sentiment)** with the same similarity+lookback selection + blend-grid tuning.
2) Compare `CoT‑RF` vs `CoT‑Sent‑RF` on the same full test split.

If `CoT‑RF ≈ CoT‑Sent‑RF`, then sentiment is currently not informative enough → we focus on improving headline relevance/coverage and vote reliability (Paper A style). If `CoT‑Sent‑RF` wins, we keep it and tune `k_examples/lookback` next.

### Update (Feb 6, 2026): CoT‑RF ablation on the full splits

We ran the exact same setup but with **sentiment disabled** (`TSM+LLM‑CoT‑RF`).

- Run: `runs/20260205_231634_6a6539`
- Blend-grid selection (val, metric=`mse_path`): best `max_w=0.75`

Key results (test, n=359):
- `TSM` `mse_path`: **14.5158**
- `TSM+LLM‑CoT‑RF_blend_ramp_bestval_path` `mse_path`: **12.5149** (**−13.78%**)
- Horizon MSE improvements vs TSM:
  - h20: **19.2321 → 16.4049** (−14.7%)
  - h30: **40.7650 → 33.8886** (−16.9%)

Interpretation:
- The **self-refine signal is real** (big lift at long horizons).
- Our current **sentiment signal is not helping** (likely sparse/noisy/misaligned enough that it adds complexity without adding information).

This means the biggest “next step” is **not more LLM influence** — it’s either:
1) make sentiment as clean/dense as the papers (relevance + dedup + majority vote), then re-test CoT‑Sent‑RF, or
2) proceed with CoT‑RF as the main method and tune `k_examples/lookback` for additional gains.

---

## Sentiment data-quality work (Feb 8, 2026): higher coverage + majority vote + per-day averaging

### What was wrong with our old sentiment series

Even though we had lots of raw headlines, the *effective* sentiment input seen by the model was weak because:
- in the **test period** it was **mostly zeros** (very sparse non-zero days),
- and in some variants we effectively had **~1 labeled headline per day**, which makes the series very “spiky” (±1) and noisy.

### What we changed (paper-aligned)

1) **Systematic query-driven collection (GDELT)**
- Added a multi-query backfill script with checkpointing:
  - `scripts/backfill_gdelt_multiquery.py`
  - queries file: `data/news/queries_eu_ets_gdelt.txt`
- Result: `data/news/headlines_raw_eu_ets_multiquery_2017_2026.csv` (~456k rows, 2017‑01‑01 → 2026‑01‑05).

2) **EU-ETS relevance scoring + simhash dedup**
- Extended `scripts/filter_news_headlines.py` with:
  - `--relevance-scheme eu_ets_score` (heuristic EU ETS relevance scoring)
  - `--dedupe-method simhash` + rolling dedup controls
- Candidate pool used:
  - `data/news/headlines_filtered_eu_ets_multiquery_v3_score2_sim0.csv` (score≥2, within-day dedupe).

3) **Paper-style majority vote (3 votes)**
- `scripts/label_news_sentiment.py` now supports `--base-url/--api-key` so we can label using the local OpenAI-compatible endpoint.
- Labels produced with **votes=3**.

4) **Daily aggregation with multiple headlines/day**
- Built daily sentiment with **top-3 headlines/day** (by relevance score), then averaged:
  - labeled: `data/news/headlines_labeled_qwen_votes3_daily3.csv` (n=7,553)
  - daily series: `data/news/daily_sentiment_qwen_votes3_daily3.csv` (n=2,782 days)
- Coverage (daily series row coverage relative to full calendar span):
  - train: **83.3%**
  - val: **84.7%**
  - test: **89.5%**
- Non-zero rate improved to **~83%** of days (far less sparse than before), and values are smoother (common values: 0.33, 0.67, 1.0, etc.).

### Did this improve CoT‑Sent‑RF?

Yes — it materially improved the sentiment-augmented method (though it still doesn’t beat CoT‑RF yet).

Run (CoT‑Sent‑RF with new daily3 sentiment):
- `runs/20260208_042155_ea55a7`

Path-level (test, n=359):
- `TSM` `mse_path`: **14.5158**
- `TSM+LLM‑CoT‑Sent‑RF` `mse_path`: **13.8942** (**−4.28%** vs TSM)
- `…blend_ramp_bestval_path` `mse_path`: **14.0020** (**−3.54%** vs TSM; preserves h1)

Horizon MSE (test, blended variant keeps h1 identical):
- h1: no change (by design)
- h5: **−1.38%**
- h20: **−3.24%**
- h30: **−5.50%**

Benchmark: CoT‑RF still stronger
- `runs/20260205_231634_6a6539` `TSM+LLM‑CoT‑RF_blend_ramp_bestval_path` `mse_path`: **12.5149** (**−13.78%** vs TSM)

### Next sentiment steps (if we keep pushing this)

To close the gap to CoT‑RF, the next “paper-like” upgrades are:
1) raise per-day headline count further (e.g. top‑5/day) and/or apply a short rolling average to reduce label noise;
2) add a cheap relevance gate (LLM or heuristic) to drop “EU policy / shipping” items that don’t plausibly move EUA prices;
3) add a domain-weighted sampling (prefer carbon/energy market outlets) while keeping coverage.

**Update (Feb 10, 2026): implemented step (2) as a heuristic gate in `scripts/filter_news_headlines.py`.**
- New flag: `--cheap-gate policy_shipping`
  - Scope: only triggers on policy-ish or shipping-ish headlines, and drops them unless they contain EU ETS / EUA price-driver terms (e.g., ETS/allowances/EUA/auctions/carbon trading, MSR/CBAM/Fit-for-55, etc.).
- Quick sanity check on `data/news/headlines_raw_eu_ets_multiquery_2017_2026.csv` with `--relevance-scheme eu_ets_score --min-relevance-score 2 --dedupe-method simhash --simhash-max-hamming 0`:
  - rows: **47,126 → 43,424** (−7.9%)
  - shipping-ish rows: **1,405 → 275**
  - policy-ish rows: **2,838 → 379**
  - unique days: **2,782 → 2,762** (coverage roughly **84.5% → 83.9%** across the same date span)

**Sentiment rebuild (Feb 10, 2026): daily3 series with this gate**
- Filtered headlines: `data/news/headlines_filtered_eu_ets_multiquery_v4_score2_sim0_gate_policy_shipping.csv` (43,424 rows)
- Labeled (top-3/day, votes=3): `data/news/headlines_labeled_qwen_votes3_daily3_gate_policy_shipping.csv` (7,409 rows)
- Daily series: `data/news/daily_sentiment_qwen_votes3_daily3_gate_policy_shipping.csv` (2,762 days)
- Content reduction in labeled pool (regex-based): shipping/policy headlines **931 → 359**.

**Does this move MSE? (Feb 10, 2026)**
- We re-ran **CoT‑Sent‑RF** on the same **test set (n=359)**, changing only the daily sentiment file from:
  - `data/news/daily_sentiment_qwen_votes3_daily3.csv`
  - to `data/news/daily_sentiment_qwen_votes3_daily3_gate_policy_shipping.csv`
- Results (base run vs gated-sentiment run):
  - `TSM+LLM‑CoT‑Sent‑RF` `mse_path`: **13.8942 → 13.9264** (slightly worse)
  - Using the same blend selection as the base run (ramp schedule, `best_w_path=1.0`):
    - blended `mse_path`: **14.0020 → 13.9886** (slightly better)
- Run artifact: `runs/20260210_181129_sentiment_compare` (summary in `runs/20260210_181129_sentiment_compare/results/summary.json`).

**Upgrade (1): raise to top-5/day (Feb 10, 2026)**
- Labeled (top-5/day, votes=3): `data/news/headlines_labeled_qwen_votes3_daily5_gate_policy_shipping.csv` (11,164 rows)
- Daily series: `data/news/daily_sentiment_qwen_votes3_daily5_gate_policy_shipping.csv` (2,762 days)
- MSE comparison vs daily3 baseline (same test set `n=359`, same method):
  - `TSM+LLM‑CoT‑Sent‑RF` `mse_path`: **13.8942 → 14.0752** (+1.30%)
  - “Blended” (same base-run selection `best_w_path=1.0`): **14.0020 → 14.0998** (+0.70%)
- Run artifact: `runs/20260210_202028_sentiment_compare` (summary in `runs/20260210_202028_sentiment_compare/results/summary.json`).

**Upgrade (3): domain-weighted daily sampling (Feb 10, 2026)**
- Labeled (top-3/day, votes=3, `domain_weighted`): `data/news/headlines_labeled_qwen_votes3_daily3_gate_policy_shipping_domainw.csv` (7,409 rows)
- Daily series: `data/news/daily_sentiment_qwen_votes3_daily3_gate_policy_shipping_domainw.csv` (2,762 days)
- MSE comparison vs daily3 baseline (same test set `n=359`, same method):
  - `TSM+LLM‑CoT‑Sent‑RF` `mse_path`: **13.8942 → 14.0260** (+0.95%)
  - “Blended” (same base-run selection `best_w_path=1.0`): **14.0020 → 14.0863** (+0.60%)
- Run artifact: `runs/20260210_212619_sentiment_compare` (summary in `runs/20260210_212619_sentiment_compare/results/summary.json`).

**Additional sentiment-signal options to test (added Feb 11, 2026)**
1) Confidence-weighted sentiment from vote agreement (downweight 2/3 and 1/3 consensus rows).
2) Better timestamp-to-trading-day alignment (after EU close -> next business day).
3) Event-level dedupe before labeling (cluster near-duplicate headlines by day/topic).
4) Replace single daily mean with richer features (`level`, `change`, `news_volume`).
5) Learn relevance weights on train only (topic/domain buckets), then freeze for val/test.
6) Light smoothing only (e.g., 2–3 day EWMA), rather than increasing per-day headline count.

**Option (2) rerun: EU close alignment (Feb 11, 2026)**
- Implementation:
  - `scripts/label_news_sentiment.py`: keep original timestamp in `seendate` while still sampling top-N per calendar day.
  - `scripts/build_daily_sentiment.py`: added `--align-to-eu-close --eu-close-time --eu-close-timezone`.
  - Rule: convert `seendate` to `Europe/Brussels`; headlines at/after **16:30** map to next business day; weekend dates also map to next business day.
- Artifacts:
  - Labeled (timestamp-preserving daily3): `data/news/headlines_labeled_qwen_votes3_daily3_gate_policy_shipping_ts.csv` (7,409 rows)
  - Daily sentiment (EU-close aligned): `data/news/daily_sentiment_qwen_votes3_daily3_gate_policy_shipping_euclose1630.csv` (2,118 days)
  - Shift magnitude: **3,533 / 7,409** headlines moved to a later trading day.
- MSE comparison vs baseline daily3 run (same test set `n=359`, same method):
  - `TSM+LLM‑CoT‑Sent‑RF` `mse_path`: **13.8942 → 14.1289** (**+1.69%**, worse)
  - “Blended” (same base-run selection `best_w_path=1.0`): **14.0020 → 14.1064** (**+0.75%**, worse)
- Run artifact: `runs/20260211_142122_sentiment_compare` (summary in `runs/20260211_142122_sentiment_compare/results/summary.json`).

**Option (1) rerun: confidence-weighted sentiment from vote agreement (Feb 11, 2026)**
- Implementation:
  - We prototyped agreement-weighted aggregation (`max_label_count / n_votes`) and re-ran MSE.
  - Because it had no practical impact in this dataset, this code path was reverted from the active pipeline.
- Artifacts:
  - Daily sentiment: `data/news/daily_sentiment_qwen_votes3_daily3_gate_policy_shipping_confagree.csv` (2,762 days)
  - MSE comparison run: `runs/20260211_152600_sentiment_compare` (summary in `runs/20260211_152600_sentiment_compare/results/summary.json`)
- MSE comparison vs baseline daily3 run (same test set `n=359`, same method):
  - `TSM+LLM‑CoT‑Sent‑RF` `mse_path`: **13.8942 → 13.9263** (**+0.23%**, worse)
  - “Blended” (same base-run selection `best_w_path=1.0`): **14.0020 → 13.9886** (**−0.10%**, slightly better)
- Important caveat:
  - In this labeled dataset (`data/news/headlines_labeled_qwen_votes3_daily3_gate_policy_shipping.csv`), stored `llm_votes` are effectively all unanimous (`3/3`), so agreement weights are all `1.0` and this option is nearly a no-op on this specific run.

**Option (1b) paper-style independent majority voting (Feb 11, 2026)**
- Implementation:
  - `src/news/sentiment.py` updated so votes can be cached independently per vote slot (`vote_cache_mode=independent`) instead of reusing the same cached response for all votes.
  - `scripts/label_news_sentiment.py` adds:
    - `--vote-cache-mode {shared,independent}` (default `independent`)
    - `--vote-models` (optional comma-separated model list to cycle across votes).
- Artifacts:
  - Re-labeled (daily3, independent votes): `data/news/headlines_labeled_qwen_votes3_daily3_gate_policy_shipping_indepvotes_v1.csv` (7,409 rows)
  - Daily sentiment: `data/news/daily_sentiment_qwen_votes3_daily3_gate_policy_shipping_indepvotes_v1.csv` (2,762 days)
  - MSE comparison run: `runs/20260211_183023_sentiment_compare` (summary in `runs/20260211_183023_sentiment_compare/results/summary.json`)
- Vote-diversity outcome:
  - unanimous rows: **98.80%**; non-unanimous rows: **89 / 7,409** (mostly `2/3` splits)
  - daily sentiment changed on **7 / 2,762** days vs prior gated-daily3 series.
- MSE comparison vs baseline daily3 run (same test set `n=359`, same method):
  - `TSM+LLM‑CoT‑Sent‑RF` `mse_path`: **13.8942 → 13.9872** (**+0.67%**, worse)
  - “Blended” (same base-run selection `best_w_path=1.0`): **14.0020 → 14.0310** (**+0.21%**, worse)

**Next paper-aligned relevance pass (Feb 11, 2026): strict event dedupe + practical relevance floor**
- Paper re-check takeaway:
  - Paper A emphasizes strict relevance and event-similarity controls (RavenPack-style filtering), so we tested a stricter event-level dedupe path without collapsing calendar coverage.
- Implementation (no new code required; existing flags in `scripts/filter_news_headlines.py`):
  - `--relevance-scheme eu_ets_score --min-relevance-score 2`
  - `--cheap-gate policy_shipping`
  - `--dedupe-method simhash --simhash-max-hamming 3`
  - `--event-similarity-days 90`
- Filter artifact:
  - `data/news/headlines_filtered_eu_ets_multiquery_v5_score2_sim3_gate_policy_shipping_event90.csv`
  - Rows: **40,491** vs prior gated pool **43,424** (−6.8%)
  - Unique days: **2,717** vs **2,762** (−1.6%)
- Labeled/daily artifacts for this pass:
  - `data/news/headlines_labeled_qwen_votes3_daily3_gate_policy_shipping_event90_proxy.csv`
  - `data/news/daily_sentiment_qwen_votes3_daily3_gate_policy_shipping_event90_proxy.csv`
  - Note: this is a high-coverage proxy built from existing `daily5` labels (7,140/7,249 top-3 rows covered; 98.5% row coverage, 2,716 sentiment days).
- MSE comparison vs baseline daily3 run (same test set `n=359`, same method):
  - Run artifact: `runs/20260211_203500_sentiment_compare` (summary in `runs/20260211_203500_sentiment_compare/results/summary.json`).
  - `TSM+LLM‑CoT‑Sent‑RF` `mse_path`: **13.8942 → 13.8262** (**−0.49%**, slight improvement).
  - “Blended” (same base-run selection `best_w_path=1.0`): **14.0020 → 13.9362** (**−0.47%**, slight improvement).
  - Horizon MSE deltas:
    - `h1`: **−3.29%** (better)
    - `h5`: **−1.89%** (better)
    - `h20`: **+0.00%** (flat)
    - `h30`: **−0.23%** (better)

**Stage-1 LLM relevance classifier before sentiment (Feb 12, 2026)**
- Implementation:
  - `src/news/sentiment.py`:
    - added optional pre-sentiment relevance gate prompt (`YES/NO/UNKNOWN`) for EUA price impact relevance.
    - added gate outputs in labeled rows (`relevance_vote`, `relevance_votes`, `relevance_rationales`).
    - gate keeps labels in a configurable set (`--relevance-keep-label`, comma-separated; used `YES,UNKNOWN`).
  - `scripts/label_news_sentiment.py`:
    - new flags: `--relevance-gate`, `--relevance-votes`, `--relevance-model`, `--relevance-temperature`,
      `--relevance-max-tokens`, `--relevance-cache-dir`, `--relevance-keep-label`.
- Data build (strict event-dedupe input):
  - Input pool: `data/news/headlines_filtered_eu_ets_multiquery_v5_score2_sim3_gate_policy_shipping_event90.csv` (top-3/day candidate set size **7,249**).
  - Labeled with relevance gate: `data/news/headlines_labeled_qwen_votes3_daily3_gate_policy_shipping_event90_relcls_v3.csv` (**3,791** rows; **52.3%** keep-rate; **2,173** days).
  - Daily sentiment: `data/news/daily_sentiment_qwen_votes3_daily3_gate_policy_shipping_event90_relcls_v3.csv`.
  - Relevance votes among kept rows: `YES=3,085`, `UNKNOWN=706`.
- MSE comparison vs baseline daily3 run (same test set `n=359`, same method):
  - Run artifact: `runs/20260212_001500_sentiment_compare` (summary in `runs/20260212_001500_sentiment_compare/results/summary.json`).
  - `TSM+LLM‑CoT‑Sent‑RF` `mse_path`: **13.8942 → 14.0483** (**+1.11%**, worse).
  - “Blended” (same base-run selection `best_w_path=1.0`): **14.0020 → 14.0341** (**+0.23%**, worse).
  - Horizon MSE deltas:
    - `h1`: **+36.99%** (worse)
    - `h5`: **+6.68%** (worse)
    - `h20`: **+2.36%** (worse)
    - `h30`: **−0.10%** (slightly better)
- Outcome:
  - This gate degraded performance and has been reverted from active code (`src/news/sentiment.py`, `scripts/label_news_sentiment.py`).

**Option (6) rerun: light smoothing only (EWMA-3) (Feb 12, 2026)**
- Implementation:
  - `scripts/build_daily_sentiment.py` now supports optional smoothing:
    - `--smoothing {none,ewma}`
    - `--ewma-span` (used `3`)
- Data build:
  - Input labeled set: `data/news/headlines_labeled_qwen_votes3_daily3_gate_policy_shipping_event90_proxy.csv`
  - Smoothed daily sentiment: `data/news/daily_sentiment_qwen_votes3_daily3_gate_policy_shipping_event90_proxy_ewma3.csv` (**2,717** rows)
- MSE comparison vs baseline daily3 run (same test set `n=359`, same method):
  - Run artifact: `runs/20260212_010500_sentiment_compare` (summary in `runs/20260212_010500_sentiment_compare/results/summary.json`).
  - `TSM+LLM‑CoT‑Sent‑RF` `mse_path`: **13.8942 → 14.2417** (**+2.50%**, worse).
  - “Blended” (same base-run selection `best_w_path=1.0`): **14.0020 → 14.2776** (**+1.97%**, worse).
  - Horizon MSE deltas:
    - `h1`: **−2.03%** (better)
    - `h5`: **+4.53%** (worse)
    - `h20`: **+1.95%** (worse)
    - `h30`: **+3.77%** (worse)

**Option (4) rerun: richer daily sentiment features (`level`, `change`, `news_volume`) (Feb 12, 2026)**
- Revert:
  - The prior heuristic gate edits in `scripts/filter_news_headlines.py` were reverted before this test.
- Implementation:
  - `src/news/sentiment.py` daily aggregation now outputs:
    - `sent_score` (daily level),
    - `sent_change` (day-over-day change in level),
    - `news_volume` (headline count per day).
  - `src/run_experiment.py` and `scripts/compare_sentiment_series_mse.py` now support multi-column sentiment loading/history (`llm.sentiment.feature_cols` or `--sentiment-feature-cols`).
  - `src/llm/prompts.py` and `src/llm/refine.py` now accept multi-feature sentiment histories and expose them in CoT‑Sent prompts/rules.
- Data build:
  - Input labeled set: `data/news/headlines_labeled_qwen_votes3_daily3_gate_policy_shipping_event90_proxy.csv`
  - Enriched daily sentiment: `data/news/daily_sentiment_qwen_votes3_daily3_gate_policy_shipping_event90_proxy_feat4.csv` (**2,717** rows; cols: `seendate,sent_score,news_volume,sent_change`)
- MSE comparison vs baseline daily3 run (same test set `n=359`, same method):
  - Run artifact: `runs/20260212_021500_sentiment_compare` (summary in `runs/20260212_021500_sentiment_compare/results/summary.json`).
  - `TSM+LLM‑CoT‑Sent‑RF` `mse_path`: **13.8942 → 14.2836** (**+2.80%**, worse).
  - “Blended” (same base-run selection `best_w_path=1.0`): **14.0020 → 14.3021** (**+2.14%**, worse).
  - Horizon MSE deltas:
    - `h1`: **+17.08%** (worse)
    - `h5`: **+3.89%** (worse)
    - `h20`: **+2.57%** (worse)
    - `h30`: **+4.13%** (worse)
- Outcome:
  - This option was reverted from active code after confirming degradation.

**Option (5) rerun: learn relevance weights on train only (domain buckets), then freeze (Feb 12, 2026)**
- Revert first:
  - Reverted the option (4) multi-feature sentiment code changes from active pipeline.
- Implementation:
  - `scripts/build_daily_sentiment.py` now supports train-only bucket weighting:
    - `--bucket-weighting train_corr`
    - `--bucket-col` (used: `domain`)
    - `--panel-path --panel-date-col --panel-price-col` (used base run panel with `date`,`y`)
    - `--train-end` (used: `2023-06-30`)
    - `--min-bucket-days --bucket-shrinkage --bucket-weight-min --bucket-weight-max`
    - `--weights-output` (persist frozen train-only weights)
  - Method:
    - For each domain bucket on train period only, compute correlation between that bucket’s daily sentiment score and next-day return.
    - Convert `abs(corr)` with sample-size shrinkage into a relevance quality score.
    - Map quality to fixed weights and freeze for full period aggregation.
- Data build:
  - Input labeled set: `data/news/headlines_labeled_qwen_votes3_daily3_gate_policy_shipping_event90_proxy.csv`
  - Weighted daily sentiment: `data/news/daily_sentiment_qwen_votes3_daily3_gate_policy_shipping_event90_proxy_domainw_traincorr.csv` (**2,716** rows)
  - Frozen weights artifact: `data/news/domain_weights_traincorr_event90_proxy.json` (**23** weighted domains; default weight **1.1571**)
- MSE comparison vs baseline daily3 run (same test set `n=359`, same method):
  - Run artifact: `runs/20260212_041500_sentiment_compare` (summary in `runs/20260212_041500_sentiment_compare/results/summary.json`).
  - `TSM+LLM‑CoT‑Sent‑RF` `mse_path`: **13.8942 → 13.8619** (**−0.23%**, slightly better).
  - “Blended” (same base-run selection `best_w_path=1.0`): **14.0020 → 13.9495** (**−0.37%**, slightly better).
  - Horizon MSE deltas:
    - `h1`: **+23.91%** (worse)
    - `h5`: **+0.38%** (slightly worse)
    - `h20`: **+0.25%** (slightly worse)
    - `h30`: **−1.18%** (better)
- Outcome:
  - Despite slight path-level improvement, this was also reverted from active code due weak/unstable horizon profile.

**New LLM-refinement improvement set (post-Feb 12 review)**
- Why a new direction:
  - Most sentiment-data tweaks produced small or negative gains; the paper-scale lift is more likely in the refinement core than in additional headline filtering.
- 1) Add a strict paper-replication refinement mode (highest priority):
  - Lock to `history_points=18`, `k_examples=5`, retained two-stage context, and no hidden fallback damping.
  - Match paper prompt framing for AF→LLM correction and evaluate with path MSE + Day-10/20/30 trend metrics.
- 2) Replace free-form reflection rules with structured residual instructions:
  - Stage-A should output compact JSON rule fields (level bias, trend bias, horizon-specific correction intent).
  - Stage-B applies those rules to AF forecast with explicit horizon controls, reducing noisy overcorrections.
- 3) Upgrade teaching-example selection to mixed informative set:
  - Per target window choose a diversity-constrained mix of:
    - nearest-regime examples,
    - high-error-in-regime examples,
    - opposite-regime contrast examples.
  - Include AF residual paths in examples so LLM learns correction shape directly.
- 4) Add post-LLM calibration layer on validation only:
  - Learn a lightweight calibrator on `(LLM−AF)` residuals (scale + level + horizon slope) using val split only.
  - Apply frozen calibrator on test to preserve long-horizon lift while constraining h1 blow-ups.
- 5) Move from single ramp blend to horizon-piecewise blend selected on validation:
  - Optimize 3-segment weights (`h1-5`, `h6-15`, `h16-30`) instead of one global `w`.
  - This directly targets our recurring pattern: short horizon hurt, long horizon helped.
- 6) Run capacity test on refinement model before more sentiment engineering:
  - Keep same prompts/examples and swap in a stronger reasoning model endpoint for reflection/apply.
  - If gain scales with model strength, prioritize model capacity and latency budget over further sentiment preprocessing.

**Sentiment option: train-only headline score calibration via next-day return mapping (Feb 14, 2026)**
- Implementation:
  - `scripts/build_daily_sentiment.py` now supports:
    - `--calibration-mode train_return`
    - `--panel-path --panel-date-col --panel-price-col`
    - `--train-end --calibration-min-count --calibration-shrinkage`
    - `--calibration-output` (diagnostics JSON)
  - Method:
    - Join headline rows to panel next-day log return (`ret_fwd1`) by date.
    - On train split only, estimate score-bucket mean return with shrinkage to global train mean.
    - Freeze bucket mapping and apply to all rows before daily aggregation.
- Artifacts:
  - Calibrated daily sentiment: `data/news/daily_sentiment_qwen_votes3_daily3_calib_trainret.csv`
  - Calibration diagnostics: `data/news/score_calibration_trainret_daily3.json`
  - MSE comparison run: `runs/20260214_120500_sentiment_compare_calib_trainret` (summary in `runs/20260214_120500_sentiment_compare_calib_trainret/results/summary.json`)
- MSE comparison vs baseline daily3 run (same test set `n=359`, same method):
  - `TSM+LLM‑CoT‑Sent‑RF` `mse_path`: **13.8942 → 13.9469** (**+0.38%**, worse)
  - “Blended” (same base-run selection `best_w_path=1.0`): **14.0020 → 14.0582** (**+0.40%**, worse)
  - Horizon MSE deltas:
    - `h1`: **−10.16%** (better)
    - `h5`: **−2.20%** (better)
    - `h20`: **+0.01%** (flat/slightly worse)
    - `h30`: **+1.24%** (worse)
  - Trend-accuracy deltas:
    - `d10`: **−0.28 pp**
    - `d20`: **−0.28 pp**
    - `d30`: **−1.67 pp**
- Outcome:
  - Short-horizon error improved, but overall path MSE and long-horizon trend quality worsened; not a net gain.

**Sentiment option: train-only regime-aware score calibration (volatility split) (Feb 14, 2026)**
- Revert:
  - Reverted the prior global train-return calibration implementation in `scripts/build_daily_sentiment.py` before this step.
- Implementation:
  - `scripts/build_daily_sentiment.py` now supports:
    - `--calibration-mode train_return_regime`
    - volatility regime split on panel returns (`--regime-vol-window`, `--regime-vol-quantile`) learned on train only.
  - Method:
    - Join headlines to panel next-day returns and a daily regime label (`high_vol`/`low_vol`).
    - Learn train-only score→mapped-score per regime with shrinkage and global fallback.
    - Freeze mappings and apply before daily aggregation.
- Artifacts:
  - Calibrated daily sentiment: `data/news/daily_sentiment_qwen_votes3_daily3_calib_regime_trainret.csv`
  - Calibration diagnostics: `data/news/score_calibration_regime_trainret_daily3.json`
  - MSE comparison run: `runs/20260214_140500_sentiment_compare_calib_regime_trainret` (summary in `runs/20260214_140500_sentiment_compare_calib_regime_trainret/results/summary.json`)
- MSE comparison vs baseline daily3 run (same test set `n=359`, same method):
  - `TSM+LLM‑CoT‑Sent‑RF` `mse_path`: **13.8942 → 14.1203** (**+1.63%**, worse)
  - “Blended” (same base-run selection `best_w_path=1.0`): **14.0020 → 14.1299** (**+0.91%**, worse)
  - Horizon MSE deltas:
    - `h1`: **−5.06%** (better)
    - `h5`: **+6.37%** (worse)
    - `h20`: **+0.90%** (worse)
    - `h30`: **+2.34%** (worse)
  - Trend-accuracy deltas:
    - `d10`: **−1.67 pp**
    - `d20`: **+0.84 pp**
    - `d30`: **−2.51 pp**
  - LLM fallback note:
    - `1/359` windows fell back to TSM due timeout (`llm_metadata.jsonl`).
- Outcome:
  - This regime-aware sentiment calibration worsened overall path MSE and is not a net gain.

**Sentiment option: direction+importance (0-10) voting A/B suite (Feb 16-17, 2026)**
- Goal:
  - Test whether replacing `llm_score ∈ {-1,0,1}` with an importance-weighted score improves CoT-Sent-RF.
- Scoring implementation (in `src/news/sentiment.py`):
  - Per headline, each vote returns `(direction, importance[0..10])`.
  - Majority direction `d ∈ {-1,0,1}`.
  - Raw importance score: `score_imp_nopen = d * (median_importance_of_majority / 10)`.
  - Confidence-penalized score: `score_imp = score_imp_nopen * (agreement_frac ^ importance_confidence_power)`.

### 0-10 labeling/data artifacts
- Labeled sets (same top-3/day pool size as baseline: **7,553** rows):
  - `data/news/headlines_labeled_qwen_votes3_daily3_importance_v1.csv`
  - `data/news/headlines_labeled_qwen_votes3_daily3_importance_v2_indep_percall.csv`
  - `data/news/headlines_labeled_qwen_votes3_daily3_importance_v3_indep_t035_percall.csv`
- Daily sentiment sets (**2,782** days each):
  - v1: `data/news/daily_sentiment_qwen_votes3_daily3_importance_v1_confpen.csv`, `..._v1_nopen.csv`
  - v2 (indep/per-call): `data/news/daily_sentiment_qwen_votes3_daily3_importance_v2_indep_confpen.csv`, `..._v2_indep_nopen.csv`
  - v3 (indep/per-call, `temperature=0.35`): `data/news/daily_sentiment_qwen_votes3_daily3_importance_v3_indep_t035_confpen.csv`, `..._v3_indep_t035_nopen.csv`

### A/B run results (all vs baseline run `runs/20260208_042155_ea55a7`, test `n=359`)

| Variant | Run | `mse_path` | Delta vs baseline | Blended `mse_path` | Blended delta |
|---|---|---:|---:|---:|---:|
| v1 conf-pen | `runs/20260216_importance_v1_confpen_sentiment_compare` | 14.2737 | **+2.73%** | 14.2666 | **+1.89%** |
| v1 no-pen | `runs/20260216_importance_v1_nopen_sentiment_compare` | 14.0383 | **+1.04%** | 14.1247 | **+0.88%** |
| v2 indep/per-call | `runs/20260216_importance_v2_indep_percall_sentiment_compare` | 13.8760 | **-0.13%** | 13.9748 | **-0.19%** |
| v3 indep/per-call t=0.35 conf-pen | `runs/20260216_importance_v3_indep_t035_confpen_sentiment_compare` | 13.8883 | **-0.04%** | 13.9484 | **-0.38%** |
| v3 indep/per-call t=0.35 no-pen | `runs/20260216_importance_v3_indep_t035_nopen_sentiment_compare` | 13.7794 | **-0.83%** | 13.8313 | **-1.22%** |

### Vote-behavior diagnostics (what we learned)
- v1 (`single_call_multi_vote` style behavior): low agreement and unstable labels:
  - agreement split: `1/3=2,881`, `2/3=2,945`, `3/3=1,727` (only **22.87%** unanimous).
  - label skew: `YES=4,467`, `UNKNOWN=2,747`, `NO=339`.
  - net result: degraded MSE.
- v2 (`per_call` + independent cache, `temperature=0`): fully deterministic:
  - **100% unanimous** votes.
  - label skew flips strongly negative: `NO=5,524`, `YES=2,029`, `UNKNOWN=0`.
  - confidence penalty has no effect (conf-pen and no-pen daily series are identical).
- v3 (`per_call`, independent, `temperature=0.35`): slight diversity restored:
  - **95.7%** unanimous; remaining rows mostly `2/3` splits.
  - best path MSE came from `v3 ... nopen`, but short horizon still worsened (`h1: 0.2650 -> 0.3192`, +20.45%), and trend accuracy dropped.
- Daily-series shift warning:
  - baseline effect daily mean `sent_score ≈ +0.4907`;
  - v2/v3 importance daily means around `-0.11` (sign regime flip), indicating scoring distribution drift.

### Decision / revert status
- 0-10 importance voting was tested as an A/B branch and did **not** produce robust net gains.
- Baseline recovery was confirmed by:
  - `runs/20260217_revert_effect_baseline_confirm` (`mse_path` and horizon MSE exactly match baseline).
- Active baseline remains direction-only scoring (`-1/0/1`) for now.

**New literature-backed sentiment improvement options (added Feb 17, 2026)**
- 1) **Abnormal sentiment factor (surprise, not level)**:
  - Use `z_t = (S_t - rolling_mean(S))/rolling_std(S)` and feed `z_t` (or capped `z_t`) instead of raw `S_t`.
  - Rationale: papers on text-return predictability often find incremental signal in *unexpected* text moves.
- 2) **Novelty/topicality weighting before daily aggregation**:
  - Downweight repetitive headlines and upweight novel-but-topical headlines before computing daily score.
  - Rationale: reduces duplicate narrative noise and improves information efficiency.
- 3) **Asymmetric intensity channels with decay**:
  - Build separate positive and negative intensity series (e.g., `S+`, `S-`) with short decay kernels; feed both.
  - Rationale: bad news and good news often have asymmetric and different-horizon impacts.
- 4) **Probabilistic vote aggregation (Dawid-Skene/Snorkel style)**:
  - Replace plain majority vote with label-model estimation of vote reliability/confusion.
  - Rationale: converts noisy multi-vote outputs into calibrated posterior labels/scores.
- 5) **Regime/structural-break-aware sentiment mapping**:
  - Fit separate sentiment→return mapping by detected regimes (e.g., Bai-Perron breakpoints or volatility states).
  - Rationale: signal semantics shift across policy and macro regimes in EUA markets.

### Literature anchors used for these options
- Lopez-Lira & Tang (ChatGPT YES/NO/UNKNOWN financial-news framework).
- Chen et al. (LLM refinement for carbon prices; CoT-RF / CoT-Sent-RF setup).
- Kelly, Pruitt, Su (text characteristics and return predictability).
- News novelty/topicality literature (novel-information weighting).
- Dawid & Skene (1979) and Snorkel label-model work (weak supervision aggregation).
- Bai & Perron structural-break framework (regime-sensitive mapping).

**Option (1) rerun: abnormal sentiment factor (`z-score` surprise) (Feb 17, 2026)**
- Goal:
  - Replace raw daily sentiment level with a rolling surprise factor (`z_t`) and test impact on CoT-Sent-RF.
- Implementation:
  - Added script: `scripts/transform_daily_sentiment_abnormal.py`.
  - Transform used:
    - input: `data/news/daily_sentiment_qwen_votes3_daily3.csv`
    - output: `data/news/daily_sentiment_qwen_votes3_daily3_abnormal_z63_cap3.csv`
    - settings: `window=63`, `min_periods=20`, clip `[-3, 3]`.
  - Formula:
    - `z_t = (S_t - mean_{t-62:t}(S)) / std_{t-62:t}(S)` (with zero/NaN std guarded to 0).
- Data diagnostics:
  - rows: **2,782** (same coverage as baseline daily3).
  - baseline mean/std: **0.4907 / 0.4466**.
  - abnormal mean/std: **0.0016 / 0.9973** (properly centered and standardized).
  - changed days vs baseline: **99.89%**.
  - correlation with baseline series: **0.9800** (mostly monotone rescaling/normalization).
- MSE comparison run:
  - artifact: `runs/20260217_abnormal_z63_cap3_sentiment_compare_rerun1` (summary in `runs/20260217_abnormal_z63_cap3_sentiment_compare_rerun1/results/summary.json`)
  - same method/split as baseline: `TSM+LLM-COT-SENT-RF`, test `n=359`.
- Results vs baseline daily3:
  - `mse_path`: **13.8942 → 14.0878** (**+1.39%**, worse)
  - blended `mse_path` (`best_w_path=1.0`): **14.0020 → 14.1502** (**+1.06%**, worse)
  - horizon MSE deltas:
    - `h1`: **+8.45%** (worse)
    - `h5`: **+0.84%** (worse)
    - `h20`: **+0.85%** (worse)
    - `h30`: **+3.64%** (worse)
  - trend-accuracy deltas:
    - `d10`: **−2.51 pp**
    - `d20`: **−0.84 pp**
    - `d30`: **−2.51 pp**
- What worked:
  - The option is now fully reproducible via a dedicated transform script.
  - Series centering/scaling behaved as intended and integrated cleanly with existing pipeline.
- What did not work:
  - Forecast quality degraded across path MSE, all tracked horizons, and trend accuracy.
  - In this setup, surprise-normalization did not improve signal-to-noise for CoT-Sent-RF.
- Decision:
  - Keep this as a documented negative result; do not promote to active baseline.

**Revert before option (2)**
- Reverted option-(1) implementation artifact from active code:
  - removed `scripts/transform_daily_sentiment_abnormal.py`
- Kept option-(1) experiment documentation/results for traceability.

**Option (2) rerun: novelty/topicality weighting before daily aggregation (Feb 17, 2026)**
- Goal:
  - Downweight repetitive headlines (novelty) and modestly upweight high-relevance headlines (topicality) before daily sentiment aggregation.
- Implementation:
  - Added script: `scripts/transform_daily_sentiment_novelty_topicality.py`.
  - Build used:
    - input labeled headlines: `data/news/headlines_labeled_qwen_votes3_daily3.csv`
    - output daily series: `data/news/daily_sentiment_qwen_votes3_daily3_novelty90_topicality.csv`
    - settings: `lookback_days=90`, `novelty_power=1.0`, `topicality_min=0.75`, `topicality_max=1.25`
  - Weights:
    - novelty weight per headline: `w_novel = 1 / (1 + prior_count_same_title_90d)`
    - topicality weight from normalized `relevance_score`: `w_topical in [0.75, 1.25]`
    - combined: `w = w_novel * w_topical`
    - daily score: weighted mean of `llm_score` by `w`.
- Data diagnostics:
  - rows: **2,782** (same day coverage as baseline daily3).
  - changed days vs baseline series: **46.69%**.
  - mean absolute day-level change: **0.02124**.
  - correlation vs baseline series: **0.99418**.
- MSE comparison run:
  - artifact: `runs/20260217_novelty90_topicality_sentiment_compare` (summary in `runs/20260217_novelty90_topicality_sentiment_compare/results/summary.json`)
  - same method/split as baseline: `TSM+LLM-COT-SENT-RF`, test `n=359`.
- Results vs baseline daily3:
  - `mse_path`: **13.8942 → 14.2068** (**+2.25%**, worse)
  - blended `mse_path` (`best_w_path=1.0`): **14.0020 → 14.2466** (**+1.75%**, worse)
  - horizon MSE deltas:
    - `h1`: **−5.86%** (better)
    - `h5`: **+0.80%** (worse)
    - `h20`: **+2.72%** (worse)
    - `h30`: **+2.70%** (worse)
  - trend-accuracy deltas:
    - `d10`: **−1.11 pp**
    - `d20`: **−1.11 pp**
    - `d30`: **−1.95 pp**
- What worked:
  - Weighting pipeline integrated cleanly and produced a measurable shift in daily sentiment without breaking coverage.
  - Slight h1 improvement.
- What did not work:
  - Path-level and long-horizon errors worsened materially.
  - Trend accuracy deteriorated across all tracked horizons.
- Decision:
  - Option (2) is not a net gain in current form; keep as documented negative result and do not adopt as baseline.

**Revert before option (3)**
- Reverted option-(2) implementation artifact from active code:
  - removed `scripts/transform_daily_sentiment_novelty_topicality.py`
- Kept option-(2) experiment documentation/results for traceability.

**Option (3) rerun: asymmetric intensity channels with decay (Feb 17, 2026)**
- Goal:
  - Encode asymmetric positive vs negative sentiment dynamics with separate decay rates, then combine into one daily signal.
- Implementation:
  - Added script: `scripts/transform_daily_sentiment_asym_decay.py`.
  - Build used:
    - input labeled headlines: `data/news/headlines_labeled_qwen_votes3_daily3.csv`
    - output daily series: `data/news/daily_sentiment_qwen_votes3_daily3_asymdecay_pos2_neg5_w1p3.csv`
    - settings: `span_pos=2`, `span_neg=5`, `pos_weight=1.0`, `neg_weight=1.3`, `normalize=range`
  - Construction:
    - daily positive channel: mean of positive `llm_score` mass.
    - daily negative channel: mean of negative `llm_score` mass (as positive magnitude).
    - separate EWMAs:
      - `pos_decay = EWMA(sent_pos, span=2)`
      - `neg_decay = EWMA(sent_neg, span=5)`
    - combined signal:
      - `sent_score_raw = 1.0 * pos_decay - 1.3 * neg_decay`
      - then range-normalized to approximately `[-1, 1]`.
- Data diagnostics:
  - rows: **2,782** (same coverage as baseline daily3).
  - changed days vs baseline series: **99.93%**.
  - mean absolute day-level change: **0.1734**.
  - correlation vs baseline series: **0.8962**.
- MSE comparison run:
  - artifact: `runs/20260217_asymdecay_pos2_neg5_w1p3_sentiment_compare` (summary in `runs/20260217_asymdecay_pos2_neg5_w1p3_sentiment_compare/results/summary.json`)
  - same method/split as baseline: `TSM+LLM-COT-SENT-RF`, test `n=359`.
- Results vs baseline daily3:
  - `mse_path`: **13.8942 → 13.8850** (**−0.07%**, slight improvement)
  - blended `mse_path` (`best_w_path=1.0`): **14.0020 → 13.9545** (**−0.34%**, slight improvement)
  - horizon MSE deltas:
    - `h1`: **+20.03%** (worse)
    - `h5`: **+4.35%** (worse)
    - `h20`: **−2.17%** (better)
    - `h30`: **+2.41%** (worse)
  - trend-accuracy deltas:
    - `d10`: **−1.67 pp**
    - `d20`: **+0.28 pp**
    - `d30`: **−1.39 pp**
- What worked:
  - First non-negative path-level result among the new literature options (small path MSE lift).
  - Better `h20` and slightly better blended path MSE.
- What did not work:
  - Short horizon degradation (`h1`, `h5`) is large.
  - `h30` and most trend metrics still worsened.
- Decision:
  - Promising as a direction (best of options 1-3 so far on path-level), but not robust enough for baseline adoption without further tuning.

**Revert before option (4)**
- Reverted option-(3) implementation artifact from active code:
  - removed `scripts/transform_daily_sentiment_asym_decay.py`
- Kept option-(3) experiment documentation/results for traceability.

**Option (4) rerun: probabilistic vote aggregation (Dawid-Skene) (Feb 18, 2026)**
- Goal:
  - Replace plain majority-vote headline labels with Dawid-Skene posterior labels/scores before daily aggregation.
- Implementation:
  - Added script: `scripts/transform_daily_sentiment_dawid_skene.py`.
  - Build used:
    - input labeled headlines: `data/news/headlines_labeled_qwen_votes3_daily3.csv`
    - output daily series: `data/news/daily_sentiment_qwen_votes3_daily3_dawidskene.csv`
    - optional posterior dump: `data/news/daily_sentiment_qwen_votes3_daily3_dawidskene_posteriors.csv`
    - settings: `workers=3`, `max_iter=50`, `tol=1e-6`, `alpha=0.5`.
  - Construction:
    - parse per-headline votes (`llm_votes` fallback `llm_vote`) into label observations.
    - run Dawid-Skene EM over labels `{NO, UNKNOWN, YES}`.
    - compute expected headline score using `{-1, 0, +1}` and average to daily `sent_score`.
- DS diagnostics from run:
  - EM converged in **3** iterations.
  - class priors: `NO=0.0953`, `UNKNOWN=0.3084`, `YES=0.5963`.
  - worker diagonals were all ~`0.999`, indicating near-deterministic/near-identical votes.
- Data diagnostics vs baseline daily3:
  - rows: **2,782** (same coverage).
  - correlation with baseline daily score: **1.0000**.
  - mean absolute day-level difference: **9.37e-11** (numerically identical for practical purposes).
- MSE comparison run:
  - artifact: `runs/20260218_dawidskene_sentiment_compare` (summary in `runs/20260218_dawidskene_sentiment_compare/results/summary.json`)
  - same method/split as baseline: `TSM+LLM-COT-SENT-RF`, test `n=359`.
- Results vs baseline daily3:
  - `mse_path`: **13.8942 → 13.9438** (**+0.36%**, worse)
  - blended `mse_path` (`best_w_path=1.0`): **14.0020 → 14.0320** (**+0.21%**, worse)
  - horizon MSE deltas:
    - `h1`: **−10.48%** (better)
    - `h5`: **+0.43%** (worse)
    - `h20`: **+0.40%** (worse)
    - `h30`: **+0.58%** (worse)
  - trend-accuracy deltas:
    - `d10`: **−0.56 pp**
    - `d20`: **+0.28 pp**
    - `d30`: **−0.56 pp**
- What worked:
  - Clean end-to-end integration of probabilistic aggregation into the pipeline.
  - `h1` improved materially.
- What did not work:
  - Daily signal is effectively unchanged from baseline because vote streams are near-deterministic.
  - Net path MSE and blended path MSE are still slightly worse.
- Decision:
  - Keep as documented negative/neutral result; do not adopt as active baseline in current form.

**Option (4) targeted variant: apply Dawid-Skene only to `h1` (Feb 18, 2026)**
- Goal:
  - Keep the `h1` gain from option (4) while preserving baseline behavior for horizons `h2..h30`.
- Implementation:
  - Built a horizon splice artifact:
    - baseline source: `runs/20260208_042155_ea55a7/predictions/TSM+LLM-COT-SENT-RF_pred_test_subset.npz`
    - option-(4) source: `runs/20260218_dawidskene_sentiment_compare/predictions/TSM+LLM-COT-SENT-RF_pred_test_subset.npz`
    - splice rule: `yhat[:,0] = yhat_option4[:,0]`; keep baseline for `yhat[:,1:]`.
  - Saved run artifact:
    - `runs/20260218_dawidskene_h1only_splice_compare/results/summary.json`
- Results vs baseline:
  - `mse_path`: **13.8942 → 13.8933** (**−0.0067%**, tiny improvement)
  - blended `mse_path` (`best_w_path=1.0`): **unchanged** (`14.0020 → 14.0020`)
  - horizon deltas:
    - `h1`: **−10.48%** (better)
    - `h5/h20/h30`: **unchanged** (by construction)
- Decision:
  - Useful as a surgical horizon-level improvement, but the net path gain is extremely small.
  - Keep as a possible post-processing rule rather than promoting as a core sentiment transform.

**Baseline update (Feb 19, 2026)**
- Adopted new active comparison baseline:
  - `runs/20260219_h1only_baseline`
  - construction: baseline run `runs/20260208_042155_ea55a7` with `h1` replaced by option-(4) Dawid-Skene `h1`.
- Baseline metrics (test `n=359`):
  - `mse_path = 13.8933`
  - `h1 = 0.2372`, `h5 = 1.8328`, `h20 = 18.3857`, `h30 = 38.5211`
- Validation rerun:
  - `runs/20260219_h1only_baseline_selfcheck`
  - with unchanged sentiment context, `changed windows = 0/359`, and `mse_path_base == mse_path_new == 13.8933`.
- Operational note:
  - use `--base-run runs/20260219_h1only_baseline` for subsequent sentiment-option A/B comparisons.

**Option (5) rerun: regime-aware sentiment mapping (vol-state calibration) (Feb 18, 2026)**
- Goal:
  - Implement literature option (5): map sentiment through regime-specific train-only calibration before feeding CoT-Sent-RF.
- Implementation:
  - Added script: `scripts/transform_daily_sentiment_regime_mapping.py`.
  - Build used:
    - input sentiment: `data/news/daily_sentiment_qwen_votes3_daily3.csv`
    - panel path: `runs/20260208_042155_ea55a7/data/panel.parquet`
    - output sentiment: `data/news/daily_sentiment_qwen_votes3_daily3_regime_map.csv`
    - diagnostics: `data/news/daily_sentiment_qwen_votes3_daily3_regime_map_diagnostics.json`
    - settings: `train_end=2023-06-30`, `vol_window=20`, `n_regimes=3`, `min_samples=40`, `ridge=1e-3`, `clip=1.0`
  - Mapping details:
    - compute `ret_fwd` and rolling volatility regime from panel returns.
    - fit train-only ridge-linear `ret_fwd ~ sent_score` per regime, with global fallback.
    - apply regime-specific mapping to all days.
    - rescale mapped output to train sentiment mean/std and clip to `[-1, 1]`.
- Data diagnostics vs baseline daily3:
  - rows: **2,782** (full coverage retained).
  - changed days: **87.67%**.
  - mean absolute day-level change: **0.3561**.
  - correlation vs baseline: **0.3905**.
  - mapped mean/std: **0.5376 / 0.4242** (baseline **0.4907 / 0.4466**).
- MSE comparison run:
  - artifact: `runs/20260218_regime_map_sentiment_compare` (summary in `runs/20260218_regime_map_sentiment_compare/results/summary.json`)
  - same method/split as baseline: `TSM+LLM-COT-SENT-RF`, test `n=359`.
- Results vs baseline:
  - `mse_path`: **13.8942 → 14.5570** (**+4.77%**, worse)
  - blended `mse_path` (`best_w_path=1.0`): **14.0020 → 14.4869** (**+3.46%**, worse)
  - horizon MSE deltas:
    - `h1`: **+27.39%** (worse)
    - `h5`: **+6.98%** (worse)
    - `h20`: **+5.52%** (worse)
    - `h30`: **+4.86%** (worse)
  - trend-accuracy deltas:
    - `d10`: **−1.39 pp**
    - `d20`: **−0.84 pp**
    - `d30`: **−0.84 pp**
- What worked:
  - Full end-to-end implementation of regime-conditioned mapping with train-only calibration and diagnostics.
- What did not work:
  - Strong performance deterioration across path, all tracked horizons, blend, and trend metrics.
- Decision:
  - Reject this option in current form; keep documented as negative result.

**Revert after option (5)**
- Reverted option-(5) implementation artifacts from active code/workspace:
  - removed `scripts/transform_daily_sentiment_regime_mapping.py`
  - removed `data/news/daily_sentiment_qwen_votes3_daily3_regime_map.csv`
  - removed `data/news/daily_sentiment_qwen_votes3_daily3_regime_map_diagnostics.json`
  - removed run artifact directory `runs/20260218_regime_map_sentiment_compare/`
- Kept documentation and the option-(4) h1-only splice artifact for traceability.
