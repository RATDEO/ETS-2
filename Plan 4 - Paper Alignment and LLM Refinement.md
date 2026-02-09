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
