# 5 Strategies to Recover CoT/SENT Signal Performance

## Background

After fixing three concrete data leakage paths in the main pipeline (Section 20 of the runbook), the LLM-enhanced variants lost their edge over the base Autoformer TSM. The clean post-leakage results are:

| Model | Path MSE | h1 | h5 | h20 | h30 |
|---|---|---|---|---|---|
| **TSM (retuned clean)** | **20.990** | 1.388 | 6.432 | 26.411 | 41.795 |
| CoT-RF raw | 23.442 | 1.514 | 6.631 | 29.402 | 49.218 |
| CoT-RF blend (val-tuned, w=0.0) | 20.990 | — | — | — | — |
| **CoT-SENT-RF raw (paper-style)** | **20.604** | 1.434 | 6.455 | 25.793 | 41.088 |
| CoT-SENT-RF blend ramp w=0.50 | 20.739 | 1.390 | 6.410 | 26.040 | 41.390 |

The paper-style `TSM+LLM-COT-SENT-RF` recovers a real but modest 1.6% improvement on held-out test. The CoT-RF variant without sentiment adds nothing (val-selected blend weight collapses to 0.0). Additional signal tuning (`qwen_votes3_daily3`, `feat4_train_linear_proxy`) did not beat the baseline paper signal on the exact held-out test.

## Root Causes

The LLM-enhanced variants underperform because:

1. **The 4B Qwen model produces vague qualitative rules** that translate to noise in price space (e.g., "reduce forecast by 2-5%", "apply downward bias of 3-8%")
2. **The reflect→apply pipeline gives the LLM the entire 30-step forecast to rewrite**, which invites correlated drift across all horizons
3. **Sentiment is injected as a flat numeric array in the prompt** with no event semantics — the LLM cannot learn conditional relationships
4. **Teaching examples cluster in similar regimes** (similarity-based selection on mean/std/slope) and lack error-diversity
5. **There is no learned correction layer** to backstop the LLM's imprecision — the fixed ramp-blend grid is too coarse

---

## Strategy 1: Upgrade to a Larger LLM (or Two-Tier Model Setup)

### Problem

The current model is **Qwen3-VL-4B** running locally at `http://192.168.1.140:9877/v1`. The reflection rules it produces are generic platitudes that lack the quantitative precision needed to beat the TSM's already-reasonable output. The 4B parameter count is the single biggest bottleneck.

### Approach

- **Primary:** Switch to a materially larger model — Qwen-32B, Qwen-72B, or a frontier API model (GPT-4o, Claude) — for at least the reflection stage (Stage A). The apply stage (Stage B) could remain on the 4B model since it just executes a structured JSON fill.
- **Two-tier variant:** Run reflection on a large model, cache the rules, then run apply on the cheap/fast 4B model. This keeps cost/latency controlled while getting much higher quality reasoning in the rules.

### Implementation

The `LLMRefiner` in `src/llm/refine.py` already supports `provider`/`model`/`base_url` per config. The changes needed are:

1. Add a `reflect_model` override to the `cot_rf` config block so Stage A and Stage B can target different endpoints
2. Add a `model_override` parameter to `_call_llm` and `_call_llm_messages` methods
3. Wire the two-tier routing in the `refine()` method's two-stage reflect→apply block

### Validation

Re-run `scripts/tune_paper_cot_sent_validation.py` against the same frozen checkpoint from `runs/20260228_145058_40d91f` with the new model, comparing path MSE on the same 226-sample validation split. This is a zero-risk experiment — it touches no training, only the LLM inference stage.

### Expected Impact

**High.** The quality of reflection rules directly determines whether the apply stage produces useful corrections or noise. A 32B+ model should produce rules with concrete quantitative anchoring rather than vague percentage ranges.

### Cost

LLM inference only — no TSM retraining required.

---

## Strategy 2: Replace Free-Text Rewrite with Structured Per-Horizon Delta Corrections

### Problem

The current pipeline asks the LLM to output `{"yhat": [30 prices]}` — a full 30-step price path. This means the LLM can (and does) introduce correlated drift across all horizons. The model has 30 degrees of freedom when it really needs 4.

### Approach

Instead of asking the LLM to rewrite the full path, constrain it to output **bounded percentage adjustments** at each of the 4 key horizons only (`h1, h5, h20, h30`), then interpolate the adjustments across the full 30-step path.

### Prompt Change

The apply prompt in `CoTSentRFApplyTemplate` (`src/llm/prompts.py`) would change from:

```
Provide your refined 30-day forecast as JSON:
{"yhat": [day1_price, day2_price, ..., day30_price]}
```

To:

```
Provide percentage adjustments at 4 key horizons (clamped to ±3%):
{"adjustments": {"h1": +0.3, "h5": -0.5, "h20": +0.8, "h30": -0.2}}

Where each value is the % change to apply to the model's forecast at that horizon.
Intermediate steps will be interpolated automatically.
```

### Why This Helps

- Reduces the LLM's degrees of freedom from 30 to 4
- Eliminates correlated path drift
- Makes the output interpretable and auditable
- The 4B model is far more likely to get 4 directional nudges right than 30 precise price values
- Hard clamping prevents catastrophic over-corrections

### Implementation

1. New `CoTSentRFApplyDeltaHorizonTemplate` in `src/llm/prompts.py`
2. New `parse_horizon_deltas()` parser in `src/llm/refine.py`
3. Interpolation + clamp logic in the `refine()` method
4. New method name `TSM+LLM-COT-SENT-RF-HDELTA` to distinguish from the existing full-delta variant

### Expected Impact

**High.** This is the most architecturally sound fix. Even a weak model can say "undershoot at h20 by 0.5%" more reliably than rewriting 30 price values.

### Cost

LLM inference only — no TSM retraining required. New prompt template + parser code.

---

## Strategy 3: Inject Sentiment as a Statistical Feature Inside the TSM

### Problem

Currently, sentiment only appears inside the LLM's text prompt as a flat numeric array. The Autoformer never sees it. This means the TSM baseline has no access to news signal at all — the entire burden of interpreting sentiment falls on the 4B LLM.

### Approach

Add the daily sentiment series (`sent_score` from `daily_sentiment.csv` and/or `qwen_votes3_daily3`) as an input feature column to the Autoformer panel in `src/data/panel_builder.py`. The TSM then learns the sentiment→price relationship statistically during training.

### Two-Channel Design

1. `TSM-SENT` = Autoformer trained with sentiment as an input feature → produces a sentiment-aware base forecast
2. `TSM-SENT+LLM-COT-SENT-RF` = the LLM refiner on top of this stronger base

### Why This Helps

The current TSM uses 8-11 features (Brent, coal, KEUA, KRBN, GRN, auction, ICAP, VSTOXX) but no news signal. Even a noisy sentiment column gives the model a covariate it currently lacks. Critically, this lets the model learn the *conditional effect* of sentiment (e.g., "positive sentiment after a drawdown" vs. "positive sentiment at a peak"), which the LLM prompt cannot express.

The comparison then shifts from:

- **Old:** "Can a 4B LLM interpret flat sentiment numbers in text?" (no)
- **New:** "Can an Autoformer learn a statistical relationship with a sentiment feature?" (much more likely)

### Implementation

1. **Feature engineering** in `src/data/panel_builder.py`: merge `daily_sentiment.csv` onto the panel, forward-fill, and expose 2-3 derived columns:
   - `sent_score` (raw daily sentiment)
   - `sent_score_3d_ma` (3-day moving average for smoothing)
   - `sent_volume` (daily news volume as a secondary signal)
2. Bump `enc_in`/`dec_in` in config to account for the new columns
3. Retune TSM hyperparameters on validation only via `scripts/tune_clean_tsm_cot_rf.py`
4. Re-run CoT-SENT-RF on top of the new `TSM-SENT` base

### Expected Impact

**Medium-High.** The Autoformer is a well-regularised statistical learner. If there is any predictive signal in daily sentiment, it will find it more reliably than a 4B LLM parsing a text array.

### Cost

Requires a TSM retrain (hyperparameter search on validation), but this is fast (~minutes per candidate on current hardware).

---

## Strategy 4: Diversify Teaching-Example Selection with Error-Type Stratification

### Problem

The current similarity-based selection (Euclidean distance on `[mean, std, slope]` over the last 18 days, lookback 365) results in K=5 examples that cluster in the same market regime. The LLM sees five nearly-identical error patterns and derives rules that only apply to that regime.

### Approach

Replace the similarity selector with an **error-stratified** selector that ensures the 5 examples represent diverse forecast-error modes:

1. Compute the TSM forecast error profile for each candidate window (signed error at h1, h5, h20, h30)
2. Cluster the candidate pool into K error archetypes (e.g., "overshoot short-term", "undershoot long-term", "lag error", "direction wrong", "near-perfect")
3. Pick one example per cluster, preferring the one closest in time to the current window

### Complementary Idea

Always include one "near-perfect" example so the LLM can see what a good forecast looks like, not just failures. The existing `high_error` selector was tested but only picks the worst cases, teaching the LLM that the model is always terrible — which produces over-corrections.

### Why This Helps

The reflection rules should cover multiple failure modes, not just one. With error-stratified examples, the LLM is more likely to produce rules like:

- "When the model overshoots at h5, reduce by X"
- "When the model undershoots at h20-h30, increase by Y"
- "When the model is close at h1, don't touch it"

Instead of the current generic rules it produces from seeing 5 similar examples.

### Implementation

1. New `error_stratified` example selection mode in the teaching-example builder section of `src/run_experiment.py`
2. K-means or quantile-based clustering on the 4D error vector `[err_h1, err_h5, err_h20, err_h30]`
3. One-per-cluster selection, with time-proximity tiebreaking within each cluster
4. Add `example_selection: error_stratified` as a config option in the `cot_rf` section

### Expected Impact

**Medium.** This improves the information content of the reflection stage without requiring a larger model or new architecture. The impact is bounded by the LLM's ability to synthesize diverse patterns.

### Cost

LLM inference only — no TSM retraining required. New example selection logic.

---

## Strategy 5: Add a Lightweight Learned Post-Hoc Calibration Layer

### Problem

Even with a better LLM and better prompts, the LLM output will still be noisy. The current ramp-blend grid searches over only 5 weights `[0.0, 0.25, 0.5, 0.75, 1.0]` applied uniformly. A per-horizon learned layer could discover that h1 should always be pure TSM while h20-h30 can accept modest LLM contribution.

### Approach

After generating both `y_tsm` and `y_llm` on the validation set, fit a per-horizon Ridge regression:

$$\hat{y}_h = \alpha_h \cdot y_{tsm,h} + \beta_h \cdot y_{llm,h} + \gamma_h$$

where $\alpha_h, \beta_h, \gamma_h$ are learned on validation for each horizon $h \in \{1,...,30\}$, with regularization to prevent overfitting on ~226 samples.

### Extension

Add 2-3 conditioning features to the meta-learner:

- Recent volatility (20-day rolling std)
- Recent sentiment score (3-day average)
- TSM-LLM forecast divergence (absolute difference)

This makes the blend weight **state-dependent** — e.g., trust LLM more in high-sentiment-divergence regimes, less when TSM and LLM already agree.

### Why This Helps

The findings already show that:

- h1 should always be pure TSM (the `h1base` override was needed)
- Longer horizons sometimes benefit from LLM contribution
- The benefit is regime-dependent

A learned per-horizon calibration layer can encode all of this automatically, replacing the brittle manual ramp schedule.

### Implementation

1. New `src/eval/meta_blend.py` module with a `MetaBlender` class
2. **Training:** After the main run completes, load `tsm` and `TSM+LLM-COT-SENT-RF` validation predictions, fit per-horizon Ridge/ElasticNet
3. **Test application:** Apply the fitted `MetaBlender` to test predictions
4. **Integration** into `scripts/tune_paper_cot_sent_validation.py` as an alternative to the fixed ramp grid
5. Cross-validation within the validation set to select regularization strength

### Expected Impact

**Medium.** This is a clean-up layer that maximises the value of whatever improvement comes from Strategies 1-4. On its own it cannot fix fundamentally noisy LLM outputs, but it can ensure that whatever signal exists is optimally extracted.

### Cost

Requires validation predictions from both TSM and LLM — no retraining of the TSM itself. Fitting is near-instant (Ridge regression on 226 samples × 30 horizons).

---

## Execution Priority

| Order | Strategy | Cost | Expected Impact | Dependencies |
|---|---|---|---|---|
| 1 | Strategy 1: Larger LLM | LLM inference only | High | None |
| 2 | Strategy 2: Horizon delta corrections | LLM inference only | High | None |
| 3 | Strategy 4: Error-stratified examples | LLM inference only | Medium | None |
| 4 | Strategy 3: Sentiment as TSM feature | TSM retrain | Medium-High | None |
| 5 | Strategy 5: Learned calibration layer | Post-hoc fitting | Medium | Requires predictions from 1-4 |

### Recommended Bundles

**Bundle A (cheap, fast, no retraining):** Strategies 1 + 2 + 4

- Better model + structured deltas + diverse examples
- Re-run LLM inference only against the frozen TSM checkpoint
- Estimated wall time: hours (dominated by LLM calls)

**Bundle B (independent, parallel):** Strategy 3

- Retrain TSM with sentiment features
- Can run in parallel with Bundle A on the same machine
- Estimated wall time: ~1 hour for hyperparameter search

**Bundle C (post-hoc, after A and B):** Strategy 5

- Fit the learned calibration layer on top of the best outputs from A and B
- Estimated wall time: minutes

### Validation Protocol

All strategies should be evaluated on the same frozen test split (`n=382`, post-leakage) against the same baseline:

- **Baseline checkpoint:** `runs/20260228_145058_40d91f`
- **Baseline TSM path MSE:** `20.942` (same-run) / `20.990` (retuned clean)
- **Current best CoT-SENT-RF:** `20.604` (raw paper-style, baseline signal)
- **Target:** Path MSE below `20.0` with consistent improvement across horizons

Results should be reported using the same `four_option_comparison_with_cot_ramp.csv` format for direct comparison with all prior runs documented in the runbook.
