# UK ETS Production Live LLM System Blueprint

## Objective

Build a production-ready UK ETS forecasting system that:

- updates daily using all information available up to the forecast origin date,
- uses the LLM only when it is likely to improve the base model,
- preserves the real long-horizon LLM edge we observed offline,
- avoids leakage,
- and remains operationally simple enough to run every day.

The system should optimize for **live decision quality**, not for maximum prompt complexity.

## Core Thesis

The production product should **not** be:

- "LLM on every forecast"
- "one shared online memory bank for every base model"
- "generic prompt optimization"

The production product **should** be:

- a strong traditional base forecaster,
- plus a selective LLM residual corrector,
- plus a learned live decision policy,
- plus tightly curated horizon-specific memory.

The system is successful if it:

- improves the base forecast when the LLM is used,
- abstains when the LLM is not likely to help,
- and updates causally as new outcomes mature.

## Current Evidence

### What is already proven

1. The LLM can improve a traditional base forecaster in fixed-holdout settings.
   - Best historical `35B` `TSM+LLM`: about `+5.46%` path-MSE improvement.
   - Best historical `4B` numeric-tool path: about `+4.35%` path-MSE improvement.

2. The live-online architecture is much harsher than the frozen setup.
   - The offline edge mostly collapses when we move to causal online memory.

3. In the current live-online system, only `TSM` still shows a net positive lift.
   - Best current live-online single policy: `TSM + regime_specific`
   - Gain vs base: about `+0.665%`

4. The useful LLM signal is still mostly long-horizon.
   - `h20` and `h30` improve.
   - `h1` is unchanged.
   - `h5` is often slightly worse.

### What is not yet solved

1. The live apply policy is too weak.
2. The online memory admission rule is still too generic.
3. The same online policy does not transfer across base models.
4. Combining individually helpful policies tends to degrade performance.

## Production System To Build

### 1. Forecasting stack

Use a **multi-model stack**, but only one model should be the primary LLM target.

- Primary base model: `TSM`
- Shadow benchmark model: `linear_ridge`
- Optional monitoring models: `linear_lasso`, `naive_persistence`, `seasonal_naive`

Production rule:

- The LLM refiner is only active on `TSM` initially.
- Other bases remain benchmarks until they can beat their raw base consistently in live-online evaluation.

This preserves edge by refusing to deploy LLM refinement where it still hurts.

### 2. LLM role

The LLM should be a **selective long-horizon residual corrector**.

It is not responsible for:

- full-path reforecasting,
- short-horizon micro-adjustment,
- or unconditional daily intervention.

The LLM should focus on:

- `h20` and `h30`,
- regime interpretation,
- medium/long-horizon directional and magnitude corrections,
- and abstention when evidence is weak.

### 3. Daily causal update loop

For each trading day `t`:

1. Ingest all market and exogenous data available by end-of-day `t`.
2. Recompute point-in-time features.
3. Refit or refresh the base model on all history up to `t`.
4. Generate the next forecast from the base model.
5. Compute live gate features for the LLM decision.
6. If the apply gate says `no`, ship the base forecast.
7. If the apply gate says `yes`, call the LLM with:
   - numeric analysis tool,
   - deterministic verifier,
   - curated positive/negative memory cases,
   - horizon-specific context.
8. Store the final forecast packet.
9. When older forecasts fully mature, score whether the LLM helped and update memory/gate datasets.

This gives a truly daily-updating production system without requiring a standing validation split in live inference.

## Components

### A. Apply gate

This is the most important missing component.

It should be a small learned model that predicts:

- `P(LLM helps vs base at h20/h30 | information available at time t)`

Inputs should include:

- base forecast levels and slopes,
- base forecast uncertainty proxies,
- recent realized volatility,
- auction / non-auction state,
- energy / weather regime tags,
- positive-memory similarity,
- negative-memory similarity,
- memory support counts,
- numeric tool outputs,
- verifier outputs from a hypothetical dry-run or pre-verification features.

Recommended models:

- primary: logistic regression or gradient-boosted trees
- fallback: monotonic scorecard / threshold rule

Why:

- deployable,
- interpretable,
- easy to retrain daily or weekly,
- low latency.

### B. Memory admission policy

A realized case should not be appended blindly.

Each matured case should be labeled ex post as:

- `positive`
- `negative`
- `discard`

using a horizon-weighted helpfulness score.

Recommended helpfulness score:

- reward path improvement,
- reward `h20` and `h30` improvement more than `h5`,
- penalize any large `h5` damage,
- penalize sign mistakes,
- penalize unstable oversized corrections.

Admission rule:

- only `positive` cases enter the positive bank,
- only clearly harmful cases enter the negative bank,
- neutral/noisy cases are dropped.

### C. Memory structure

The memory bank should be split, not shared.

Use:

- positive `h20` bank
- positive `h30` bank
- negative `h20` bank
- negative `h30` bank

Each case also gets regime tags:

- volatility bucket
- trend bucket
- auction day flag
- weather / heating-demand regime
- energy-stress regime

Retrieval then becomes:

- retrieve from matching regime buckets first,
- prefer high-utility cases,
- keep memory size small,
- show both supporting and cautionary examples.

### D. Horizon-specific maturity

Do not wait for `t+30` to learn everything.

Use horizon-specific promotion:

- `h20` memory eligibility at `t+20`
- `h30` memory eligibility at `t+30`

This allows the live system to adapt faster without leakage.

### E. Deterministic post-LLM controls

Keep:

- numeric analysis tool
- deterministic verifier

These should remain in production because they are low-variance controls.

Do not add:

- more free-form prompting complexity
- open-ended retrieval
- generic live web search

## Architecture Decision

### System to ship first

Ship only:

- `TSM`
- `4B` non-reasoning endpoint
- numeric analysis tool
- deterministic verifier
- learned apply gate
- regime-specific memory
- horizon-specific memory banks
- positive/negative memory separation

Why:

- This is the only branch with demonstrated live-online positive lift.
- It is fast enough to run daily.
- It is simpler than the 35B reasoning path.

### Role of 35B

Use `35B` as an offline teacher, not the default live model.

Suggested use:

- overnight teacher labeling,
- high-value case analysis,
- periodic policy distillation,
- creation of stronger positive/negative memory summaries.

This preserves the benefit of the larger model without making the daily live path too slow.

## Experimental Program

### Phase 1: Fix live apply policy

#### Experiment group 1: learned gates

Test:

1. Logistic gate on base/regime/memory features
2. Gradient-boosted tree gate
3. Two-stage gate:
   - stage 1: should we call the LLM?
   - stage 2: should we accept the LLM correction?

Primary benchmark:

- current `TSM + regime_specific`

Expected gain:

- improve live-online `TSM` from `+0.665%` to roughly `+1.25% to +2.00%`

Rationale:

- current gating is heuristic and underpowered
- learned gating should reduce false positive LLM usage

### Phase 2: Fix online memory

#### Experiment group 2: horizon-specific positive/negative memory

Test:

1. `h20` bank only
2. `h30` bank only
3. split `h20/h30` banks
4. split banks plus negative examples in prompt
5. split banks plus regime filtering

Expected gain:

- additional `+0.25% to +1.00%` on live `TSM`

Rationale:

- current shared memory bank mixes incompatible cases

### Phase 3: Improve admission quality

#### Experiment group 3: memory promotion rules

Test:

1. strict helpfulness margin
2. helpfulness margin + `h5` damage cap
3. support-weighted admission
4. probation bank before full promotion

Expected gain:

- mainly robustness rather than large headline lift
- reduce collapse in bad regimes

### Phase 4: 35B teacher distillation

#### Experiment group 4: teacher-assisted live policy

Use archived or freshly generated `35B` corrections to train:

1. apply gate labels
2. memory summaries
3. delta calibrators

Expected gain:

- another `+0.25% to +1.00%` if teacher signal is consistent

Rationale:

- 35B historically showed a stronger offline edge
- production can borrow that intelligence without paying 35B latency every day

## Scientific Evaluation Protocol

All future experiments must be judged under:

- causal online memory only,
- rolling-origin folds,
- point-in-time features only,
- horizon-specific memory maturity,
- frozen experimental definitions before evaluation,
- daily MTM portfolio backtests,
- post-`2025-01-01` contamination-aware reporting.

Primary score:

- path MSE

Secondary scores:

- `h20` MSE
- `h30` MSE
- daily MTM Sharpe
- turnover
- apply rate
- regret vs raw base model

## Success Criteria

### Minimum viable production release

For `TSM + live LLM`:

- live-online path MSE improvement > `+1.0%`
- no degradation on `h20`
- positive `h30` improvement
- apply rate below `35%`
- no evidence of strategy instability across rolling folds

### Strong release target

- live-online path MSE improvement `+1.5% to +2.5%`
- improved Sharpe or at least no economic degradation
- consistent positive performance in the post-cutoff regime

## What Not To Do

- Do not keep broadening prompt complexity.
- Do not deploy the LLM on every base model just because it can be wired in.
- Do not treat the combined policy as automatically better than the best single policy.
- Do not keep one undifferentiated online memory bank.
- Do not judge the product by single-regime offline gains.

## Recommended Immediate Build Order

1. Implement learned apply gate for `TSM`
2. Implement split positive/negative `h20/h30` memory banks
3. Add horizon-specific memory promotion at `t+20` and `t+30`
4. Retrain under rolling online folds
5. Promote only if live-online `TSM` beats the current `regime_specific` benchmark

## Bottom Line

The product we should build is **not** “an LLM that refines every model every day.”

It is:

- a `TSM`-centered live forecasting system,
- with a selective LLM long-horizon correction layer,
- driven by a learned apply policy,
- updated daily with causally matured positive/negative memory,
- and supervised scientifically with rolling online evaluation.

That is the most credible path to a production system that preserves the LLM edge.
