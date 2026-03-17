# UK ETS Online-Memory Multi-Base LLM Sweep: Detailed Analysis

Generated: 2026-03-15

## Executive Summary

This tranche implemented the first production-style `online_realized_memory` admission and live-gating policy that:

- admits only fully realized prior cases,
- scores them by ex-post LLM helpfulness versus the base model,
- separates positive and negative memories,
- and uses a deterministic live gate to decide whether to apply the LLM at a new forecast date.

The policy was then tested on the full 144-window UK holdout across all current baseline forecasters on both the 4B and 35B non-thinking endpoints.

The core result is clear:

- the new online-memory policy **does improve every tested base model** on this holdout,
- but the gains are **small**, generally well below the earlier frozen single-regime LLM wins,
- and the improvements are still concentrated at `h20` and `h30`, with `h5` often slightly worse and `h1` effectively unchanged.

So this work **does** move the project in the right production direction. It shows that a causal online-memory LLM layer can create a positive refinement effect across multiple traditional forecasting bases. But it also shows that the current online design is still only a **modest residual corrector**, not yet a large live edge.

## Objectives And Expectations

The pre-registered plan for this tranche is in [plan.md](/Users/davidwilkinson/Desktop/ETS%202/reports/uk_ets_online_memory_multi_base/20260315_175555/plan.md) and [planned_candidates.csv](/Users/davidwilkinson/Desktop/ETS%202/reports/uk_ets_online_memory_multi_base/20260315_175555/planned_candidates.csv).

Main objectives:

1. convert offline helpfulness into a deployable online memory admission policy;
2. test whether that policy can create real gains in a production-style online benchmark;
3. create tailored LLM refinement pathways for all major base forecasters, not only `TSM`;
4. compare the 4B and 35B endpoints under the same online-memory design.

The expectations were directionally reasonable but too optimistic in magnitude. None of the candidates reached the forecast MSE ranges that were pre-registered for the stronger models.

## Experimental Setup

All runs used:

- the full 144-window UK holdout,
- `test_pool_mode=online_realized_memory`,
- a causal realized-memory horizon of `30` days,
- positive and negative memory banks,
- and a deterministic gate based on memory similarity and helpfulness consensus.

Important methodological points:

- No future leakage was introduced into the memory bank.
- A test case only became memory-eligible once its full future path was realized.
- The LLM therefore only had access to `train + val + previously realized test cases`.
- This is closer to a production architecture than the earlier frozen-memory benchmark.

## Raw Results

The full raw table is in [candidate_results.csv](/Users/davidwilkinson/Desktop/ETS%202/reports/uk_ets_online_memory_multi_base/20260315_175555/candidate_results.csv).

### 4B Results

| Base | Best 4B pathway | Base path MSE | Best LLM path MSE | Relative gain |
|---|---|---:|---:|---:|
| `tsm` | `4b_tsm_verifier_conservative` | `26.850794` | `26.709540` | `+0.526%` |
| `linear_ridge` | `4b_linear_ridge_numeric_verifier` | `22.486069` | `22.423316` | `+0.279%` |
| `linear_lasso` | `4b_linear_lasso_numeric_verifier` | `24.239772` | `24.131677` | `+0.446%` |
| `naive_persistence` | `4b_naive_persistence_relaxed` | `32.051094` | `31.897689` | `+0.479%` |
| `seasonal_naive` | `4b_seasonal_naive_relaxed` | `34.237653` | `34.084646` | `+0.447%` |

### 35B Results

| Base | Best 35B pathway | Base path MSE | Best LLM path MSE | Relative gain |
|---|---|---:|---:|---:|
| `tsm` | `35b_tsm_numeric_only_online` | `26.850794` | `26.650132` | `+0.747%` |
| `linear_ridge` | `35b_linear_ridge_numeric_only_online` | `22.486069` | `22.402214` | `+0.373%` |
| `linear_lasso` | `35b_linear_lasso_numeric_only_online` | `24.239772` | `24.153255` | `+0.357%` |
| `naive_persistence` | `35b_naive_persistence_relaxed_online` | `32.051094` | `31.694960` | `+1.111%` |
| `seasonal_naive` | `35b_seasonal_naive_relaxed_online` | `34.237653` | `33.950294` | `+0.839%` |

## What Actually Worked

### 1. The online-memory policy generalized across every tested base

This is the most important positive result.

Every tested base model improved:

- `TSM`
- `linear_ridge`
- `linear_lasso`
- `naive_persistence`
- `seasonal_naive`

So the current project can now make the claim that, on the main UK holdout, the LLM refinement layer is not just a `TSM`-only phenomenon. It can add a positive refinement effect on top of multiple traditional forecasters.

### 2. The live gate is doing real work

Across essentially every candidate, the gate only allowed about `33-34` LLM applications out of `144` possible windows.

That means:

- only about `23-24%` of windows actually used the LLM,
- the remaining `~110` windows stayed on the base forecast,
- and the LLM gains come from **selective use**, not from applying it everywhere.

That is exactly the right qualitative behavior for a production-grade online refinement layer.

### 3. The gains are still mostly long-horizon

Across the whole sweep:

- `h1` was effectively unchanged,
- `h20` and `h30` usually improved,
- `h5` was often slightly worse.

Examples:

- `4b_tsm_verifier_conservative`
  - `h20`: `+0.596%`
  - `h30`: `+0.666%`
  - `h5`: `-0.426%`
- `35b_tsm_numeric_only_online`
  - `h20`: `+0.821%`
  - `h30`: `+0.947%`
  - `h5`: `-0.561%`
- `35b_linear_ridge_numeric_only_online`
  - `h20`: `+0.247%`
  - `h30`: `+0.739%`
  - `h5`: `-0.558%`

So the LLM is still behaving like a long-horizon residual corrector, not a full-path improver.

## What Did Not Work

### 1. The realized gains were much smaller than expected

This is the main negative result.

The pre-registered expectations were too optimistic. In particular:

- the stronger models (`tsm`, `linear_ridge`, `linear_lasso`) did not come close to the hoped-for `1-5%` range under the online-memory architecture,
- the gains landed mostly in the `0.26-0.75%` range for strong models,
- and only the weak baselines crossed `0.8-1.1%`, but on much worse absolute forecast levels.

So the online-memory policy is useful, but it is not yet producing the magnitude of live improvement we wanted.

### 2. The weak models improved relatively more, but remain poor in absolute terms

The biggest relative lift was:

- `35b_naive_persistence_relaxed_online`: `+1.111%`

But the absolute path MSE is still `31.694960`, far worse than:

- `35b_linear_ridge_numeric_only_online`: `22.402214`
- `35b_tsm_numeric_only_online`: `26.650132`

That means the LLM can help weak bases, but this does not make them competitive with the stronger traditional models.

### 3. The production-style online architecture is much harsher than the frozen regime-local setup

Compared with the earlier frozen / single-regime-style wins:

- the online-memory gains are dramatically smaller,
- they are more consistent across bases,
- and they are much more believable as a live architecture.

That is scientifically good, but commercially sobering.

## Cross-Model Takeaways

### TSM

`TSM` still benefits the most among the strong models, especially on 35B.

Best result:

- `35b_tsm_numeric_only_online`: `26.850794 -> 26.650132` (`+0.747%`)

Interpretation:

- `TSM` still seems to leave the most useful long-horizon residual structure for the LLM to exploit.
- The 35B endpoint helps more than 4B here.
- Under online memory, the gain is real but much smaller than the earlier frozen wins.

### Linear Ridge

`linear_ridge` remains the strongest absolute base model in this tranche.

Best result:

- `35b_linear_ridge_numeric_only_online`: `22.486069 -> 22.402214` (`+0.373%`)

Interpretation:

- Ridge is already strong and stable, so there is less residual error left for the LLM.
- The LLM does still help slightly, almost entirely at longer horizons.
- The absolute best model across this sweep is still the ridge family, not TSM.

### Linear Lasso

`linear_lasso` improves a little more than ridge on 4B, but not on 35B.

Best results:

- 4B: `24.239772 -> 24.131677` (`+0.446%`)
- 35B: `24.239772 -> 24.153255` (`+0.357%`)

Interpretation:

- Lasso is somewhat more improvable than ridge under 4B verifier gating.
- But the total effect is still small.

### Naive And Seasonal Naive

These showed the biggest relative gains on 35B, but are still not strong bases.

Best results:

- `naive_persistence`: `+1.111%`
- `seasonal_naive`: `+0.839%`

Interpretation:

- The LLM can patch them selectively.
- That is interesting scientifically.
- But it does not make them competitive with ridge or even TSM.

## Best Pathways To Keep

### Keep as active 4B pathways

- `tsm`: `4b_tsm_verifier_conservative`
- `linear_ridge`: `4b_linear_ridge_numeric_verifier`
- `linear_lasso`: `4b_linear_lasso_numeric_verifier`

These are the strongest 4B versions for the three bases that matter most.

### Keep as active 35B pathways

- `tsm`: `35b_tsm_numeric_only_online`
- `linear_ridge`: `35b_linear_ridge_numeric_only_online`
- `linear_lasso`: `35b_linear_lasso_numeric_only_online`

### Keep only as research controls

- `naive_persistence`
- `seasonal_naive`

They improve, but not enough in absolute terms to matter for deployment.

## Main Conclusion For The Project

This tranche succeeded in one important strategic sense:

- it converted the offline helpfulness idea into a **causal online-memory architecture**,
- and showed that this architecture can create a positive LLM refinement effect across all current traditional base forecasters.

But it also clarified the harder truth:

- once the architecture is made production-like,
- the gains are much smaller than the old frozen single-regime numbers,
- and the LLM currently behaves as a **small, selective long-horizon residual corrector**, not a transformative live overlay.

That is still a meaningful result for the thesis:

- the LLM layer is real,
- it is not just a TSM artifact,
- it generalizes across multiple bases,
- but under a more realistic online design its effect is modest.

## Recommended Next Steps

1. Learn the live gate instead of hand-setting it.
   Use offline validation labels to fit a simple deployable gate from pre-decision features rather than manual thresholds.

2. Move to horizon-specific admission timing.
   For `h20`-focused deployment, admit `h20` memories at `t+20` instead of waiting for the full `t+30` path.

3. Tighten memory promotion further.
   Promote only cases where the LLM improves both path error and the target trading horizon, not just a weighted aggregate.

4. Add regime-conditioned memory quotas.
   The current bank is still too generic. Positive/negative memories should be balanced across volatility, auction, and energy-stress regimes.

5. Benchmark the online-memory winners under the stricter multi-fold scientific protocol.
   This current sweep is the right production-style holdout test, but the next scientific step is to bring the best base-specific online policies into the multi-fold evaluation framework.
