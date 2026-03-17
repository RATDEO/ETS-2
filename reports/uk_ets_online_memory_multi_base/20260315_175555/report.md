# UK ETS Online-Memory Multi-Base LLM Sweep

Generated: 2026-03-15T18:44:29.899435

## Objectives

1. Convert the new offline helpfulness concept into a deployable online-memory policy.
2. Evaluate whether that policy improves the production-style online LLM refinement path, not just the frozen-memory benchmark.
3. Tailor the LLM refinement path across all current baseline forecasters, not only TSM.
4. Measure absolute and relative lift over each base model on the full 144-window UK holdout.

## Method

- Each run used `test_pool_mode=online_realized_memory`.
- Only fully realized prior cases were eligible for online memory admission.
- A realized case entered memory only if its ex-post LLM helpfulness score met the configured positive or negative threshold.
- Positive and negative memories were stored separately and retrieved independently.
- A deterministic live gate decided whether to apply the LLM using only pre-decision information plus admitted memory similarity/consensus.
- Full 144-window holdout runs were used for all candidates.

## Planned Candidates

| Candidate | Endpoint | Base | Objective | Expected Result |
|---|---|---|---|---|
| `4b_tsm_verifier_conservative` | `4b` | `tsm` | Recover the late-horizon TSM edge in online memory while preventing noisy recent cases from entering the bank. | Expected path MSE: 24.90-25.35. Expected gain vs base TSM: 1.1%-2.9%. |
| `4b_tsm_numeric_only_conservative` | `4b` | `tsm` | Test whether the verifier is over-regularizing online TSM memory and whether numeric-only corrections carry the same signal with less clipping. | Expected path MSE: 25.00-25.45. Expected gain vs base TSM: 0.7%-2.5%. |
| `4b_linear_ridge_numeric_only` | `4b` | `linear_ridge` | Exploit ridge's already-stable shape with long-horizon numeric correction and a cautious online memory gate. | Expected path MSE: 22.05-22.35. Expected gain vs base ridge: 0.6%-1.9%. |
| `4b_linear_ridge_numeric_verifier` | `4b` | `linear_ridge` | Test whether ridge benefits from extra verifier clipping once online memory selects only historically helpful regimes. | Expected path MSE: 22.10-22.40. Expected gain vs base ridge: 0.4%-1.7%. |
| `4b_linear_lasso_numeric_only` | `4b` | `linear_lasso` | Use a stricter online gate to protect the noisier lasso base while still letting the LLM repair large long-horizon miss cases. | Expected path MSE: 23.40-23.95. Expected gain vs base lasso: 1.2%-3.5%. |
| `4b_linear_lasso_numeric_verifier` | `4b` | `linear_lasso` | Check whether verifier clipping stabilizes lasso enough to turn noisy residuals into a cleaner online correction pathway. | Expected path MSE: 23.30-23.90. Expected gain vs base lasso: 1.4%-3.9%. |
| `4b_naive_persistence_relaxed` | `4b` | `naive_persistence` | See whether online memory can salvage a weak naive base by only applying LLM corrections on historically large profitable long-horizon miss cases. | Expected path MSE: 29.50-31.20. Expected gain vs naive: 2.7%-8.0%. |
| `4b_seasonal_naive_relaxed` | `4b` | `seasonal_naive` | Test whether selective online memory can convert seasonal naive into a viable long-horizon correction target without overtrading. | Expected path MSE: 31.00-33.20. Expected gain vs seasonal naive: 3.0%-9.5%. |
| `35b_tsm_numeric_only_online` | `35b` | `tsm` | Port the curated online memory policy to the 35B non-thinking endpoint while keeping the tool stack lightweight. | Expected path MSE: 24.40-25.00. Expected gain vs base TSM: 2.5%-4.8%. |
| `35b_linear_ridge_numeric_only_online` | `35b` | `linear_ridge` | Use 35B as a higher-capacity long-horizon residual corrector for the strongest linear baseline under online memory. | Expected path MSE: 21.85-22.20. Expected gain vs base ridge: 1.3%-2.8%. |
| `35b_linear_lasso_numeric_only_online` | `35b` | `linear_lasso` | See whether 35B can stabilize lasso's more volatile path enough for real online improvement without the verifier. | Expected path MSE: 23.00-23.70. Expected gain vs base lasso: 2.2%-5.1%. |
| `35b_naive_persistence_relaxed_online` | `35b` | `naive_persistence` | Use 35B only on strong positive-memory long-horizon setups to see how much of the naive baseline can be repaired online. | Expected path MSE: 28.50-30.50. Expected gain vs naive: 4.8%-11.1%. |
| `35b_seasonal_naive_relaxed_online` | `35b` | `seasonal_naive` | Test whether 35B can turn seasonal naive into a selective online correction target when only the most helpful memories are admitted. | Expected path MSE: 29.80-31.90. Expected gain vs seasonal naive: 6.8%-13.0%. |

## Results

| Candidate | Base | Endpoint | Base path MSE | LLM path MSE | Abs gain | % gain | h5 % | h20 % | h30 % | Approx applied |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `35b_linear_lasso_numeric_only_online` | `linear_lasso` | `35b` | 24.239772 | 24.153255 | +0.086518 | +0.36% | -0.81% | +0.29% | +0.58% | 34 |
| `35b_linear_ridge_numeric_only_online` | `linear_ridge` | `35b` | 22.486069 | 22.402214 | +0.083855 | +0.37% | -0.56% | +0.25% | +0.74% | 33 |
| `35b_naive_persistence_relaxed_online` | `naive_persistence` | `35b` | 32.051094 | 31.694960 | +0.356134 | +1.11% | +0.84% | +1.15% | +1.04% | 34 |
| `35b_seasonal_naive_relaxed_online` | `seasonal_naive` | `35b` | 34.237653 | 33.950294 | +0.287359 | +0.84% | -0.35% | +0.93% | +0.80% | 34 |
| `35b_tsm_numeric_only_online` | `tsm` | `35b` | 26.850794 | 26.650132 | +0.200662 | +0.75% | -0.56% | +0.82% | +0.95% | 34 |
| `4b_linear_lasso_numeric_verifier` | `linear_lasso` | `4b` | 24.239772 | 24.131677 | +0.108095 | +0.45% | -1.23% | +0.41% | +0.71% | 34 |
| `4b_linear_lasso_numeric_only` | `linear_lasso` | `4b` | 24.239772 | 24.150740 | +0.089033 | +0.37% | -0.67% | +0.37% | +0.46% | 34 |
| `4b_linear_ridge_numeric_verifier` | `linear_ridge` | `4b` | 22.486069 | 22.423316 | +0.062753 | +0.28% | -0.31% | +0.14% | +0.59% | 33 |
| `4b_linear_ridge_numeric_only` | `linear_ridge` | `4b` | 22.486069 | 22.426553 | +0.059516 | +0.26% | +0.06% | +0.20% | +0.43% | 33 |
| `4b_naive_persistence_relaxed` | `naive_persistence` | `4b` | 32.051094 | 31.897689 | +0.153405 | +0.48% | -0.05% | +0.51% | +0.51% | 34 |
| `4b_seasonal_naive_relaxed` | `seasonal_naive` | `4b` | 34.237653 | 34.084646 | +0.153007 | +0.45% | -0.31% | +0.45% | +0.40% | 34 |
| `4b_tsm_verifier_conservative` | `tsm` | `4b` | 26.850794 | 26.709540 | +0.141254 | +0.53% | -0.43% | +0.60% | +0.67% | 34 |
| `4b_tsm_numeric_only_conservative` | `tsm` | `4b` | 26.850794 | 26.732169 | +0.118625 | +0.44% | -0.14% | +0.51% | +0.52% | 34 |

## Best By Base

### 4B

- `linear_lasso`: `4b_linear_lasso_numeric_verifier`
  base path MSE `24.239772` -> LLM `24.131677` (+0.45%)
  horizon gains: `h1 +0.00%`, `h5 -1.23%`, `h20 +0.41%`, `h30 +0.71%`
  objective: Check whether verifier clipping stabilizes lasso enough to turn noisy residuals into a cleaner online correction pathway.
  expected: Expected path MSE: 23.30-23.90. Expected gain vs base lasso: 1.4%-3.9%.
- `linear_ridge`: `4b_linear_ridge_numeric_verifier`
  base path MSE `22.486069` -> LLM `22.423316` (+0.28%)
  horizon gains: `h1 +0.00%`, `h5 -0.31%`, `h20 +0.14%`, `h30 +0.59%`
  objective: Test whether ridge benefits from extra verifier clipping once online memory selects only historically helpful regimes.
  expected: Expected path MSE: 22.10-22.40. Expected gain vs base ridge: 0.4%-1.7%.
- `naive_persistence`: `4b_naive_persistence_relaxed`
  base path MSE `32.051094` -> LLM `31.897689` (+0.48%)
  horizon gains: `h1 +0.00%`, `h5 -0.05%`, `h20 +0.51%`, `h30 +0.51%`
  objective: See whether online memory can salvage a weak naive base by only applying LLM corrections on historically large profitable long-horizon miss cases.
  expected: Expected path MSE: 29.50-31.20. Expected gain vs naive: 2.7%-8.0%.
- `seasonal_naive`: `4b_seasonal_naive_relaxed`
  base path MSE `34.237653` -> LLM `34.084646` (+0.45%)
  horizon gains: `h1 +0.00%`, `h5 -0.31%`, `h20 +0.45%`, `h30 +0.40%`
  objective: Test whether selective online memory can convert seasonal naive into a viable long-horizon correction target without overtrading.
  expected: Expected path MSE: 31.00-33.20. Expected gain vs seasonal naive: 3.0%-9.5%.
- `tsm`: `4b_tsm_verifier_conservative`
  base path MSE `26.850794` -> LLM `26.709540` (+0.53%)
  horizon gains: `h1 -0.00%`, `h5 -0.43%`, `h20 +0.60%`, `h30 +0.67%`
  objective: Recover the late-horizon TSM edge in online memory while preventing noisy recent cases from entering the bank.
  expected: Expected path MSE: 24.90-25.35. Expected gain vs base TSM: 1.1%-2.9%.

### 35B

- `linear_lasso`: `35b_linear_lasso_numeric_only_online`
  base path MSE `24.239772` -> LLM `24.153255` (+0.36%)
  horizon gains: `h1 +0.00%`, `h5 -0.81%`, `h20 +0.29%`, `h30 +0.58%`
  objective: See whether 35B can stabilize lasso's more volatile path enough for real online improvement without the verifier.
  expected: Expected path MSE: 23.00-23.70. Expected gain vs base lasso: 2.2%-5.1%.
- `linear_ridge`: `35b_linear_ridge_numeric_only_online`
  base path MSE `22.486069` -> LLM `22.402214` (+0.37%)
  horizon gains: `h1 +0.00%`, `h5 -0.56%`, `h20 +0.25%`, `h30 +0.74%`
  objective: Use 35B as a higher-capacity long-horizon residual corrector for the strongest linear baseline under online memory.
  expected: Expected path MSE: 21.85-22.20. Expected gain vs base ridge: 1.3%-2.8%.
- `naive_persistence`: `35b_naive_persistence_relaxed_online`
  base path MSE `32.051094` -> LLM `31.694960` (+1.11%)
  horizon gains: `h1 +0.00%`, `h5 +0.84%`, `h20 +1.15%`, `h30 +1.04%`
  objective: Use 35B only on strong positive-memory long-horizon setups to see how much of the naive baseline can be repaired online.
  expected: Expected path MSE: 28.50-30.50. Expected gain vs naive: 4.8%-11.1%.
- `seasonal_naive`: `35b_seasonal_naive_relaxed_online`
  base path MSE `34.237653` -> LLM `33.950294` (+0.84%)
  horizon gains: `h1 +0.00%`, `h5 -0.35%`, `h20 +0.93%`, `h30 +0.80%`
  objective: Test whether 35B can turn seasonal naive into a selective online correction target when only the most helpful memories are admitted.
  expected: Expected path MSE: 29.80-31.90. Expected gain vs seasonal naive: 6.8%-13.0%.
- `tsm`: `35b_tsm_numeric_only_online`
  base path MSE `26.850794` -> LLM `26.650132` (+0.75%)
  horizon gains: `h1 -0.00%`, `h5 -0.56%`, `h20 +0.82%`, `h30 +0.95%`
  objective: Port the curated online memory policy to the 35B non-thinking endpoint while keeping the tool stack lightweight.
  expected: Expected path MSE: 24.40-25.00. Expected gain vs base TSM: 2.5%-4.8%.

## Interpretation

- Positive path gain means the LLM improved the base model on the full holdout.
- `Approx applied` is the number of apply-stage LLM calls; the remainder of the 144 windows were effectively gated to the base forecast.
- This is a production-style single-holdout online-memory benchmark, not the stricter frozen multi-fold scientific benchmark.
