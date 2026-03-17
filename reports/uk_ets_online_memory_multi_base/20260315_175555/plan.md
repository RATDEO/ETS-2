# Planned UK ETS Online-Memory Multi-Base Sweep

Generated: 2026-03-15T17:55:55.768917

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