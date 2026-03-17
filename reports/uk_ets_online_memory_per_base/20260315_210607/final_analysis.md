# Final Analysis: Per-Base Online Memory Policy Sweep

## Scope
- Endpoint: `4b`
- Evaluation mode: production-style `online_realized_memory`
- Goal: test each proposed online-memory improvement one at a time on every base forecaster, then combine the successful policies per base

## Comparability Note
- These numbers should be compared within this sweep only.
- Absolute path-MSE levels are on the current compact/normalized pipeline and are not directly comparable to older reports that operated on different panel states or evaluation plumbing.

## Policies Tested One at a Time
- `baseline`
- `learned_gate`
- `horizon_specific`
- `regime_specific`
- `high_utility_admission`
- `prototype_compression`

## Single-Policy Results

### TSM
| Policy | LLM path MSE | Gain vs base | Delta vs online baseline |
|---|---:|---:|---:|
| `regime_specific` | `4.174480` | `+0.665%` | `+0.069303` |
| `horizon_specific` | `4.210744` | `-0.198%` | `+0.033039` |
| `high_utility_admission` | `4.230978` | `-0.679%` | `+0.012805` |
| `baseline` | `4.243783` | `-0.984%` | `+0.000000` |
| `learned_gate` | `4.243783` | `-0.984%` | `+0.000000` |
| `prototype_compression` | `4.243783` | `-0.984%` | `+0.000000` |

Best single policy: `regime_specific`

Key horizon effects:
- `h20`: `+2.046%`
- `h30`: `+0.065%`
- `h5`: still slightly worse

### linear_ridge
| Policy | LLM path MSE | Gain vs base | Delta vs online baseline |
|---|---:|---:|---:|
| `horizon_specific` | `7.414174` | `-0.185%` | `+0.124378` |
| `high_utility_admission` | `7.532910` | `-1.790%` | `+0.005642` |
| `baseline` | `7.538552` | `-1.866%` | `+0.000000` |
| `learned_gate` | `7.538552` | `-1.866%` | `+0.000000` |
| `regime_specific` | `7.538552` | `-1.866%` | `+0.000000` |
| `prototype_compression` | `7.538552` | `-1.866%` | `+0.000000` |

Best single policy: `horizon_specific`

Key horizon effects for best single:
- `h20`: `-0.736%`
- `h30`: `-0.590%`

Conclusion: the online LLM still hurts `linear_ridge` even with the best policy found.

### linear_lasso
| Policy | LLM path MSE | Gain vs base | Delta vs online baseline |
|---|---:|---:|---:|
| `horizon_specific` | `17.463384` | `-1.872%` | `+0.340396` |
| `high_utility_admission` | `17.719655` | `-3.367%` | `+0.084125` |
| `baseline` | `17.803780` | `-3.858%` | `+0.000000` |
| `learned_gate` | `17.803780` | `-3.858%` | `+0.000000` |
| `regime_specific` | `17.803780` | `-3.858%` | `+0.000000` |
| `prototype_compression` | `17.803780` | `-3.858%` | `+0.000000` |

Best single policy: `horizon_specific`

Conclusion: `linear_lasso` benefits relative to its weak online LLM baseline, but not enough to beat the raw base.

### naive_persistence
| Policy | LLM path MSE | Gain vs base | Delta vs online baseline |
|---|---:|---:|---:|
| `horizon_specific` | `7.831994` | `-1.239%` | `+0.077742` |
| `baseline` | `7.909735` | `-2.244%` | `+0.000000` |
| `learned_gate` | `7.909735` | `-2.244%` | `+0.000000` |
| `regime_specific` | `7.909735` | `-2.244%` | `+0.000000` |
| `high_utility_admission` | `7.909735` | `-2.244%` | `+0.000000` |
| `prototype_compression` | `7.909735` | `-2.244%` | `+0.000000` |

Best single policy: `horizon_specific`

Conclusion: same pattern as the linear bases. The policy improves the online LLM baseline but not the base forecaster.

### seasonal_naive
| Policy | LLM path MSE | Gain vs base | Delta vs online baseline |
|---|---:|---:|---:|
| `horizon_specific` | `8.813463` | `-1.160%` | `+0.100330` |
| `baseline` | `8.913793` | `-2.311%` | `+0.000000` |
| `learned_gate` | `8.913793` | `-2.311%` | `+0.000000` |
| `high_utility_admission` | `8.913793` | `-2.311%` | `+0.000000` |
| `prototype_compression` | `8.913793` | `-2.311%` | `+0.000000` |
| `regime_specific` | `8.918308` | `-2.363%` | `-0.004515` |

Best single policy: `horizon_specific`

Conclusion: same as `naive_persistence`.

## Combined Per-Base Results
| Base | Combined policy path MSE | Gain vs base | Delta vs best single |
|---|---:|---:|---:|
| `tsm` | `4.189792` | `+0.301%` | `-0.015313` |
| `linear_ridge` | `7.538552` | `-1.866%` | `-0.124378` |
| `linear_lasso` | `17.803780` | `-3.858%` | `-0.340396` |
| `naive_persistence` | `7.909735` | `-2.244%` | `-0.077742` |
| `seasonal_naive` | `8.913793` | `-2.311%` | `-0.100330` |

Combined successful variants per base:
- `tsm`: `horizon_specific + regime_specific + high_utility_admission`
- `linear_ridge`: `horizon_specific + high_utility_admission`
- `linear_lasso`: `horizon_specific + high_utility_admission`
- `naive_persistence`: `horizon_specific`
- `seasonal_naive`: `horizon_specific`

## Main Findings
1. The only base model that gets a positive live-online improvement over the raw base is `TSM`.
2. The best live-online `TSM` policy is still a single policy, not a combination: `regime_specific`.
3. `horizon_specific` is the most transferable policy across non-TSM bases, but it only reduces damage; it does not create a net positive refinement effect.
4. `learned_gate` and `prototype_compression` are effectively no-ops in this live setup.
5. `regime_specific` is highly model-dependent. It helps `TSM`, does nothing for `linear_ridge`, `linear_lasso`, and `naive_persistence`, and slightly hurts `seasonal_naive`.
6. Blindly combining all policies that beat the online LLM baseline is a bad rule. For every base, the combined run is worse than the best single-policy run.

## Operational Recommendation
- Keep `TSM + regime_specific online memory` as the only live-online candidate worth promoting from this tranche.
- Do not deploy online-memory refinement on `linear_ridge`, `linear_lasso`, `naive_persistence`, or `seasonal_naive` yet.
- If we continue improving the live system, the next tranche should be `TSM`-specific:
  - stronger regime tagging
  - better positive/negative memory separation
  - learned apply/no-apply gate using regime features
  - horizon-specific memory banks with `h20` admission at `t+20`

## Artifacts
- Singles CSV: [single_policy_results.csv](/Users/davidwilkinson/Desktop/ETS%202/reports/uk_ets_online_memory_per_base/20260315_210607/single_policy_results.csv)
- Combined CSV: [combined_policy_results.csv](/Users/davidwilkinson/Desktop/ETS%202/reports/uk_ets_online_memory_per_base/20260315_210607/combined_policy_results.csv)
- Summary by base: [best_by_base.csv](/Users/davidwilkinson/Desktop/ETS%202/reports/uk_ets_online_memory_per_base/20260315_210607/best_by_base.csv)
