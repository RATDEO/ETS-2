# TSM Live Policy Build Plan

## Objective
- Build the first production-oriented live TSM policy with a learned apply gate and split h20/h30 memory banks.
- Compare each change against the current best live-online TSM benchmark.

## Candidates
- `baseline_regime_specific`: Reproduce the current best live-online TSM policy using regime-specific memory only. Expected path MSE around 4.16-4.20. This is the live benchmark to beat.
- `split_horizon_banks_regime`: Split positive/negative online memory into separate h20 and h30 banks while keeping regime filtering. Expected path MSE around 4.13-4.18. Separate horizon banks should reduce contradictory long-horizon memories.
- `learned_logistic_regime`: Replace the heuristic apply rule with a validation-fit logistic live gate while keeping regime-specific memory. Expected path MSE around 4.12-4.18. A learned gate should reduce false positives and preserve only useful LLM calls.
- `learned_logistic_split_horizon_regime`: Combine a learned logistic apply gate with split h20/h30 memory banks and regime filtering. Expected path MSE around 4.08-4.16. This is the main candidate to beat the current TSM live benchmark.
- `learned_logistic_split_horizon_regime_high_utility`: Add stricter long-horizon utility-based promotion on top of the learned gate and split horizon banks. Expected path MSE around 4.08-4.17. This may improve memory quality if noisy positive cases are currently diluting the bank.
- `learned_gbdt_split_horizon_regime`: Use a nonlinear boosted-tree live gate with split horizon banks and regime filtering. Expected path MSE around 4.08-4.16. This is the more flexible learned-gate alternative if logistic is too linear.
