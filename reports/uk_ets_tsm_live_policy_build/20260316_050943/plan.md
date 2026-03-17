# TSM Live Policy Build Plan

## Objective
- Improve the production-style online TSM+LLM system by replacing heuristic apply logic with learned gates and by testing horizon-specific online memory policies.
- Measure each policy on the full live-online holdout, then compare combinations against the current regime-specific benchmark.

## Candidates
- `learned_gbdt_regime`: Use a nonlinear boosted-tree live gate with regime-specific memory and no split horizon banks. Expected path MSE around 3.95-4.10. If the gain is mostly from the gate, this should stay close to or beat the split-bank boosted-tree run.
- `learned_gbdt_split_horizon_regime`: Use a nonlinear boosted-tree live gate with split horizon banks and regime filtering. Expected path MSE around 4.08-4.16. This is the more flexible learned-gate alternative if logistic is too linear.
- `learned_logistic_regime`: Replace the heuristic apply rule with a validation-fit logistic live gate while keeping regime-specific memory. Expected path MSE around 4.12-4.18. A learned gate should reduce false positives and preserve only useful LLM calls.
- `learned_logistic_regime_high_utility`: Add stricter long-horizon utility-based promotion on top of the learned logistic gate without split horizon banks. Expected path MSE around 4.00-4.12. If stricter promotion helps on its own, it should improve on the plain learned logistic gate.
- `learned_logistic_split_horizon_regime`: Combine a learned logistic apply gate with split h20/h30 memory banks and regime filtering. Expected path MSE around 4.08-4.16. This is the main candidate to beat the current TSM live benchmark.
- `baseline_regime_specific`: Reproduce the current best live-online TSM policy using regime-specific memory only. Expected path MSE around 4.16-4.20. This is the live benchmark to beat.
- `learned_logistic_split_horizon_regime_high_utility`: Add stricter long-horizon utility-based promotion on top of the learned gate and split horizon banks. Expected path MSE around 4.08-4.17. This may improve memory quality if noisy positive cases are currently diluting the bank.
- `split_horizon_banks_regime`: Split positive/negative online memory into separate h20 and h30 banks while keeping regime filtering. Expected path MSE around 4.13-4.18. Separate horizon banks should reduce contradictory long-horizon memories.
