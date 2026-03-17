# TSM Live Policy Shortlist

## Objective
- Rerun the live-online TSM shortlist after clarifying the learned-gate validation-selection artifact.
- Keep only the production-relevant candidates: old heuristic baseline, learned logistic gate, and learned boosted-tree gate.

## Candidates
- `baseline_regime_specific`: Reproduce the current best live-online TSM policy using regime-specific memory only. Expected path MSE around 4.16-4.20. This is the live benchmark to beat.
- `learned_logistic_regime`: Replace the heuristic apply rule with a validation-fit logistic live gate while keeping regime-specific memory. Expected path MSE around 4.12-4.18. A learned gate should reduce false positives and preserve only useful LLM calls.
- `learned_gbdt_regime`: Use a nonlinear boosted-tree live gate with regime-specific memory and no split horizon banks. Expected path MSE around 3.95-4.10. If the gain is mostly from the gate, this should stay close to or beat the split-bank boosted-tree run.
