# W1-W4 Refinement Benchmark Plan

## Objective
- Keep the improved Huber + energy-interaction base fixed.
- Keep the current best heuristic gate fixed.
- Test only narrower refinement-layer variants.

## Baseline
- Imported from the completed improved-base live-policy benchmark:
  - `baseline_regime_specific`
  - observed mean uplift vs improved base: about `-0.12%`

## Candidates
- `minimal_long_horizon_only`: Bias the refiner toward abstention by freezing h5, shrinking h20/h30 bounds, and using the minimal apply style.
  Expected outcome: Expected mean uplift vs the improved base: roughly 0.0% to +0.6%. Should reduce W1/W4 overcorrection but may give back some of W3.
- `citation_bounded_residual`: Treat refinement as a small evidence-grounded residual correction using citation_bounded style and tighter structured caps.
  Expected outcome: Expected mean uplift vs the improved base: roughly +0.2% to +1.0%. Best chance to preserve W3 while reducing W1/W4 damage.
