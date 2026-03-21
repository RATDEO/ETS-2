# W1-W4 Effective Retrieval Benchmark Plan

## Objective
- Keep the improved Huber + energy-interaction base fixed.
- Keep the current best heuristic gate fixed.
- Increase the effective retrieved example set, not just the requested k.

## Baseline
- Imported from the completed improved-base live-policy benchmark:
  - `baseline_regime_specific`
  - observed mean uplift vs improved base: about `-0.12%`

## Candidates
- `effective_retrieval_wide6`: Widen the actual matched-example set and support-memory budget so the refiner can see more than three matched cases.
  Expected outcome: Expected mean uplift vs current heuristic: roughly flat to +0.5%. If example starvation is the bottleneck, this should help W3 without blowing up W4.
- `effective_retrieval_wide8`: Push the effective retrieval width further with more matched examples and a larger online-memory budget.
  Expected outcome: Expected mean uplift vs current heuristic: -0.3% to +0.6%. Higher upside if example diversity matters, but also the clearest risk of prompt dilution.
