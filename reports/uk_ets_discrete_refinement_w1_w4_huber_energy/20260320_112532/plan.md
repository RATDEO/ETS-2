# W1-W4 Discrete Residual Benchmark Plan

## Objective
- Keep the improved Huber + energy-interaction base fixed.
- Keep the current best heuristic gate fixed.
- Replace free-form continuous refinement with a discrete residual action menu.

## Baseline
- Imported from the completed improved-base live-policy benchmark:
  - `baseline_regime_specific`
  - observed mean uplift vs improved base: about `-0.12%`

## Candidates
- `discrete_residual_balanced`: Use a discrete residual action menu with tiny h5 and small long-horizon moves, preserving only a few bounded corrections.
  Expected outcome: Expected mean uplift vs improved base: roughly 0.0% to +0.5%. Should keep some W3 help while reducing W1/W4 overcorrection.
- `discrete_residual_long_only`: Freeze h5 and allow only discrete h20/h30 residual actions, forcing the refiner into a sparse long-horizon correction role.
  Expected outcome: Expected mean uplift vs improved base: roughly -0.2% to +0.4%. Lower variance than the balanced variant, but a real risk of giving up too much W3 upside.
