# Selective Regime Filter Plan

## Objective
- Test whether a preregistered selective regime filter can preserve the positive residual cases from `effective_retrieval_wide8` while suppressing harmful ones.
- Evaluate scientifically via leave-one-window-out selection.

## Protocol
- Use the completed improved-base `wide8` W1-W4 runs only.
- For each held-out window, derive helpful regimes from the other three windows only.
- Apply the frozen regime filter offline to the held-out saved predictions by reverting non-selected cases from LLM output back to raw base output.

## Variants
- `vm_regime_filter`: select on `(vol_regime, move_regime)`.
- `vmr_regime_filter`: select on `(vol_regime, move_regime, retrieval_regime)`.

## Selection Rule
- Positive mean path uplift in training windows.
- Helpful strict rate at least as high as harmful strict rate.
- Minimum support threshold: `20` cases for VM, `12` cases for VMR.

## Expected Outcomes
- `vm_regime_filter`: expected to improve mean W1-W4 path MSE by `0.1%` to `0.5%` vs `wide8`; likely still close to the improved base.
- `vmr_regime_filter`: higher variance; may help `W2/W4` more, but may overfit and lose `W1/W3` support.

## Primary Endpoint
- Mean W1-W4 path MSE versus both the improved base and the current always-on `wide8` refiner.
