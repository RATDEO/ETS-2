# Selective Regime Filter Analysis

## Scope
- Base system held fixed at the improved UK hard-regime anchor.
- Source refinement system held fixed at `effective_retrieval_wide8`.
- No new LLM calls or new fitting on test-window rows.
- Evaluation uses leave-one-window-out regime selection:
  - derive helpful regimes from three windows
  - freeze the regime filter
  - apply it to the held-out fourth window by reverting non-selected cases to base predictions

## Variants
- `vm_regime_filter`
  - selection space: `(vol_regime, move_regime)`
  - minimum support: `20`
- `vmr_regime_filter`
  - selection space: `(vol_regime, move_regime, retrieval_regime)`
  - minimum support: `12`

## Main Result
- `vm_regime_filter` is the better of the two selective designs.
- Mean `W1-W4` path MSE:
  - improved base: `53.0267`
  - always-on `wide8`: `52.7082`
  - `vm_regime_filter`: `52.8957`
  - `vmr_regime_filter`: `53.0556`

So:
- `vm_regime_filter` improves on always-on `wide8` only in the sense of reducing some harmful applications in `W2/W4`, but it gives back too much of the `W3` gain.
- `vmr_regime_filter` is too sparse and effectively collapses toward abstention.
- Neither variant beats the current improved base decisively and robustly out of sample.

## Window-Level Interpretation
- `W1`
  - both filters hurt versus `wide8`
  - the selective rule removed too many useful cases
- `W2`
  - both filters help versus `wide8`
  - this is the cleanest harmful-window suppression result
- `W3`
  - `vm_regime_filter` keeps some gain but loses a large share of the original `wide8` improvement
  - `vmr_regime_filter` removes everything and falls back to base
- `W4`
  - both filters are slightly worse than both base and `wide8`

## Scientific Interpretation
- The helpful-case dataset is useful for diagnosis, but simple hard regime-bucket selection is too blunt.
- The out-of-sample leave-one-window-out protocol is the right benchmark design here.
- The negative result is informative:
  - “helpful regime” is not captured well enough by coarse `(volatility, base move, retrieval support)` buckets alone.
  - selective refinement likely needs a more continuous residual/uplift score, not hard bucket membership.

## Implication For Next Tranche
- Keep the helpful-case dataset as the scientific foundation.
- Do not promote either regime-filter variant into the main live stack.
- Next implementation should move from hard regime buckets to a continuous residual/uplift model trained only on prior windows, then evaluated on held-out windows.
