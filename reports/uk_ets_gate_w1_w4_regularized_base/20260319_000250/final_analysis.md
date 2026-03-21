# W1-W4 Base And Gate Follow-Up

Generated: 2026-03-19

## Scope

This note closes the follow-up requested after the repeated-holdout regime diagnostic:

- fix the base `TSM` problem on the difficult historical windows `W1-W4`
- then test whether restoring a wider learned gate schema helps the `LLM` on those harder regimes

All runs in this follow-up used the UK-specific data root:

- `uk_ets/Data_auto_uk`

That matters. Earlier live runs were often falling back to the generic `Data/` tree, which means the old `W0` frontier and this corrected `W1-W4` benchmark are not directly comparable.

## 1. Base Model Result

The best fix for the hard windows was not an older short-sequence `DLinear` variant. It was a more regularized version of the current shared live `DLinear` recipe.

Best base candidate from the regularization sweep:

- `learning_rate=0.001`
- `max_epochs=40`
- `early_stopping_patience=8`
- `dropout=0.5`
- `weight_decay=0.02`

Source benchmark:

- [/Users/davidwilkinson/Desktop/ETS 2/reports/uk_ets_tsm_w1_w4_regularized_benchmark/20260317_143631/results.csv](/Users/davidwilkinson/Desktop/ETS%202/reports/uk_ets_tsm_w1_w4_regularized_benchmark/20260317_143631/results.csv)

Mean `TSM` path MSE on `W1-W4`:

- previous corrected live baseline: `61.326236`
- regularized shared `DLinear`: `54.606079`

That is a `10.96%` reduction in mean path MSE on the difficult windows.

Per-window `TSM` path MSE change:

- `W1`: `19.302713 -> 23.734299` worse
- `W2`: `22.382441 -> 19.702139` better
- `W3`: `97.654152 -> 70.206314` much better
- `W4`: `105.965637 -> 104.781563` slightly better

So the base issue is real, and the best current fix is stronger regularization on the shared live `DLinear`, not reverting to the older tuned short-sequence variants.

## 2. Gate Feature Result

I then froze that better base and reran the live `TSM+LLM` stack on `W1-W4` with two learned-gate schemas:

- `gate_top20`: current promoted schema
- `gate_top29`: original wider schema used before the `W0` feature-count ablation

Results file:

- [/Users/davidwilkinson/Desktop/ETS 2/reports/uk_ets_gate_w1_w4_regularized_base/20260319_000250/results.csv](/Users/davidwilkinson/Desktop/ETS%202/reports/uk_ets_gate_w1_w4_regularized_base/20260319_000250/results.csv)

Mean `LLM` path MSE across `W1-W4`:

- `gate_top20`: `54.613995`
- `gate_top29`: `54.614541`

Mean path improvement vs the same regularized `TSM` base:

- `gate_top20`: `+0.0249%`
- `gate_top29`: `+0.0189%`

So the wider gate is not better in aggregate on the difficult windows.

Per-window `LLM` path MSE:

- `W1`
  - `top20`: `23.793163`
  - `top29`: `23.801416`
  - wider gate worse by `0.008253`
- `W2`
  - `top20`: `19.705550`
  - `top29`: `19.705550`
  - identical
- `W3`
  - `top20`: `69.492012`
  - `top29`: `69.481338`
  - wider gate better by `0.010674`
- `W4`
  - `top20`: `105.465253`
  - `top29`: `105.469861`
  - wider gate worse by `0.004608`

So the exact answer to the gate question is:

- restoring `29` features does **not** materially improve the hard-regime performance
- it is marginally better only on `W3`
- it is worse on `W1` and `W4`
- and effectively identical on `W2`

## 3. Main Conclusion

The hard-window problem is primarily a base-model/regime problem, not a gate-feature-count problem.

What helped:

- correcting the data root to use `uk_ets/Data_auto_uk`
- regularizing the shared live `DLinear`

What did **not** help materially:

- restoring the old `29`-feature learned gate schema

That means the next high-value work should not be “add back more gate columns.” The remaining bottleneck is more likely:

- regime-specific base-model behavior
- gate calibration / safe-region definition
- or richer regime features rather than simply more of the old memory-summary columns

## 4. Recommended Next Step

Use the regularized shared `DLinear` as the new hard-regime base candidate, keep the leaner `top20` gate, and test a `safe-regime` gate layer on top of that:

- allow LLM refinement only when volatility and base long-horizon move size fall inside historically safe regions
- then compare that against the current `top20` gate on the same repeated holdouts

That is the shortest path left that still matches the evidence from `W1-W4`.
