# W1-W4 Base Rebuild Plan

Generated: 2026-03-19T18:52:53.913890

## Objective

Improve the hard-regime base `TSM` (`DLinear`) on `W1-W4` before spending more time on the LLM gate.

## Diagnosis

- The corrected regularized shared `DLinear` already improved mean `W1-W4` base path MSE from `61.326236` to `54.606079`.
- Restoring the wider `29`-feature gate did not materially help.
- The remaining weakness is concentrated in the base model, especially `W3/W4` long horizons.

## Planned Steps

1. Longer-context `DLinear` sweep
   - Test `seq_len` `20 / 40 / 60 / 90`.
   - Keep the best known regularized shared recipe fixed.
   - Expected outcome: `40` or `60` should outperform `20` if the hard windows need longer regime memory.
   - Success threshold: beat mean `W1-W4` base path MSE `54.606079`.

2. Shared vs individual `DLinear`
   - Freeze the best `seq_len` from step 1.
   - Test `dlinear_individual=true` against shared.
   - Expected outcome: modest gain if per-channel decomposition helps the harder windows.

3. Loss shaping
   - Freeze the best architecture from steps 1-2.
   - Test robust and/or horizon-weighted training.
   - Expected outcome: improve `h20/h30` tail behavior in `W3/W4`.

4. Curated engineered base features
   - Add only tightly engineered UK energy/weather regime features.
   - Avoid the broad raw expansion that previously degraded the base.

5. Hybrid base if needed
   - If pure `DLinear` stalls, test a validation-selected `DLinear + ridge` base blend.

## Current Step

Run step 1 only: longer-context `seq_len` sweep on the same `W1-W4` benchmark geometry.
