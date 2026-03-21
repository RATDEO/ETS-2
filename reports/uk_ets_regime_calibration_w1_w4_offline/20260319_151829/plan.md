# W1-W4 Conservative Regime Calibration Plan

Generated: 2026-03-19T15:18:29.895426

## Objective

Test whether the current `top20` learned gate improves on `W1-W4` when we keep the same forecast stack and only raise the apply threshold in harder regime buckets.

This is an exact offline evaluation because it only blocks additional LLM applications relative to the completed `gate_top20` runs.

## Method

- Reuse the completed regularized-base `gate_top20` runs
- Rebuild the learned GBDT gate from saved validation features/labels using the same configuration
- Use the validation selection slice only to choose regime-bucket thresholds
- Constrain every bucket threshold to be `>=` the run's original global threshold
- Apply the calibrated thresholds to the saved test probabilities
- Replace newly blocked rows with raw `TSM` predictions and recompute metrics exactly

## Candidates

- `regime_cal_move_default`: `moderate_move` vs `large_move` buckets using `|h20| > 2.5%` or `|h30| > 4.0%`
- `regime_cal_move_vol_default`: the same move split plus `profile_vol_pct > 3.8` into a 4-bucket scheme

## Expected outcome

- Hard safe-regime blocking already failed, so expected gains are modest
- Success condition: beat the current `gate_top20` mean `W1-W4` path MSE `54.613995`
- Expected lift if this works: roughly `+0.1%` to `+0.5%` on mean `W1-W4` path MSE
