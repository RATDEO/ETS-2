# Final Analysis

The conservative regime-calibration test failed.

What was tested:
- The completed `gate_top20` `W1-W4` runs were reused exactly.
- The learned GBDT gate was rebuilt from the saved validation features and labels using the same configuration.
- Regime buckets were defined from:
  - `base_move_h20_pct`
  - `base_move_h30_pct`
  - optionally `profile_vol_pct`
- Bucket thresholds were selected on the validation-selection slice only.
- Every bucket threshold was constrained to be greater than or equal to the original global threshold, so this evaluation only blocked extra LLM applications and could be computed exactly from the saved test artifacts.

Expected outcome:
- modest lift over the current `gate_top20` baseline, ideally by suppressing bad `W1/W4` applications without destroying the only clearly positive older window (`W3`)

Actual outcome:
- baseline `gate_top20` mean `W1-W4` path MSE: `54.613995`
- `regime_cal_move_default`: `54.713856`
- `regime_cal_move_vol_default`: `54.713856`

Interpretation:
- The regime-calibrated thresholds were mostly just higher versions of the existing global thresholds.
- That reduced apply rate materially, from the baseline live behavior down to about `17.16%` average retained application in the offline conservative test.
- The damage reduction in `W4` was not enough to offset the lost upside in `W3`.
- Adding the volatility split did not change the result, which means the extra bucket granularity was not the missing piece.

Practical conclusion:
- The current `W1-W4` weakness is not solved by threshold engineering alone.
- We should keep:
  - the corrected regularized shared `DLinear` base
  - the `top20` learned gate
- We should stop spending time on simple or conservative threshold calibration layers and move to a stronger regime/uplift model if we want the gate to generalize across harder historical windows.
