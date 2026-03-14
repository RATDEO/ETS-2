# UK ETS 4B Delta Calibration Sweep

Generated: 2026-03-11T02:41:32.535909

## Checkpoint

- Base checkpoint run: `/Users/davidwilkinson/Desktop/ETS 2/runs/20260310_195100_69e04e`
- Checkpoint raw `TSM+LLM-COT-RF-HDELTA` path MSE: `22.198106`
- Reference `linear_ridge` path MSE: `22.419955`

## Candidate Results

| Candidate | Reported model | Change | Aim | Expected | Actual | Vs checkpoint | Outcome | h5 vs checkpoint | h20 vs checkpoint | h30 vs checkpoint |
|---|---|---|---|---|---:|---:|---|---:|---:|---:|
| control_recent_high_error_k4 | TSM+LLM-COT-RF-HDELTA | Re-run the current 4B checkpoint without post-LLM calibration. | Verify the new calibration-capable runner reproduces the checkpoint exactly before shrinking any horizons. | 22.14-22.24 | 22.198106 | +0.000000 | within | +0.000000 | +0.000000 | +0.000000 |
| recent_high_error_k4_delta_per_h5_h20_h30 | TSM+LLM-COT-RF-HDELTA_delta_calibrated | Fit separate validation scales for `h5`, `h20`, and `h30` while leaving the rest of the path untouched. | Test whether the 4B path needs horizon-specific trust rather than a single long-horizon shrink factor. | 22.08-22.20 | 22.268015 | +0.069909 | missed | -0.026414 | +0.931902 | +1.191790 |
| recent_high_error_k4_delta_shared_h20_h30 | TSM+LLM-COT-RF-HDELTA_delta_calibrated | Apply one shared validation-trained shrink factor to the `h20` and `h30` deltas only. | Keep the current checkpoint structure, but reduce any long-horizon overshoot without disturbing early horizons. | 22.10-22.20 | 22.268895 | +0.070790 | missed | +0.000000 | +0.931902 | +1.191790 |
| utility_mmr_augmented_delta_shared_h20_h30 | TSM+LLM-COT-RF-HDELTA_delta_calibrated | Use the best structural retrieval challenger from the prior sweep, then apply shared long-horizon delta shrinkage. | See whether the utility-MMR variant can recover its small `h5` gain once `h20/h30` are calibrated down on validation. | 22.08-22.20 | 22.277640 | +0.079535 | missed | -0.025721 | +0.931902 | +1.191790 |

## Best Candidate

- Candidate: `control_recent_high_error_k4`
- Reported model: `TSM+LLM-COT-RF-HDELTA`
- Actual path MSE: `22.198106`
- Delta vs checkpoint: `+0.000000`
- Delta vs base TSM: `-0.528247`
- Delta vs ridge: `-0.221849`
- Run dir: `/Users/davidwilkinson/Desktop/ETS 2/runs/20260311_005418_f0e1b8`
