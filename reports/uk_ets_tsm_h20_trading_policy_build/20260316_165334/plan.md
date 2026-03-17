# TSM H20 Trading Policy Build

## Objective
- Freeze the promoted live forecast stack (`DLinear + learned_gbdt(top20) + selective LLM`).
- Calibrate trading-conversion rules on validation only, then evaluate them on the common test holdout.
- Focus on `h20` as the primary production trading horizon while reporting `h30` as a secondary check.

## Expected Outcomes
- `sign_threshold`: likely close to existing sign-only performance; mainly a control.
- `linear_size` / `tanh_size`: expected to improve `h20` if the LLM gain is mostly a magnitude effect.
- `uplift_linear_size`: expected to help if the LLM edge is concentrated in higher-uplift windows.
- `regime_linear_size` / `regime_tanh_size`: expected to be strongest if the learned gate probability captures tradable context, not just forecast MSE.

## Policies
- `sign_threshold`: Pure sign or thresholded sign based on forecast magnitude. Expected to confirm whether magnitude filtering alone is enough to monetize the improved h20 forecast.
- `linear_size`: Size linearly with forecast return magnitude. Expected to convert better magnitude calibration into higher Sharpe if the LLM mostly improves forecast sizing.
- `tanh_size`: Use smooth tanh sizing on forecast magnitude. Expected to be more stable than linear sizing if extreme forecasts are noisy.
- `uplift_linear_size`: Size by LLM-vs-base uplift while preserving the LLM sign. Expected to help if the incremental LLM edge is concentrated in larger-adjustment windows.
- `regime_linear_size`: Scale linear position size by the learned live-gate probability. Expected to be the most product-faithful policy if the learned gate probability tracks when the LLM truly adds economic value.
- `regime_tanh_size`: Scale tanh position size by the learned live-gate probability. Expected to be more robust than raw regime-linear sizing if both probability and magnitude are noisy.
