# TSM Live Trading Conversion Plan

## Objective
- Rerun the verified live TSM baseline and learned-gbdt winner with validation LLM predictions exported.
- Calibrate trading conversion rules on validation only and apply them to the common test holdout.
- Determine whether the verified LLM magnitude gains can be monetized once the trading rule uses more than just forecast sign.

## Policies
- `sign_threshold`: Trade only when the absolute forecast return is large enough. Expected to monetize magnitude gains if the learned gate mainly improves conviction, not direction.
- `linear_size`: Size linearly with forecast return magnitude, clipped to [-1, 1]. Expected to reward better long-horizon magnitude estimates if the LLM is improving sizing quality.
- `tanh_size`: Use a smooth tanh sizing rule on forecast return magnitude. Expected to be more stable than linear sizing if extreme forecast returns are noisy.
- `uplift_gate`: Only trade the LLM forecast when its horizon return differs enough from raw TSM. Expected to help if LLM gains are concentrated in a smaller subset of high-uplift windows.
- `uplift_linear_size`: Size by the absolute LLM-vs-TSM uplift while keeping the LLM sign. Expected to help if incremental LLM edge is informative even when raw TSM direction is unchanged.
