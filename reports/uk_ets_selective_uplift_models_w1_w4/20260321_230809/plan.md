# Continuous Uplift Model Plan

## Objective
- Replace coarse hard regime buckets with a continuous out-of-sample uplift score.
- Keep the benchmark strictly leave-one-window-out and causal in the same sense as the previous selective filter test.

## Features
- Pre-decision only: volatility, range, momentum, auction flags, energy interaction features, base move/bias features, retrieval counts, prompt-length metadata, and upstream `llm_applied` state.
- No realized error fields and no post-response adjustment fields are used as predictors.

## Variants
- `uplift_logistic_helpful_strict`: balanced logistic classifier on `helpful_strict`.
- `uplift_gbdt_helpful_strict`: histogram GBDT classifier on `helpful_strict`.
- `uplift_gbdt_path_reg`: histogram GBDT regressor on realized `path_uplift_abs`.

## Selection Protocol
- For each held-out window, fit the model on the other three windows only.
- Tune the keep-threshold on the same training windows only by maximizing mean training uplift versus the improved base.
- Freeze the threshold and score the held-out window.

## Expected Outcomes
- Logistic classifier: likely conservative; expected to beat the hard regime filter but may still trail always-on `wide8`.
- GBDT classifier: best chance of preserving `W3` while trimming `W2/W4`; target is modest improvement over `wide8` of `0.1%` to `0.5%`.
- GBDT regressor: highest variance; may help if uplift magnitude is learnable, but could overfit on the small sample.

## Primary Endpoint
- Mean W1-W4 path MSE versus the improved base and the current always-on `wide8` refiner.
