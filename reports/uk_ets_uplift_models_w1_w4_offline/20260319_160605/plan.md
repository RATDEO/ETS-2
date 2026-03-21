# W1-W4 Uplift Model Benchmark Plan

Generated: 2026-03-19T16:06:05.696253

## Objective

Test stronger learned uplift models for the online apply policy using the completed `W1-W4` runs, while keeping the underlying forecast stack fixed.

## What is being tested

- `uplift_logistic`: logistic classifier that predicts whether the LLM helps
- `uplift_gbdt`: nonlinear GBDT classifier for the same target

## Training protocol

- Windows are processed chronologically: `W4 -> W3 -> W2 -> W1`
- For each target window, the uplift model is trained on:
  - all prior windows' validation and test rows
  - the current window's validation rows
- Thresholds are selected on a chronological holdout inside that training pool

## Important constraint

This is a conservative exact offline benchmark.

Because we only have saved final predictions for the completed `gate_top20` runs, these uplift models can only block additional LLM applications relative to the baseline. They cannot add new LLM applications where the baseline skipped them.

## Expected outcome

- The logistic uplift model should be a stable low-variance baseline, but may be flat
- The GBDT uplift model has the better chance of improving `W1-W4`
- Success condition: beat the current `gate_top20` mean `W1-W4` path MSE `54.613995`
- Expected lift if the idea works even conservatively: roughly `+0.2%` to `+1.0%` on mean `W1-W4` path MSE
