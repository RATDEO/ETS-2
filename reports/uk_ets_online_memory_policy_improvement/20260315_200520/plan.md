# Online Memory Policy Sweep Plan

## Objectives
- Replace heuristic online gating with validation-fit threshold rules.
- Speed online learning with horizon-specific admission.
- Reduce contradictory memories with regime-specific retrieval.
- Improve memory quality with utility-gated admission.
- Control prompt dilution with prototype compression.
- Combine only the successful single-policy changes and port them across bases.

## Pre-Registered Singles
- `4b_tsm_baseline`: Reproduce the current live-online TSM baseline under the refactored runner. Expected path MSE near 26.65-26.80. This should reproduce the current live-online 4B TSM baseline.
- `4b_tsm_learned_gate`: Convert offline helpfulness into validation-fit threshold rules for live gating. Expected path MSE 26.45-26.68. Offline-tuned thresholds should reduce false positives and modestly lift live TSM performance.
- `4b_tsm_horizon_specific`: Admit h20 memories earlier than h30 to make online adaptation faster and more target-aligned. Expected path MSE 26.50-26.70. Earlier h20 memory availability should help long horizons without materially changing h1/h5.
- `4b_tsm_regime_specific`: Only retrieve online memories from matching regime buckets. Expected path MSE 26.50-26.72. Regime filtering should cut contradictory memories and slightly improve net signal quality.
- `4b_tsm_high_utility_admission`: Promote only cases that materially help long horizons while avoiding h5 damage. Expected path MSE 26.45-26.70. Promoting only truly helpful cases should reduce noisy bank growth.
- `4b_tsm_prototype_compression`: Compress raw memory banks into a smaller, more diverse case set. Expected path MSE 26.50-26.72. Fewer but cleaner memories should suit 4B.
- `35b_tsm_baseline`: Reproduce the current live-online TSM baseline under the refactored runner. Expected path MSE near 26.55-26.72. This should reproduce the current live-online 35B TSM baseline.
- `35b_tsm_learned_gate`: Convert offline helpfulness into validation-fit threshold rules for live gating. Expected path MSE 26.30-26.58. 35B should benefit more from a validation-fit live gate than 4B.
- `35b_tsm_horizon_specific`: Admit h20 memories earlier than h30 to make online adaptation faster and more target-aligned. Expected path MSE 26.35-26.60. Faster h20 admission should let 35B exploit long-horizon improvements earlier.
- `35b_tsm_regime_specific`: Only retrieve online memories from matching regime buckets. Expected path MSE 26.35-26.62. Regime filtering should help 35B avoid low-similarity memories.
- `35b_tsm_high_utility_admission`: Promote only cases that materially help long horizons while avoiding h5 damage. Expected path MSE 26.30-26.60. Higher-capacity 35B should benefit from more selective positive-memory promotion.
- `35b_tsm_prototype_compression`: Compress raw memory banks into a smaller, more diverse case set. Expected path MSE 26.40-26.64. Compression may help less than on 4B but should still control prompt clutter.
