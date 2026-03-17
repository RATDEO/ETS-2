# Per-Base Online Memory Policy Sweep Plan

## Objectives
- Test each proposed online-memory policy one at a time on every base model.
- Measure whether the policy helps the live causal setup relative to the same-base online baseline.
- Combine only the winning policies for each base separately.
- Verify whether policy gains transfer across strong and weak base forecasters.

## Pre-Registered Singles
- `4b_tsm_baseline`: Reproduce the current live-online baseline under the refactored runner. Expected to reproduce the current 4B online-memory baseline for tsm.
- `4b_tsm_learned_gate`: Convert offline helpfulness into validation-fit threshold rules for live gating. Expected small lift if tsm benefits from validation-fit online gating; likely limited on strong bases.
- `4b_tsm_horizon_specific`: Admit h20 memories earlier than h30 to make online adaptation faster and more target-aligned. Expected to help if tsm needs faster h20 adaptation without waiting for full h30 realization.
- `4b_tsm_regime_specific`: Only retrieve online memories from matching regime buckets. Expected to help if tsm suffers from contradictory memories across regimes.
- `4b_tsm_high_utility_admission`: Promote only cases that materially help long horizons while avoiding h5 damage. Expected to help if tsm benefits from only promoting materially positive long-horizon memories.
- `4b_tsm_prototype_compression`: Compress raw memory banks into a smaller, more diverse case set. Expected to help if tsm benefits from smaller, less cluttered online memory prompts.
- `4b_linear_ridge_baseline`: Reproduce the current live-online baseline under the refactored runner. Expected to reproduce the current 4B online-memory baseline for linear_ridge.
- `4b_linear_ridge_learned_gate`: Convert offline helpfulness into validation-fit threshold rules for live gating. Expected small lift if linear_ridge benefits from validation-fit online gating; likely limited on strong bases.
- `4b_linear_ridge_horizon_specific`: Admit h20 memories earlier than h30 to make online adaptation faster and more target-aligned. Expected to help if linear_ridge needs faster h20 adaptation without waiting for full h30 realization.
- `4b_linear_ridge_regime_specific`: Only retrieve online memories from matching regime buckets. Expected to help if linear_ridge suffers from contradictory memories across regimes.
- `4b_linear_ridge_high_utility_admission`: Promote only cases that materially help long horizons while avoiding h5 damage. Expected to help if linear_ridge benefits from only promoting materially positive long-horizon memories.
- `4b_linear_ridge_prototype_compression`: Compress raw memory banks into a smaller, more diverse case set. Expected to help if linear_ridge benefits from smaller, less cluttered online memory prompts.
- `4b_linear_lasso_baseline`: Reproduce the current live-online baseline under the refactored runner. Expected to reproduce the current 4B online-memory baseline for linear_lasso.
- `4b_linear_lasso_learned_gate`: Convert offline helpfulness into validation-fit threshold rules for live gating. Expected small lift if linear_lasso benefits from validation-fit online gating; likely limited on weaker bases.
- `4b_linear_lasso_horizon_specific`: Admit h20 memories earlier than h30 to make online adaptation faster and more target-aligned. Expected to help if linear_lasso needs faster h20 adaptation without waiting for full h30 realization.
- `4b_linear_lasso_regime_specific`: Only retrieve online memories from matching regime buckets. Expected to help if linear_lasso suffers from contradictory memories across regimes.
- `4b_linear_lasso_high_utility_admission`: Promote only cases that materially help long horizons while avoiding h5 damage. Expected to help if linear_lasso benefits from only promoting materially positive long-horizon memories.
- `4b_linear_lasso_prototype_compression`: Compress raw memory banks into a smaller, more diverse case set. Expected to help if linear_lasso benefits from smaller, less cluttered online memory prompts.
- `4b_naive_persistence_baseline`: Reproduce the current live-online baseline under the refactored runner. Expected to reproduce the current 4B online-memory baseline for naive_persistence.
- `4b_naive_persistence_learned_gate`: Convert offline helpfulness into validation-fit threshold rules for live gating. Expected small lift if naive_persistence benefits from validation-fit online gating; likely limited on weaker bases.
- `4b_naive_persistence_horizon_specific`: Admit h20 memories earlier than h30 to make online adaptation faster and more target-aligned. Expected to help if naive_persistence needs faster h20 adaptation without waiting for full h30 realization.
- `4b_naive_persistence_regime_specific`: Only retrieve online memories from matching regime buckets. Expected to help if naive_persistence suffers from contradictory memories across regimes.
- `4b_naive_persistence_high_utility_admission`: Promote only cases that materially help long horizons while avoiding h5 damage. Expected to help if naive_persistence benefits from only promoting materially positive long-horizon memories.
- `4b_naive_persistence_prototype_compression`: Compress raw memory banks into a smaller, more diverse case set. Expected to help if naive_persistence benefits from smaller, less cluttered online memory prompts.
- `4b_seasonal_naive_baseline`: Reproduce the current live-online baseline under the refactored runner. Expected to reproduce the current 4B online-memory baseline for seasonal_naive.
- `4b_seasonal_naive_learned_gate`: Convert offline helpfulness into validation-fit threshold rules for live gating. Expected small lift if seasonal_naive benefits from validation-fit online gating; likely limited on weaker bases.
- `4b_seasonal_naive_horizon_specific`: Admit h20 memories earlier than h30 to make online adaptation faster and more target-aligned. Expected to help if seasonal_naive needs faster h20 adaptation without waiting for full h30 realization.
- `4b_seasonal_naive_regime_specific`: Only retrieve online memories from matching regime buckets. Expected to help if seasonal_naive suffers from contradictory memories across regimes.
- `4b_seasonal_naive_high_utility_admission`: Promote only cases that materially help long horizons while avoiding h5 damage. Expected to help if seasonal_naive benefits from only promoting materially positive long-horizon memories.
- `4b_seasonal_naive_prototype_compression`: Compress raw memory banks into a smaller, more diverse case set. Expected to help if seasonal_naive benefits from smaller, less cluttered online memory prompts.
