# TSM Live Gate Feature Ablation Report

## Scope
- Objective: test whether the current `29`-feature learned `hist_gbdt` live gate is actually optimal.
- Method: rerun the live `TSM` online-memory winner while changing only `llm.cot_rf.online_memory_policy.gate.learned.feature_columns`.
- Ranking source: validation mutual information from [feature_ranking.csv](/Users/davidwilkinson/Desktop/ETS%202/reports/uk_ets_tsm_live_gate_feature_ablation/20260316_133421/feature_ranking.csv).

## Status
- Completed candidates:
  - `full_29_control`
  - `top_25_mi`
  - `top_20_mi`
  - `top_15_mi`
  - `top_10_mi`
- Stopped before `top_5_mi` and the family-only tails because the top-k curve was already decisive and further endpoint usage was not justified for the specific question of feature-count optimality.

## Results
- `full_29_control` (`29` features): path MSE `3.754735`
- `top_25_mi` (`25` features): path MSE `3.729913`
- `top_20_mi` (`20` features): path MSE `3.729913`
- `top_15_mi` (`15` features): path MSE `3.771059`
- `top_10_mi` (`10` features): path MSE `3.889859`

## Interpretation
- `29` features is not optimal.
- The best observed region is `20-25` features.
- Compression from `29 -> 25` or `29 -> 20` improves the live gate.
- Compression from `20 -> 15` starts to lose performance.
- Compression to `10` features is clearly too aggressive.

## Practical Recommendation
- Promote a `20`-feature or `25`-feature learned-gate schema for the live `TSM` product, not the full `29`.
- Prefer `20` if we value a smaller production surface with no observed loss versus `25`.
- Keep the remaining family-only ablations optional unless we specifically want interpretability by feature family rather than feature count.

## Top Ranked Features
Top validation-ranked features from [feature_ranking.csv](/Users/davidwilkinson/Desktop/ETS%202/reports/uk_ets_tsm_live_gate_feature_ablation/20260316_133421/feature_ranking.csv):
- `positive_count_h30`
- `positive_count`
- `positive_signal_h30`
- `positive_signal`
- `net_signal`
- `negative_mean_helpfulness`
- `base_move_h5_pct`
- `profile_fc_h5`
- `negative_count`
- `positive_mean_helpfulness`
- `profile_vol_pct`
- `positive_mean_similarity`

## Artifacts
- Plan: [plan.md](/Users/davidwilkinson/Desktop/ETS%202/reports/uk_ets_tsm_live_gate_feature_ablation/20260316_133421/plan.md)
- Candidate results: [candidate_results.csv](/Users/davidwilkinson/Desktop/ETS%202/reports/uk_ets_tsm_live_gate_feature_ablation/20260316_133421/candidate_results.csv)
- Feature ranking: [feature_ranking.csv](/Users/davidwilkinson/Desktop/ETS%202/reports/uk_ets_tsm_live_gate_feature_ablation/20260316_133421/feature_ranking.csv)
