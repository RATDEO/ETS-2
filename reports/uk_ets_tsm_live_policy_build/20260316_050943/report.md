# TSM Live Policy Build Report

Best candidate: `learned_gbdt_regime` at path MSE `3.729913`.
Baseline live benchmark: `baseline_regime_specific` at `4.174480`.

## Ranking
- `learned_gbdt_regime`: path MSE `3.729913`, gain vs base `11.244%`, delta vs baseline `+0.444567`, apply calls `180`, `h20` gain `16.124%`, `h30` gain `13.577%`.
- `learned_gbdt_split_horizon_regime`: path MSE `3.923331`, gain vs base `6.642%`, delta vs baseline `+0.251149`, apply calls `172`, `h20` gain `15.092%`, `h30` gain `3.177%`.
- `learned_logistic_regime`: path MSE `4.012189`, gain vs base `4.527%`, delta vs baseline `+0.162291`, apply calls `139`, `h20` gain `5.793%`, `h30` gain `4.824%`.
- `learned_logistic_regime_high_utility`: path MSE `4.050407`, gain vs base `3.618%`, delta vs baseline `+0.124073`, apply calls `142`, `h20` gain `5.245%`, `h30` gain `4.353%`.
- `learned_logistic_split_horizon_regime`: path MSE `4.117427`, gain vs base `2.023%`, delta vs baseline `+0.057053`, apply calls `117`, `h20` gain `3.473%`, `h30` gain `2.330%`.
- `baseline_regime_specific`: path MSE `4.174480`, gain vs base `0.665%`, delta vs baseline `+0.000000`, apply calls `36`, `h20` gain `2.046%`, `h30` gain `0.065%`.
- `learned_logistic_split_horizon_regime_high_utility`: path MSE `4.204232`, gain vs base `-0.043%`, delta vs baseline `-0.029752`, apply calls `109`, `h20` gain `-0.279%`, `h30` gain `0.875%`.
- `split_horizon_banks_regime`: path MSE `4.214644`, gain vs base `-0.290%`, delta vs baseline `-0.040164`, apply calls `26`, `h20` gain `-1.023%`, `h30` gain `0.879%`.

## Key Findings
- The current heuristic live benchmark was real but limited: `baseline_regime_specific` improved path MSE by only `0.665%`.
- A learned logistic gate with regime-specific memory was the first major step-up, cutting path MSE to `4.012189` and improving `h20` by `5.793%`.
- Split h20/h30 banks did not help. They weakened both the logistic and boosted-tree policies relative to their no-split counterparts.
- Stricter long-horizon admission rules also did not help. They were positive only when paired with the strong logistic gate, and still underperformed the plain logistic version.
- The winning policy is a nonlinear boosted-tree gate on top of regime-specific memory without split banks: `learned_gbdt_regime` at `3.729913`, which is an `11.244%` gain over base `TSM` and a `0.444567` gain over the old live benchmark.

## Gate Behavior
- All learned-gate runs chose a probability threshold of `0.30` on the validation slice.
- The boosted-tree winner was more aggressive than the heuristic baseline, with `180` reflect/apply calls versus `36` for the old benchmark. That extra activity paid off because the gate concentrated the added LLM usage at `h20/h30` rather than bluntly increasing `h5` noise.

## Caveats
- The saved `select_size=0` in the gate-selection JSON is a reporting artifact of the final refit bundle, not evidence that no holdout selection happened. Threshold selection was still done on the post-training slice of the validation frame; after that, the gate was refit on all validation rows and the refit bundle is what got serialized. The reporting should still be cleaned up so the pre-refit split is recorded explicitly.
- The weather/auction proxy warnings in these runs are expected here because this live-policy build used the standard `Data/` tree, not the expanded UK auto-data branch. They do not affect relative comparison across these candidates.

## Recommendation
- Promote `learned_gbdt_regime` as the new production-style TSM live policy candidate.
- Do not promote split horizon banks or stricter admission thresholds into the live default yet.
- Next engineering step: fix the within-validation gate selection split, then rerun only `baseline_regime_specific`, `learned_logistic_regime`, and `learned_gbdt_regime` as the clean shortlist.

## Artifacts
- Candidate table: [candidate_results.csv](/Users/davidwilkinson/Desktop/ETS 2/reports/uk_ets_tsm_live_policy_build/20260316_050943/candidate_results.csv)
- Plan: [plan.md](/Users/davidwilkinson/Desktop/ETS 2/reports/uk_ets_tsm_live_policy_build/20260316_050943/plan.md)
