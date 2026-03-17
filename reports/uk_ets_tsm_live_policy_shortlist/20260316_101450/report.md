# TSM Live Policy Shortlist Report

Best candidate: `learned_gbdt_regime` at path MSE `3.729913`.
Baseline: `baseline_regime_specific` at `4.174480`.

## Ranking
- `learned_gbdt_regime`: path MSE `3.729913`, gain vs base `11.244%`, delta vs baseline `+0.444567`, apply calls `180`, `h20` gain `16.124%`, `h30` gain `13.577%`.
- `learned_logistic_regime`: path MSE `4.086013`, gain vs base `2.770%`, delta vs baseline `+0.088467`, apply calls `131`, `h20` gain `3.824%`, `h30` gain `2.678%`.
- `baseline_regime_specific`: path MSE `4.174480`, gain vs base `0.665%`, delta vs baseline `+0.000000`, apply calls `36`, `h20` gain `2.046%`, `h30` gain `0.065%`.

## Interpretation
- This rerun uses the same live-online architecture as the full build, but the learned-gate selection artifact now records the threshold-selection split explicitly.
- The goal is not to change the policy family again; it is to confirm which of the shortlisted gates survives a clean rerun.

## Selection Semantics
- The learned-gate threshold was chosen on a real within-validation split: `54` validation rows for gate fitting and `26` validation rows for threshold selection.
- After threshold selection, the final gate bundle was refit on all `80` validation rows. That is why the saved selection artifact shows `refit_train_size=80` and `refit_select_size=0`.
- So `refit_select_size=0` does not mean threshold selection was skipped. It means the serialized artifact is the post-selection refit model.

## Artifacts
- Candidate table: [candidate_results.csv](/Users/davidwilkinson/Desktop/ETS 2/reports/uk_ets_tsm_live_policy_shortlist/20260316_101450/candidate_results.csv)
- Plan: [plan.md](/Users/davidwilkinson/Desktop/ETS 2/reports/uk_ets_tsm_live_policy_shortlist/20260316_101450/plan.md)
