# Frozen compact-model benchmark

Candidate family, regularisation and shrinkage were selected on validation data only. The chosen model was then refit on train+validation and evaluated once on the frozen test split.

| run_id                 | selected_candidate            |   mse_path |   reported_tsm_mse |   reported_linear_ridge_mse |   improvement_vs_tsm_pct |   improvement_vs_linear_ridge_pct |
|:-----------------------|:------------------------------|-----------:|-------------------:|----------------------------:|-------------------------:|----------------------------------:|
| 20260322_194840_381bee | ridge_target_a100             |    22.0755 |            23.2579 |                     21.5837 |                   5.0839 |                           -2.2787 |
| 20260322_213347_11a76f | ridge_summary_a1              |    20.9025 |            18.5983 |                     90.3629 |                 -12.3889 |                           76.8683 |
| 20260322_230818_cd22ab | extra_trees_summary_leaf2_mf1 |    42.4720 |            68.5154 |                    101.9225 |                  38.0110 |                           58.3291 |
| 20260323_003616_212adb | extra_trees_full_leaf2_mf1    |    61.9194 |           101.7353 |                    310.4285 |                  39.1367 |                           80.0536 |

## Aggregate

- Mean improvement_vs_tsm_pct: **17.46%**; wins: **3/4**
- Mean improvement_vs_linear_ridge_pct: **53.24%**; wins: **3/4**
- Mean improvement_vs_naive_persistence_pct: **-7.13%**; wins: **1/4**

The benchmark also records price-reconstruction alignment. A near-zero `alignment_mae` confirms that the saved target windows were converted back to price paths correctly.