# Online Memory Policy Improvement Report

## Single-Policy Results
### 4B TSM
- `4b_tsm_regime_specific`: path MSE `4.174480` vs baseline `4.243783`; delta `+0.069303`; applied `36` windows. Expected: Expected path MSE 26.50-26.72. Regime filtering should cut contradictory memories and slightly improve net signal quality.
- `4b_tsm_horizon_specific`: path MSE `4.210744` vs baseline `4.243783`; delta `+0.033039`; applied `26` windows. Expected: Expected path MSE 26.50-26.70. Earlier h20 memory availability should help long horizons without materially changing h1/h5.
- `4b_tsm_high_utility_admission`: path MSE `4.230978` vs baseline `4.243783`; delta `+0.012805`; applied `35` windows. Expected: Expected path MSE 26.45-26.70. Promoting only truly helpful cases should reduce noisy bank growth.
- `4b_tsm_baseline`: path MSE `4.243783` vs baseline `4.243783`; delta `+0.000000`; applied `35` windows. Expected: Expected path MSE near 26.65-26.80. This should reproduce the current live-online 4B TSM baseline.
- `4b_tsm_learned_gate`: path MSE `4.243783` vs baseline `4.243783`; delta `+0.000000`; applied `115` windows. Expected: Expected path MSE 26.45-26.68. Offline-tuned thresholds should reduce false positives and modestly lift live TSM performance.
- `4b_tsm_prototype_compression`: path MSE `4.243783` vs baseline `4.243783`; delta `+0.000000`; applied `35` windows. Expected: Expected path MSE 26.50-26.72. Fewer but cleaner memories should suit 4B.
- Successful singles promoted for `4b`: `horizon_specific`, `regime_specific`, `high_utility_admission`

## Combined Multi-Base Results
### 4B Combined Policy
- `tsm`: path MSE `4.189792`; gain vs base `0.301%`; vs previous live baseline `+84.313%`; `h20` gain `-0.103%`; `h30` gain `1.567%`.
- `linear_ridge`: path MSE `7.416483`; gain vs base `-0.216%`; vs previous live baseline `+66.925%`; `h20` gain `-0.698%`; `h30` gain `-0.723%`.
- `naive_persistence`: path MSE `7.830430`; gain vs base `-1.219%`; vs previous live baseline `+75.451%`; `h20` gain `-1.378%`; `h30` gain `-1.480%`.
- `seasonal_naive`: path MSE `8.813463`; gain vs base `-1.160%`; vs previous live baseline `+74.142%`; `h20` gain `-1.258%`; `h30` gain `-1.675%`.
- `linear_lasso`: path MSE `17.545483`; gain vs base `-2.351%`; vs previous live baseline `+27.293%`; `h20` gain `-2.387%`; `h30` gain `-2.270%`.

## Artifacts
- Singles CSV: `single_policy_results.csv`
- Combined CSV: `combined_multi_base_results.csv`
