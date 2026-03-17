# Per-Base Online Memory Policy Sweep Report

## Interpretation Note
- These results should be compared within this sweep only.
- Absolute path-MSE levels are not directly comparable to older reports that used different panel states and evaluation plumbing.

## Single-Policy Results
### 4B
#### tsm
- Base path MSE `4.202436`; baseline online LLM path MSE `4.243783`.
- `4b_tsm_regime_specific`: LLM path MSE `4.174480`; gain vs base `0.665%`; delta vs same-base online baseline `+0.069303`; applied `36` windows.
- `4b_tsm_horizon_specific`: LLM path MSE `4.210744`; gain vs base `-0.198%`; delta vs same-base online baseline `+0.033039`; applied `26` windows.
- `4b_tsm_high_utility_admission`: LLM path MSE `4.230978`; gain vs base `-0.679%`; delta vs same-base online baseline `+0.012805`; applied `35` windows.
- `4b_tsm_baseline`: LLM path MSE `4.243783`; gain vs base `-0.984%`; delta vs same-base online baseline `+0.000000`; applied `35` windows.
- `4b_tsm_learned_gate`: LLM path MSE `4.243783`; gain vs base `-0.984%`; delta vs same-base online baseline `+0.000000`; applied `115` windows.
- `4b_tsm_prototype_compression`: LLM path MSE `4.243783`; gain vs base `-0.984%`; delta vs same-base online baseline `+0.000000`; applied `35` windows.
- Successful singles promoted: `horizon_specific`, `regime_specific`, `high_utility_admission`

#### linear_ridge
- Base path MSE `7.400466`; baseline online LLM path MSE `7.538552`.
- `4b_linear_ridge_horizon_specific`: LLM path MSE `7.414174`; gain vs base `-0.185%`; delta vs same-base online baseline `+0.124378`; applied `23` windows.
- `4b_linear_ridge_high_utility_admission`: LLM path MSE `7.532910`; gain vs base `-1.790%`; delta vs same-base online baseline `+0.005642`; applied `33` windows.
- `4b_linear_ridge_baseline`: LLM path MSE `7.538552`; gain vs base `-1.866%`; delta vs same-base online baseline `+0.000000`; applied `33` windows.
- `4b_linear_ridge_learned_gate`: LLM path MSE `7.538552`; gain vs base `-1.866%`; delta vs same-base online baseline `+0.000000`; applied `113` windows.
- `4b_linear_ridge_regime_specific`: LLM path MSE `7.538552`; gain vs base `-1.866%`; delta vs same-base online baseline `+0.000000`; applied `33` windows.
- `4b_linear_ridge_prototype_compression`: LLM path MSE `7.538552`; gain vs base `-1.866%`; delta vs same-base online baseline `+0.000000`; applied `33` windows.
- Successful singles promoted: `ridge_horizon_specific`, `ridge_high_utility_admission`

#### linear_lasso
- Base path MSE `17.142476`; baseline online LLM path MSE `17.803780`.
- `4b_linear_lasso_horizon_specific`: LLM path MSE `17.463384`; gain vs base `-1.872%`; delta vs same-base online baseline `+0.340396`; applied `26` windows.
- `4b_linear_lasso_high_utility_admission`: LLM path MSE `17.719655`; gain vs base `-3.367%`; delta vs same-base online baseline `+0.084125`; applied `36` windows.
- `4b_linear_lasso_baseline`: LLM path MSE `17.803780`; gain vs base `-3.858%`; delta vs same-base online baseline `+0.000000`; applied `34` windows.
- `4b_linear_lasso_learned_gate`: LLM path MSE `17.803780`; gain vs base `-3.858%`; delta vs same-base online baseline `+0.000000`; applied `114` windows.
- `4b_linear_lasso_regime_specific`: LLM path MSE `17.803780`; gain vs base `-3.858%`; delta vs same-base online baseline `+0.000000`; applied `34` windows.
- `4b_linear_lasso_prototype_compression`: LLM path MSE `17.803780`; gain vs base `-3.858%`; delta vs same-base online baseline `+0.000000`; applied `34` windows.
- Successful singles promoted: `lasso_horizon_specific`, `lasso_high_utility_admission`

#### naive_persistence
- Base path MSE `7.736115`; baseline online LLM path MSE `7.909735`.
- `4b_naive_persistence_horizon_specific`: LLM path MSE `7.831994`; gain vs base `-1.239%`; delta vs same-base online baseline `+0.077742`; applied `24` windows.
- `4b_naive_persistence_baseline`: LLM path MSE `7.909735`; gain vs base `-2.244%`; delta vs same-base online baseline `+0.000000`; applied `34` windows.
- `4b_naive_persistence_learned_gate`: LLM path MSE `7.909735`; gain vs base `-2.244%`; delta vs same-base online baseline `+0.000000`; applied `114` windows.
- `4b_naive_persistence_regime_specific`: LLM path MSE `7.909735`; gain vs base `-2.244%`; delta vs same-base online baseline `+0.000000`; applied `34` windows.
- `4b_naive_persistence_high_utility_admission`: LLM path MSE `7.909735`; gain vs base `-2.244%`; delta vs same-base online baseline `+0.000000`; applied `34` windows.
- `4b_naive_persistence_prototype_compression`: LLM path MSE `7.909735`; gain vs base `-2.244%`; delta vs same-base online baseline `+0.000000`; applied `34` windows.
- Successful singles promoted: `persistence_horizon_specific`

#### seasonal_naive
- Base path MSE `8.712429`; baseline online LLM path MSE `8.913793`.
- `4b_seasonal_naive_horizon_specific`: LLM path MSE `8.813463`; gain vs base `-1.160%`; delta vs same-base online baseline `+0.100330`; applied `24` windows.
- `4b_seasonal_naive_baseline`: LLM path MSE `8.913793`; gain vs base `-2.311%`; delta vs same-base online baseline `+0.000000`; applied `34` windows.
- `4b_seasonal_naive_learned_gate`: LLM path MSE `8.913793`; gain vs base `-2.311%`; delta vs same-base online baseline `+0.000000`; applied `114` windows.
- `4b_seasonal_naive_high_utility_admission`: LLM path MSE `8.913793`; gain vs base `-2.311%`; delta vs same-base online baseline `+0.000000`; applied `34` windows.
- `4b_seasonal_naive_prototype_compression`: LLM path MSE `8.913793`; gain vs base `-2.311%`; delta vs same-base online baseline `+0.000000`; applied `34` windows.
- `4b_seasonal_naive_regime_specific`: LLM path MSE `8.918308`; gain vs base `-2.363%`; delta vs same-base online baseline `-0.004515`; applied `34` windows.
- Successful singles promoted: `naive_horizon_specific`

## Combined Per-Base Results
### 4B
- `tsm`: combined path MSE `4.189792`; gain vs base `0.301%`; delta vs best single `-0.015313`; `h20` gain `-0.103%`; `h30` gain `1.567%`.
- `linear_ridge`: combined path MSE `7.538552`; gain vs base `-1.866%`; delta vs best single `-0.124378`; `h20` gain `-3.117%`; `h30` gain `-2.386%`.
- `linear_lasso`: combined path MSE `17.803780`; gain vs base `-3.858%`; delta vs best single `-0.340396`; `h20` gain `-4.059%`; `h30` gain `-3.764%`.
- `naive_persistence`: combined path MSE `7.909735`; gain vs base `-2.244%`; delta vs best single `-0.077742`; `h20` gain `-2.766%`; `h30` gain `-2.681%`.
- `seasonal_naive`: combined path MSE `8.913793`; gain vs base `-2.311%`; delta vs best single `-0.100330`; `h20` gain `-3.555%`; `h30` gain `-3.032%`.

## Main Findings
### 4B
- Combined-policy positive bases: `tsm` (0.301%)
- Best single-policy by base: `linear_lasso` -> `lasso_horizon_specific` (-1.872%), `linear_ridge` -> `ridge_horizon_specific` (-0.185%), `naive_persistence` -> `persistence_horizon_specific` (-1.239%), `seasonal_naive` -> `naive_horizon_specific` (-1.160%), `tsm` -> `regime_specific` (0.665%)

