# Leakage-safe selective policy benchmark v2

## Protocol

Model and threshold selection uses forward out-of-fold predictions from older windows only. The selected policy is refit on all older windows and evaluated once on W2 and W1.

`precall` uses strictly last-observed features and can avoid inference. `postcall` uses response metadata and can veto an adjustment, but cannot save inference cost.

## Results

| policy   | test_window   | selected_candidate   |   keep_share_applied |   base_mse |   always_accept_mse |   selected_mse |   improvement_vs_base_pct |   improvement_vs_always_accept_pct |
|:---------|:--------------|:---------------------|---------------------:|-----------:|--------------------:|---------------:|--------------------------:|-----------------------------------:|
| precall  | W2            | ridge_a0.1           |             1.000000 |  18.598327 |           18.628799 |      18.628799 |                 -0.163846 |                           0.000000 |
| precall  | W1            | ridge_a0.1           |             1.000000 |  23.257937 |           23.189336 |      23.189336 |                  0.294959 |                           0.000000 |
| postcall | W2            | ridge_a100           |             1.000000 |  18.598327 |           18.628799 |      18.628799 |                 -0.163846 |                           0.000000 |
| postcall | W1            | ridge_a100           |             0.630769 |  23.257937 |           23.189336 |      23.187629 |                  0.302296 |                           0.007359 |

## Macro

| policy   |   n_test_windows |   macro_base_mse |   macro_always_accept_mse |   macro_selected_mse |   improvement_vs_base_pct |   improvement_vs_always_accept_pct |   mean_keep_share_applied |
|:---------|-----------------:|-----------------:|--------------------------:|---------------------:|--------------------------:|-----------------------------------:|--------------------------:|
| postcall |                2 |        20.928132 |                 20.909067 |            20.908214 |                  0.095171 |                           0.004081 |                  0.815385 |
| precall  |                2 |        20.928132 |                 20.909067 |            20.909067 |                  0.091095 |                           0.000000 |                  1.000000 |

## Block bootstrap

- `precall` vs base: `-0.092444` to `0.165607` (**inconclusive**).
- `precall` vs always accept: `0.000000` to `0.000000` (**inconclusive**).
- `postcall` vs base: `-0.104383` to `0.177959` (**inconclusive**).
- `postcall` vs always accept: `-0.024268` to `0.027288` (**inconclusive**).

## Safe features

- `precall`: `asof_y`, `asof_y_return`, `asof_target_range_pct`, `asof_target_volume`, `asof_y_vol_20d`, `asof_y_ma_5d`, `asof_y_momentum_20d`, `asof_is_auction_day`, `asof_uk_icap_secondary_print_day`, `asof_coal_brent_ratio`, `asof_coal_brent_ratio_z20`, `asof_uk_power_gas_vol_ratio_20d`, `asof_uk_gas_hdd18_surprise_interaction`, `asof_uk_hdd18_7d_ma`, `asof_uka_brent_ratio`, `asof_uka_brent_ratio_z20`, `asof_uk_gas_vol_20d`, `asof_uk_temp_mean_c`, `base_move_h20_pct`, `base_move_h30_pct`
- `postcall`: `apply_prompt_length`, `teaching_examples`, `matched_teaching_count`, `support_example_count`, `positive_memory_count`, `negative_memory_count`, `dynamic_frozen_horizons`, `adjust_h1`, `adjust_h5`, `adjust_h20`, `adjust_h30`

## Limits

- Thirty-step paths overlap; confidence intervals therefore use 30-origin circular blocks.
- Only historical LLM-applied rows reveal the LLM counterfactual.
- These windows end in 2025; current production claims require newly frozen rolling-origin tests.
