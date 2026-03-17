# Repeated Holdout Regime Diagnostic

Generated: 2026-03-17T13:42:27.069212

## Main Finding
- `W0` is not just a better result; it is a different operating regime for the live stack.
- `W0` path improvement was `11.24%`, while the mean of the four earlier windows was `-1.64%`.

## Why W0 Works Better
- Lower volatility regime: `W0 mean y_vol_20d = 0.012` vs earlier-window mean `0.021`.
- Lower return noise: `W0 mean abs return = 0.0092` vs earlier-window mean `0.0172`.
- Much higher gate conviction: `W0 mean apply probability = 0.778` and apply rate `97.09%`.
- Earlier windows are mostly lower-confidence: their mean apply rate was `28.09%`.

## Important Diagnostic
- The biggest structural issue is validation-to-test helpfulness shift in the gate labels.
- `W0` was relatively aligned: validation positive rate `46.25%`, test positive rate `45.63%`.
- Some failing windows had large shifts, for example `W3` validation positive rate `18.75%` vs test `88.97%`.

## Regime Table

| window                   |   path_improvement_pct |   h20_improvement_pct |   h30_improvement_pct |   gate_threshold |   val_positive_rate |   test_positive_rate |   positive_rate_shift |   apply_rate |   mean_apply_prob |   n_test_samples |   mean_y_vol_20d |   mean_abs_return |   std_return |   auction_day_share |   net_price_change_pct |
|:-------------------------|-----------------------:|----------------------:|----------------------:|-----------------:|--------------------:|---------------------:|----------------------:|-------------:|------------------:|-----------------:|-----------------:|------------------:|-------------:|--------------------:|-----------------------:|
| W0_2025-07-01_2026-03-04 |             0.11244    |            0.161237   |             0.135765  |              0.3 |              0.4625 |             0.456311 |           -0.00618932 |    0.970874  |          0.777518 |              103 |        0.0120672 |        0.00915595 |    0.011455  |            0.719697 |              0.236186  |
| W1_2024-10-27_2025-06-30 |            -0.00965268 |           -0.00977259 |            -0.01866   |              0.7 |              0.3375 |             0.598592 |            0.261092   |    0.316901  |          0.490695 |              142 |        0.0191549 |        0.0158116  |    0.0190655 |            0.812865 |              0.0387048 |
| W2_2024-02-23_2024-10-26 |            -0.0538135  |           -0.0712984  |            -0.0461139 |              0.4 |              0.4    |             0.462069 |            0.062069   |    0.57931   |          0.440368 |              145 |        0.0226308 |        0.0179693  |    0.0233361 |            0.91954  |              0.275376  |
| W3_2023-06-21_2024-02-22 |             0.0126824  |            0.0162965  |             0.0140545 |              0.7 |              0.1875 |             0.889655 |            0.702155   |    0.0206897 |          0.100929 |              145 |        0.0190426 |        0.0157048  |    0.0199719 |            0.850575 |             -0.421134  |
| W4_2022-10-17_2023-06-20 |            -0.0146555  |           -0.0165274  |            -0.0185925 |              0.5 |              0.125  |             0.696552 |            0.571552   |    0.206897  |          0.226666 |              145 |        0.0241853 |        0.0193238  |    0.0241292 |            0.833333 |              0.399941  |

## Top Gate-Feature Shifts (W0 vs average of W1-W4, using validation features)

| feature                   |   W0_mean |   other_windows_mean |   difference_W0_minus_others |   abs_difference |
|:--------------------------|----------:|---------------------:|-----------------------------:|-----------------:|
| profile_fc_h30            |  1.53319  |             4.66373  |                    -3.13054  |         3.13054  |
| base_move_h30_pct         |  1.53319  |             4.66373  |                    -3.13054  |         3.13054  |
| profile_fc_h20            |  0.917048 |             3.87166  |                    -2.95461  |         2.95461  |
| base_move_h20_pct         |  0.917048 |             3.87166  |                    -2.95461  |         2.95461  |
| negative_mean_helpfulness | -3.86814  |            -1.37602  |                    -2.49213  |         2.49213  |
| profile_fc_h5             |  0.238492 |             1.21817  |                    -0.979681 |         0.979681 |
| base_move_h5_pct          |  0.238492 |             1.21817  |                    -0.979681 |         0.979681 |
| negative_signal           |  0.736908 |             0.295222 |                     0.441686 |         0.441686 |
| negative_signal_h30       |  0.736908 |             0.295222 |                     0.441686 |         0.441686 |
| negative_count            |  0.2375   |             0.5625   |                    -0.325    |         0.325    |
| profile_vol_pct           |  3.6149   |             3.38184  |                     0.233058 |         0.233058 |
| profile_change_5          | -0.456817 |            -0.316476 |                    -0.140341 |         0.140341 |

## Figures
![Window price shapes](/Users/davidwilkinson/Desktop/ETS 2/reports/uk_ets_repeated_holdout_regime_diagnostic/20260317_134225/window_price_shapes.png)

![Window volatility shapes](/Users/davidwilkinson/Desktop/ETS 2/reports/uk_ets_repeated_holdout_regime_diagnostic/20260317_134225/window_volatility_shapes.png)

![Window overlay](/Users/davidwilkinson/Desktop/ETS 2/reports/uk_ets_repeated_holdout_regime_diagnostic/20260317_134225/window_overlay_shapes.png)
