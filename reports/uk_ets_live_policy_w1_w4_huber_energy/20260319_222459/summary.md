# W1-W4 Live Policy Benchmark On Improved Base

Generated: 2026-03-20T00:19:25.689274

## Setup

- Base frozen at the improved hard-regime anchor: shared `DLinear`, `seq_len=20`, regularized, `Huber(beta=0.5)`, `energy_interactions16`.
- Data root fixed to `uk_ets/Data_auto_uk`.
- Live-policy shortlist rerun on top of this stronger base.
- `learned_gbdt_regime` imported from the completed improved-base benchmark because it is the same production stack already rerun end to end.

## Results

| candidate                | window   | train_end   | val_end    | test_end   | run_dir                                                         |   tsm_path_mse |   tsm_h20 |   tsm_h30 |   llm_path_mse |   llm_h20 |   llm_h30 |   path_improvement_pct |   h20_improvement_pct |   h30_improvement_pct |
|:-------------------------|:---------|:------------|:-----------|:-----------|:----------------------------------------------------------------|---------------:|----------:|----------:|---------------:|----------:|----------:|-----------------------:|----------------------:|----------------------:|
| learned_gbdt_regime      | W1       | 2023-10-27  | 2024-10-26 | 2025-06-30 | /Users/davidwilkinson/Desktop/ETS 2/runs/20260319_201906_205013 |        23.2579 |   28.1896 |   36.6271 |        23.3228 |   28.3314 |   36.6918 |           -0.00278772  |          -0.00503227  |          -0.00176529  |
| learned_gbdt_regime      | W2       | 2023-02-22  | 2024-02-22 | 2024-10-26 | /Users/davidwilkinson/Desktop/ETS 2/runs/20260319_204752_59a68f |        18.5983 |   25.401  |   35.6188 |        18.5929 |   25.3957 |   35.5727 |            0.000294185 |           0.00020936  |           0.00129328  |
| learned_gbdt_regime      | W3       | 2022-06-20  | 2023-06-20 | 2024-02-22 | /Users/davidwilkinson/Desktop/ETS 2/runs/20260319_210311_0fbd36 |        68.5154 |   89.8141 |  109.466  |        67.7858 |   88.6945 |  107.693  |            0.0106481   |           0.0124653   |           0.0162017   |
| learned_gbdt_regime      | W4       | 2021-10-16  | 2022-10-16 | 2023-06-20 | /Users/davidwilkinson/Desktop/ETS 2/runs/20260319_211833_83c717 |       101.735  |  124.654  |  216.043  |       103.085  |  126.79   |  218.306  |           -0.0132707   |          -0.0171358   |          -0.0104757   |
| baseline_regime_specific | W1       | 2023-10-27  | 2024-10-26 | 2025-06-30 | /Users/davidwilkinson/Desktop/ETS 2/runs/20260319_223103_53a87b |        23.2579 |   28.1896 |   36.6271 |        23.251  |   28.2659 |   36.6611 |            0.00030018  |          -0.00270744  |          -0.000927203 |
| baseline_regime_specific | W2       | 2023-02-22  | 2024-02-22 | 2024-10-26 | /Users/davidwilkinson/Desktop/ETS 2/runs/20260319_223836_2d42e3 |        18.5983 |   25.401  |   35.6188 |        18.6358 |   25.4152 |   35.7932 |           -0.00201365  |          -0.000559438 |          -0.00489561  |
| baseline_regime_specific | W3       | 2022-06-20  | 2023-06-20 | 2024-02-22 | /Users/davidwilkinson/Desktop/ETS 2/runs/20260319_224506_9db393 |        68.5154 |   89.8141 |  109.466  |        68.2779 |   89.5729 |  108.126  |            0.00346614  |           0.0026858   |           0.0122421   |
| baseline_regime_specific | W4       | 2021-10-16  | 2022-10-16 | 2023-06-20 | /Users/davidwilkinson/Desktop/ETS 2/runs/20260319_225250_5a77f0 |       101.735  |  124.654  |  216.043  |       102.405  |  125.804  |  216.882  |           -0.00658773  |          -0.00922131  |          -0.0038811   |
| learned_logistic_regime  | W1       | 2023-10-27  | 2024-10-26 | 2025-06-30 | /Users/davidwilkinson/Desktop/ETS 2/runs/20260319_225904_8480f9 |        23.2579 |   28.1896 |   36.6271 |        23.5584 |   28.6347 |   37.114  |           -0.0129192   |          -0.0157901   |          -0.0132915   |
| learned_logistic_regime  | W2       | 2023-02-22  | 2024-02-22 | 2024-10-26 | /Users/davidwilkinson/Desktop/ETS 2/runs/20260319_232012_46aa27 |        18.5983 |   25.401  |   35.6188 |        18.5453 |   25.35   |   35.5633 |            0.0028491   |           0.00200711  |           0.00155858  |
| learned_logistic_regime  | W3       | 2022-06-20  | 2023-06-20 | 2024-02-22 | /Users/davidwilkinson/Desktop/ETS 2/runs/20260319_234146_9dc7b2 |        68.5154 |   89.8141 |  109.466  |        67.7858 |   88.6945 |  107.693  |            0.0106481   |           0.0124653   |           0.0162017   |
| learned_logistic_regime  | W4       | 2021-10-16  | 2022-10-16 | 2023-06-20 | /Users/davidwilkinson/Desktop/ETS 2/runs/20260319_235719_94d960 |       101.735  |  124.654  |  216.043  |       103.546  |  127.324  |  219.378  |           -0.0177982   |          -0.021416    |          -0.0154363   |

## Mean By Candidate

| candidate                |   tsm_path_mse |   llm_path_mse |   path_improvement_pct |   h20_improvement_pct |   h30_improvement_pct |
|:-------------------------|---------------:|---------------:|-----------------------:|----------------------:|----------------------:|
| baseline_regime_specific |        53.0267 |        53.1425 |            -0.00120876 |           -0.0024506  |            0.00063455 |
| learned_gbdt_regime      |        53.0267 |        53.1967 |            -0.00127904 |           -0.00237336 |            0.00131349 |
| learned_logistic_regime  |        53.0267 |        53.3589 |            -0.00430504 |           -0.00568343 |           -0.0027419  |
