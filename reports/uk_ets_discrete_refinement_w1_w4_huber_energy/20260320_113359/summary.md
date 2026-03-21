# W1-W4 Discrete Residual Benchmark On Improved Base

Generated: 2026-03-20T13:11:12.489093

## Setup

- Base fixed at the improved hard-regime anchor.
- Gate fixed at the current best heuristic live policy.
- Only refinement behavior changes across candidates.

## Results

| candidate                   | window   | train_end   | val_end    | test_end   | run_dir                                                         |   tsm_path_mse |   tsm_h20 |   tsm_h30 |   llm_path_mse |   llm_h20 |   llm_h30 |   path_improvement_pct |   h20_improvement_pct |   h30_improvement_pct |
|:----------------------------|:---------|:------------|:-----------|:-----------|:----------------------------------------------------------------|---------------:|----------:|----------:|---------------:|----------:|----------:|-----------------------:|----------------------:|----------------------:|
| baseline_regime_specific    | W1       | 2023-10-27  | 2024-10-26 | 2025-06-30 | /Users/davidwilkinson/Desktop/ETS 2/runs/20260319_223103_53a87b |        23.2579 |   28.1896 |   36.6271 |        23.251  |   28.2659 |   36.6611 |            0.00030018  |          -0.00270744  |          -0.000927203 |
| baseline_regime_specific    | W2       | 2023-02-22  | 2024-02-22 | 2024-10-26 | /Users/davidwilkinson/Desktop/ETS 2/runs/20260319_223836_2d42e3 |        18.5983 |   25.401  |   35.6188 |        18.6358 |   25.4152 |   35.7932 |           -0.00201365  |          -0.000559438 |          -0.00489561  |
| baseline_regime_specific    | W3       | 2022-06-20  | 2023-06-20 | 2024-02-22 | /Users/davidwilkinson/Desktop/ETS 2/runs/20260319_224506_9db393 |        68.5154 |   89.8141 |  109.466  |        68.2779 |   89.5729 |  108.126  |            0.00346614  |           0.0026858   |           0.0122421   |
| baseline_regime_specific    | W4       | 2021-10-16  | 2022-10-16 | 2023-06-20 | /Users/davidwilkinson/Desktop/ETS 2/runs/20260319_225250_5a77f0 |       101.735  |  124.654  |  216.043  |       102.405  |  125.804  |  216.882  |           -0.00658773  |          -0.00922131  |          -0.0038811   |
| discrete_residual_balanced  | W1       | 2023-10-27  | 2024-10-26 | 2025-06-30 | /Users/davidwilkinson/Desktop/ETS 2/runs/20260320_113359_63ca63 |        23.2579 |   28.1896 |   36.6271 |        23.2542 |   28.1838 |   36.6224 |            0.000158752 |           0.000205844 |           0.000130865 |
| discrete_residual_balanced  | W2       | 2023-02-22  | 2024-02-22 | 2024-10-26 | /Users/davidwilkinson/Desktop/ETS 2/runs/20260320_114115_4584bd |        18.5983 |   25.401  |   35.6188 |        18.621  |   25.4569 |   35.6228 |           -0.00121874  |          -0.00219844  |          -0.000114106 |
| discrete_residual_balanced  | W3       | 2022-06-20  | 2023-06-20 | 2024-02-22 | /Users/davidwilkinson/Desktop/ETS 2/runs/20260320_115855_f1ff06 |        68.5154 |   89.8141 |  109.466  |        68.7518 |   90.2481 |  109.983  |           -0.0034509   |          -0.00483278  |          -0.00471582  |
| discrete_residual_balanced  | W4       | 2021-10-16  | 2022-10-16 | 2023-06-20 | /Users/davidwilkinson/Desktop/ETS 2/runs/20260320_121549_2fe539 |       101.735  |  124.654  |  216.043  |       101.94   |  125.06   |  216.317  |           -0.00200934  |          -0.00325193  |          -0.00126934  |
| discrete_residual_long_only | W1       | 2023-10-27  | 2024-10-26 | 2025-06-30 | /Users/davidwilkinson/Desktop/ETS 2/runs/20260320_122050_f74345 |        23.2579 |   28.1896 |   36.6271 |        23.2519 |   28.1646 |   36.622  |            0.000257951 |           0.000886527 |           0.000140052 |
| discrete_residual_long_only | W2       | 2023-02-22  | 2024-02-22 | 2024-10-26 | /Users/davidwilkinson/Desktop/ETS 2/runs/20260320_122741_73d77c |        18.5983 |   25.401  |   35.6188 |        18.6082 |   25.4173 |   35.6384 |           -0.000530988 |          -0.000640051 |          -0.000551031 |
| discrete_residual_long_only | W3       | 2022-06-20  | 2023-06-20 | 2024-02-22 | /Users/davidwilkinson/Desktop/ETS 2/runs/20260320_124507_97dc16 |        68.5154 |   89.8141 |  109.466  |        68.7475 |   90.0931 |  110.165  |           -0.00338876  |          -0.00310684  |          -0.00637817  |
| discrete_residual_long_only | W4       | 2021-10-16  | 2022-10-16 | 2023-06-20 | /Users/davidwilkinson/Desktop/ETS 2/runs/20260320_130626_2dfc6b |       101.735  |  124.654  |  216.043  |       101.946  |  125.015  |  216.456  |           -0.00207419  |          -0.00289507  |          -0.00190955  |

## Mean By Candidate

| candidate                   |   tsm_path_mse |   llm_path_mse |   path_improvement_pct |   h20_improvement_pct |   h30_improvement_pct |
|:----------------------------|---------------:|---------------:|-----------------------:|----------------------:|----------------------:|
| discrete_residual_long_only |        53.0267 |        53.1385 |            -0.001434   |           -0.00143886 |           -0.00217467 |
| discrete_residual_balanced  |        53.0267 |        53.1417 |            -0.00163006 |           -0.00251933 |           -0.0014921  |
| baseline_regime_specific    |        53.0267 |        53.1425 |            -0.00120876 |           -0.0024506  |            0.00063455 |
