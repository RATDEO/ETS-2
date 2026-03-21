# W1-W4 Refinement Benchmark On Improved Base

Generated: 2026-03-20T03:02:58.305480

## Setup

- Base fixed at the improved hard-regime anchor.
- Gate fixed at the current best heuristic live policy.
- Only refinement behavior changes across candidates.

## Results

| candidate                 | window   | train_end   | val_end    | test_end   | run_dir                                                         |   tsm_path_mse |   tsm_h20 |   tsm_h30 |   llm_path_mse |   llm_h20 |   llm_h30 |   path_improvement_pct |   h20_improvement_pct |   h30_improvement_pct |
|:--------------------------|:---------|:------------|:-----------|:-----------|:----------------------------------------------------------------|---------------:|----------:|----------:|---------------:|----------:|----------:|-----------------------:|----------------------:|----------------------:|
| baseline_regime_specific  | W1       | 2023-10-27  | 2024-10-26 | 2025-06-30 | /Users/davidwilkinson/Desktop/ETS 2/runs/20260319_223103_53a87b |        23.2579 |   28.1896 |   36.6271 |        23.251  |   28.2659 |   36.6611 |            0.00030018  |          -0.00270744  |          -0.000927203 |
| baseline_regime_specific  | W2       | 2023-02-22  | 2024-02-22 | 2024-10-26 | /Users/davidwilkinson/Desktop/ETS 2/runs/20260319_223836_2d42e3 |        18.5983 |   25.401  |   35.6188 |        18.6358 |   25.4152 |   35.7932 |           -0.00201365  |          -0.000559438 |          -0.00489561  |
| baseline_regime_specific  | W3       | 2022-06-20  | 2023-06-20 | 2024-02-22 | /Users/davidwilkinson/Desktop/ETS 2/runs/20260319_224506_9db393 |        68.5154 |   89.8141 |  109.466  |        68.2779 |   89.5729 |  108.126  |            0.00346614  |           0.0026858   |           0.0122421   |
| baseline_regime_specific  | W4       | 2021-10-16  | 2022-10-16 | 2023-06-20 | /Users/davidwilkinson/Desktop/ETS 2/runs/20260319_225250_5a77f0 |       101.735  |  124.654  |  216.043  |       102.405  |  125.804  |  216.882  |           -0.00658773  |          -0.00922131  |          -0.0038811   |
| minimal_long_horizon_only | W1       | 2023-10-27  | 2024-10-26 | 2025-06-30 | /Users/davidwilkinson/Desktop/ETS 2/runs/20260320_002859_0d4fb9 |        23.2579 |   28.1896 |   36.6271 |        23.2579 |   28.1896 |   36.6271 |            5.68435e-08 |           2.48541e-08 |           4.65155e-10 |
| minimal_long_horizon_only | W2       | 2023-02-22  | 2024-02-22 | 2024-10-26 | /Users/davidwilkinson/Desktop/ETS 2/runs/20260320_004636_2a3b4d |        18.5983 |   25.401  |   35.6188 |        18.5983 |   25.401  |   35.6188 |            5.45038e-08 |          -1.29672e-07 |          -7.87592e-09 |
| minimal_long_horizon_only | W3       | 2022-06-20  | 2023-06-20 | 2024-02-22 | /Users/davidwilkinson/Desktop/ETS 2/runs/20260320_010937_a8d911 |        68.5154 |   89.8141 |  109.466  |        68.5154 |   89.8141 |  109.466  |           -5.91073e-08 |           3.93601e-08 |           1.11249e-07 |
| minimal_long_horizon_only | W4       | 2021-10-16  | 2022-10-16 | 2023-06-20 | /Users/davidwilkinson/Desktop/ETS 2/runs/20260320_012743_2f5783 |       101.735  |  124.654  |  216.043  |       101.735  |  124.654  |  216.043  |            7.96189e-08 |           1.52309e-08 |           5.98164e-08 |
| citation_bounded_residual | W1       | 2023-10-27  | 2024-10-26 | 2025-06-30 | /Users/davidwilkinson/Desktop/ETS 2/runs/20260320_014552_50fe36 |        23.2579 |   28.1896 |   36.6271 |        23.2579 |   28.1896 |   36.6271 |            5.68435e-08 |           2.48541e-08 |           4.65155e-10 |
| citation_bounded_residual | W2       | 2023-02-22  | 2024-02-22 | 2024-10-26 | /Users/davidwilkinson/Desktop/ETS 2/runs/20260320_020828_b180e8 |        18.5983 |   25.401  |   35.6188 |        18.5983 |   25.401  |   35.6188 |            5.45038e-08 |          -1.29672e-07 |          -7.87592e-09 |
| citation_bounded_residual | W3       | 2022-06-20  | 2023-06-20 | 2024-02-22 | /Users/davidwilkinson/Desktop/ETS 2/runs/20260320_022639_9ba22f |        68.5154 |   89.8141 |  109.466  |        68.5154 |   89.8141 |  109.466  |           -5.91073e-08 |           3.93601e-08 |           1.11249e-07 |
| citation_bounded_residual | W4       | 2021-10-16  | 2022-10-16 | 2023-06-20 | /Users/davidwilkinson/Desktop/ETS 2/runs/20260320_024441_e953d0 |       101.735  |  124.654  |  216.043  |       101.735  |  124.654  |  216.043  |            7.96189e-08 |           1.52309e-08 |           5.98164e-08 |

## Mean By Candidate

| candidate                 |   tsm_path_mse |   llm_path_mse |   path_improvement_pct |   h20_improvement_pct |   h30_improvement_pct |
|:--------------------------|---------------:|---------------:|-----------------------:|----------------------:|----------------------:|
| citation_bounded_residual |        53.0267 |        53.0267 |            3.29647e-08 |          -1.25566e-08 |           4.09138e-08 |
| minimal_long_horizon_only |        53.0267 |        53.0267 |            3.29647e-08 |          -1.25566e-08 |           4.09138e-08 |
| baseline_regime_specific  |        53.0267 |        53.1425 |           -0.00120876  |          -0.0024506   |           0.00063455  |
