# W1-W4 Effective Retrieval Benchmark On Improved Base

Generated: 2026-03-20T18:53:41.808434

## Setup

- Base fixed at the improved hard-regime anchor.
- Gate fixed at the current best heuristic live policy.
- Only effective retrieval width changes across candidates.

## Results

| candidate                 | window   | train_end   | val_end    | test_end   | run_dir                                                         |   tsm_path_mse |   tsm_h20 |   tsm_h30 |   llm_path_mse |   llm_h20 |   llm_h30 |   path_improvement_pct |   h20_improvement_pct |   h30_improvement_pct |
|:--------------------------|:---------|:------------|:-----------|:-----------|:----------------------------------------------------------------|---------------:|----------:|----------:|---------------:|----------:|----------:|-----------------------:|----------------------:|----------------------:|
| baseline_regime_specific  | W1       | 2023-10-27  | 2024-10-26 | 2025-06-30 | /Users/davidwilkinson/Desktop/ETS 2/runs/20260319_223103_53a87b |        23.2579 |   28.1896 |   36.6271 |        23.251  |   28.2659 |   36.6611 |            0.00030018  |          -0.00270744  |          -0.000927203 |
| baseline_regime_specific  | W2       | 2023-02-22  | 2024-02-22 | 2024-10-26 | /Users/davidwilkinson/Desktop/ETS 2/runs/20260319_223836_2d42e3 |        18.5983 |   25.401  |   35.6188 |        18.6358 |   25.4152 |   35.7932 |           -0.00201365  |          -0.000559438 |          -0.00489561  |
| baseline_regime_specific  | W3       | 2022-06-20  | 2023-06-20 | 2024-02-22 | /Users/davidwilkinson/Desktop/ETS 2/runs/20260319_224506_9db393 |        68.5154 |   89.8141 |  109.466  |        68.2779 |   89.5729 |  108.126  |            0.00346614  |           0.0026858   |           0.0122421   |
| baseline_regime_specific  | W4       | 2021-10-16  | 2022-10-16 | 2023-06-20 | /Users/davidwilkinson/Desktop/ETS 2/runs/20260319_225250_5a77f0 |       101.735  |  124.654  |  216.043  |       102.405  |  125.804  |  216.882  |           -0.00658773  |          -0.00922131  |          -0.0038811   |
| effective_retrieval_wide6 | W1       | 2023-10-27  | 2024-10-26 | 2025-06-30 | /Users/davidwilkinson/Desktop/ETS 2/runs/20260320_174901_d5259e |        23.2579 |   28.1896 |   36.6271 |        23.0486 |   27.9199 |   36.3035 |            0.00900128  |           0.00956764  |           0.00883713  |
| effective_retrieval_wide6 | W2       | 2023-02-22  | 2024-02-22 | 2024-10-26 | /Users/davidwilkinson/Desktop/ETS 2/runs/20260320_175735_4cfdb4 |        18.5983 |   25.401  |   35.6188 |        18.6276 |   25.4045 |   35.7722 |           -0.00157361  |          -0.000136687 |          -0.0043065   |
| effective_retrieval_wide6 | W3       | 2022-06-20  | 2023-06-20 | 2024-02-22 | /Users/davidwilkinson/Desktop/ETS 2/runs/20260320_180500_a61b19 |        68.5154 |   89.8141 |  109.466  |        67.7545 |   88.7298 |  107.391  |            0.0111047   |           0.012073    |           0.0189611   |
| effective_retrieval_wide6 | W4       | 2021-10-16  | 2022-10-16 | 2023-06-20 | /Users/davidwilkinson/Desktop/ETS 2/runs/20260320_181330_6b1127 |       101.735  |  124.654  |  216.043  |       101.609  |  124.353  |  215.457  |            0.00124548  |           0.00241466  |           0.00271441  |
| effective_retrieval_wide8 | W1       | 2023-10-27  | 2024-10-26 | 2025-06-30 | /Users/davidwilkinson/Desktop/ETS 2/runs/20260320_182015_da9e9d |        23.2579 |   28.1896 |   36.6271 |        23.1893 |   28.0257 |   36.7724 |            0.00294965  |           0.00581359  |          -0.00396588  |
| effective_retrieval_wide8 | W2       | 2023-02-22  | 2024-02-22 | 2024-10-26 | /Users/davidwilkinson/Desktop/ETS 2/runs/20260320_183041_c7484b |        18.5983 |   25.401  |   35.6188 |        18.6288 |   25.422  |   35.7289 |           -0.0016384   |          -0.000826312 |          -0.00309241  |
| effective_retrieval_wide8 | W3       | 2022-06-20  | 2023-06-20 | 2024-02-22 | /Users/davidwilkinson/Desktop/ETS 2/runs/20260320_183637_a4d80c |        68.5154 |   89.8141 |  109.466  |        67.2902 |   88.0071 |  107.04   |            0.0178819   |           0.0201187   |           0.0221621   |
| effective_retrieval_wide8 | W4       | 2021-10-16  | 2022-10-16 | 2023-06-20 | /Users/davidwilkinson/Desktop/ETS 2/runs/20260320_184725_2cded6 |       101.735  |  124.654  |  216.043  |       101.725  |  124.665  |  215.601  |            0.000105658 |          -8.44144e-05 |           0.00204451  |

## Mean By Candidate

| candidate                 |   tsm_path_mse |   llm_path_mse |   path_improvement_pct |   h20_improvement_pct |   h30_improvement_pct |
|:--------------------------|---------------:|---------------:|-----------------------:|----------------------:|----------------------:|
| effective_retrieval_wide8 |        53.0267 |        52.7082 |             0.0048247  |            0.0062554  |            0.00428709 |
| effective_retrieval_wide6 |        53.0267 |        52.7598 |             0.00494445 |            0.00597966 |            0.00655153 |
| baseline_regime_specific  |        53.0267 |        53.1425 |            -0.00120876 |           -0.0024506  |            0.00063455 |
