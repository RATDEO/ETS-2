# W1-W4 Live Gate Benchmark With Improved Huber Energy Base

Generated: 2026-03-19T21:37:01.754994

## Setup

- Live stack frozen except for the improved base.
- Same `top20` learned gate feature schema.
- Same online-memory / live LLM refinement path.
- Base changed to shared `DLinear` with `Huber(beta=0.5)` and the curated `energy_interactions16` feature slate.
- Data root fixed to `uk_ets/Data_auto_uk`.

## Results

| candidate                 | window   | train_end   | val_end    | test_end   | run_dir                                                         |   tsm_path_mse |   tsm_h20 |   tsm_h30 |   llm_path_mse |   llm_h20 |   llm_h30 |   path_improvement_pct |   h20_improvement_pct |   h30_improvement_pct |
|:--------------------------|:---------|:------------|:-----------|:-----------|:----------------------------------------------------------------|---------------:|----------:|----------:|---------------:|----------:|----------:|-----------------------:|----------------------:|----------------------:|
| gate_top20_huber_energy16 | W1       | 2023-10-27  | 2024-10-26 | 2025-06-30 | /Users/davidwilkinson/Desktop/ETS 2/runs/20260319_201906_205013 |        23.2579 |   28.1896 |   36.6271 |        23.3228 |   28.3314 |   36.6918 |           -0.00278772  |           -0.00503227 |           -0.00176529 |
| gate_top20_huber_energy16 | W2       | 2023-02-22  | 2024-02-22 | 2024-10-26 | /Users/davidwilkinson/Desktop/ETS 2/runs/20260319_204752_59a68f |        18.5983 |   25.401  |   35.6188 |        18.5929 |   25.3957 |   35.5727 |            0.000294185 |            0.00020936 |            0.00129328 |
| gate_top20_huber_energy16 | W3       | 2022-06-20  | 2023-06-20 | 2024-02-22 | /Users/davidwilkinson/Desktop/ETS 2/runs/20260319_210311_0fbd36 |        68.5154 |   89.8141 |  109.466  |        67.7858 |   88.6945 |  107.693  |            0.0106481   |            0.0124653  |            0.0162017  |
| gate_top20_huber_energy16 | W4       | 2021-10-16  | 2022-10-16 | 2023-06-20 | /Users/davidwilkinson/Desktop/ETS 2/runs/20260319_211833_83c717 |       101.735  |  124.654  |  216.043  |       103.085  |  126.79   |  218.306  |           -0.0132707   |           -0.0171358  |           -0.0104757  |

## Comparison vs Prior Regularized-Base Gate Benchmark

| window   |   baseline_tsm_path_mse |   baseline_llm_path_mse |   baseline_path_improvement_pct |   tsm_path_mse |   llm_path_mse |   path_improvement_pct |   base_path_gain_pct |   llm_path_gain_pct |
|:---------|------------------------:|------------------------:|--------------------------------:|---------------:|---------------:|-----------------------:|---------------------:|--------------------:|
| W1       |                 23.7343 |                 23.7932 |                     -0.00248012 |        23.2579 |        23.3228 |           -0.00278772  |            0.0200705 |           0.0197699 |
| W2       |                 19.7021 |                 19.7056 |                     -0.00017315 |        18.5983 |        18.5929 |            0.000294185 |            0.0560249 |           0.056466  |
| W3       |                 70.2063 |                 69.492  |                      0.0101743  |        68.5154 |        67.7858 |            0.0106481   |            0.0240854 |           0.0245525 |
| W4       |                104.782  |                105.465  |                     -0.00652491 |       101.735  |       103.085  |           -0.0132707   |            0.0290727 |           0.0225655 |

## Mean Comparison

- Prior base mean path MSE: `54.606079`
- New base mean path MSE: `53.026727`
- Prior live-stack mean path MSE: `54.613995`
- New live-stack mean path MSE: `53.196703`
- Prior mean LLM uplift vs base: `0.02%`
- New mean LLM uplift vs base: `-0.13%`
