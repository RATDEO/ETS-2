# W1-W4 Hybrid Base Benchmark

Generated: 2026-03-19T20:06:38.595790

## Setup

- Base-only benchmark on the current improved `Huber` `DLinear` anchor with the energy-interactions feature slate.
- Fit `linear_ridge` on the exact same train split reconstructed from saved datasets.
- Select the ridge/DLinear blend on validation only across weights `0, 0.25, 0.5, 0.75, 1.0` and schedules `uniform, ramp`.

## Test Results

| model                  | window   | train_end   | val_end    | test_end   | run_dir                                                         |   path_mse |         h1 |         h5 |       h20 |       h30 |
|:-----------------------|:---------|:------------|:-----------|:-----------|:----------------------------------------------------------------|-----------:|-----------:|-----------:|----------:|----------:|
| dlinear_huber_energy16 | W1       | 2023-10-27  | 2024-10-26 | 2025-06-30 | /Users/davidwilkinson/Desktop/ETS 2/runs/20260319_200627_d17b1f |    23.2579 |    2.61888 |    9.57823 |   28.1896 |   36.6271 |
| linear_ridge           | W1       | 2023-10-27  | 2024-10-26 | 2025-06-30 | /Users/davidwilkinson/Desktop/ETS 2/runs/20260319_200627_d17b1f |  1777.97   | 1658.85    | 1693.99    | 1815.9    | 1893.51   |
| ridge_dlinear_blend    | W1       | 2023-10-27  | 2024-10-26 | 2025-06-30 | /Users/davidwilkinson/Desktop/ETS 2/runs/20260319_200627_d17b1f |    23.2579 |    2.61888 |    9.57823 |   28.1896 |   36.6271 |
| dlinear_huber_energy16 | W2       | 2023-02-22  | 2024-02-22 | 2024-10-26 | /Users/davidwilkinson/Desktop/ETS 2/runs/20260319_200632_785633 |    18.5983 |    1.41346 |    4.1155  |   25.401  |   35.6188 |
| linear_ridge           | W2       | 2023-02-22  | 2024-02-22 | 2024-10-26 | /Users/davidwilkinson/Desktop/ETS 2/runs/20260319_200632_785633 |  1594.6    | 1575.13    | 1584.25    | 1596.62   | 1615.69   |
| ridge_dlinear_blend    | W2       | 2023-02-22  | 2024-02-22 | 2024-10-26 | /Users/davidwilkinson/Desktop/ETS 2/runs/20260319_200632_785633 |    18.5983 |    1.41346 |    4.1155  |   25.401  |   35.6188 |
| dlinear_huber_energy16 | W3       | 2022-06-20  | 2023-06-20 | 2024-02-22 | /Users/davidwilkinson/Desktop/ETS 2/runs/20260319_200635_22304b |    68.5154 |   10.4356  |   22.648   |   89.8141 |  109.466  |
| linear_ridge           | W3       | 2022-06-20  | 2023-06-20 | 2024-02-22 | /Users/davidwilkinson/Desktop/ETS 2/runs/20260319_200635_22304b |  1676.46   | 1823.48    | 1772.71    | 1630.18   | 1563.96   |
| ridge_dlinear_blend    | W3       | 2022-06-20  | 2023-06-20 | 2024-02-22 | /Users/davidwilkinson/Desktop/ETS 2/runs/20260319_200635_22304b |    68.5154 |   10.4356  |   22.648   |   89.8141 |  109.466  |
| dlinear_huber_energy16 | W4       | 2021-10-16  | 2022-10-16 | 2023-06-20 | /Users/davidwilkinson/Desktop/ETS 2/runs/20260319_200637_4e4dc6 |   101.735  |   11.4088  |   27.1403  |  124.654  |  216.043  |
| linear_ridge           | W4       | 2021-10-16  | 2022-10-16 | 2023-06-20 | /Users/davidwilkinson/Desktop/ETS 2/runs/20260319_200637_4e4dc6 |  4843.58   | 5059.98    | 5019.06    | 4753.68   | 4646.26   |
| ridge_dlinear_blend    | W4       | 2021-10-16  | 2022-10-16 | 2023-06-20 | /Users/davidwilkinson/Desktop/ETS 2/runs/20260319_200637_4e4dc6 |   101.735  |   11.4088  |   27.1403  |  124.654  |  216.043  |

## Mean By Model

| model                  |   path_mse |         h1 |        h5 |       h20 |       h30 |
|:-----------------------|-----------:|-----------:|----------:|----------:|----------:|
| ridge_dlinear_blend    |    53.0267 |    6.46919 |   15.8705 |   67.0148 |   99.4388 |
| dlinear_huber_energy16 |    53.0267 |    6.46919 |   15.8705 |   67.0148 |   99.4388 |
| linear_ridge           |  2473.15   | 2529.36    | 2517.5    | 2449.09   | 2429.86   |

## Blend Selections

| window   | run_dir                                                         | schedule   |   weight |   val_mse_path |
|:---------|:----------------------------------------------------------------|:-----------|---------:|---------------:|
| W1       | /Users/davidwilkinson/Desktop/ETS 2/runs/20260319_200627_d17b1f | uniform    |        1 |        21.6698 |
| W2       | /Users/davidwilkinson/Desktop/ETS 2/runs/20260319_200632_785633 | uniform    |        1 |        69.0892 |
| W3       | /Users/davidwilkinson/Desktop/ETS 2/runs/20260319_200635_22304b | uniform    |        1 |       122.583  |
| W4       | /Users/davidwilkinson/Desktop/ETS 2/runs/20260319_200637_4e4dc6 | uniform    |        1 |        95.5953 |

## Best Model Per Window

| model                  | window   | train_end   | val_end    | test_end   | run_dir                                                         |   path_mse |       h1 |       h5 |      h20 |      h30 |
|:-----------------------|:---------|:------------|:-----------|:-----------|:----------------------------------------------------------------|-----------:|---------:|---------:|---------:|---------:|
| ridge_dlinear_blend    | W1       | 2023-10-27  | 2024-10-26 | 2025-06-30 | /Users/davidwilkinson/Desktop/ETS 2/runs/20260319_200627_d17b1f |    23.2579 |  2.61888 |  9.57823 |  28.1896 |  36.6271 |
| ridge_dlinear_blend    | W2       | 2023-02-22  | 2024-02-22 | 2024-10-26 | /Users/davidwilkinson/Desktop/ETS 2/runs/20260319_200632_785633 |    18.5983 |  1.41346 |  4.1155  |  25.401  |  35.6188 |
| dlinear_huber_energy16 | W3       | 2022-06-20  | 2023-06-20 | 2024-02-22 | /Users/davidwilkinson/Desktop/ETS 2/runs/20260319_200635_22304b |    68.5154 | 10.4356  | 22.648   |  89.8141 | 109.466  |
| ridge_dlinear_blend    | W4       | 2021-10-16  | 2022-10-16 | 2023-06-20 | /Users/davidwilkinson/Desktop/ETS 2/runs/20260319_200637_4e4dc6 |   101.735  | 11.4088  | 27.1403  | 124.654  | 216.043  |

## Headline

- Best mean model: `ridge_dlinear_blend` with path MSE `53.026725`.
- Baseline `dlinear_huber_energy16` path MSE: `53.026727`.
- Improvement vs baseline: `0.00%`.
