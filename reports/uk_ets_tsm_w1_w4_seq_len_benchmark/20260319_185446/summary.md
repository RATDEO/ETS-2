# W1-W4 Seq-Len Sweep

Generated: 2026-03-19T18:56:11.422257

Data dir forced to `uk_ets/Data_auto_uk`.

## Setup

- Base only: `llm.methods=[]`.
- Fixed regularized shared `DLinear`: `lr=0.001`, `epochs=40`, `patience=8`, `dropout=0.5`, `weight_decay=0.02`.
- Only `seq_len` and matching `label_len` vary.

## Results

| candidate        | window   | train_end   | val_end    | test_end   | run_dir                                                         |   seq_len |   label_len | status          |   path_mse |        h1 |        h5 |      h20 |      h30 |
|:-----------------|:---------|:------------|:-----------|:-----------|:----------------------------------------------------------------|----------:|------------:|:----------------|-----------:|----------:|----------:|---------:|---------:|
| seq20_reg_shared | W1       | 2023-10-27  | 2024-10-26 | 2025-06-30 | /Users/davidwilkinson/Desktop/ETS 2/runs/20260319_185446_1f26f9 |        20 |          10 | ok              |    23.7343 |   2.59219 |   9.47455 |  28.5422 |  38.9594 |
| seq20_reg_shared | W2       | 2023-02-22  | 2024-02-22 | 2024-10-26 | /Users/davidwilkinson/Desktop/ETS 2/runs/20260319_185450_140ba4 |        20 |          10 | ok              |    19.7021 |   1.50372 |   4.50719 |  26.414  |  37.0306 |
| seq20_reg_shared | W3       | 2022-06-20  | 2023-06-20 | 2024-02-22 | /Users/davidwilkinson/Desktop/ETS 2/runs/20260319_185453_d4c38e |        20 |          10 | ok              |    70.2063 |   9.62048 |  22.6419  |  90.6118 | 119.532  |
| seq20_reg_shared | W4       | 2021-10-16  | 2022-10-16 | 2023-06-20 | /Users/davidwilkinson/Desktop/ETS 2/runs/20260319_185455_9a3d0d |        20 |          10 | ok              |   104.782  |  11.3508  |  26.7213  | 131.854  | 222.8    |
| seq40_reg_shared | W1       | 2023-10-27  | 2024-10-26 | 2025-06-30 | /Users/davidwilkinson/Desktop/ETS 2/runs/20260319_185456_2c8ec8 |        40 |          20 | ok              |    23.4244 |   2.58954 |  10.1308  |  28.057  |  37.3159 |
| seq40_reg_shared | W2       | 2023-02-22  | 2024-02-22 | 2024-10-26 | /Users/davidwilkinson/Desktop/ETS 2/runs/20260319_185459_ac0cc0 |        40 |          20 | ok              |    19.0704 |   1.52395 |   4.83092 |  25.0825 |  38.5284 |
| seq40_reg_shared | W3       | 2022-06-20  | 2023-06-20 | 2024-02-22 | /Users/davidwilkinson/Desktop/ETS 2/runs/20260319_185501_2b0d5c |        40 |          20 | ok              |    84.4378 |  11.6623  |  28.8721  | 106.058  | 135.514  |
| seq40_reg_shared | W4       | 2021-10-16  | 2022-10-16 | 2023-06-20 | /Users/davidwilkinson/Desktop/ETS 2/runs/20260319_185503_ea8769 |        40 |          20 | ok              |   137.484  |   9.28642 |  27.2446  | 169.857  | 284.211  |
| seq60_reg_shared | W1       | 2023-10-27  | 2024-10-26 | 2025-06-30 | /Users/davidwilkinson/Desktop/ETS 2/runs/20260319_185505_6c5611 |        60 |          30 | ok              |    24.3116 |   2.83422 |  10.518   |  29.7453 |  37.4633 |
| seq60_reg_shared | W2       | 2023-02-22  | 2024-02-22 | 2024-10-26 | /Users/davidwilkinson/Desktop/ETS 2/runs/20260319_185508_4214e8 |        60 |          30 | ok              |    21.689  |   1.59379 |   5.17354 |  30.0084 |  42.0088 |
| seq60_reg_shared | W3       | 2022-06-20  | 2023-06-20 | 2024-02-22 | /Users/davidwilkinson/Desktop/ETS 2/runs/20260319_185510_41c432 |        60 |          30 | ok              |    98.425  |  10.9057  |  32.2641  | 118.453  | 171.287  |
| seq60_reg_shared | W4       | 2021-10-16  | 2022-10-16 | 2023-06-20 | /Users/davidwilkinson/Desktop/ETS 2/runs/20260319_185512_e541a8 |        60 |          30 | ok              |   165.231  |  17.3637  |  33.2313  | 215.073  | 349.192  |
| seq90_reg_shared | W1       | 2023-10-27  | 2024-10-26 | 2025-06-30 | /Users/davidwilkinson/Desktop/ETS 2/runs/20260319_185513_567cb8 |        90 |          45 | ok              |    26.3941 |   2.55648 |  10.0423  |  33.4404 |  42.3564 |
| seq90_reg_shared | W2       | 2023-02-22  | 2024-02-22 | 2024-10-26 | /Users/davidwilkinson/Desktop/ETS 2/runs/20260319_185515_6b99eb |        90 |          45 | ok              |    23.0094 |   2.00908 |   5.55585 |  30.9833 |  43.9287 |
| seq90_reg_shared | W3       | 2022-06-20  | 2023-06-20 | 2024-02-22 | /Users/davidwilkinson/Desktop/ETS 2/runs/20260319_185517_a824ba |        90 |          45 | ok              |    81.9424 |  12.8973  |  37.4461  |  99.0241 | 136.174  |
| seq90_reg_shared | W4       | 2021-10-16  | 2022-10-16 | 2023-06-20 | /Users/davidwilkinson/Desktop/ETS 2/runs/20260319_185519_a399f9 |        90 |          45 | tsm_unavailable |   nan      | nan       | nan       | nan      | nan      |

## Mean By Candidate

| candidate        |   path_mse |      h1 |      h5 |     h20 |      h30 |   n_valid_windows |
|:-----------------|-----------:|--------:|--------:|--------:|---------:|------------------:|
| seq90_reg_shared |    43.782  | 5.82096 | 17.6814 | 54.4826 |  74.1529 |                 3 |
| seq20_reg_shared |    54.6061 | 6.2668  | 15.8362 | 69.3554 | 104.581  |                 4 |
| seq40_reg_shared |    66.1043 | 6.26555 | 17.7696 | 82.2638 | 123.892  |                 4 |
| seq60_reg_shared |    77.4142 | 8.17437 | 20.2968 | 98.32   | 149.988  |                 4 |

## Mean By Candidate (Common Feasible Windows Only)

| candidate        |   path_mse |      h1 |      h5 |     h20 |     h30 |
|:-----------------|-----------:|--------:|--------:|--------:|--------:|
| seq20_reg_shared |    37.8809 | 4.57213 | 12.2079 | 48.5226 | 65.174  |
| seq40_reg_shared |    42.3109 | 5.2586  | 14.6113 | 53.0659 | 70.4527 |
| seq90_reg_shared |    43.782  | 5.82096 | 17.6814 | 54.4826 | 74.1529 |
| seq60_reg_shared |    48.1419 | 5.11125 | 15.9852 | 59.4023 | 83.5864 |

## Best Candidate Per Window

| candidate        | window   | train_end   | val_end    | test_end   | run_dir                                                         |   seq_len |   label_len | status   |   path_mse |       h1 |       h5 |      h20 |      h30 |
|:-----------------|:---------|:------------|:-----------|:-----------|:----------------------------------------------------------------|----------:|------------:|:---------|-----------:|---------:|---------:|---------:|---------:|
| seq40_reg_shared | W1       | 2023-10-27  | 2024-10-26 | 2025-06-30 | /Users/davidwilkinson/Desktop/ETS 2/runs/20260319_185456_2c8ec8 |        40 |          20 | ok       |    23.4244 |  2.58954 | 10.1308  |  28.057  |  37.3159 |
| seq40_reg_shared | W2       | 2023-02-22  | 2024-02-22 | 2024-10-26 | /Users/davidwilkinson/Desktop/ETS 2/runs/20260319_185459_ac0cc0 |        40 |          20 | ok       |    19.0704 |  1.52395 |  4.83092 |  25.0825 |  38.5284 |
| seq20_reg_shared | W3       | 2022-06-20  | 2023-06-20 | 2024-02-22 | /Users/davidwilkinson/Desktop/ETS 2/runs/20260319_185453_d4c38e |        20 |          10 | ok       |    70.2063 |  9.62048 | 22.6419  |  90.6118 | 119.532  |
| seq20_reg_shared | W4       | 2021-10-16  | 2022-10-16 | 2023-06-20 | /Users/davidwilkinson/Desktop/ETS 2/runs/20260319_185455_9a3d0d |        20 |          10 | ok       |   104.782  | 11.3508  | 26.7213  | 131.854  | 222.8    |

## Infeasible Candidates

| candidate        | window   | train_end   | val_end    | test_end   | run_dir                                                         |   seq_len |   label_len | status          |   path_mse |   h1 |   h5 |   h20 |   h30 |
|:-----------------|:---------|:------------|:-----------|:-----------|:----------------------------------------------------------------|----------:|------------:|:----------------|-----------:|-----:|-----:|------:|------:|
| seq90_reg_shared | W4       | 2021-10-16  | 2022-10-16 | 2023-06-20 | /Users/davidwilkinson/Desktop/ETS 2/runs/20260319_185519_a399f9 |        90 |          45 | tsm_unavailable |        nan |  nan |  nan |   nan |   nan |

## Headline

- Best mean candidate: `seq20_reg_shared` with path MSE `54.606079`.
- Baseline `seq20_reg_shared` path MSE: `54.606079`.
- Improvement vs `seq20_reg_shared`: `0.00%`.
- Feasible windows required for headline: `4`.
