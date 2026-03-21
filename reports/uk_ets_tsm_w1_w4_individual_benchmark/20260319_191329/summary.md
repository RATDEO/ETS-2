# W1-W4 Shared vs Individual DLinear

Generated: 2026-03-19T19:14:04.677938

## Setup

- Base only: `llm.methods=[]`.
- Fixed regularized `DLinear`: `seq_len=20`, `label_len=10`, `lr=0.001`, `epochs=40`, `patience=8`, `dropout=0.5`, `weight_decay=0.02`.
- Only `dlinear_individual` varies.

## Results

| candidate            | window   | train_end   | val_end    | test_end   | run_dir                                                         | dlinear_individual   |   path_mse |       h1 |       h5 |      h20 |      h30 |
|:---------------------|:---------|:------------|:-----------|:-----------|:----------------------------------------------------------------|:---------------------|-----------:|---------:|---------:|---------:|---------:|
| shared_seq20_reg     | W1       | 2023-10-27  | 2024-10-26 | 2025-06-30 | /Users/davidwilkinson/Desktop/ETS 2/runs/20260319_191329_9e2b85 | False                |    23.7343 |  2.59219 |  9.47455 |  28.5422 |  38.9594 |
| shared_seq20_reg     | W2       | 2023-02-22  | 2024-02-22 | 2024-10-26 | /Users/davidwilkinson/Desktop/ETS 2/runs/20260319_191334_e2eec7 | False                |    19.7021 |  1.50372 |  4.50719 |  26.414  |  37.0306 |
| shared_seq20_reg     | W3       | 2022-06-20  | 2023-06-20 | 2024-02-22 | /Users/davidwilkinson/Desktop/ETS 2/runs/20260319_191337_ba7ea0 | False                |    70.2063 |  9.62048 | 22.6419  |  90.6118 | 119.532  |
| shared_seq20_reg     | W4       | 2021-10-16  | 2022-10-16 | 2023-06-20 | /Users/davidwilkinson/Desktop/ETS 2/runs/20260319_191339_dc0176 | False                |   104.782  | 11.3508  | 26.7213  | 131.854  | 222.8    |
| individual_seq20_reg | W1       | 2023-10-27  | 2024-10-26 | 2025-06-30 | /Users/davidwilkinson/Desktop/ETS 2/runs/20260319_191340_d88333 | True                 |    18.6647 |  2.34703 |  9.05681 |  23.3492 |  23.7779 |
| individual_seq20_reg | W2       | 2023-02-22  | 2024-02-22 | 2024-10-26 | /Users/davidwilkinson/Desktop/ETS 2/runs/20260319_191349_e7f2ae | True                 |    23.2618 |  1.50975 |  4.77282 |  33.3758 |  43.8381 |
| individual_seq20_reg | W3       | 2022-06-20  | 2023-06-20 | 2024-02-22 | /Users/davidwilkinson/Desktop/ETS 2/runs/20260319_191356_1f1bc4 | True                 |   112.858  |  9.2518  | 23.1195  | 160.037  | 225.02   |
| individual_seq20_reg | W4       | 2021-10-16  | 2022-10-16 | 2023-06-20 | /Users/davidwilkinson/Desktop/ETS 2/runs/20260319_191401_898c92 | True                 |    87.2756 | 10.3837  | 25.2711  | 129.074  | 152.644  |

## Mean By Candidate

| candidate            |   path_mse |      h1 |      h5 |     h20 |     h30 |
|:---------------------|-----------:|--------:|--------:|--------:|--------:|
| shared_seq20_reg     |    54.6061 | 6.2668  | 15.8362 | 69.3554 | 104.581 |
| individual_seq20_reg |    60.5151 | 5.87307 | 15.5551 | 86.4589 | 111.32  |

## Best Candidate Per Window

| candidate            | window   | train_end   | val_end    | test_end   | run_dir                                                         | dlinear_individual   |   path_mse |       h1 |       h5 |      h20 |      h30 |
|:---------------------|:---------|:------------|:-----------|:-----------|:----------------------------------------------------------------|:---------------------|-----------:|---------:|---------:|---------:|---------:|
| individual_seq20_reg | W1       | 2023-10-27  | 2024-10-26 | 2025-06-30 | /Users/davidwilkinson/Desktop/ETS 2/runs/20260319_191340_d88333 | True                 |    18.6647 |  2.34703 |  9.05681 |  23.3492 |  23.7779 |
| shared_seq20_reg     | W2       | 2023-02-22  | 2024-02-22 | 2024-10-26 | /Users/davidwilkinson/Desktop/ETS 2/runs/20260319_191334_e2eec7 | False                |    19.7021 |  1.50372 |  4.50719 |  26.414  |  37.0306 |
| shared_seq20_reg     | W3       | 2022-06-20  | 2023-06-20 | 2024-02-22 | /Users/davidwilkinson/Desktop/ETS 2/runs/20260319_191337_ba7ea0 | False                |    70.2063 |  9.62048 | 22.6419  |  90.6118 | 119.532  |
| individual_seq20_reg | W4       | 2021-10-16  | 2022-10-16 | 2023-06-20 | /Users/davidwilkinson/Desktop/ETS 2/runs/20260319_191401_898c92 | True                 |    87.2756 | 10.3837  | 25.2711  | 129.074  | 152.644  |

## Headline

- Best mean candidate: `shared_seq20_reg` with path MSE `54.606079`.
- Shared baseline path MSE: `54.606079`.
- Improvement vs shared baseline: `0.00%`.
