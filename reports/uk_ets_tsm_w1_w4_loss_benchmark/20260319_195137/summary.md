# W1-W4 Loss Shaping Benchmark

Generated: 2026-03-19T19:52:16.468106

## Setup

- Base only: `llm.methods=[]`.
- Fixed shared regularized `DLinear`: `seq_len=20`, `label_len=10`, `lr=0.001`, `epochs=40`, `patience=8`, `dropout=0.5`, `weight_decay=0.02`.
- Tested `mse`, `huber`, `weighted_mse`, and `weighted_huber`.
- Tail weights: `{1: 1.0, 5: 1.0, 20: 2.0, 30: 3.0}`.

## Results

| candidate                       | window   | train_end   | val_end    | test_end   | run_dir                                                         | loss_type      |   path_mse |       h1 |       h5 |      h20 |      h30 |
|:--------------------------------|:---------|:------------|:-----------|:-----------|:----------------------------------------------------------------|:---------------|-----------:|---------:|---------:|---------:|---------:|
| mse_shared_seq20_reg            | W1       | 2023-10-27  | 2024-10-26 | 2025-06-30 | /Users/davidwilkinson/Desktop/ETS 2/runs/20260319_195137_a858dc | mse            |    23.7343 |  2.59219 |  9.47455 |  28.5422 |  38.9594 |
| mse_shared_seq20_reg            | W2       | 2023-02-22  | 2024-02-22 | 2024-10-26 | /Users/davidwilkinson/Desktop/ETS 2/runs/20260319_195142_355cfd | mse            |    19.7021 |  1.50372 |  4.50719 |  26.414  |  37.0306 |
| mse_shared_seq20_reg            | W3       | 2022-06-20  | 2023-06-20 | 2024-02-22 | /Users/davidwilkinson/Desktop/ETS 2/runs/20260319_195145_7344a6 | mse            |    70.2063 |  9.62048 | 22.6419  |  90.6118 | 119.532  |
| mse_shared_seq20_reg            | W4       | 2021-10-16  | 2022-10-16 | 2023-06-20 | /Users/davidwilkinson/Desktop/ETS 2/runs/20260319_195147_5b5396 | mse            |   104.782  | 11.3508  | 26.7213  | 131.854  | 222.8    |
| huber_shared_seq20_reg          | W1       | 2023-10-27  | 2024-10-26 | 2025-06-30 | /Users/davidwilkinson/Desktop/ETS 2/runs/20260319_195149_55ee1e | huber          |    22.9453 |  2.62117 |  9.54875 |  27.6949 |  35.9526 |
| huber_shared_seq20_reg          | W2       | 2023-02-22  | 2024-02-22 | 2024-10-26 | /Users/davidwilkinson/Desktop/ETS 2/runs/20260319_195152_ae51e3 | huber          |    17.9647 |  1.41456 |  4.1137  |  24.3242 |  34.3057 |
| huber_shared_seq20_reg          | W3       | 2022-06-20  | 2023-06-20 | 2024-02-22 | /Users/davidwilkinson/Desktop/ETS 2/runs/20260319_195154_edf602 | huber          |    69.8799 | 10.4121  | 22.6172  |  91.956  | 112.369  |
| huber_shared_seq20_reg          | W4       | 2021-10-16  | 2022-10-16 | 2023-06-20 | /Users/davidwilkinson/Desktop/ETS 2/runs/20260319_195156_07e0f6 | huber          |   102.833  | 11.3908  | 26.9551  | 127.085  | 218.574  |
| weighted_mse_shared_seq20_reg   | W1       | 2023-10-27  | 2024-10-26 | 2025-06-30 | /Users/davidwilkinson/Desktop/ETS 2/runs/20260319_195158_d174d3 | weighted_mse   |    23.2063 |  2.55328 |  9.34433 |  28.0916 |  37.8385 |
| weighted_mse_shared_seq20_reg   | W2       | 2023-02-22  | 2024-02-22 | 2024-10-26 | /Users/davidwilkinson/Desktop/ETS 2/runs/20260319_195201_928cae | weighted_mse   |    19.0165 |  1.48476 |  4.29003 |  25.9721 |  35.5318 |
| weighted_mse_shared_seq20_reg   | W3       | 2022-06-20  | 2023-06-20 | 2024-02-22 | /Users/davidwilkinson/Desktop/ETS 2/runs/20260319_195203_03641d | weighted_mse   |    72.5933 |  9.73771 | 23.3965  |  93.1299 | 123.086  |
| weighted_mse_shared_seq20_reg   | W4       | 2021-10-16  | 2022-10-16 | 2023-06-20 | /Users/davidwilkinson/Desktop/ETS 2/runs/20260319_195205_77693f | weighted_mse   |   106.904  | 11.5215  | 27.8369  | 133.572  | 225.882  |
| weighted_huber_shared_seq20_reg | W1       | 2023-10-27  | 2024-10-26 | 2025-06-30 | /Users/davidwilkinson/Desktop/ETS 2/runs/20260319_195207_a4fbd2 | weighted_huber |    22.9817 |  2.62002 |  9.55623 |  27.7199 |  36.0256 |
| weighted_huber_shared_seq20_reg | W2       | 2023-02-22  | 2024-02-22 | 2024-10-26 | /Users/davidwilkinson/Desktop/ETS 2/runs/20260319_195210_406e35 | weighted_huber |    18.4599 |  1.4293  |  4.22979 |  24.7976 |  35.2707 |
| weighted_huber_shared_seq20_reg | W3       | 2022-06-20  | 2023-06-20 | 2024-02-22 | /Users/davidwilkinson/Desktop/ETS 2/runs/20260319_195212_a1b037 | weighted_huber |    70.659  | 10.4422  | 22.8314  |  92.7158 | 113.178  |
| weighted_huber_shared_seq20_reg | W4       | 2021-10-16  | 2022-10-16 | 2023-06-20 | /Users/davidwilkinson/Desktop/ETS 2/runs/20260319_195214_3636a0 | weighted_huber |   103.074  | 11.4151  | 27.1705  | 127.142  | 218.839  |

## Mean By Candidate

| candidate                       |   path_mse |      h1 |      h5 |     h20 |     h30 |
|:--------------------------------|-----------:|--------:|--------:|--------:|--------:|
| huber_shared_seq20_reg          |    53.4058 | 6.45964 | 15.8087 | 67.765  | 100.3   |
| weighted_huber_shared_seq20_reg |    53.7936 | 6.47668 | 15.947  | 68.0939 | 100.828 |
| mse_shared_seq20_reg            |    54.6061 | 6.2668  | 15.8362 | 69.3554 | 104.581 |
| weighted_mse_shared_seq20_reg   |    55.4301 | 6.32432 | 16.217  | 70.1915 | 105.585 |

## Best Candidate Per Window

| candidate              | window   | train_end   | val_end    | test_end   | run_dir                                                         | loss_type   |   path_mse |       h1 |       h5 |      h20 |      h30 |
|:-----------------------|:---------|:------------|:-----------|:-----------|:----------------------------------------------------------------|:------------|-----------:|---------:|---------:|---------:|---------:|
| huber_shared_seq20_reg | W1       | 2023-10-27  | 2024-10-26 | 2025-06-30 | /Users/davidwilkinson/Desktop/ETS 2/runs/20260319_195149_55ee1e | huber       |    22.9453 |  2.62117 |  9.54875 |  27.6949 |  35.9526 |
| huber_shared_seq20_reg | W2       | 2023-02-22  | 2024-02-22 | 2024-10-26 | /Users/davidwilkinson/Desktop/ETS 2/runs/20260319_195152_ae51e3 | huber       |    17.9647 |  1.41456 |  4.1137  |  24.3242 |  34.3057 |
| huber_shared_seq20_reg | W3       | 2022-06-20  | 2023-06-20 | 2024-02-22 | /Users/davidwilkinson/Desktop/ETS 2/runs/20260319_195154_edf602 | huber       |    69.8799 | 10.4121  | 22.6172  |  91.956  | 112.369  |
| huber_shared_seq20_reg | W4       | 2021-10-16  | 2022-10-16 | 2023-06-20 | /Users/davidwilkinson/Desktop/ETS 2/runs/20260319_195156_07e0f6 | huber       |   102.833  | 11.3908  | 26.9551  | 127.085  | 218.574  |

## Headline

- Best mean candidate: `huber_shared_seq20_reg` with path MSE `53.405818`.
- Baseline `mse_shared_seq20_reg` path MSE: `54.606079`.
- Improvement vs baseline: `2.20%`.
