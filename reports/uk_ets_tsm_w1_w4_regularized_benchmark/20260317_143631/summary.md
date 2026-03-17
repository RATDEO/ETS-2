# Regularized W1-W4 TSM Benchmark

Generated: 2026-03-17T14:37:30.218935

Data dir forced to `uk_ets/Data_auto_uk`.

## Results

| candidate                   | window   | train_end   | val_end    | test_end   | run_dir                                                         |   path_mse |       h1 |       h5 |      h20 |      h30 |
|:----------------------------|:---------|:------------|:-----------|:-----------|:----------------------------------------------------------------|-----------:|---------:|---------:|---------:|---------:|
| baseline_live_auto_uk       | W1       | 2023-10-27  | 2024-10-26 | 2025-06-30 | /Users/davidwilkinson/Desktop/ETS 2/runs/20260317_143631_bc0cf4 |    19.3027 |  2.47884 |  8.52204 |  22.9989 |  32.474  |
| baseline_live_auto_uk       | W2       | 2023-02-22  | 2024-02-22 | 2024-10-26 | /Users/davidwilkinson/Desktop/ETS 2/runs/20260317_143639_b278a8 |    22.3824 |  1.44316 |  4.44662 |  31.7191 |  42.5939 |
| baseline_live_auto_uk       | W3       | 2022-06-20  | 2023-06-20 | 2024-02-22 | /Users/davidwilkinson/Desktop/ETS 2/runs/20260317_143642_584550 |    97.6542 |  9.4751  | 21.7134  | 136.676  | 193.508  |
| baseline_live_auto_uk       | W4       | 2021-10-16  | 2022-10-16 | 2023-06-20 | /Users/davidwilkinson/Desktop/ETS 2/runs/20260317_143644_54f084 |   105.966  | 10.1602  | 23.6144  | 135.654  | 242.566  |
| shared_lr1e3_e40_pat8       | W1       | 2023-10-27  | 2024-10-26 | 2025-06-30 | /Users/davidwilkinson/Desktop/ETS 2/runs/20260317_143646_c875f8 |    23.7427 |  2.59257 |  9.47688 |  28.5499 |  38.9766 |
| shared_lr1e3_e40_pat8       | W2       | 2023-02-22  | 2024-02-22 | 2024-10-26 | /Users/davidwilkinson/Desktop/ETS 2/runs/20260317_143648_3842f9 |    19.7078 |  1.50387 |  4.50835 |  26.42   |  37.0407 |
| shared_lr1e3_e40_pat8       | W3       | 2022-06-20  | 2023-06-20 | 2024-02-22 | /Users/davidwilkinson/Desktop/ETS 2/runs/20260317_143650_b33e55 |    70.2512 |  9.62144 | 22.65    |  90.6596 | 119.639  |
| shared_lr1e3_e40_pat8       | W4       | 2021-10-16  | 2022-10-16 | 2023-06-20 | /Users/davidwilkinson/Desktop/ETS 2/runs/20260317_143652_8d7b93 |   104.805  | 11.3519  | 26.7268  | 131.879  | 222.86   |
| shared_lr5e4_e60_pat10      | W1       | 2023-10-27  | 2024-10-26 | 2025-06-30 | /Users/davidwilkinson/Desktop/ETS 2/runs/20260317_143654_9eb555 |    24.712  |  2.64859 |  9.7501  |  29.3238 |  41.0525 |
| shared_lr5e4_e60_pat10      | W2       | 2023-02-22  | 2024-02-22 | 2024-10-26 | /Users/davidwilkinson/Desktop/ETS 2/runs/20260317_143658_45d4d5 |    20.3298 |  1.52023 |  4.68472 |  27.0026 |  38.2744 |
| shared_lr5e4_e60_pat10      | W3       | 2022-06-20  | 2023-06-20 | 2024-02-22 | /Users/davidwilkinson/Desktop/ETS 2/runs/20260317_143701_0cafee |    76.0628 |  9.90098 | 24.0962  |  97.3723 | 129.304  |
| shared_lr5e4_e60_pat10      | W4       | 2021-10-16  | 2022-10-16 | 2023-06-20 | /Users/davidwilkinson/Desktop/ETS 2/runs/20260317_143703_bbdd04 |   107.258  | 11.4916  | 27.7147  | 134.337  | 226.765  |
| shared_lr1e3_dropout05_wd02 | W1       | 2023-10-27  | 2024-10-26 | 2025-06-30 | /Users/davidwilkinson/Desktop/ETS 2/runs/20260317_143705_459592 |    23.7343 |  2.59219 |  9.47455 |  28.5422 |  38.9594 |
| shared_lr1e3_dropout05_wd02 | W2       | 2023-02-22  | 2024-02-22 | 2024-10-26 | /Users/davidwilkinson/Desktop/ETS 2/runs/20260317_143707_0969b2 |    19.7021 |  1.50372 |  4.50719 |  26.414  |  37.0306 |
| shared_lr1e3_dropout05_wd02 | W3       | 2022-06-20  | 2023-06-20 | 2024-02-22 | /Users/davidwilkinson/Desktop/ETS 2/runs/20260317_143709_ad28c5 |    70.2063 |  9.62048 | 22.6419  |  90.6118 | 119.532  |
| shared_lr1e3_dropout05_wd02 | W4       | 2021-10-16  | 2022-10-16 | 2023-06-20 | /Users/davidwilkinson/Desktop/ETS 2/runs/20260317_143711_27841e |   104.782  | 11.3508  | 26.7213  | 131.854  | 222.8    |
| shared_lr5e4_dropout05_wd02 | W1       | 2023-10-27  | 2024-10-26 | 2025-06-30 | /Users/davidwilkinson/Desktop/ETS 2/runs/20260317_143712_3bc7d3 |    24.7024 |  2.64814 |  9.74751 |  29.3153 |  41.032  |
| shared_lr5e4_dropout05_wd02 | W2       | 2023-02-22  | 2024-02-22 | 2024-10-26 | /Users/davidwilkinson/Desktop/ETS 2/runs/20260317_143715_cd7f7c |    20.3243 |  1.52007 |  4.68352 |  26.9968 |  38.2642 |
| shared_lr5e4_dropout05_wd02 | W3       | 2022-06-20  | 2023-06-20 | 2024-02-22 | /Users/davidwilkinson/Desktop/ETS 2/runs/20260317_143718_871b02 |    76.0129 |  9.8998  | 24.0874  |  97.3168 | 129.189  |
| shared_lr5e4_dropout05_wd02 | W4       | 2021-10-16  | 2022-10-16 | 2023-06-20 | /Users/davidwilkinson/Desktop/ETS 2/runs/20260317_143720_4ace97 |   107.238  | 11.4908  | 27.71    | 134.316  | 226.716  |
| shared_lr1e3_compact12      | W1       | 2023-10-27  | 2024-10-26 | 2025-06-30 | /Users/davidwilkinson/Desktop/ETS 2/runs/20260317_143722_3ef2ef |    24.0529 |  2.59011 |  9.51572 |  29.0139 |  39.6643 |
| shared_lr1e3_compact12      | W2       | 2023-02-22  | 2024-02-22 | 2024-10-26 | /Users/davidwilkinson/Desktop/ETS 2/runs/20260317_143725_9c8ea4 |    19.9003 |  1.50481 |  4.53733 |  26.7202 |  37.4332 |
| shared_lr1e3_compact12      | W3       | 2022-06-20  | 2023-06-20 | 2024-02-22 | /Users/davidwilkinson/Desktop/ETS 2/runs/20260317_143727_904a88 |    70.9777 |  9.60165 | 22.834   |  91.9657 | 120.66   |
| shared_lr1e3_compact12      | W4       | 2021-10-16  | 2022-10-16 | 2023-06-20 | /Users/davidwilkinson/Desktop/ETS 2/runs/20260317_143728_797730 |   104.502  | 11.3756  | 26.8533  | 131.24   | 222.096  |

## Mean By Candidate

| candidate                   |   path_mse |      h1 |      h5 |     h20 |     h30 |
|:----------------------------|-----------:|--------:|--------:|--------:|--------:|
| shared_lr1e3_dropout05_wd02 |    54.6061 | 6.2668  | 15.8362 | 69.3554 | 104.581 |
| shared_lr1e3_e40_pat8       |    54.6268 | 6.26745 | 15.8405 | 69.3771 | 104.629 |
| shared_lr1e3_compact12      |    54.8582 | 6.26805 | 15.9351 | 69.735  | 104.963 |
| shared_lr5e4_dropout05_wd02 |    57.0695 | 6.38969 | 16.5571 | 71.9862 | 108.8   |
| shared_lr5e4_e60_pat10      |    57.0907 | 6.39036 | 16.5614 | 72.009  | 108.849 |
| baseline_live_auto_uk       |    61.3262 | 5.88933 | 14.5741 | 81.7619 | 127.785 |
