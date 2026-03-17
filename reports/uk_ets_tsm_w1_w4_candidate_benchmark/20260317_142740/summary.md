# W1-W4 TSM Candidate Benchmark

Generated: 2026-03-17T14:28:44.824454

Data dir forced to `uk_ets/Data_auto_uk` to use the UK-specific weather and auction proxy packs.

## Results

| candidate                        | window   | train_end   | val_end    | test_end   | run_dir                                                         |   path_mse |       h1 |       h5 |      h20 |      h30 |
|:---------------------------------|:---------|:------------|:-----------|:-----------|:----------------------------------------------------------------|-----------:|---------:|---------:|---------:|---------:|
| live_auto_uk                     | W1       | 2023-10-27  | 2024-10-26 | 2025-06-30 | /Users/davidwilkinson/Desktop/ETS 2/runs/20260317_142740_260777 |    19.3027 |  2.47884 |  8.52204 |  22.9989 |  32.474  |
| live_auto_uk                     | W2       | 2023-02-22  | 2024-02-22 | 2024-10-26 | /Users/davidwilkinson/Desktop/ETS 2/runs/20260317_142746_5c58a7 |    22.3824 |  1.44316 |  4.44662 |  31.7191 |  42.5939 |
| live_auto_uk                     | W3       | 2022-06-20  | 2023-06-20 | 2024-02-22 | /Users/davidwilkinson/Desktop/ETS 2/runs/20260317_142749_819826 |    97.6542 |  9.4751  | 21.7134  | 136.676  | 193.508  |
| live_auto_uk                     | W4       | 2021-10-16  | 2022-10-16 | 2023-06-20 | /Users/davidwilkinson/Desktop/ETS 2/runs/20260317_142751_9a377b |   105.966  | 10.1602  | 23.6144  | 135.654  | 242.566  |
| tuned_auto_uk                    | W1       | 2023-10-27  | 2024-10-26 | 2025-06-30 | /Users/davidwilkinson/Desktop/ETS 2/runs/20260317_142753_696095 |    19.6191 |  2.39182 |  9.03312 |  24.6706 |  27.6891 |
| tuned_auto_uk                    | W2       | 2023-02-22  | 2024-02-22 | 2024-10-26 | /Users/davidwilkinson/Desktop/ETS 2/runs/20260317_142803_3bebd9 |    30.4398 |  1.47413 |  5.30667 |  41.962  |  62.1257 |
| tuned_auto_uk                    | W3       | 2022-06-20  | 2023-06-20 | 2024-02-22 | /Users/davidwilkinson/Desktop/ETS 2/runs/20260317_142810_573c0f |   122.274  | 11.4638  | 24.3603  | 161.322  | 269.898  |
| tuned_auto_uk                    | W4       | 2021-10-16  | 2022-10-16 | 2023-06-20 | /Users/davidwilkinson/Desktop/ETS 2/runs/20260317_142816_254842 |   144.721  | 12.5315  | 27.2608  | 221.824  | 286.533  |
| hybrid_tuned_model_live_features | W1       | 2023-10-27  | 2024-10-26 | 2025-06-30 | /Users/davidwilkinson/Desktop/ETS 2/runs/20260317_142820_96575c |    19.7024 |  2.29865 |  9.25162 |  25.5447 |  24.1503 |
| hybrid_tuned_model_live_features | W2       | 2023-02-22  | 2024-02-22 | 2024-10-26 | /Users/davidwilkinson/Desktop/ETS 2/runs/20260317_142828_5e4d37 |    23.6298 |  1.58954 |  5.37973 |  34.9603 |  42.9306 |
| hybrid_tuned_model_live_features | W3       | 2022-06-20  | 2023-06-20 | 2024-02-22 | /Users/davidwilkinson/Desktop/ETS 2/runs/20260317_142835_697048 |   123.586  |  9.87906 | 27.176   | 174.742  | 238.509  |
| hybrid_tuned_model_live_features | W4       | 2021-10-16  | 2022-10-16 | 2023-06-20 | /Users/davidwilkinson/Desktop/ETS 2/runs/20260317_142840_fe788c |    92.6145 | 10.9752  | 25.0892  | 139.762  | 154.309  |

## Mean By Candidate

| candidate                        |   path_mse |      h1 |      h5 |      h20 |     h30 |
|:---------------------------------|-----------:|--------:|--------:|---------:|--------:|
| hybrid_tuned_model_live_features |    64.8833 | 6.18561 | 16.7241 |  93.7523 | 114.975 |
| live_auto_uk                     |    61.3262 | 5.88933 | 14.5741 |  81.7619 | 127.785 |
| tuned_auto_uk                    |    79.2634 | 6.96532 | 16.4902 | 112.444  | 161.562 |
