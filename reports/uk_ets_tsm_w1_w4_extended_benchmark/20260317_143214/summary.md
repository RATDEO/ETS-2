# Extended W1-W4 TSM Candidate Benchmark

Generated: 2026-03-17T14:34:47.572078

Data dir forced to `uk_ets/Data_auto_uk` to use the UK-specific weather and auction proxy packs.

## Results

| candidate                      | window   | train_end   | val_end    | test_end   | run_dir                                                         |   path_mse |       h1 |        h5 |      h20 |      h30 |
|:-------------------------------|:---------|:------------|:-----------|:-----------|:----------------------------------------------------------------|-----------:|---------:|----------:|---------:|---------:|
| live_auto_uk                   | W1       | 2023-10-27  | 2024-10-26 | 2025-06-30 | /Users/davidwilkinson/Desktop/ETS 2/runs/20260317_143214_4a56b0 |    19.3027 |  2.47884 |   8.52204 |  22.9989 |  32.474  |
| live_auto_uk                   | W2       | 2023-02-22  | 2024-02-22 | 2024-10-26 | /Users/davidwilkinson/Desktop/ETS 2/runs/20260317_143218_4d6f58 |    22.3824 |  1.44316 |   4.44662 |  31.7191 |  42.5939 |
| live_auto_uk                   | W3       | 2022-06-20  | 2023-06-20 | 2024-02-22 | /Users/davidwilkinson/Desktop/ETS 2/runs/20260317_143220_ddae64 |    97.6542 |  9.4751  |  21.7134  | 136.676  | 193.508  |
| live_auto_uk                   | W4       | 2021-10-16  | 2022-10-16 | 2023-06-20 | /Users/davidwilkinson/Desktop/ETS 2/runs/20260317_143222_8c8b8a |   105.966  | 10.1602  |  23.6144  | 135.654  | 242.566  |
| live_short_ret_s10_l5          | W1       | 2023-10-27  | 2024-10-26 | 2025-06-30 | /Users/davidwilkinson/Desktop/ETS 2/runs/20260317_143223_0c9e93 |    22.6538 |  2.63798 |  10.8429  |  29.9202 |  34.6741 |
| live_short_ret_s10_l5          | W2       | 2023-02-22  | 2024-02-22 | 2024-10-26 | /Users/davidwilkinson/Desktop/ETS 2/runs/20260317_143231_1cf708 |    27.7528 |  1.84499 |   6.44094 |  38.0176 |  53.209  |
| live_short_ret_s10_l5          | W3       | 2022-06-20  | 2023-06-20 | 2024-02-22 | /Users/davidwilkinson/Desktop/ETS 2/runs/20260317_143237_76c227 |   101.386  |  9.22336 |  28.3236  | 137.86   | 198.809  |
| live_short_ret_s10_l5          | W4       | 2021-10-16  | 2022-10-16 | 2023-06-20 | /Users/davidwilkinson/Desktop/ETS 2/runs/20260317_143241_1f34cf |   144.854  | 11.9202  |  41.8177  | 194.238  | 308.557  |
| live_short_price_s10_l5        | W1       | 2023-10-27  | 2024-10-26 | 2025-06-30 | /Users/davidwilkinson/Desktop/ETS 2/runs/20260317_143245_976e05 |    48.6448 |  5.39755 |  18.2588  |  41.6513 |  61.6504 |
| live_short_price_s10_l5        | W2       | 2023-02-22  | 2024-02-22 | 2024-10-26 | /Users/davidwilkinson/Desktop/ETS 2/runs/20260317_143252_78cf5c |   133.782  |  4.2662  |  19.8558  | 111.368  | 262.794  |
| live_short_price_s10_l5        | W3       | 2022-06-20  | 2023-06-20 | 2024-02-22 | /Users/davidwilkinson/Desktop/ETS 2/runs/20260317_143258_36072c |   161.604  | 13.5702  |  37.99    | 167.44   | 362.248  |
| live_short_price_s10_l5        | W4       | 2021-10-16  | 2022-10-16 | 2023-06-20 | /Users/davidwilkinson/Desktop/ETS 2/runs/20260317_143302_4ec847 |   441.728  | 55.597   |  78.1841  |  93.6062 | 758.284  |
| live_mid_ret_s20_l10           | W1       | 2023-10-27  | 2024-10-26 | 2025-06-30 | /Users/davidwilkinson/Desktop/ETS 2/runs/20260317_143305_6435cc |    19.7024 |  2.29865 |   9.25162 |  25.5447 |  24.1503 |
| live_mid_ret_s20_l10           | W2       | 2023-02-22  | 2024-02-22 | 2024-10-26 | /Users/davidwilkinson/Desktop/ETS 2/runs/20260317_143313_526a42 |    23.6298 |  1.58954 |   5.37973 |  34.9603 |  42.9306 |
| live_mid_ret_s20_l10           | W3       | 2022-06-20  | 2023-06-20 | 2024-02-22 | /Users/davidwilkinson/Desktop/ETS 2/runs/20260317_143319_8fa644 |   123.586  |  9.87906 |  27.176   | 174.742  | 238.509  |
| live_mid_ret_s20_l10           | W4       | 2021-10-16  | 2022-10-16 | 2023-06-20 | /Users/davidwilkinson/Desktop/ETS 2/runs/20260317_143324_c52e00 |    92.6145 | 10.9752  |  25.0892  | 139.762  | 154.309  |
| live_mid_price_s20_l10         | W1       | 2023-10-27  | 2024-10-26 | 2025-06-30 | /Users/davidwilkinson/Desktop/ETS 2/runs/20260317_143327_e8b6e5 |    42.1224 |  7.06111 |  23.4972  |  55.7218 |  60.7238 |
| live_mid_price_s20_l10         | W2       | 2023-02-22  | 2024-02-22 | 2024-10-26 | /Users/davidwilkinson/Desktop/ETS 2/runs/20260317_143333_b86c0f |    96.3216 |  4.04237 |  34.2532  | 140.455  | 184.94   |
| live_mid_price_s20_l10         | W3       | 2022-06-20  | 2023-06-20 | 2024-02-22 | /Users/davidwilkinson/Desktop/ETS 2/runs/20260317_143337_9d1014 |   124.51   | 12.8143  |  72.4613  | 238.225  | 257.306  |
| live_mid_price_s20_l10         | W4       | 2021-10-16  | 2022-10-16 | 2023-06-20 | /Users/davidwilkinson/Desktop/ETS 2/runs/20260317_143341_b51db2 |   858.7    | 32.9412  | 736.327   | 433.285  | 442.373  |
| compact_short_ret_s10_l5       | W1       | 2023-10-27  | 2024-10-26 | 2025-06-30 | /Users/davidwilkinson/Desktop/ETS 2/runs/20260317_143344_ca6366 |    22.051  |  2.70756 |   9.96521 |  29.6356 |  31.0223 |
| compact_short_ret_s10_l5       | W2       | 2023-02-22  | 2024-02-22 | 2024-10-26 | /Users/davidwilkinson/Desktop/ETS 2/runs/20260317_143350_ad0a25 |    26.4559 |  1.77717 |   5.75031 |  39.0062 |  48.8798 |
| compact_short_ret_s10_l5       | W3       | 2022-06-20  | 2023-06-20 | 2024-02-22 | /Users/davidwilkinson/Desktop/ETS 2/runs/20260317_143355_57b6c7 |    95.032  | 13.1286  |  25.8534  | 137.759  | 179.669  |
| compact_short_ret_s10_l5       | W4       | 2021-10-16  | 2022-10-16 | 2023-06-20 | /Users/davidwilkinson/Desktop/ETS 2/runs/20260317_143358_4e031b |   117.441  | 15.8048  |  28.9276  | 174.721  | 212.406  |
| tuned_feature_short_ret_s10_l5 | W1       | 2023-10-27  | 2024-10-26 | 2025-06-30 | /Users/davidwilkinson/Desktop/ETS 2/runs/20260317_143401_ae6631 |    22.2197 |  2.43569 |   9.53763 |  28.5406 |  35.2231 |
| tuned_feature_short_ret_s10_l5 | W2       | 2023-02-22  | 2024-02-22 | 2024-10-26 | /Users/davidwilkinson/Desktop/ETS 2/runs/20260317_143409_27ddfc |    27.485  |  1.406   |   4.91165 |  40.9168 |  54.7341 |
| tuned_feature_short_ret_s10_l5 | W3       | 2022-06-20  | 2023-06-20 | 2024-02-22 | /Users/davidwilkinson/Desktop/ETS 2/runs/20260317_143416_3a5ac2 |   100.107  |  9.92411 |  24.1645  | 141.032  | 204.785  |
| tuned_feature_short_ret_s10_l5 | W4       | 2021-10-16  | 2022-10-16 | 2023-06-20 | /Users/davidwilkinson/Desktop/ETS 2/runs/20260317_143421_09a923 |   143.025  |  8.75424 |  22.5226  | 227.096  | 298.001  |
| tuned_feature_mid_ret_s20_l10  | W1       | 2023-10-27  | 2024-10-26 | 2025-06-30 | /Users/davidwilkinson/Desktop/ETS 2/runs/20260317_143424_463c4e |    19.6191 |  2.39182 |   9.03312 |  24.6706 |  27.6891 |
| tuned_feature_mid_ret_s20_l10  | W2       | 2023-02-22  | 2024-02-22 | 2024-10-26 | /Users/davidwilkinson/Desktop/ETS 2/runs/20260317_143432_d16d43 |    30.4398 |  1.47413 |   5.30667 |  41.962  |  62.1257 |
| tuned_feature_mid_ret_s20_l10  | W3       | 2022-06-20  | 2023-06-20 | 2024-02-22 | /Users/davidwilkinson/Desktop/ETS 2/runs/20260317_143439_885394 |   122.274  | 11.4638  |  24.3603  | 161.322  | 269.898  |
| tuned_feature_mid_ret_s20_l10  | W4       | 2021-10-16  | 2022-10-16 | 2023-06-20 | /Users/davidwilkinson/Desktop/ETS 2/runs/20260317_143444_95e6d2 |   144.721  | 12.5315  |  27.2608  | 221.824  | 286.533  |

## Mean By Candidate

| candidate                      |   path_mse |       h1 |       h5 |      h20 |     h30 |
|:-------------------------------|-----------:|---------:|---------:|---------:|--------:|
| live_auto_uk                   |    61.3262 |  5.88933 |  14.5741 |  81.7619 | 127.785 |
| live_mid_ret_s20_l10           |    64.8833 |  6.18561 |  16.7241 |  93.7523 | 114.975 |
| compact_short_ret_s10_l5       |    65.2451 |  8.35453 |  17.6241 |  95.2804 | 117.994 |
| tuned_feature_short_ret_s10_l5 |    73.2091 |  5.63001 |  15.2841 | 109.396  | 148.186 |
| live_short_ret_s10_l5          |    74.1615 |  6.40663 |  21.8563 | 100.009  | 148.812 |
| tuned_feature_mid_ret_s20_l10  |    79.2634 |  6.96532 |  16.4902 | 112.444  | 161.562 |
| live_short_price_s10_l5        |   196.44   | 19.7077  |  38.5722 | 103.516  | 361.244 |
| live_mid_price_s20_l10         |   280.413  | 14.2148  | 216.635  | 216.922  | 236.336 |

## Best Candidate Per Window

| candidate                | window   | train_end   | val_end    | test_end   | run_dir                                                         |   path_mse |       h1 |       h5 |      h20 |      h30 |
|:-------------------------|:---------|:------------|:-----------|:-----------|:----------------------------------------------------------------|-----------:|---------:|---------:|---------:|---------:|
| live_auto_uk             | W1       | 2023-10-27  | 2024-10-26 | 2025-06-30 | /Users/davidwilkinson/Desktop/ETS 2/runs/20260317_143214_4a56b0 |    19.3027 |  2.47884 |  8.52204 |  22.9989 |  32.474  |
| live_auto_uk             | W2       | 2023-02-22  | 2024-02-22 | 2024-10-26 | /Users/davidwilkinson/Desktop/ETS 2/runs/20260317_143218_4d6f58 |    22.3824 |  1.44316 |  4.44662 |  31.7191 |  42.5939 |
| compact_short_ret_s10_l5 | W3       | 2022-06-20  | 2023-06-20 | 2024-02-22 | /Users/davidwilkinson/Desktop/ETS 2/runs/20260317_143355_57b6c7 |    95.032  | 13.1286  | 25.8534  | 137.759  | 179.669  |
| live_mid_ret_s20_l10     | W4       | 2021-10-16  | 2022-10-16 | 2023-06-20 | /Users/davidwilkinson/Desktop/ETS 2/runs/20260317_143324_c52e00 |    92.6145 | 10.9752  | 25.0892  | 139.762  | 154.309  |
