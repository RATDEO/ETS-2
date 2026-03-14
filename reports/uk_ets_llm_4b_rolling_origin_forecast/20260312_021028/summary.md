# UK ETS Rolling-Origin Forecast Check

Generated: 2026-03-12T02:44:45.323250

Aim:
- Increase evidential confidence by testing the live 4B forecast winner on multiple disjoint UK test blocks.
- Expected outcome: if the LLM edge is real, it should beat base TSM on path MSE in most folds and accumulate materially more than 8 h20 trades / 5 h30 trades overall.

## Horizon Summary

- h20: 5 folds, 542 total test samples, 29 non-overlap trades
  LLM mean path MSE `27.870351` vs TSM `28.204165`
  LLM beat TSM on path MSE in `3/5` folds
  LLM mean net return `23.76%` vs TSM `24.65%`
  LLM mean net Sharpe `1.173` vs TSM `1.220`
  LLM beat TSM on net return in `2/5` folds
- h30: 5 folds, 542 total test samples, 21 non-overlap trades
  LLM mean path MSE `27.870351` vs TSM `28.204165`
  LLM beat TSM on path MSE in `3/5` folds
  LLM mean net return `19.19%` vs TSM `19.91%`
  LLM mean net Sharpe `0.933` vs TSM `1.029`
  LLM beat TSM on net return in `1/5` folds

## Fold Results

| fold                  | train_end   | val_end    | test_end   | run_dir                                                         | model                 |   path_mse |   n_test_samples |   horizon |   n_trades_non_overlap |   trade_rate_non_overlap |   net_total_return_offset_avg |   net_annualized_return_offset_avg |   net_sharpe_offset_avg |   net_max_drawdown_offset_avg |
|:----------------------|:------------|:-----------|:-----------|:----------------------------------------------------------------|:----------------------|-----------:|-----------------:|----------:|-----------------------:|-------------------------:|------------------------------:|-----------------------------------:|------------------------:|------------------------------:|
| 2023H2                | 2022-12-31  | 2023-06-30 | 2023-12-31 | /Users/davidwilkinson/Desktop/ETS 2/runs/20260312_021028_41ad8a | tsm                   |    70.8512 |               99 |        20 |                      5 |                        1 |                    -0.164864  |                         -0.350019  |               -0.672903 |                     0.209512  |
| 2023H2                | 2022-12-31  | 2023-06-30 | 2023-12-31 | /Users/davidwilkinson/Desktop/ETS 2/runs/20260312_021028_41ad8a | tsm                   |    70.8512 |               99 |        30 |                      4 |                        1 |                    -0.12329   |                         -0.240678  |               -1.44784  |                     0.0993991 |
| 2023H2                | 2022-12-31  | 2023-06-30 | 2023-12-31 | /Users/davidwilkinson/Desktop/ETS 2/runs/20260312_021028_41ad8a | TSM+LLM-COT-RF-HDELTA |    68.6113 |               99 |        20 |                      5 |                        1 |                    -0.170562  |                         -0.358951  |               -0.70734  |                     0.209512  |
| 2023H2                | 2022-12-31  | 2023-06-30 | 2023-12-31 | /Users/davidwilkinson/Desktop/ETS 2/runs/20260312_021028_41ad8a | TSM+LLM-COT-RF-HDELTA |    68.6113 |               99 |        30 |                      4 |                        1 |                    -0.12329   |                         -0.240678  |               -1.44784  |                     0.0993991 |
| 2024H1                | 2023-06-30  | 2023-12-31 | 2024-06-30 | /Users/davidwilkinson/Desktop/ETS 2/runs/20260312_021906_a19c17 | tsm                   |    17.8423 |              100 |        20 |                      5 |                        1 |                     0.300616  |                          1.07295   |                1.95379  |                     0.0483598 |
| 2024H1                | 2023-06-30  | 2023-12-31 | 2024-06-30 | /Users/davidwilkinson/Desktop/ETS 2/runs/20260312_021906_a19c17 | tsm                   |    17.8423 |              100 |        30 |                      4 |                        1 |                     0.353844  |                          1.23473   |                2.53001  |                     0.0194285 |
| 2024H1                | 2023-06-30  | 2023-12-31 | 2024-06-30 | /Users/davidwilkinson/Desktop/ETS 2/runs/20260312_021906_a19c17 | TSM+LLM-COT-RF-HDELTA |    17.7631 |              100 |        20 |                      5 |                        1 |                     0.322381  |                          1.15796   |                2.09103  |                     0.0483598 |
| 2024H1                | 2023-06-30  | 2023-12-31 | 2024-06-30 | /Users/davidwilkinson/Desktop/ETS 2/runs/20260312_021906_a19c17 | TSM+LLM-COT-RF-HDELTA |    17.7631 |              100 |        30 |                      4 |                        1 |                     0.34      |                          1.1852    |                2.38191  |                     0.0250884 |
| 2024H2                | 2023-12-31  | 2024-06-30 | 2024-12-31 | /Users/davidwilkinson/Desktop/ETS 2/runs/20260312_022747_2f56c6 | tsm                   |    11.0631 |              101 |        20 |                      6 |                        1 |                    -0.0494373 |                         -0.0883053 |               -0.710926 |                     0.111275  |
| 2024H2                | 2023-12-31  | 2024-06-30 | 2024-12-31 | /Users/davidwilkinson/Desktop/ETS 2/runs/20260312_022747_2f56c6 | tsm                   |    11.0631 |              101 |        30 |                      4 |                        1 |                    -0.0911428 |                         -0.160864  |               -0.958007 |                     0.123871  |
| 2024H2                | 2023-12-31  | 2024-06-30 | 2024-12-31 | /Users/davidwilkinson/Desktop/ETS 2/runs/20260312_022747_2f56c6 | TSM+LLM-COT-RF-HDELTA |    11.7355 |              101 |        20 |                      6 |                        1 |                    -0.0772157 |                         -0.15773   |               -0.922384 |                     0.120814  |
| 2024H2                | 2023-12-31  | 2024-06-30 | 2024-12-31 | /Users/davidwilkinson/Desktop/ETS 2/runs/20260312_022747_2f56c6 | TSM+LLM-COT-RF-HDELTA |    11.7355 |              101 |        30 |                      4 |                        1 |                    -0.106255  |                         -0.207604  |               -1.18634  |                     0.130669  |
| 2025H1                | 2024-06-30  | 2024-12-31 | 2025-06-30 | /Users/davidwilkinson/Desktop/ETS 2/runs/20260312_023621_733f60 | tsm                   |    18.5379 |               98 |        20 |                      5 |                        1 |                     0.626321  |                          2.82709   |                3.18093  |                     0.0501498 |
| 2025H1                | 2024-06-30  | 2024-12-31 | 2025-06-30 | /Users/davidwilkinson/Desktop/ETS 2/runs/20260312_023621_733f60 | tsm                   |    18.5379 |               98 |        30 |                      4 |                        1 |                     0.261842  |                          1.05502   |                2.43969  |                     0.0564076 |
| 2025H1                | 2024-06-30  | 2024-12-31 | 2025-06-30 | /Users/davidwilkinson/Desktop/ETS 2/runs/20260312_023621_733f60 | TSM+LLM-COT-RF-HDELTA |    19.3889 |               98 |        20 |                      5 |                        1 |                     0.586826  |                          2.66051   |                3.01965  |                     0.0631027 |
| 2025H1                | 2024-06-30  | 2024-12-31 | 2025-06-30 | /Users/davidwilkinson/Desktop/ETS 2/runs/20260312_023621_733f60 | TSM+LLM-COT-RF-HDELTA |    19.3889 |               98 |        30 |                      4 |                        1 |                     0.231499  |                          0.922266  |                2.25475  |                     0.0669674 |
| 2025H2_2026Q1_holdout | 2024-06-30  | 2025-06-30 | 2026-03-04 | /Users/davidwilkinson/Desktop/ETS 2/runs/20260311_135144_bb26cf | tsm                   |    22.7264 |              144 |        20 |                      8 |                        1 |                     0.519939  |                          1.15311   |                2.35072  |                     0.0587584 |
| 2025H2_2026Q1_holdout | 2024-06-30  | 2025-06-30 | 2026-03-04 | /Users/davidwilkinson/Desktop/ETS 2/runs/20260311_135144_bb26cf | tsm                   |    22.7264 |              144 |        30 |                      5 |                        1 |                     0.594474  |                          1.27831   |                2.58097  |                     0.0202192 |
| 2025H2_2026Q1_holdout | 2024-06-30  | 2025-06-30 | 2026-03-04 | /Users/davidwilkinson/Desktop/ETS 2/runs/20260311_135144_bb26cf | TSM+LLM-COT-RF-HDELTA |    21.8529 |              144 |        20 |                      8 |                        1 |                     0.526453  |                          1.1722    |                2.384    |                     0.0587584 |
| 2025H2_2026Q1_holdout | 2024-06-30  | 2025-06-30 | 2026-03-04 | /Users/davidwilkinson/Desktop/ETS 2/runs/20260311_135144_bb26cf | TSM+LLM-COT-RF-HDELTA |    21.8529 |              144 |        30 |                      5 |                        1 |                     0.617702  |                          1.33626   |                2.66202  |                     0.0171465 |

Interpretation:
- This is more informative than a single terminal holdout because the test blocks are disjoint in time.
- It is still not a production-grade trading study; the next step after this is a longer rolling history or a monthly rebalance study with cost/slippage stress tests.
