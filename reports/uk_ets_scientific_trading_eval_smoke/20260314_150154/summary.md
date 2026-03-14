# UK ETS Scientific Trading Evaluation

Generated: 2026-03-14T15:01:54.544296

Protocol choices:
- Primary trading horizon: `h20`
- Execution lag: `1` trading day(s) after the forecast decision date
- Holding period: `20` close-to-close sessions
- Capital allocation: one staggered trade slot per horizon day, capped at `20` concurrent positions
- Transaction cost: `10.0` bps per side
- Slippage stress: `5.0` additional bps per side

LLM training-cutoff note:
- Cutoff used: `2025-01-01`
- Prediction sets whose evaluation dates lie entirely on or after this cutoff are treated as materially cleaner with respect to possible LLM pretraining contamination.
- Prediction sets earlier than this remain scientifically usable for research, but the contamination question is more ambiguous.

## Fold Results

| fold                  | train_end   | val_end    | test_end   | run_dir                                                         | model                 |   primary_horizon |   embargo_days |   n_raw_test_windows |   n_eval_windows_after_embargo |   path_mse_embargoed |   n_days |   n_trade_decisions |   n_executed_trades |   execution_lag_days |   hold_days |   max_concurrent_positions |   transaction_cost_bps |   slippage_bps |   mean_daily_return |   daily_volatility |   daily_sharpe_annualized |   total_return |   annualized_return |   max_drawdown |   avg_net_exposure |   avg_gross_exposure |   trade_hit_rate | llm_cutoff_status   | llm_cutoff_date   |   llm_n_pre_cutoff |   llm_n_post_cutoff | llm_cutoff_note                                                                                                                                                                                                                    |
|:----------------------|:------------|:-----------|:-----------|:----------------------------------------------------------------|:----------------------|------------------:|---------------:|---------------------:|-------------------------------:|---------------------:|---------:|--------------------:|--------------------:|---------------------:|------------:|---------------------------:|-----------------------:|---------------:|--------------------:|-------------------:|--------------------------:|---------------:|--------------------:|---------------:|-------------------:|---------------------:|-----------------:|:--------------------|:------------------|-------------------:|--------------------:|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| 2025H2_2026Q1_holdout | 2024-06-30  | 2025-06-30 | 2026-03-04 | /Users/davidwilkinson/Desktop/ETS 2/runs/20260313_020215_90a3c5 | tsm                   |                20 |             30 |                  144 |                            122 |              26.2143 |     1233 |                 122 |                 122 |                    1 |          20 |                         20 |                     10 |              5 |         0.000285139 |         0.00434349 |                   1.04212 |       0.404964 |           0.0719629 |      0.0628466 |          0.0583131 |            0.0989457 |         0.721311 | post_cutoff_only    | 2025-01-01        |                  0 |                 122 | All prediction-set dates are on or after 2025-01-01. Relative to the assumed LLM training cutoff around early 2025, this segment is the cleaner regime for assessing whether results could reflect memorized pretraining exposure. |
| 2025H2_2026Q1_holdout | 2024-06-30  | 2025-06-30 | 2026-03-04 | /Users/davidwilkinson/Desktop/ETS 2/runs/20260313_020215_90a3c5 | TSM+LLM-COT-RF-HDELTA |                20 |             30 |                  144 |                            122 |              24.8687 |     1233 |                 122 |                 122 |                    1 |          20 |                         20 |                     10 |              5 |         0.000306764 |         0.0043826  |                   1.11115 |       0.44262  |           0.0777733 |      0.0628466 |          0.0609895 |            0.0989457 |         0.737705 | post_cutoff_only    | 2025-01-01        |                  0 |                 122 | All prediction-set dates are on or after 2025-01-01. Relative to the assumed LLM training cutoff around early 2025, this segment is the cleaner regime for assessing whether results could reflect memorized pretraining exposure. |

## Model Summary

- `tsm`: mean embargoed path MSE `26.214286`, mean total return `40.50%`, mean annualized return `7.20%`, mean Sharpe `1.042`, mean max drawdown `6.28%`
  LLM cutoff statuses: post_cutoff_only
- `TSM+LLM-COT-RF-HDELTA`: mean embargoed path MSE `24.868659`, mean total return `44.26%`, mean annualized return `7.78%`, mean Sharpe `1.111`, mean max drawdown `6.28%`
  LLM cutoff statuses: post_cutoff_only

Interpretation:
- These results are stricter than the earlier offset-averaged horizon backtests because they use embargoed forecast windows and a daily marked-to-market portfolio construction.
- They are still historical evidence. The first confirmatory economic test remains prospective paper trading after the strategy freeze.
