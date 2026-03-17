# TSM Live Policy Verification And Financial Conversion

## Verification
- Both compared runs use the same 103 test windows, the same `y_true`, the same `eval_indices`, and the same raw `TSM` base predictions.
- `baseline_heuristic`: base path MSE `4.202436`, refined path MSE `4.174480`, `h20` `4.112015 -> 4.027897`, `h30` `9.131735 -> 9.125793`.
- `learned_gbdt`: base path MSE `4.202436`, refined path MSE `3.729913`, `h20` `4.112015 -> 3.449006`, `h30` `9.131735 -> 7.891961`.

## Financial Conversion Test
- Method: direct active-window marked-to-market backtest on the verified test predictions only.
- Trading rule: sign of predicted horizon return at `h20` or `h30`, with `1`-day execution lag, hold `= horizon`, `10` bps transaction cost, `5` bps slippage, max concurrent positions `= horizon`, active-window trim enabled.
- This is a clean conversion test of the verified winner, but it is still a single-holdout backtest, not a full rolling scientific evaluation.

### h20
- `baseline_heuristic`: executed trades `103`, total return `16.09%`, annualized return `35.76%`, Sharpe `1.964`, max drawdown `3.53%`, directional accuracy `0.903`.
- `learned_gbdt`: executed trades `103`, total return `16.09%`, annualized return `35.76%`, Sharpe `1.964`, max drawdown `3.53%`, directional accuracy `0.903`.
- `raw_tsm`: executed trades `103`, total return `16.09%`, annualized return `35.76%`, Sharpe `1.964`, max drawdown `3.53%`, directional accuracy `0.903`.
- Signal disagreement vs raw TSM for `baseline_heuristic`: `0` windows (`0.00%`).
- Signal disagreement vs raw TSM for `learned_gbdt`: `0` windows (`0.00%`).

### h30
- `baseline_heuristic`: executed trades `101`, total return `17.48%`, annualized return `36.33%`, Sharpe `2.134`, max drawdown `3.48%`, directional accuracy `0.971`.
- `learned_gbdt`: executed trades `101`, total return `17.48%`, annualized return `36.33%`, Sharpe `2.134`, max drawdown `3.48%`, directional accuracy `0.971`.
- `raw_tsm`: executed trades `101`, total return `17.48%`, annualized return `36.33%`, Sharpe `2.134`, max drawdown `3.48%`, directional accuracy `0.971`.
- Signal disagreement vs raw TSM for `baseline_heuristic`: `0` windows (`0.00%`).
- Signal disagreement vs raw TSM for `learned_gbdt`: `0` windows (`0.00%`).

