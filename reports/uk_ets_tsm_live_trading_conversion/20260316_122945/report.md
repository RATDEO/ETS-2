# TSM Live Trading Conversion Report

## Setup
- Reran `baseline_regime_specific` and `learned_gbdt_regime` with `llm.export_val_predictions=true`.
- Used the saved validation predictions to calibrate trading rules and then applied them to the common test holdout.
- Costs: `10` bps transaction cost plus `5` bps slippage, `1`-day execution lag, active-window MTM.

## Verification
- `same_val_dates`: `True`
- `same_test_dates`: `True`
- `same_val_base_pred`: `True`
- `same_test_base_pred`: `True`

## Best Policies
- `learned_gbdt` `h20` best policy `linear_size` with test Sharpe `2.006`, total return `9.24%`, annualized return `19.85%`, max drawdown `2.03%`, trades `103`, param `0.031381`.
- `baseline_heuristic` `h20` best policy `tanh_size` with test Sharpe `1.936`, total return `6.51%`, annualized return `13.79%`, max drawdown `1.67%`, trades `103`, param `0.033185`.
- `raw_tsm` `h20` best policy `tanh_size` with test Sharpe `1.871`, total return `6.24%`, annualized return `13.20%`, max drawdown `1.68%`, trades `103`, param `0.034760`.
- `learned_gbdt` `h30` best policy `uplift_linear_size` with test Sharpe `2.171`, total return `16.07%`, annualized return `33.19%`, max drawdown `3.48%`, trades `91`, param `0.000000`.
- `raw_tsm` `h30` best policy `tanh_size` with test Sharpe `2.134`, total return `17.46%`, annualized return `36.29%`, max drawdown `3.48%`, trades `101`, param `0.004219`.
- `baseline_heuristic` `h30` best policy `linear_size` with test Sharpe `2.134`, total return `17.48%`, annualized return `36.33%`, max drawdown `3.48%`, trades `101`, param `0.002934`.

## Interpretation
- If the best `learned_gbdt` policy beats the best `raw_tsm` policy on the same horizon, then the forecast-magnitude gain is monetizable under a leakage-safe validation-calibrated rule.
- If the best policies are still tied or nearly tied, the remaining bottleneck is the trading conversion rule or the economic objective itself, not the forecast MSE.

## Artifacts
- Plan: [plan.md](/Users/davidwilkinson/Desktop/ETS 2/reports/uk_ets_tsm_live_trading_conversion/20260316_122945/plan.md)
- Policy results: [policy_results.csv](/Users/davidwilkinson/Desktop/ETS 2/reports/uk_ets_tsm_live_trading_conversion/20260316_122945/policy_results.csv)
- Best policy table: [best_policy_by_model_horizon.csv](/Users/davidwilkinson/Desktop/ETS 2/reports/uk_ets_tsm_live_trading_conversion/20260316_122945/best_policy_by_model_horizon.csv)
- Test weights: [policy_test_weights.csv](/Users/davidwilkinson/Desktop/ETS 2/reports/uk_ets_tsm_live_trading_conversion/20260316_122945/policy_test_weights.csv)
