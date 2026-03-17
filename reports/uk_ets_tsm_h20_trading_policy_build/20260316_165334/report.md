# TSM H20 Trading Policy Build Report

## Setup
- Forecast stack frozen at `DLinear + learned_gbdt(top20) + selective LLM`.
- Reran the heuristic baseline and the learned-gbdt winner with validation exports.
- Calibrated trading rules on validation only and applied them to the common test holdout.
- Costs: `10` bps transaction cost + `5` bps slippage, `1`-day execution lag, active-window MTM.

## Verification
- `same_val_dates`: `True`
- `same_test_dates`: `True`
- `same_val_base_pred`: `True`
- `same_test_base_pred`: `True`

## Best Policies
- `learned_gbdt` `h20` best policy `linear_size` with test Sharpe `2.006`, total return `9.24%`, annualized return `19.85%`, max drawdown `2.03%`, trades `103`, scale/threshold `0.031381`, prob floor `0.000000`.
- `baseline_heuristic` `h20` best policy `tanh_size` with test Sharpe `1.936`, total return `6.51%`, annualized return `13.79%`, max drawdown `1.67%`, trades `103`, scale/threshold `0.033185`, prob floor `0.000000`.
- `raw_tsm` `h20` best policy `tanh_size` with test Sharpe `1.871`, total return `6.24%`, annualized return `13.20%`, max drawdown `1.68%`, trades `103`, scale/threshold `0.034760`, prob floor `0.000000`.
- `learned_gbdt` `h30` best policy `uplift_linear_size` with test Sharpe `2.171`, total return `16.07%`, annualized return `33.19%`, max drawdown `3.48%`, trades `91`, scale/threshold `0.000000`, prob floor `0.000000`.
- `raw_tsm` `h30` best policy `tanh_size` with test Sharpe `2.134`, total return `17.48%`, annualized return `36.33%`, max drawdown `3.48%`, trades `101`, scale/threshold `0.001794`, prob floor `0.000000`.
- `baseline_heuristic` `h30` best policy `tanh_size` with test Sharpe `2.134`, total return `17.48%`, annualized return `36.33%`, max drawdown `3.48%`, trades `101`, scale/threshold `0.000673`, prob floor `0.000000`.

## Selected H20 Production Candidate
- `learned_gbdt` selected policy: `linear_size`
- Test Sharpe: `2.006`
- Total return: `9.24%`
- Annualized return: `19.85%`
- Max drawdown: `2.03%`
- Executed trades: `103`

## Interpretation
- If the selected `learned_gbdt` h20 policy beats both the raw TSM and heuristic LLM controls, the remaining edge is economically convertible under a validation-safe sizing rule.
- Regime-conditioned sizing is the key product test here: it asks whether the same learned gate that improves forecast MSE can also control trade aggressiveness in a useful way.

## Artifacts
- Plan: [plan.md](/Users/davidwilkinson/Desktop/ETS 2/reports/uk_ets_tsm_h20_trading_policy_build/20260316_165334/plan.md)
- Policy results: [policy_results.csv](/Users/davidwilkinson/Desktop/ETS 2/reports/uk_ets_tsm_h20_trading_policy_build/20260316_165334/policy_results.csv)
- Best policy table: [best_policy_by_model_horizon.csv](/Users/davidwilkinson/Desktop/ETS 2/reports/uk_ets_tsm_h20_trading_policy_build/20260316_165334/best_policy_by_model_horizon.csv)
- Selected h20 ledger: [selected_h20_trade_ledger.csv](/Users/davidwilkinson/Desktop/ETS 2/reports/uk_ets_tsm_h20_trading_policy_build/20260316_165334/selected_h20_trade_ledger.csv)
- Selected h20 MTM: [selected_h20_daily_mtm.csv](/Users/davidwilkinson/Desktop/ETS 2/reports/uk_ets_tsm_h20_trading_policy_build/20260316_165334/selected_h20_daily_mtm.csv)
