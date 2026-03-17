# UK ETS LASSO Predictor Screen

Generated: 2026-03-15T02:06:54.765372
Config: `/Users/davidwilkinson/Desktop/ETS 2/uk_ets/config/uk_ets_llm_4b_current_default.yaml`
Data dir: `/Users/davidwilkinson/Desktop/ETS 2/uk_ets/Data_auto_uk`

Method:
- Exact repo lasso-selection path from `src/run_experiment.py::select_features_via_lasso`.
- Training split only, using the configured UK train boundary.
- Current settings: summary stats = `last, change_5, change_20, std_20`; mode = weighted aggregate over horizons `[1,5,20,30]` with weights `[0.1,0.2,0.3,0.4]`.
- Candidate numeric feature count: `70` including target and all available exogenous columns.
- Train windows used: `754`.

Selected features:
- `y_return` (target_derived)
- `coal_brent_ratio` (energy)
- `uk_icap_secondary` (uk_icap)
- `uk_icap_primary_lag_20` (uk_icap)
- `eurusd` (fx)
- `uk_temp_mean_c` (weather)
- `y_ma_20d` (target_derived)
- `coal_vol_5d` (energy)
- `uk_gas_vol_20d` (energy)
- `uk_temp_min_c` (weather)
- `uk_temp_max_c` (weather)
- `auction_price_lag_20` (uk_auction)

Top energy/weather features by aggregate lasso score:
- `coal_brent_ratio`: `6.912119` (energy)
- `uk_temp_mean_c`: `2.919832` (weather)
- `coal_vol_5d`: `1.823187` (energy)
- `uk_gas_vol_20d`: `1.807653` (energy)
- `uk_temp_min_c`: `1.685534` (weather)
- `uk_temp_max_c`: `1.576873` (weather)
- `uk_hdd18_7d_ma`: `1.433378` (weather)
- `brent_vol_5d`: `1.373884` (energy)
- `uk_gas_price`: `1.193030` (energy)
- `uk_hdd18`: `1.157757` (weather)
- `brent_price`: `0.974334` (energy)
- `uk_power_vol_5d`: `0.910154` (energy)
