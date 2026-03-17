# UK ETS LASSO Predictor Screen

Generated: 2026-03-15T03:35:32.337105
Config: `/Users/davidwilkinson/Desktop/ETS 2/uk_ets/config/uk_ets_llm_4b_current_default.yaml`
Data dir: `/Users/davidwilkinson/Desktop/ETS 2/uk_ets/Data_auto_uk`

Method:
- Exact repo lasso-selection path from `src/run_experiment.py::select_features_via_lasso`.
- Training split only, using the configured UK train boundary.
- Current settings: summary stats = `last, change_5, change_20, std_20`; mode = weighted aggregate over horizons `[1,5,20,30]` with weights `[0.1,0.2,0.3,0.4]`.
- Candidate numeric feature count: `95` including target and all available exogenous columns.
- Train windows used: `754`.

Selected features:
- `y_return` (target_derived)
- `coal_brent_ratio` (energy)
- `uk_icap_secondary` (uk_icap)
- `uka_brent_ratio` (other)
- `uk_power_gas_vol_ratio_20d` (energy)
- `eurusd` (fx)
- `auction_price_rolling_mean_20` (uk_auction)
- `uk_power_vol_20d` (energy)
- `brent_price` (energy)
- `uk_gas_hdd18_surprise_interaction` (energy)
- `auction_price_lag_1` (uk_auction)
- `uka_brent_ratio_z20` (other)

Top energy/weather features by aggregate lasso score:
- `coal_brent_ratio`: `5.688057` (energy)
- `uk_power_gas_vol_ratio_20d`: `3.185948` (energy)
- `uk_power_vol_20d`: `2.050599` (energy)
- `brent_price`: `1.953021` (energy)
- `uk_gas_hdd18_surprise_interaction`: `1.761117` (energy)
- `uk_gas_vol_20d`: `1.433140` (energy)
- `uk_gas_price`: `1.207727` (energy)
- `brent_vol_5d`: `1.025563` (energy)
- `coal_vol_5d`: `0.940094` (energy)
- `uk_temp_mean_c`: `0.779376` (weather)
- `brent_return`: `0.671630` (energy)
- `coal_brent_ratio_z20`: `0.602090` (energy)
