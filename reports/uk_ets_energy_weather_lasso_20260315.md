# UK ETS Energy + Weather LASSO Screen

Date: 2026-03-15

## Scope

This note records the first UK ETS feature-expansion pass adding:

- UK SAP gas from ONS / National Gas
- UK system electricity price from ONS / Elexon BMRS
- UK weather-demand proxy from Open-Meteo ERA5-Land
- Existing Brent and API2 ARA coal benchmarks

The lasso screen was run with the repo's existing selection path in
`src/run_experiment.py::select_features_via_lasso`, using the UK train split only.

## Data Added

- `uk_ets/Data_auto_uk/energy-benchmarks/uk-sap-gas-pence-per-kwh.csv`
- `uk_ets/Data_auto_uk/energy-benchmarks/uk-system-electricity-price-pence-per-kwh.csv`
- `uk_ets/Data_auto_uk/weather-proxy/uk_weather_daily_feature_pack.csv`

Weather pack assumptions:

- Equal-weight Great Britain metro composite
- Locations: London, Birmingham, Manchester, Leeds, Glasgow
- Variables: mean/min/max temperature, HDD18, 7-day HDD mean, 7-day mean temperature
- Weather is lagged by 1 day when merged into the panel to avoid same-day aggregation leakage

## Coverage vs UKA Target Calendar

Target calendar:

- 1,240 UKA futures dates
- 2021-05-19 to 2026-03-13

Key feature coverage in the aligned panel:

- `uk_gas_price`: 100.00%
- `uk_gas_return`: 99.92%
- `uk_power_price`: 100.00%
- `uk_power_return`: 99.76%
- `uk_temp_mean_c`: 99.68%
- `uk_hdd18`: 99.68%
- `uk_hdd18_7d_ma`: 99.92%
- `brent_return`: 99.60%
- `coal_return`: 95.97%

## LASSO Method

Config:

- summary stats: `last`, `change_5`, `change_20`, `std_20`
- horizons: `1, 5, 20, 30`
- mode: weighted aggregate
- weights: `0.1, 0.2, 0.3, 0.4`
- alpha: `0.01`
- train windows: `754`
- candidate numeric features: `70`

Outputs:

- `reports/uk_ets_lasso_energy_weather/20260315_020653/summary.md`
- `reports/uk_ets_lasso_energy_weather/20260315_020653/lasso_feature_scores.csv`
- `reports/uk_ets_lasso_energy_weather/20260315_020653/energy_weather_feature_scores.csv`
- `reports/uk_ets_lasso_energy_weather/20260315_020653/selected_features.json`

## Selected Features

Selected by the repo's current lasso path:

- `y_return`
- `coal_brent_ratio`
- `uk_icap_secondary`
- `uk_icap_primary_lag_20`
- `eurusd`
- `uk_temp_mean_c`
- `y_ma_20d`
- `coal_vol_5d`
- `uk_gas_vol_20d`
- `uk_temp_min_c`
- `uk_temp_max_c`
- `auction_price_lag_20`

## Interpretation

What survived strongly:

- UK weather is real in the screen. `uk_temp_mean_c`, `uk_temp_min_c`, and `uk_temp_max_c` were all selected.
- UK gas also matters, but mainly through volatility rather than raw return in this first pass. `uk_gas_vol_20d` was selected.
- Energy structure matters. `coal_brent_ratio` was the strongest single energy term by a large margin.

What is promising but not in the final capped selected set:

- `uk_hdd18_7d_ma`
- `uk_hdd18`
- `uk_power_return`
- `uk_gas_return`
- `uk_power_gas_spread`

So the first take is:

1. Weather clearly lines up with UK ETS.
2. Gas and fuel relationships line up too, but more through regime/volatility and spread structure than simple daily directional returns.
3. UK power is not absent, but it is weaker than weather and gas in this first lasso screen.

## Practical Next Step

The next defensible experiment is a forecast rerun with:

- current UK energy stack kept
- weather features kept
- lasso feature selection enabled explicitly

and then compare:

- current default feature slate
- lasso-selected slate from this screen

on the same frozen UK benchmark split.
