# EU ETS Futures Forecasting with TSM + LLM Refinement

**Authors**: Auto-generated

**Generated**: 2026-03-12 13:38:47

---

## Abstract

This study reproduces and extends the methodology of recent research on using Large Language Models (LLMs) 
for carbon price forecasting, applying it to the European Union Emissions Trading System (EU ETS). 
We implement a hybrid approach combining time series models (TSM) with LLM-based forecast refinement, 
evaluating performance across multiple horizons (1, 5, 20, and 30 days ahead).

Our methodology follows a two-stage approach: first training a deep learning time series model to produce 
30-step forecast paths, then applying LLM refinement to improve predictions. We evaluate multiple prompting 
strategies including direct prompting (DP), chain-of-thought (CoT), and TSM+LLM refinement.

Results are evaluated using mean squared error (MSE) for price prediction accuracy and classification accuracy 
for trend direction prediction. Statistical significance is assessed using paired t-tests and Wilcoxon signed-rank tests.

Key findings:
- Dataset spans 2021-05-19 00:00:00 to 2026-03-04 00:00:00
- 1233 daily observations
- Best performing method: linear_ridge_llm_subset


## 1. Introduction

The European Union Emissions Trading System (EU ETS) is the world's largest carbon market, 
covering approximately 40% of the EU's greenhouse gas emissions. Accurate forecasting of 
EU ETS allowance prices is crucial for compliance planning, investment decisions, and 
policy analysis.

### 1.1 Background

Carbon markets have grown significantly since the establishment of the EU ETS in 2005. 
The price of EU Allowances (EUAs) is influenced by multiple factors including:
- Energy prices (particularly natural gas and coal)
- Economic activity and industrial output
- Regulatory decisions and policy announcements
- Market speculation and hedging activities
- Auction volumes and timing

### 1.2 Related Work

Recent advances in Large Language Models have shown promising results in financial forecasting. 
The work "Can Large Language Models forecast carbon price movements?" demonstrated that LLMs 
can effectively refine quantitative model forecasts for Chinese carbon markets.

### 1.3 Contributions

This study makes the following contributions:
1. Reproduction of the Meta-DTS methodology for EU ETS futures
2. Implementation of multiple prompting strategies for forecast refinement
3. Comprehensive evaluation at multiple forecast horizons
4. Robustness analysis including noise injection tests


## 2. Data

### 2.1 Data Sources

This study uses the following data sources for EU ETS forecasting:

1. **Primary Market (Auctions)**: EEX emissions auction data from 2017-2025
2. **Secondary Market (ICAP)**: Daily EUA prices from the ICAP Allowance Price Explorer
3. **Carbon Market Indices**: ETF indices tracking carbon markets (KEUA, KRBN, GRN, KCCA, KSET)
4. **Energy Benchmarks**: Brent crude oil spot prices and Rotterdam coal futures
5. **Volatility Index**: VSTOXX for market volatility proxy
6. **Exchange Rates**: ECB EUR/USD reference rates for currency conversion

### 2.2 Target Variable

The primary target variable is the EU ETS secondary market price in EUR, derived from 
the ICAP allowance price explorer data. This represents actual trading prices rather 
than auction clearing prices.

### 2.3 Data Coverage

- **Date Range**: 2021-05-19 00:00:00 to 2026-03-04 00:00:00
- **Total Observations**: 1,233
- **Number of Features**: 45

### 2.4 Preprocessing

Key preprocessing steps:
1. Currency conversion: All USD-denominated series converted to EUR using ECB rates
2. Calendar alignment: All features aligned to the EUA trading calendar
3. Missing value handling: Forward-fill up to 5 days for market holidays
4. Feature engineering: Returns, rolling statistics, and momentum indicators


## 3. Methods

### 3.1 Problem Formulation

Given a sequence of daily EUA prices $y_{t-L+1}, ..., y_t$ and exogenous features 
$X_{t-L+1}, ..., X_t$, we aim to forecast the next $H$ days:

$$\hat{y}_{t+1}, ..., \hat{y}_{t+H}$$

where $L = 20$ (lookback window) and $H = 30$ (forecast horizon).

### 3.2 Baseline Models

We implement several baseline models for comparison:

1. **Naive Persistence**: $\hat{y}_{t+h} = y_t$ for all horizons
2. **Seasonal Naive**: $\hat{y}_{t+h} = y_{t+h-5}$ (weekly seasonality)
3. **Linear Regression**: Ridge regression on lagged features
4. **ARIMA**: Autoregressive integrated moving average model

### 3.3 Time Series Model (TSM)

The primary TSM uses an Autoformer-style architecture with:
- **Model dimension**: 32
- **Attention heads**: 2
- **Encoder layers**: 1
- **Feed-forward dimension**: 64
- **Dropout**: 0.3

The model uses series decomposition to separate trend and seasonal components, 
with auto-correlation attention for efficient long-range dependency modeling.

### 3.4 LLM Refinement

We implement four prompting strategies:

1. **Direct Prompting (DP)**: LLM forecasts directly from historical data
2. **Chain-of-Thought (CoT)**: Includes step-by-step reasoning
3. **CoT with Refinement (CoT-RF)**: CoT followed by self-critique and refinement
4. **TSM+LLM**: LLM refines the TSM model's forecast path

The TSM+LLM method is the primary approach, where the LLM receives:
- Recent price history (30 days)
- Summary statistics (mean, volatility, trend)
- The TSM's 30-day forecast
- Market context (optional)

### 3.5 Evaluation Metrics

**Price Prediction (Regression)**:
- Mean Squared Error (MSE): $\text{MSE} = \frac{1}{n} \sum_{i=1}^n (y_i - \hat{y}_i)^2$
- Root Mean Squared Error (RMSE)
- Mean Absolute Error (MAE)
- Mean Absolute Percentage Error (MAPE)

**Trend Classification**:
- Three-way classification: Up, Flat, Down
- Threshold based on recent volatility: $\tau = 0.25 \times \sigma_{20d}$
- Classification accuracy at each horizon

### 3.6 Statistical Tests

- **Paired t-test**: For comparing mean squared errors between methods
- **Wilcoxon signed-rank test**: Non-parametric alternative
- Significance level: $\alpha = 0.05$


## 4. Results

### 4.1 MSE by Horizon

Table 1 presents the Mean Squared Error at forecast horizons of 1, 5, 20, and 30 days.

|    |   horizon |        mse |     rmse |      mae |      mape | model                        |
|---:|----------:|-----------:|---------:|---------:|----------:|:-----------------------------|
|  0 |         1 |  1.06139   | 1.03024  | 0.784861 |  1.39543  | naive_persistence            |
|  1 |         5 |  5.2119    | 2.28296  | 1.67     |  2.90235  | naive_persistence            |
|  2 |        20 | 44.2759    | 6.65401  | 4.40799  |  7.94956  | naive_persistence            |
|  3 |        30 | 79.9881    | 8.94361  | 6.37986  | 12.0727   | naive_persistence            |
|  4 |         1 |  5.16853   | 2.27344  | 1.65132  |  2.90012  | seasonal_naive               |
|  5 |         5 |  5.2119    | 2.28296  | 1.67     |  2.90235  | seasonal_naive               |
|  6 |        20 | 44.2759    | 6.65401  | 4.40799  |  7.94956  | seasonal_naive               |
|  7 |        30 | 79.9881    | 8.94361  | 6.37986  | 12.0727   | seasonal_naive               |
|  8 |         1 |  1.06154   | 1.03031  | 0.781283 |  1.38304  | linear_ridge                 |
|  9 |         5 |  4.59363   | 2.14328  | 1.56632  |  2.70398  | linear_ridge                 |
| 10 |        20 | 30.1544    | 5.4913   | 3.32793  |  6.07247  | linear_ridge                 |
| 11 |        30 | 56.9094    | 7.54383  | 4.78768  |  9.18917  | linear_ridge                 |
| 12 |         1 |  1.1541    | 1.07429  | 0.830618 |  1.48473  | linear_lasso                 |
| 13 |         5 |  4.73942   | 2.17702  | 1.61188  |  2.79853  | linear_lasso                 |
| 14 |        20 | 32.1131    | 5.66684  | 3.43547  |  6.16613  | linear_lasso                 |
| 15 |        30 | 62.0974    | 7.88019  | 5.23504  |  9.89649  | linear_lasso                 |
| 16 |         1 |  1.22496   | 1.10678  | 0.823182 |  1.44975  | tsm                          |
| 17 |         5 |  4.58705   | 2.14174  | 1.52805  |  2.62472  | tsm                          |
| 18 |        20 | 29.8651    | 5.46489  | 3.58359  |  6.46353  | tsm                          |
| 19 |        30 | 59.4805    | 7.71236  | 5.39684  | 10.1968   | tsm                          |
| 20 |         1 |  4.62512   | 2.15061  | 2.15061  |  4.66408  | TSM+LLM-COT-RF-HDELTA        |
| 21 |         5 |  0.766717  | 0.875624 | 0.875624 |  1.85239  | TSM+LLM-COT-RF-HDELTA        |
| 22 |        20 |  0.42052   | 0.648475 | 0.648475 |  1.28973  | TSM+LLM-COT-RF-HDELTA        |
| 23 |        30 |  2.40517   | 1.55086  | 1.55086  |  3.02784  | TSM+LLM-COT-RF-HDELTA        |
| 24 |         1 |  1.7956    | 1.34     | 1.34     |  2.90609  | naive_persistence_llm_subset |
| 25 |         5 |  0.0324001 | 0.18     | 0.18     |  0.380792 | naive_persistence_llm_subset |
| 26 |        20 |  8.00889   | 2.83     | 2.83     |  5.62848  | naive_persistence_llm_subset |
| 27 |        30 | 14.2129    | 3.77     | 3.77     |  7.36041  | naive_persistence_llm_subset |
| 28 |         1 | 16.6464    | 4.08     | 4.08     |  8.8484   | seasonal_naive_llm_subset    |
| 29 |         5 |  0.0324001 | 0.18     | 0.18     |  0.380792 | seasonal_naive_llm_subset    |
| 30 |        20 |  8.00889   | 2.83     | 2.83     |  5.62848  | seasonal_naive_llm_subset    |
| 31 |        30 | 14.2129    | 3.77     | 3.77     |  7.36041  | seasonal_naive_llm_subset    |
| 32 |         1 |  2.83341   | 1.68327  | 1.68327  |  3.65056  | linear_ridge_llm_subset      |
| 33 |         5 |  0.278277  | 0.527519 | 0.527519 |  1.11597  | linear_ridge_llm_subset      |
| 34 |        20 |  0.0482077 | 0.219563 | 0.219563 |  0.43668  | linear_ridge_llm_subset      |
| 35 |        30 |  0.48864   | 0.699028 | 0.699028 |  1.36476  | linear_ridge_llm_subset      |
| 36 |         1 |  5.55559   | 2.35703  | 2.35703  |  5.11175  | linear_lasso_llm_subset      |
| 37 |         5 |  1.68519   | 1.29815  | 1.29815  |  2.74624  | linear_lasso_llm_subset      |
| 38 |        20 |  0.0298434 | 0.172752 | 0.172752 |  0.343581 | linear_lasso_llm_subset      |
| 39 |        30 |  0.594294  | 0.770905 | 0.770905 |  1.50508  | linear_lasso_llm_subset      |
| 40 |         1 |  4.62511   | 2.15061  | 2.15061  |  4.66408  | tsm_llm_subset               |
| 41 |         5 |  0.404614  | 0.636093 | 0.636093 |  1.34566  | tsm_llm_subset               |
| 42 |        20 |  0.156104  | 0.3951   | 0.3951   |  0.785799 | tsm_llm_subset               |
| 43 |        30 |  3.2327    | 1.79797  | 1.79797  |  3.51029  | tsm_llm_subset               |

### 4.2 Trend Classification Accuracy

Table 2 shows the accuracy of predicting price direction (up/flat/down).

|    |   horizon |   accuracy |   accuracy_up |   accuracy_down |   accuracy_flat |   n_up |   n_down |   n_flat | model                        |
|---:|----------:|-----------:|--------------:|----------------:|----------------:|-------:|---------:|---------:|:-----------------------------|
|  0 |         1 |  0.395833  |     0         |       0         |       1         |     52 |       35 |       57 | naive_persistence            |
|  1 |         5 |  0.236111  |     0         |       0         |       1         |     70 |       40 |       34 | naive_persistence            |
|  2 |        20 |  0.118056  |     0         |       0         |       1         |     90 |       37 |       17 | naive_persistence            |
|  3 |        30 |  0.0763889 |     0         |       0         |       1         |    100 |       33 |       11 | naive_persistence            |
|  4 |         1 |  0.375     |     0.365385  |       0.628571  |       0.22807   |     52 |       35 |       57 | seasonal_naive               |
|  5 |         5 |  0.236111  |     0         |       0         |       1         |     70 |       40 |       34 | seasonal_naive               |
|  6 |        20 |  0.118056  |     0         |       0         |       1         |     90 |       37 |       17 | seasonal_naive               |
|  7 |        30 |  0.0763889 |     0         |       0         |       1         |    100 |       33 |       11 | seasonal_naive               |
|  8 |         1 |  0.430556  |     0.0576923 |       0.0857143 |       0.982456  |     52 |       35 |       57 | linear_ridge                 |
|  9 |         5 |  0.451389  |     0.514286  |       0.125     |       0.705882  |     70 |       40 |       34 | linear_ridge                 |
| 10 |        20 |  0.708333  |     0.955556  |       0.351351  |       0.176471  |     90 |       37 |       17 | linear_ridge                 |
| 11 |        30 |  0.833333  |     0.99      |       0.606061  |       0.0909091 |    100 |       33 |       11 | linear_ridge                 |
| 12 |         1 |  0.423611  |     0.288462  |       0.114286  |       0.736842  |     52 |       35 |       57 | linear_lasso                 |
| 13 |         5 |  0.388889  |     0.414286  |       0.15      |       0.617647  |     70 |       40 |       34 | linear_lasso                 |
| 14 |        20 |  0.708333  |     0.922222  |       0.459459  |       0.117647  |     90 |       37 |       17 | linear_lasso                 |
| 15 |        30 |  0.736111  |     0.75      |       0.757576  |       0.545455  |    100 |       33 |       11 | linear_lasso                 |
| 16 |         1 |  0.409722  |     0.192308  |       0.114286  |       0.789474  |     52 |       35 |       57 | tsm                          |
| 17 |         5 |  0.451389  |     0.428571  |       0.4       |       0.558824  |     70 |       40 |       34 | tsm                          |
| 18 |        20 |  0.534722  |     0.566667  |       0.567568  |       0.294118  |     90 |       37 |       17 | tsm                          |
| 19 |        30 |  0.618056  |     0.59      |       0.818182  |       0.272727  |    100 |       33 |       11 | tsm                          |
| 20 |         1 |  0         |   nan         |       0         |     nan         |      0 |        1 |        0 | TSM+LLM-COT-RF-HDELTA        |
| 21 |         5 |  0         |   nan         |     nan         |       0         |      0 |        0 |        1 | TSM+LLM-COT-RF-HDELTA        |
| 22 |        20 |  1         |     1         |     nan         |     nan         |      1 |        0 |        0 | TSM+LLM-COT-RF-HDELTA        |
| 23 |        30 |  1         |     1         |     nan         |     nan         |      1 |        0 |        0 | TSM+LLM-COT-RF-HDELTA        |
| 24 |         1 |  0         |   nan         |       0         |     nan         |      0 |        1 |        0 | naive_persistence_llm_subset |
| 25 |         5 |  1         |   nan         |     nan         |       1         |      0 |        0 |        1 | naive_persistence_llm_subset |
| 26 |        20 |  0         |     0         |     nan         |     nan         |      1 |        0 |        0 | naive_persistence_llm_subset |
| 27 |        30 |  0         |     0         |     nan         |     nan         |      1 |        0 |        0 | naive_persistence_llm_subset |
| 28 |         1 |  0         |   nan         |       0         |     nan         |      0 |        1 |        0 | seasonal_naive_llm_subset    |
| 29 |         5 |  1         |   nan         |     nan         |       1         |      0 |        0 |        1 | seasonal_naive_llm_subset    |
| 30 |        20 |  0         |     0         |     nan         |     nan         |      1 |        0 |        0 | seasonal_naive_llm_subset    |
| 31 |        30 |  0         |     0         |     nan         |     nan         |      1 |        0 |        0 | seasonal_naive_llm_subset    |
| 32 |         1 |  0         |   nan         |       0         |     nan         |      0 |        1 |        0 | linear_ridge_llm_subset      |
| 33 |         5 |  1         |   nan         |     nan         |       1         |      0 |        0 |        1 | linear_ridge_llm_subset      |
| 34 |        20 |  1         |     1         |     nan         |     nan         |      1 |        0 |        0 | linear_ridge_llm_subset      |
| 35 |        30 |  1         |     1         |     nan         |     nan         |      1 |        0 |        0 | linear_ridge_llm_subset      |
| 36 |         1 |  0         |   nan         |       0         |     nan         |      0 |        1 |        0 | linear_lasso_llm_subset      |
| 37 |         5 |  0         |   nan         |     nan         |       0         |      0 |        0 |        1 | linear_lasso_llm_subset      |
| 38 |        20 |  1         |     1         |     nan         |     nan         |      1 |        0 |        0 | linear_lasso_llm_subset      |
| 39 |        30 |  1         |     1         |     nan         |     nan         |      1 |        0 |        0 | linear_lasso_llm_subset      |
| 40 |         1 |  0         |   nan         |       0         |     nan         |      0 |        1 |        0 | tsm_llm_subset               |
| 41 |         5 |  1         |   nan         |     nan         |       1         |      0 |        0 |        1 | tsm_llm_subset               |
| 42 |        20 |  1         |     1         |     nan         |     nan         |      1 |        0 |        0 | tsm_llm_subset               |
| 43 |        30 |  1         |     1         |     nan         |     nan         |      1 |        0 |        0 | tsm_llm_subset               |

### 4.3 Statistical Significance

Table 3 reports the results of significance tests comparing methods.

|    |   horizon | model                     | baseline                     |   mean_error_model |   mean_error_baseline |   improvement_pct |   t_statistic |   t_pvalue | t_significant   |   wilcoxon_statistic |   wilcoxon_pvalue | wilcoxon_significant   |   dm_statistic |   dm_pvalue | dm_significant   | model_better   |
|---:|----------:|:--------------------------|:-----------------------------|-------------------:|----------------------:|------------------:|--------------:|-----------:|:----------------|---------------------:|------------------:|:-----------------------|---------------:|------------:|:-----------------|:---------------|
|  0 |         1 | seasonal_naive_llm_subset | naive_persistence_llm_subset |         16.6464    |             1.7956    |    -827.065       |           nan |        nan | False           |                  nan |               nan | False                  |            nan |         nan | False            | False          |
|  1 |         5 | seasonal_naive_llm_subset | naive_persistence_llm_subset |          0.0324001 |             0.0324001 |      -3.0541e-06  |           nan |        nan | False           |                  nan |               nan | False                  |            nan |         nan | False            | False          |
|  2 |        20 | seasonal_naive_llm_subset | naive_persistence_llm_subset |          8.00889   |             8.00889   |       5.31555e-06 |           nan |        nan | False           |                  nan |               nan | False                  |            nan |         nan | False            | True           |
|  3 |        30 | seasonal_naive_llm_subset | naive_persistence_llm_subset |         14.2129    |            14.2129    |      -3.01678e-06 |           nan |        nan | False           |                  nan |               nan | False                  |            nan |         nan | False            | False          |
|  4 |         1 | linear_ridge_llm_subset   | naive_persistence_llm_subset |          2.83341   |             1.7956    |     -57.7973      |           nan |        nan | False           |                  nan |               nan | False                  |            nan |         nan | False            | False          |
|  5 |         5 | linear_ridge_llm_subset   | naive_persistence_llm_subset |          0.278277  |             0.0324001 |    -758.875       |           nan |        nan | False           |                  nan |               nan | False                  |            nan |         nan | False            | False          |
|  6 |        20 | linear_ridge_llm_subset   | naive_persistence_llm_subset |          0.0482077 |             8.00889   |      99.3981      |           nan |        nan | False           |                  nan |               nan | False                  |            nan |         nan | False            | True           |
|  7 |        30 | linear_ridge_llm_subset   | naive_persistence_llm_subset |          0.48864   |            14.2129    |      96.562       |           nan |        nan | False           |                  nan |               nan | False                  |            nan |         nan | False            | True           |
|  8 |         1 | linear_lasso_llm_subset   | naive_persistence_llm_subset |          5.55559   |             1.7956    |    -209.4         |           nan |        nan | False           |                  nan |               nan | False                  |            nan |         nan | False            | False          |
|  9 |         5 | linear_lasso_llm_subset   | naive_persistence_llm_subset |          1.68519   |             0.0324001 |   -5101.19        |           nan |        nan | False           |                  nan |               nan | False                  |            nan |         nan | False            | False          |
| 10 |        20 | linear_lasso_llm_subset   | naive_persistence_llm_subset |          0.0298434 |             8.00889   |      99.6274      |           nan |        nan | False           |                  nan |               nan | False                  |            nan |         nan | False            | True           |
| 11 |        30 | linear_lasso_llm_subset   | naive_persistence_llm_subset |          0.594294  |            14.2129    |      95.8186      |           nan |        nan | False           |                  nan |               nan | False                  |            nan |         nan | False            | True           |
| 12 |         1 | tsm_llm_subset            | naive_persistence_llm_subset |          4.62511   |             1.7956    |    -157.58        |           nan |        nan | False           |                  nan |               nan | False                  |            nan |         nan | False            | False          |
| 13 |         5 | tsm_llm_subset            | naive_persistence_llm_subset |          0.404614  |             0.0324001 |   -1148.81        |           nan |        nan | False           |                  nan |               nan | False                  |            nan |         nan | False            | False          |
| 14 |        20 | tsm_llm_subset            | naive_persistence_llm_subset |          0.156104  |             8.00889   |      98.0509      |           nan |        nan | False           |                  nan |               nan | False                  |            nan |         nan | False            | True           |
| 15 |        30 | tsm_llm_subset            | naive_persistence_llm_subset |          3.2327    |            14.2129    |      77.2552      |           nan |        nan | False           |                  nan |               nan | False                  |            nan |         nan | False            | True           |
| 16 |         1 | TSM+LLM-COT-RF-HDELTA     | naive_persistence_llm_subset |          4.62512   |             1.7956    |    -157.58        |           nan |        nan | False           |                  nan |               nan | False                  |            nan |         nan | False            | False          |
| 17 |         5 | TSM+LLM-COT-RF-HDELTA     | naive_persistence_llm_subset |          0.766717  |             0.0324001 |   -2266.4         |           nan |        nan | False           |                  nan |               nan | False                  |            nan |         nan | False            | False          |
| 18 |        20 | TSM+LLM-COT-RF-HDELTA     | naive_persistence_llm_subset |          0.42052   |             8.00889   |      94.7493      |           nan |        nan | False           |                  nan |               nan | False                  |            nan |         nan | False            | True           |
| 19 |        30 | TSM+LLM-COT-RF-HDELTA     | naive_persistence_llm_subset |          2.40517   |            14.2129    |      83.0776      |           nan |        nan | False           |                  nan |               nan | False                  |            nan |         nan | False            | True           |


## 5. Robustness Analysis

### 5.1 Noise Injection Test

To assess robustness, we inject noise into forecast paths at levels of 5%, 10%, 20%, 
and 30% of the forecast standard deviation.

|    |   horizon |        mse |     rmse |      mae |      mape |   noise_level | model                 |
|---:|----------:|-----------:|---------:|---------:|----------:|--------------:|:----------------------|
|  0 |         1 |  1.21489   | 1.10222  | 0.820331 |  1.44583  |          0.05 | tsm                   |
|  1 |         5 |  4.59811   | 2.14432  | 1.52671  |  2.62331  |          0.05 | tsm                   |
|  2 |        20 | 29.8561    | 5.46407  | 3.58635  |  6.46801  |          0.05 | tsm                   |
|  3 |        30 | 59.4497    | 7.71036  | 5.3919   | 10.1879   |          0.05 | tsm                   |
|  4 |         1 |  1.20679   | 1.09854  | 0.817577 |  1.44208  |          0.1  | tsm                   |
|  5 |         5 |  4.61202   | 2.14756  | 1.52665  |  2.62418  |          0.1  | tsm                   |
|  6 |        20 | 29.8493    | 5.46345  | 3.58911  |  6.47248  |          0.1  | tsm                   |
|  7 |        30 | 59.4206    | 7.70847  | 5.38697  | 10.1789   |          0.1  | tsm                   |
|  8 |         1 |  1.19649   | 1.09384  | 0.812703 |  1.4359   |          0.2  | tsm                   |
|  9 |         5 |  4.6484    | 2.15601  | 1.52773  |  2.62835  |          0.2  | tsm                   |
| 10 |        20 | 29.8426    | 5.46284  | 3.59462  |  6.48143  |          0.2  | tsm                   |
| 11 |        30 | 59.3676    | 7.70504  | 5.37711  | 10.161    |          0.2  | tsm                   |
| 12 |         1 |  1.19405   | 1.09273  | 0.810592 |  1.43459  |          0.3  | tsm                   |
| 13 |         5 |  4.69618   | 2.16707  | 1.53203  |  2.63806  |          0.3  | tsm                   |
| 14 |        20 | 29.8449    | 5.46305  | 3.60096  |  6.49201  |          0.3  | tsm                   |
| 15 |        30 | 59.3218    | 7.70206  | 5.36768  | 10.1439   |          0.3  | tsm                   |
| 16 |         1 |  4.7438    | 2.17803  | 2.17803  |  4.72354  |          0.05 | TSM+LLM-COT-RF-HDELTA |
| 17 |         5 |  0.744249  | 0.862698 | 0.862698 |  1.82504  |          0.05 | TSM+LLM-COT-RF-HDELTA |
| 18 |        20 |  0.325489  | 0.570516 | 0.570516 |  1.13468  |          0.05 | TSM+LLM-COT-RF-HDELTA |
| 19 |        30 |  2.45537   | 1.56696  | 1.56696  |  3.05928  |          0.05 | TSM+LLM-COT-RF-HDELTA |
| 20 |         1 |  4.86399   | 2.20545  | 2.20545  |  4.78301  |          0.1  | TSM+LLM-COT-RF-HDELTA |
| 21 |         5 |  0.722114  | 0.849773 | 0.849773 |  1.7977   |          0.1  | TSM+LLM-COT-RF-HDELTA |
| 22 |        20 |  0.242613  | 0.492557 | 0.492557 |  0.979629 |          0.1  | TSM+LLM-COT-RF-HDELTA |
| 23 |        30 |  2.50609   | 1.58306  | 1.58306  |  3.09071  |          0.1  | TSM+LLM-COT-RF-HDELTA |
| 24 |         1 |  5.10888   | 2.26028  | 2.26028  |  4.90193  |          0.2  | TSM+LLM-COT-RF-HDELTA |
| 25 |         5 |  0.678849  | 0.823923 | 0.823923 |  1.74301  |          0.2  | TSM+LLM-COT-RF-HDELTA |
| 26 |        20 |  0.113326  | 0.33664  | 0.33664  |  0.66953  |          0.2  | TSM+LLM-COT-RF-HDELTA |
| 27 |        30 |  2.60908   | 1.61527  | 1.61527  |  3.15358  |          0.2  | TSM+LLM-COT-RF-HDELTA |
| 28 |         1 |  5.35978   | 2.31512  | 2.31512  |  5.02086  |          0.3  | TSM+LLM-COT-RF-HDELTA |
| 29 |         5 |  0.636919  | 0.798072 | 0.798072 |  1.68833  |          0.3  | TSM+LLM-COT-RF-HDELTA |
| 30 |        20 |  0.0326604 | 0.180722 | 0.180722 |  0.359431 |          0.3  | TSM+LLM-COT-RF-HDELTA |
| 31 |        30 |  2.71415   | 1.64747  | 1.64747  |  3.21645  |          0.3  | TSM+LLM-COT-RF-HDELTA |

### 5.2 Ablation Study

We compare the contribution of different components:
- TSM only (no LLM refinement)
- LLM only (direct prompting)
- TSM + LLM (main method)
- With/without exogenous features

### 5.3 Temporal Stability

We evaluate performance across different market regimes:


## 6. Discussion

### 6.1 Key Findings

- **Price accuracy**: linear_ridge_llm_subset has the lowest average MSE (0.912).
- **TSM vs naive**: TSM MSE is 0.7x the naive baseline on average.
- **Directional accuracy**: linear_ridge_llm_subset has the highest average trend accuracy (0.750).

### 6.2 Comparison to Original Paper

This study applies the Meta-DTS methodology to the EU ETS market. Key similarities 
and differences with the original Chinese carbon market study:

- **Similarities**: Same evaluation metrics (MSE, trend accuracy), similar horizons
- **Differences**: Different market dynamics, regulatory environment, and data sources

### 6.3 Limitations

1. **Data limitations**: Limited secondary market data availability
2. **LLM costs**: API costs limit extensive hyperparameter tuning
3. **Market regime changes**: EU ETS underwent significant regulatory changes
4. **Comparison scope**: Direct comparison to the original paper is limited by market differences

### 6.4 Future Work

- Incorporate news sentiment analysis
- Extend to other carbon markets (California, UK ETS)
- Investigate ensemble approaches
- Explore fine-tuned LLMs for carbon market analysis


## 7. Conclusion

This run summarizes a reproducible pipeline for EU ETS carbon price forecasting.
Price accuracy is best for linear_ridge_llm_subset, while directional accuracy is highest for linear_ridge_llm_subset.
The framework remains suitable for future runs with full LLM and robustness evaluations enabled.


## Appendix A: Reproducibility

### A.1 Configuration

```yaml
# Key configuration parameters
Random Seed: 42
Prediction Length: 30
Sequence Length: 20
TSM Type: dlinear
LLM Model: qwen3.5-35b-a3b-ud-q4-k-xl
```

### A.2 Environment

- Python: 3.10+
- Key packages: torch, pytorch-forecasting, openai, pandas, numpy

### A.3 Compute Resources

- Device preference: auto
- Data loader workers: 0
- Pin memory: False

### A.4 Data Availability

Raw data sources:
- ICAP Allowance Price Explorer
- EEX Emissions Auctions
- ECB Exchange Rates
- EIA Energy Prices


