# EU ETS Futures Forecasting with TSM + LLM Refinement

**Authors**: Auto-generated

**Generated**: 2026-03-05 01:53:26

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
- 348 daily observations
- Best performing method: linear_ridge


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
- **Total Observations**: 348
- **Number of Features**: 59

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

where $L = 120$ (lookback window) and $H = 30$ (forecast horizon).

### 3.2 Baseline Models

We implement several baseline models for comparison:

1. **Naive Persistence**: $\hat{y}_{t+h} = y_t$ for all horizons
2. **Seasonal Naive**: $\hat{y}_{t+h} = y_{t+h-5}$ (weekly seasonality)
3. **Linear Regression**: Ridge regression on lagged features
4. **ARIMA**: Autoregressive integrated moving average model

### 3.3 Time Series Model (TSM)

The primary TSM uses an Autoformer-style architecture with:
- **Model dimension**: 512
- **Attention heads**: 8
- **Encoder layers**: 2
- **Feed-forward dimension**: 2048
- **Dropout**: 0.05

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

|    |   horizon |        mse |      rmse |       mae |     mape | model                        |
|---:|----------:|-----------:|----------:|----------:|---------:|:-----------------------------|
|  0 |         1 |   3.19845  |  1.78842  |  1.5      |  3.13376 | naive_persistence            |
|  1 |         5 |  51.6027   |  7.1835   |  6.3575   | 11.6069  | naive_persistence            |
|  2 |        20 |   5.67175  |  2.38154  |  2.06     |  4.58605 | naive_persistence            |
|  3 |        30 |  13.4305   |  3.66477  |  3.1825   |  7.31159 | naive_persistence            |
|  4 |         1 |  18.2369   |  4.27046  |  3.52     |  7.52739 | seasonal_naive               |
|  5 |         5 |  51.6027   |  7.1835   |  6.3575   | 11.6069  | seasonal_naive               |
|  6 |        20 |   5.67175  |  2.38154  |  2.06     |  4.58605 | seasonal_naive               |
|  7 |        30 |  13.4305   |  3.66477  |  3.1825   |  7.31159 | seasonal_naive               |
|  8 |         1 |   2.43339  |  1.55993  |  1.32657  |  2.78938 | linear_ridge                 |
|  9 |         5 |  32.6113   |  5.71063  |  5.1835   |  9.49325 | linear_ridge                 |
| 10 |        20 |   4.15363  |  2.03805  |  1.96486  |  4.32953 | linear_ridge                 |
| 11 |        30 |   2.95076  |  1.71778  |  1.45709  |  3.33288 | linear_ridge                 |
| 12 |         1 |   3.28989  |  1.8138   |  1.62105  |  3.38547 | linear_lasso                 |
| 13 |         5 |  48.8026   |  6.98588  |  6.32767  | 11.5782  | linear_lasso                 |
| 14 |        20 |   0.639777 |  0.799861 |  0.791409 |  1.74917 | linear_lasso                 |
| 15 |        30 |   3.09323  |  1.75876  |  1.61733  |  3.64396 | linear_lasso                 |
| 16 |         1 |   4.55379  |  2.13396  |  1.66183  |  3.43719 | tsm                          |
| 17 |         5 |  85.6108   |  9.25261  |  8.65138  | 15.8942  | tsm                          |
| 18 |        20 |   8.49134  |  2.91399  |  2.33591  |  5.11171 | tsm                          |
| 19 |        30 |   5.17084  |  2.27395  |  2.09125  |  4.75751 | tsm                          |
| 20 |         1 |   3.20385  |  1.78993  |  1.47     |  3.14185 | DP                           |
| 21 |         5 |  23.7494   |  4.87333  |  3.8125   |  6.91617 | DP                           |
| 22 |        20 | 302.681    | 17.3977   | 16.1025   | 35.8495  | DP                           |
| 23 |        30 | 699.131    | 26.4411   | 24.665    | 55.9603  | DP                           |
| 24 |         1 |   3.08873  |  1.75748  |  1.3825   |  2.90809 | CoT                          |
| 25 |         5 |  48.1712   |  6.94055  |  4.85     |  8.70346 | CoT                          |
| 26 |        20 | 102.801    | 10.1391   |  9.405    | 20.7907  | CoT                          |
| 27 |        30 | 222.4      | 14.9131   | 13.4325   | 30.7961  | CoT                          |
| 28 |         1 |   3.08873  |  1.75748  |  1.3825   |  2.90809 | CoT-RF                       |
| 29 |         5 |  32.6662   |  5.71544  |  4.5      |  8.1409  | CoT-RF                       |
| 30 |        20 |  66.7055   |  8.16734  |  7.59     | 16.8211  | CoT-RF                       |
| 31 |        30 | 134.421    | 11.594    | 10.8525   | 24.7946  | CoT-RF                       |
| 32 |         1 |   3.19845  |  1.78842  |  1.5      |  3.13375 | TSM+LLM                      |
| 33 |         5 |  56.6681   |  7.52782  |  6.2375   | 11.3217  | TSM+LLM                      |
| 34 |        20 |  22.7233   |  4.76689  |  3.705    |  8.28063 | TSM+LLM                      |
| 35 |        30 |  35.0342   |  5.91897  |  4.6925   | 10.5235  | TSM+LLM                      |
| 36 |         1 |   3.19845  |  1.78842  |  1.5      |  3.13376 | naive_persistence_llm_subset |
| 37 |         5 |  51.6027   |  7.1835   |  6.3575   | 11.6069  | naive_persistence_llm_subset |
| 38 |        20 |   5.67175  |  2.38154  |  2.06     |  4.58605 | naive_persistence_llm_subset |
| 39 |        30 |  13.4305   |  3.66477  |  3.1825   |  7.31159 | naive_persistence_llm_subset |
| 40 |         1 |  18.2369   |  4.27046  |  3.52     |  7.52739 | seasonal_naive_llm_subset    |
| 41 |         5 |  51.6027   |  7.1835   |  6.3575   | 11.6069  | seasonal_naive_llm_subset    |
| 42 |        20 |   5.67175  |  2.38154  |  2.06     |  4.58605 | seasonal_naive_llm_subset    |
| 43 |        30 |  13.4305   |  3.66477  |  3.1825   |  7.31159 | seasonal_naive_llm_subset    |
| 44 |         1 |   2.43339  |  1.55993  |  1.32657  |  2.78938 | linear_ridge_llm_subset      |
| 45 |         5 |  32.6113   |  5.71063  |  5.1835   |  9.49325 | linear_ridge_llm_subset      |
| 46 |        20 |   4.15363  |  2.03805  |  1.96486  |  4.32953 | linear_ridge_llm_subset      |
| 47 |        30 |   2.95076  |  1.71778  |  1.45709  |  3.33288 | linear_ridge_llm_subset      |
| 48 |         1 |   3.28989  |  1.8138   |  1.62105  |  3.38547 | linear_lasso_llm_subset      |
| 49 |         5 |  48.8026   |  6.98588  |  6.32767  | 11.5782  | linear_lasso_llm_subset      |
| 50 |        20 |   0.639777 |  0.799861 |  0.791409 |  1.74917 | linear_lasso_llm_subset      |
| 51 |        30 |   3.09323  |  1.75876  |  1.61733  |  3.64396 | linear_lasso_llm_subset      |
| 52 |         1 |   4.55379  |  2.13396  |  1.66183  |  3.43719 | tsm_llm_subset               |
| 53 |         5 |  85.6108   |  9.25261  |  8.65138  | 15.8942  | tsm_llm_subset               |
| 54 |        20 |   8.49134  |  2.91399  |  2.33591  |  5.11171 | tsm_llm_subset               |
| 55 |        30 |   5.17084  |  2.27395  |  2.09125  |  4.75751 | tsm_llm_subset               |

### 4.2 Trend Classification Accuracy

Table 2 shows the accuracy of predicting price direction (up/flat/down).

|    |   horizon |   accuracy |   accuracy_up |   accuracy_down |   accuracy_flat |   n_up |   n_down |   n_flat | model                        |
|---:|----------:|-----------:|--------------:|----------------:|----------------:|-------:|---------:|---------:|:-----------------------------|
|  0 |         1 |       0.25 |          0    |        0        |               1 |      2 |        1 |        1 | naive_persistence            |
|  1 |         5 |       0    |          0    |      nan        |             nan |      4 |        0 |        0 | naive_persistence            |
|  2 |        20 |       0.25 |        nan    |        0        |               1 |      0 |        3 |        1 | naive_persistence            |
|  3 |        30 |       0    |        nan    |        0        |             nan |      0 |        4 |        0 | naive_persistence            |
|  4 |         1 |       0.25 |          0.5  |        0        |               0 |      2 |        1 |        1 | seasonal_naive               |
|  5 |         5 |       0    |          0    |      nan        |             nan |      4 |        0 |        0 | seasonal_naive               |
|  6 |        20 |       0.25 |        nan    |        0        |               1 |      0 |        3 |        1 | seasonal_naive               |
|  7 |        30 |       0    |        nan    |        0        |             nan |      0 |        4 |        0 | seasonal_naive               |
|  8 |         1 |       0.5  |          0.5  |        0        |               1 |      2 |        1 |        1 | linear_ridge                 |
|  9 |         5 |       0.75 |          0.75 |      nan        |             nan |      4 |        0 |        0 | linear_ridge                 |
| 10 |        20 |       0.5  |        nan    |        0.666667 |               0 |      0 |        3 |        1 | linear_ridge                 |
| 11 |        30 |       1    |        nan    |        1        |             nan |      0 |        4 |        0 | linear_ridge                 |
| 12 |         1 |       0.5  |          0.5  |        0        |               1 |      2 |        1 |        1 | linear_lasso                 |
| 13 |         5 |       0.25 |          0.25 |      nan        |             nan |      4 |        0 |        0 | linear_lasso                 |
| 14 |        20 |       0.75 |        nan    |        1        |               0 |      0 |        3 |        1 | linear_lasso                 |
| 15 |        30 |       1    |        nan    |        1        |             nan |      0 |        4 |        0 | linear_lasso                 |
| 16 |         1 |       0.5  |          0    |        1        |               1 |      2 |        1 |        1 | tsm                          |
| 17 |         5 |       0    |          0    |      nan        |             nan |      4 |        0 |        0 | tsm                          |
| 18 |        20 |       0.75 |        nan    |        1        |               0 |      0 |        3 |        1 | tsm                          |
| 19 |        30 |       1    |        nan    |        1        |             nan |      0 |        4 |        0 | tsm                          |
| 20 |         1 |       0.5  |          1    |        0        |               0 |      2 |        1 |        1 | DP                           |
| 21 |         5 |       1    |          1    |      nan        |             nan |      4 |        0 |        0 | DP                           |
| 22 |        20 |       0    |        nan    |        0        |               0 |      0 |        3 |        1 | DP                           |
| 23 |        30 |       0    |        nan    |        0        |             nan |      0 |        4 |        0 | DP                           |
| 24 |         1 |       0.5  |          0.5  |        0        |               1 |      2 |        1 |        1 | CoT                          |
| 25 |         5 |       0.75 |          0.75 |      nan        |             nan |      4 |        0 |        0 | CoT                          |
| 26 |        20 |       0    |        nan    |        0        |               0 |      0 |        3 |        1 | CoT                          |
| 27 |        30 |       0.25 |        nan    |        0.25     |             nan |      0 |        4 |        0 | CoT                          |
| 28 |         1 |       0.5  |          0.5  |        0        |               1 |      2 |        1 |        1 | CoT-RF                       |
| 29 |         5 |       1    |          1    |      nan        |             nan |      4 |        0 |        0 | CoT-RF                       |
| 30 |        20 |       0    |        nan    |        0        |               0 |      0 |        3 |        1 | CoT-RF                       |
| 31 |        30 |       0    |        nan    |        0        |             nan |      0 |        4 |        0 | CoT-RF                       |
| 32 |         1 |       0.25 |          0    |        0        |               1 |      2 |        1 |        1 | TSM+LLM                      |
| 33 |         5 |       0.5  |          0.5  |      nan        |             nan |      4 |        0 |        0 | TSM+LLM                      |
| 34 |        20 |       0.5  |        nan    |        0.666667 |               0 |      0 |        3 |        1 | TSM+LLM                      |
| 35 |        30 |       0.75 |        nan    |        0.75     |             nan |      0 |        4 |        0 | TSM+LLM                      |
| 36 |         1 |       0.25 |          0    |        0        |               1 |      2 |        1 |        1 | naive_persistence_llm_subset |
| 37 |         5 |       0    |          0    |      nan        |             nan |      4 |        0 |        0 | naive_persistence_llm_subset |
| 38 |        20 |       0.25 |        nan    |        0        |               1 |      0 |        3 |        1 | naive_persistence_llm_subset |
| 39 |        30 |       0    |        nan    |        0        |             nan |      0 |        4 |        0 | naive_persistence_llm_subset |
| 40 |         1 |       0.25 |          0.5  |        0        |               0 |      2 |        1 |        1 | seasonal_naive_llm_subset    |
| 41 |         5 |       0    |          0    |      nan        |             nan |      4 |        0 |        0 | seasonal_naive_llm_subset    |
| 42 |        20 |       0.25 |        nan    |        0        |               1 |      0 |        3 |        1 | seasonal_naive_llm_subset    |
| 43 |        30 |       0    |        nan    |        0        |             nan |      0 |        4 |        0 | seasonal_naive_llm_subset    |
| 44 |         1 |       0.5  |          0.5  |        0        |               1 |      2 |        1 |        1 | linear_ridge_llm_subset      |
| 45 |         5 |       0.75 |          0.75 |      nan        |             nan |      4 |        0 |        0 | linear_ridge_llm_subset      |
| 46 |        20 |       0.5  |        nan    |        0.666667 |               0 |      0 |        3 |        1 | linear_ridge_llm_subset      |
| 47 |        30 |       1    |        nan    |        1        |             nan |      0 |        4 |        0 | linear_ridge_llm_subset      |
| 48 |         1 |       0.5  |          0.5  |        0        |               1 |      2 |        1 |        1 | linear_lasso_llm_subset      |
| 49 |         5 |       0.25 |          0.25 |      nan        |             nan |      4 |        0 |        0 | linear_lasso_llm_subset      |
| 50 |        20 |       0.75 |        nan    |        1        |               0 |      0 |        3 |        1 | linear_lasso_llm_subset      |
| 51 |        30 |       1    |        nan    |        1        |             nan |      0 |        4 |        0 | linear_lasso_llm_subset      |
| 52 |         1 |       0.5  |          0    |        1        |               1 |      2 |        1 |        1 | tsm_llm_subset               |
| 53 |         5 |       0    |          0    |      nan        |             nan |      4 |        0 |        0 | tsm_llm_subset               |
| 54 |        20 |       0.75 |        nan    |        1        |               0 |      0 |        3 |        1 | tsm_llm_subset               |
| 55 |        30 |       1    |        nan    |        1        |             nan |      0 |        4 |        0 | tsm_llm_subset               |

### 4.3 Statistical Significance

Table 3 reports the results of significance tests comparing methods.

|    |   horizon | model                     | baseline                     |   mean_error_model |   mean_error_baseline |   improvement_pct |   t_statistic |   t_pvalue | t_significant   |   wilcoxon_statistic |   wilcoxon_pvalue | wilcoxon_significant   |      dm_statistic |   dm_pvalue | dm_significant   | model_better   |
|---:|----------:|:--------------------------|:-----------------------------|-------------------:|----------------------:|------------------:|--------------:|-----------:|:----------------|---------------------:|------------------:|:-----------------------|------------------:|------------:|:-----------------|:---------------|
|  0 |         1 | seasonal_naive_llm_subset | naive_persistence_llm_subset |          18.2369   |               3.19845 |    -470.177       |     1.23955   |  0.30327   | False           |                  nan |               nan | False                  |       1.23955     | 0.215142    | False            | False          |
|  1 |         5 | seasonal_naive_llm_subset | naive_persistence_llm_subset |          51.6027   |              51.6027  |      -2.82835e-06 |     1.52095   |  0.225614  | False           |                  nan |               nan | False                  |       0.291901    | 0.770362    | False            | False          |
|  2 |        20 | seasonal_naive_llm_subset | naive_persistence_llm_subset |           5.67175  |               5.67175 |       4.84253e-06 |     0.214215  |  0.844113  | False           |                  nan |               nan | False                  |       0.00579093  | 0.99538     | False            | True           |
|  3 |        30 | seasonal_naive_llm_subset | naive_persistence_llm_subset |          13.4305   |              13.4305  |      -5.66362e-06 |     1.23239   |  0.305581  | False           |                  nan |               nan | False                  |       0.0806052   | 0.935756    | False            | False          |
|  4 |         1 | linear_ridge_llm_subset   | naive_persistence_llm_subset |           2.43339  |               3.19845 |      23.9199      |    -0.524797  |  0.636036  | False           |                  nan |               nan | False                  |      -0.524797    | 0.599724    | False            | True           |
|  5 |         5 | linear_ridge_llm_subset   | naive_persistence_llm_subset |          32.6113   |              51.6027  |      36.8031      |    -1.53572   |  0.222183  | False           |                  nan |               nan | False                  |      -4.57834     | 4.68688e-06 | True             | True           |
|  6 |        20 | linear_ridge_llm_subset   | naive_persistence_llm_subset |           4.15363  |               5.67175 |      26.7663      |    -0.448063  |  0.684489  | False           |                  nan |               nan | False                  | -303624           | 0           | True             | True           |
|  7 |        30 | linear_ridge_llm_subset   | naive_persistence_llm_subset |           2.95076  |              13.4305  |      78.0295      |    -1.63492   |  0.200579  | False           |                  nan |               nan | False                  |      -2.09595e+06 | 0           | True             | True           |
|  8 |         1 | linear_lasso_llm_subset   | naive_persistence_llm_subset |           3.28989  |               3.19845 |      -2.85866     |     0.0916902 |  0.932724  | False           |                  nan |               nan | False                  |       0.0916902   | 0.926944    | False            | False          |
|  9 |         5 | linear_lasso_llm_subset   | naive_persistence_llm_subset |          48.8026   |              51.6027  |       5.42639     |    -0.518807  |  0.639737  | False           |                  nan |               nan | False                  |      -0.971932    | 0.331085    | False            | True           |
| 10 |        20 | linear_lasso_llm_subset   | naive_persistence_llm_subset |           0.639777 |               5.67175 |      88.7199      |    -2.1204    |  0.124135  | False           |                  nan |               nan | False                  |      -1.00639e+06 | 0           | True             | True           |
| 11 |        30 | linear_lasso_llm_subset   | naive_persistence_llm_subset |           3.09323  |              13.4305  |      76.9686      |    -1.2157    |  0.311037  | False           |                  nan |               nan | False                  |      -2.06746e+06 | 0           | True             | True           |
| 12 |         1 | tsm_llm_subset            | naive_persistence_llm_subset |           4.55379  |               3.19845 |     -42.3746      |     0.943756  |  0.414919  | False           |                  nan |               nan | False                  |       0.943756    | 0.345294    | False            | False          |
| 13 |         5 | tsm_llm_subset            | naive_persistence_llm_subset |          85.6108   |              51.6027  |     -65.9036      |     3.75943   |  0.0329028 | True            |                  nan |               nan | False                  |       6.80161e+06 | 0           | True             | False          |
| 14 |        20 | tsm_llm_subset            | naive_persistence_llm_subset |           8.49134  |               5.67175 |     -49.7129      |     0.427642  |  0.697751  | False           |                  nan |               nan | False                  |  563918           | 0           | True             | False          |
| 15 |        30 | tsm_llm_subset            | naive_persistence_llm_subset |           5.17084  |              13.4305  |      61.4993      |    -1.06497   |  0.365002  | False           |                  nan |               nan | False                  |      -1.65194e+06 | 0           | True             | True           |
| 16 |         1 | DP                        | naive_persistence_llm_subset |           3.20385  |               3.19845 |      -0.16877     |     0.0022898 |  0.998317  | False           |                  nan |               nan | False                  |       0.0022898   | 0.998173    | False            | False          |
| 17 |         5 | DP                        | naive_persistence_llm_subset |          23.7494   |              51.6027  |      53.9765      |    -2.57471   |  0.082156  | False           |                  nan |               nan | False                  |      -5.57067e+06 | 0           | True             | True           |
| 18 |        20 | DP                        | naive_persistence_llm_subset |         302.681    |               5.67175 |   -5236.64        |     2.19273   |  0.115958  | False           |                  nan |               nan | False                  |       5.94018e+07 | 0           | True             | False          |
| 19 |        30 | DP                        | naive_persistence_llm_subset |         699.131    |              13.4305  |   -5105.54        |     2.25401   |  0.109531  | False           |                  nan |               nan | False                  |       1.3714e+08  | 0           | True             | False          |
| 20 |         1 | CoT                       | naive_persistence_llm_subset |           3.08873  |               3.19845 |       3.43064     |    -0.255477  |  0.814869  | False           |                  nan |               nan | False                  |      -0.255477    | 0.798355    | False            | True           |
| 21 |         5 | CoT                       | naive_persistence_llm_subset |          48.1712   |              51.6027  |       6.64981     |    -0.275114  |  0.801088  | False           |                  nan |               nan | False                  |      -0.638723    | 0.523003    | False            | True           |
| 22 |        20 | CoT                       | naive_persistence_llm_subset |         102.801    |               5.67175 |   -1712.52        |     2.27916   |  0.107018  | False           |                  nan |               nan | False                  |       1.94259e+07 | 0           | True             | False          |
| 23 |        30 | CoT                       | naive_persistence_llm_subset |         222.4      |              13.4305  |   -1555.93        |     1.98522   |  0.141338  | False           |                  nan |               nan | False                  |       4.17939e+07 | 0           | True             | False          |
| 24 |         1 | CoT-RF                    | naive_persistence_llm_subset |           3.08873  |               3.19845 |       3.43064     |    -0.255477  |  0.814869  | False           |                  nan |               nan | False                  |      -0.255477    | 0.798355    | False            | True           |
| 25 |         5 | CoT-RF                    | naive_persistence_llm_subset |          32.6662   |              51.6027  |      36.6967      |    -2.74931   |  0.0707833 | False           |                  nan |               nan | False                  |      -3.7873e+06  | 0           | True             | True           |
| 26 |        20 | CoT-RF                    | naive_persistence_llm_subset |          66.7055   |               5.67175 |   -1076.1         |     2.64199   |  0.0775236 | False           |                  nan |               nan | False                  |       3.99989     | 6.33707e-05 | True             | False          |
| 27 |        30 | CoT-RF                    | naive_persistence_llm_subset |         134.421    |              13.4305  |    -900.858       |     2.71707   |  0.0727292 | False           |                  nan |               nan | False                  |       5.13133     | 2.87698e-07 | True             | False          |
| 28 |         1 | TSM+LLM                   | naive_persistence_llm_subset |           3.19845  |               3.19845 |       8.34868e-05 |    -1.28059   |  0.290369  | False           |                  nan |               nan | False                  |      -0.509098    | 0.610684    | False            | True           |
| 29 |         5 | TSM+LLM                   | naive_persistence_llm_subset |          56.6681   |              51.6027  |      -9.81614     |     0.61358   |  0.582882  | False           |                  nan |               nan | False                  |       0.973605    | 0.330252    | False            | False          |
| 30 |        20 | TSM+LLM                   | naive_persistence_llm_subset |          22.7233   |               5.67175 |    -300.639       |     1.2368    |  0.304156  | False           |                  nan |               nan | False                  |       3.4103e+06  | 0           | True             | False          |
| 31 |        30 | TSM+LLM                   | naive_persistence_llm_subset |          35.0342   |              13.4305  |    -160.855       |     0.795562  |  0.484419  | False           |                  nan |               nan | False                  |       4.32074e+06 | 0           | True             | False          |


## 5. Robustness Analysis

### 5.1 Noise Injection Test

To assess robustness, we inject noise into forecast paths at levels of 5%, 10%, 20%, 
and 30% of the forecast standard deviation.

|    |   horizon |       mse |     rmse |      mae |     mape |   noise_level | model   |
|---:|----------:|----------:|---------:|---------:|---------:|--------------:|:--------|
|  0 |         1 |   4.57166 |  2.13814 |  1.66402 |  3.44122 |          0.05 | tsm     |
|  1 |         5 |  85.3936  |  9.24087 |  8.64508 | 15.884   |          0.05 | tsm     |
|  2 |        20 |   8.6584  |  2.94252 |  2.36538 |  5.17542 |          0.05 | tsm     |
|  3 |        30 |   5.01431 |  2.23927 |  2.05737 |  4.67906 |          0.05 | tsm     |
|  4 |         1 |   4.59055 |  2.14256 |  1.6662  |  3.44526 |          0.1  | tsm     |
|  5 |         5 |  85.1789  |  9.22924 |  8.63878 | 15.8738  |          0.1  | tsm     |
|  6 |        20 |   8.83586 |  2.97252 |  2.39485 |  5.23913 |          0.1  | tsm     |
|  7 |        30 |   4.86236 |  2.20508 |  2.02349 |  4.60062 |          0.1  | tsm     |
|  8 |         1 |   4.63136 |  2.15206 |  1.67057 |  3.45332 |          0.2  | tsm     |
|  9 |         5 |  84.7567  |  9.20634 |  8.62618 | 15.8534  |          0.2  | tsm     |
| 10 |        20 |   9.22202 |  3.03678 |  2.45379 |  5.36656 |          0.2  | tsm     |
| 11 |        30 |   4.57223 |  2.13828 |  1.95572 |  4.44373 |          0.2  | tsm     |
| 12 |         1 |   4.6762  |  2.16245 |  1.67494 |  3.46138 |          0.3  | tsm     |
| 13 |         5 |  84.3442  |  9.18391 |  8.61358 | 15.833   |          0.3  | tsm     |
| 14 |        20 |   9.64981 |  3.10641 |  2.51274 |  5.49398 |          0.3  | tsm     |
| 15 |        30 |   4.30044 |  2.07375 |  1.88796 |  4.28685 |          0.3  | tsm     |
| 16 |         1 |   3.22136 |  1.79481 |  1.52108 |  3.24495 |          0.05 | DP      |
| 17 |         5 |  22.8142  |  4.77642 |  3.71079 |  6.72613 |          0.05 | DP      |
| 18 |        20 | 287.313   | 16.9503  | 15.6981  | 34.9448  |          0.05 | DP      |
| 19 |        30 | 701.714   | 26.4899  | 24.7584  | 56.1769  |          0.05 | DP      |
| 20 |         1 |   3.2933  |  1.81474 |  1.57215 |  3.34806 |          0.1  | DP      |
| 21 |         5 |  21.9273  |  4.68266 |  3.60908 |  6.53609 |          0.1  | DP      |
| 22 |        20 | 272.421   | 16.5052  | 15.2936  | 34.04    |          0.1  | DP      |
| 23 |        30 | 704.366   | 26.5399  | 24.8518  | 56.3935  |          0.1  | DP      |
| 24 |         1 |   3.60047 |  1.89749 |  1.67431 |  3.55427 |          0.2  | DP      |
| 25 |         5 |  20.2985  |  4.50538 |  3.40567 |  6.15601 |          0.2  | DP      |
| 26 |        20 | 244.062   | 15.6225  | 14.4847  | 32.2305  |          0.2  | DP      |
| 27 |        30 | 709.876   | 26.6435  | 25.0386  | 56.8267  |          0.2  | DP      |
| 28 |         1 |   4.12536 |  2.0311  |  1.77646 |  3.76048 |          0.3  | DP      |
| 29 |         5 |  18.8629  |  4.34314 |  3.20225 |  5.77592 |          0.3  | DP      |
| 30 |        20 | 217.604   | 14.7514  | 13.6758  | 30.4211  |          0.3  | DP      |
| 31 |        30 | 715.66    | 26.7518  | 25.2254  | 57.2599  |          0.3  | DP      |
| 32 |         1 |   3.08003 |  1.755   |  1.38666 |  2.91541 |          0.05 | CoT     |
| 33 |         5 |  47.4956  |  6.89171 |  4.83732 |  8.68319 |          0.05 | CoT     |
| 34 |        20 | 101.08    | 10.0539  |  9.3339  | 20.6289  |          0.05 | CoT     |
| 35 |        30 | 224.866   | 14.9955  | 13.4861  | 30.9223  |          0.05 | CoT     |
| 36 |         1 |   3.07838 |  1.75453 |  1.39082 |  2.92273 |          0.1  | CoT     |
| 37 |         5 |  46.8332  |  6.84348 |  4.82465 |  8.66292 |          0.1  | CoT     |
| 38 |        20 |  99.4233  |  9.97112 |  9.26281 | 20.467   |          0.1  | CoT     |
| 39 |        30 | 227.359   | 15.0784  | 13.5398  | 31.0486  |          0.1  | CoT     |
| 40 |         1 |   3.09622 |  1.75961 |  1.39913 |  2.93737 |          0.2  | CoT     |
| 41 |         5 |  45.5479  |  6.74892 |  4.79929 |  8.62239 |          0.2  | CoT     |
| 42 |        20 |  96.3041  |  9.81346 |  9.12062 | 20.1433  |          0.2  | CoT     |
| 43 |        30 | 232.42    | 15.2453  | 13.647   | 31.3011  |          0.2  | CoT     |
| 44 |         1 |   3.14224 |  1.77264 |  1.40745 |  2.952   |          0.3  | CoT     |
| 45 |         5 |  44.3155  |  6.65699 |  4.77394 |  8.58185 |          0.3  | CoT     |
| 46 |        20 |  93.4437  |  9.66663 |  8.97843 | 19.8196  |          0.3  | CoT     |
| 47 |        30 | 237.585   | 15.4138  | 13.7543  | 31.5536  |          0.3  | CoT     |
| 48 |         1 |   3.08388 |  1.7561  |  1.39178 |  2.92671 |          0.05 | CoT-RF  |
| 49 |         5 |  32.3219  |  5.68524 |  4.49311 |  8.13063 |          0.05 | CoT-RF  |
| 50 |        20 |  64.9379  |  8.0584  |  7.46829 | 16.5492  |          0.05 | CoT-RF  |
| 51 |        30 | 135.43    | 11.6374  | 10.8951  | 24.8936  |          0.05 | CoT-RF  |
| 52 |         1 |   3.08365 |  1.75603 |  1.40106 |  2.94533 |          0.1  | CoT-RF  |
| 53 |         5 |  31.9832  |  5.65537 |  4.48623 |  8.12036 |          0.1  | CoT-RF  |
| 54 |        20 |  63.2109  |  7.95053 |  7.34658 | 16.2773  |          0.1  | CoT-RF  |
| 55 |        30 | 136.45    | 11.6812  | 10.9377  | 24.9926  |          0.1  | CoT-RF  |
| 56 |         1 |   3.09704 |  1.75984 |  1.41962 |  2.98256 |          0.2  | CoT-RF  |
| 57 |         5 |  31.3224  |  5.59664 |  4.47245 |  8.09981 |          0.2  | CoT-RF  |
| 58 |        20 |  59.8787  |  7.73813 |  7.10316 | 15.7335  |          0.2  | CoT-RF  |
| 59 |        30 | 138.52    | 11.7694  | 11.0229  | 25.1906  |          0.2  | CoT-RF  |
| 60 |         1 |   3.1289  |  1.76887 |  1.43818 |  3.01979 |          0.3  | CoT-RF  |
| 61 |         5 |  30.684   |  5.53931 |  4.45868 |  8.07927 |          0.3  | CoT-RF  |
| 62 |        20 |  56.7089  |  7.53053 |  6.85974 | 15.1897  |          0.3  | CoT-RF  |
| 63 |        30 | 140.63    | 11.8588  | 11.1081  | 25.3887  |          0.3  | CoT-RF  |
| 64 |         1 |   3.23406 |  1.79835 |  1.51156 |  3.15705 |          0.05 | TSM+LLM |
| 65 |         5 |  56.022   |  7.48478 |  6.22157 | 11.2955  |          0.05 | TSM+LLM |
| 66 |        20 |  22.7473  |  4.76941 |  3.75037 |  8.37784 |          0.05 | TSM+LLM |
| 67 |        30 |  34.5917  |  5.88147 |  4.62414 | 10.3668  |          0.05 | TSM+LLM |
| 68 |         1 |   3.27355 |  1.80929 |  1.52313 |  3.18035 |          0.1  | TSM+LLM |
| 69 |         5 |  55.3836  |  7.44202 |  6.20564 | 11.2692  |          0.1  | TSM+LLM |
| 70 |        20 |  22.8146  |  4.77647 |  3.79574 |  8.47505 |          0.1  | TSM+LLM |
| 71 |        30 |  34.1603  |  5.84468 |  4.55579 | 10.2101  |          0.1  | TSM+LLM |
| 72 |         1 |   3.36415 |  1.83416 |  1.54625 |  3.22694 |          0.2  | TSM+LLM |
| 73 |         5 |  54.1304  |  7.35734 |  6.17378 | 11.2168  |          0.2  | TSM+LLM |
| 74 |        20 |  23.0791  |  4.80407 |  3.88647 |  8.66947 |          0.2  | TSM+LLM |
| 75 |        30 |  33.3311  |  5.77331 |  4.41907 |  9.89667 |          0.2  | TSM+LLM |
| 76 |         1 |   3.47026 |  1.86286 |  1.56938 |  3.27353 |          0.3  | TSM+LLM |
| 77 |         5 |  52.9084  |  7.27381 |  6.14192 | 11.1644  |          0.3  | TSM+LLM |
| 78 |        20 |  23.5166  |  4.8494  |  3.97721 |  8.8639  |          0.3  | TSM+LLM |
| 79 |        30 |  32.5466  |  5.70496 |  4.28235 |  9.58324 |          0.3  | TSM+LLM |

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

- **Price accuracy**: linear_ridge has the lowest average MSE (10.537).
- **TSM vs naive**: TSM MSE is 1.4x the naive baseline on average.
- **Directional accuracy**: linear_ridge has the highest average trend accuracy (0.688).

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
Price accuracy is best for linear_ridge, while directional accuracy is highest for linear_ridge.
The framework remains suitable for future runs with full LLM and robustness evaluations enabled.


## Appendix A: Reproducibility

### A.1 Configuration

```yaml
# Key configuration parameters
Random Seed: 42
Prediction Length: 30
Sequence Length: 120
TSM Type: autoformer
LLM Model: qwen3-vl-4b-gpu
```

### A.2 Environment

- Python: 3.10+
- Key packages: torch, pytorch-forecasting, openai, pandas, numpy

### A.3 Compute Resources

- Device preference: auto
- Data loader workers: 0
- Pin memory: True

### A.4 Data Availability

Raw data sources:
- ICAP Allowance Price Explorer
- EEX Emissions Auctions
- ECB Exchange Rates
- EIA Energy Prices


