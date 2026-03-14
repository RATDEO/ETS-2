# EU ETS Futures Forecasting with TSM + LLM Refinement

**Authors**: Auto-generated

**Generated**: 2026-03-05 01:50:47

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
| 16 |         1 |   3.80385  |  1.95035  |  1.62     |  3.45112 | DP                           |
| 17 |         5 |  12.3144   |  3.50918  |  2.7375   |  4.98715 | DP                           |
| 18 |        20 | 339.815    | 18.4341   | 17.84     | 39.6357  | DP                           |
| 19 |        30 | 699.126    | 26.441    | 25.4275   | 57.6087  | DP                           |
| 20 |         1 |   3.08873  |  1.75748  |  1.3825   |  2.90809 | CoT                          |
| 21 |         5 |  38.8112   |  6.22987  |  4.375    |  7.85566 | CoT                          |
| 22 |        20 | 118.891    | 10.9037   |  9.355    | 20.7173  | CoT                          |
| 23 |        30 | 199.942    | 14.1401   | 11.6025   | 26.6762  | CoT                          |
| 24 |         1 |   3.08873  |  1.75748  |  1.3825   |  2.90809 | CoT-RF                       |
| 25 |         5 |  34.1812   |  5.84647  |  4.575    |  8.27191 | CoT-RF                       |
| 26 |        20 |  42.3125   |  6.50481  |  5.94     | 13.2426  | CoT-RF                       |
| 27 |        30 |  76.2745   |  8.73353  |  7.7025   | 17.3379  | CoT-RF                       |
| 28 |         1 |   3.19845  |  1.78842  |  1.5      |  3.13376 | naive_persistence_llm_subset |
| 29 |         5 |  51.6027   |  7.1835   |  6.3575   | 11.6069  | naive_persistence_llm_subset |
| 30 |        20 |   5.67175  |  2.38154  |  2.06     |  4.58605 | naive_persistence_llm_subset |
| 31 |        30 |  13.4305   |  3.66477  |  3.1825   |  7.31159 | naive_persistence_llm_subset |
| 32 |         1 |  18.2369   |  4.27046  |  3.52     |  7.52739 | seasonal_naive_llm_subset    |
| 33 |         5 |  51.6027   |  7.1835   |  6.3575   | 11.6069  | seasonal_naive_llm_subset    |
| 34 |        20 |   5.67175  |  2.38154  |  2.06     |  4.58605 | seasonal_naive_llm_subset    |
| 35 |        30 |  13.4305   |  3.66477  |  3.1825   |  7.31159 | seasonal_naive_llm_subset    |
| 36 |         1 |   2.43339  |  1.55993  |  1.32657  |  2.78938 | linear_ridge_llm_subset      |
| 37 |         5 |  32.6113   |  5.71063  |  5.1835   |  9.49325 | linear_ridge_llm_subset      |
| 38 |        20 |   4.15363  |  2.03805  |  1.96486  |  4.32953 | linear_ridge_llm_subset      |
| 39 |        30 |   2.95076  |  1.71778  |  1.45709  |  3.33288 | linear_ridge_llm_subset      |
| 40 |         1 |   3.28989  |  1.8138   |  1.62105  |  3.38547 | linear_lasso_llm_subset      |
| 41 |         5 |  48.8026   |  6.98588  |  6.32767  | 11.5782  | linear_lasso_llm_subset      |
| 42 |        20 |   0.639777 |  0.799861 |  0.791409 |  1.74917 | linear_lasso_llm_subset      |
| 43 |        30 |   3.09323  |  1.75876  |  1.61733  |  3.64396 | linear_lasso_llm_subset      |

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
| 16 |         1 |       0.5  |          1    |        0        |               0 |      2 |        1 |        1 | DP                           |
| 17 |         5 |       1    |          1    |      nan        |             nan |      4 |        0 |        0 | DP                           |
| 18 |        20 |       0    |        nan    |        0        |               0 |      0 |        3 |        1 | DP                           |
| 19 |        30 |       0    |        nan    |        0        |             nan |      0 |        4 |        0 | DP                           |
| 20 |         1 |       0.5  |          0.5  |        0        |               1 |      2 |        1 |        1 | CoT                          |
| 21 |         5 |       0.75 |          0.75 |      nan        |             nan |      4 |        0 |        0 | CoT                          |
| 22 |        20 |       0.25 |        nan    |        0        |               1 |      0 |        3 |        1 | CoT                          |
| 23 |        30 |       0    |        nan    |        0        |             nan |      0 |        4 |        0 | CoT                          |
| 24 |         1 |       0.5  |          0.5  |        0        |               1 |      2 |        1 |        1 | CoT-RF                       |
| 25 |         5 |       1    |          1    |      nan        |             nan |      4 |        0 |        0 | CoT-RF                       |
| 26 |        20 |       0    |        nan    |        0        |               0 |      0 |        3 |        1 | CoT-RF                       |
| 27 |        30 |       0.25 |        nan    |        0.25     |             nan |      0 |        4 |        0 | CoT-RF                       |
| 28 |         1 |       0.25 |          0    |        0        |               1 |      2 |        1 |        1 | naive_persistence_llm_subset |
| 29 |         5 |       0    |          0    |      nan        |             nan |      4 |        0 |        0 | naive_persistence_llm_subset |
| 30 |        20 |       0.25 |        nan    |        0        |               1 |      0 |        3 |        1 | naive_persistence_llm_subset |
| 31 |        30 |       0    |        nan    |        0        |             nan |      0 |        4 |        0 | naive_persistence_llm_subset |
| 32 |         1 |       0.25 |          0.5  |        0        |               0 |      2 |        1 |        1 | seasonal_naive_llm_subset    |
| 33 |         5 |       0    |          0    |      nan        |             nan |      4 |        0 |        0 | seasonal_naive_llm_subset    |
| 34 |        20 |       0.25 |        nan    |        0        |               1 |      0 |        3 |        1 | seasonal_naive_llm_subset    |
| 35 |        30 |       0    |        nan    |        0        |             nan |      0 |        4 |        0 | seasonal_naive_llm_subset    |
| 36 |         1 |       0.5  |          0.5  |        0        |               1 |      2 |        1 |        1 | linear_ridge_llm_subset      |
| 37 |         5 |       0.75 |          0.75 |      nan        |             nan |      4 |        0 |        0 | linear_ridge_llm_subset      |
| 38 |        20 |       0.5  |        nan    |        0.666667 |               0 |      0 |        3 |        1 | linear_ridge_llm_subset      |
| 39 |        30 |       1    |        nan    |        1        |             nan |      0 |        4 |        0 | linear_ridge_llm_subset      |
| 40 |         1 |       0.5  |          0.5  |        0        |               1 |      2 |        1 |        1 | linear_lasso_llm_subset      |
| 41 |         5 |       0.25 |          0.25 |      nan        |             nan |      4 |        0 |        0 | linear_lasso_llm_subset      |
| 42 |        20 |       0.75 |        nan    |        1        |               0 |      0 |        3 |        1 | linear_lasso_llm_subset      |
| 43 |        30 |       1    |        nan    |        1        |             nan |      0 |        4 |        0 | linear_lasso_llm_subset      |

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
| 12 |         1 | DP                        | naive_persistence_llm_subset |           3.80385  |               3.19845 |     -18.9278      |     0.314411  |  0.773806  | False           |                  nan |               nan | False                  |       0.314411    | 0.753209    | False            | False          |
| 13 |         5 | DP                        | naive_persistence_llm_subset |          12.3144   |              51.6027  |      76.1362      |    -1.98761   |  0.141011  | False           |                  nan |               nan | False                  |      -5.9452      | 2.76123e-09 | True             | True           |
| 14 |        20 | DP                        | naive_persistence_llm_subset |         339.815    |               5.67175 |   -5891.35        |     3.12719   |  0.0521794 | False           |                  nan |               nan | False                  |       6.68285e+07 | 0           | True             | False          |
| 15 |        30 | DP                        | naive_persistence_llm_subset |         699.126    |              13.4305  |   -5105.5         |     2.82679   |  0.0663652 | False           |                  nan |               nan | False                  |       1.37139e+08 | 0           | True             | False          |
| 16 |         1 | CoT                       | naive_persistence_llm_subset |           3.08873  |               3.19845 |       3.43064     |    -0.255477  |  0.814869  | False           |                  nan |               nan | False                  |      -0.255477    | 0.798355    | False            | True           |
| 17 |         5 | CoT                       | naive_persistence_llm_subset |          38.8112   |              51.6027  |      24.7884      |    -2.11147   |  0.125192  | False           |                  nan |               nan | False                  |      -2.5583e+06  | 0           | True             | True           |
| 18 |        20 | CoT                       | naive_persistence_llm_subset |         118.891    |               5.67175 |   -1996.2         |     2.46437   |  0.0905158 | False           |                  nan |               nan | False                  |       2.26439e+07 | 0           | True             | False          |
| 19 |        30 | CoT                       | naive_persistence_llm_subset |         199.942    |              13.4305  |   -1388.71        |     1.64564   |  0.19839   | False           |                  nan |               nan | False                  |       3.73022e+07 | 0           | True             | False          |
| 20 |         1 | CoT-RF                    | naive_persistence_llm_subset |           3.08873  |               3.19845 |       3.43064     |    -0.255477  |  0.814869  | False           |                  nan |               nan | False                  |      -0.255477    | 0.798355    | False            | True           |
| 21 |         5 | CoT-RF                    | naive_persistence_llm_subset |          34.1812   |              51.6027  |      33.7608      |    -3.09061   |  0.0536879 | False           |                  nan |               nan | False                  |      -3.4843e+06  | 0           | True             | True           |
| 22 |        20 | CoT-RF                    | naive_persistence_llm_subset |          42.3125   |               5.67175 |    -646.022       |     1.95322   |  0.145815  | False           |                  nan |               nan | False                  |       7.32815e+06 | 0           | True             | False          |
| 23 |        30 | CoT-RF                    | naive_persistence_llm_subset |          76.2745   |              13.4305  |    -467.919       |     1.61876   |  0.203932  | False           |                  nan |               nan | False                  |       1.25688e+07 | 0           | True             | False          |


## 5. Robustness Analysis

### 5.1 Noise Injection Test

To assess robustness, we inject noise into forecast paths at levels of 5%, 10%, 20%, 
and 30% of the forecast standard deviation.

|    |   horizon |       mse |     rmse |      mae |     mape |   noise_level | model   |
|---:|----------:|----------:|---------:|---------:|---------:|--------------:|:--------|
|  0 |         1 |   3.98429 |  1.99607 |  1.69675 |  3.60845 |          0.05 | DP      |
|  1 |         5 |  11.3965  |  3.37587 |  2.6226  |  4.7764  |          0.05 | DP      |
|  2 |        20 | 322.706   | 17.964   | 17.4049  | 38.6659  |          0.05 | DP      |
|  3 |        30 | 702.022   | 26.4957  | 25.5239  | 57.8312  |          0.05 | DP      |
|  4 |         1 |   4.21698 |  2.05353 |  1.77349 |  3.76578 |          0.1  | DP      |
|  5 |         5 |  10.5378  |  3.2462  |  2.5077  |  4.56564 |          0.1  | DP      |
|  6 |        20 | 306.129   | 17.4965  | 16.9697  | 37.6961  |          0.1  | DP      |
|  7 |        30 | 704.977   | 26.5514  | 25.6203  | 58.0537  |          0.1  | DP      |
|  8 |         1 |   4.83913 |  2.1998  |  1.92699 |  4.08044 |          0.2  | DP      |
|  9 |         5 |   8.99798 |  2.99966 |  2.2779  |  4.14413 |          0.2  | DP      |
| 10 |        20 | 274.568   | 16.5701  | 16.0995  | 35.7566  |          0.2  | DP      |
| 11 |        30 | 711.066   | 26.6658  | 25.8131  | 58.4987  |          0.2  | DP      |
| 12 |         1 |   5.6703  |  2.38124 |  2.08048 |  4.3951  |          0.3  | DP      |
| 13 |         5 |   7.69486 |  2.77396 |  2.18031 |  3.96952 |          0.3  | DP      |
| 14 |        20 | 245.131   | 15.6567  | 15.2292  | 33.817   |          0.3  | DP      |
| 15 |        30 | 717.392   | 26.7842  | 26.0059  | 58.9437  |          0.3  | DP      |
| 16 |         1 |   3.00312 |  1.73295 |  1.37193 |  2.88451 |          0.05 | CoT     |
| 17 |         5 |  38.5858  |  6.21174 |  4.37468 |  7.8563  |          0.05 | CoT     |
| 18 |        20 | 116.013   | 10.7709  |  9.22784 | 20.4324  |          0.05 | CoT     |
| 19 |        30 | 202.471   | 14.2292  | 11.6742  | 26.8431  |          0.05 | CoT     |
| 20 |         1 |   2.92508 |  1.71029 |  1.36137 |  2.86093 |          0.1  | CoT     |
| 21 |         5 |  38.3735  |  6.19464 |  4.37437 |  7.85695 |          0.1  | CoT     |
| 22 |        20 | 113.196   | 10.6393  |  9.10068 | 20.1475  |          0.1  | CoT     |
| 23 |        30 | 205.028   | 14.3188  | 11.7458  | 27.0099  |          0.1  | CoT     |
| 24 |         1 |   2.7917  |  1.67084 |  1.34023 |  2.81376 |          0.2  | CoT     |
| 25 |         5 |  37.989   |  6.16352 |  4.37374 |  7.85824 |          0.2  | CoT     |
| 26 |        20 | 107.745   | 10.38    |  8.84636 | 19.5777  |          0.2  | CoT     |
| 27 |        30 | 210.228   | 14.4993  | 11.8892  | 27.3436  |          0.2  | CoT     |
| 28 |         1 |   2.68859 |  1.63969 |  1.3191  |  2.76659 |          0.3  | CoT     |
| 29 |         5 |  37.6575  |  6.13657 |  4.37311 |  7.85953 |          0.3  | CoT     |
| 30 |        20 | 102.541   | 10.1263  |  8.59203 | 19.0079  |          0.3  | CoT     |
| 31 |        30 | 215.543   | 14.6814  | 12.0325  | 27.6773  |          0.3  | CoT     |
| 32 |         1 |   3.1093  |  1.76332 |  1.3957  |  2.93482 |          0.05 | CoT-RF  |
| 33 |         5 |  33.6844  |  5.80383 |  4.56264 |  8.25231 |          0.05 | CoT-RF  |
| 34 |        20 |  40.3568  |  6.3527  |  5.79957 | 12.9298  |          0.05 | CoT-RF  |
| 35 |        30 |  76.6435  |  8.75463 |  7.75371 | 17.4566  |          0.05 | CoT-RF  |
| 36 |         1 |   3.13515 |  1.77064 |  1.4089  |  2.96155 |          0.1  | CoT-RF  |
| 37 |         5 |  33.1955  |  5.76155 |  4.55027 |  8.23272 |          0.1  | CoT-RF  |
| 38 |        20 |  38.4526  |  6.20101 |  5.65914 | 12.617   |          0.1  | CoT-RF  |
| 39 |        30 |  77.025   |  8.77639 |  7.80493 | 17.5753  |          0.1  | CoT-RF  |
| 40 |         1 |   3.20267 |  1.7896  |  1.4353  |  3.015   |          0.2  | CoT-RF  |
| 41 |         5 |  32.2411  |  5.67813 |  4.52554 |  8.19353 |          0.2  | CoT-RF  |
| 42 |        20 |  34.7985  |  5.89903 |  5.37828 | 11.9913  |          0.2  | CoT-RF  |
| 43 |        30 |  77.8257  |  8.82189 |  7.90735 | 17.8126  |          0.2  | CoT-RF  |
| 44 |         1 |   3.29129 |  1.81419 |  1.4617  |  3.06845 |          0.3  | CoT-RF  |
| 45 |         5 |  31.3182  |  5.59626 |  4.50081 |  8.15435 |          0.3  | CoT-RF  |
| 46 |        20 |  31.3503  |  5.59914 |  5.09742 | 11.3657  |          0.3  | CoT-RF  |
| 47 |        30 |  78.6765  |  8.86998 |  8.00978 | 18.05    |          0.3  | CoT-RF  |

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
- Data loader workers: 4
- Pin memory: True

### A.4 Data Availability

Raw data sources:
- ICAP Allowance Price Explorer
- EEX Emissions Auctions
- ECB Exchange Rates
- EIA Energy Prices


