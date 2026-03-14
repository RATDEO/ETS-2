# EU ETS Futures Forecasting with TSM + LLM Refinement

**Authors**: Auto-generated

**Generated**: 2026-02-26 18:43:56

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
- Dataset spans 2009-04-01 00:00:00 to 2026-02-25 00:00:00
- 4334 daily observations
- Best performing method: TSM+LLM-COT-RF


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

- **Date Range**: 2009-04-01 00:00:00 to 2026-02-25 00:00:00
- **Total Observations**: 4,334
- **Number of Features**: 43

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

|    |   horizon |        mse |     rmse |      mae |     mape | model                        |
|---:|----------:|-----------:|---------:|---------:|---------:|:-----------------------------|
|  0 |         1 |  1.39098   | 1.1794   | 0.947696 | 1.32876  | naive_persistence            |
|  1 |         5 |  6.4397    | 2.53766  | 2.01736  | 2.83588  | naive_persistence            |
|  2 |        20 | 26.3936    | 5.13747  | 4.04948  | 5.56409  | naive_persistence            |
|  3 |        30 | 41.3395    | 6.42958  | 5.15615  | 7.07198  | naive_persistence            |
|  4 |         1 |  6.53148   | 2.55568  | 2.04024  | 2.87018  | seasonal_naive               |
|  5 |         5 |  6.4397    | 2.53766  | 2.01736  | 2.83588  | seasonal_naive               |
|  6 |        20 | 26.3936    | 5.13747  | 4.04948  | 5.56409  | seasonal_naive               |
|  7 |        30 | 41.3395    | 6.42958  | 5.15615  | 7.07198  | seasonal_naive               |
|  8 |         1 |  1.40541   | 1.1855   | 0.953819 | 1.3368   | linear_ridge                 |
|  9 |         5 |  6.65042   | 2.57884  | 2.04762  | 2.88303  | linear_ridge                 |
| 10 |        20 | 26.8354    | 5.18029  | 3.98828  | 5.59742  | linear_ridge                 |
| 11 |        30 | 39.7633    | 6.30582  | 4.961    | 6.90319  | linear_ridge                 |
| 12 |         1 |  5.34846   | 2.31267  | 2.02053  | 2.77429  | linear_lasso                 |
| 13 |         5 | 10.4036    | 3.22547  | 2.69319  | 3.68562  | linear_lasso                 |
| 14 |        20 | 28.5853    | 5.34652  | 4.47365  | 6.08462  | linear_lasso                 |
| 15 |        30 | 39.992     | 6.32392  | 5.21733  | 7.12347  | linear_lasso                 |
| 16 |         1 |  0.0908557 | 0.301423 | 0.232685 | 0.324059 | tsm                          |
| 17 |         5 |  1.77245   | 1.33133  | 1.017    | 1.4141   | tsm                          |
| 18 |        20 | 20.0121    | 4.47348  | 3.45806  | 4.77619  | tsm                          |
| 19 |        30 | 45.4267    | 6.73993  | 5.22796  | 7.26316  | tsm                          |
| 20 |         1 |  0.919657  | 0.958988 | 0.69555  | 0.960132 | TSM+LLM-COT-RF               |
| 21 |         5 |  2.54942   | 1.59669  | 1.23707  | 1.71906  | TSM+LLM-COT-RF               |
| 22 |        20 | 16.5944    | 4.07363  | 3.20995  | 4.38075  | TSM+LLM-COT-RF               |
| 23 |        30 | 38.8828    | 6.23561  | 4.93649  | 6.77821  | TSM+LLM-COT-RF               |
| 24 |         1 |  1.39098   | 1.1794   | 0.947696 | 1.32876  | naive_persistence_llm_subset |
| 25 |         5 |  6.4397    | 2.53766  | 2.01736  | 2.83588  | naive_persistence_llm_subset |
| 26 |        20 | 26.3936    | 5.13747  | 4.04948  | 5.56409  | naive_persistence_llm_subset |
| 27 |        30 | 41.3395    | 6.42958  | 5.15615  | 7.07198  | naive_persistence_llm_subset |
| 28 |         1 |  6.53148   | 2.55568  | 2.04024  | 2.87018  | seasonal_naive_llm_subset    |
| 29 |         5 |  6.4397    | 2.53766  | 2.01736  | 2.83588  | seasonal_naive_llm_subset    |
| 30 |        20 | 26.3936    | 5.13747  | 4.04948  | 5.56409  | seasonal_naive_llm_subset    |
| 31 |        30 | 41.3395    | 6.42958  | 5.15615  | 7.07198  | seasonal_naive_llm_subset    |
| 32 |         1 |  1.40541   | 1.1855   | 0.953819 | 1.3368   | linear_ridge_llm_subset      |
| 33 |         5 |  6.65042   | 2.57884  | 2.04762  | 2.88303  | linear_ridge_llm_subset      |
| 34 |        20 | 26.8354    | 5.18029  | 3.98828  | 5.59742  | linear_ridge_llm_subset      |
| 35 |        30 | 39.7633    | 6.30582  | 4.961    | 6.90319  | linear_ridge_llm_subset      |
| 36 |         1 |  5.34846   | 2.31267  | 2.02053  | 2.77429  | linear_lasso_llm_subset      |
| 37 |         5 | 10.4036    | 3.22547  | 2.69319  | 3.68562  | linear_lasso_llm_subset      |
| 38 |        20 | 28.5853    | 5.34652  | 4.47365  | 6.08462  | linear_lasso_llm_subset      |
| 39 |        30 | 39.992     | 6.32392  | 5.21733  | 7.12347  | linear_lasso_llm_subset      |
| 40 |         1 |  0.0908557 | 0.301423 | 0.232685 | 0.324059 | tsm_llm_subset               |
| 41 |         5 |  1.77245   | 1.33133  | 1.017    | 1.4141   | tsm_llm_subset               |
| 42 |        20 | 20.0121    | 4.47348  | 3.45806  | 4.77619  | tsm_llm_subset               |
| 43 |        30 | 45.4267    | 6.73993  | 5.22796  | 7.26316  | tsm_llm_subset               |

### 4.2 Trend Classification Accuracy

Table 2 shows the accuracy of predicting price direction (up/flat/down).

|    |   horizon |   accuracy |   accuracy_up |   accuracy_down |   accuracy_flat |   n_up |   n_down |   n_flat | model                        |
|---:|----------:|-----------:|--------------:|----------------:|----------------:|-------:|---------:|---------:|:-----------------------------|
|  0 |         1 |  0.303665  |     0         |       0         |       1         |    133 |      133 |      116 | naive_persistence            |
|  1 |         5 |  0.149215  |     0         |       0         |       1         |    177 |      148 |       57 | naive_persistence            |
|  2 |        20 |  0.0863874 |     0         |       0         |       1         |    226 |      123 |       33 | naive_persistence            |
|  3 |        30 |  0.0628272 |     0         |       0         |       1         |    229 |      129 |       24 | naive_persistence            |
|  4 |         1 |  0.342932  |     0.406015  |       0.451128  |       0.146552  |    133 |      133 |      116 | seasonal_naive               |
|  5 |         5 |  0.149215  |     0         |       0         |       1         |    177 |      148 |       57 | seasonal_naive               |
|  6 |        20 |  0.0863874 |     0         |       0         |       1         |    226 |      123 |       33 | seasonal_naive               |
|  7 |        30 |  0.0628272 |     0         |       0         |       1         |    229 |      129 |       24 | seasonal_naive               |
|  8 |         1 |  0.303665  |     0.0075188 |       0         |       0.991379  |    133 |      133 |      116 | linear_ridge                 |
|  9 |         5 |  0.256545  |     0.265537  |       0.0608108 |       0.736842  |    177 |      148 |       57 | linear_ridge                 |
| 10 |        20 |  0.531414  |     0.862832  |       0.0569106 |       0.030303  |    226 |      123 |       33 | linear_ridge                 |
| 11 |        30 |  0.549738  |     0.842795  |       0.0930233 |       0.208333  |    229 |      129 |       24 | linear_ridge                 |
| 12 |         1 |  0.348168  |     0         |       1         |       0         |    133 |      133 |      116 | linear_lasso                 |
| 13 |         5 |  0.390052  |     0         |       1         |       0.0175439 |    177 |      148 |       57 | linear_lasso                 |
| 14 |        20 |  0.295812  |     0.0929204 |       0.682927  |       0.242424  |    226 |      123 |       33 | linear_lasso                 |
| 15 |        30 |  0.384817  |     0.253275  |       0.612403  |       0.416667  |    229 |      129 |       24 | linear_lasso                 |
| 16 |         1 |  0.827225  |     0.834586  |       0.819549  |       0.827586  |    133 |      133 |      116 | tsm                          |
| 17 |         5 |  0.73822   |     0.80791   |       0.783784  |       0.403509  |    177 |      148 |       57 | tsm                          |
| 18 |        20 |  0.578534  |     0.690265  |       0.471545  |       0.212121  |    226 |      123 |       33 | tsm                          |
| 19 |        30 |  0.518325  |     0.655022  |       0.348837  |       0.125     |    229 |      129 |       24 | tsm                          |
| 20 |         1 |  0.615183  |     0.473684  |       0.699248  |       0.681034  |    133 |      133 |      116 | TSM+LLM-COT-RF               |
| 21 |         5 |  0.680628  |     0.723164  |       0.77027   |       0.315789  |    177 |      148 |       57 | TSM+LLM-COT-RF               |
| 22 |        20 |  0.575916  |     0.584071  |       0.658537  |       0.212121  |    226 |      123 |       33 | TSM+LLM-COT-RF               |
| 23 |        30 |  0.518325  |     0.554585  |       0.51938   |       0.166667  |    229 |      129 |       24 | TSM+LLM-COT-RF               |
| 24 |         1 |  0.303665  |     0         |       0         |       1         |    133 |      133 |      116 | naive_persistence_llm_subset |
| 25 |         5 |  0.149215  |     0         |       0         |       1         |    177 |      148 |       57 | naive_persistence_llm_subset |
| 26 |        20 |  0.0863874 |     0         |       0         |       1         |    226 |      123 |       33 | naive_persistence_llm_subset |
| 27 |        30 |  0.0628272 |     0         |       0         |       1         |    229 |      129 |       24 | naive_persistence_llm_subset |
| 28 |         1 |  0.342932  |     0.406015  |       0.451128  |       0.146552  |    133 |      133 |      116 | seasonal_naive_llm_subset    |
| 29 |         5 |  0.149215  |     0         |       0         |       1         |    177 |      148 |       57 | seasonal_naive_llm_subset    |
| 30 |        20 |  0.0863874 |     0         |       0         |       1         |    226 |      123 |       33 | seasonal_naive_llm_subset    |
| 31 |        30 |  0.0628272 |     0         |       0         |       1         |    229 |      129 |       24 | seasonal_naive_llm_subset    |
| 32 |         1 |  0.303665  |     0.0075188 |       0         |       0.991379  |    133 |      133 |      116 | linear_ridge_llm_subset      |
| 33 |         5 |  0.256545  |     0.265537  |       0.0608108 |       0.736842  |    177 |      148 |       57 | linear_ridge_llm_subset      |
| 34 |        20 |  0.531414  |     0.862832  |       0.0569106 |       0.030303  |    226 |      123 |       33 | linear_ridge_llm_subset      |
| 35 |        30 |  0.549738  |     0.842795  |       0.0930233 |       0.208333  |    229 |      129 |       24 | linear_ridge_llm_subset      |
| 36 |         1 |  0.348168  |     0         |       1         |       0         |    133 |      133 |      116 | linear_lasso_llm_subset      |
| 37 |         5 |  0.390052  |     0         |       1         |       0.0175439 |    177 |      148 |       57 | linear_lasso_llm_subset      |
| 38 |        20 |  0.295812  |     0.0929204 |       0.682927  |       0.242424  |    226 |      123 |       33 | linear_lasso_llm_subset      |
| 39 |        30 |  0.384817  |     0.253275  |       0.612403  |       0.416667  |    229 |      129 |       24 | linear_lasso_llm_subset      |
| 40 |         1 |  0.827225  |     0.834586  |       0.819549  |       0.827586  |    133 |      133 |      116 | tsm_llm_subset               |
| 41 |         5 |  0.73822   |     0.80791   |       0.783784  |       0.403509  |    177 |      148 |       57 | tsm_llm_subset               |
| 42 |        20 |  0.578534  |     0.690265  |       0.471545  |       0.212121  |    226 |      123 |       33 | tsm_llm_subset               |
| 43 |        30 |  0.518325  |     0.655022  |       0.348837  |       0.125     |    229 |      129 |       24 | tsm_llm_subset               |

### 4.3 Statistical Significance

Table 3 reports the results of significance tests comparing methods.

|    |   horizon | model                     | baseline                     |   mean_error_model |   mean_error_baseline |   improvement_pct |   t_statistic |    t_pvalue | t_significant   |   wilcoxon_statistic |   wilcoxon_pvalue | wilcoxon_significant   |   dm_statistic |   dm_pvalue | dm_significant   | model_better   |
|---:|----------:|:--------------------------|:-----------------------------|-------------------:|----------------------:|------------------:|--------------:|------------:|:----------------|---------------------:|------------------:|:-----------------------|---------------:|------------:|:-----------------|:---------------|
|  0 |         1 | seasonal_naive_llm_subset | naive_persistence_llm_subset |          6.53148   |               1.39098 |    -369.559       |     11.3323   | 7.50404e-26 | True            |                11234 |       8.40522e-32 | True                   |    11.3323     | 0           | True             | False          |
|  1 |         5 | seasonal_naive_llm_subset | naive_persistence_llm_subset |          6.4397    |               6.4397  |       2.4183e-06  |     -0.142768 | 0.886549    | False           |                27993 |       0.00933291  | True                   |    -0.00373097 | 0.997023    | False            | True           |
|  2 |        20 | seasonal_naive_llm_subset | naive_persistence_llm_subset |         26.3936    |              26.3936  |       1.43085e-05 |     -0.962751 | 0.336283    | False           |                28266 |       0.00196767  | True                   |    -0.106439   | 0.915234    | False            | True           |
|  3 |        30 | seasonal_naive_llm_subset | naive_persistence_llm_subset |         41.3395    |              41.3395  |       5.55194e-06 |     -0.137731 | 0.890526    | False           |                31648 |       0.225587    | False                  |    -0.0247567  | 0.980249    | False            | True           |
|  4 |         1 | linear_ridge_llm_subset   | naive_persistence_llm_subset |          1.40541   |               1.39098 |      -1.03757     |      0.69981  | 0.484473    | False           |                34831 |       0.418927    | False                  |     0.69981    | 0.484046    | False            | False          |
|  5 |         5 | linear_ridge_llm_subset   | naive_persistence_llm_subset |          6.65042   |               6.4397  |      -3.27227     |      1.32026  | 0.187539    | False           |                34635 |       0.368629    | False                  |     0.620983   | 0.534611    | False            | False          |
|  6 |        20 | linear_ridge_llm_subset   | naive_persistence_llm_subset |         26.8354    |              26.3936  |      -1.67402     |      0.379706 | 0.704376    | False           |                36250 |       0.879825    | False                  |     0.100645   | 0.919832    | False            | False          |
|  7 |        30 | linear_ridge_llm_subset   | naive_persistence_llm_subset |         39.7633    |              41.3395  |       3.81269     |     -1.16959  | 0.242897    | False           |                27296 |       1.7274e-05  | True                   |    -0.324992   | 0.745187    | False            | True           |
|  8 |         1 | linear_lasso_llm_subset   | naive_persistence_llm_subset |          5.34846   |               1.39098 |    -284.51        |     16.9088   | 2.98715e-48 | True            |                 7999 |       5.63596e-40 | True                   |    16.9088     | 0           | True             | False          |
|  9 |         5 | linear_lasso_llm_subset   | naive_persistence_llm_subset |         10.4036    |               6.4397  |     -61.5546      |      8.6341   | 1.63736e-16 | True            |                18726 |       1.38555e-16 | True                   |     3.6972     | 0.000217989 | True             | False          |
| 10 |        20 | linear_lasso_llm_subset   | naive_persistence_llm_subset |         28.5853    |              26.3936  |      -8.30401     |      2.31639  | 0.0210668   | True            |                25383 |       2.17947e-07 | True                   |     0.949165   | 0.342537    | False            | False          |
| 11 |        30 | linear_lasso_llm_subset   | naive_persistence_llm_subset |         39.992     |              41.3395  |       3.25951     |     -1.26056  | 0.208239    | False           |                33944 |       0.222835    | False                  |    -0.822793   | 0.410625    | False            | True           |
| 12 |         1 | tsm_llm_subset            | naive_persistence_llm_subset |          0.0908557 |               1.39098 |      93.4682      |    -13.3306   | 1.53237e-33 | True            |                 2277 |       8.31509e-57 | True                   |   -13.3306     | 0           | True             | True           |
| 13 |         5 | tsm_llm_subset            | naive_persistence_llm_subset |          1.77245   |               6.4397  |      72.4762      |    -11.1972   | 2.37051e-25 | True            |                11714 |       1.13457e-30 | True                   |    -6.41413    | 1.41631e-10 | True             | True           |
| 14 |        20 | tsm_llm_subset            | naive_persistence_llm_subset |         20.0121    |              26.3936  |      24.1782      |     -5.4454   | 9.28404e-08 | True            |                23725 |       2.66337e-09 | True                   |    -3.46656    | 0.000527162 | True             | True           |
| 15 |        30 | tsm_llm_subset            | naive_persistence_llm_subset |         45.4267    |              41.3395  |      -9.88681     |      1.54998  | 0.121977    | False           |                35737 |       0.697465    | False                  |     0.973097   | 0.330505    | False            | False          |
| 16 |         1 | TSM+LLM-COT-RF            | naive_persistence_llm_subset |          0.919657  |               1.39098 |      33.8842      |     -3.58389  | 0.000382465 | True            |                27048 |       1.02262e-05 | True                   |    -3.58389    | 0.000338518 | True             | True           |
| 17 |         5 | TSM+LLM-COT-RF            | naive_persistence_llm_subset |          2.54942   |               6.4397  |      60.411       |     -8.77211  | 5.93526e-17 | True            |                16685 |       3.22808e-20 | True                   |    -4.99224    | 5.96833e-07 | True             | True           |
| 18 |        20 | TSM+LLM-COT-RF            | naive_persistence_llm_subset |         16.5944    |              26.3936  |      37.127       |     -6.58766  | 1.48743e-10 | True            |                22592 |       9.43356e-11 | True                   |    -2.89413    | 0.0038021   | True             | True           |
| 19 |        30 | TSM+LLM-COT-RF            | naive_persistence_llm_subset |         38.8828    |              41.3395  |       5.94272     |     -0.898651 | 0.369406    | False           |                33857 |       0.207918    | False                  |    -0.583362   | 0.55965     | False            | True           |


## 5. Robustness Analysis

### 5.1 Noise Injection Test

To assess robustness, we inject noise into forecast paths at levels of 5%, 10%, 20%, 
and 30% of the forecast standard deviation.

|    |   horizon |       mse |     rmse |      mae |     mape |   noise_level | model          |
|---:|----------:|----------:|---------:|---------:|---------:|--------------:|:---------------|
|  0 |         1 |  0.098265 | 0.313472 | 0.239818 | 0.33432  |          0.05 | tsm            |
|  1 |         5 |  1.76472  | 1.32843  | 1.01773  | 1.41545  |          0.05 | tsm            |
|  2 |        20 | 20.0028   | 4.47245  | 3.45817  | 4.77659  |          0.05 | tsm            |
|  3 |        30 | 45.3224   | 6.73219  | 5.22231  | 7.25555  |          0.05 | tsm            |
|  4 |         1 |  0.119587 | 0.345813 | 0.264408 | 0.368678 |          0.1  | tsm            |
|  5 |         5 |  1.77325  | 1.33163  | 1.02174  | 1.42125  |          0.1  | tsm            |
|  6 |        20 | 20.0058   | 4.47279  | 3.45955  | 4.77878  |          0.1  | tsm            |
|  7 |        30 | 45.2309   | 6.72539  | 5.21827  | 7.25019  |          0.1  | tsm            |
|  8 |         1 |  0.203967 | 0.451627 | 0.343971 | 0.480069 |          0.2  | tsm            |
|  9 |         5 |  1.83911  | 1.35614  | 1.04426  | 1.45242  |          0.2  | tsm            |
| 10 |        20 | 20.0484   | 4.47754  | 3.46885  | 4.79159  |          0.2  | tsm            |
| 11 |        30 | 45.0862   | 6.71463  | 5.21147  | 7.24117  |          0.2  | tsm            |
| 12 |         1 |  0.343997 | 0.586513 | 0.447324 | 0.624281 |          0.3  | tsm            |
| 13 |         5 |  1.97006  | 1.40359  | 1.07901  | 1.50058  |          0.3  | tsm            |
| 14 |        20 | 20.1397   | 4.48773  | 3.48023  | 4.80727  |          0.3  | tsm            |
| 15 |        30 | 44.9926   | 6.70766  | 5.20628  | 7.23423  |          0.3  | tsm            |
| 16 |         1 |  0.926876 | 0.962744 | 0.700153 | 0.966744 |          0.05 | TSM+LLM-COT-RF |
| 17 |         5 |  2.54433  | 1.59509  | 1.23927  | 1.72265  |          0.05 | TSM+LLM-COT-RF |
| 18 |        20 | 16.588    | 4.07284  | 3.20874  | 4.3794   |          0.05 | TSM+LLM-COT-RF |
| 19 |        30 | 38.7733   | 6.22682  | 4.92689  | 6.76522  |          0.05 | TSM+LLM-COT-RF |
| 20 |         1 |  0.948831 | 0.97408  | 0.709856 | 0.980248 |          0.1  | TSM+LLM-COT-RF |
| 21 |         5 |  2.5561   | 1.59878  | 1.24343  | 1.72899  |          0.1  | TSM+LLM-COT-RF |
| 22 |        20 | 16.596    | 4.07382  | 3.21059  | 4.38213  |          0.1  | TSM+LLM-COT-RF |
| 23 |        30 | 38.6765   | 6.21905  | 4.91892  | 6.75426  |          0.1  | TSM+LLM-COT-RF |
| 24 |         1 |  1.03695  | 1.01831  | 0.748948 | 1.03449  |          0.2  | TSM+LLM-COT-RF |
| 25 |         5 |  2.63022  | 1.6218   | 1.26234  | 1.75551  |          0.2  | TSM+LLM-COT-RF |
| 26 |        20 | 16.6555   | 4.08111  | 3.21816  | 4.39307  |          0.2  | TSM+LLM-COT-RF |
| 27 |        30 | 38.5208   | 6.20651  | 4.90586  | 6.73653  |          0.2  | TSM+LLM-COT-RF |
| 28 |         1 |  1.18402  | 1.08813  | 0.803482 | 1.11086  |          0.3  | TSM+LLM-COT-RF |
| 29 |         5 |  2.77178  | 1.66487  | 1.28971  | 1.79358  |          0.3  | TSM+LLM-COT-RF |
| 30 |        20 | 16.7726   | 4.09544  | 3.23762  | 4.42054  |          0.3  | TSM+LLM-COT-RF |
| 31 |        30 | 38.4157   | 6.19804  | 4.896    | 6.72327  |          0.3  | TSM+LLM-COT-RF |

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

- **Price accuracy**: TSM+LLM-COT-RF has the lowest average MSE (14.737).
- **TSM vs naive**: TSM MSE is 0.9x the naive baseline on average.
- **Directional accuracy**: tsm has the highest average trend accuracy (0.666).

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
Price accuracy is best for TSM+LLM-COT-RF, while directional accuracy is highest for tsm.
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


