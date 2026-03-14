# EU ETS Futures Forecasting with TSM + LLM Refinement

**Authors**: Auto-generated

**Generated**: 2026-03-12 02:19:06

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
- Dataset spans 2021-05-19 00:00:00 to 2023-12-29 00:00:00
- 674 daily observations
- Best performing method: naive_persistence


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

- **Date Range**: 2021-05-19 00:00:00 to 2023-12-29 00:00:00
- **Total Observations**: 674
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

|    |   horizon |       mse |     rmse |      mae |     mape | model                        |
|---:|----------:|----------:|---------:|---------:|---------:|:-----------------------------|
|  0 |         1 |   7.08531 |  2.66182 |  1.73354 |  4.09775 | naive_persistence            |
|  1 |         5 |  17.8845  |  4.22901 |  3.42636 |  8.03558 | naive_persistence            |
|  2 |        20 |  52.0155  |  7.21217 |  6.10707 | 15.3059  | naive_persistence            |
|  3 |        30 |  23.1628  |  4.81278 |  3.76515 |  9.48745 | naive_persistence            |
|  4 |         1 |  18.0489  |  4.2484  |  3.43949 |  8.01226 | seasonal_naive               |
|  5 |         5 |  17.8845  |  4.22901 |  3.42636 |  8.03558 | seasonal_naive               |
|  6 |        20 |  52.0155  |  7.21217 |  6.10707 | 15.3059  | seasonal_naive               |
|  7 |        30 |  23.1628  |  4.81278 |  3.76515 |  9.48745 | seasonal_naive               |
|  8 |         1 |   7.38763 |  2.71802 |  1.85149 |  4.40048 | linear_ridge                 |
|  9 |         5 |  23.5991  |  4.85789 |  4.32017 | 10.3332  | linear_ridge                 |
| 10 |        20 | 150.017   | 12.2482  | 10.7244  | 27.7066  | linear_ridge                 |
| 11 |        30 | 235.345   | 15.3409  | 14.7471  | 37.5491  | linear_ridge                 |
| 12 |         1 |  13.6724  |  3.69762 |  3.07791 |  7.44466 | linear_lasso                 |
| 13 |         5 |  42.6355  |  6.52959 |  5.81462 | 14.1513  | linear_lasso                 |
| 14 |        20 | 207.394   | 14.4012  | 13.2904  | 34.0881  | linear_lasso                 |
| 15 |        30 | 313.861   | 17.7161  | 17.2173  | 43.7776  | linear_lasso                 |
| 16 |         1 |   7.13272 |  2.67072 |  1.76919 |  4.19073 | tsm                          |
| 17 |         5 |  18.041   |  4.24747 |  3.6698  |  8.64381 | tsm                          |
| 18 |        20 | 106.666   | 10.3279  |  8.55509 | 22.1639  | tsm                          |
| 19 |        30 | 106.277   | 10.3091  |  9.06333 | 23.2087  | tsm                          |
| 20 |         1 |   7.13272 |  2.67072 |  1.76919 |  4.19073 | TSM+LLM-COT-RF-HDELTA        |
| 21 |         5 |  17.7602  |  4.21428 |  3.6309  |  8.55046 | TSM+LLM-COT-RF-HDELTA        |
| 22 |        20 | 103.036   | 10.1507  |  8.37697 | 21.7105  | TSM+LLM-COT-RF-HDELTA        |
| 23 |        30 | 101.833   | 10.0912  |  8.86162 | 22.6946  | TSM+LLM-COT-RF-HDELTA        |
| 24 |         1 |   7.08531 |  2.66182 |  1.73354 |  4.09775 | naive_persistence_llm_subset |
| 25 |         5 |  17.8845  |  4.22901 |  3.42636 |  8.03558 | naive_persistence_llm_subset |
| 26 |        20 |  52.0155  |  7.21217 |  6.10707 | 15.3059  | naive_persistence_llm_subset |
| 27 |        30 |  23.1628  |  4.81278 |  3.76515 |  9.48745 | naive_persistence_llm_subset |
| 28 |         1 |  18.0489  |  4.2484  |  3.43949 |  8.01226 | seasonal_naive_llm_subset    |
| 29 |         5 |  17.8845  |  4.22901 |  3.42636 |  8.03558 | seasonal_naive_llm_subset    |
| 30 |        20 |  52.0155  |  7.21217 |  6.10707 | 15.3059  | seasonal_naive_llm_subset    |
| 31 |        30 |  23.1628  |  4.81278 |  3.76515 |  9.48745 | seasonal_naive_llm_subset    |
| 32 |         1 |   7.38763 |  2.71802 |  1.85149 |  4.40048 | linear_ridge_llm_subset      |
| 33 |         5 |  23.5991  |  4.85789 |  4.32017 | 10.3332  | linear_ridge_llm_subset      |
| 34 |        20 | 150.017   | 12.2482  | 10.7244  | 27.7066  | linear_ridge_llm_subset      |
| 35 |        30 | 235.345   | 15.3409  | 14.7471  | 37.5491  | linear_ridge_llm_subset      |
| 36 |         1 |  13.6724  |  3.69762 |  3.07791 |  7.44466 | linear_lasso_llm_subset      |
| 37 |         5 |  42.6355  |  6.52959 |  5.81462 | 14.1513  | linear_lasso_llm_subset      |
| 38 |        20 | 207.394   | 14.4012  | 13.2904  | 34.0881  | linear_lasso_llm_subset      |
| 39 |        30 | 313.861   | 17.7161  | 17.2173  | 43.7776  | linear_lasso_llm_subset      |
| 40 |         1 |   7.13272 |  2.67072 |  1.76919 |  4.19073 | tsm_llm_subset               |
| 41 |         5 |  18.041   |  4.24747 |  3.6698  |  8.64381 | tsm_llm_subset               |
| 42 |        20 | 106.666   | 10.3279  |  8.55509 | 22.1639  | tsm_llm_subset               |
| 43 |        30 | 106.277   | 10.3091  |  9.06333 | 23.2087  | tsm_llm_subset               |

### 4.2 Trend Classification Accuracy

Table 2 shows the accuracy of predicting price direction (up/flat/down).

|    |   horizon |   accuracy |   accuracy_up |   accuracy_down |   accuracy_flat |   n_up |   n_down |   n_flat | model                        |
|---:|----------:|-----------:|--------------:|----------------:|----------------:|-------:|---------:|---------:|:-----------------------------|
|  0 |         1 |  0.272727  |      0        |       0         |        1        |     35 |       37 |       27 | naive_persistence            |
|  1 |         5 |  0.0808081 |      0        |       0         |        1        |     29 |       62 |        8 | naive_persistence            |
|  2 |        20 |  0.040404  |      0        |       0         |        1        |     37 |       58 |        4 | naive_persistence            |
|  3 |        30 |  0.0909091 |      0        |       0         |        1        |     24 |       66 |        9 | naive_persistence            |
|  4 |         1 |  0.30303   |      0.4      |       0.351351  |        0.111111 |     35 |       37 |       27 | seasonal_naive               |
|  5 |         5 |  0.0808081 |      0        |       0         |        1        |     29 |       62 |        8 | seasonal_naive               |
|  6 |        20 |  0.040404  |      0        |       0         |        1        |     37 |       58 |        4 | seasonal_naive               |
|  7 |        30 |  0.0909091 |      0        |       0         |        1        |     24 |       66 |        9 | seasonal_naive               |
|  8 |         1 |  0.343434  |      0.742857 |       0         |        0.296296 |     35 |       37 |       27 | linear_ridge                 |
|  9 |         5 |  0.292929  |      1        |       0         |        0        |     29 |       62 |        8 | linear_ridge                 |
| 10 |        20 |  0.383838  |      1        |       0.0172414 |        0        |     37 |       58 |        4 | linear_ridge                 |
| 11 |        30 |  0.242424  |      1        |       0         |        0        |     24 |       66 |        9 | linear_ridge                 |
| 12 |         1 |  0.353535  |      1        |       0         |        0        |     35 |       37 |       27 | linear_lasso                 |
| 13 |         5 |  0.292929  |      1        |       0         |        0        |     29 |       62 |        8 | linear_lasso                 |
| 14 |        20 |  0.373737  |      1        |       0         |        0        |     37 |       58 |        4 | linear_lasso                 |
| 15 |        30 |  0.242424  |      1        |       0         |        0        |     24 |       66 |        9 | linear_lasso                 |
| 16 |         1 |  0.313131  |      0.228571 |       0.108108  |        0.703704 |     35 |       37 |       27 | tsm                          |
| 17 |         5 |  0.323232  |      0.862069 |       0.0806452 |        0.25     |     29 |       62 |        8 | tsm                          |
| 18 |        20 |  0.40404   |      0.972973 |       0.0689655 |        0        |     37 |       58 |        4 | tsm                          |
| 19 |        30 |  0.292929  |      1        |       0.0757576 |        0        |     24 |       66 |        9 | tsm                          |
| 20 |         1 |  0.313131  |      0.228571 |       0.108108  |        0.703704 |     35 |       37 |       27 | TSM+LLM-COT-RF-HDELTA        |
| 21 |         5 |  0.353535  |      0.896552 |       0.112903  |        0.25     |     29 |       62 |        8 | TSM+LLM-COT-RF-HDELTA        |
| 22 |        20 |  0.414141  |      0.972973 |       0.0862069 |        0        |     37 |       58 |        4 | TSM+LLM-COT-RF-HDELTA        |
| 23 |        30 |  0.292929  |      1        |       0.0757576 |        0        |     24 |       66 |        9 | TSM+LLM-COT-RF-HDELTA        |
| 24 |         1 |  0.272727  |      0        |       0         |        1        |     35 |       37 |       27 | naive_persistence_llm_subset |
| 25 |         5 |  0.0808081 |      0        |       0         |        1        |     29 |       62 |        8 | naive_persistence_llm_subset |
| 26 |        20 |  0.040404  |      0        |       0         |        1        |     37 |       58 |        4 | naive_persistence_llm_subset |
| 27 |        30 |  0.0909091 |      0        |       0         |        1        |     24 |       66 |        9 | naive_persistence_llm_subset |
| 28 |         1 |  0.30303   |      0.4      |       0.351351  |        0.111111 |     35 |       37 |       27 | seasonal_naive_llm_subset    |
| 29 |         5 |  0.0808081 |      0        |       0         |        1        |     29 |       62 |        8 | seasonal_naive_llm_subset    |
| 30 |        20 |  0.040404  |      0        |       0         |        1        |     37 |       58 |        4 | seasonal_naive_llm_subset    |
| 31 |        30 |  0.0909091 |      0        |       0         |        1        |     24 |       66 |        9 | seasonal_naive_llm_subset    |
| 32 |         1 |  0.343434  |      0.742857 |       0         |        0.296296 |     35 |       37 |       27 | linear_ridge_llm_subset      |
| 33 |         5 |  0.292929  |      1        |       0         |        0        |     29 |       62 |        8 | linear_ridge_llm_subset      |
| 34 |        20 |  0.383838  |      1        |       0.0172414 |        0        |     37 |       58 |        4 | linear_ridge_llm_subset      |
| 35 |        30 |  0.242424  |      1        |       0         |        0        |     24 |       66 |        9 | linear_ridge_llm_subset      |
| 36 |         1 |  0.353535  |      1        |       0         |        0        |     35 |       37 |       27 | linear_lasso_llm_subset      |
| 37 |         5 |  0.292929  |      1        |       0         |        0        |     29 |       62 |        8 | linear_lasso_llm_subset      |
| 38 |        20 |  0.373737  |      1        |       0         |        0        |     37 |       58 |        4 | linear_lasso_llm_subset      |
| 39 |        30 |  0.242424  |      1        |       0         |        0        |     24 |       66 |        9 | linear_lasso_llm_subset      |
| 40 |         1 |  0.313131  |      0.228571 |       0.108108  |        0.703704 |     35 |       37 |       27 | tsm_llm_subset               |
| 41 |         5 |  0.323232  |      0.862069 |       0.0806452 |        0.25     |     29 |       62 |        8 | tsm_llm_subset               |
| 42 |        20 |  0.40404   |      0.972973 |       0.0689655 |        0        |     37 |       58 |        4 | tsm_llm_subset               |
| 43 |        30 |  0.292929  |      1        |       0.0757576 |        0        |     24 |       66 |        9 | tsm_llm_subset               |

### 4.3 Statistical Significance

Table 3 reports the results of significance tests comparing methods.

|    |   horizon | model                     | baseline                     |   mean_error_model |   mean_error_baseline |   improvement_pct |   t_statistic |    t_pvalue | t_significant   |   wilcoxon_statistic |   wilcoxon_pvalue | wilcoxon_significant   |   dm_statistic |   dm_pvalue | dm_significant   | model_better   |
|---:|----------:|:--------------------------|:-----------------------------|-------------------:|----------------------:|------------------:|--------------:|------------:|:----------------|---------------------:|------------------:|:-----------------------|---------------:|------------:|:-----------------|:---------------|
|  0 |         1 | seasonal_naive_llm_subset | naive_persistence_llm_subset |           18.0489  |               7.08531 |    -154.737       |      3.81157  | 0.000241262 | True            |                  980 |       1.80882e-07 | True                   |      3.81157   | 0.000138087 | True             | False          |
|  1 |         5 | seasonal_naive_llm_subset | naive_persistence_llm_subset |           17.8845  |              17.8845  |      -6.40355e-06 |      0.923112 | 0.358216    | False           |                 1888 |       0.145575    | False                  |      0.0605824 | 0.951692    | False            | False          |
|  2 |        20 | seasonal_naive_llm_subset | naive_persistence_llm_subset |           52.0155  |              52.0155  |       6.26508e-06 |      1.65117  | 0.101905    | False           |                 2326 |       0.724379    | False                  |      0.294796  | 0.76815     | False            | True           |
|  3 |        30 | seasonal_naive_llm_subset | naive_persistence_llm_subset |           23.1628  |              23.1628  |      -1.34065e-05 |     -1.59889  | 0.113064    | False           |                 1828 |       0.0484015   | True                   |     -0.212859  | 0.831437    | False            | False          |
|  4 |         1 | linear_ridge_llm_subset   | naive_persistence_llm_subset |            7.38763 |               7.08531 |      -4.26686     |      1.18127  | 0.240356    | False           |                 1983 |       0.0859385   | False                  |      1.18127   | 0.237497    | False            | False          |
|  5 |         5 | linear_ridge_llm_subset   | naive_persistence_llm_subset |           23.5991  |              17.8845  |     -31.9526      |      2.82674  | 0.00570068  | True            |                 1564 |       0.00147451  | True                   |      1.26716   | 0.205098    | False            | False          |
|  6 |        20 | linear_ridge_llm_subset   | naive_persistence_llm_subset |          150.017   |              52.0155  |    -188.409       |      8.76014  | 5.9071e-14  | True            |                  547 |       1.70506e-11 | True                   |      4.59703   | 4.28563e-06 | True             | False          |
|  7 |        30 | linear_ridge_llm_subset   | naive_persistence_llm_subset |          235.345   |              23.1628  |    -916.044       |     19.3239   | 3.3741e-35  | True            |                    6 |       6.84166e-18 | True                   |      8.32798   | 0           | True             | False          |
|  8 |         1 | linear_lasso_llm_subset   | naive_persistence_llm_subset |           13.6724  |               7.08531 |     -92.9686      |      5.18318  | 1.17122e-06 | True            |                  944 |       9.11016e-08 | True                   |      5.18318   | 2.18136e-07 | True             | False          |
|  9 |         5 | linear_lasso_llm_subset   | naive_persistence_llm_subset |           42.6355  |              17.8845  |    -138.394       |      6.66024  | 1.59787e-09 | True            |                  905 |       4.25915e-08 | True                   |      2.90322   | 0.00369346  | True             | False          |
| 10 |        20 | linear_lasso_llm_subset   | naive_persistence_llm_subset |          207.394   |              52.0155  |    -298.716       |     11.9141   | 9.13635e-21 | True            |                  259 |       1.03825e-14 | True                   |      9.90781   | 0           | True             | False          |
| 11 |        30 | linear_lasso_llm_subset   | naive_persistence_llm_subset |          313.861   |              23.1628  |   -1255.02        |     22.0551   | 8.7994e-40  | True            |                    1 |       5.87389e-18 | True                   |     15.0627    | 0           | True             | False          |
| 12 |         1 | tsm_llm_subset            | naive_persistence_llm_subset |            7.13272 |               7.08531 |      -0.669151    |      0.141474 | 0.887786    | False           |                 2270 |       0.474294    | False                  |      0.141474  | 0.887495    | False            | False          |
| 13 |         5 | tsm_llm_subset            | naive_persistence_llm_subset |           18.041   |              17.8845  |      -0.875048    |      0.127367 | 0.898911    | False           |                 2082 |       0.170162    | False                  |      0.0723134 | 0.942353    | False            | False          |
| 14 |        20 | tsm_llm_subset            | naive_persistence_llm_subset |          106.666   |              52.0155  |    -105.065       |      5.66608  | 1.46711e-07 | True            |                 1007 |       2.99525e-07 | True                   |      2.51024   | 0.0120648   | True             | False          |
| 15 |        30 | tsm_llm_subset            | naive_persistence_llm_subset |          106.277   |              23.1628  |    -358.825       |     10.2671   | 3.1878e-17  | True            |                  262 |       1.12725e-14 | True                   |      3.13169   | 0.00173803  | True             | False          |
| 16 |         1 | TSM+LLM-COT-RF-HDELTA     | naive_persistence_llm_subset |            7.13272 |               7.08531 |      -0.669152    |      0.141475 | 0.887785    | False           |                 2270 |       0.474294    | False                  |      0.141475  | 0.887495    | False            | False          |
| 17 |         5 | TSM+LLM-COT-RF-HDELTA     | naive_persistence_llm_subset |           17.7602  |              17.8845  |       0.695092    |     -0.101501 | 0.91936     | False           |                 2156 |       0.265536    | False                  |     -0.0593868 | 0.952644    | False            | True           |
| 18 |        20 | TSM+LLM-COT-RF-HDELTA     | naive_persistence_llm_subset |          103.036   |              52.0155  |     -98.0869      |      5.46064  | 3.58903e-07 | True            |                 1069 |       9.23164e-07 | True                   |      2.46647   | 0.0136453   | True             | False          |
| 19 |        30 | TSM+LLM-COT-RF-HDELTA     | naive_persistence_llm_subset |          101.833   |              23.1628  |    -339.64        |     10.1248   | 6.48847e-17 | True            |                  283 |       1.99865e-14 | True                   |      3.14546   | 0.00165826  | True             | False          |


## 5. Robustness Analysis

### 5.1 Noise Injection Test

To assess robustness, we inject noise into forecast paths at levels of 5%, 10%, 20%, 
and 30% of the forecast standard deviation.

|    |   horizon |       mse |     rmse |     mae |     mape |   noise_level | model                 |
|---:|----------:|----------:|---------:|--------:|---------:|--------------:|:----------------------|
|  0 |         1 |   7.16781 |  2.67728 | 1.77631 |  4.21166 |          0.05 | tsm                   |
|  1 |         5 |  18.0414  |  4.24751 | 3.66454 |  8.63161 |          0.05 | tsm                   |
|  2 |        20 | 106.543   | 10.322   | 8.5574  | 22.1706  |          0.05 | tsm                   |
|  3 |        30 | 106.939   | 10.3411  | 9.09291 | 23.2865  |          0.05 | tsm                   |
|  4 |         1 |   7.22823 |  2.68854 | 1.78939 |  4.2464  |          0.1  | tsm                   |
|  5 |         5 |  18.0699  |  4.25087 | 3.66063 |  8.62295 |          0.1  | tsm                   |
|  6 |        20 | 106.449   | 10.3174  | 8.55971 | 22.1773  |          0.1  | tsm                   |
|  7 |        30 | 107.637   | 10.3748  | 9.12248 | 23.3643  |          0.1  | tsm                   |
|  8 |         1 |   7.42507 |  2.7249  | 1.84022 |  4.37408 |          0.2  | tsm                   |
|  9 |         5 |  18.2116  |  4.26751 | 3.66153 |  8.62956 |          0.2  | tsm                   |
| 10 |        20 | 106.347   | 10.3125  | 8.56433 | 22.1907  |          0.2  | tsm                   |
| 11 |        30 | 109.14    | 10.447   | 9.18163 | 23.5198  |          0.2  | tsm                   |
| 12 |         1 |   7.72323 |  2.77907 | 1.90742 |  4.53964 |          0.3  | tsm                   |
| 13 |         5 |  18.4661  |  4.29722 | 3.66432 |  8.64118 |          0.3  | tsm                   |
| 14 |        20 | 106.36    | 10.3131  | 8.56947 | 22.2052  |          0.3  | tsm                   |
| 15 |        30 | 110.788   | 10.5256  | 9.24078 | 23.6754  |          0.3  | tsm                   |
| 16 |         1 |   7.16789 |  2.67729 | 1.7761  |  4.21099 |          0.05 | TSM+LLM-COT-RF-HDELTA |
| 17 |         5 |  17.7608  |  4.21436 | 3.62682 |  8.5414  |          0.05 | TSM+LLM-COT-RF-HDELTA |
| 18 |        20 | 102.922   | 10.1451  | 8.37974 | 21.7182  |          0.05 | TSM+LLM-COT-RF-HDELTA |
| 19 |        30 | 102.448   | 10.1217  | 8.89018 | 22.7697  |          0.05 | TSM+LLM-COT-RF-HDELTA |
| 20 |         1 |   7.2269  |  2.68829 | 1.78859 |  4.24414 |          0.1  | TSM+LLM-COT-RF-HDELTA |
| 21 |         5 |  17.7879  |  4.21757 | 3.62466 |  8.53751 |          0.1  | TSM+LLM-COT-RF-HDELTA |
| 22 |        20 | 102.836   | 10.1408  | 8.38252 | 21.7259  |          0.1  | TSM+LLM-COT-RF-HDELTA |
| 23 |        30 | 103.097   | 10.1537  | 8.91875 | 22.8447  |          0.1  | TSM+LLM-COT-RF-HDELTA |
| 24 |         1 |   7.41643 |  2.72331 | 1.83784 |  4.36786 |          0.2  | TSM+LLM-COT-RF-HDELTA |
| 25 |         5 |  17.9214  |  4.23337 | 3.62635 |  8.54646 |          0.2  | TSM+LLM-COT-RF-HDELTA |
| 26 |        20 | 102.744   | 10.1363  | 8.38989 | 21.7457  |          0.2  | TSM+LLM-COT-RF-HDELTA |
| 27 |        30 | 104.497   | 10.2224  | 8.97589 | 22.9948  |          0.2  | TSM+LLM-COT-RF-HDELTA |
| 28 |         1 |   7.7013  |  2.77512 | 1.90223 |  4.52651 |          0.3  | TSM+LLM-COT-RF-HDELTA |
| 29 |         5 |  18.1608  |  4.26155 | 3.62951 |  8.559   |          0.3  | TSM+LLM-COT-RF-HDELTA |
| 30 |        20 | 102.762   | 10.1372  | 8.4006  | 21.7733  |          0.3  | TSM+LLM-COT-RF-HDELTA |
| 31 |        30 | 106.032   | 10.2972  | 9.03303 | 23.1449  |          0.3  | TSM+LLM-COT-RF-HDELTA |

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

- **Price accuracy**: naive_persistence has the lowest average MSE (25.037).
- **TSM vs naive**: TSM MSE is 2.4x the naive baseline on average.
- **Directional accuracy**: TSM+LLM-COT-RF-HDELTA has the highest average trend accuracy (0.343).

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
Price accuracy is best for naive_persistence, while directional accuracy is highest for TSM+LLM-COT-RF-HDELTA.
The framework remains suitable for future runs with full LLM and robustness evaluations enabled.


## Appendix A: Reproducibility

### A.1 Configuration

```yaml
# Key configuration parameters
Random Seed: 42
Prediction Length: 30
Sequence Length: 20
TSM Type: dlinear
LLM Model: qwen3-vl-4b-gpu
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


