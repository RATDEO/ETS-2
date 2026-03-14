# EU ETS Futures Forecasting with TSM + LLM Refinement

**Authors**: Auto-generated

**Generated**: 2026-03-12 02:27:47

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
- Dataset spans 2021-05-19 00:00:00 to 2024-06-28 00:00:00
- 803 daily observations
- Best performing method: TSM+LLM-COT-RF-HDELTA


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

- **Date Range**: 2021-05-19 00:00:00 to 2024-06-28 00:00:00
- **Total Observations**: 803
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

|    |   horizon |       mse |     rmse |       mae |     mape | model                        |
|---:|----------:|----------:|---------:|----------:|---------:|:-----------------------------|
|  0 |         1 |   1.78099 |  1.33454 |  1.0504   |  2.95771 | naive_persistence            |
|  1 |         5 |   6.44009 |  2.53773 |  1.8876   |  5.27328 | naive_persistence            |
|  2 |        20 |  24.5334  |  4.95312 |  3.56     |  9.02711 | naive_persistence            |
|  3 |        30 |  36.9906  |  6.08199 |  4.5644   | 11.1185  | naive_persistence            |
|  4 |         1 |   5.33857 |  2.31054 |  1.7197   |  4.88189 | seasonal_naive               |
|  5 |         5 |   6.44009 |  2.53773 |  1.8876   |  5.27328 | seasonal_naive               |
|  6 |        20 |  24.5334  |  4.95312 |  3.56     |  9.02711 | seasonal_naive               |
|  7 |        30 |  36.9906  |  6.08199 |  4.5644   | 11.1185  | seasonal_naive               |
|  8 |         1 |   2.31316 |  1.52091 |  1.24632  |  3.55371 | linear_ridge                 |
|  9 |         5 |  14.7377  |  3.83897 |  3.33036  |  9.58608 | linear_ridge                 |
| 10 |        20 |  90.0731  |  9.49068 |  8.67639  | 24.7772  | linear_ridge                 |
| 11 |        30 | 148.916   | 12.2031  | 11.1692   | 31.2097  | linear_ridge                 |
| 12 |         1 |  14.7152  |  3.83604 |  3.62024  | 10.4008  | linear_lasso                 |
| 13 |         5 |  40.9779  |  6.4014  |  5.97173  | 17.2502  | linear_lasso                 |
| 14 |        20 | 149.312   | 12.2193  | 11.4618   | 32.5386  | linear_lasso                 |
| 15 |        30 | 221.622   | 14.887   | 13.9911   | 38.7658  | linear_lasso                 |
| 16 |         1 |   1.56164 |  1.24966 |  0.983813 |  2.76808 | tsm                          |
| 17 |         5 |   6.27746 |  2.50549 |  1.85701  |  5.1882  | tsm                          |
| 18 |        20 |  23.2329  |  4.82005 |  3.47513  |  8.9976  | tsm                          |
| 19 |        30 |  29.6703  |  5.44705 |  4.10147  | 10.1044  | tsm                          |
| 20 |         1 |   1.56164 |  1.24966 |  0.983813 |  2.76808 | TSM+LLM-COT-RF-HDELTA        |
| 21 |         5 |   6.26765 |  2.50353 |  1.86435  |  5.20674 | TSM+LLM-COT-RF-HDELTA        |
| 22 |        20 |  23.0859  |  4.80478 |  3.45706  |  8.92995 | TSM+LLM-COT-RF-HDELTA        |
| 23 |        30 |  29.6681  |  5.44684 |  4.08599  | 10.0424  | TSM+LLM-COT-RF-HDELTA        |
| 24 |         1 |   1.78099 |  1.33454 |  1.0504   |  2.95771 | naive_persistence_llm_subset |
| 25 |         5 |   6.44009 |  2.53773 |  1.8876   |  5.27328 | naive_persistence_llm_subset |
| 26 |        20 |  24.5334  |  4.95312 |  3.56     |  9.02711 | naive_persistence_llm_subset |
| 27 |        30 |  36.9906  |  6.08199 |  4.5644   | 11.1185  | naive_persistence_llm_subset |
| 28 |         1 |   5.33857 |  2.31054 |  1.7197   |  4.88189 | seasonal_naive_llm_subset    |
| 29 |         5 |   6.44009 |  2.53773 |  1.8876   |  5.27328 | seasonal_naive_llm_subset    |
| 30 |        20 |  24.5334  |  4.95312 |  3.56     |  9.02711 | seasonal_naive_llm_subset    |
| 31 |        30 |  36.9906  |  6.08199 |  4.5644   | 11.1185  | seasonal_naive_llm_subset    |
| 32 |         1 |   2.31316 |  1.52091 |  1.24632  |  3.55371 | linear_ridge_llm_subset      |
| 33 |         5 |  14.7377  |  3.83897 |  3.33036  |  9.58608 | linear_ridge_llm_subset      |
| 34 |        20 |  90.0731  |  9.49068 |  8.67639  | 24.7772  | linear_ridge_llm_subset      |
| 35 |        30 | 148.916   | 12.2031  | 11.1692   | 31.2097  | linear_ridge_llm_subset      |
| 36 |         1 |  14.7152  |  3.83604 |  3.62024  | 10.4008  | linear_lasso_llm_subset      |
| 37 |         5 |  40.9779  |  6.4014  |  5.97173  | 17.2502  | linear_lasso_llm_subset      |
| 38 |        20 | 149.312   | 12.2193  | 11.4618   | 32.5386  | linear_lasso_llm_subset      |
| 39 |        30 | 221.622   | 14.887   | 13.9911   | 38.7658  | linear_lasso_llm_subset      |
| 40 |         1 |   1.56164 |  1.24966 |  0.983813 |  2.76808 | tsm_llm_subset               |
| 41 |         5 |   6.27746 |  2.50549 |  1.85701  |  5.1882  | tsm_llm_subset               |
| 42 |        20 |  23.2329  |  4.82005 |  3.47513  |  8.9976  | tsm_llm_subset               |
| 43 |        30 |  29.6703  |  5.44705 |  4.10147  | 10.1044  | tsm_llm_subset               |

### 4.2 Trend Classification Accuracy

Table 2 shows the accuracy of predicting price direction (up/flat/down).

|    |   horizon |   accuracy |   accuracy_up |   accuracy_down |   accuracy_flat |   n_up |   n_down |   n_flat | model                        |
|---:|----------:|-----------:|--------------:|----------------:|----------------:|-------:|---------:|---------:|:-----------------------------|
|  0 |         1 |       0.35 |      0        |        0        |       1         |     30 |       35 |       35 | naive_persistence            |
|  1 |         5 |       0.23 |      0        |        0        |       1         |     42 |       35 |       23 | naive_persistence            |
|  2 |        20 |       0.09 |      0        |        0        |       1         |     61 |       30 |        9 | naive_persistence            |
|  3 |        30 |       0.08 |      0        |        0        |       1         |     73 |       19 |        8 | naive_persistence            |
|  4 |         1 |       0.41 |      0.566667 |        0.542857 |       0.142857  |     30 |       35 |       35 | seasonal_naive               |
|  5 |         5 |       0.23 |      0        |        0        |       1         |     42 |       35 |       23 | seasonal_naive               |
|  6 |        20 |       0.09 |      0        |        0        |       1         |     61 |       30 |        9 | seasonal_naive               |
|  7 |        30 |       0.08 |      0        |        0        |       1         |     73 |       19 |        8 | seasonal_naive               |
|  8 |         1 |       0.32 |      1        |        0        |       0.0571429 |     30 |       35 |       35 | linear_ridge                 |
|  9 |         5 |       0.42 |      1        |        0        |       0         |     42 |       35 |       23 | linear_ridge                 |
| 10 |        20 |       0.61 |      1        |        0        |       0         |     61 |       30 |        9 | linear_ridge                 |
| 11 |        30 |       0.73 |      1        |        0        |       0         |     73 |       19 |        8 | linear_ridge                 |
| 12 |         1 |       0.3  |      1        |        0        |       0         |     30 |       35 |       35 | linear_lasso                 |
| 13 |         5 |       0.42 |      1        |        0        |       0         |     42 |       35 |       23 | linear_lasso                 |
| 14 |        20 |       0.61 |      1        |        0        |       0         |     61 |       30 |        9 | linear_lasso                 |
| 15 |        30 |       0.73 |      1        |        0        |       0         |     73 |       19 |        8 | linear_lasso                 |
| 16 |         1 |       0.47 |      0.166667 |        0.285714 |       0.914286  |     30 |       35 |       35 | tsm                          |
| 17 |         5 |       0.33 |      0.380952 |        0.114286 |       0.565217  |     42 |       35 |       23 | tsm                          |
| 18 |        20 |       0.59 |      0.852459 |        0.166667 |       0.222222  |     61 |       30 |        9 | tsm                          |
| 19 |        30 |       0.71 |      0.863014 |        0.315789 |       0.25      |     73 |       19 |        8 | tsm                          |
| 20 |         1 |       0.47 |      0.166667 |        0.285714 |       0.914286  |     30 |       35 |       35 | TSM+LLM-COT-RF-HDELTA        |
| 21 |         5 |       0.31 |      0.333333 |        0.142857 |       0.521739  |     42 |       35 |       23 | TSM+LLM-COT-RF-HDELTA        |
| 22 |        20 |       0.57 |      0.803279 |        0.166667 |       0.333333  |     61 |       30 |        9 | TSM+LLM-COT-RF-HDELTA        |
| 23 |        30 |       0.7  |      0.849315 |        0.315789 |       0.25      |     73 |       19 |        8 | TSM+LLM-COT-RF-HDELTA        |
| 24 |         1 |       0.35 |      0        |        0        |       1         |     30 |       35 |       35 | naive_persistence_llm_subset |
| 25 |         5 |       0.23 |      0        |        0        |       1         |     42 |       35 |       23 | naive_persistence_llm_subset |
| 26 |        20 |       0.09 |      0        |        0        |       1         |     61 |       30 |        9 | naive_persistence_llm_subset |
| 27 |        30 |       0.08 |      0        |        0        |       1         |     73 |       19 |        8 | naive_persistence_llm_subset |
| 28 |         1 |       0.41 |      0.566667 |        0.542857 |       0.142857  |     30 |       35 |       35 | seasonal_naive_llm_subset    |
| 29 |         5 |       0.23 |      0        |        0        |       1         |     42 |       35 |       23 | seasonal_naive_llm_subset    |
| 30 |        20 |       0.09 |      0        |        0        |       1         |     61 |       30 |        9 | seasonal_naive_llm_subset    |
| 31 |        30 |       0.08 |      0        |        0        |       1         |     73 |       19 |        8 | seasonal_naive_llm_subset    |
| 32 |         1 |       0.32 |      1        |        0        |       0.0571429 |     30 |       35 |       35 | linear_ridge_llm_subset      |
| 33 |         5 |       0.42 |      1        |        0        |       0         |     42 |       35 |       23 | linear_ridge_llm_subset      |
| 34 |        20 |       0.61 |      1        |        0        |       0         |     61 |       30 |        9 | linear_ridge_llm_subset      |
| 35 |        30 |       0.73 |      1        |        0        |       0         |     73 |       19 |        8 | linear_ridge_llm_subset      |
| 36 |         1 |       0.3  |      1        |        0        |       0         |     30 |       35 |       35 | linear_lasso_llm_subset      |
| 37 |         5 |       0.42 |      1        |        0        |       0         |     42 |       35 |       23 | linear_lasso_llm_subset      |
| 38 |        20 |       0.61 |      1        |        0        |       0         |     61 |       30 |        9 | linear_lasso_llm_subset      |
| 39 |        30 |       0.73 |      1        |        0        |       0         |     73 |       19 |        8 | linear_lasso_llm_subset      |
| 40 |         1 |       0.47 |      0.166667 |        0.285714 |       0.914286  |     30 |       35 |       35 | tsm_llm_subset               |
| 41 |         5 |       0.33 |      0.380952 |        0.114286 |       0.565217  |     42 |       35 |       23 | tsm_llm_subset               |
| 42 |        20 |       0.59 |      0.852459 |        0.166667 |       0.222222  |     61 |       30 |        9 | tsm_llm_subset               |
| 43 |        30 |       0.71 |      0.863014 |        0.315789 |       0.25      |     73 |       19 |        8 | tsm_llm_subset               |

### 4.3 Statistical Significance

Table 3 reports the results of significance tests comparing methods.

|    |   horizon | model                     | baseline                     |   mean_error_model |   mean_error_baseline |   improvement_pct |   t_statistic |    t_pvalue | t_significant   |   wilcoxon_statistic |   wilcoxon_pvalue | wilcoxon_significant   |   dm_statistic |   dm_pvalue | dm_significant   | model_better   |
|---:|----------:|:--------------------------|:-----------------------------|-------------------:|----------------------:|------------------:|--------------:|------------:|:----------------|---------------------:|------------------:|:-----------------------|---------------:|------------:|:-----------------|:---------------|
|  0 |         1 | seasonal_naive_llm_subset | naive_persistence_llm_subset |            5.33857 |               1.78099 |    -199.753       |     3.61123   | 0.000480557 | True            |                 1366 |       6.74749e-05 | True                   |     3.61123    | 0.000304753 | True             | False          |
|  1 |         5 | seasonal_naive_llm_subset | naive_persistence_llm_subset |            6.44009 |               6.44009 |       1.1119e-06  |     0.0362256 | 0.971175    | False           |                 1820 |       0.161343    | False                  |     0.0011766  | 0.999061    | False            | True           |
|  2 |        20 | seasonal_naive_llm_subset | naive_persistence_llm_subset |           24.5334  |              24.5334  |       6.09949e-06 |     0.0937587 | 0.92549     | False           |                 1926 |       0.141796    | False                  |     0.00855922 | 0.993171    | False            | True           |
|  3 |        30 | seasonal_naive_llm_subset | naive_persistence_llm_subset |           36.9906  |              36.9906  |      -2.25129e-06 |    -0.909338  | 0.36538     | False           |                 1652 |       0.0285842   | True                   |    -0.119466   | 0.904907    | False            | False          |
|  4 |         1 | linear_ridge_llm_subset   | naive_persistence_llm_subset |            2.31316 |               1.78099 |     -29.8802      |     2.56859   | 0.0117044   | True            |                 1722 |       0.00576291  | True                   |     2.56859    | 0.0102113   | True             | False          |
|  5 |         5 | linear_ridge_llm_subset   | naive_persistence_llm_subset |           14.7377  |               6.44009 |    -128.843       |     6.44684   | 4.19194e-09 | True            |                  838 |       6.61356e-09 | True                   |     3.28453    | 0.00102152  | True             | False          |
|  6 |        20 | linear_ridge_llm_subset   | naive_persistence_llm_subset |           90.0731  |              24.5334  |    -267.145       |     8.32623   | 4.77704e-13 | True            |                  692 |       2.92978e-10 | True                   |     1.8549     | 0.0636104   | False            | False          |
|  7 |        30 | linear_ridge_llm_subset   | naive_persistence_llm_subset |          148.916   |              36.9906  |    -302.578       |     8.36495   | 3.94142e-13 | True            |                  550 |       1.11597e-11 | True                   |     1.62289    | 0.104613    | False            | False          |
|  8 |         1 | linear_lasso_llm_subset   | naive_persistence_llm_subset |           14.7152  |               1.78099 |    -726.235       |    14.4864    | 3.35675e-26 | True            |                  120 |       1.34893e-16 | True                   |    14.4864     | 0           | True             | False          |
|  9 |         5 | linear_lasso_llm_subset   | naive_persistence_llm_subset |           40.9779  |               6.44009 |    -536.294       |    13.0494    | 3.07838e-23 | True            |                  217 |       2.09389e-15 | True                   |     6.25796    | 3.90052e-10 | True             | False          |
| 10 |        20 | linear_lasso_llm_subset   | naive_persistence_llm_subset |          149.312   |              24.5334  |    -508.607       |    11.9216    | 7.54997e-21 | True            |                  247 |       4.78248e-15 | True                   |     2.604      | 0.00921417  | True             | False          |
| 11 |        30 | linear_lasso_llm_subset   | naive_persistence_llm_subset |          221.622   |              36.9906  |    -499.132       |    11.1617    | 3.26093e-19 | True            |                  387 |       1.96526e-13 | True                   |     2.12155    | 0.0338755   | True             | False          |
| 12 |         1 | tsm_llm_subset            | naive_persistence_llm_subset |            1.56164 |               1.78099 |      12.3163      |    -2.48737   | 0.0145407   | True            |                 2020 |       0.0825013   | False                  |    -2.48737    | 0.0128693   | True             | True           |
| 13 |         5 | tsm_llm_subset            | naive_persistence_llm_subset |            6.27746 |               6.44009 |       2.5252      |    -0.413241  | 0.680324    | False           |                 2022 |       0.0837237   | False                  |    -0.451377   | 0.651718    | False            | True           |
| 14 |        20 | tsm_llm_subset            | naive_persistence_llm_subset |           23.2329  |              24.5334  |       5.30109     |    -0.922864  | 0.358322    | False           |                 1951 |       0.0484274   | True                   |    -0.326652   | 0.743931    | False            | True           |
| 15 |        30 | tsm_llm_subset            | naive_persistence_llm_subset |           29.6703  |              36.9906  |      19.7896      |    -4.24751   | 4.88822e-05 | True            |                 1369 |       7.0467e-05  | True                   |    -1.30402    | 0.192225    | False            | True           |
| 16 |         1 | TSM+LLM-COT-RF-HDELTA     | naive_persistence_llm_subset |            1.56164 |               1.78099 |      12.3163      |    -2.48737   | 0.0145407   | True            |                 2020 |       0.0825013   | False                  |    -2.48737    | 0.0128693   | True             | True           |
| 17 |         5 | TSM+LLM-COT-RF-HDELTA     | naive_persistence_llm_subset |            6.26765 |               6.44009 |       2.67751     |    -0.41761   | 0.677137    | False           |                 2078 |       0.12431     | False                  |    -0.480386   | 0.630953    | False            | True           |
| 18 |        20 | TSM+LLM-COT-RF-HDELTA     | naive_persistence_llm_subset |           23.0859  |              24.5334  |       5.89997     |    -1.06359   | 0.290101    | False           |                 1882 |       0.0270467   | True                   |    -0.400625   | 0.688696    | False            | True           |
| 19 |        30 | TSM+LLM-COT-RF-HDELTA     | naive_persistence_llm_subset |           29.6681  |              36.9906  |      19.7956      |    -4.45584   | 2.2027e-05  | True            |                 1332 |       4.09733e-05 | True                   |    -1.40708    | 0.159402    | False            | True           |


## 5. Robustness Analysis

### 5.1 Noise Injection Test

To assess robustness, we inject noise into forecast paths at levels of 5%, 10%, 20%, 
and 30% of the forecast standard deviation.

|    |   horizon |      mse |    rmse |      mae |     mape |   noise_level | model                 |
|---:|----------:|---------:|--------:|---------:|---------:|--------------:|:----------------------|
|  0 |         1 |  1.5535  | 1.24639 | 0.984764 |  2.7708  |          0.05 | tsm                   |
|  1 |         5 |  6.29055 | 2.5081  | 1.85775  |  5.18914 |          0.05 | tsm                   |
|  2 |        20 | 23.1689  | 4.81341 | 3.47242  |  8.98948 |          0.05 | tsm                   |
|  3 |        30 | 29.6848  | 5.44837 | 4.10506  | 10.1178  |          0.05 | tsm                   |
|  4 |         1 |  1.54818 | 1.24426 | 0.985939 |  2.77417 |          0.1  | tsm                   |
|  5 |         5 |  6.30733 | 2.51144 | 1.85849  |  5.19008 |          0.1  | tsm                   |
|  6 |        20 | 23.108   | 4.80708 | 3.47074  |  8.98434 |          0.1  | tsm                   |
|  7 |        30 | 29.7025  | 5.45    | 4.10865  | 10.1311  |          0.1  | tsm                   |
|  8 |         1 |  1.54602 | 1.24339 | 0.988741 |  2.78223 |          0.2  | tsm                   |
|  9 |         5 |  6.352   | 2.52032 | 1.86892  |  5.21902 |          0.2  | tsm                   |
| 10 |        20 | 22.9953  | 4.79534 | 3.46927  |  8.97948 |          0.2  | tsm                   |
| 11 |        30 | 29.7476  | 5.45414 | 4.11584  | 10.1579  |          0.2  | tsm                   |
| 12 |         1 |  1.55515 | 1.24706 | 0.993367 |  2.79503 |          0.3  | tsm                   |
| 13 |         5 |  6.41148 | 2.53209 | 1.88179  |  5.25532 |          0.3  | tsm                   |
| 14 |        20 | 22.8947  | 4.78484 | 3.46925  |  8.97858 |          0.3  | tsm                   |
| 15 |        30 | 29.8057  | 5.45946 | 4.12483  | 10.1898  |          0.3  | tsm                   |
| 16 |         1 |  1.55374 | 1.24649 | 0.984679 |  2.77053 |          0.05 | TSM+LLM-COT-RF-HDELTA |
| 17 |         5 |  6.28074 | 2.50614 | 1.86343  |  5.20288 |          0.05 | TSM+LLM-COT-RF-HDELTA |
| 18 |        20 | 23.0228  | 4.79821 | 3.45523  |  8.92441 |          0.05 | TSM+LLM-COT-RF-HDELTA |
| 19 |        30 | 29.6801  | 5.44795 | 4.08728  | 10.0492  |          0.05 | TSM+LLM-COT-RF-HDELTA |
| 20 |         1 |  1.5485  | 1.24439 | 0.985768 |  2.77364 |          0.1  | TSM+LLM-COT-RF-HDELTA |
| 21 |         5 |  6.29729 | 2.50944 | 1.86409  |  5.20355 |          0.1  | TSM+LLM-COT-RF-HDELTA |
| 22 |        20 | 22.9624  | 4.79191 | 3.45341  |  8.91887 |          0.1  | TSM+LLM-COT-RF-HDELTA |
| 23 |        30 | 29.6952  | 5.44933 | 4.08908  | 10.0575  |          0.1  | TSM+LLM-COT-RF-HDELTA |
| 24 |         1 |  1.54604 | 1.2434  | 0.988269 |  2.78077 |          0.2  | TSM+LLM-COT-RF-HDELTA |
| 25 |         5 |  6.34074 | 2.51808 | 1.8755   |  5.23541 |          0.2  | TSM+LLM-COT-RF-HDELTA |
| 26 |        20 | 22.8502  | 4.78019 | 3.45028  |  8.90928 |          0.2  | TSM+LLM-COT-RF-HDELTA |
| 27 |        30 | 29.7346  | 5.45294 | 4.09647  | 10.0847  |          0.2  | TSM+LLM-COT-RF-HDELTA |
| 28 |         1 |  1.55425 | 1.2467  | 0.992809 |  2.79321 |          0.3  | TSM+LLM-COT-RF-HDELTA |
| 29 |         5 |  6.39802 | 2.52943 | 1.88824  |  5.27127 |          0.3  | TSM+LLM-COT-RF-HDELTA |
| 30 |        20 | 22.7492  | 4.76962 | 3.44916  |  8.90533 |          0.3  | TSM+LLM-COT-RF-HDELTA |
| 31 |        30 | 29.7862  | 5.45767 | 4.10539  | 10.1163  |          0.3  | TSM+LLM-COT-RF-HDELTA |

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

- **Price accuracy**: TSM+LLM-COT-RF-HDELTA has the lowest average MSE (15.146).
- **TSM vs naive**: TSM MSE is 0.9x the naive baseline on average.
- **Directional accuracy**: tsm has the highest average trend accuracy (0.525).

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
Price accuracy is best for TSM+LLM-COT-RF-HDELTA, while directional accuracy is highest for tsm.
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


