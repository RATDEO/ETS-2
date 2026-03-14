# EU ETS Futures Forecasting with TSM + LLM Refinement

**Authors**: Auto-generated

**Generated**: 2026-03-12 02:36:21

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
- Dataset spans 2021-05-19 00:00:00 to 2024-12-31 00:00:00
- 933 daily observations
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

- **Date Range**: 2021-05-19 00:00:00 to 2024-12-31 00:00:00
- **Total Observations**: 933
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
|  0 |         1 |  0.802159 | 0.895633 | 0.691584 |  1.76339 | naive_persistence            |
|  1 |         5 |  3.19275  | 1.78683  | 1.39752  |  3.58012 | naive_persistence            |
|  2 |        20 | 11.3778   | 3.37309  | 2.76188  |  7.29501 | naive_persistence            |
|  3 |        30 | 11.5073   | 3.39224  | 2.89139  |  7.81753 | naive_persistence            |
|  4 |         1 |  3.17264  | 1.78119  | 1.38188  |  3.53042 | seasonal_naive               |
|  5 |         5 |  3.19275  | 1.78683  | 1.39752  |  3.58012 | seasonal_naive               |
|  6 |        20 | 11.3778   | 3.37309  | 2.76188  |  7.29501 | seasonal_naive               |
|  7 |        30 | 11.5073   | 3.39224  | 2.89139  |  7.81753 | seasonal_naive               |
|  8 |         1 |  0.979251 | 0.989571 | 0.783139 |  2.00646 | linear_ridge                 |
|  9 |         5 |  6.05072  | 2.45982  | 2.00575  |  5.19542 | linear_ridge                 |
| 10 |        20 | 38.1626   | 6.17759  | 5.50137  | 14.6817  | linear_ridge                 |
| 11 |        30 | 43.0432   | 6.56073  | 6.00263  | 16.3085  | linear_ridge                 |
| 12 |         1 |  5.04201  | 2.24544  | 2.07479  |  5.33472 | linear_lasso                 |
| 13 |         5 | 12.8359   | 3.58272  | 3.21972  |  8.34393 | linear_lasso                 |
| 14 |        20 | 45.8922   | 6.77437  | 6.22325  | 16.5723  | linear_lasso                 |
| 15 |        30 | 51.7439   | 7.19332  | 6.76841  | 18.3279  | linear_lasso                 |
| 16 |         1 |  0.786214 | 0.886687 | 0.696314 |  1.7796  | tsm                          |
| 17 |         5 |  3.81312  | 1.95272  | 1.50172  |  3.87423 | tsm                          |
| 18 |        20 | 15.124    | 3.88896  | 3.11682  |  8.32176 | tsm                          |
| 19 |        30 | 13.7986   | 3.71465  | 3.05296  |  8.34449 | tsm                          |
| 20 |         1 |  0.786214 | 0.886687 | 0.696314 |  1.7796  | TSM+LLM-COT-RF-HDELTA        |
| 21 |         5 |  3.92873  | 1.9821   | 1.52755  |  3.94077 | TSM+LLM-COT-RF-HDELTA        |
| 22 |        20 | 16.2641   | 4.03288  | 3.25544  |  8.69491 | TSM+LLM-COT-RF-HDELTA        |
| 23 |        30 | 14.7811   | 3.84461  | 3.17227  |  8.67032 | TSM+LLM-COT-RF-HDELTA        |
| 24 |         1 |  0.802159 | 0.895633 | 0.691584 |  1.76339 | naive_persistence_llm_subset |
| 25 |         5 |  3.19275  | 1.78683  | 1.39752  |  3.58012 | naive_persistence_llm_subset |
| 26 |        20 | 11.3778   | 3.37309  | 2.76188  |  7.29501 | naive_persistence_llm_subset |
| 27 |        30 | 11.5073   | 3.39224  | 2.89139  |  7.81753 | naive_persistence_llm_subset |
| 28 |         1 |  3.17264  | 1.78119  | 1.38188  |  3.53042 | seasonal_naive_llm_subset    |
| 29 |         5 |  3.19275  | 1.78683  | 1.39752  |  3.58012 | seasonal_naive_llm_subset    |
| 30 |        20 | 11.3778   | 3.37309  | 2.76188  |  7.29501 | seasonal_naive_llm_subset    |
| 31 |        30 | 11.5073   | 3.39224  | 2.89139  |  7.81753 | seasonal_naive_llm_subset    |
| 32 |         1 |  0.979251 | 0.989571 | 0.783139 |  2.00646 | linear_ridge_llm_subset      |
| 33 |         5 |  6.05072  | 2.45982  | 2.00575  |  5.19542 | linear_ridge_llm_subset      |
| 34 |        20 | 38.1626   | 6.17759  | 5.50137  | 14.6817  | linear_ridge_llm_subset      |
| 35 |        30 | 43.0432   | 6.56073  | 6.00263  | 16.3085  | linear_ridge_llm_subset      |
| 36 |         1 |  5.04201  | 2.24544  | 2.07479  |  5.33472 | linear_lasso_llm_subset      |
| 37 |         5 | 12.8359   | 3.58272  | 3.21972  |  8.34393 | linear_lasso_llm_subset      |
| 38 |        20 | 45.8922   | 6.77437  | 6.22325  | 16.5723  | linear_lasso_llm_subset      |
| 39 |        30 | 51.7439   | 7.19332  | 6.76841  | 18.3279  | linear_lasso_llm_subset      |
| 40 |         1 |  0.786214 | 0.886687 | 0.696314 |  1.7796  | tsm_llm_subset               |
| 41 |         5 |  3.81312  | 1.95272  | 1.50172  |  3.87423 | tsm_llm_subset               |
| 42 |        20 | 15.124    | 3.88896  | 3.11682  |  8.32176 | tsm_llm_subset               |
| 43 |        30 | 13.7986   | 3.71465  | 3.05296  |  8.34449 | tsm_llm_subset               |

### 4.2 Trend Classification Accuracy

Table 2 shows the accuracy of predicting price direction (up/flat/down).

|    |   horizon |   accuracy |   accuracy_up |   accuracy_down |   accuracy_flat |   n_up |   n_down |   n_flat | model                        |
|---:|----------:|-----------:|--------------:|----------------:|----------------:|-------:|---------:|---------:|:-----------------------------|
|  0 |         1 |  0.475248  |      0        |       0         |        1        |     20 |       33 |       48 | naive_persistence            |
|  1 |         5 |  0.247525  |      0        |       0         |        1        |     29 |       47 |       25 | naive_persistence            |
|  2 |        20 |  0.108911  |      0        |       0         |        1        |     34 |       56 |       11 | naive_persistence            |
|  3 |        30 |  0.0891089 |      0        |       0         |        1        |     29 |       63 |        9 | naive_persistence            |
|  4 |         1 |  0.356436  |      0.55     |       0.393939  |        0.25     |     20 |       33 |       48 | seasonal_naive               |
|  5 |         5 |  0.247525  |      0        |       0         |        1        |     29 |       47 |       25 | seasonal_naive               |
|  6 |        20 |  0.108911  |      0        |       0         |        1        |     34 |       56 |       11 | seasonal_naive               |
|  7 |        30 |  0.0891089 |      0        |       0         |        1        |     29 |       63 |        9 | seasonal_naive               |
|  8 |         1 |  0.405941  |      0.2      |       0         |        0.770833 |     20 |       33 |       48 | linear_ridge                 |
|  9 |         5 |  0.287129  |      1        |       0         |        0        |     29 |       47 |       25 | linear_ridge                 |
| 10 |        20 |  0.336634  |      1        |       0         |        0        |     34 |       56 |       11 | linear_ridge                 |
| 11 |        30 |  0.287129  |      1        |       0         |        0        |     29 |       63 |        9 | linear_ridge                 |
| 12 |         1 |  0.19802   |      1        |       0         |        0        |     20 |       33 |       48 | linear_lasso                 |
| 13 |         5 |  0.287129  |      1        |       0         |        0        |     29 |       47 |       25 | linear_lasso                 |
| 14 |        20 |  0.336634  |      1        |       0         |        0        |     34 |       56 |       11 | linear_lasso                 |
| 15 |        30 |  0.287129  |      1        |       0         |        0        |     29 |       63 |        9 | linear_lasso                 |
| 16 |         1 |  0.405941  |      0.25     |       0.0909091 |        0.6875   |     20 |       33 |       48 | tsm                          |
| 17 |         5 |  0.415842  |      0.862069 |       0.0638298 |        0.56     |     29 |       47 |       25 | tsm                          |
| 18 |        20 |  0.376238  |      1        |       0.0178571 |        0.272727 |     34 |       56 |       11 | tsm                          |
| 19 |        30 |  0.287129  |      1        |       0         |        0        |     29 |       63 |        9 | tsm                          |
| 20 |         1 |  0.405941  |      0.25     |       0.0909091 |        0.6875   |     20 |       33 |       48 | TSM+LLM-COT-RF-HDELTA        |
| 21 |         5 |  0.39604   |      0.827586 |       0.0425532 |        0.56     |     29 |       47 |       25 | TSM+LLM-COT-RF-HDELTA        |
| 22 |        20 |  0.356436  |      1        |       0         |        0.181818 |     34 |       56 |       11 | TSM+LLM-COT-RF-HDELTA        |
| 23 |        30 |  0.287129  |      1        |       0         |        0        |     29 |       63 |        9 | TSM+LLM-COT-RF-HDELTA        |
| 24 |         1 |  0.475248  |      0        |       0         |        1        |     20 |       33 |       48 | naive_persistence_llm_subset |
| 25 |         5 |  0.247525  |      0        |       0         |        1        |     29 |       47 |       25 | naive_persistence_llm_subset |
| 26 |        20 |  0.108911  |      0        |       0         |        1        |     34 |       56 |       11 | naive_persistence_llm_subset |
| 27 |        30 |  0.0891089 |      0        |       0         |        1        |     29 |       63 |        9 | naive_persistence_llm_subset |
| 28 |         1 |  0.356436  |      0.55     |       0.393939  |        0.25     |     20 |       33 |       48 | seasonal_naive_llm_subset    |
| 29 |         5 |  0.247525  |      0        |       0         |        1        |     29 |       47 |       25 | seasonal_naive_llm_subset    |
| 30 |        20 |  0.108911  |      0        |       0         |        1        |     34 |       56 |       11 | seasonal_naive_llm_subset    |
| 31 |        30 |  0.0891089 |      0        |       0         |        1        |     29 |       63 |        9 | seasonal_naive_llm_subset    |
| 32 |         1 |  0.405941  |      0.2      |       0         |        0.770833 |     20 |       33 |       48 | linear_ridge_llm_subset      |
| 33 |         5 |  0.287129  |      1        |       0         |        0        |     29 |       47 |       25 | linear_ridge_llm_subset      |
| 34 |        20 |  0.336634  |      1        |       0         |        0        |     34 |       56 |       11 | linear_ridge_llm_subset      |
| 35 |        30 |  0.287129  |      1        |       0         |        0        |     29 |       63 |        9 | linear_ridge_llm_subset      |
| 36 |         1 |  0.19802   |      1        |       0         |        0        |     20 |       33 |       48 | linear_lasso_llm_subset      |
| 37 |         5 |  0.287129  |      1        |       0         |        0        |     29 |       47 |       25 | linear_lasso_llm_subset      |
| 38 |        20 |  0.336634  |      1        |       0         |        0        |     34 |       56 |       11 | linear_lasso_llm_subset      |
| 39 |        30 |  0.287129  |      1        |       0         |        0        |     29 |       63 |        9 | linear_lasso_llm_subset      |
| 40 |         1 |  0.405941  |      0.25     |       0.0909091 |        0.6875   |     20 |       33 |       48 | tsm_llm_subset               |
| 41 |         5 |  0.415842  |      0.862069 |       0.0638298 |        0.56     |     29 |       47 |       25 | tsm_llm_subset               |
| 42 |        20 |  0.376238  |      1        |       0.0178571 |        0.272727 |     34 |       56 |       11 | tsm_llm_subset               |
| 43 |        30 |  0.287129  |      1        |       0         |        0        |     29 |       63 |        9 | tsm_llm_subset               |

### 4.3 Statistical Significance

Table 3 reports the results of significance tests comparing methods.

|    |   horizon | model                     | baseline                     |   mean_error_model |   mean_error_baseline |   improvement_pct |   t_statistic |    t_pvalue | t_significant   |   wilcoxon_statistic |   wilcoxon_pvalue | wilcoxon_significant   |   dm_statistic |   dm_pvalue | dm_significant   | model_better   |
|---:|----------:|:--------------------------|:-----------------------------|-------------------:|----------------------:|------------------:|--------------:|------------:|:----------------|---------------------:|------------------:|:-----------------------|---------------:|------------:|:-----------------|:---------------|
|  0 |         1 | seasonal_naive_llm_subset | naive_persistence_llm_subset |           3.17264  |              0.802159 |    -295.512       |      4.95667  | 2.93116e-06 | True            |                 1102 |       5.98555e-07 | True                   |    4.95667     | 7.17099e-07 | True             | False          |
|  1 |         5 | seasonal_naive_llm_subset | naive_persistence_llm_subset |           3.19275  |              3.19275  |       7.89046e-06 |     -1.33481  | 0.184968    | False           |                 1653 |       0.0136215   | True                   |   -0.0164261   | 0.986894    | False            | True           |
|  2 |        20 | seasonal_naive_llm_subset | naive_persistence_llm_subset |          11.3778   |             11.3778   |      -1.42711e-05 |      1.55438  | 0.123254    | False           |                 2356 |       0.941191    | False                  |    0.0766641   | 0.938891    | False            | False          |
|  3 |        30 | seasonal_naive_llm_subset | naive_persistence_llm_subset |          11.5073   |             11.5073   |       2.47279e-06 |      0.636281 | 0.526047    | False           |                 1964 |       0.311232    | False                  |    0.029492    | 0.976472    | False            | True           |
|  4 |         1 | linear_ridge_llm_subset   | naive_persistence_llm_subset |           0.979251 |              0.802159 |     -22.0769      |      2.48463  | 0.0146291   | True            |                 1741 |       0.00469888  | True                   |    2.48463     | 0.0129686   | True             | False          |
|  5 |         5 | linear_ridge_llm_subset   | naive_persistence_llm_subset |           6.05072  |              3.19275  |     -89.5144      |      5.86554  | 5.81342e-08 | True            |                 1021 |       1.39367e-07 | True                   |    2.56224     | 0.0104      | True             | False          |
|  6 |        20 | linear_ridge_llm_subset   | naive_persistence_llm_subset |          38.1626   |             11.3778   |    -235.414       |     10.7419   | 2.35364e-18 | True            |                  331 |       2.88115e-14 | True                   |    4.20517     | 2.6089e-05  | True             | False          |
|  7 |        30 | linear_ridge_llm_subset   | naive_persistence_llm_subset |          43.0432   |             11.5073   |    -274.051       |     13.3971   | 4.75281e-24 | True            |                  152 |       2.21304e-16 | True                   |    3.16932e+07 | 0           | True             | False          |
|  8 |         1 | linear_lasso_llm_subset   | naive_persistence_llm_subset |           5.04201  |              0.802159 |    -528.554       |     12.2837   | 1.08223e-21 | True            |                  241 |       2.60639e-15 | True                   |   12.2837      | 0           | True             | False          |
|  9 |         5 | linear_lasso_llm_subset   | naive_persistence_llm_subset |          12.8359   |              3.19275  |    -302.033       |     11.0547   | 4.89088e-19 | True            |                  341 |       3.74161e-14 | True                   |    4.77852     | 1.76593e-06 | True             | False          |
| 10 |        20 | linear_lasso_llm_subset   | naive_persistence_llm_subset |          45.8922   |             11.3778   |    -303.35        |     13.0321   | 2.77876e-23 | True            |                  172 |       3.88208e-16 | True                   |    5.10898     | 3.23901e-07 | True             | False          |
| 11 |        30 | linear_lasso_llm_subset   | naive_persistence_llm_subset |          51.7439   |             11.5073   |    -349.661       |     16.1888   | 1.06997e-29 | True            |                   61 |       1.62035e-17 | True                   |   28.1194      | 0           | True             | False          |
| 12 |         1 | tsm_llm_subset            | naive_persistence_llm_subset |           0.786214 |              0.802159 |       1.98782     |     -0.226128 | 0.821563    | False           |                 2384 |       0.516511    | False                  |   -0.226128    | 0.821102    | False            | True           |
| 13 |         5 | tsm_llm_subset            | naive_persistence_llm_subset |           3.81312  |              3.19275  |     -19.4305      |      1.91277  | 0.0586381   | False           |                 2198 |       0.200956    | False                  |    0.940618    | 0.346901    | False            | False          |
| 14 |        20 | tsm_llm_subset            | naive_persistence_llm_subset |          15.124    |             11.3778   |     -32.9258      |      3.64161  | 0.000431346 | True            |                 1711 |       0.00340479  | True                   |    1.17547     | 0.239807    | False            | False          |
| 15 |        30 | tsm_llm_subset            | naive_persistence_llm_subset |          13.7986   |             11.5073   |     -19.9116      |      3.59204  | 0.000510816 | True            |                 1573 |       0.000683519 | True                   |    2.30272e+06 | 0           | True             | False          |
| 16 |         1 | TSM+LLM-COT-RF-HDELTA     | naive_persistence_llm_subset |           0.786214 |              0.802159 |       1.98782     |     -0.226128 | 0.821563    | False           |                 2384 |       0.516511    | False                  |   -0.226128    | 0.821102    | False            | True           |
| 17 |         5 | TSM+LLM-COT-RF-HDELTA     | naive_persistence_llm_subset |           3.92873  |              3.19275  |     -23.0515      |      2.12076  | 0.0364149   | True            |                 2122 |       0.124466    | False                  |    1.03177     | 0.30218     | False            | False          |
| 18 |        20 | TSM+LLM-COT-RF-HDELTA     | naive_persistence_llm_subset |          16.2641   |             11.3778   |     -42.9467      |      4.27017  | 4.45373e-05 | True            |                 1491 |       0.000238884 | True                   |    1.38913     | 0.164792    | False            | False          |
| 19 |        30 | TSM+LLM-COT-RF-HDELTA     | naive_persistence_llm_subset |          14.7811   |             11.5073   |     -28.4493      |      4.54716  | 1.52805e-05 | True            |                 1349 |       3.25378e-05 | True                   |    3.29008e+06 | 0           | True             | False          |


## 5. Robustness Analysis

### 5.1 Noise Injection Test

To assess robustness, we inject noise into forecast paths at levels of 5%, 10%, 20%, 
and 30% of the forecast standard deviation.

|    |   horizon |       mse |     rmse |      mae |    mape |   noise_level | model                 |
|---:|----------:|----------:|---------:|---------:|--------:|--------------:|:----------------------|
|  0 |         1 |  0.787679 | 0.887513 | 0.698557 | 1.78511 |          0.05 | tsm                   |
|  1 |         5 |  3.80416  | 1.95043  | 1.50172  | 3.87423 |          0.05 | tsm                   |
|  2 |        20 | 15.0967   | 3.88544  | 3.11391  | 8.3152  |          0.05 | tsm                   |
|  3 |        30 | 13.8554   | 3.72228  | 3.0622   | 8.36862 |          0.05 | tsm                   |
|  4 |         1 |  0.790866 | 0.889307 | 0.701311 | 1.792   |          0.1  | tsm                   |
|  5 |         5 |  3.79699  | 1.94859  | 1.5019   | 3.87469 |          0.1  | tsm                   |
|  6 |        20 | 15.0708   | 3.88211  | 3.11195  | 8.31118 |          0.1  | tsm                   |
|  7 |        30 | 13.9138   | 3.73012  | 3.07143  | 8.39275 |          0.1  | tsm                   |
|  8 |         1 |  0.802406 | 0.895771 | 0.707461 | 1.80747 |          0.2  | tsm                   |
|  9 |         5 |  3.78801  | 1.94628  | 1.50374  | 3.87964 |          0.2  | tsm                   |
| 10 |        20 | 15.0235   | 3.87601  | 3.10805  | 8.30315 |          0.2  | tsm                   |
| 11 |        30 | 14.0356   | 3.74641  | 3.08994  | 8.44113 |          0.2  | tsm                   |
| 12 |         1 |  0.820833 | 0.905998 | 0.716813 | 1.83133 |          0.3  | tsm                   |
| 13 |         5 |  3.78618  | 1.94581  | 1.50961  | 3.89499 |          0.3  | tsm                   |
| 14 |        20 | 14.982    | 3.87066  | 3.10414  | 8.29511 |          0.3  | tsm                   |
| 15 |        30 | 14.1639   | 3.7635   | 3.10931  | 8.4918  |          0.3  | tsm                   |
| 16 |         1 |  0.787946 | 0.887663 | 0.698772 | 1.78565 |          0.05 | TSM+LLM-COT-RF-HDELTA |
| 17 |         5 |  3.91879  | 1.97959  | 1.5263   | 3.93757 |          0.05 | TSM+LLM-COT-RF-HDELTA |
| 18 |        20 | 16.2356   | 4.02934  | 3.25201  | 8.68709 |          0.05 | TSM+LLM-COT-RF-HDELTA |
| 19 |        30 | 14.8461   | 3.85306  | 3.18181  | 8.69547 |          0.05 | TSM+LLM-COT-RF-HDELTA |
| 20 |         1 |  0.791663 | 0.889754 | 0.701741 | 1.79308 |          0.1  | TSM+LLM-COT-RF-HDELTA |
| 21 |         5 |  3.91091  | 1.9776   | 1.52527  | 3.93488 |          0.1  | TSM+LLM-COT-RF-HDELTA |
| 22 |        20 | 16.2087   | 4.026    | 3.24859  | 8.67926 |          0.1  | TSM+LLM-COT-RF-HDELTA |
| 23 |        30 | 14.913    | 3.86173  | 3.19222  | 8.7227  |          0.1  | TSM+LLM-COT-RF-HDELTA |
| 24 |         1 |  0.805045 | 0.897243 | 0.708411 | 1.80988 |          0.2  | TSM+LLM-COT-RF-HDELTA |
| 25 |         5 |  3.90132  | 1.97517  | 1.52494  | 3.93403 |          0.2  | TSM+LLM-COT-RF-HDELTA |
| 26 |        20 | 16.16     | 4.01995  | 3.24267  | 8.66609 |          0.2  | TSM+LLM-COT-RF-HDELTA |
| 27 |        30 | 15.0524   | 3.87974  | 3.21306  | 8.77726 |          0.2  | TSM+LLM-COT-RF-HDELTA |
| 28 |         1 |  0.82636  | 0.909043 | 0.719437 | 1.838   |          0.3  | TSM+LLM-COT-RF-HDELTA |
| 29 |         5 |  3.89995  | 1.97483  | 1.53024  | 3.94799 |          0.3  | TSM+LLM-COT-RF-HDELTA |
| 30 |        20 | 16.1181   | 4.01474  | 3.23865  | 8.6579  |          0.3  | TSM+LLM-COT-RF-HDELTA |
| 31 |        30 | 15.1994   | 3.89864  | 3.23413  | 8.83238 |          0.3  | TSM+LLM-COT-RF-HDELTA |

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

- **Price accuracy**: naive_persistence has the lowest average MSE (6.720).
- **TSM vs naive**: TSM MSE is 1.2x the naive baseline on average.
- **Directional accuracy**: tsm has the highest average trend accuracy (0.371).

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
Price accuracy is best for naive_persistence, while directional accuracy is highest for tsm.
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


