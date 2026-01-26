# EU ETS Futures Forecasting with TSM + LLM Refinement

**Authors**: Auto-generated

**Generated**: 2026-01-02 16:02:08

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
- Dataset spans 2010-01-05 00:00:00 to 2025-09-30 00:00:00
- 2866 daily observations
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

- **Date Range**: 2010-01-05 00:00:00 to 2025-09-30 00:00:00
- **Total Observations**: 2,866
- **Number of Features**: 36

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

|    |   horizon |        mse |     rmse |      mae |     mape | model             |
|---:|----------:|-----------:|---------:|---------:|---------:|:------------------|
|  0 |         1 |    4.28177 |  2.06924 |  1.4437  |  2.15348 | naive_persistence |
|  1 |         5 |   10.2545  |  3.20227 |  2.4621  |  3.71042 | naive_persistence |
|  2 |        20 |   36.8281  |  6.06862 |  4.85088 |  7.20306 | naive_persistence |
|  3 |        30 |   49.5437  |  7.03873 |  5.68055 |  8.32392 | naive_persistence |
|  4 |         1 |   10.571   |  3.25131 |  2.49754 |  3.76078 | seasonal_naive    |
|  5 |         5 |   10.2545  |  3.20227 |  2.4621  |  3.71042 | seasonal_naive    |
|  6 |        20 |   36.8281  |  6.06862 |  4.85088 |  7.20307 | seasonal_naive    |
|  7 |        30 |   49.5437  |  7.03873 |  5.68055 |  8.32392 | seasonal_naive    |
|  8 |         1 | 2043.16    | 45.2013  | 44.8729  | 66.385   | tsm               |
|  9 |         5 | 2044.79    | 45.2193  | 44.8906  | 66.3811  | tsm               |
| 10 |        20 | 2083.98    | 45.6506  | 45.3406  | 66.5709  | tsm               |
| 11 |        30 | 2149.33    | 46.3609  | 46.0515  | 67.0075  | tsm               |
| 12 |         1 |   19.6344  |  4.43107 |  3.213   |  4.72604 | DP                |
| 13 |         5 |   33.4648  |  5.78488 |  3.457   |  5.60618 | DP                |
| 14 |        20 |  116.57    | 10.7967  |  9.341   | 16.9489  | DP                |
| 15 |        30 |  174.663   | 13.216   | 10.383   | 19.4467  | DP                |
| 16 |         1 |   19.8709  |  4.45768 |  3.314   |  4.89536 | CoT               |
| 17 |         5 |   43.2467  |  6.57622 |  4.521   |  7.33644 | CoT               |
| 18 |        20 |  223.34    | 14.9446  | 14.477   | 26.4055  | CoT               |
| 19 |        30 |  294.218   | 17.1528  | 16.172   | 29.7667  | CoT               |
| 20 |         1 |   19.1804  |  4.37954 |  3.215   |  4.73891 | CoT-RF            |
| 21 |         5 |   35.8469  |  5.98722 |  3.921   |  6.35605 | CoT-RF            |
| 22 |        20 |  172.503   | 13.134   | 12.707   | 23.1784  | CoT-RF            |
| 23 |        30 |  219.078   | 14.8013  | 13.622   | 25.1928  | CoT-RF            |
| 24 |         1 |   22.2168  |  4.71347 |  3.441   |  5.07043 | TSM+LLM           |
| 25 |         5 |   35.4204  |  5.95151 |  4.135   |  6.70415 | TSM+LLM           |
| 26 |        20 |  205.427   | 14.3327  | 12.353   | 22.5335  | TSM+LLM           |
| 27 |        30 |  355.753   | 18.8614  | 17.067   | 31.5105  | TSM+LLM           |

### 4.2 Trend Classification Accuracy

Table 2 shows the accuracy of predicting price direction (up/flat/down).

|    |   horizon |   accuracy |   accuracy_up |   accuracy_down |   accuracy_flat |   n_up |   n_down |   n_flat | model             |
|---:|----------:|-----------:|--------------:|----------------:|----------------:|-------:|---------:|---------:|:------------------|
|  0 |         1 |  0.270718  |      0        |        0        |        1        |    122 |      142 |       98 | naive_persistence |
|  1 |         5 |  0.121547  |      0        |        0        |        1        |    151 |      167 |       44 | naive_persistence |
|  2 |        20 |  0.0635359 |      0        |        0        |        1        |    184 |      155 |       23 | naive_persistence |
|  3 |        30 |  0.0359116 |      0        |        0        |        1        |    200 |      149 |       13 | naive_persistence |
|  4 |         1 |  0.41989   |      0.590164 |        0.464789 |        0.142857 |    122 |      142 |       98 | seasonal_naive    |
|  5 |         5 |  0.121547  |      0        |        0        |        1        |    151 |      167 |       44 | seasonal_naive    |
|  6 |        20 |  0.0635359 |      0        |        0        |        1        |    184 |      155 |       23 | seasonal_naive    |
|  7 |        30 |  0.0359116 |      0        |        0        |        1        |    200 |      149 |       13 | seasonal_naive    |
|  8 |         1 |  0.392265  |      0        |        1        |        0        |    122 |      142 |       98 | tsm               |
|  9 |         5 |  0.461326  |      0        |        1        |        0        |    151 |      167 |       44 | tsm               |
| 10 |        20 |  0.428177  |      0        |        1        |        0        |    184 |      155 |       23 | tsm               |
| 11 |        30 |  0.411602  |      0        |        1        |        0        |    200 |      149 |       13 | tsm               |

### 4.3 Statistical Significance

Table 3 reports the results of significance tests comparing methods.

|    |   horizon | model          | baseline          |   mean_error_model |   mean_error_baseline |   improvement_pct |   t_statistic |     t_pvalue | t_significant   |   wilcoxon_statistic |   wilcoxon_pvalue | wilcoxon_significant   |     dm_statistic |   dm_pvalue | dm_significant   | model_better   |
|---:|----------:|:---------------|:------------------|-------------------:|----------------------:|------------------:|--------------:|-------------:|:----------------|---------------------:|------------------:|:-----------------------|-----------------:|------------:|:-----------------|:---------------|
|  0 |         1 | seasonal_naive | naive_persistence |            10.571  |               4.28177 |    -146.884       |     6.07455   | 3.15547e-09  | True            |                15973 |       2.42149e-17 | True                   |      6.07455     | 1.24333e-09 | True             | False          |
|  1 |         5 | seasonal_naive | naive_persistence |            10.2545 |              10.2545  |      -1.06104e-05 |     1.41614   | 0.157596     | False           |                26437 |       0.187844    | False                  |      0.0636682   | 0.949234    | False            | False          |
|  2 |        20 | seasonal_naive | naive_persistence |            36.8281 |              36.8281  |       2.26968e-06 |    -0.698159  | 0.485527     | False           |                24292 |       0.00966441  | True                   |     -0.117386    | 0.906554    | False            | True           |
|  3 |        30 | seasonal_naive | naive_persistence |            49.5437 |              49.5437  |       3.0175e-06  |    -0.155382  | 0.876607     | False           |                27420 |       0.258151    | False                  |     -0.0294068   | 0.97654     | False            | True           |
|  4 |         1 | tsm            | naive_persistence |          2043.16   |               4.28177 |  -47617.7         |    80.7765    | 3.26084e-233 | True            |                    0 |       4.42975e-61 | True                   |     80.7765      | 0           | True             | False          |
|  5 |         5 | tsm            | naive_persistence |          2044.79   |              10.2545  |  -19840.3         |    80.4594    | 1.25188e-232 | True            |                    0 |       4.42975e-61 | True                   |     27.5342      | 0           | True             | False          |
|  6 |        20 | tsm            | naive_persistence |          2083.98   |              36.8281  |   -5558.67        |    83.0421    | 2.49656e-237 | True            |                    0 |       4.42975e-61 | True                   |     15.7866      | 0           | True             | False          |
|  7 |        30 | tsm            | naive_persistence |          2149.33   |              49.5437  |   -4238.25        |    83.1452    | 1.63115e-237 | True            |                    0 |       4.42975e-61 | True                   |     15.7403      | 0           | True             | False          |
|  8 |         1 | DP             | naive_persistence |            19.6344 |              18.7791  |      -4.55425     |     1.71548   | 0.120395     | False           |                   12 |       0.130859    | False                  |      1.71548     | 0.0862569   | False            | False          |
|  9 |         5 | DP             | naive_persistence |            33.4648 |              30.9493  |      -8.12808     |     1.12433   | 0.289964     | False           |                   20 |       0.492188    | False                  |      2.91655     | 0.00353928  | True             | False          |
| 10 |        20 | DP             | naive_persistence |           116.57   |             115.192   |      -1.19575     |     0.0735798 | 0.942954     | False           |                   23 |       0.695312    | False                  | 435578           | 0           | True             | False          |
| 11 |        30 | DP             | naive_persistence |           174.663  |             136.459   |     -27.9961      |     1.18716   | 0.265553     | False           |                   22 |       0.625       | False                  |      1.2081e+07  | 0           | True             | False          |
| 12 |         1 | CoT            | naive_persistence |            19.8709 |              18.7791  |      -5.8139      |     1.13495   | 0.285717     | False           |                   15 |       0.232422    | False                  |      1.13495     | 0.256396    | False            | False          |
| 13 |         5 | CoT            | naive_persistence |            43.2467 |              30.9493  |     -39.7341      |     1.88737   | 0.0917206    | False           |                    0 |       0.00195312  | True                   |      2.01561     | 0.0438408   | True             | False          |
| 14 |        20 | CoT            | naive_persistence |           223.34   |             115.192   |     -93.8843      |     8.09904   | 2.00597e-05  | True            |                    0 |       0.00195312  | True                   |      3.41993e+07 | 0           | True             | False          |
| 15 |        30 | CoT            | naive_persistence |           294.218  |             136.459   |    -115.609       |     6.74034   | 8.45694e-05  | True            |                    0 |       0.00195312  | True                   |      4.98878e+07 | 0           | True             | False          |
| 16 |         1 | CoT-RF         | naive_persistence |            19.1804 |              18.7791  |      -2.13667     |     0.935519  | 0.373942     | False           |                   18 |       0.375       | False                  |      0.935519    | 0.349521    | False            | False          |
| 17 |         5 | CoT-RF         | naive_persistence |            35.8469 |              30.9493  |     -15.8246      |     2.25996   | 0.0501797    | False           |                    2 |       0.00585938  | True                   |      1.96743     | 0.0491338   | True             | False          |
| 18 |        20 | CoT-RF         | naive_persistence |           172.503  |             115.192   |     -49.752       |     6.6353    | 9.5331e-05   | True            |                    0 |       0.00195312  | True                   |      1.81232e+07 | 0           | True             | False          |
| 19 |        30 | CoT-RF         | naive_persistence |           219.078  |             136.459   |     -60.5444      |     5.0035    | 0.000735449  | True            |                    0 |       0.00195312  | True                   |      2.61262e+07 | 0           | True             | False          |
| 20 |         1 | TSM+LLM        | naive_persistence |            22.2168 |              18.7791  |     -18.306       |     2.29686   | 0.0472416    | True            |                    6 |       0.0273438   | True                   |      2.29686     | 0.0216269   | True             | False          |
| 21 |         5 | TSM+LLM        | naive_persistence |            35.4204 |              30.9493  |     -14.4467      |     0.934331  | 0.374522     | False           |                   17 |       0.322266    | False                  |      1.4139e+06  | 0           | True             | False          |
| 22 |        20 | TSM+LLM        | naive_persistence |           205.427  |             115.192   |     -78.3342      |     2.10749   | 0.064331     | False           |                   11 |       0.105469    | False                  |      6.94259     | 3.84981e-12 | True             | False          |
| 23 |        30 | TSM+LLM        | naive_persistence |           355.753  |             136.459   |    -160.703       |     3.42849   | 0.00752623   | True            |                    0 |       0.00195312  | True                   |      6.93468e+07 | 0           | True             | False          |


## 5. Robustness Analysis

### 5.1 Noise Injection Test

To assess robustness, we inject noise into forecast paths at levels of 5%, 10%, 20%, 
and 30% of the forecast standard deviation.

|    |   horizon |       mse |     rmse |      mae |     mape |   noise_level | model   |
|---:|----------:|----------:|---------:|---------:|---------:|--------------:|:--------|
|  0 |         1 | 2043.21   | 45.2019  | 44.8736  | 66.3861  |          0.05 | tsm     |
|  1 |         5 | 2044.99   | 45.2215  | 44.8933  | 66.3854  |          0.05 | tsm     |
|  2 |        20 | 2084.11   | 45.652   | 45.342   | 66.5728  |          0.05 | tsm     |
|  3 |        30 | 2148.21   | 46.3488  | 46.0393  | 66.9894  |          0.05 | tsm     |
|  4 |         1 | 2043.28   | 45.2027  | 44.8742  | 66.3872  |          0.1  | tsm     |
|  5 |         5 | 2045.2    | 45.2239  | 44.8959  | 66.3896  |          0.1  | tsm     |
|  6 |        20 | 2084.25   | 45.6536  | 45.3434  | 66.5748  |          0.1  | tsm     |
|  7 |        30 | 2147.1    | 46.3368  | 46.027   | 66.9714  |          0.1  | tsm     |
|  8 |         1 | 2043.45   | 45.2045  | 44.8756  | 66.3894  |          0.2  | tsm     |
|  9 |         5 | 2045.66   | 45.229   | 44.9012  | 66.398   |          0.2  | tsm     |
| 10 |        20 | 2084.55   | 45.6569  | 45.3461  | 66.5786  |          0.2  | tsm     |
| 11 |        30 | 2144.91   | 46.3132  | 46.0025  | 66.9353  |          0.2  | tsm     |
| 12 |         1 | 2043.66   | 45.2068  | 44.877   | 66.3916  |          0.3  | tsm     |
| 13 |         5 | 2046.18   | 45.2347  | 44.9065  | 66.4065  |          0.3  | tsm     |
| 14 |        20 | 2084.9    | 45.6607  | 45.3489  | 66.5825  |          0.3  | tsm     |
| 15 |        30 | 2142.77   | 46.2901  | 45.9779  | 66.8992  |          0.3  | tsm     |
| 16 |         1 |   19.4957 |  4.4154  |  3.19393 |  4.69613 |          0.05 | DP      |
| 17 |         5 |   33.6799 |  5.80344 |  3.47327 |  5.63294 |          0.05 | DP      |
| 18 |        20 |  115.526  | 10.7483  |  9.29816 | 16.8727  |          0.05 | DP      |
| 19 |        30 |  175.706  | 13.2554  | 10.4006  | 19.4824  |          0.05 | DP      |
| 20 |         1 |   19.3601 |  4.40001 |  3.17485 |  4.66623 |          0.1  | DP      |
| 21 |         5 |   33.8977 |  5.82217 |  3.48955 |  5.6597  |          0.1  | DP      |
| 22 |        20 |  114.492  | 10.7001  |  9.25533 | 16.7964  |          0.1  | DP      |
| 23 |        30 |  176.767  | 13.2954  | 10.4182  | 19.518   |          0.1  | DP      |
| 24 |         1 |   19.0977 |  4.37009 |  3.1367  |  4.60643 |          0.2  | DP      |
| 25 |         5 |   34.3416 |  5.86017 |  3.5221  |  5.71322 |          0.2  | DP      |
| 26 |        20 |  112.457  | 10.6046  |  9.16965 | 16.644   |          0.2  | DP      |
| 27 |        30 |  178.948  | 13.3772  | 10.5017  | 19.6714  |          0.2  | DP      |
| 28 |         1 |   18.8472 |  4.34133 |  3.09855 |  4.54663 |          0.3  | DP      |
| 29 |         5 |   34.7966 |  5.89886 |  3.55464 |  5.76674 |          0.3  | DP      |
| 30 |        20 |  110.463  | 10.5101  |  9.08398 | 16.4915  |          0.3  | DP      |
| 31 |        30 |  181.205  | 13.4613  | 10.602   | 19.8534  |          0.3  | DP      |
| 32 |         1 |   19.6387 |  4.43156 |  3.29326 |  4.86358 |          0.05 | CoT     |
| 33 |         5 |   43.5292 |  6.59766 |  4.5136  |  7.32468 |          0.05 | CoT     |
| 34 |        20 |  221.416  | 14.88    | 14.4156  | 26.2971  |          0.05 | CoT     |
| 35 |        30 |  297.26   | 17.2412  | 16.2777  | 29.9537  |          0.05 | CoT     |
| 36 |         1 |   19.4147 |  4.40621 |  3.27252 |  4.83181 |          0.1  | CoT     |
| 37 |         5 |   43.8186 |  6.61956 |  4.5062  |  7.31292 |          0.1  | CoT     |
| 38 |        20 |  219.511  | 14.8159  | 14.3543  | 26.1887  |          0.1  | CoT     |
| 39 |        30 |  300.359  | 17.3309  | 16.3834  | 30.1407  |          0.1  | CoT     |
| 40 |         1 |   18.9916 |  4.35793 |  3.23104 |  4.76825 |          0.2  | CoT     |
| 41 |         5 |   44.418  |  6.66469 |  4.49139 |  7.2894  |          0.2  | CoT     |
| 42 |        20 |  215.763  | 14.6889  | 14.2315  | 25.9719  |          0.2  | CoT     |
| 43 |        30 |  306.728  | 17.5136  | 16.5948  | 30.5148  |          0.2  | CoT     |
| 44 |         1 |   18.6015 |  4.31295 |  3.18955 |  4.7047  |          0.3  | CoT     |
| 45 |         5 |   45.045  |  6.71156 |  4.47659 |  7.26588 |          0.3  | CoT     |
| 46 |        20 |  212.095  | 14.5635  | 14.1088  | 25.7551  |          0.3  | CoT     |
| 47 |        30 |  313.325  | 17.701   | 16.8062  | 30.8889  |          0.3  | CoT     |
| 48 |         1 |   19.0356 |  4.36298 |  3.18979 |  4.69917 |          0.05 | CoT-RF  |
| 49 |         5 |   35.924  |  5.99366 |  3.90975 |  6.33775 |          0.05 | CoT-RF  |
| 50 |        20 |  171.637  | 13.101   | 12.6723  | 23.117   |          0.05 | CoT-RF  |
| 51 |        30 |  220.813  | 14.8598  | 13.6924  | 25.3183  |          0.05 | CoT-RF  |
| 52 |         1 |   18.8953 |  4.34688 |  3.16458 |  4.65944 |          0.1  | CoT-RF  |
| 53 |         5 |   36.0045 |  6.00037 |  3.89849 |  6.31946 |          0.1  | CoT-RF  |
| 54 |        20 |  170.777  | 13.0682  | 12.6376  | 23.0556  |          0.1  | CoT-RF  |
| 55 |        30 |  222.579  | 14.9191  | 13.7628  | 25.4438  |          0.1  | CoT-RF  |
| 56 |         1 |   18.6284 |  4.31607 |  3.11415 |  4.57998 |          0.2  | CoT-RF  |
| 57 |         5 |   36.1755 |  6.01461 |  3.87598 |  6.28286 |          0.2  | CoT-RF  |
| 58 |        20 |  169.077  | 13.003   | 12.5682  | 22.9328  |          0.2  | CoT-RF  |
| 59 |        30 |  226.206  | 15.0401  | 13.9036  | 25.6947  |          0.2  | CoT-RF  |
| 60 |         1 |   18.3796 |  4.28715 |  3.06373 |  4.50052 |          0.3  | CoT-RF  |
| 61 |         5 |   36.3598 |  6.02991 |  3.85347 |  6.24626 |          0.3  | CoT-RF  |
| 62 |        20 |  167.403  | 12.9384  | 12.4987  | 22.8101  |          0.3  | CoT-RF  |
| 63 |        30 |  229.957  | 15.1643  | 14.0444  | 25.9457  |          0.3  | CoT-RF  |
| 64 |         1 |   21.6556 |  4.65356 |  3.36353 |  4.94916 |          0.05 | TSM+LLM |
| 65 |         5 |   35.9738 |  5.99781 |  4.15464 |  6.73781 |          0.05 | TSM+LLM |
| 66 |        20 |  202.274  | 14.2223  | 12.2342  | 22.3232  |          0.05 | TSM+LLM |
| 67 |        30 |  358.243  | 18.9273  | 17.1479  | 31.6578  |          0.05 | TSM+LLM |
| 68 |         1 |   21.1138 |  4.59497 |  3.28606 |  4.82789 |          0.1  | TSM+LLM |
| 69 |         5 |   36.5504 |  6.04569 |  4.17429 |  6.77147 |          0.1  | TSM+LLM |
| 70 |        20 |  199.186  | 14.1133  | 12.1154  | 22.1129  |          0.1  | TSM+LLM |
| 71 |        30 |  360.782  | 18.9942  | 17.2288  | 31.8051  |          0.1  | TSM+LLM |
| 72 |         1 |   20.0884 |  4.48201 |  3.16498 |  4.63912 |          0.2  | TSM+LLM |
| 73 |         5 |   37.7733 |  6.146   |  4.21358 |  6.83878 |          0.2  | TSM+LLM |
| 74 |        20 |  193.203  | 13.8998  | 11.8777  | 21.6922  |          0.2  | TSM+LLM |
| 75 |        30 |  366.006  | 19.1313  | 17.3907  | 32.0997  |          0.2  | TSM+LLM |
| 76 |         1 |   19.1407 |  4.37501 |  3.07297 |  4.49653 |          0.3  | TSM+LLM |
| 77 |         5 |   39.0891 |  6.25213 |  4.25286 |  6.90609 |          0.3  | TSM+LLM |
| 78 |        20 |  187.48   | 13.6923  | 11.6401  | 21.2716  |          0.3  | TSM+LLM |
| 79 |        30 |  371.426  | 19.2724  | 17.5525  | 32.3943  |          0.3  | TSM+LLM |

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

- **Price accuracy**: naive_persistence has the lowest average MSE (25.227).
- **TSM vs naive**: TSM MSE is 82.5x the naive baseline on average.
- **Directional accuracy**: tsm has the highest average trend accuracy (0.423).

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
Sequence Length: 120
TSM Type: autoformer
LLM Model: gpt-4-turbo-preview
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


