# EU ETS Futures Forecasting with TSM + LLM Refinement

**Authors**: Auto-generated

**Generated**: 2026-03-06 14:01:09

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

|    |   horizon |      mse |    rmse |      mae |     mape | model                                         |
|---:|----------:|---------:|--------:|---------:|---------:|:----------------------------------------------|
|  0 |         1 |  1.06139 | 1.03024 | 0.784861 |  1.39543 | naive_persistence                             |
|  1 |         5 |  5.2119  | 2.28296 | 1.67     |  2.90235 | naive_persistence                             |
|  2 |        20 | 44.2759  | 6.65401 | 4.40799  |  7.94956 | naive_persistence                             |
|  3 |        30 | 79.9881  | 8.94361 | 6.37986  | 12.0727  | naive_persistence                             |
|  4 |         1 |  5.16853 | 2.27344 | 1.65132  |  2.90012 | seasonal_naive                                |
|  5 |         5 |  5.2119  | 2.28296 | 1.67     |  2.90235 | seasonal_naive                                |
|  6 |        20 | 44.2759  | 6.65401 | 4.40799  |  7.94956 | seasonal_naive                                |
|  7 |        30 | 79.9881  | 8.94361 | 6.37986  | 12.0727  | seasonal_naive                                |
|  8 |         1 |  1.06154 | 1.03031 | 0.781283 |  1.38304 | linear_ridge                                  |
|  9 |         5 |  4.59363 | 2.14328 | 1.56632  |  2.70398 | linear_ridge                                  |
| 10 |        20 | 30.1544  | 5.4913  | 3.32793  |  6.07247 | linear_ridge                                  |
| 11 |        30 | 56.9094  | 7.54383 | 4.78768  |  9.18917 | linear_ridge                                  |
| 12 |         1 |  1.1541  | 1.07429 | 0.830618 |  1.48473 | linear_lasso                                  |
| 13 |         5 |  4.73942 | 2.17702 | 1.61188  |  2.79853 | linear_lasso                                  |
| 14 |        20 | 32.1131  | 5.66684 | 3.43547  |  6.16613 | linear_lasso                                  |
| 15 |        30 | 62.0974  | 7.88019 | 5.23504  |  9.89649 | linear_lasso                                  |
| 16 |         1 |  1.20574 | 1.09806 | 0.855881 |  1.51781 | tsm                                           |
| 17 |         5 |  5.64911 | 2.37679 | 1.80329  |  3.12164 | tsm                                           |
| 18 |        20 | 40.2462  | 6.34399 | 4.28359  |  7.6831  | tsm                                           |
| 19 |        30 | 72.369   | 8.507   | 6.23287  | 11.696   | tsm                                           |
| 20 |         1 |  1.18834 | 1.09011 | 0.843542 |  1.50022 | TSM+LLM-COT-RF                                |
| 21 |         5 |  5.14291 | 2.2678  | 1.71535  |  2.98841 | TSM+LLM-COT-RF                                |
| 22 |        20 | 42.9263  | 6.55182 | 4.36437  |  7.89077 | TSM+LLM-COT-RF                                |
| 23 |        30 | 83.7506  | 9.15153 | 6.48569  | 12.2882  | TSM+LLM-COT-RF                                |
| 24 |         1 |  1.20574 | 1.09806 | 0.855881 |  1.51781 | TSM+LLM-COT-RF_blend_ramp_bestval_path        |
| 25 |         5 |  5.59217 | 2.36478 | 1.7916   |  3.10262 | TSM+LLM-COT-RF_blend_ramp_bestval_path        |
| 26 |        20 | 40.7115  | 6.38055 | 4.26878  |  7.68057 | TSM+LLM-COT-RF_blend_ramp_bestval_path        |
| 27 |        30 | 77.1687  | 8.78457 | 6.31244  | 11.909   | TSM+LLM-COT-RF_blend_ramp_bestval_path        |
| 28 |         1 |  1.20574 | 1.09806 | 0.855881 |  1.51781 | TSM+LLM-COT-RF_blend_ramp_bestval_path_h1base |
| 29 |         5 |  5.59217 | 2.36478 | 1.7916   |  3.10262 | TSM+LLM-COT-RF_blend_ramp_bestval_path_h1base |
| 30 |        20 | 40.7115  | 6.38055 | 4.26878  |  7.68057 | TSM+LLM-COT-RF_blend_ramp_bestval_path_h1base |
| 31 |        30 | 77.1687  | 8.78457 | 6.31244  | 11.909   | TSM+LLM-COT-RF_blend_ramp_bestval_path_h1base |
| 32 |         1 |  1.20574 | 1.09806 | 0.855881 |  1.51781 | TSM+LLM-COT-RF_blend_ramp_bestval_h1          |
| 33 |         5 |  5.64911 | 2.37679 | 1.80329  |  3.12164 | TSM+LLM-COT-RF_blend_ramp_bestval_h1          |
| 34 |        20 | 40.2462  | 6.34399 | 4.28359  |  7.6831  | TSM+LLM-COT-RF_blend_ramp_bestval_h1          |
| 35 |        30 | 72.369   | 8.507   | 6.23287  | 11.696   | TSM+LLM-COT-RF_blend_ramp_bestval_h1          |
| 36 |         1 |  1.20574 | 1.09806 | 0.855881 |  1.51781 | TSM+LLM-COT-RF_blend_ramp_bestval_h5          |
| 37 |         5 |  5.53849 | 2.3534  | 1.78018  |  3.08406 | TSM+LLM-COT-RF_blend_ramp_bestval_h5          |
| 38 |        20 | 41.5789  | 6.44817 | 4.2869   |  7.73677 | TSM+LLM-COT-RF_blend_ramp_bestval_h5          |
| 39 |        30 | 83.7506  | 9.15153 | 6.48569  | 12.2882  | TSM+LLM-COT-RF_blend_ramp_bestval_h5          |
| 40 |         1 |  1.20574 | 1.09806 | 0.855881 |  1.51781 | TSM+LLM-COT-RF_blend_ramp_bestval_h20         |
| 41 |         5 |  5.59217 | 2.36478 | 1.7916   |  3.10262 | TSM+LLM-COT-RF_blend_ramp_bestval_h20         |
| 42 |        20 | 40.7115  | 6.38055 | 4.26878  |  7.68057 | TSM+LLM-COT-RF_blend_ramp_bestval_h20         |
| 43 |        30 | 77.1687  | 8.78457 | 6.31244  | 11.909   | TSM+LLM-COT-RF_blend_ramp_bestval_h20         |
| 44 |         1 |  1.20574 | 1.09806 | 0.855881 |  1.51781 | TSM+LLM-COT-RF_blend_ramp_bestval_h30         |
| 45 |         5 |  5.62023 | 2.3707  | 1.79744  |  3.11213 | TSM+LLM-COT-RF_blend_ramp_bestval_h30         |
| 46 |        20 | 40.4286  | 6.35835 | 4.27441  |  7.67859 | TSM+LLM-COT-RF_blend_ramp_bestval_h30         |
| 47 |        30 | 74.5461  | 8.63401 | 6.25272  | 11.767   | TSM+LLM-COT-RF_blend_ramp_bestval_h30         |
| 48 |         1 |  1.06139 | 1.03024 | 0.784861 |  1.39543 | naive_persistence_llm_subset                  |
| 49 |         5 |  5.2119  | 2.28296 | 1.67     |  2.90235 | naive_persistence_llm_subset                  |
| 50 |        20 | 44.2759  | 6.65401 | 4.40799  |  7.94956 | naive_persistence_llm_subset                  |
| 51 |        30 | 79.9881  | 8.94361 | 6.37986  | 12.0727  | naive_persistence_llm_subset                  |
| 52 |         1 |  5.16853 | 2.27344 | 1.65132  |  2.90012 | seasonal_naive_llm_subset                     |
| 53 |         5 |  5.2119  | 2.28296 | 1.67     |  2.90235 | seasonal_naive_llm_subset                     |
| 54 |        20 | 44.2759  | 6.65401 | 4.40799  |  7.94956 | seasonal_naive_llm_subset                     |
| 55 |        30 | 79.9881  | 8.94361 | 6.37986  | 12.0727  | seasonal_naive_llm_subset                     |
| 56 |         1 |  1.06154 | 1.03031 | 0.781283 |  1.38304 | linear_ridge_llm_subset                       |
| 57 |         5 |  4.59363 | 2.14328 | 1.56632  |  2.70398 | linear_ridge_llm_subset                       |
| 58 |        20 | 30.1544  | 5.4913  | 3.32793  |  6.07247 | linear_ridge_llm_subset                       |
| 59 |        30 | 56.9094  | 7.54383 | 4.78768  |  9.18917 | linear_ridge_llm_subset                       |
| 60 |         1 |  1.1541  | 1.07429 | 0.830618 |  1.48473 | linear_lasso_llm_subset                       |
| 61 |         5 |  4.73942 | 2.17702 | 1.61188  |  2.79853 | linear_lasso_llm_subset                       |
| 62 |        20 | 32.1131  | 5.66684 | 3.43547  |  6.16613 | linear_lasso_llm_subset                       |
| 63 |        30 | 62.0974  | 7.88019 | 5.23504  |  9.89649 | linear_lasso_llm_subset                       |
| 64 |         1 |  1.20574 | 1.09806 | 0.855881 |  1.51781 | tsm_llm_subset                                |
| 65 |         5 |  5.64911 | 2.37679 | 1.80329  |  3.12164 | tsm_llm_subset                                |
| 66 |        20 | 40.2462  | 6.34399 | 4.28359  |  7.6831  | tsm_llm_subset                                |
| 67 |        30 | 72.369   | 8.507   | 6.23287  | 11.696   | tsm_llm_subset                                |

### 4.2 Trend Classification Accuracy

Table 2 shows the accuracy of predicting price direction (up/flat/down).

|    |   horizon |   accuracy |   accuracy_up |   accuracy_down |   accuracy_flat |   n_up |   n_down |   n_flat | model                                         |
|---:|----------:|-----------:|--------------:|----------------:|----------------:|-------:|---------:|---------:|:----------------------------------------------|
|  0 |         1 |  0.395833  |     0         |       0         |       1         |     52 |       35 |       57 | naive_persistence                             |
|  1 |         5 |  0.236111  |     0         |       0         |       1         |     70 |       40 |       34 | naive_persistence                             |
|  2 |        20 |  0.118056  |     0         |       0         |       1         |     90 |       37 |       17 | naive_persistence                             |
|  3 |        30 |  0.0763889 |     0         |       0         |       1         |    100 |       33 |       11 | naive_persistence                             |
|  4 |         1 |  0.375     |     0.365385  |       0.628571  |       0.22807   |     52 |       35 |       57 | seasonal_naive                                |
|  5 |         5 |  0.236111  |     0         |       0         |       1         |     70 |       40 |       34 | seasonal_naive                                |
|  6 |        20 |  0.118056  |     0         |       0         |       1         |     90 |       37 |       17 | seasonal_naive                                |
|  7 |        30 |  0.0763889 |     0         |       0         |       1         |    100 |       33 |       11 | seasonal_naive                                |
|  8 |         1 |  0.430556  |     0.0576923 |       0.0857143 |       0.982456  |     52 |       35 |       57 | linear_ridge                                  |
|  9 |         5 |  0.451389  |     0.514286  |       0.125     |       0.705882  |     70 |       40 |       34 | linear_ridge                                  |
| 10 |        20 |  0.708333  |     0.955556  |       0.351351  |       0.176471  |     90 |       37 |       17 | linear_ridge                                  |
| 11 |        30 |  0.833333  |     0.99      |       0.606061  |       0.0909091 |    100 |       33 |       11 | linear_ridge                                  |
| 12 |         1 |  0.423611  |     0.288462  |       0.114286  |       0.736842  |     52 |       35 |       57 | linear_lasso                                  |
| 13 |         5 |  0.388889  |     0.414286  |       0.15      |       0.617647  |     70 |       40 |       34 | linear_lasso                                  |
| 14 |        20 |  0.708333  |     0.922222  |       0.459459  |       0.117647  |     90 |       37 |       17 | linear_lasso                                  |
| 15 |        30 |  0.736111  |     0.75      |       0.757576  |       0.545455  |    100 |       33 |       11 | linear_lasso                                  |
| 16 |         1 |  0.381944  |     0.0192308 |       0.114286  |       0.877193  |     52 |       35 |       57 | tsm                                           |
| 17 |         5 |  0.256944  |     0.0714286 |       0.425     |       0.441176  |     70 |       40 |       34 | tsm                                           |
| 18 |        20 |  0.388889  |     0.288889  |       0.702703  |       0.235294  |     90 |       37 |       17 | tsm                                           |
| 19 |        30 |  0.361111  |     0.19      |       0.909091  |       0.272727  |    100 |       33 |       11 | tsm                                           |
| 20 |         1 |  0.388889  |     0.0192308 |       0.0285714 |       0.947368  |     52 |       35 |       57 | TSM+LLM-COT-RF                                |
| 21 |         5 |  0.3125    |     0.157143  |       0.3       |       0.647059  |     70 |       40 |       34 | TSM+LLM-COT-RF                                |
| 22 |        20 |  0.381944  |     0.377778  |       0.459459  |       0.235294  |     90 |       37 |       17 | TSM+LLM-COT-RF                                |
| 23 |        30 |  0.361111  |     0.29      |       0.545455  |       0.454545  |    100 |       33 |       11 | TSM+LLM-COT-RF                                |
| 24 |         1 |  0.381944  |     0.0192308 |       0.114286  |       0.877193  |     52 |       35 |       57 | TSM+LLM-COT-RF_blend_ramp_bestval_path        |
| 25 |         5 |  0.256944  |     0.0714286 |       0.425     |       0.441176  |     70 |       40 |       34 | TSM+LLM-COT-RF_blend_ramp_bestval_path        |
| 26 |        20 |  0.381944  |     0.3       |       0.621622  |       0.294118  |     90 |       37 |       17 | TSM+LLM-COT-RF_blend_ramp_bestval_path        |
| 27 |        30 |  0.388889  |     0.28      |       0.727273  |       0.363636  |    100 |       33 |       11 | TSM+LLM-COT-RF_blend_ramp_bestval_path        |
| 28 |         1 |  0.381944  |     0.0192308 |       0.114286  |       0.877193  |     52 |       35 |       57 | TSM+LLM-COT-RF_blend_ramp_bestval_path_h1base |
| 29 |         5 |  0.256944  |     0.0714286 |       0.425     |       0.441176  |     70 |       40 |       34 | TSM+LLM-COT-RF_blend_ramp_bestval_path_h1base |
| 30 |        20 |  0.381944  |     0.3       |       0.621622  |       0.294118  |     90 |       37 |       17 | TSM+LLM-COT-RF_blend_ramp_bestval_path_h1base |
| 31 |        30 |  0.388889  |     0.28      |       0.727273  |       0.363636  |    100 |       33 |       11 | TSM+LLM-COT-RF_blend_ramp_bestval_path_h1base |
| 32 |         1 |  0.381944  |     0.0192308 |       0.114286  |       0.877193  |     52 |       35 |       57 | TSM+LLM-COT-RF_blend_ramp_bestval_h1          |
| 33 |         5 |  0.256944  |     0.0714286 |       0.425     |       0.441176  |     70 |       40 |       34 | TSM+LLM-COT-RF_blend_ramp_bestval_h1          |
| 34 |        20 |  0.388889  |     0.288889  |       0.702703  |       0.235294  |     90 |       37 |       17 | TSM+LLM-COT-RF_blend_ramp_bestval_h1          |
| 35 |        30 |  0.361111  |     0.19      |       0.909091  |       0.272727  |    100 |       33 |       11 | TSM+LLM-COT-RF_blend_ramp_bestval_h1          |
| 36 |         1 |  0.381944  |     0.0192308 |       0.114286  |       0.877193  |     52 |       35 |       57 | TSM+LLM-COT-RF_blend_ramp_bestval_h5          |
| 37 |         5 |  0.25      |     0.0714286 |       0.4       |       0.441176  |     70 |       40 |       34 | TSM+LLM-COT-RF_blend_ramp_bestval_h5          |
| 38 |        20 |  0.423611  |     0.355556  |       0.594595  |       0.411765  |     90 |       37 |       17 | TSM+LLM-COT-RF_blend_ramp_bestval_h5          |
| 39 |        30 |  0.361111  |     0.29      |       0.545455  |       0.454545  |    100 |       33 |       11 | TSM+LLM-COT-RF_blend_ramp_bestval_h5          |
| 40 |         1 |  0.381944  |     0.0192308 |       0.114286  |       0.877193  |     52 |       35 |       57 | TSM+LLM-COT-RF_blend_ramp_bestval_h20         |
| 41 |         5 |  0.256944  |     0.0714286 |       0.425     |       0.441176  |     70 |       40 |       34 | TSM+LLM-COT-RF_blend_ramp_bestval_h20         |
| 42 |        20 |  0.381944  |     0.3       |       0.621622  |       0.294118  |     90 |       37 |       17 | TSM+LLM-COT-RF_blend_ramp_bestval_h20         |
| 43 |        30 |  0.388889  |     0.28      |       0.727273  |       0.363636  |    100 |       33 |       11 | TSM+LLM-COT-RF_blend_ramp_bestval_h20         |
| 44 |         1 |  0.381944  |     0.0192308 |       0.114286  |       0.877193  |     52 |       35 |       57 | TSM+LLM-COT-RF_blend_ramp_bestval_h30         |
| 45 |         5 |  0.256944  |     0.0714286 |       0.425     |       0.441176  |     70 |       40 |       34 | TSM+LLM-COT-RF_blend_ramp_bestval_h30         |
| 46 |        20 |  0.381944  |     0.277778  |       0.675676  |       0.294118  |     90 |       37 |       17 | TSM+LLM-COT-RF_blend_ramp_bestval_h30         |
| 47 |        30 |  0.381944  |     0.23      |       0.878788  |       0.272727  |    100 |       33 |       11 | TSM+LLM-COT-RF_blend_ramp_bestval_h30         |
| 48 |         1 |  0.395833  |     0         |       0         |       1         |     52 |       35 |       57 | naive_persistence_llm_subset                  |
| 49 |         5 |  0.236111  |     0         |       0         |       1         |     70 |       40 |       34 | naive_persistence_llm_subset                  |
| 50 |        20 |  0.118056  |     0         |       0         |       1         |     90 |       37 |       17 | naive_persistence_llm_subset                  |
| 51 |        30 |  0.0763889 |     0         |       0         |       1         |    100 |       33 |       11 | naive_persistence_llm_subset                  |
| 52 |         1 |  0.375     |     0.365385  |       0.628571  |       0.22807   |     52 |       35 |       57 | seasonal_naive_llm_subset                     |
| 53 |         5 |  0.236111  |     0         |       0         |       1         |     70 |       40 |       34 | seasonal_naive_llm_subset                     |
| 54 |        20 |  0.118056  |     0         |       0         |       1         |     90 |       37 |       17 | seasonal_naive_llm_subset                     |
| 55 |        30 |  0.0763889 |     0         |       0         |       1         |    100 |       33 |       11 | seasonal_naive_llm_subset                     |
| 56 |         1 |  0.430556  |     0.0576923 |       0.0857143 |       0.982456  |     52 |       35 |       57 | linear_ridge_llm_subset                       |
| 57 |         5 |  0.451389  |     0.514286  |       0.125     |       0.705882  |     70 |       40 |       34 | linear_ridge_llm_subset                       |
| 58 |        20 |  0.708333  |     0.955556  |       0.351351  |       0.176471  |     90 |       37 |       17 | linear_ridge_llm_subset                       |
| 59 |        30 |  0.833333  |     0.99      |       0.606061  |       0.0909091 |    100 |       33 |       11 | linear_ridge_llm_subset                       |
| 60 |         1 |  0.423611  |     0.288462  |       0.114286  |       0.736842  |     52 |       35 |       57 | linear_lasso_llm_subset                       |
| 61 |         5 |  0.388889  |     0.414286  |       0.15      |       0.617647  |     70 |       40 |       34 | linear_lasso_llm_subset                       |
| 62 |        20 |  0.708333  |     0.922222  |       0.459459  |       0.117647  |     90 |       37 |       17 | linear_lasso_llm_subset                       |
| 63 |        30 |  0.736111  |     0.75      |       0.757576  |       0.545455  |    100 |       33 |       11 | linear_lasso_llm_subset                       |
| 64 |         1 |  0.381944  |     0.0192308 |       0.114286  |       0.877193  |     52 |       35 |       57 | tsm_llm_subset                                |
| 65 |         5 |  0.256944  |     0.0714286 |       0.425     |       0.441176  |     70 |       40 |       34 | tsm_llm_subset                                |
| 66 |        20 |  0.388889  |     0.288889  |       0.702703  |       0.235294  |     90 |       37 |       17 | tsm_llm_subset                                |
| 67 |        30 |  0.361111  |     0.19      |       0.909091  |       0.272727  |    100 |       33 |       11 | tsm_llm_subset                                |

### 4.3 Statistical Significance

Table 3 reports the results of significance tests comparing methods.

|    |   horizon | model                                         | baseline                     |   mean_error_model |   mean_error_baseline |   improvement_pct |   t_statistic |    t_pvalue | t_significant   |   wilcoxon_statistic |   wilcoxon_pvalue | wilcoxon_significant   |   dm_statistic |   dm_pvalue | dm_significant   | model_better   |
|---:|----------:|:----------------------------------------------|:-----------------------------|-------------------:|----------------------:|------------------:|--------------:|------------:|:----------------|---------------------:|------------------:|:-----------------------|---------------:|------------:|:-----------------|:---------------|
|  0 |         1 | seasonal_naive_llm_subset                     | naive_persistence_llm_subset |            5.16853 |               1.06139 |    -386.957       |    5.07044    | 1.21001e-06 | True            |                 1834 |       1.45104e-11 | True                   |     5.07044    | 3.96893e-07 | True             | False          |
|  1 |         5 | seasonal_naive_llm_subset                     | naive_persistence_llm_subset |            5.2119  |               5.2119  |       5.49269e-07 |    1.19107    | 0.235598    | False           |                 4258 |       0.556904    | False                  |     0.0263232  | 0.979       | False            | True           |
|  2 |        20 | seasonal_naive_llm_subset                     | naive_persistence_llm_subset |           44.2759  |              44.2759  |       2.22491e-06 |    0.67977    | 0.497749    | False           |                 4233 |       0.183904    | False                  |     0.216494   | 0.828603    | False            | True           |
|  3 |        30 | seasonal_naive_llm_subset                     | naive_persistence_llm_subset |           79.9881  |              79.9881  |       3.104e-06   |   -0.578827   | 0.563616    | False           |                 3978 |       0.082287    | False                  |    -0.245123   | 0.806361    | False            | True           |
|  4 |         1 | linear_ridge_llm_subset                       | naive_persistence_llm_subset |            1.06154 |               1.06139 |      -0.0135551   |    0.00242328 | 0.99807     | False           |                 5155 |       0.896859    | False                  |     0.00242328 | 0.998067    | False            | False          |
|  5 |         5 | linear_ridge_llm_subset                       | naive_persistence_llm_subset |            4.59363 |               5.2119  |      11.8626      |   -2.00294    | 0.0470741   | True            |                 3955 |       0.0116426   | True                   |    -1.15055    | 0.249918    | False            | True           |
|  6 |        20 | linear_ridge_llm_subset                       | naive_persistence_llm_subset |           30.1544  |              44.2759  |      31.8943      |   -5.67076    | 7.5947e-08  | True            |                 2425 |       2.4884e-08  | True                   |    -2.3465     | 0.0189507   | True             | True           |
|  7 |        30 | linear_ridge_llm_subset                       | naive_persistence_llm_subset |           56.9094  |              79.9881  |      28.8527      |   -7.86447    | 8.23678e-13 | True            |                  726 |       3.17675e-19 | True                   |    -1.91128    | 0.0559692   | False            | True           |
|  8 |         1 | linear_lasso_llm_subset                       | naive_persistence_llm_subset |            1.1541  |               1.06139 |      -8.73423     |    1.16037    | 0.24783     | False           |                 4553 |       0.183452    | False                  |     1.16037    | 0.245896    | False            | False          |
|  9 |         5 | linear_lasso_llm_subset                       | naive_persistence_llm_subset |            4.73942 |               5.2119  |       9.06537     |   -1.85969    | 0.0649833   | False           |                 4418 |       0.109725    | False                  |    -1.14841    | 0.250799    | False            | True           |
| 10 |        20 | linear_lasso_llm_subset                       | naive_persistence_llm_subset |           32.1131  |              44.2759  |      27.4704      |   -5.48329    | 1.83923e-07 | True            |                 1499 |       1.16409e-13 | True                   |    -2.64988    | 0.00805199  | True             | True           |
| 11 |        30 | linear_lasso_llm_subset                       | naive_persistence_llm_subset |           62.0974  |              79.9881  |      22.3667      |   -6.53209    | 1.0595e-09  | True            |                  443 |       1.62218e-21 | True                   |    -1.64973    | 0.0989982   | False            | True           |
| 12 |         1 | tsm_llm_subset                                | naive_persistence_llm_subset |            1.20574 |               1.06139 |     -13.5994      |    2.1918     | 0.0300118   | True            |                 3956 |       0.0117088   | True                   |     2.1918     | 0.0283939   | True             | False          |
| 13 |         5 | tsm_llm_subset                                | naive_persistence_llm_subset |            5.64911 |               5.2119  |      -8.38883     |    1.54257    | 0.125144    | False           |                 4440 |       0.119813    | False                  |     0.774825   | 0.438443    | False            | False          |
| 14 |        20 | tsm_llm_subset                                | naive_persistence_llm_subset |           40.2462  |              44.2759  |       9.10133     |   -2.80664    | 0.0057048   | True            |                 4600 |       0.216284    | False                  |    -1.24077    | 0.214692    | False            | True           |
| 15 |        30 | tsm_llm_subset                                | naive_persistence_llm_subset |           72.369   |              79.9881  |       9.52529     |   -3.08456    | 0.00244758  | True            |                 4970 |       0.618078    | False                  |    -0.821835   | 0.411171    | False            | True           |
| 16 |         1 | TSM+LLM-COT-RF                                | naive_persistence_llm_subset |            1.18834 |               1.06139 |     -11.9604      |    2.04274    | 0.0429156   | True            |                 3928 |       0.00997654  | True                   |     2.04274    | 0.0410786   | True             | False          |
| 17 |         5 | TSM+LLM-COT-RF                                | naive_persistence_llm_subset |            5.14291 |               5.2119  |       1.32361     |   -0.298652   | 0.765639    | False           |                 4870 |       0.485173    | False                  |    -0.215127   | 0.829668    | False            | True           |
| 18 |        20 | TSM+LLM-COT-RF                                | naive_persistence_llm_subset |           42.9263  |              44.2759  |       3.04806     |   -0.511567   | 0.609743    | False           |                 4579 |       0.201126    | False                  |    -0.762181   | 0.445952    | False            | True           |
| 19 |        30 | TSM+LLM-COT-RF                                | naive_persistence_llm_subset |           83.7506  |              79.9881  |      -4.70381     |    0.633304   | 0.527547    | False           |                 5020 |       0.689996    | False                  |     1.18801    | 0.234831    | False            | False          |
| 20 |         1 | TSM+LLM-COT-RF_blend_ramp_bestval_path        | naive_persistence_llm_subset |            1.20574 |               1.06139 |     -13.5994      |    2.1918     | 0.0300118   | True            |                 3956 |       0.0117088   | True                   |     2.1918     | 0.0283939   | True             | False          |
| 21 |         5 | TSM+LLM-COT-RF_blend_ramp_bestval_path        | naive_persistence_llm_subset |            5.59217 |               5.2119  |      -7.29619     |    1.41042    | 0.160587    | False           |                 4490 |       0.145436    | False                  |     0.705913   | 0.480242    | False            | False          |
| 22 |        20 | TSM+LLM-COT-RF_blend_ramp_bestval_path        | naive_persistence_llm_subset |           40.7115  |              44.2759  |       8.0504      |   -2.32496    | 0.021481    | True            |                 4239 |       0.0504166   | False                  |    -1.42994    | 0.152735    | False            | True           |
| 23 |        30 | TSM+LLM-COT-RF_blend_ramp_bestval_path        | naive_persistence_llm_subset |           77.1687  |              79.9881  |       3.52478     |   -0.895068   | 0.372255    | False           |                 4680 |       0.281514    | False                  |    -0.678165   | 0.497667    | False            | True           |
| 24 |         1 | TSM+LLM-COT-RF_blend_ramp_bestval_path_h1base | naive_persistence_llm_subset |            1.20574 |               1.06139 |     -13.5994      |    2.1918     | 0.0300118   | True            |                 3956 |       0.0117088   | True                   |     2.1918     | 0.0283939   | True             | False          |
| 25 |         5 | TSM+LLM-COT-RF_blend_ramp_bestval_path_h1base | naive_persistence_llm_subset |            5.59217 |               5.2119  |      -7.29619     |    1.41042    | 0.160587    | False           |                 4490 |       0.145436    | False                  |     0.705913   | 0.480242    | False            | False          |
| 26 |        20 | TSM+LLM-COT-RF_blend_ramp_bestval_path_h1base | naive_persistence_llm_subset |           40.7115  |              44.2759  |       8.0504      |   -2.32496    | 0.021481    | True            |                 4239 |       0.0504166   | False                  |    -1.42994    | 0.152735    | False            | True           |
| 27 |        30 | TSM+LLM-COT-RF_blend_ramp_bestval_path_h1base | naive_persistence_llm_subset |           77.1687  |              79.9881  |       3.52478     |   -0.895068   | 0.372255    | False           |                 4680 |       0.281514    | False                  |    -0.678165   | 0.497667    | False            | True           |
| 28 |         1 | TSM+LLM-COT-RF_blend_ramp_bestval_h1          | naive_persistence_llm_subset |            1.20574 |               1.06139 |     -13.5994      |    2.1918     | 0.0300118   | True            |                 3956 |       0.0117088   | True                   |     2.1918     | 0.0283939   | True             | False          |
| 29 |         5 | TSM+LLM-COT-RF_blend_ramp_bestval_h1          | naive_persistence_llm_subset |            5.64911 |               5.2119  |      -8.38883     |    1.54257    | 0.125144    | False           |                 4440 |       0.119813    | False                  |     0.774825   | 0.438443    | False            | False          |
| 30 |        20 | TSM+LLM-COT-RF_blend_ramp_bestval_h1          | naive_persistence_llm_subset |           40.2462  |              44.2759  |       9.10134     |   -2.80664    | 0.0057048   | True            |                 4600 |       0.216284    | False                  |    -1.24077    | 0.214692    | False            | True           |
| 31 |        30 | TSM+LLM-COT-RF_blend_ramp_bestval_h1          | naive_persistence_llm_subset |           72.369   |              79.9881  |       9.52528     |   -3.08456    | 0.00244759  | True            |                 4970 |       0.618078    | False                  |    -0.821835   | 0.411171    | False            | True           |
| 32 |         1 | TSM+LLM-COT-RF_blend_ramp_bestval_h5          | naive_persistence_llm_subset |            1.20574 |               1.06139 |     -13.5994      |    2.1918     | 0.0300118   | True            |                 3956 |       0.0117088   | True                   |     2.1918     | 0.0283939   | True             | False          |
| 33 |         5 | TSM+LLM-COT-RF_blend_ramp_bestval_h5          | naive_persistence_llm_subset |            5.53849 |               5.2119  |      -6.2662      |    1.2691     | 0.206468    | False           |                 4568 |       0.193503    | False                  |     0.634964   | 0.525452    | False            | False          |
| 34 |        20 | TSM+LLM-COT-RF_blend_ramp_bestval_h5          | naive_persistence_llm_subset |           41.5789  |              44.2759  |       6.09133     |   -1.38083    | 0.169485    | False           |                 4261 |       0.0558072   | False                  |    -1.51105    | 0.130775    | False            | True           |
| 35 |        30 | TSM+LLM-COT-RF_blend_ramp_bestval_h5          | naive_persistence_llm_subset |           83.7506  |              79.9881  |      -4.70381     |    0.633304   | 0.527547    | False           |                 5020 |       0.689996    | False                  |     1.18801    | 0.234831    | False            | False          |
| 36 |         1 | TSM+LLM-COT-RF_blend_ramp_bestval_h20         | naive_persistence_llm_subset |            1.20574 |               1.06139 |     -13.5994      |    2.1918     | 0.0300118   | True            |                 3956 |       0.0117088   | True                   |     2.1918     | 0.0283939   | True             | False          |
| 37 |         5 | TSM+LLM-COT-RF_blend_ramp_bestval_h20         | naive_persistence_llm_subset |            5.59217 |               5.2119  |      -7.29619     |    1.41042    | 0.160587    | False           |                 4490 |       0.145436    | False                  |     0.705913   | 0.480242    | False            | False          |
| 38 |        20 | TSM+LLM-COT-RF_blend_ramp_bestval_h20         | naive_persistence_llm_subset |           40.7115  |              44.2759  |       8.0504      |   -2.32496    | 0.021481    | True            |                 4239 |       0.0504166   | False                  |    -1.42994    | 0.152735    | False            | True           |
| 39 |        30 | TSM+LLM-COT-RF_blend_ramp_bestval_h20         | naive_persistence_llm_subset |           77.1687  |              79.9881  |       3.52478     |   -0.895068   | 0.372255    | False           |                 4680 |       0.281514    | False                  |    -0.678165   | 0.497667    | False            | True           |
| 40 |         1 | TSM+LLM-COT-RF_blend_ramp_bestval_h30         | naive_persistence_llm_subset |            1.20574 |               1.06139 |     -13.5994      |    2.1918     | 0.0300118   | True            |                 3956 |       0.0117088   | True                   |     2.1918     | 0.0283939   | True             | False          |
| 41 |         5 | TSM+LLM-COT-RF_blend_ramp_bestval_h30         | naive_persistence_llm_subset |            5.62023 |               5.2119  |      -7.83468     |    1.47768    | 0.141692    | False           |                 4458 |       0.128596    | False                  |     0.740635   | 0.458915    | False            | False          |
| 42 |        20 | TSM+LLM-COT-RF_blend_ramp_bestval_h30         | naive_persistence_llm_subset |           40.4286  |              44.2759  |       8.68939     |   -2.67088    | 0.00844214  | True            |                 4409 |       0.105796    | False                  |    -1.33524    | 0.181799    | False            | True           |
| 43 |        30 | TSM+LLM-COT-RF_blend_ramp_bestval_h30         | naive_persistence_llm_subset |           74.5461  |              79.9881  |       6.80354     |   -2.1958     | 0.0297178   | True            |                 4708 |       0.307215    | False                  |    -0.79849    | 0.424586    | False            | True           |


## 5. Robustness Analysis

### 5.1 Noise Injection Test

To assess robustness, we inject noise into forecast paths at levels of 5%, 10%, 20%, 
and 30% of the forecast standard deviation.

|     |   horizon |      mse |    rmse |      mae |     mape |   noise_level | model                                         |
|----:|----------:|---------:|--------:|---------:|---------:|--------------:|:----------------------------------------------|
|   0 |         1 |  1.20222 | 1.09646 | 0.854421 |  1.51606 |          0.05 | tsm                                           |
|   1 |         5 |  5.65917 | 2.3789  | 1.8026   |  3.12079 |          0.05 | tsm                                           |
|   2 |        20 | 40.2456  | 6.34394 | 4.28232  |  7.6806  |          0.05 | tsm                                           |
|   3 |        30 | 72.3504  | 8.5059  | 6.22953  | 11.69    |          0.05 | tsm                                           |
|   4 |         1 |  1.19999 | 1.09544 | 0.853554 |  1.51538 |          0.1  | tsm                                           |
|   5 |         5 |  5.67089 | 2.38136 | 1.80194  |  3.11996 |          0.1  | tsm                                           |
|   6 |        20 | 40.2464  | 6.34401 | 4.28108  |  7.67814 |          0.1  | tsm                                           |
|   7 |        30 | 72.3328  | 8.50487 | 6.2262   | 11.684   |          0.1  | tsm                                           |
|   8 |         1 |  1.19937 | 1.09516 | 0.85291  |  1.51572 |          0.2  | tsm                                           |
|   9 |         5 |  5.69935 | 2.38733 | 1.80217  |  3.12112 |          0.2  | tsm                                           |
|  10 |        20 | 40.2522  | 6.34446 | 4.27922  |  7.67436 |          0.2  | tsm                                           |
|  11 |        30 | 72.301   | 8.503   | 6.21953  | 11.6721  |          0.2  | tsm                                           |
|  12 |         1 |  1.20386 | 1.0972  | 0.854502 |  1.51948 |          0.3  | tsm                                           |
|  13 |         5 |  5.73449 | 2.39468 | 1.80254  |  3.12249 |          0.3  | tsm                                           |
|  14 |        20 | 40.2634  | 6.34534 | 4.27937  |  7.67435 |          0.3  | tsm                                           |
|  15 |        30 | 72.2734  | 8.50138 | 6.21286  | 11.6602  |          0.3  | tsm                                           |
|  16 |         1 |  1.17566 | 1.08428 | 0.840015 |  1.49511 |          0.05 | TSM+LLM-COT-RF                                |
|  17 |         5 |  5.14945 | 2.26924 | 1.71712  |  2.99183 |          0.05 | TSM+LLM-COT-RF                                |
|  18 |        20 | 42.9332  | 6.55234 | 4.36295  |  7.8885  |          0.05 | TSM+LLM-COT-RF                                |
|  19 |        30 | 83.765   | 9.15232 | 6.48249  | 12.2828  |          0.05 | TSM+LLM-COT-RF                                |
|  20 |         1 |  1.16499 | 1.07935 | 0.836767 |  1.49042 |          0.1  | TSM+LLM-COT-RF                                |
|  21 |         5 |  5.15791 | 2.2711  | 1.71889  |  2.99526 |          0.1  | TSM+LLM-COT-RF                                |
|  22 |        20 | 42.9416  | 6.55299 | 4.36182  |  7.88682 |          0.1  | TSM+LLM-COT-RF                                |
|  23 |        30 | 83.7807  | 9.15318 | 6.47929  | 12.2773  |          0.1  | TSM+LLM-COT-RF                                |
|  24 |         1 |  1.14972 | 1.07225 | 0.830385 |  1.48125 |          0.2  | TSM+LLM-COT-RF                                |
|  25 |         5 |  5.18061 | 2.27609 | 1.72243  |  3.0021  |          0.2  | TSM+LLM-COT-RF                                |
|  26 |        20 | 42.963   | 6.55461 | 4.36153  |  7.88711 |          0.2  | TSM+LLM-COT-RF                                |
|  27 |        30 | 83.816   | 9.15511 | 6.47288  | 12.2664  |          0.2  | TSM+LLM-COT-RF                                |
|  28 |         1 |  1.14252 | 1.06889 | 0.82663  |  1.4769  |          0.3  | TSM+LLM-COT-RF                                |
|  29 |         5 |  5.211   | 2.28276 | 1.72749  |  3.01164 |          0.3  | TSM+LLM-COT-RF                                |
|  30 |        20 | 42.9903  | 6.5567  | 4.36531  |  7.89419 |          0.3  | TSM+LLM-COT-RF                                |
|  31 |        30 | 83.8565  | 9.15732 | 6.46648  | 12.2555  |          0.3  | TSM+LLM-COT-RF                                |
|  32 |         1 |  1.1993  | 1.09512 | 0.853853 |  1.51514 |          0.05 | TSM+LLM-COT-RF_blend_ramp_bestval_path        |
|  33 |         5 |  5.59806 | 2.36602 | 1.78982  |  3.09991 |          0.05 | TSM+LLM-COT-RF_blend_ramp_bestval_path        |
|  34 |        20 | 40.709   | 6.38036 | 4.26831  |  7.6798  |          0.05 | TSM+LLM-COT-RF_blend_ramp_bestval_path        |
|  35 |        30 | 77.1658  | 8.7844  | 6.30883  | 11.9027  |          0.05 | TSM+LLM-COT-RF_blend_ramp_bestval_path        |
|  36 |         1 |  1.1944  | 1.09288 | 0.852511 |  1.51371 |          0.1  | TSM+LLM-COT-RF_blend_ramp_bestval_path        |
|  37 |         5 |  5.60572 | 2.36764 | 1.7887   |  3.09839 |          0.1  | TSM+LLM-COT-RF_blend_ramp_bestval_path        |
|  38 |        20 | 40.7079  | 6.38028 | 4.26785  |  7.67904 |          0.1  | TSM+LLM-COT-RF_blend_ramp_bestval_path        |
|  39 |        30 | 77.164   | 8.7843  | 6.30521  | 11.8964  |          0.1  | TSM+LLM-COT-RF_blend_ramp_bestval_path        |
|  40 |         1 |  1.18919 | 1.0905  | 0.849827 |  1.51084 |          0.2  | TSM+LLM-COT-RF_blend_ramp_bestval_path        |
|  41 |         5 |  5.62637 | 2.372   | 1.78646  |  3.09534 |          0.2  | TSM+LLM-COT-RF_blend_ramp_bestval_path        |
|  42 |        20 | 40.7098  | 6.38042 | 4.26759  |  7.67871 |          0.2  | TSM+LLM-COT-RF_blend_ramp_bestval_path        |
|  43 |        30 | 77.1637  | 8.78429 | 6.29797  | 11.8837  |          0.2  | TSM+LLM-COT-RF_blend_ramp_bestval_path        |
|  44 |         1 |  1.19012 | 1.09093 | 0.849131 |  1.51113 |          0.3  | TSM+LLM-COT-RF_blend_ramp_bestval_path        |
|  45 |         5 |  5.65412 | 2.37784 | 1.78436  |  3.09251 |          0.3  | TSM+LLM-COT-RF_blend_ramp_bestval_path        |
|  46 |        20 | 40.7169  | 6.38098 | 4.26835  |  7.68035 |          0.3  | TSM+LLM-COT-RF_blend_ramp_bestval_path        |
|  47 |        30 | 77.1681  | 8.78453 | 6.29074  | 11.8711  |          0.3  | TSM+LLM-COT-RF_blend_ramp_bestval_path        |
|  48 |         1 |  1.1993  | 1.09512 | 0.853853 |  1.51514 |          0.05 | TSM+LLM-COT-RF_blend_ramp_bestval_path_h1base |
|  49 |         5 |  5.59806 | 2.36602 | 1.78982  |  3.09991 |          0.05 | TSM+LLM-COT-RF_blend_ramp_bestval_path_h1base |
|  50 |        20 | 40.709   | 6.38036 | 4.26831  |  7.6798  |          0.05 | TSM+LLM-COT-RF_blend_ramp_bestval_path_h1base |
|  51 |        30 | 77.1658  | 8.7844  | 6.30883  | 11.9027  |          0.05 | TSM+LLM-COT-RF_blend_ramp_bestval_path_h1base |
|  52 |         1 |  1.1944  | 1.09288 | 0.852511 |  1.51371 |          0.1  | TSM+LLM-COT-RF_blend_ramp_bestval_path_h1base |
|  53 |         5 |  5.60572 | 2.36764 | 1.7887   |  3.09839 |          0.1  | TSM+LLM-COT-RF_blend_ramp_bestval_path_h1base |
|  54 |        20 | 40.7079  | 6.38028 | 4.26785  |  7.67904 |          0.1  | TSM+LLM-COT-RF_blend_ramp_bestval_path_h1base |
|  55 |        30 | 77.164   | 8.7843  | 6.30521  | 11.8964  |          0.1  | TSM+LLM-COT-RF_blend_ramp_bestval_path_h1base |
|  56 |         1 |  1.18919 | 1.0905  | 0.849827 |  1.51084 |          0.2  | TSM+LLM-COT-RF_blend_ramp_bestval_path_h1base |
|  57 |         5 |  5.62637 | 2.372   | 1.78646  |  3.09534 |          0.2  | TSM+LLM-COT-RF_blend_ramp_bestval_path_h1base |
|  58 |        20 | 40.7098  | 6.38042 | 4.26759  |  7.67871 |          0.2  | TSM+LLM-COT-RF_blend_ramp_bestval_path_h1base |
|  59 |        30 | 77.1637  | 8.78429 | 6.29797  | 11.8837  |          0.2  | TSM+LLM-COT-RF_blend_ramp_bestval_path_h1base |
|  60 |         1 |  1.19012 | 1.09093 | 0.849131 |  1.51113 |          0.3  | TSM+LLM-COT-RF_blend_ramp_bestval_path_h1base |
|  61 |         5 |  5.65412 | 2.37784 | 1.78436  |  3.09251 |          0.3  | TSM+LLM-COT-RF_blend_ramp_bestval_path_h1base |
|  62 |        20 | 40.7169  | 6.38098 | 4.26835  |  7.68035 |          0.3  | TSM+LLM-COT-RF_blend_ramp_bestval_path_h1base |
|  63 |        30 | 77.1681  | 8.78453 | 6.29074  | 11.8711  |          0.3  | TSM+LLM-COT-RF_blend_ramp_bestval_path_h1base |
|  64 |         1 |  1.20222 | 1.09646 | 0.854421 |  1.51606 |          0.05 | TSM+LLM-COT-RF_blend_ramp_bestval_h1          |
|  65 |         5 |  5.65917 | 2.3789  | 1.8026   |  3.12079 |          0.05 | TSM+LLM-COT-RF_blend_ramp_bestval_h1          |
|  66 |        20 | 40.2456  | 6.34394 | 4.28232  |  7.6806  |          0.05 | TSM+LLM-COT-RF_blend_ramp_bestval_h1          |
|  67 |        30 | 72.3504  | 8.5059  | 6.22953  | 11.69    |          0.05 | TSM+LLM-COT-RF_blend_ramp_bestval_h1          |
|  68 |         1 |  1.19999 | 1.09544 | 0.853554 |  1.51538 |          0.1  | TSM+LLM-COT-RF_blend_ramp_bestval_h1          |
|  69 |         5 |  5.67089 | 2.38136 | 1.80194  |  3.11996 |          0.1  | TSM+LLM-COT-RF_blend_ramp_bestval_h1          |
|  70 |        20 | 40.2464  | 6.34401 | 4.28108  |  7.67814 |          0.1  | TSM+LLM-COT-RF_blend_ramp_bestval_h1          |
|  71 |        30 | 72.3328  | 8.50487 | 6.2262   | 11.684   |          0.1  | TSM+LLM-COT-RF_blend_ramp_bestval_h1          |
|  72 |         1 |  1.19937 | 1.09516 | 0.85291  |  1.51572 |          0.2  | TSM+LLM-COT-RF_blend_ramp_bestval_h1          |
|  73 |         5 |  5.69935 | 2.38733 | 1.80217  |  3.12112 |          0.2  | TSM+LLM-COT-RF_blend_ramp_bestval_h1          |
|  74 |        20 | 40.2522  | 6.34446 | 4.27922  |  7.67436 |          0.2  | TSM+LLM-COT-RF_blend_ramp_bestval_h1          |
|  75 |        30 | 72.301   | 8.503   | 6.21953  | 11.6721  |          0.2  | TSM+LLM-COT-RF_blend_ramp_bestval_h1          |
|  76 |         1 |  1.20386 | 1.0972  | 0.854502 |  1.51948 |          0.3  | TSM+LLM-COT-RF_blend_ramp_bestval_h1          |
|  77 |         5 |  5.73449 | 2.39468 | 1.80254  |  3.12249 |          0.3  | TSM+LLM-COT-RF_blend_ramp_bestval_h1          |
|  78 |        20 | 40.2634  | 6.34534 | 4.27937  |  7.67435 |          0.3  | TSM+LLM-COT-RF_blend_ramp_bestval_h1          |
|  79 |        30 | 72.2734  | 8.50138 | 6.21286  | 11.6602  |          0.3  | TSM+LLM-COT-RF_blend_ramp_bestval_h1          |
|  80 |         1 |  1.19386 | 1.09264 | 0.851934 |  1.51196 |          0.05 | TSM+LLM-COT-RF_blend_ramp_bestval_h5          |
|  81 |         5 |  5.54486 | 2.35475 | 1.77875  |  3.08197 |          0.05 | TSM+LLM-COT-RF_blend_ramp_bestval_h5          |
|  82 |        20 | 41.5762  | 6.44796 | 4.28682  |  7.73676 |          0.05 | TSM+LLM-COT-RF_blend_ramp_bestval_h5          |
|  83 |        30 | 83.7661  | 9.15238 | 6.48221  | 12.2824  |          0.05 | TSM+LLM-COT-RF_blend_ramp_bestval_h5          |
|  84 |         1 |  1.18459 | 1.08839 | 0.848673 |  1.50735 |          0.1  | TSM+LLM-COT-RF_blend_ramp_bestval_h5          |
|  85 |         5 |  5.55382 | 2.35665 | 1.77732  |  3.07988 |          0.1  | TSM+LLM-COT-RF_blend_ramp_bestval_h5          |
|  86 |        20 | 41.5754  | 6.4479  | 4.28706  |  7.73732 |          0.1  | TSM+LLM-COT-RF_blend_ramp_bestval_h5          |
|  87 |        30 | 83.7834  | 9.15333 | 6.47872  | 12.2765  |          0.1  | TSM+LLM-COT-RF_blend_ramp_bestval_h5          |
|  88 |         1 |  1.17386 | 1.08345 | 0.845485 |  1.50362 |          0.2  | TSM+LLM-COT-RF_blend_ramp_bestval_h5          |
|  89 |         5 |  5.57947 | 2.36209 | 1.77563  |  3.07751 |          0.2  | TSM+LLM-COT-RF_blend_ramp_bestval_h5          |
|  90 |        20 | 41.5794  | 6.44821 | 4.28837  |  7.73995 |          0.2  | TSM+LLM-COT-RF_blend_ramp_bestval_h5          |
|  91 |        30 | 83.8229  | 9.15548 | 6.47175  | 12.2647  |          0.2  | TSM+LLM-COT-RF_blend_ramp_bestval_h5          |
|  92 |         1 |  1.17356 | 1.08331 | 0.84458  |  1.50358 |          0.3  | TSM+LLM-COT-RF_blend_ramp_bestval_h5          |
|  93 |         5 |  5.61545 | 2.36969 | 1.77598  |  3.0783  |          0.3  | TSM+LLM-COT-RF_blend_ramp_bestval_h5          |
|  94 |        20 | 41.5908  | 6.44909 | 4.29072  |  7.74457 |          0.3  | TSM+LLM-COT-RF_blend_ramp_bestval_h5          |
|  95 |        30 | 83.8691  | 9.15801 | 6.46477  | 12.2529  |          0.3  | TSM+LLM-COT-RF_blend_ramp_bestval_h5          |
|  96 |         1 |  1.1993  | 1.09512 | 0.853853 |  1.51514 |          0.05 | TSM+LLM-COT-RF_blend_ramp_bestval_h20         |
|  97 |         5 |  5.59806 | 2.36602 | 1.78982  |  3.09991 |          0.05 | TSM+LLM-COT-RF_blend_ramp_bestval_h20         |
|  98 |        20 | 40.709   | 6.38036 | 4.26831  |  7.6798  |          0.05 | TSM+LLM-COT-RF_blend_ramp_bestval_h20         |
|  99 |        30 | 77.1658  | 8.7844  | 6.30883  | 11.9027  |          0.05 | TSM+LLM-COT-RF_blend_ramp_bestval_h20         |
| 100 |         1 |  1.1944  | 1.09288 | 0.852511 |  1.51371 |          0.1  | TSM+LLM-COT-RF_blend_ramp_bestval_h20         |
| 101 |         5 |  5.60572 | 2.36764 | 1.7887   |  3.09839 |          0.1  | TSM+LLM-COT-RF_blend_ramp_bestval_h20         |
| 102 |        20 | 40.7079  | 6.38028 | 4.26785  |  7.67904 |          0.1  | TSM+LLM-COT-RF_blend_ramp_bestval_h20         |
| 103 |        30 | 77.164   | 8.7843  | 6.30521  | 11.8964  |          0.1  | TSM+LLM-COT-RF_blend_ramp_bestval_h20         |
| 104 |         1 |  1.18919 | 1.0905  | 0.849827 |  1.51084 |          0.2  | TSM+LLM-COT-RF_blend_ramp_bestval_h20         |
| 105 |         5 |  5.62637 | 2.372   | 1.78646  |  3.09534 |          0.2  | TSM+LLM-COT-RF_blend_ramp_bestval_h20         |
| 106 |        20 | 40.7098  | 6.38042 | 4.26759  |  7.67871 |          0.2  | TSM+LLM-COT-RF_blend_ramp_bestval_h20         |
| 107 |        30 | 77.1637  | 8.78429 | 6.29797  | 11.8837  |          0.2  | TSM+LLM-COT-RF_blend_ramp_bestval_h20         |
| 108 |         1 |  1.19012 | 1.09093 | 0.849131 |  1.51113 |          0.3  | TSM+LLM-COT-RF_blend_ramp_bestval_h20         |
| 109 |         5 |  5.65412 | 2.37784 | 1.78436  |  3.09251 |          0.3  | TSM+LLM-COT-RF_blend_ramp_bestval_h20         |
| 110 |        20 | 40.7169  | 6.38098 | 4.26835  |  7.68035 |          0.3  | TSM+LLM-COT-RF_blend_ramp_bestval_h20         |
| 111 |        30 | 77.1681  | 8.78453 | 6.29074  | 11.8711  |          0.3  | TSM+LLM-COT-RF_blend_ramp_bestval_h20         |
| 112 |         1 |  1.20114 | 1.09596 | 0.854325 |  1.51591 |          0.05 | TSM+LLM-COT-RF_blend_ramp_bestval_h30         |
| 113 |         5 |  5.62784 | 2.37231 | 1.79597  |  3.10994 |          0.05 | TSM+LLM-COT-RF_blend_ramp_bestval_h30         |
| 114 |        20 | 40.4273  | 6.35825 | 4.27361  |  7.6771  |          0.05 | TSM+LLM-COT-RF_blend_ramp_bestval_h30         |
| 115 |        30 | 74.5345  | 8.63334 | 6.24903  | 11.7604  |          0.05 | TSM+LLM-COT-RF_blend_ramp_bestval_h30         |
| 116 |         1 |  1.19784 | 1.09446 | 0.853455 |  1.51524 |          0.1  | TSM+LLM-COT-RF_blend_ramp_bestval_h30         |
| 117 |         5 |  5.63708 | 2.37425 | 1.79521  |  3.10902 |          0.1  | TSM+LLM-COT-RF_blend_ramp_bestval_h30         |
| 118 |        20 | 40.4274  | 6.35826 | 4.27283  |  7.67562 |          0.1  | TSM+LLM-COT-RF_blend_ramp_bestval_h30         |
| 119 |        30 | 74.5239  | 8.63272 | 6.24533  | 11.7539  |          0.1  | TSM+LLM-COT-RF_blend_ramp_bestval_h30         |
| 120 |         1 |  1.19516 | 1.09323 | 0.851873 |  1.51415 |          0.2  | TSM+LLM-COT-RF_blend_ramp_bestval_h30         |
| 121 |         5 |  5.66047 | 2.37917 | 1.7941   |  3.10793 |          0.2  | TSM+LLM-COT-RF_blend_ramp_bestval_h30         |
| 122 |        20 | 40.4314  | 6.35857 | 4.27192  |  7.67383 |          0.2  | TSM+LLM-COT-RF_blend_ramp_bestval_h30         |
| 123 |        30 | 74.5059  | 8.63168 | 6.23794  | 11.7408  |          0.2  | TSM+LLM-COT-RF_blend_ramp_bestval_h30         |
| 124 |         1 |  1.1977  | 1.0944  | 0.852124 |  1.51588 |          0.3  | TSM+LLM-COT-RF_blend_ramp_bestval_h30         |
| 125 |         5 |  5.6904  | 2.38546 | 1.79312  |  3.10704 |          0.3  | TSM+LLM-COT-RF_blend_ramp_bestval_h30         |
| 126 |        20 | 40.4405  | 6.35929 | 4.27225  |  7.67443 |          0.3  | TSM+LLM-COT-RF_blend_ramp_bestval_h30         |
| 127 |        30 | 74.4921  | 8.63088 | 6.23055  | 11.7277  |          0.3  | TSM+LLM-COT-RF_blend_ramp_bestval_h30         |

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

- **Price accuracy**: linear_ridge_llm_subset has the lowest average MSE (23.180).
- **TSM vs naive**: TSM MSE is 0.9x the naive baseline on average.
- **Directional accuracy**: linear_ridge_llm_subset has the highest average trend accuracy (0.606).

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


