# EU ETS Futures Forecasting with TSM + LLM Refinement

**Authors**: Auto-generated

**Generated**: 2026-03-06 14:56:41

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

|    |   horizon |      mse |    rmse |      mae |     mape | model                                                  |
|---:|----------:|---------:|--------:|---------:|---------:|:-------------------------------------------------------|
|  0 |         1 |  1.06139 | 1.03024 | 0.784861 |  1.39543 | naive_persistence                                      |
|  1 |         5 |  5.2119  | 2.28296 | 1.67     |  2.90235 | naive_persistence                                      |
|  2 |        20 | 44.2759  | 6.65401 | 4.40799  |  7.94956 | naive_persistence                                      |
|  3 |        30 | 79.9881  | 8.94361 | 6.37986  | 12.0727  | naive_persistence                                      |
|  4 |         1 |  5.16853 | 2.27344 | 1.65132  |  2.90012 | seasonal_naive                                         |
|  5 |         5 |  5.2119  | 2.28296 | 1.67     |  2.90235 | seasonal_naive                                         |
|  6 |        20 | 44.2759  | 6.65401 | 4.40799  |  7.94956 | seasonal_naive                                         |
|  7 |        30 | 79.9881  | 8.94361 | 6.37986  | 12.0727  | seasonal_naive                                         |
|  8 |         1 |  1.06154 | 1.03031 | 0.781283 |  1.38304 | linear_ridge                                           |
|  9 |         5 |  4.59363 | 2.14328 | 1.56632  |  2.70398 | linear_ridge                                           |
| 10 |        20 | 30.1544  | 5.4913  | 3.32793  |  6.07247 | linear_ridge                                           |
| 11 |        30 | 56.9094  | 7.54383 | 4.78768  |  9.18917 | linear_ridge                                           |
| 12 |         1 |  1.1541  | 1.07429 | 0.830618 |  1.48473 | linear_lasso                                           |
| 13 |         5 |  4.73942 | 2.17702 | 1.61188  |  2.79853 | linear_lasso                                           |
| 14 |        20 | 32.1131  | 5.66684 | 3.43547  |  6.16613 | linear_lasso                                           |
| 15 |        30 | 62.0974  | 7.88019 | 5.23504  |  9.89649 | linear_lasso                                           |
| 16 |         1 |  1.20574 | 1.09806 | 0.855881 |  1.51781 | tsm                                                    |
| 17 |         5 |  5.64911 | 2.37679 | 1.80329  |  3.12164 | tsm                                                    |
| 18 |        20 | 40.2462  | 6.34399 | 4.28359  |  7.6831  | tsm                                                    |
| 19 |        30 | 72.369   | 8.507   | 6.23287  | 11.696   | tsm                                                    |
| 20 |         1 |  1.13673 | 1.06618 | 0.811458 |  1.43468 | linear_ridge+LLM-COT-RF                                |
| 21 |         5 |  5.27228 | 2.29615 | 1.6709   |  2.88414 | linear_ridge+LLM-COT-RF                                |
| 22 |        20 | 37.8793  | 6.15461 | 4.21153  |  7.54086 | linear_ridge+LLM-COT-RF                                |
| 23 |        30 | 67.7587  | 8.23157 | 6.14632  | 11.4616  | linear_ridge+LLM-COT-RF                                |
| 24 |         1 |  1.06154 | 1.03031 | 0.781283 |  1.38304 | linear_ridge+LLM-COT-RF_blend_ramp_bestval_path        |
| 25 |         5 |  4.64183 | 2.15449 | 1.57308  |  2.71491 | linear_ridge+LLM-COT-RF_blend_ramp_bestval_path        |
| 26 |        20 | 32.2685  | 5.68054 | 3.59313  |  6.49116 | linear_ridge+LLM-COT-RF_blend_ramp_bestval_path        |
| 27 |        30 | 63.679   | 7.97991 | 5.73294  | 10.7614  | linear_ridge+LLM-COT-RF_blend_ramp_bestval_path        |
| 28 |         1 |  1.06154 | 1.03031 | 0.781283 |  1.38304 | linear_ridge+LLM-COT-RF_blend_ramp_bestval_path_h1base |
| 29 |         5 |  4.64183 | 2.15449 | 1.57308  |  2.71491 | linear_ridge+LLM-COT-RF_blend_ramp_bestval_path_h1base |
| 30 |        20 | 32.2685  | 5.68054 | 3.59313  |  6.49116 | linear_ridge+LLM-COT-RF_blend_ramp_bestval_path_h1base |
| 31 |        30 | 63.679   | 7.97991 | 5.73294  | 10.7614  | linear_ridge+LLM-COT-RF_blend_ramp_bestval_path_h1base |
| 32 |         1 |  1.06154 | 1.03031 | 0.781283 |  1.38304 | linear_ridge+LLM-COT-RF_blend_ramp_bestval_h1          |
| 33 |         5 |  4.59363 | 2.14328 | 1.56632  |  2.70398 | linear_ridge+LLM-COT-RF_blend_ramp_bestval_h1          |
| 34 |        20 | 30.1544  | 5.4913  | 3.32793  |  6.07247 | linear_ridge+LLM-COT-RF_blend_ramp_bestval_h1          |
| 35 |        30 | 56.9094  | 7.54383 | 4.78768  |  9.18917 | linear_ridge+LLM-COT-RF_blend_ramp_bestval_h1          |
| 36 |         1 |  1.06154 | 1.03031 | 0.781283 |  1.38304 | linear_ridge+LLM-COT-RF_blend_ramp_bestval_h5          |
| 37 |         5 |  4.65902 | 2.15848 | 1.57536  |  2.71861 | linear_ridge+LLM-COT-RF_blend_ramp_bestval_h5          |
| 38 |        20 | 33.6953  | 5.80477 | 3.76304  |  6.77605 | linear_ridge+LLM-COT-RF_blend_ramp_bestval_h5          |
| 39 |        30 | 67.7587  | 8.23157 | 6.14632  | 11.4616  | linear_ridge+LLM-COT-RF_blend_ramp_bestval_h5          |
| 40 |         1 |  1.06154 | 1.03031 | 0.781283 |  1.38304 | linear_ridge+LLM-COT-RF_blend_ramp_bestval_h20         |
| 41 |         5 |  4.65902 | 2.15848 | 1.57536  |  2.71861 | linear_ridge+LLM-COT-RF_blend_ramp_bestval_h20         |
| 42 |        20 | 33.6953  | 5.80477 | 3.76304  |  6.77605 | linear_ridge+LLM-COT-RF_blend_ramp_bestval_h20         |
| 43 |        30 | 67.7587  | 8.23157 | 6.14632  | 11.4616  | linear_ridge+LLM-COT-RF_blend_ramp_bestval_h20         |
| 44 |         1 |  1.06154 | 1.03031 | 0.781283 |  1.38304 | linear_ridge+LLM-COT-RF_blend_ramp_bestval_h30         |
| 45 |         5 |  4.60913 | 2.14689 | 1.56855  |  2.70759 | linear_ridge+LLM-COT-RF_blend_ramp_bestval_h30         |
| 46 |        20 | 30.4981  | 5.52251 | 3.35334  |  6.09895 | linear_ridge+LLM-COT-RF_blend_ramp_bestval_h30         |
| 47 |        30 | 58.2544  | 7.63245 | 5.00644  |  9.54155 | linear_ridge+LLM-COT-RF_blend_ramp_bestval_h30         |
| 48 |         1 |  1.06139 | 1.03024 | 0.784861 |  1.39543 | naive_persistence_llm_subset                           |
| 49 |         5 |  5.2119  | 2.28296 | 1.67     |  2.90235 | naive_persistence_llm_subset                           |
| 50 |        20 | 44.2759  | 6.65401 | 4.40799  |  7.94956 | naive_persistence_llm_subset                           |
| 51 |        30 | 79.9881  | 8.94361 | 6.37986  | 12.0727  | naive_persistence_llm_subset                           |
| 52 |         1 |  5.16853 | 2.27344 | 1.65132  |  2.90012 | seasonal_naive_llm_subset                              |
| 53 |         5 |  5.2119  | 2.28296 | 1.67     |  2.90235 | seasonal_naive_llm_subset                              |
| 54 |        20 | 44.2759  | 6.65401 | 4.40799  |  7.94956 | seasonal_naive_llm_subset                              |
| 55 |        30 | 79.9881  | 8.94361 | 6.37986  | 12.0727  | seasonal_naive_llm_subset                              |
| 56 |         1 |  1.06154 | 1.03031 | 0.781283 |  1.38304 | linear_ridge_llm_subset                                |
| 57 |         5 |  4.59363 | 2.14328 | 1.56632  |  2.70398 | linear_ridge_llm_subset                                |
| 58 |        20 | 30.1544  | 5.4913  | 3.32793  |  6.07247 | linear_ridge_llm_subset                                |
| 59 |        30 | 56.9094  | 7.54383 | 4.78768  |  9.18917 | linear_ridge_llm_subset                                |
| 60 |         1 |  1.1541  | 1.07429 | 0.830618 |  1.48473 | linear_lasso_llm_subset                                |
| 61 |         5 |  4.73942 | 2.17702 | 1.61188  |  2.79853 | linear_lasso_llm_subset                                |
| 62 |        20 | 32.1131  | 5.66684 | 3.43547  |  6.16613 | linear_lasso_llm_subset                                |
| 63 |        30 | 62.0974  | 7.88019 | 5.23504  |  9.89649 | linear_lasso_llm_subset                                |
| 64 |         1 |  1.20574 | 1.09806 | 0.855881 |  1.51781 | tsm_llm_subset                                         |
| 65 |         5 |  5.64911 | 2.37679 | 1.80329  |  3.12164 | tsm_llm_subset                                         |
| 66 |        20 | 40.2462  | 6.34399 | 4.28359  |  7.6831  | tsm_llm_subset                                         |
| 67 |        30 | 72.369   | 8.507   | 6.23287  | 11.696   | tsm_llm_subset                                         |

### 4.2 Trend Classification Accuracy

Table 2 shows the accuracy of predicting price direction (up/flat/down).

|    |   horizon |   accuracy |   accuracy_up |   accuracy_down |   accuracy_flat |   n_up |   n_down |   n_flat | model                                                  |
|---:|----------:|-----------:|--------------:|----------------:|----------------:|-------:|---------:|---------:|:-------------------------------------------------------|
|  0 |         1 |  0.395833  |     0         |       0         |       1         |     52 |       35 |       57 | naive_persistence                                      |
|  1 |         5 |  0.236111  |     0         |       0         |       1         |     70 |       40 |       34 | naive_persistence                                      |
|  2 |        20 |  0.118056  |     0         |       0         |       1         |     90 |       37 |       17 | naive_persistence                                      |
|  3 |        30 |  0.0763889 |     0         |       0         |       1         |    100 |       33 |       11 | naive_persistence                                      |
|  4 |         1 |  0.375     |     0.365385  |       0.628571  |       0.22807   |     52 |       35 |       57 | seasonal_naive                                         |
|  5 |         5 |  0.236111  |     0         |       0         |       1         |     70 |       40 |       34 | seasonal_naive                                         |
|  6 |        20 |  0.118056  |     0         |       0         |       1         |     90 |       37 |       17 | seasonal_naive                                         |
|  7 |        30 |  0.0763889 |     0         |       0         |       1         |    100 |       33 |       11 | seasonal_naive                                         |
|  8 |         1 |  0.430556  |     0.0576923 |       0.0857143 |       0.982456  |     52 |       35 |       57 | linear_ridge                                           |
|  9 |         5 |  0.451389  |     0.514286  |       0.125     |       0.705882  |     70 |       40 |       34 | linear_ridge                                           |
| 10 |        20 |  0.708333  |     0.955556  |       0.351351  |       0.176471  |     90 |       37 |       17 | linear_ridge                                           |
| 11 |        30 |  0.833333  |     0.99      |       0.606061  |       0.0909091 |    100 |       33 |       11 | linear_ridge                                           |
| 12 |         1 |  0.423611  |     0.288462  |       0.114286  |       0.736842  |     52 |       35 |       57 | linear_lasso                                           |
| 13 |         5 |  0.388889  |     0.414286  |       0.15      |       0.617647  |     70 |       40 |       34 | linear_lasso                                           |
| 14 |        20 |  0.708333  |     0.922222  |       0.459459  |       0.117647  |     90 |       37 |       17 | linear_lasso                                           |
| 15 |        30 |  0.736111  |     0.75      |       0.757576  |       0.545455  |    100 |       33 |       11 | linear_lasso                                           |
| 16 |         1 |  0.381944  |     0.0192308 |       0.114286  |       0.877193  |     52 |       35 |       57 | tsm                                                    |
| 17 |         5 |  0.256944  |     0.0714286 |       0.425     |       0.441176  |     70 |       40 |       34 | tsm                                                    |
| 18 |        20 |  0.388889  |     0.288889  |       0.702703  |       0.235294  |     90 |       37 |       17 | tsm                                                    |
| 19 |        30 |  0.361111  |     0.19      |       0.909091  |       0.272727  |    100 |       33 |       11 | tsm                                                    |
| 20 |         1 |  0.416667  |     0         |       0.114286  |       0.982456  |     52 |       35 |       57 | linear_ridge+LLM-COT-RF                                |
| 21 |         5 |  0.3125    |     0.128571  |       0.225     |       0.794118  |     70 |       40 |       34 | linear_ridge+LLM-COT-RF                                |
| 22 |        20 |  0.486111  |     0.444444  |       0.756757  |       0.117647  |     90 |       37 |       17 | linear_ridge+LLM-COT-RF                                |
| 23 |        30 |  0.527778  |     0.45      |       0.878788  |       0.181818  |    100 |       33 |       11 | linear_ridge+LLM-COT-RF                                |
| 24 |         1 |  0.430556  |     0.0576923 |       0.0857143 |       0.982456  |     52 |       35 |       57 | linear_ridge+LLM-COT-RF_blend_ramp_bestval_path        |
| 25 |         5 |  0.4375    |     0.457143  |       0.125     |       0.764706  |     70 |       40 |       34 | linear_ridge+LLM-COT-RF_blend_ramp_bestval_path        |
| 26 |        20 |  0.673611  |     0.833333  |       0.486486  |       0.235294  |     90 |       37 |       17 | linear_ridge+LLM-COT-RF_blend_ramp_bestval_path        |
| 27 |        30 |  0.569444  |     0.52      |       0.848485  |       0.181818  |    100 |       33 |       11 | linear_ridge+LLM-COT-RF_blend_ramp_bestval_path        |
| 28 |         1 |  0.430556  |     0.0576923 |       0.0857143 |       0.982456  |     52 |       35 |       57 | linear_ridge+LLM-COT-RF_blend_ramp_bestval_path_h1base |
| 29 |         5 |  0.4375    |     0.457143  |       0.125     |       0.764706  |     70 |       40 |       34 | linear_ridge+LLM-COT-RF_blend_ramp_bestval_path_h1base |
| 30 |        20 |  0.673611  |     0.833333  |       0.486486  |       0.235294  |     90 |       37 |       17 | linear_ridge+LLM-COT-RF_blend_ramp_bestval_path_h1base |
| 31 |        30 |  0.569444  |     0.52      |       0.848485  |       0.181818  |    100 |       33 |       11 | linear_ridge+LLM-COT-RF_blend_ramp_bestval_path_h1base |
| 32 |         1 |  0.430556  |     0.0576923 |       0.0857143 |       0.982456  |     52 |       35 |       57 | linear_ridge+LLM-COT-RF_blend_ramp_bestval_h1          |
| 33 |         5 |  0.451389  |     0.514286  |       0.125     |       0.705882  |     70 |       40 |       34 | linear_ridge+LLM-COT-RF_blend_ramp_bestval_h1          |
| 34 |        20 |  0.708333  |     0.955556  |       0.351351  |       0.176471  |     90 |       37 |       17 | linear_ridge+LLM-COT-RF_blend_ramp_bestval_h1          |
| 35 |        30 |  0.833333  |     0.99      |       0.606061  |       0.0909091 |    100 |       33 |       11 | linear_ridge+LLM-COT-RF_blend_ramp_bestval_h1          |
| 36 |         1 |  0.430556  |     0.0576923 |       0.0857143 |       0.982456  |     52 |       35 |       57 | linear_ridge+LLM-COT-RF_blend_ramp_bestval_h5          |
| 37 |         5 |  0.416667  |     0.414286  |       0.125     |       0.764706  |     70 |       40 |       34 | linear_ridge+LLM-COT-RF_blend_ramp_bestval_h5          |
| 38 |        20 |  0.576389  |     0.611111  |       0.567568  |       0.411765  |     90 |       37 |       17 | linear_ridge+LLM-COT-RF_blend_ramp_bestval_h5          |
| 39 |        30 |  0.527778  |     0.45      |       0.878788  |       0.181818  |    100 |       33 |       11 | linear_ridge+LLM-COT-RF_blend_ramp_bestval_h5          |
| 40 |         1 |  0.430556  |     0.0576923 |       0.0857143 |       0.982456  |     52 |       35 |       57 | linear_ridge+LLM-COT-RF_blend_ramp_bestval_h20         |
| 41 |         5 |  0.416667  |     0.414286  |       0.125     |       0.764706  |     70 |       40 |       34 | linear_ridge+LLM-COT-RF_blend_ramp_bestval_h20         |
| 42 |        20 |  0.576389  |     0.611111  |       0.567568  |       0.411765  |     90 |       37 |       17 | linear_ridge+LLM-COT-RF_blend_ramp_bestval_h20         |
| 43 |        30 |  0.527778  |     0.45      |       0.878788  |       0.181818  |    100 |       33 |       11 | linear_ridge+LLM-COT-RF_blend_ramp_bestval_h20         |
| 44 |         1 |  0.430556  |     0.0576923 |       0.0857143 |       0.982456  |     52 |       35 |       57 | linear_ridge+LLM-COT-RF_blend_ramp_bestval_h30         |
| 45 |         5 |  0.4375    |     0.485714  |       0.125     |       0.705882  |     70 |       40 |       34 | linear_ridge+LLM-COT-RF_blend_ramp_bestval_h30         |
| 46 |        20 |  0.715278  |     0.944444  |       0.405405  |       0.176471  |     90 |       37 |       17 | linear_ridge+LLM-COT-RF_blend_ramp_bestval_h30         |
| 47 |        30 |  0.826389  |     0.95      |       0.666667  |       0.181818  |    100 |       33 |       11 | linear_ridge+LLM-COT-RF_blend_ramp_bestval_h30         |
| 48 |         1 |  0.395833  |     0         |       0         |       1         |     52 |       35 |       57 | naive_persistence_llm_subset                           |
| 49 |         5 |  0.236111  |     0         |       0         |       1         |     70 |       40 |       34 | naive_persistence_llm_subset                           |
| 50 |        20 |  0.118056  |     0         |       0         |       1         |     90 |       37 |       17 | naive_persistence_llm_subset                           |
| 51 |        30 |  0.0763889 |     0         |       0         |       1         |    100 |       33 |       11 | naive_persistence_llm_subset                           |
| 52 |         1 |  0.375     |     0.365385  |       0.628571  |       0.22807   |     52 |       35 |       57 | seasonal_naive_llm_subset                              |
| 53 |         5 |  0.236111  |     0         |       0         |       1         |     70 |       40 |       34 | seasonal_naive_llm_subset                              |
| 54 |        20 |  0.118056  |     0         |       0         |       1         |     90 |       37 |       17 | seasonal_naive_llm_subset                              |
| 55 |        30 |  0.0763889 |     0         |       0         |       1         |    100 |       33 |       11 | seasonal_naive_llm_subset                              |
| 56 |         1 |  0.430556  |     0.0576923 |       0.0857143 |       0.982456  |     52 |       35 |       57 | linear_ridge_llm_subset                                |
| 57 |         5 |  0.451389  |     0.514286  |       0.125     |       0.705882  |     70 |       40 |       34 | linear_ridge_llm_subset                                |
| 58 |        20 |  0.708333  |     0.955556  |       0.351351  |       0.176471  |     90 |       37 |       17 | linear_ridge_llm_subset                                |
| 59 |        30 |  0.833333  |     0.99      |       0.606061  |       0.0909091 |    100 |       33 |       11 | linear_ridge_llm_subset                                |
| 60 |         1 |  0.423611  |     0.288462  |       0.114286  |       0.736842  |     52 |       35 |       57 | linear_lasso_llm_subset                                |
| 61 |         5 |  0.388889  |     0.414286  |       0.15      |       0.617647  |     70 |       40 |       34 | linear_lasso_llm_subset                                |
| 62 |        20 |  0.708333  |     0.922222  |       0.459459  |       0.117647  |     90 |       37 |       17 | linear_lasso_llm_subset                                |
| 63 |        30 |  0.736111  |     0.75      |       0.757576  |       0.545455  |    100 |       33 |       11 | linear_lasso_llm_subset                                |
| 64 |         1 |  0.381944  |     0.0192308 |       0.114286  |       0.877193  |     52 |       35 |       57 | tsm_llm_subset                                         |
| 65 |         5 |  0.256944  |     0.0714286 |       0.425     |       0.441176  |     70 |       40 |       34 | tsm_llm_subset                                         |
| 66 |        20 |  0.388889  |     0.288889  |       0.702703  |       0.235294  |     90 |       37 |       17 | tsm_llm_subset                                         |
| 67 |        30 |  0.361111  |     0.19      |       0.909091  |       0.272727  |    100 |       33 |       11 | tsm_llm_subset                                         |

### 4.3 Statistical Significance

Table 3 reports the results of significance tests comparing methods.

|    |   horizon | model                                                  | baseline                     |   mean_error_model |   mean_error_baseline |   improvement_pct |   t_statistic |    t_pvalue | t_significant   |   wilcoxon_statistic |   wilcoxon_pvalue | wilcoxon_significant   |   dm_statistic |   dm_pvalue | dm_significant   | model_better   |
|---:|----------:|:-------------------------------------------------------|:-----------------------------|-------------------:|----------------------:|------------------:|--------------:|------------:|:----------------|---------------------:|------------------:|:-----------------------|---------------:|------------:|:-----------------|:---------------|
|  0 |         1 | seasonal_naive_llm_subset                              | naive_persistence_llm_subset |            5.16853 |               1.06139 |    -386.957       |    5.07044    | 1.21001e-06 | True            |                 1834 |       1.45104e-11 | True                   |     5.07044    | 3.96893e-07 | True             | False          |
|  1 |         5 | seasonal_naive_llm_subset                              | naive_persistence_llm_subset |            5.2119  |               5.2119  |       5.49269e-07 |    1.19107    | 0.235598    | False           |                 4258 |       0.556904    | False                  |     0.0263232  | 0.979       | False            | True           |
|  2 |        20 | seasonal_naive_llm_subset                              | naive_persistence_llm_subset |           44.2759  |              44.2759  |       2.22491e-06 |    0.67977    | 0.497749    | False           |                 4233 |       0.183904    | False                  |     0.216494   | 0.828603    | False            | True           |
|  3 |        30 | seasonal_naive_llm_subset                              | naive_persistence_llm_subset |           79.9881  |              79.9881  |       3.104e-06   |   -0.578827   | 0.563616    | False           |                 3978 |       0.082287    | False                  |    -0.245123   | 0.806361    | False            | True           |
|  4 |         1 | linear_ridge_llm_subset                                | naive_persistence_llm_subset |            1.06154 |               1.06139 |      -0.0135551   |    0.00242328 | 0.99807     | False           |                 5155 |       0.896859    | False                  |     0.00242328 | 0.998067    | False            | False          |
|  5 |         5 | linear_ridge_llm_subset                                | naive_persistence_llm_subset |            4.59363 |               5.2119  |      11.8626      |   -2.00294    | 0.0470741   | True            |                 3955 |       0.0116426   | True                   |    -1.15055    | 0.249918    | False            | True           |
|  6 |        20 | linear_ridge_llm_subset                                | naive_persistence_llm_subset |           30.1544  |              44.2759  |      31.8943      |   -5.67076    | 7.5947e-08  | True            |                 2425 |       2.4884e-08  | True                   |    -2.3465     | 0.0189507   | True             | True           |
|  7 |        30 | linear_ridge_llm_subset                                | naive_persistence_llm_subset |           56.9094  |              79.9881  |      28.8527      |   -7.86447    | 8.23678e-13 | True            |                  726 |       3.17675e-19 | True                   |    -1.91128    | 0.0559692   | False            | True           |
|  8 |         1 | linear_lasso_llm_subset                                | naive_persistence_llm_subset |            1.1541  |               1.06139 |      -8.73423     |    1.16037    | 0.24783     | False           |                 4553 |       0.183452    | False                  |     1.16037    | 0.245896    | False            | False          |
|  9 |         5 | linear_lasso_llm_subset                                | naive_persistence_llm_subset |            4.73942 |               5.2119  |       9.06537     |   -1.85969    | 0.0649833   | False           |                 4418 |       0.109725    | False                  |    -1.14841    | 0.250799    | False            | True           |
| 10 |        20 | linear_lasso_llm_subset                                | naive_persistence_llm_subset |           32.1131  |              44.2759  |      27.4704      |   -5.48329    | 1.83923e-07 | True            |                 1499 |       1.16409e-13 | True                   |    -2.64988    | 0.00805199  | True             | True           |
| 11 |        30 | linear_lasso_llm_subset                                | naive_persistence_llm_subset |           62.0974  |              79.9881  |      22.3667      |   -6.53209    | 1.0595e-09  | True            |                  443 |       1.62218e-21 | True                   |    -1.64973    | 0.0989982   | False            | True           |
| 12 |         1 | tsm_llm_subset                                         | naive_persistence_llm_subset |            1.20574 |               1.06139 |     -13.5994      |    2.1918     | 0.0300118   | True            |                 3956 |       0.0117088   | True                   |     2.1918     | 0.0283939   | True             | False          |
| 13 |         5 | tsm_llm_subset                                         | naive_persistence_llm_subset |            5.64911 |               5.2119  |      -8.38883     |    1.54257    | 0.125144    | False           |                 4440 |       0.119813    | False                  |     0.774825   | 0.438443    | False            | False          |
| 14 |        20 | tsm_llm_subset                                         | naive_persistence_llm_subset |           40.2462  |              44.2759  |       9.10133     |   -2.80664    | 0.0057048   | True            |                 4600 |       0.216284    | False                  |    -1.24077    | 0.214692    | False            | True           |
| 15 |        30 | tsm_llm_subset                                         | naive_persistence_llm_subset |           72.369   |              79.9881  |       9.52529     |   -3.08456    | 0.00244758  | True            |                 4970 |       0.618078    | False                  |    -0.821835   | 0.411171    | False            | True           |
| 16 |         1 | linear_ridge+LLM-COT-RF                                | naive_persistence_llm_subset |            1.13673 |               1.06139 |      -7.09818     |    1.52544    | 0.129359    | False           |                 4592 |       0.210416    | False                  |     1.52544    | 0.12715     | False            | False          |
| 17 |         5 | linear_ridge+LLM-COT-RF                                | naive_persistence_llm_subset |            5.27228 |               5.2119  |      -1.15864     |    0.214359   | 0.830573    | False           |                 5120 |       0.841926    | False                  |     0.142285   | 0.886855    | False            | False          |
| 18 |        20 | linear_ridge+LLM-COT-RF                                | naive_persistence_llm_subset |           37.8793  |              44.2759  |      14.4471      |   -2.77225    | 0.00630861  | True            |                 4645 |       0.251495    | False                  |    -1.04654    | 0.295312    | False            | True           |
| 19 |        30 | linear_ridge+LLM-COT-RF                                | naive_persistence_llm_subset |           67.7587  |              79.9881  |      15.289       |   -3.22482    | 0.00156209  | True            |                 4496 |       0.148774    | False                  |    -0.806075   | 0.420199    | False            | True           |
| 20 |         1 | linear_ridge+LLM-COT-RF_blend_ramp_bestval_path        | naive_persistence_llm_subset |            1.06154 |               1.06139 |      -0.0135551   |    0.00242328 | 0.99807     | False           |                 5155 |       0.896859    | False                  |     0.00242328 | 0.998067    | False            | False          |
| 21 |         5 | linear_ridge+LLM-COT-RF_blend_ramp_bestval_path        | naive_persistence_llm_subset |            4.64183 |               5.2119  |      10.9379      |   -1.89425    | 0.0602117   | False           |                 3950 |       0.0113166   | True                   |    -1.09993    | 0.271363    | False            | True           |
| 22 |        20 | linear_ridge+LLM-COT-RF_blend_ramp_bestval_path        | naive_persistence_llm_subset |           32.2685  |              44.2759  |      27.1193      |   -5.69917    | 6.63222e-08 | True            |                 1786 |       7.46589e-12 | True                   |    -2.30288    | 0.0212856   | True             | True           |
| 23 |        30 | linear_ridge+LLM-COT-RF_blend_ramp_bestval_path        | naive_persistence_llm_subset |           63.679   |              79.9881  |      20.3894      |   -4.86712    | 2.95723e-06 | True            |                 3328 |       0.000161151 | True                   |    -1.16084    | 0.245706    | False            | True           |
| 24 |         1 | linear_ridge+LLM-COT-RF_blend_ramp_bestval_path_h1base | naive_persistence_llm_subset |            1.06154 |               1.06139 |      -0.0135551   |    0.00242328 | 0.99807     | False           |                 5155 |       0.896859    | False                  |     0.00242328 | 0.998067    | False            | False          |
| 25 |         5 | linear_ridge+LLM-COT-RF_blend_ramp_bestval_path_h1base | naive_persistence_llm_subset |            4.64183 |               5.2119  |      10.9379      |   -1.89425    | 0.0602117   | False           |                 3950 |       0.0113166   | True                   |    -1.09993    | 0.271363    | False            | True           |
| 26 |        20 | linear_ridge+LLM-COT-RF_blend_ramp_bestval_path_h1base | naive_persistence_llm_subset |           32.2685  |              44.2759  |      27.1193      |   -5.69917    | 6.63222e-08 | True            |                 1786 |       7.46589e-12 | True                   |    -2.30288    | 0.0212856   | True             | True           |
| 27 |        30 | linear_ridge+LLM-COT-RF_blend_ramp_bestval_path_h1base | naive_persistence_llm_subset |           63.679   |              79.9881  |      20.3894      |   -4.86712    | 2.95723e-06 | True            |                 3328 |       0.000161151 | True                   |    -1.16084    | 0.245706    | False            | True           |
| 28 |         1 | linear_ridge+LLM-COT-RF_blend_ramp_bestval_h1          | naive_persistence_llm_subset |            1.06154 |               1.06139 |      -0.0135551   |    0.00242328 | 0.99807     | False           |                 5155 |       0.896859    | False                  |     0.00242328 | 0.998067    | False            | False          |
| 29 |         5 | linear_ridge+LLM-COT-RF_blend_ramp_bestval_h1          | naive_persistence_llm_subset |            4.59363 |               5.2119  |      11.8626      |   -2.00294    | 0.0470741   | True            |                 3955 |       0.0116426   | True                   |    -1.15055    | 0.249918    | False            | True           |
| 30 |        20 | linear_ridge+LLM-COT-RF_blend_ramp_bestval_h1          | naive_persistence_llm_subset |           30.1544  |              44.2759  |      31.8943      |   -5.67076    | 7.5947e-08  | True            |                 2425 |       2.4884e-08  | True                   |    -2.3465     | 0.0189507   | True             | True           |
| 31 |        30 | linear_ridge+LLM-COT-RF_blend_ramp_bestval_h1          | naive_persistence_llm_subset |           56.9094  |              79.9881  |      28.8527      |   -7.86447    | 8.23678e-13 | True            |                  726 |       3.17675e-19 | True                   |    -1.91128    | 0.0559692   | False            | True           |
| 32 |         1 | linear_ridge+LLM-COT-RF_blend_ramp_bestval_h5          | naive_persistence_llm_subset |            1.06154 |               1.06139 |      -0.0135551   |    0.00242328 | 0.99807     | False           |                 5155 |       0.896859    | False                  |     0.00242328 | 0.998067    | False            | False          |
| 33 |         5 | linear_ridge+LLM-COT-RF_blend_ramp_bestval_h5          | naive_persistence_llm_subset |            4.65902 |               5.2119  |      10.608       |   -1.85176    | 0.0661225   | False           |                 3943 |       0.0108739   | True                   |    -1.07947    | 0.280377    | False            | True           |
| 34 |        20 | linear_ridge+LLM-COT-RF_blend_ramp_bestval_h5          | naive_persistence_llm_subset |           33.6953  |              44.2759  |      23.8968      |   -5.05516    | 1.29513e-06 | True            |                 2343 |       9.60197e-09 | True                   |    -1.9905     | 0.0465354   | True             | True           |
| 35 |        30 | linear_ridge+LLM-COT-RF_blend_ramp_bestval_h5          | naive_persistence_llm_subset |           67.7587  |              79.9881  |      15.289       |   -3.22482    | 0.00156209  | True            |                 4496 |       0.148774    | False                  |    -0.806075   | 0.420199    | False            | True           |
| 36 |         1 | linear_ridge+LLM-COT-RF_blend_ramp_bestval_h20         | naive_persistence_llm_subset |            1.06154 |               1.06139 |      -0.0135551   |    0.00242328 | 0.99807     | False           |                 5155 |       0.896859    | False                  |     0.00242328 | 0.998067    | False            | False          |
| 37 |         5 | linear_ridge+LLM-COT-RF_blend_ramp_bestval_h20         | naive_persistence_llm_subset |            4.65902 |               5.2119  |      10.608       |   -1.85176    | 0.0661225   | False           |                 3943 |       0.0108739   | True                   |    -1.07947    | 0.280377    | False            | True           |
| 38 |        20 | linear_ridge+LLM-COT-RF_blend_ramp_bestval_h20         | naive_persistence_llm_subset |           33.6953  |              44.2759  |      23.8968      |   -5.05516    | 1.29513e-06 | True            |                 2343 |       9.60197e-09 | True                   |    -1.9905     | 0.0465354   | True             | True           |
| 39 |        30 | linear_ridge+LLM-COT-RF_blend_ramp_bestval_h20         | naive_persistence_llm_subset |           67.7587  |              79.9881  |      15.289       |   -3.22482    | 0.00156209  | True            |                 4496 |       0.148774    | False                  |    -0.806075   | 0.420199    | False            | True           |
| 40 |         1 | linear_ridge+LLM-COT-RF_blend_ramp_bestval_h30         | naive_persistence_llm_subset |            1.06154 |               1.06139 |      -0.0135551   |    0.00242328 | 0.99807     | False           |                 5155 |       0.896859    | False                  |     0.00242328 | 0.998067    | False            | False          |
| 41 |         5 | linear_ridge+LLM-COT-RF_blend_ramp_bestval_h30         | naive_persistence_llm_subset |            4.60913 |               5.2119  |      11.5652      |   -1.96977    | 0.0507968   | False           |                 3947 |       0.0111249   | True                   |    -1.1354     | 0.256207    | False            | True           |
| 42 |        20 | linear_ridge+LLM-COT-RF_blend_ramp_bestval_h30         | naive_persistence_llm_subset |           30.4981  |              44.2759  |      31.1181      |   -5.94642    | 2.0059e-08  | True            |                 2157 |       1.00538e-09 | True                   |    -2.46847    | 0.0135691   | True             | True           |
| 43 |        30 | linear_ridge+LLM-COT-RF_blend_ramp_bestval_h30         | naive_persistence_llm_subset |           58.2544  |              79.9881  |      27.1712      |   -7.46527    | 7.46216e-12 | True            |                  385 |       5.29037e-22 | True                   |    -1.73949    | 0.0819488   | False            | True           |


## 5. Robustness Analysis

### 5.1 Noise Injection Test

To assess robustness, we inject noise into forecast paths at levels of 5%, 10%, 20%, 
and 30% of the forecast standard deviation.

|     |   horizon |      mse |    rmse |      mae |     mape |   noise_level | model                                                  |
|----:|----------:|---------:|--------:|---------:|---------:|--------------:|:-------------------------------------------------------|
|   0 |         1 |  1.20222 | 1.09646 | 0.854421 |  1.51606 |          0.05 | tsm                                                    |
|   1 |         5 |  5.65917 | 2.3789  | 1.8026   |  3.12079 |          0.05 | tsm                                                    |
|   2 |        20 | 40.2456  | 6.34394 | 4.28232  |  7.6806  |          0.05 | tsm                                                    |
|   3 |        30 | 72.3504  | 8.5059  | 6.22953  | 11.69    |          0.05 | tsm                                                    |
|   4 |         1 |  1.19999 | 1.09544 | 0.853554 |  1.51538 |          0.1  | tsm                                                    |
|   5 |         5 |  5.67089 | 2.38136 | 1.80194  |  3.11996 |          0.1  | tsm                                                    |
|   6 |        20 | 40.2464  | 6.34401 | 4.28108  |  7.67814 |          0.1  | tsm                                                    |
|   7 |        30 | 72.3328  | 8.50487 | 6.2262   | 11.684   |          0.1  | tsm                                                    |
|   8 |         1 |  1.19937 | 1.09516 | 0.85291  |  1.51572 |          0.2  | tsm                                                    |
|   9 |         5 |  5.69935 | 2.38733 | 1.80217  |  3.12112 |          0.2  | tsm                                                    |
|  10 |        20 | 40.2522  | 6.34446 | 4.27922  |  7.67436 |          0.2  | tsm                                                    |
|  11 |        30 | 72.301   | 8.503   | 6.21953  | 11.6721  |          0.2  | tsm                                                    |
|  12 |         1 |  1.20386 | 1.0972  | 0.854502 |  1.51948 |          0.3  | tsm                                                    |
|  13 |         5 |  5.73449 | 2.39468 | 1.80254  |  3.12249 |          0.3  | tsm                                                    |
|  14 |        20 | 40.2634  | 6.34534 | 4.27937  |  7.67435 |          0.3  | tsm                                                    |
|  15 |        30 | 72.2734  | 8.50138 | 6.21286  | 11.6602  |          0.3  | tsm                                                    |
|  16 |         1 |  1.1306  | 1.0633  | 0.810873 |  1.43465 |          0.05 | linear_ridge+LLM-COT-RF                                |
|  17 |         5 |  5.28106 | 2.29806 | 1.671    |  2.88419 |          0.05 | linear_ridge+LLM-COT-RF                                |
|  18 |        20 | 37.8897  | 6.15546 | 4.21344  |  7.54443 |          0.05 | linear_ridge+LLM-COT-RF                                |
|  19 |        30 | 67.6843  | 8.22705 | 6.14001  | 11.4503  |          0.05 | linear_ridge+LLM-COT-RF                                |
|  20 |         1 |  1.12689 | 1.06155 | 0.811156 |  1.43596 |          0.1  | linear_ridge+LLM-COT-RF                                |
|  21 |         5 |  5.29262 | 2.30057 | 1.67124  |  2.8845  |          0.1  | linear_ridge+LLM-COT-RF                                |
|  22 |        20 | 37.9021  | 6.15647 | 4.21534  |  7.54799 |          0.1  | linear_ridge+LLM-COT-RF                                |
|  23 |        30 | 67.612   | 8.22265 | 6.13369  | 11.439   |          0.1  | linear_ridge+LLM-COT-RF                                |
|  24 |         1 |  1.12671 | 1.06146 | 0.814931 |  1.44378 |          0.2  | linear_ridge+LLM-COT-RF                                |
|  25 |         5 |  5.32404 | 2.30739 | 1.67413  |  2.88936 |          0.2  | linear_ridge+LLM-COT-RF                                |
|  26 |        20 | 37.9329  | 6.15897 | 4.21969  |  7.55599 |          0.2  | linear_ridge+LLM-COT-RF                                |
|  27 |        30 | 67.4736  | 8.21423 | 6.12436  | 11.4228  |          0.2  | linear_ridge+LLM-COT-RF                                |
|  28 |         1 |  1.13619 | 1.06592 | 0.822046 |  1.45763 |          0.3  | linear_ridge+LLM-COT-RF                                |
|  29 |         5 |  5.36657 | 2.31658 | 1.67836  |  2.89661 |          0.3  | linear_ridge+LLM-COT-RF                                |
|  30 |        20 | 37.9717  | 6.16212 | 4.22523  |  7.56594 |          0.3  | linear_ridge+LLM-COT-RF                                |
|  31 |        30 | 67.3436  | 8.20631 | 6.1172   | 11.4105  |          0.3  | linear_ridge+LLM-COT-RF                                |
|  32 |         1 |  1.0553  | 1.02728 | 0.780056 |  1.38158 |          0.05 | linear_ridge+LLM-COT-RF_blend_ramp_bestval_path        |
|  33 |         5 |  4.65124 | 2.15667 | 1.576    |  2.72015 |          0.05 | linear_ridge+LLM-COT-RF_blend_ramp_bestval_path        |
|  34 |        20 | 32.2781  | 5.68138 | 3.59532  |  6.49525 |          0.05 | linear_ridge+LLM-COT-RF_blend_ramp_bestval_path        |
|  35 |        30 | 63.622   | 7.97634 | 5.72715  | 10.7509  |          0.05 | linear_ridge+LLM-COT-RF_blend_ramp_bestval_path        |
|  36 |         1 |  1.05118 | 1.02527 | 0.779752 |  1.38165 |          0.1  | linear_ridge+LLM-COT-RF_blend_ramp_bestval_path        |
|  37 |         5 |  4.66339 | 2.15949 | 1.57898  |  2.72547 |          0.1  | linear_ridge+LLM-COT-RF_blend_ramp_bestval_path        |
|  38 |        20 | 32.2896  | 5.6824  | 3.59763  |  6.49955 |          0.1  | linear_ridge+LLM-COT-RF_blend_ramp_bestval_path        |
|  39 |        30 | 63.5669  | 7.97289 | 5.72135  | 10.7405  |          0.1  | linear_ridge+LLM-COT-RF_blend_ramp_bestval_path        |
|  40 |         1 |  1.04928 | 1.02434 | 0.782584 |  1.3875  |          0.2  | linear_ridge+LLM-COT-RF_blend_ramp_bestval_path        |
|  41 |         5 |  4.69591 | 2.16701 | 1.58593  |  2.73789 |          0.2  | linear_ridge+LLM-COT-RF_blend_ramp_bestval_path        |
|  42 |        20 | 32.3186  | 5.68494 | 3.60277  |  6.50902 |          0.2  | linear_ridge+LLM-COT-RF_blend_ramp_bestval_path        |
|  43 |        30 | 63.4628  | 7.96636 | 5.71169  | 10.7234  |          0.2  | linear_ridge+LLM-COT-RF_blend_ramp_bestval_path        |
|  44 |         1 |  1.05585 | 1.02755 | 0.788597 |  1.39922 |          0.3  | linear_ridge+LLM-COT-RF_blend_ramp_bestval_path        |
|  45 |         5 |  4.73939 | 2.17702 | 1.59347  |  2.75151 |          0.3  | linear_ridge+LLM-COT-RF_blend_ramp_bestval_path        |
|  46 |        20 | 32.3554  | 5.68818 | 3.60915  |  6.52051 |          0.3  | linear_ridge+LLM-COT-RF_blend_ramp_bestval_path        |
|  47 |        30 | 63.3668  | 7.96032 | 5.70435  | 10.7106  |          0.3  | linear_ridge+LLM-COT-RF_blend_ramp_bestval_path        |
|  48 |         1 |  1.0553  | 1.02728 | 0.780056 |  1.38158 |          0.05 | linear_ridge+LLM-COT-RF_blend_ramp_bestval_path_h1base |
|  49 |         5 |  4.65124 | 2.15667 | 1.576    |  2.72015 |          0.05 | linear_ridge+LLM-COT-RF_blend_ramp_bestval_path_h1base |
|  50 |        20 | 32.2781  | 5.68138 | 3.59532  |  6.49525 |          0.05 | linear_ridge+LLM-COT-RF_blend_ramp_bestval_path_h1base |
|  51 |        30 | 63.622   | 7.97634 | 5.72715  | 10.7509  |          0.05 | linear_ridge+LLM-COT-RF_blend_ramp_bestval_path_h1base |
|  52 |         1 |  1.05118 | 1.02527 | 0.779752 |  1.38165 |          0.1  | linear_ridge+LLM-COT-RF_blend_ramp_bestval_path_h1base |
|  53 |         5 |  4.66339 | 2.15949 | 1.57898  |  2.72547 |          0.1  | linear_ridge+LLM-COT-RF_blend_ramp_bestval_path_h1base |
|  54 |        20 | 32.2896  | 5.6824  | 3.59763  |  6.49955 |          0.1  | linear_ridge+LLM-COT-RF_blend_ramp_bestval_path_h1base |
|  55 |        30 | 63.5669  | 7.97289 | 5.72135  | 10.7405  |          0.1  | linear_ridge+LLM-COT-RF_blend_ramp_bestval_path_h1base |
|  56 |         1 |  1.04928 | 1.02434 | 0.782584 |  1.3875  |          0.2  | linear_ridge+LLM-COT-RF_blend_ramp_bestval_path_h1base |
|  57 |         5 |  4.69591 | 2.16701 | 1.58593  |  2.73789 |          0.2  | linear_ridge+LLM-COT-RF_blend_ramp_bestval_path_h1base |
|  58 |        20 | 32.3186  | 5.68494 | 3.60277  |  6.50902 |          0.2  | linear_ridge+LLM-COT-RF_blend_ramp_bestval_path_h1base |
|  59 |        30 | 63.4628  | 7.96636 | 5.71169  | 10.7234  |          0.2  | linear_ridge+LLM-COT-RF_blend_ramp_bestval_path_h1base |
|  60 |         1 |  1.05585 | 1.02755 | 0.788597 |  1.39922 |          0.3  | linear_ridge+LLM-COT-RF_blend_ramp_bestval_path_h1base |
|  61 |         5 |  4.73939 | 2.17702 | 1.59347  |  2.75151 |          0.3  | linear_ridge+LLM-COT-RF_blend_ramp_bestval_path_h1base |
|  62 |        20 | 32.3554  | 5.68818 | 3.60915  |  6.52051 |          0.3  | linear_ridge+LLM-COT-RF_blend_ramp_bestval_path_h1base |
|  63 |        30 | 63.3668  | 7.96032 | 5.70435  | 10.7106  |          0.3  | linear_ridge+LLM-COT-RF_blend_ramp_bestval_path_h1base |
|  64 |         1 |  1.05593 | 1.02758 | 0.78003  |  1.38133 |          0.05 | linear_ridge+LLM-COT-RF_blend_ramp_bestval_h1          |
|  65 |         5 |  4.60235 | 2.14531 | 1.57012  |  2.71084 |          0.05 | linear_ridge+LLM-COT-RF_blend_ramp_bestval_h1          |
|  66 |        20 | 30.1652  | 5.49229 | 3.3282   |  6.07279 |          0.05 | linear_ridge+LLM-COT-RF_blend_ramp_bestval_h1          |
|  67 |        30 | 56.8828  | 7.54207 | 4.78163  |  9.17831 |          0.05 | linear_ridge+LLM-COT-RF_blend_ramp_bestval_h1          |
|  68 |         1 |  1.05375 | 1.02653 | 0.779948 |  1.38163 |          0.1  | linear_ridge+LLM-COT-RF_blend_ramp_bestval_h1          |
|  69 |         5 |  4.61548 | 2.14837 | 1.57419  |  2.71819 |          0.1  | linear_ridge+LLM-COT-RF_blend_ramp_bestval_h1          |
|  70 |        20 | 30.1796  | 5.4936  | 3.32846  |  6.07311 |          0.1  | linear_ridge+LLM-COT-RF_blend_ramp_bestval_h1          |
|  71 |        30 | 56.86    | 7.54056 | 4.77579  |  9.16779 |          0.1  | linear_ridge+LLM-COT-RF_blend_ramp_bestval_h1          |
|  72 |         1 |  1.0597  | 1.02942 | 0.7878   |  1.39651 |          0.2  | linear_ridge+LLM-COT-RF_blend_ramp_bestval_h1          |
|  73 |         5 |  4.65498 | 2.15754 | 1.58274  |  2.7337  |          0.2  | linear_ridge+LLM-COT-RF_blend_ramp_bestval_h1          |
|  74 |        20 | 30.2191  | 5.49719 | 3.33163  |  6.07868 |          0.2  | linear_ridge+LLM-COT-RF_blend_ramp_bestval_h1          |
|  75 |        30 | 56.8261  | 7.53831 | 4.76928  |  9.15665 |          0.2  | linear_ridge+LLM-COT-RF_blend_ramp_bestval_h1          |
|  76 |         1 |  1.07936 | 1.03892 | 0.802095 |  1.42352 |          0.3  | linear_ridge+LLM-COT-RF_blend_ramp_bestval_h1          |
|  77 |         5 |  4.71212 | 2.17074 | 1.59315  |  2.75297 |          0.3  | linear_ridge+LLM-COT-RF_blend_ramp_bestval_h1          |
|  78 |        20 | 30.2728  | 5.50207 | 3.33841  |  6.0906  |          0.3  | linear_ridge+LLM-COT-RF_blend_ramp_bestval_h1          |
|  79 |        30 | 56.8076  | 7.53708 | 4.77361  |  9.1661  |          0.3  | linear_ridge+LLM-COT-RF_blend_ramp_bestval_h1          |
|  80 |         1 |  1.05443 | 1.02686 | 0.780602 |  1.38256 |          0.05 | linear_ridge+LLM-COT-RF_blend_ramp_bestval_h5          |
|  81 |         5 |  4.66651 | 2.16021 | 1.57757  |  2.7226  |          0.05 | linear_ridge+LLM-COT-RF_blend_ramp_bestval_h5          |
|  82 |        20 | 33.7068  | 5.80576 | 3.76486  |  6.77943 |          0.05 | linear_ridge+LLM-COT-RF_blend_ramp_bestval_h5          |
|  83 |        30 | 67.6697  | 8.22616 | 6.13897  | 11.4485  |          0.05 | linear_ridge+LLM-COT-RF_blend_ramp_bestval_h5          |
|  84 |         1 |  1.05022 | 1.0248  | 0.780708 |  1.38334 |          0.1  | linear_ridge+LLM-COT-RF_blend_ramp_bestval_h5          |
|  85 |         5 |  4.67733 | 2.16271 | 1.58012  |  2.72718 |          0.1  | linear_ridge+LLM-COT-RF_blend_ramp_bestval_h5          |
|  86 |        20 | 33.7207  | 5.80696 | 3.76682  |  6.78307 |          0.1  | linear_ridge+LLM-COT-RF_blend_ramp_bestval_h5          |
|  87 |        30 | 67.5832  | 8.2209  | 6.13163  | 11.4354  |          0.1  | linear_ridge+LLM-COT-RF_blend_ramp_bestval_h5          |
|  88 |         1 |  1.05048 | 1.02493 | 0.785195 |  1.39249 |          0.2  | linear_ridge+LLM-COT-RF_blend_ramp_bestval_h5          |
|  89 |         5 |  4.70891 | 2.17    | 1.58616  |  2.73804 |          0.2  | linear_ridge+LLM-COT-RF_blend_ramp_bestval_h5          |
|  90 |        20 | 33.756   | 5.80999 | 3.77125  |  6.79122 |          0.2  | linear_ridge+LLM-COT-RF_blend_ramp_bestval_h5          |
|  91 |        30 | 67.4178  | 8.21083 | 6.11965  | 11.4144  |          0.2  | linear_ridge+LLM-COT-RF_blend_ramp_bestval_h5          |
|  92 |         1 |  1.06231 | 1.03069 | 0.794113 |  1.40974 |          0.3  | linear_ridge+LLM-COT-RF_blend_ramp_bestval_h5          |
|  93 |         5 |  4.75377 | 2.18031 | 1.59353  |  2.75125 |          0.3  | linear_ridge+LLM-COT-RF_blend_ramp_bestval_h5          |
|  94 |        20 | 33.8011  | 5.81387 | 3.77691  |  6.80136 |          0.3  | linear_ridge+LLM-COT-RF_blend_ramp_bestval_h5          |
|  95 |        30 | 67.2625  | 8.20137 | 6.11013  | 11.398   |          0.3  | linear_ridge+LLM-COT-RF_blend_ramp_bestval_h5          |
|  96 |         1 |  1.05443 | 1.02686 | 0.780602 |  1.38256 |          0.05 | linear_ridge+LLM-COT-RF_blend_ramp_bestval_h20         |
|  97 |         5 |  4.66651 | 2.16021 | 1.57757  |  2.7226  |          0.05 | linear_ridge+LLM-COT-RF_blend_ramp_bestval_h20         |
|  98 |        20 | 33.7068  | 5.80576 | 3.76486  |  6.77943 |          0.05 | linear_ridge+LLM-COT-RF_blend_ramp_bestval_h20         |
|  99 |        30 | 67.6697  | 8.22616 | 6.13897  | 11.4485  |          0.05 | linear_ridge+LLM-COT-RF_blend_ramp_bestval_h20         |
| 100 |         1 |  1.05022 | 1.0248  | 0.780708 |  1.38334 |          0.1  | linear_ridge+LLM-COT-RF_blend_ramp_bestval_h20         |
| 101 |         5 |  4.67733 | 2.16271 | 1.58012  |  2.72718 |          0.1  | linear_ridge+LLM-COT-RF_blend_ramp_bestval_h20         |
| 102 |        20 | 33.7207  | 5.80696 | 3.76682  |  6.78307 |          0.1  | linear_ridge+LLM-COT-RF_blend_ramp_bestval_h20         |
| 103 |        30 | 67.5832  | 8.2209  | 6.13163  | 11.4354  |          0.1  | linear_ridge+LLM-COT-RF_blend_ramp_bestval_h20         |
| 104 |         1 |  1.05048 | 1.02493 | 0.785195 |  1.39249 |          0.2  | linear_ridge+LLM-COT-RF_blend_ramp_bestval_h20         |
| 105 |         5 |  4.70891 | 2.17    | 1.58616  |  2.73804 |          0.2  | linear_ridge+LLM-COT-RF_blend_ramp_bestval_h20         |
| 106 |        20 | 33.756   | 5.80999 | 3.77125  |  6.79122 |          0.2  | linear_ridge+LLM-COT-RF_blend_ramp_bestval_h20         |
| 107 |        30 | 67.4178  | 8.21083 | 6.11965  | 11.4144  |          0.2  | linear_ridge+LLM-COT-RF_blend_ramp_bestval_h20         |
| 108 |         1 |  1.06231 | 1.03069 | 0.794113 |  1.40974 |          0.3  | linear_ridge+LLM-COT-RF_blend_ramp_bestval_h20         |
| 109 |         5 |  4.75377 | 2.18031 | 1.59353  |  2.75125 |          0.3  | linear_ridge+LLM-COT-RF_blend_ramp_bestval_h20         |
| 110 |        20 | 33.8011  | 5.81387 | 3.77691  |  6.80136 |          0.3  | linear_ridge+LLM-COT-RF_blend_ramp_bestval_h20         |
| 111 |        30 | 67.2625  | 8.20137 | 6.11013  | 11.398   |          0.3  | linear_ridge+LLM-COT-RF_blend_ramp_bestval_h20         |
| 112 |         1 |  1.05665 | 1.02793 | 0.779838 |  1.38106 |          0.05 | linear_ridge+LLM-COT-RF_blend_ramp_bestval_h30         |
| 113 |         5 |  4.61872 | 2.14912 | 1.57198  |  2.71375 |          0.05 | linear_ridge+LLM-COT-RF_blend_ramp_bestval_h30         |
| 114 |        20 | 30.5092  | 5.52351 | 3.35626  |  6.10444 |          0.05 | linear_ridge+LLM-COT-RF_blend_ramp_bestval_h30         |
| 115 |        30 | 58.2236  | 7.63044 | 5.00192  |  9.53335 |          0.05 | linear_ridge+LLM-COT-RF_blend_ramp_bestval_h30         |
| 116 |         1 |  1.05415 | 1.02672 | 0.779552 |  1.38107 |          0.1  | linear_ridge+LLM-COT-RF_blend_ramp_bestval_h30         |
| 117 |         5 |  4.63159 | 2.15211 | 1.57588  |  2.72075 |          0.1  | linear_ridge+LLM-COT-RF_blend_ramp_bestval_h30         |
| 118 |        20 | 30.5228  | 5.52475 | 3.35943  |  6.11036 |          0.1  | linear_ridge+LLM-COT-RF_blend_ramp_bestval_h30         |
| 119 |        30 | 58.1954  | 7.62859 | 4.99739  |  9.52516 |          0.1  | linear_ridge+LLM-COT-RF_blend_ramp_bestval_h30         |
| 120 |         1 |  1.0563  | 1.02776 | 0.784035 |  1.38971 |          0.2  | linear_ridge+LLM-COT-RF_blend_ramp_bestval_h30         |
| 121 |         5 |  4.66718 | 2.16037 | 1.5838   |  2.73496 |          0.2  | linear_ridge+LLM-COT-RF_blend_ramp_bestval_h30         |
| 122 |        20 | 30.5577  | 5.5279  | 3.36627  |  6.12302 |          0.2  | linear_ridge+LLM-COT-RF_blend_ramp_bestval_h30         |
| 123 |        30 | 58.1472  | 7.62543 | 4.9905   |  9.51305 |          0.2  | linear_ridge+LLM-COT-RF_blend_ramp_bestval_h30         |
| 124 |         1 |  1.06798 | 1.03343 | 0.79343  |  1.40742 |          0.3  | linear_ridge+LLM-COT-RF_blend_ramp_bestval_h30         |
| 125 |         5 |  4.71593 | 2.17162 | 1.59329  |  2.75245 |          0.3  | linear_ridge+LLM-COT-RF_blend_ramp_bestval_h30         |
| 126 |        20 | 30.6026  | 5.53196 | 3.37498  |  6.1388  |          0.3  | linear_ridge+LLM-COT-RF_blend_ramp_bestval_h30         |
| 127 |        30 | 58.1095  | 7.62296 | 4.98598  |  9.50534 |          0.3  | linear_ridge+LLM-COT-RF_blend_ramp_bestval_h30         |

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

- **Price accuracy**: linear_ridge has the lowest average MSE (23.180).
- **TSM vs naive**: TSM MSE is 0.9x the naive baseline on average.
- **Directional accuracy**: linear_ridge has the highest average trend accuracy (0.606).

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


