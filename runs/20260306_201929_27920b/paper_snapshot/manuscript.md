# EU ETS Futures Forecasting with TSM + LLM Refinement

**Authors**: Auto-generated

**Generated**: 2026-03-06 20:55:28

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
| 16 |         1 |  1.22496 | 1.10678 | 0.823182 |  1.44975 | tsm                                           |
| 17 |         5 |  4.58705 | 2.14174 | 1.52805  |  2.62472 | tsm                                           |
| 18 |        20 | 29.8651  | 5.46489 | 3.58359  |  6.46353 | tsm                                           |
| 19 |        30 | 59.4805  | 7.71236 | 5.39684  | 10.1968  | tsm                                           |
| 20 |         1 |  1.13176 | 1.06384 | 0.793264 |  1.4016  | TSM+LLM-COT-RF                                |
| 21 |         5 |  5.10481 | 2.25938 | 1.64674  |  2.83632 | TSM+LLM-COT-RF                                |
| 22 |        20 | 36.6352  | 6.0527  | 3.95313  |  7.13654 | TSM+LLM-COT-RF                                |
| 23 |        30 | 69.9611  | 8.36427 | 5.89264  | 11.1375  | TSM+LLM-COT-RF                                |
| 24 |         1 |  1.22496 | 1.10678 | 0.823182 |  1.44975 | TSM+LLM-COT-RF_blend_ramp_bestval_path        |
| 25 |         5 |  4.60873 | 2.14679 | 1.53112  |  2.63137 | TSM+LLM-COT-RF_blend_ramp_bestval_path        |
| 26 |        20 | 32.8471  | 5.73124 | 3.73504  |  6.7418  | TSM+LLM-COT-RF_blend_ramp_bestval_path        |
| 27 |        30 | 67.0022  | 8.18549 | 5.75456  | 10.8773  | TSM+LLM-COT-RF_blend_ramp_bestval_path        |
| 28 |         1 |  1.22496 | 1.10678 | 0.823182 |  1.44975 | TSM+LLM-COT-RF_blend_ramp_bestval_path_h1base |
| 29 |         5 |  4.60873 | 2.14679 | 1.53112  |  2.63137 | TSM+LLM-COT-RF_blend_ramp_bestval_path_h1base |
| 30 |        20 | 32.8471  | 5.73124 | 3.73504  |  6.7418  | TSM+LLM-COT-RF_blend_ramp_bestval_path_h1base |
| 31 |        30 | 67.0022  | 8.18549 | 5.75456  | 10.8773  | TSM+LLM-COT-RF_blend_ramp_bestval_path_h1base |
| 32 |         1 |  1.22496 | 1.10678 | 0.823182 |  1.44975 | TSM+LLM-COT-RF_blend_ramp_bestval_h1          |
| 33 |         5 |  4.58705 | 2.14174 | 1.52805  |  2.62472 | TSM+LLM-COT-RF_blend_ramp_bestval_h1          |
| 34 |        20 | 29.8651  | 5.46489 | 3.58359  |  6.46353 | TSM+LLM-COT-RF_blend_ramp_bestval_h1          |
| 35 |        30 | 59.4805  | 7.71236 | 5.39684  | 10.1968  | TSM+LLM-COT-RF_blend_ramp_bestval_h1          |
| 36 |         1 |  1.22496 | 1.10678 | 0.823182 |  1.44975 | TSM+LLM-COT-RF_blend_ramp_bestval_h5          |
| 37 |         5 |  4.61759 | 2.14886 | 1.53272  |  2.63462 | TSM+LLM-COT-RF_blend_ramp_bestval_h5          |
| 38 |        20 | 33.9891  | 5.83002 | 3.79704  |  6.8555  | TSM+LLM-COT-RF_blend_ramp_bestval_h5          |
| 39 |        30 | 69.9611  | 8.36427 | 5.89264  | 11.1375  | TSM+LLM-COT-RF_blend_ramp_bestval_h5          |
| 40 |         1 |  1.22496 | 1.10678 | 0.823182 |  1.44975 | TSM+LLM-COT-RF_blend_ramp_bestval_h20         |
| 41 |         5 |  4.60873 | 2.14679 | 1.53112  |  2.63137 | TSM+LLM-COT-RF_blend_ramp_bestval_h20         |
| 42 |        20 | 32.8471  | 5.73124 | 3.73504  |  6.7418  | TSM+LLM-COT-RF_blend_ramp_bestval_h20         |
| 43 |        30 | 67.0022  | 8.18549 | 5.75456  | 10.8773  | TSM+LLM-COT-RF_blend_ramp_bestval_h20         |
| 44 |         1 |  1.22496 | 1.10678 | 0.823182 |  1.44975 | TSM+LLM-COT-RF_blend_ramp_bestval_h30         |
| 45 |         5 |  4.60068 | 2.14492 | 1.52992  |  2.62884 | TSM+LLM-COT-RF_blend_ramp_bestval_h30         |
| 46 |        20 | 31.7791  | 5.6373  | 3.67732  |  6.63608 | TSM+LLM-COT-RF_blend_ramp_bestval_h30         |
| 47 |        30 | 64.2692  | 8.0168  | 5.62321  | 10.6291  | TSM+LLM-COT-RF_blend_ramp_bestval_h30         |
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
| 64 |         1 |  1.22496 | 1.10678 | 0.823182 |  1.44975 | tsm_llm_subset                                |
| 65 |         5 |  4.58705 | 2.14174 | 1.52805  |  2.62472 | tsm_llm_subset                                |
| 66 |        20 | 29.8651  | 5.46489 | 3.58359  |  6.46353 | tsm_llm_subset                                |
| 67 |        30 | 59.4805  | 7.71236 | 5.39684  | 10.1968  | tsm_llm_subset                                |

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
| 16 |         1 |  0.409722  |     0.192308  |       0.114286  |       0.789474  |     52 |       35 |       57 | tsm                                           |
| 17 |         5 |  0.451389  |     0.428571  |       0.4       |       0.558824  |     70 |       40 |       34 | tsm                                           |
| 18 |        20 |  0.534722  |     0.566667  |       0.567568  |       0.294118  |     90 |       37 |       17 | tsm                                           |
| 19 |        30 |  0.618056  |     0.59      |       0.818182  |       0.272727  |    100 |       33 |       11 | tsm                                           |
| 20 |         1 |  0.409722  |     0.0576923 |       0.171429  |       0.877193  |     52 |       35 |       57 | TSM+LLM-COT-RF                                |
| 21 |         5 |  0.354167  |     0.257143  |       0.3       |       0.617647  |     70 |       40 |       34 | TSM+LLM-COT-RF                                |
| 22 |        20 |  0.388889  |     0.4       |       0.432432  |       0.235294  |     90 |       37 |       17 | TSM+LLM-COT-RF                                |
| 23 |        30 |  0.472222  |     0.41      |       0.666667  |       0.454545  |    100 |       33 |       11 | TSM+LLM-COT-RF                                |
| 24 |         1 |  0.409722  |     0.192308  |       0.114286  |       0.789474  |     52 |       35 |       57 | TSM+LLM-COT-RF_blend_ramp_bestval_path        |
| 25 |         5 |  0.451389  |     0.414286  |       0.4       |       0.588235  |     70 |       40 |       34 | TSM+LLM-COT-RF_blend_ramp_bestval_path        |
| 26 |        20 |  0.479167  |     0.477778  |       0.540541  |       0.352941  |     90 |       37 |       17 | TSM+LLM-COT-RF_blend_ramp_bestval_path        |
| 27 |        30 |  0.534722  |     0.47      |       0.727273  |       0.545455  |    100 |       33 |       11 | TSM+LLM-COT-RF_blend_ramp_bestval_path        |
| 28 |         1 |  0.409722  |     0.192308  |       0.114286  |       0.789474  |     52 |       35 |       57 | TSM+LLM-COT-RF_blend_ramp_bestval_path_h1base |
| 29 |         5 |  0.451389  |     0.414286  |       0.4       |       0.588235  |     70 |       40 |       34 | TSM+LLM-COT-RF_blend_ramp_bestval_path_h1base |
| 30 |        20 |  0.479167  |     0.477778  |       0.540541  |       0.352941  |     90 |       37 |       17 | TSM+LLM-COT-RF_blend_ramp_bestval_path_h1base |
| 31 |        30 |  0.534722  |     0.47      |       0.727273  |       0.545455  |    100 |       33 |       11 | TSM+LLM-COT-RF_blend_ramp_bestval_path_h1base |
| 32 |         1 |  0.409722  |     0.192308  |       0.114286  |       0.789474  |     52 |       35 |       57 | TSM+LLM-COT-RF_blend_ramp_bestval_h1          |
| 33 |         5 |  0.451389  |     0.428571  |       0.4       |       0.558824  |     70 |       40 |       34 | TSM+LLM-COT-RF_blend_ramp_bestval_h1          |
| 34 |        20 |  0.534722  |     0.566667  |       0.567568  |       0.294118  |     90 |       37 |       17 | TSM+LLM-COT-RF_blend_ramp_bestval_h1          |
| 35 |        30 |  0.618056  |     0.59      |       0.818182  |       0.272727  |    100 |       33 |       11 | TSM+LLM-COT-RF_blend_ramp_bestval_h1          |
| 36 |         1 |  0.409722  |     0.192308  |       0.114286  |       0.789474  |     52 |       35 |       57 | TSM+LLM-COT-RF_blend_ramp_bestval_h5          |
| 37 |         5 |  0.451389  |     0.414286  |       0.4       |       0.588235  |     70 |       40 |       34 | TSM+LLM-COT-RF_blend_ramp_bestval_h5          |
| 38 |        20 |  0.430556  |     0.444444  |       0.486486  |       0.235294  |     90 |       37 |       17 | TSM+LLM-COT-RF_blend_ramp_bestval_h5          |
| 39 |        30 |  0.472222  |     0.41      |       0.666667  |       0.454545  |    100 |       33 |       11 | TSM+LLM-COT-RF_blend_ramp_bestval_h5          |
| 40 |         1 |  0.409722  |     0.192308  |       0.114286  |       0.789474  |     52 |       35 |       57 | TSM+LLM-COT-RF_blend_ramp_bestval_h20         |
| 41 |         5 |  0.451389  |     0.414286  |       0.4       |       0.588235  |     70 |       40 |       34 | TSM+LLM-COT-RF_blend_ramp_bestval_h20         |
| 42 |        20 |  0.479167  |     0.477778  |       0.540541  |       0.352941  |     90 |       37 |       17 | TSM+LLM-COT-RF_blend_ramp_bestval_h20         |
| 43 |        30 |  0.534722  |     0.47      |       0.727273  |       0.545455  |    100 |       33 |       11 | TSM+LLM-COT-RF_blend_ramp_bestval_h20         |
| 44 |         1 |  0.409722  |     0.192308  |       0.114286  |       0.789474  |     52 |       35 |       57 | TSM+LLM-COT-RF_blend_ramp_bestval_h30         |
| 45 |         5 |  0.451389  |     0.428571  |       0.4       |       0.558824  |     70 |       40 |       34 | TSM+LLM-COT-RF_blend_ramp_bestval_h30         |
| 46 |        20 |  0.513889  |     0.533333  |       0.540541  |       0.352941  |     90 |       37 |       17 | TSM+LLM-COT-RF_blend_ramp_bestval_h30         |
| 47 |        30 |  0.5625    |     0.51      |       0.727273  |       0.545455  |    100 |       33 |       11 | TSM+LLM-COT-RF_blend_ramp_bestval_h30         |
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
| 64 |         1 |  0.409722  |     0.192308  |       0.114286  |       0.789474  |     52 |       35 |       57 | tsm_llm_subset                                |
| 65 |         5 |  0.451389  |     0.428571  |       0.4       |       0.558824  |     70 |       40 |       34 | tsm_llm_subset                                |
| 66 |        20 |  0.534722  |     0.566667  |       0.567568  |       0.294118  |     90 |       37 |       17 | tsm_llm_subset                                |
| 67 |        30 |  0.618056  |     0.59      |       0.818182  |       0.272727  |    100 |       33 |       11 | tsm_llm_subset                                |

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
| 12 |         1 | tsm_llm_subset                                | naive_persistence_llm_subset |            1.22496 |               1.06139 |     -15.4104      |    1.3854     | 0.168087    | False           |                 5127 |       0.85286     | False                  |     1.3854     | 0.16593     | False            | False          |
| 13 |         5 | tsm_llm_subset                                | naive_persistence_llm_subset |            4.58705 |               5.2119  |      11.989       |   -1.14818    | 0.25281     | False           |                 3573 |       0.0010212   | True                   |    -0.594844   | 0.551947    | False            | True           |
| 14 |        20 | tsm_llm_subset                                | naive_persistence_llm_subset |           29.8651  |              44.2759  |      32.5477      |   -4.95125    | 2.04896e-06 | True            |                 2397 |       1.80292e-08 | True                   |    -1.72788    | 0.0840099   | False            | True           |
| 15 |        30 | tsm_llm_subset                                | naive_persistence_llm_subset |           59.4805  |              79.9881  |      25.6383      |   -5.43666    | 2.28549e-07 | True            |                 1244 |       2.20288e-15 | True                   |    -1.29984    | 0.193657    | False            | True           |
| 16 |         1 | TSM+LLM-COT-RF                                | naive_persistence_llm_subset |            1.13176 |               1.06139 |      -6.62946     |    0.694358   | 0.488584    | False           |                 4973 |       0.6223      | False                  |     0.694358   | 0.487458    | False            | False          |
| 17 |         5 | TSM+LLM-COT-RF                                | naive_persistence_llm_subset |            5.10481 |               5.2119  |       2.05462     |   -0.258701   | 0.796238    | False           |                 4531 |       0.169418    | False                  |    -0.132056   | 0.89494     | False            | True           |
| 18 |        20 | TSM+LLM-COT-RF                                | naive_persistence_llm_subset |           36.6352  |              44.2759  |      17.257       |   -2.94664    | 0.00375211  | True            |                 3748 |       0.00332886  | True                   |    -1.69087    | 0.0908624   | False            | True           |
| 19 |        30 | TSM+LLM-COT-RF                                | naive_persistence_llm_subset |           69.9611  |              79.9881  |      12.5356      |   -2.75144    | 0.00670174  | True            |                 2859 |       2.49482e-06 | True                   |    -1.19921    | 0.230445    | False            | True           |
| 20 |         1 | TSM+LLM-COT-RF_blend_ramp_bestval_path        | naive_persistence_llm_subset |            1.22496 |               1.06139 |     -15.4104      |    1.3854     | 0.168087    | False           |                 5127 |       0.85286     | False                  |     1.3854     | 0.16593     | False            | False          |
| 21 |         5 | TSM+LLM-COT-RF_blend_ramp_bestval_path        | naive_persistence_llm_subset |            4.60873 |               5.2119  |      11.5729      |   -1.15667    | 0.249337    | False           |                 3519 |       0.000693055 | True                   |    -0.593647   | 0.552748    | False            | True           |
| 22 |        20 | TSM+LLM-COT-RF_blend_ramp_bestval_path        | naive_persistence_llm_subset |           32.8471  |              44.2759  |      25.8125      |   -4.78193    | 4.27005e-06 | True            |                 2718 |       6.04576e-07 | True                   |    -1.74815    | 0.0804383   | False            | True           |
| 23 |        30 | TSM+LLM-COT-RF_blend_ramp_bestval_path        | naive_persistence_llm_subset |           67.0022  |              79.9881  |      16.2348      |   -4.1451     | 5.79224e-05 | True            |                 2420 |       2.34978e-08 | True                   |    -1.24528    | 0.213028    | False            | True           |
| 24 |         1 | TSM+LLM-COT-RF_blend_ramp_bestval_path_h1base | naive_persistence_llm_subset |            1.22496 |               1.06139 |     -15.4104      |    1.3854     | 0.168087    | False           |                 5127 |       0.85286     | False                  |     1.3854     | 0.16593     | False            | False          |
| 25 |         5 | TSM+LLM-COT-RF_blend_ramp_bestval_path_h1base | naive_persistence_llm_subset |            4.60873 |               5.2119  |      11.5729      |   -1.15667    | 0.249337    | False           |                 3519 |       0.000693055 | True                   |    -0.593647   | 0.552748    | False            | True           |
| 26 |        20 | TSM+LLM-COT-RF_blend_ramp_bestval_path_h1base | naive_persistence_llm_subset |           32.8471  |              44.2759  |      25.8125      |   -4.78193    | 4.27005e-06 | True            |                 2718 |       6.04576e-07 | True                   |    -1.74815    | 0.0804383   | False            | True           |
| 27 |        30 | TSM+LLM-COT-RF_blend_ramp_bestval_path_h1base | naive_persistence_llm_subset |           67.0022  |              79.9881  |      16.2348      |   -4.1451     | 5.79224e-05 | True            |                 2420 |       2.34978e-08 | True                   |    -1.24528    | 0.213028    | False            | True           |
| 28 |         1 | TSM+LLM-COT-RF_blend_ramp_bestval_h1          | naive_persistence_llm_subset |            1.22496 |               1.06139 |     -15.4104      |    1.3854     | 0.168087    | False           |                 5127 |       0.85286     | False                  |     1.3854     | 0.16593     | False            | False          |
| 29 |         5 | TSM+LLM-COT-RF_blend_ramp_bestval_h1          | naive_persistence_llm_subset |            4.58705 |               5.2119  |      11.989       |   -1.14818    | 0.25281     | False           |                 3573 |       0.0010212   | True                   |    -0.594844   | 0.551947    | False            | True           |
| 30 |        20 | TSM+LLM-COT-RF_blend_ramp_bestval_h1          | naive_persistence_llm_subset |           29.8651  |              44.2759  |      32.5477      |   -4.95125    | 2.04896e-06 | True            |                 2397 |       1.80292e-08 | True                   |    -1.72788    | 0.0840099   | False            | True           |
| 31 |        30 | TSM+LLM-COT-RF_blend_ramp_bestval_h1          | naive_persistence_llm_subset |           59.4805  |              79.9881  |      25.6383      |   -5.43666    | 2.28549e-07 | True            |                 1244 |       2.20288e-15 | True                   |    -1.29984    | 0.193657    | False            | True           |
| 32 |         1 | TSM+LLM-COT-RF_blend_ramp_bestval_h5          | naive_persistence_llm_subset |            1.22496 |               1.06139 |     -15.4104      |    1.3854     | 0.168087    | False           |                 5127 |       0.85286     | False                  |     1.3854     | 0.16593     | False            | False          |
| 33 |         5 | TSM+LLM-COT-RF_blend_ramp_bestval_h5          | naive_persistence_llm_subset |            4.61759 |               5.2119  |      11.4029      |   -1.15581    | 0.249687    | False           |                 3512 |       0.000658565 | True                   |    -0.591393   | 0.554257    | False            | True           |
| 34 |        20 | TSM+LLM-COT-RF_blend_ramp_bestval_h5          | naive_persistence_llm_subset |           33.9891  |              44.2759  |      23.2332      |   -4.38251    | 2.25674e-05 | True            |                 2983 |       8.14795e-06 | True                   |    -1.74738    | 0.080571    | False            | True           |
| 35 |        30 | TSM+LLM-COT-RF_blend_ramp_bestval_h5          | naive_persistence_llm_subset |           69.9611  |              79.9881  |      12.5356      |   -2.75144    | 0.00670174  | True            |                 2859 |       2.49482e-06 | True                   |    -1.19921    | 0.230445    | False            | True           |
| 36 |         1 | TSM+LLM-COT-RF_blend_ramp_bestval_h20         | naive_persistence_llm_subset |            1.22496 |               1.06139 |     -15.4104      |    1.3854     | 0.168087    | False           |                 5127 |       0.85286     | False                  |     1.3854     | 0.16593     | False            | False          |
| 37 |         5 | TSM+LLM-COT-RF_blend_ramp_bestval_h20         | naive_persistence_llm_subset |            4.60873 |               5.2119  |      11.5729      |   -1.15667    | 0.249337    | False           |                 3519 |       0.000693055 | True                   |    -0.593647   | 0.552748    | False            | True           |
| 38 |        20 | TSM+LLM-COT-RF_blend_ramp_bestval_h20         | naive_persistence_llm_subset |           32.8471  |              44.2759  |      25.8125      |   -4.78193    | 4.27005e-06 | True            |                 2718 |       6.04576e-07 | True                   |    -1.74815    | 0.0804383   | False            | True           |
| 39 |        30 | TSM+LLM-COT-RF_blend_ramp_bestval_h20         | naive_persistence_llm_subset |           67.0022  |              79.9881  |      16.2348      |   -4.1451     | 5.79224e-05 | True            |                 2420 |       2.34978e-08 | True                   |    -1.24528    | 0.213028    | False            | True           |
| 40 |         1 | TSM+LLM-COT-RF_blend_ramp_bestval_h30         | naive_persistence_llm_subset |            1.22496 |               1.06139 |     -15.4104      |    1.3854     | 0.168087    | False           |                 5127 |       0.85286     | False                  |     1.3854     | 0.16593     | False            | False          |
| 41 |         5 | TSM+LLM-COT-RF_blend_ramp_bestval_h30         | naive_persistence_llm_subset |            4.60068 |               5.2119  |      11.7273      |   -1.15562    | 0.249764    | False           |                 3542 |       0.000818556 | True                   |    -0.594952   | 0.551876    | False            | True           |
| 42 |        20 | TSM+LLM-COT-RF_blend_ramp_bestval_h30         | naive_persistence_llm_subset |           31.7791  |              44.2759  |      28.2247      |   -4.97224    | 1.8685e-06  | True            |                 2596 |       1.66731e-07 | True                   |    -1.74336    | 0.0812702   | False            | True           |
| 43 |        30 | TSM+LLM-COT-RF_blend_ramp_bestval_h30         | naive_persistence_llm_subset |           64.2692  |              79.9881  |      19.6516      |   -5.16645    | 7.87138e-07 | True            |                 1900 |       3.56558e-11 | True                   |    -1.27304    | 0.203004    | False            | True           |


## 5. Robustness Analysis

### 5.1 Noise Injection Test

To assess robustness, we inject noise into forecast paths at levels of 5%, 10%, 20%, 
and 30% of the forecast standard deviation.

|     |   horizon |      mse |    rmse |      mae |     mape |   noise_level | model                                         |
|----:|----------:|---------:|--------:|---------:|---------:|--------------:|:----------------------------------------------|
|   0 |         1 |  1.21489 | 1.10222 | 0.820331 |  1.44583 |          0.05 | tsm                                           |
|   1 |         5 |  4.59811 | 2.14432 | 1.52671  |  2.62331 |          0.05 | tsm                                           |
|   2 |        20 | 29.8561  | 5.46407 | 3.58635  |  6.46801 |          0.05 | tsm                                           |
|   3 |        30 | 59.4497  | 7.71036 | 5.3919   | 10.1879  |          0.05 | tsm                                           |
|   4 |         1 |  1.20679 | 1.09854 | 0.817577 |  1.44208 |          0.1  | tsm                                           |
|   5 |         5 |  4.61202 | 2.14756 | 1.52665  |  2.62418 |          0.1  | tsm                                           |
|   6 |        20 | 29.8493  | 5.46345 | 3.58911  |  6.47248 |          0.1  | tsm                                           |
|   7 |        30 | 59.4206  | 7.70847 | 5.38697  | 10.1789  |          0.1  | tsm                                           |
|   8 |         1 |  1.19649 | 1.09384 | 0.812703 |  1.4359  |          0.2  | tsm                                           |
|   9 |         5 |  4.6484  | 2.15601 | 1.52773  |  2.62835 |          0.2  | tsm                                           |
|  10 |        20 | 29.8426  | 5.46284 | 3.59462  |  6.48143 |          0.2  | tsm                                           |
|  11 |        30 | 59.3676  | 7.70504 | 5.37711  | 10.161   |          0.2  | tsm                                           |
|  12 |         1 |  1.19405 | 1.09273 | 0.810592 |  1.43459 |          0.3  | tsm                                           |
|  13 |         5 |  4.69618 | 2.16707 | 1.53203  |  2.63806 |          0.3  | tsm                                           |
|  14 |        20 | 29.8449  | 5.46305 | 3.60096  |  6.49201 |          0.3  | tsm                                           |
|  15 |        30 | 59.3218  | 7.70206 | 5.36768  | 10.1439  |          0.3  | tsm                                           |
|  16 |         1 |  1.12092 | 1.05873 | 0.791802 |  1.4003  |          0.05 | TSM+LLM-COT-RF                                |
|  17 |         5 |  5.11496 | 2.26163 | 1.64676  |  2.83693 |          0.05 | TSM+LLM-COT-RF                                |
|  18 |        20 | 36.6309  | 6.05235 | 3.95677  |  7.14253 |          0.05 | TSM+LLM-COT-RF                                |
|  19 |        30 | 69.9195  | 8.36179 | 5.88818  | 11.1293  |          0.05 | TSM+LLM-COT-RF                                |
|  20 |         1 |  1.11193 | 1.05448 | 0.79046  |  1.39921 |          0.1  | TSM+LLM-COT-RF                                |
|  21 |         5 |  5.12781 | 2.26447 | 1.64797  |  2.83965 |          0.1  | TSM+LLM-COT-RF                                |
|  22 |        20 | 36.6288  | 6.05217 | 3.96042  |  7.14851 |          0.1  | TSM+LLM-COT-RF                                |
|  23 |        30 | 69.8794  | 8.35939 | 5.88371  | 11.1211  |          0.1  | TSM+LLM-COT-RF                                |
|  24 |         1 |  1.09949 | 1.04857 | 0.788351 |  1.39809 |          0.2  | TSM+LLM-COT-RF                                |
|  25 |         5 |  5.16159 | 2.27191 | 1.65208  |  2.84835 |          0.2  | TSM+LLM-COT-RF                                |
|  26 |        20 | 36.631   | 6.05235 | 3.96808  |  7.16116 |          0.2  | TSM+LLM-COT-RF                                |
|  27 |        30 | 69.8036  | 8.35486 | 5.87487  | 11.1049  |          0.2  | TSM+LLM-COT-RF                                |
|  28 |         1 |  1.09445 | 1.04616 | 0.78821  |  1.40029 |          0.3  | TSM+LLM-COT-RF                                |
|  29 |         5 |  5.20616 | 2.2817  | 1.65808  |  2.86048 |          0.3  | TSM+LLM-COT-RF                                |
|  30 |        20 | 36.6418  | 6.05325 | 3.97833  |  7.17834 |          0.3  | TSM+LLM-COT-RF                                |
|  31 |        30 | 69.7339  | 8.35068 | 5.86677  | 11.09    |          0.3  | TSM+LLM-COT-RF                                |
|  32 |         1 |  1.21657 | 1.10298 | 0.820888 |  1.44664 |          0.05 | TSM+LLM-COT-RF_blend_ramp_bestval_path        |
|  33 |         5 |  4.61509 | 2.14828 | 1.52919  |  2.62891 |          0.05 | TSM+LLM-COT-RF_blend_ramp_bestval_path        |
|  34 |        20 | 32.8312  | 5.72985 | 3.73993  |  6.75007 |          0.05 | TSM+LLM-COT-RF_blend_ramp_bestval_path        |
|  35 |        30 | 66.9657  | 8.18326 | 5.75043  | 10.8698  |          0.05 | TSM+LLM-COT-RF_blend_ramp_bestval_path        |
|  36 |         1 |  1.20999 | 1.1     | 0.818691 |  1.44369 |          0.1  | TSM+LLM-COT-RF_blend_ramp_bestval_path        |
|  37 |         5 |  4.62407 | 2.15036 | 1.52866  |  2.62896 |          0.1  | TSM+LLM-COT-RF_blend_ramp_bestval_path        |
|  38 |        20 | 32.8174  | 5.72865 | 3.74482  |  6.75835 |          0.1  | TSM+LLM-COT-RF_blend_ramp_bestval_path        |
|  39 |        30 | 66.9309  | 8.18113 | 5.74631  | 10.8622  |          0.1  | TSM+LLM-COT-RF_blend_ramp_bestval_path        |
|  40 |         1 |  1.20225 | 1.09647 | 0.815198 |  1.43965 |          0.2  | TSM+LLM-COT-RF_blend_ramp_bestval_path        |
|  41 |         5 |  4.64984 | 2.15635 | 1.52953  |  2.63278 |          0.2  | TSM+LLM-COT-RF_blend_ramp_bestval_path        |
|  42 |        20 | 32.796   | 5.72678 | 3.7546   |  6.77491 |          0.2  | TSM+LLM-COT-RF_blend_ramp_bestval_path        |
|  43 |        30 | 66.8663  | 8.17718 | 5.73805  | 10.8471  |          0.2  | TSM+LLM-COT-RF_blend_ramp_bestval_path        |
|  44 |         1 |  1.20174 | 1.09624 | 0.815102 |  1.44162 |          0.3  | TSM+LLM-COT-RF_blend_ramp_bestval_path        |
|  45 |         5 |  4.68606 | 2.16473 | 1.53191  |  2.63947 |          0.3  | TSM+LLM-COT-RF_blend_ramp_bestval_path        |
|  46 |        20 | 32.7831  | 5.72566 | 3.76438  |  6.79147 |          0.3  | TSM+LLM-COT-RF_blend_ramp_bestval_path        |
|  47 |        30 | 66.8083  | 8.17363 | 5.73019  | 10.8328  |          0.3  | TSM+LLM-COT-RF_blend_ramp_bestval_path        |
|  48 |         1 |  1.21657 | 1.10298 | 0.820888 |  1.44664 |          0.05 | TSM+LLM-COT-RF_blend_ramp_bestval_path_h1base |
|  49 |         5 |  4.61509 | 2.14828 | 1.52919  |  2.62891 |          0.05 | TSM+LLM-COT-RF_blend_ramp_bestval_path_h1base |
|  50 |        20 | 32.8312  | 5.72985 | 3.73993  |  6.75007 |          0.05 | TSM+LLM-COT-RF_blend_ramp_bestval_path_h1base |
|  51 |        30 | 66.9657  | 8.18326 | 5.75043  | 10.8698  |          0.05 | TSM+LLM-COT-RF_blend_ramp_bestval_path_h1base |
|  52 |         1 |  1.20999 | 1.1     | 0.818691 |  1.44369 |          0.1  | TSM+LLM-COT-RF_blend_ramp_bestval_path_h1base |
|  53 |         5 |  4.62407 | 2.15036 | 1.52866  |  2.62896 |          0.1  | TSM+LLM-COT-RF_blend_ramp_bestval_path_h1base |
|  54 |        20 | 32.8174  | 5.72865 | 3.74482  |  6.75835 |          0.1  | TSM+LLM-COT-RF_blend_ramp_bestval_path_h1base |
|  55 |        30 | 66.9309  | 8.18113 | 5.74631  | 10.8622  |          0.1  | TSM+LLM-COT-RF_blend_ramp_bestval_path_h1base |
|  56 |         1 |  1.20225 | 1.09647 | 0.815198 |  1.43965 |          0.2  | TSM+LLM-COT-RF_blend_ramp_bestval_path_h1base |
|  57 |         5 |  4.64984 | 2.15635 | 1.52953  |  2.63278 |          0.2  | TSM+LLM-COT-RF_blend_ramp_bestval_path_h1base |
|  58 |        20 | 32.796   | 5.72678 | 3.7546   |  6.77491 |          0.2  | TSM+LLM-COT-RF_blend_ramp_bestval_path_h1base |
|  59 |        30 | 66.8663  | 8.17718 | 5.73805  | 10.8471  |          0.2  | TSM+LLM-COT-RF_blend_ramp_bestval_path_h1base |
|  60 |         1 |  1.20174 | 1.09624 | 0.815102 |  1.44162 |          0.3  | TSM+LLM-COT-RF_blend_ramp_bestval_path_h1base |
|  61 |         5 |  4.68606 | 2.16473 | 1.53191  |  2.63947 |          0.3  | TSM+LLM-COT-RF_blend_ramp_bestval_path_h1base |
|  62 |        20 | 32.7831  | 5.72566 | 3.76438  |  6.79147 |          0.3  | TSM+LLM-COT-RF_blend_ramp_bestval_path_h1base |
|  63 |        30 | 66.8083  | 8.17363 | 5.73019  | 10.8328  |          0.3  | TSM+LLM-COT-RF_blend_ramp_bestval_path_h1base |
|  64 |         1 |  1.21489 | 1.10222 | 0.820331 |  1.44583 |          0.05 | TSM+LLM-COT-RF_blend_ramp_bestval_h1          |
|  65 |         5 |  4.59811 | 2.14432 | 1.52671  |  2.62331 |          0.05 | TSM+LLM-COT-RF_blend_ramp_bestval_h1          |
|  66 |        20 | 29.8561  | 5.46407 | 3.58635  |  6.46801 |          0.05 | TSM+LLM-COT-RF_blend_ramp_bestval_h1          |
|  67 |        30 | 59.4497  | 7.71036 | 5.3919   | 10.1879  |          0.05 | TSM+LLM-COT-RF_blend_ramp_bestval_h1          |
|  68 |         1 |  1.20679 | 1.09854 | 0.817577 |  1.44208 |          0.1  | TSM+LLM-COT-RF_blend_ramp_bestval_h1          |
|  69 |         5 |  4.61202 | 2.14756 | 1.52665  |  2.62418 |          0.1  | TSM+LLM-COT-RF_blend_ramp_bestval_h1          |
|  70 |        20 | 29.8493  | 5.46345 | 3.58911  |  6.47248 |          0.1  | TSM+LLM-COT-RF_blend_ramp_bestval_h1          |
|  71 |        30 | 59.4206  | 7.70847 | 5.38697  | 10.1789  |          0.1  | TSM+LLM-COT-RF_blend_ramp_bestval_h1          |
|  72 |         1 |  1.19649 | 1.09384 | 0.812703 |  1.4359  |          0.2  | TSM+LLM-COT-RF_blend_ramp_bestval_h1          |
|  73 |         5 |  4.6484  | 2.15601 | 1.52773  |  2.62835 |          0.2  | TSM+LLM-COT-RF_blend_ramp_bestval_h1          |
|  74 |        20 | 29.8426  | 5.46284 | 3.59462  |  6.48143 |          0.2  | TSM+LLM-COT-RF_blend_ramp_bestval_h1          |
|  75 |        30 | 59.3676  | 7.70504 | 5.37711  | 10.161   |          0.2  | TSM+LLM-COT-RF_blend_ramp_bestval_h1          |
|  76 |         1 |  1.19405 | 1.09273 | 0.810592 |  1.43459 |          0.3  | TSM+LLM-COT-RF_blend_ramp_bestval_h1          |
|  77 |         5 |  4.69618 | 2.16707 | 1.53203  |  2.63806 |          0.3  | TSM+LLM-COT-RF_blend_ramp_bestval_h1          |
|  78 |        20 | 29.8449  | 5.46305 | 3.60096  |  6.49201 |          0.3  | TSM+LLM-COT-RF_blend_ramp_bestval_h1          |
|  79 |        30 | 59.3218  | 7.70206 | 5.36768  | 10.1439  |          0.3  | TSM+LLM-COT-RF_blend_ramp_bestval_h1          |
|  80 |         1 |  1.21548 | 1.10249 | 0.820509 |  1.44604 |          0.05 | TSM+LLM-COT-RF_blend_ramp_bestval_h5          |
|  81 |         5 |  4.62124 | 2.14971 | 1.53001  |  2.63088 |          0.05 | TSM+LLM-COT-RF_blend_ramp_bestval_h5          |
|  82 |        20 | 33.9764  | 5.82893 | 3.8022   |  6.86424 |          0.05 | TSM+LLM-COT-RF_blend_ramp_bestval_h5          |
|  83 |        30 | 69.9102  | 8.36123 | 5.88827  | 11.1294  |          0.05 | TSM+LLM-COT-RF_blend_ramp_bestval_h5          |
|  84 |         1 |  1.20794 | 1.09906 | 0.817933 |  1.4425  |          0.1  | TSM+LLM-COT-RF_blend_ramp_bestval_h5          |
|  85 |         5 |  4.62782 | 2.15124 | 1.52874  |  2.62972 |          0.1  | TSM+LLM-COT-RF_blend_ramp_bestval_h5          |
|  86 |        20 | 33.9658  | 5.82802 | 3.80736  |  6.87299 |          0.1  | TSM+LLM-COT-RF_blend_ramp_bestval_h5          |
|  87 |        30 | 69.861   | 8.35829 | 5.8839   | 11.1214  |          0.1  | TSM+LLM-COT-RF_blend_ramp_bestval_h5          |
|  88 |         1 |  1.19865 | 1.09483 | 0.814038 |  1.43792 |          0.2  | TSM+LLM-COT-RF_blend_ramp_bestval_h5          |
|  89 |         5 |  4.64976 | 2.15633 | 1.52839  |  2.63157 |          0.2  | TSM+LLM-COT-RF_blend_ramp_bestval_h5          |
|  90 |        20 | 33.9513  | 5.82678 | 3.81767  |  6.89048 |          0.2  | TSM+LLM-COT-RF_blend_ramp_bestval_h5          |
|  91 |        30 | 69.7681  | 8.35273 | 5.87532  | 11.1055  |          0.2  | TSM+LLM-COT-RF_blend_ramp_bestval_h5          |
|  92 |         1 |  1.19708 | 1.09411 | 0.81336  |  1.43902 |          0.3  | TSM+LLM-COT-RF_blend_ramp_bestval_h5          |
|  93 |         5 |  4.68343 | 2.16412 | 1.53066  |  2.638   |          0.3  | TSM+LLM-COT-RF_blend_ramp_bestval_h5          |
|  94 |        20 | 33.9456  | 5.82629 | 3.82798  |  6.90797 |          0.3  | TSM+LLM-COT-RF_blend_ramp_bestval_h5          |
|  95 |        30 | 69.6822  | 8.34759 | 5.86745  | 11.0909  |          0.3  | TSM+LLM-COT-RF_blend_ramp_bestval_h5          |
|  96 |         1 |  1.21657 | 1.10298 | 0.820888 |  1.44664 |          0.05 | TSM+LLM-COT-RF_blend_ramp_bestval_h20         |
|  97 |         5 |  4.61509 | 2.14828 | 1.52919  |  2.62891 |          0.05 | TSM+LLM-COT-RF_blend_ramp_bestval_h20         |
|  98 |        20 | 32.8312  | 5.72985 | 3.73993  |  6.75007 |          0.05 | TSM+LLM-COT-RF_blend_ramp_bestval_h20         |
|  99 |        30 | 66.9657  | 8.18326 | 5.75043  | 10.8698  |          0.05 | TSM+LLM-COT-RF_blend_ramp_bestval_h20         |
| 100 |         1 |  1.20999 | 1.1     | 0.818691 |  1.44369 |          0.1  | TSM+LLM-COT-RF_blend_ramp_bestval_h20         |
| 101 |         5 |  4.62407 | 2.15036 | 1.52866  |  2.62896 |          0.1  | TSM+LLM-COT-RF_blend_ramp_bestval_h20         |
| 102 |        20 | 32.8174  | 5.72865 | 3.74482  |  6.75835 |          0.1  | TSM+LLM-COT-RF_blend_ramp_bestval_h20         |
| 103 |        30 | 66.9309  | 8.18113 | 5.74631  | 10.8622  |          0.1  | TSM+LLM-COT-RF_blend_ramp_bestval_h20         |
| 104 |         1 |  1.20225 | 1.09647 | 0.815198 |  1.43965 |          0.2  | TSM+LLM-COT-RF_blend_ramp_bestval_h20         |
| 105 |         5 |  4.64984 | 2.15635 | 1.52953  |  2.63278 |          0.2  | TSM+LLM-COT-RF_blend_ramp_bestval_h20         |
| 106 |        20 | 32.796   | 5.72678 | 3.7546   |  6.77491 |          0.2  | TSM+LLM-COT-RF_blend_ramp_bestval_h20         |
| 107 |        30 | 66.8663  | 8.17718 | 5.73805  | 10.8471  |          0.2  | TSM+LLM-COT-RF_blend_ramp_bestval_h20         |
| 108 |         1 |  1.20174 | 1.09624 | 0.815102 |  1.44162 |          0.3  | TSM+LLM-COT-RF_blend_ramp_bestval_h20         |
| 109 |         5 |  4.68606 | 2.16473 | 1.53191  |  2.63947 |          0.3  | TSM+LLM-COT-RF_blend_ramp_bestval_h20         |
| 110 |        20 | 32.7831  | 5.72566 | 3.76438  |  6.79147 |          0.3  | TSM+LLM-COT-RF_blend_ramp_bestval_h20         |
| 111 |        30 | 66.8083  | 8.17363 | 5.73019  | 10.8328  |          0.3  | TSM+LLM-COT-RF_blend_ramp_bestval_h20         |
| 112 |         1 |  1.2169  | 1.10313 | 0.821034 |  1.44688 |          0.05 | TSM+LLM-COT-RF_blend_ramp_bestval_h30         |
| 113 |         5 |  4.61039 | 2.14718 | 1.52848  |  2.62716 |          0.05 | TSM+LLM-COT-RF_blend_ramp_bestval_h30         |
| 114 |        20 | 31.7638  | 5.63594 | 3.68058  |  6.64131 |          0.05 | TSM+LLM-COT-RF_blend_ramp_bestval_h30         |
| 115 |        30 | 64.2425  | 8.01514 | 5.61911  | 10.6217  |          0.05 | TSM+LLM-COT-RF_blend_ramp_bestval_h30         |
| 116 |         1 |  1.21061 | 1.10028 | 0.818982 |  1.44418 |          0.1  | TSM+LLM-COT-RF_blend_ramp_bestval_h30         |
| 117 |         5 |  4.62259 | 2.15002 | 1.52847  |  2.62806 |          0.1  | TSM+LLM-COT-RF_blend_ramp_bestval_h30         |
| 118 |        20 | 31.7507  | 5.63477 | 3.68402  |  6.64688 |          0.1  | TSM+LLM-COT-RF_blend_ramp_bestval_h30         |
| 119 |        30 | 64.2176  | 8.01359 | 5.61501  | 10.6143  |          0.1  | TSM+LLM-COT-RF_blend_ramp_bestval_h30         |
| 120 |         1 |  1.20334 | 1.09697 | 0.815556 |  1.4402  |          0.2  | TSM+LLM-COT-RF_blend_ramp_bestval_h30         |
| 121 |         5 |  4.65447 | 2.15742 | 1.53031  |  2.63343 |          0.2  | TSM+LLM-COT-RF_blend_ramp_bestval_h30         |
| 122 |        20 | 31.7305  | 5.63299 | 3.69321  |  6.66249 |          0.2  | TSM+LLM-COT-RF_blend_ramp_bestval_h30         |
| 123 |        30 | 64.1724  | 8.01077 | 5.60681  | 10.5995  |          0.2  | TSM+LLM-COT-RF_blend_ramp_bestval_h30         |
| 124 |         1 |  1.20317 | 1.09689 | 0.815499 |  1.44219 |          0.3  | TSM+LLM-COT-RF_blend_ramp_bestval_h30         |
| 125 |         5 |  4.6963  | 2.1671  | 1.53358  |  2.64159 |          0.3  | TSM+LLM-COT-RF_blend_ramp_bestval_h30         |
| 126 |        20 | 31.7187  | 5.63194 | 3.7025   |  6.67827 |          0.3  | TSM+LLM-COT-RF_blend_ramp_bestval_h30         |
| 127 |        30 | 64.1338  | 8.00836 | 5.59902  | 10.5854  |          0.3  | TSM+LLM-COT-RF_blend_ramp_bestval_h30         |

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
- **TSM vs naive**: TSM MSE is 0.7x the naive baseline on average.
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


