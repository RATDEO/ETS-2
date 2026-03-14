# EU ETS Futures Forecasting with TSM + LLM Refinement

**Authors**: Auto-generated

**Generated**: 2026-03-09 00:25:09

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

|    |   horizon |      mse |    rmse |      mae |     mape | model                                                |
|---:|----------:|---------:|--------:|---------:|---------:|:-----------------------------------------------------|
|  0 |         1 |  1.06139 | 1.03024 | 0.784861 |  1.39543 | naive_persistence                                    |
|  1 |         5 |  5.2119  | 2.28296 | 1.67     |  2.90235 | naive_persistence                                    |
|  2 |        20 | 44.2759  | 6.65401 | 4.40799  |  7.94956 | naive_persistence                                    |
|  3 |        30 | 79.9881  | 8.94361 | 6.37986  | 12.0727  | naive_persistence                                    |
|  4 |         1 |  5.16853 | 2.27344 | 1.65132  |  2.90012 | seasonal_naive                                       |
|  5 |         5 |  5.2119  | 2.28296 | 1.67     |  2.90235 | seasonal_naive                                       |
|  6 |        20 | 44.2759  | 6.65401 | 4.40799  |  7.94956 | seasonal_naive                                       |
|  7 |        30 | 79.9881  | 8.94361 | 6.37986  | 12.0727  | seasonal_naive                                       |
|  8 |         1 |  1.06154 | 1.03031 | 0.781283 |  1.38304 | linear_ridge                                         |
|  9 |         5 |  4.59363 | 2.14328 | 1.56632  |  2.70398 | linear_ridge                                         |
| 10 |        20 | 30.1544  | 5.4913  | 3.32793  |  6.07247 | linear_ridge                                         |
| 11 |        30 | 56.9094  | 7.54383 | 4.78768  |  9.18917 | linear_ridge                                         |
| 12 |         1 |  1.1541  | 1.07429 | 0.830618 |  1.48473 | linear_lasso                                         |
| 13 |         5 |  4.73942 | 2.17702 | 1.61188  |  2.79853 | linear_lasso                                         |
| 14 |        20 | 32.1131  | 5.66684 | 3.43547  |  6.16613 | linear_lasso                                         |
| 15 |        30 | 62.0974  | 7.88019 | 5.23504  |  9.89649 | linear_lasso                                         |
| 16 |         1 |  1.22496 | 1.10678 | 0.823182 |  1.44975 | tsm                                                  |
| 17 |         5 |  4.58705 | 2.14174 | 1.52805  |  2.62472 | tsm                                                  |
| 18 |        20 | 29.8651  | 5.46489 | 3.58359  |  6.46353 | tsm                                                  |
| 19 |        30 | 59.4805  | 7.71236 | 5.39684  | 10.1968  | tsm                                                  |
| 20 |         1 |  1.22496 | 1.10678 | 0.823182 |  1.44975 | TSM+LLM-COT-RF-HDELTA                                |
| 21 |         5 |  4.56152 | 2.13577 | 1.52623  |  2.62123 | TSM+LLM-COT-RF-HDELTA                                |
| 22 |        20 | 29.6716  | 5.44717 | 3.5757   |  6.45132 | TSM+LLM-COT-RF-HDELTA                                |
| 23 |        30 | 59.1243  | 7.68923 | 5.36417  | 10.1379  | TSM+LLM-COT-RF-HDELTA                                |
| 24 |         1 |  1.22496 | 1.10678 | 0.823182 |  1.44975 | TSM+LLM-COT-RF-HDELTA_rulegate_bestval               |
| 25 |         5 |  4.56799 | 2.13729 | 1.5286   |  2.6256  | TSM+LLM-COT-RF-HDELTA_rulegate_bestval               |
| 26 |        20 | 29.7389  | 5.45334 | 3.57562  |  6.45143 | TSM+LLM-COT-RF-HDELTA_rulegate_bestval               |
| 27 |        30 | 59.3272  | 7.70241 | 5.38097  | 10.1688  | TSM+LLM-COT-RF-HDELTA_rulegate_bestval               |
| 28 |         1 |  1.22496 | 1.10678 | 0.823182 |  1.44975 | TSM+LLM-COT-RF-HDELTA_blend_ramp_bestval_path        |
| 29 |         5 |  4.58705 | 2.14174 | 1.52805  |  2.62472 | TSM+LLM-COT-RF-HDELTA_blend_ramp_bestval_path        |
| 30 |        20 | 29.8651  | 5.46489 | 3.58359  |  6.46353 | TSM+LLM-COT-RF-HDELTA_blend_ramp_bestval_path        |
| 31 |        30 | 59.4805  | 7.71236 | 5.39684  | 10.1968  | TSM+LLM-COT-RF-HDELTA_blend_ramp_bestval_path        |
| 32 |         1 |  1.22496 | 1.10678 | 0.823182 |  1.44975 | TSM+LLM-COT-RF-HDELTA_blend_ramp_bestval_path_h1base |
| 33 |         5 |  4.58705 | 2.14174 | 1.52805  |  2.62472 | TSM+LLM-COT-RF-HDELTA_blend_ramp_bestval_path_h1base |
| 34 |        20 | 29.8651  | 5.46489 | 3.58359  |  6.46353 | TSM+LLM-COT-RF-HDELTA_blend_ramp_bestval_path_h1base |
| 35 |        30 | 59.4805  | 7.71236 | 5.39684  | 10.1968  | TSM+LLM-COT-RF-HDELTA_blend_ramp_bestval_path_h1base |
| 36 |         1 |  1.22496 | 1.10678 | 0.823182 |  1.44975 | TSM+LLM-COT-RF-HDELTA_blend_ramp_bestval_h1          |
| 37 |         5 |  4.58705 | 2.14174 | 1.52805  |  2.62472 | TSM+LLM-COT-RF-HDELTA_blend_ramp_bestval_h1          |
| 38 |        20 | 29.8651  | 5.46489 | 3.58359  |  6.46353 | TSM+LLM-COT-RF-HDELTA_blend_ramp_bestval_h1          |
| 39 |        30 | 59.4805  | 7.71236 | 5.39684  | 10.1968  | TSM+LLM-COT-RF-HDELTA_blend_ramp_bestval_h1          |
| 40 |         1 |  1.22496 | 1.10678 | 0.823182 |  1.44975 | TSM+LLM-COT-RF-HDELTA_blend_ramp_bestval_h5          |
| 41 |         5 |  4.58705 | 2.14174 | 1.52805  |  2.62472 | TSM+LLM-COT-RF-HDELTA_blend_ramp_bestval_h5          |
| 42 |        20 | 29.8651  | 5.46489 | 3.58359  |  6.46353 | TSM+LLM-COT-RF-HDELTA_blend_ramp_bestval_h5          |
| 43 |        30 | 59.4805  | 7.71236 | 5.39684  | 10.1968  | TSM+LLM-COT-RF-HDELTA_blend_ramp_bestval_h5          |
| 44 |         1 |  1.22496 | 1.10678 | 0.823182 |  1.44975 | TSM+LLM-COT-RF-HDELTA_blend_ramp_bestval_h20         |
| 45 |         5 |  4.58705 | 2.14174 | 1.52805  |  2.62472 | TSM+LLM-COT-RF-HDELTA_blend_ramp_bestval_h20         |
| 46 |        20 | 29.8651  | 5.46489 | 3.58359  |  6.46353 | TSM+LLM-COT-RF-HDELTA_blend_ramp_bestval_h20         |
| 47 |        30 | 59.4805  | 7.71236 | 5.39684  | 10.1968  | TSM+LLM-COT-RF-HDELTA_blend_ramp_bestval_h20         |
| 48 |         1 |  1.22496 | 1.10678 | 0.823182 |  1.44975 | TSM+LLM-COT-RF-HDELTA_blend_ramp_bestval_h30         |
| 49 |         5 |  4.58705 | 2.14174 | 1.52805  |  2.62472 | TSM+LLM-COT-RF-HDELTA_blend_ramp_bestval_h30         |
| 50 |        20 | 29.8651  | 5.46489 | 3.58359  |  6.46353 | TSM+LLM-COT-RF-HDELTA_blend_ramp_bestval_h30         |
| 51 |        30 | 59.4805  | 7.71236 | 5.39684  | 10.1968  | TSM+LLM-COT-RF-HDELTA_blend_ramp_bestval_h30         |
| 52 |         1 |  1.06139 | 1.03024 | 0.784861 |  1.39543 | naive_persistence_llm_subset                         |
| 53 |         5 |  5.2119  | 2.28296 | 1.67     |  2.90235 | naive_persistence_llm_subset                         |
| 54 |        20 | 44.2759  | 6.65401 | 4.40799  |  7.94956 | naive_persistence_llm_subset                         |
| 55 |        30 | 79.9881  | 8.94361 | 6.37986  | 12.0727  | naive_persistence_llm_subset                         |
| 56 |         1 |  5.16853 | 2.27344 | 1.65132  |  2.90012 | seasonal_naive_llm_subset                            |
| 57 |         5 |  5.2119  | 2.28296 | 1.67     |  2.90235 | seasonal_naive_llm_subset                            |
| 58 |        20 | 44.2759  | 6.65401 | 4.40799  |  7.94956 | seasonal_naive_llm_subset                            |
| 59 |        30 | 79.9881  | 8.94361 | 6.37986  | 12.0727  | seasonal_naive_llm_subset                            |
| 60 |         1 |  1.06154 | 1.03031 | 0.781283 |  1.38304 | linear_ridge_llm_subset                              |
| 61 |         5 |  4.59363 | 2.14328 | 1.56632  |  2.70398 | linear_ridge_llm_subset                              |
| 62 |        20 | 30.1544  | 5.4913  | 3.32793  |  6.07247 | linear_ridge_llm_subset                              |
| 63 |        30 | 56.9094  | 7.54383 | 4.78768  |  9.18917 | linear_ridge_llm_subset                              |
| 64 |         1 |  1.1541  | 1.07429 | 0.830618 |  1.48473 | linear_lasso_llm_subset                              |
| 65 |         5 |  4.73942 | 2.17702 | 1.61188  |  2.79853 | linear_lasso_llm_subset                              |
| 66 |        20 | 32.1131  | 5.66684 | 3.43547  |  6.16613 | linear_lasso_llm_subset                              |
| 67 |        30 | 62.0974  | 7.88019 | 5.23504  |  9.89649 | linear_lasso_llm_subset                              |
| 68 |         1 |  1.22496 | 1.10678 | 0.823182 |  1.44975 | tsm_llm_subset                                       |
| 69 |         5 |  4.58705 | 2.14174 | 1.52805  |  2.62472 | tsm_llm_subset                                       |
| 70 |        20 | 29.8651  | 5.46489 | 3.58359  |  6.46353 | tsm_llm_subset                                       |
| 71 |        30 | 59.4805  | 7.71236 | 5.39684  | 10.1968  | tsm_llm_subset                                       |

### 4.2 Trend Classification Accuracy

Table 2 shows the accuracy of predicting price direction (up/flat/down).

|    |   horizon |   accuracy |   accuracy_up |   accuracy_down |   accuracy_flat |   n_up |   n_down |   n_flat | model                                                |
|---:|----------:|-----------:|--------------:|----------------:|----------------:|-------:|---------:|---------:|:-----------------------------------------------------|
|  0 |         1 |  0.395833  |     0         |       0         |       1         |     52 |       35 |       57 | naive_persistence                                    |
|  1 |         5 |  0.236111  |     0         |       0         |       1         |     70 |       40 |       34 | naive_persistence                                    |
|  2 |        20 |  0.118056  |     0         |       0         |       1         |     90 |       37 |       17 | naive_persistence                                    |
|  3 |        30 |  0.0763889 |     0         |       0         |       1         |    100 |       33 |       11 | naive_persistence                                    |
|  4 |         1 |  0.375     |     0.365385  |       0.628571  |       0.22807   |     52 |       35 |       57 | seasonal_naive                                       |
|  5 |         5 |  0.236111  |     0         |       0         |       1         |     70 |       40 |       34 | seasonal_naive                                       |
|  6 |        20 |  0.118056  |     0         |       0         |       1         |     90 |       37 |       17 | seasonal_naive                                       |
|  7 |        30 |  0.0763889 |     0         |       0         |       1         |    100 |       33 |       11 | seasonal_naive                                       |
|  8 |         1 |  0.430556  |     0.0576923 |       0.0857143 |       0.982456  |     52 |       35 |       57 | linear_ridge                                         |
|  9 |         5 |  0.451389  |     0.514286  |       0.125     |       0.705882  |     70 |       40 |       34 | linear_ridge                                         |
| 10 |        20 |  0.708333  |     0.955556  |       0.351351  |       0.176471  |     90 |       37 |       17 | linear_ridge                                         |
| 11 |        30 |  0.833333  |     0.99      |       0.606061  |       0.0909091 |    100 |       33 |       11 | linear_ridge                                         |
| 12 |         1 |  0.423611  |     0.288462  |       0.114286  |       0.736842  |     52 |       35 |       57 | linear_lasso                                         |
| 13 |         5 |  0.388889  |     0.414286  |       0.15      |       0.617647  |     70 |       40 |       34 | linear_lasso                                         |
| 14 |        20 |  0.708333  |     0.922222  |       0.459459  |       0.117647  |     90 |       37 |       17 | linear_lasso                                         |
| 15 |        30 |  0.736111  |     0.75      |       0.757576  |       0.545455  |    100 |       33 |       11 | linear_lasso                                         |
| 16 |         1 |  0.409722  |     0.192308  |       0.114286  |       0.789474  |     52 |       35 |       57 | tsm                                                  |
| 17 |         5 |  0.451389  |     0.428571  |       0.4       |       0.558824  |     70 |       40 |       34 | tsm                                                  |
| 18 |        20 |  0.534722  |     0.566667  |       0.567568  |       0.294118  |     90 |       37 |       17 | tsm                                                  |
| 19 |        30 |  0.618056  |     0.59      |       0.818182  |       0.272727  |    100 |       33 |       11 | tsm                                                  |
| 20 |         1 |  0.409722  |     0.192308  |       0.114286  |       0.789474  |     52 |       35 |       57 | TSM+LLM-COT-RF-HDELTA                                |
| 21 |         5 |  0.451389  |     0.428571  |       0.4       |       0.558824  |     70 |       40 |       34 | TSM+LLM-COT-RF-HDELTA                                |
| 22 |        20 |  0.541667  |     0.577778  |       0.567568  |       0.294118  |     90 |       37 |       17 | TSM+LLM-COT-RF-HDELTA                                |
| 23 |        30 |  0.638889  |     0.63      |       0.818182  |       0.181818  |    100 |       33 |       11 | TSM+LLM-COT-RF-HDELTA                                |
| 24 |         1 |  0.409722  |     0.192308  |       0.114286  |       0.789474  |     52 |       35 |       57 | TSM+LLM-COT-RF-HDELTA_rulegate_bestval               |
| 25 |         5 |  0.451389  |     0.428571  |       0.4       |       0.558824  |     70 |       40 |       34 | TSM+LLM-COT-RF-HDELTA_rulegate_bestval               |
| 26 |        20 |  0.541667  |     0.577778  |       0.567568  |       0.294118  |     90 |       37 |       17 | TSM+LLM-COT-RF-HDELTA_rulegate_bestval               |
| 27 |        30 |  0.638889  |     0.62      |       0.818182  |       0.272727  |    100 |       33 |       11 | TSM+LLM-COT-RF-HDELTA_rulegate_bestval               |
| 28 |         1 |  0.409722  |     0.192308  |       0.114286  |       0.789474  |     52 |       35 |       57 | TSM+LLM-COT-RF-HDELTA_blend_ramp_bestval_path        |
| 29 |         5 |  0.451389  |     0.428571  |       0.4       |       0.558824  |     70 |       40 |       34 | TSM+LLM-COT-RF-HDELTA_blend_ramp_bestval_path        |
| 30 |        20 |  0.534722  |     0.566667  |       0.567568  |       0.294118  |     90 |       37 |       17 | TSM+LLM-COT-RF-HDELTA_blend_ramp_bestval_path        |
| 31 |        30 |  0.618056  |     0.59      |       0.818182  |       0.272727  |    100 |       33 |       11 | TSM+LLM-COT-RF-HDELTA_blend_ramp_bestval_path        |
| 32 |         1 |  0.409722  |     0.192308  |       0.114286  |       0.789474  |     52 |       35 |       57 | TSM+LLM-COT-RF-HDELTA_blend_ramp_bestval_path_h1base |
| 33 |         5 |  0.451389  |     0.428571  |       0.4       |       0.558824  |     70 |       40 |       34 | TSM+LLM-COT-RF-HDELTA_blend_ramp_bestval_path_h1base |
| 34 |        20 |  0.534722  |     0.566667  |       0.567568  |       0.294118  |     90 |       37 |       17 | TSM+LLM-COT-RF-HDELTA_blend_ramp_bestval_path_h1base |
| 35 |        30 |  0.618056  |     0.59      |       0.818182  |       0.272727  |    100 |       33 |       11 | TSM+LLM-COT-RF-HDELTA_blend_ramp_bestval_path_h1base |
| 36 |         1 |  0.409722  |     0.192308  |       0.114286  |       0.789474  |     52 |       35 |       57 | TSM+LLM-COT-RF-HDELTA_blend_ramp_bestval_h1          |
| 37 |         5 |  0.451389  |     0.428571  |       0.4       |       0.558824  |     70 |       40 |       34 | TSM+LLM-COT-RF-HDELTA_blend_ramp_bestval_h1          |
| 38 |        20 |  0.534722  |     0.566667  |       0.567568  |       0.294118  |     90 |       37 |       17 | TSM+LLM-COT-RF-HDELTA_blend_ramp_bestval_h1          |
| 39 |        30 |  0.618056  |     0.59      |       0.818182  |       0.272727  |    100 |       33 |       11 | TSM+LLM-COT-RF-HDELTA_blend_ramp_bestval_h1          |
| 40 |         1 |  0.409722  |     0.192308  |       0.114286  |       0.789474  |     52 |       35 |       57 | TSM+LLM-COT-RF-HDELTA_blend_ramp_bestval_h5          |
| 41 |         5 |  0.451389  |     0.428571  |       0.4       |       0.558824  |     70 |       40 |       34 | TSM+LLM-COT-RF-HDELTA_blend_ramp_bestval_h5          |
| 42 |        20 |  0.534722  |     0.566667  |       0.567568  |       0.294118  |     90 |       37 |       17 | TSM+LLM-COT-RF-HDELTA_blend_ramp_bestval_h5          |
| 43 |        30 |  0.618056  |     0.59      |       0.818182  |       0.272727  |    100 |       33 |       11 | TSM+LLM-COT-RF-HDELTA_blend_ramp_bestval_h5          |
| 44 |         1 |  0.409722  |     0.192308  |       0.114286  |       0.789474  |     52 |       35 |       57 | TSM+LLM-COT-RF-HDELTA_blend_ramp_bestval_h20         |
| 45 |         5 |  0.451389  |     0.428571  |       0.4       |       0.558824  |     70 |       40 |       34 | TSM+LLM-COT-RF-HDELTA_blend_ramp_bestval_h20         |
| 46 |        20 |  0.534722  |     0.566667  |       0.567568  |       0.294118  |     90 |       37 |       17 | TSM+LLM-COT-RF-HDELTA_blend_ramp_bestval_h20         |
| 47 |        30 |  0.618056  |     0.59      |       0.818182  |       0.272727  |    100 |       33 |       11 | TSM+LLM-COT-RF-HDELTA_blend_ramp_bestval_h20         |
| 48 |         1 |  0.409722  |     0.192308  |       0.114286  |       0.789474  |     52 |       35 |       57 | TSM+LLM-COT-RF-HDELTA_blend_ramp_bestval_h30         |
| 49 |         5 |  0.451389  |     0.428571  |       0.4       |       0.558824  |     70 |       40 |       34 | TSM+LLM-COT-RF-HDELTA_blend_ramp_bestval_h30         |
| 50 |        20 |  0.534722  |     0.566667  |       0.567568  |       0.294118  |     90 |       37 |       17 | TSM+LLM-COT-RF-HDELTA_blend_ramp_bestval_h30         |
| 51 |        30 |  0.618056  |     0.59      |       0.818182  |       0.272727  |    100 |       33 |       11 | TSM+LLM-COT-RF-HDELTA_blend_ramp_bestval_h30         |
| 52 |         1 |  0.395833  |     0         |       0         |       1         |     52 |       35 |       57 | naive_persistence_llm_subset                         |
| 53 |         5 |  0.236111  |     0         |       0         |       1         |     70 |       40 |       34 | naive_persistence_llm_subset                         |
| 54 |        20 |  0.118056  |     0         |       0         |       1         |     90 |       37 |       17 | naive_persistence_llm_subset                         |
| 55 |        30 |  0.0763889 |     0         |       0         |       1         |    100 |       33 |       11 | naive_persistence_llm_subset                         |
| 56 |         1 |  0.375     |     0.365385  |       0.628571  |       0.22807   |     52 |       35 |       57 | seasonal_naive_llm_subset                            |
| 57 |         5 |  0.236111  |     0         |       0         |       1         |     70 |       40 |       34 | seasonal_naive_llm_subset                            |
| 58 |        20 |  0.118056  |     0         |       0         |       1         |     90 |       37 |       17 | seasonal_naive_llm_subset                            |
| 59 |        30 |  0.0763889 |     0         |       0         |       1         |    100 |       33 |       11 | seasonal_naive_llm_subset                            |
| 60 |         1 |  0.430556  |     0.0576923 |       0.0857143 |       0.982456  |     52 |       35 |       57 | linear_ridge_llm_subset                              |
| 61 |         5 |  0.451389  |     0.514286  |       0.125     |       0.705882  |     70 |       40 |       34 | linear_ridge_llm_subset                              |
| 62 |        20 |  0.708333  |     0.955556  |       0.351351  |       0.176471  |     90 |       37 |       17 | linear_ridge_llm_subset                              |
| 63 |        30 |  0.833333  |     0.99      |       0.606061  |       0.0909091 |    100 |       33 |       11 | linear_ridge_llm_subset                              |
| 64 |         1 |  0.423611  |     0.288462  |       0.114286  |       0.736842  |     52 |       35 |       57 | linear_lasso_llm_subset                              |
| 65 |         5 |  0.388889  |     0.414286  |       0.15      |       0.617647  |     70 |       40 |       34 | linear_lasso_llm_subset                              |
| 66 |        20 |  0.708333  |     0.922222  |       0.459459  |       0.117647  |     90 |       37 |       17 | linear_lasso_llm_subset                              |
| 67 |        30 |  0.736111  |     0.75      |       0.757576  |       0.545455  |    100 |       33 |       11 | linear_lasso_llm_subset                              |
| 68 |         1 |  0.409722  |     0.192308  |       0.114286  |       0.789474  |     52 |       35 |       57 | tsm_llm_subset                                       |
| 69 |         5 |  0.451389  |     0.428571  |       0.4       |       0.558824  |     70 |       40 |       34 | tsm_llm_subset                                       |
| 70 |        20 |  0.534722  |     0.566667  |       0.567568  |       0.294118  |     90 |       37 |       17 | tsm_llm_subset                                       |
| 71 |        30 |  0.618056  |     0.59      |       0.818182  |       0.272727  |    100 |       33 |       11 | tsm_llm_subset                                       |

### 4.3 Statistical Significance

Table 3 reports the results of significance tests comparing methods.

|    |   horizon | model                                                | baseline                     |   mean_error_model |   mean_error_baseline |   improvement_pct |   t_statistic |    t_pvalue | t_significant   |   wilcoxon_statistic |   wilcoxon_pvalue | wilcoxon_significant   |   dm_statistic |   dm_pvalue | dm_significant   | model_better   |
|---:|----------:|:-----------------------------------------------------|:-----------------------------|-------------------:|----------------------:|------------------:|--------------:|------------:|:----------------|---------------------:|------------------:|:-----------------------|---------------:|------------:|:-----------------|:---------------|
|  0 |         1 | seasonal_naive_llm_subset                            | naive_persistence_llm_subset |            5.16853 |               1.06139 |    -386.957       |    5.07044    | 1.21001e-06 | True            |                 1834 |       1.45104e-11 | True                   |     5.07044    | 3.96893e-07 | True             | False          |
|  1 |         5 | seasonal_naive_llm_subset                            | naive_persistence_llm_subset |            5.2119  |               5.2119  |       5.49269e-07 |    1.19107    | 0.235598    | False           |                 4258 |       0.556904    | False                  |     0.0263232  | 0.979       | False            | True           |
|  2 |        20 | seasonal_naive_llm_subset                            | naive_persistence_llm_subset |           44.2759  |              44.2759  |       2.22491e-06 |    0.67977    | 0.497749    | False           |                 4233 |       0.183904    | False                  |     0.216494   | 0.828603    | False            | True           |
|  3 |        30 | seasonal_naive_llm_subset                            | naive_persistence_llm_subset |           79.9881  |              79.9881  |       3.104e-06   |   -0.578827   | 0.563616    | False           |                 3978 |       0.082287    | False                  |    -0.245123   | 0.806361    | False            | True           |
|  4 |         1 | linear_ridge_llm_subset                              | naive_persistence_llm_subset |            1.06154 |               1.06139 |      -0.0135551   |    0.00242328 | 0.99807     | False           |                 5155 |       0.896859    | False                  |     0.00242328 | 0.998067    | False            | False          |
|  5 |         5 | linear_ridge_llm_subset                              | naive_persistence_llm_subset |            4.59363 |               5.2119  |      11.8626      |   -2.00294    | 0.0470741   | True            |                 3955 |       0.0116426   | True                   |    -1.15055    | 0.249918    | False            | True           |
|  6 |        20 | linear_ridge_llm_subset                              | naive_persistence_llm_subset |           30.1544  |              44.2759  |      31.8943      |   -5.67076    | 7.5947e-08  | True            |                 2425 |       2.4884e-08  | True                   |    -2.3465     | 0.0189507   | True             | True           |
|  7 |        30 | linear_ridge_llm_subset                              | naive_persistence_llm_subset |           56.9094  |              79.9881  |      28.8527      |   -7.86447    | 8.23678e-13 | True            |                  726 |       3.17675e-19 | True                   |    -1.91128    | 0.0559692   | False            | True           |
|  8 |         1 | linear_lasso_llm_subset                              | naive_persistence_llm_subset |            1.1541  |               1.06139 |      -8.73423     |    1.16037    | 0.24783     | False           |                 4553 |       0.183452    | False                  |     1.16037    | 0.245896    | False            | False          |
|  9 |         5 | linear_lasso_llm_subset                              | naive_persistence_llm_subset |            4.73942 |               5.2119  |       9.06537     |   -1.85969    | 0.0649833   | False           |                 4418 |       0.109725    | False                  |    -1.14841    | 0.250799    | False            | True           |
| 10 |        20 | linear_lasso_llm_subset                              | naive_persistence_llm_subset |           32.1131  |              44.2759  |      27.4704      |   -5.48329    | 1.83923e-07 | True            |                 1499 |       1.16409e-13 | True                   |    -2.64988    | 0.00805199  | True             | True           |
| 11 |        30 | linear_lasso_llm_subset                              | naive_persistence_llm_subset |           62.0974  |              79.9881  |      22.3667      |   -6.53209    | 1.0595e-09  | True            |                  443 |       1.62218e-21 | True                   |    -1.64973    | 0.0989982   | False            | True           |
| 12 |         1 | tsm_llm_subset                                       | naive_persistence_llm_subset |            1.22496 |               1.06139 |     -15.4104      |    1.3854     | 0.168087    | False           |                 5127 |       0.85286     | False                  |     1.3854     | 0.16593     | False            | False          |
| 13 |         5 | tsm_llm_subset                                       | naive_persistence_llm_subset |            4.58705 |               5.2119  |      11.989       |   -1.14818    | 0.25281     | False           |                 3573 |       0.0010212   | True                   |    -0.594844   | 0.551947    | False            | True           |
| 14 |        20 | tsm_llm_subset                                       | naive_persistence_llm_subset |           29.8651  |              44.2759  |      32.5477      |   -4.95125    | 2.04896e-06 | True            |                 2397 |       1.80292e-08 | True                   |    -1.72788    | 0.0840099   | False            | True           |
| 15 |        30 | tsm_llm_subset                                       | naive_persistence_llm_subset |           59.4805  |              79.9881  |      25.6383      |   -5.43666    | 2.28549e-07 | True            |                 1244 |       2.20288e-15 | True                   |    -1.29984    | 0.193657    | False            | True           |
| 16 |         1 | TSM+LLM-COT-RF-HDELTA                                | naive_persistence_llm_subset |            1.22496 |               1.06139 |     -15.4104      |    1.3854     | 0.168087    | False           |                 5127 |       0.85286     | False                  |     1.3854     | 0.16593     | False            | False          |
| 17 |         5 | TSM+LLM-COT-RF-HDELTA                                | naive_persistence_llm_subset |            4.56152 |               5.2119  |      12.4788      |   -1.17915    | 0.240296    | False           |                 3635 |       0.00157243  | True                   |    -0.608519   | 0.542843    | False            | True           |
| 18 |        20 | TSM+LLM-COT-RF-HDELTA                                | naive_persistence_llm_subset |           29.6716  |              44.2759  |      32.9847      |   -4.96622    | 1.9186e-06  | True            |                 2439 |       2.92011e-08 | True                   |    -1.73225    | 0.0832287   | False            | True           |
| 19 |        30 | TSM+LLM-COT-RF-HDELTA                                | naive_persistence_llm_subset |           59.1243  |              79.9881  |      26.0836      |   -5.49043    | 1.7789e-07  | True            |                 1235 |       1.90616e-15 | True                   |    -1.31155    | 0.189671    | False            | True           |
| 20 |         1 | TSM+LLM-COT-RF-HDELTA_rulegate_bestval               | naive_persistence_llm_subset |            1.22496 |               1.06139 |     -15.4104      |    1.3854     | 0.168087    | False           |                 5127 |       0.85286     | False                  |     1.3854     | 0.16593     | False            | False          |
| 21 |         5 | TSM+LLM-COT-RF-HDELTA_rulegate_bestval               | naive_persistence_llm_subset |            4.56799 |               5.2119  |      12.3546      |   -1.18599    | 0.237593    | False           |                 3624 |       0.00145803  | True                   |    -0.611068   | 0.541155    | False            | True           |
| 22 |        20 | TSM+LLM-COT-RF-HDELTA_rulegate_bestval               | naive_persistence_llm_subset |           29.7389  |              44.2759  |      32.8327      |   -4.9742     | 1.85248e-06 | True            |                 2423 |       2.43203e-08 | True                   |    -1.73471    | 0.0827916   | False            | True           |
| 23 |        30 | TSM+LLM-COT-RF-HDELTA_rulegate_bestval               | naive_persistence_llm_subset |           59.3272  |              79.9881  |      25.83        |   -5.4726     | 1.93335e-07 | True            |                 1234 |       1.87572e-15 | True                   |    -1.30637    | 0.191426    | False            | True           |
| 24 |         1 | TSM+LLM-COT-RF-HDELTA_blend_ramp_bestval_path        | naive_persistence_llm_subset |            1.22496 |               1.06139 |     -15.4104      |    1.3854     | 0.168087    | False           |                 5127 |       0.85286     | False                  |     1.3854     | 0.16593     | False            | False          |
| 25 |         5 | TSM+LLM-COT-RF-HDELTA_blend_ramp_bestval_path        | naive_persistence_llm_subset |            4.58705 |               5.2119  |      11.989       |   -1.14818    | 0.25281     | False           |                 3573 |       0.0010212   | True                   |    -0.594844   | 0.551947    | False            | True           |
| 26 |        20 | TSM+LLM-COT-RF-HDELTA_blend_ramp_bestval_path        | naive_persistence_llm_subset |           29.8651  |              44.2759  |      32.5477      |   -4.95125    | 2.04896e-06 | True            |                 2397 |       1.80292e-08 | True                   |    -1.72788    | 0.0840099   | False            | True           |
| 27 |        30 | TSM+LLM-COT-RF-HDELTA_blend_ramp_bestval_path        | naive_persistence_llm_subset |           59.4805  |              79.9881  |      25.6383      |   -5.43666    | 2.28549e-07 | True            |                 1244 |       2.20288e-15 | True                   |    -1.29984    | 0.193657    | False            | True           |
| 28 |         1 | TSM+LLM-COT-RF-HDELTA_blend_ramp_bestval_path_h1base | naive_persistence_llm_subset |            1.22496 |               1.06139 |     -15.4104      |    1.3854     | 0.168087    | False           |                 5127 |       0.85286     | False                  |     1.3854     | 0.16593     | False            | False          |
| 29 |         5 | TSM+LLM-COT-RF-HDELTA_blend_ramp_bestval_path_h1base | naive_persistence_llm_subset |            4.58705 |               5.2119  |      11.989       |   -1.14818    | 0.25281     | False           |                 3573 |       0.0010212   | True                   |    -0.594844   | 0.551947    | False            | True           |
| 30 |        20 | TSM+LLM-COT-RF-HDELTA_blend_ramp_bestval_path_h1base | naive_persistence_llm_subset |           29.8651  |              44.2759  |      32.5477      |   -4.95125    | 2.04896e-06 | True            |                 2397 |       1.80292e-08 | True                   |    -1.72788    | 0.0840099   | False            | True           |
| 31 |        30 | TSM+LLM-COT-RF-HDELTA_blend_ramp_bestval_path_h1base | naive_persistence_llm_subset |           59.4805  |              79.9881  |      25.6383      |   -5.43666    | 2.28549e-07 | True            |                 1244 |       2.20288e-15 | True                   |    -1.29984    | 0.193657    | False            | True           |
| 32 |         1 | TSM+LLM-COT-RF-HDELTA_blend_ramp_bestval_h1          | naive_persistence_llm_subset |            1.22496 |               1.06139 |     -15.4104      |    1.3854     | 0.168087    | False           |                 5127 |       0.85286     | False                  |     1.3854     | 0.16593     | False            | False          |
| 33 |         5 | TSM+LLM-COT-RF-HDELTA_blend_ramp_bestval_h1          | naive_persistence_llm_subset |            4.58705 |               5.2119  |      11.989       |   -1.14818    | 0.25281     | False           |                 3573 |       0.0010212   | True                   |    -0.594844   | 0.551947    | False            | True           |
| 34 |        20 | TSM+LLM-COT-RF-HDELTA_blend_ramp_bestval_h1          | naive_persistence_llm_subset |           29.8651  |              44.2759  |      32.5477      |   -4.95125    | 2.04896e-06 | True            |                 2397 |       1.80292e-08 | True                   |    -1.72788    | 0.0840099   | False            | True           |
| 35 |        30 | TSM+LLM-COT-RF-HDELTA_blend_ramp_bestval_h1          | naive_persistence_llm_subset |           59.4805  |              79.9881  |      25.6383      |   -5.43666    | 2.28549e-07 | True            |                 1244 |       2.20288e-15 | True                   |    -1.29984    | 0.193657    | False            | True           |
| 36 |         1 | TSM+LLM-COT-RF-HDELTA_blend_ramp_bestval_h5          | naive_persistence_llm_subset |            1.22496 |               1.06139 |     -15.4104      |    1.3854     | 0.168087    | False           |                 5127 |       0.85286     | False                  |     1.3854     | 0.16593     | False            | False          |
| 37 |         5 | TSM+LLM-COT-RF-HDELTA_blend_ramp_bestval_h5          | naive_persistence_llm_subset |            4.58705 |               5.2119  |      11.989       |   -1.14818    | 0.25281     | False           |                 3573 |       0.0010212   | True                   |    -0.594844   | 0.551947    | False            | True           |
| 38 |        20 | TSM+LLM-COT-RF-HDELTA_blend_ramp_bestval_h5          | naive_persistence_llm_subset |           29.8651  |              44.2759  |      32.5477      |   -4.95125    | 2.04896e-06 | True            |                 2397 |       1.80292e-08 | True                   |    -1.72788    | 0.0840099   | False            | True           |
| 39 |        30 | TSM+LLM-COT-RF-HDELTA_blend_ramp_bestval_h5          | naive_persistence_llm_subset |           59.4805  |              79.9881  |      25.6383      |   -5.43666    | 2.28549e-07 | True            |                 1244 |       2.20288e-15 | True                   |    -1.29984    | 0.193657    | False            | True           |
| 40 |         1 | TSM+LLM-COT-RF-HDELTA_blend_ramp_bestval_h20         | naive_persistence_llm_subset |            1.22496 |               1.06139 |     -15.4104      |    1.3854     | 0.168087    | False           |                 5127 |       0.85286     | False                  |     1.3854     | 0.16593     | False            | False          |
| 41 |         5 | TSM+LLM-COT-RF-HDELTA_blend_ramp_bestval_h20         | naive_persistence_llm_subset |            4.58705 |               5.2119  |      11.989       |   -1.14818    | 0.25281     | False           |                 3573 |       0.0010212   | True                   |    -0.594844   | 0.551947    | False            | True           |
| 42 |        20 | TSM+LLM-COT-RF-HDELTA_blend_ramp_bestval_h20         | naive_persistence_llm_subset |           29.8651  |              44.2759  |      32.5477      |   -4.95125    | 2.04896e-06 | True            |                 2397 |       1.80292e-08 | True                   |    -1.72788    | 0.0840099   | False            | True           |
| 43 |        30 | TSM+LLM-COT-RF-HDELTA_blend_ramp_bestval_h20         | naive_persistence_llm_subset |           59.4805  |              79.9881  |      25.6383      |   -5.43666    | 2.28549e-07 | True            |                 1244 |       2.20288e-15 | True                   |    -1.29984    | 0.193657    | False            | True           |
| 44 |         1 | TSM+LLM-COT-RF-HDELTA_blend_ramp_bestval_h30         | naive_persistence_llm_subset |            1.22496 |               1.06139 |     -15.4104      |    1.3854     | 0.168087    | False           |                 5127 |       0.85286     | False                  |     1.3854     | 0.16593     | False            | False          |
| 45 |         5 | TSM+LLM-COT-RF-HDELTA_blend_ramp_bestval_h30         | naive_persistence_llm_subset |            4.58705 |               5.2119  |      11.989       |   -1.14818    | 0.25281     | False           |                 3573 |       0.0010212   | True                   |    -0.594844   | 0.551947    | False            | True           |
| 46 |        20 | TSM+LLM-COT-RF-HDELTA_blend_ramp_bestval_h30         | naive_persistence_llm_subset |           29.8651  |              44.2759  |      32.5477      |   -4.95125    | 2.04896e-06 | True            |                 2397 |       1.80292e-08 | True                   |    -1.72788    | 0.0840099   | False            | True           |
| 47 |        30 | TSM+LLM-COT-RF-HDELTA_blend_ramp_bestval_h30         | naive_persistence_llm_subset |           59.4805  |              79.9881  |      25.6383      |   -5.43666    | 2.28549e-07 | True            |                 1244 |       2.20288e-15 | True                   |    -1.29984    | 0.193657    | False            | True           |


## 5. Robustness Analysis

### 5.1 Noise Injection Test

To assess robustness, we inject noise into forecast paths at levels of 5%, 10%, 20%, 
and 30% of the forecast standard deviation.

|     |   horizon |      mse |    rmse |      mae |     mape |   noise_level | model                                                |
|----:|----------:|---------:|--------:|---------:|---------:|--------------:|:-----------------------------------------------------|
|   0 |         1 |  1.21489 | 1.10222 | 0.820331 |  1.44583 |          0.05 | tsm                                                  |
|   1 |         5 |  4.59811 | 2.14432 | 1.52671  |  2.62331 |          0.05 | tsm                                                  |
|   2 |        20 | 29.8561  | 5.46407 | 3.58635  |  6.46801 |          0.05 | tsm                                                  |
|   3 |        30 | 59.4497  | 7.71036 | 5.3919   | 10.1879  |          0.05 | tsm                                                  |
|   4 |         1 |  1.20679 | 1.09854 | 0.817577 |  1.44208 |          0.1  | tsm                                                  |
|   5 |         5 |  4.61202 | 2.14756 | 1.52665  |  2.62418 |          0.1  | tsm                                                  |
|   6 |        20 | 29.8493  | 5.46345 | 3.58911  |  6.47248 |          0.1  | tsm                                                  |
|   7 |        30 | 59.4206  | 7.70847 | 5.38697  | 10.1789  |          0.1  | tsm                                                  |
|   8 |         1 |  1.19649 | 1.09384 | 0.812703 |  1.4359  |          0.2  | tsm                                                  |
|   9 |         5 |  4.6484  | 2.15601 | 1.52773  |  2.62835 |          0.2  | tsm                                                  |
|  10 |        20 | 29.8426  | 5.46284 | 3.59462  |  6.48143 |          0.2  | tsm                                                  |
|  11 |        30 | 59.3676  | 7.70504 | 5.37711  | 10.161   |          0.2  | tsm                                                  |
|  12 |         1 |  1.19405 | 1.09273 | 0.810592 |  1.43459 |          0.3  | tsm                                                  |
|  13 |         5 |  4.69618 | 2.16707 | 1.53203  |  2.63806 |          0.3  | tsm                                                  |
|  14 |        20 | 29.8449  | 5.46305 | 3.60096  |  6.49201 |          0.3  | tsm                                                  |
|  15 |        30 | 59.3218  | 7.70206 | 5.36768  | 10.1439  |          0.3  | tsm                                                  |
|  16 |         1 |  1.21514 | 1.10233 | 0.820471 |  1.44608 |          0.05 | TSM+LLM-COT-RF-HDELTA                                |
|  17 |         5 |  4.57189 | 2.1382  | 1.52513  |  2.62032 |          0.05 | TSM+LLM-COT-RF-HDELTA                                |
|  18 |        20 | 29.6611  | 5.4462  | 3.57859  |  6.45602 |          0.05 | TSM+LLM-COT-RF-HDELTA                                |
|  19 |        30 | 59.0935  | 7.68723 | 5.3591   | 10.1287  |          0.05 | TSM+LLM-COT-RF-HDELTA                                |
|  20 |         1 |  1.20736 | 1.0988  | 0.817864 |  1.4426  |          0.1  | TSM+LLM-COT-RF-HDELTA                                |
|  21 |         5 |  4.58519 | 2.14131 | 1.52403  |  2.61941 |          0.1  | TSM+LLM-COT-RF-HDELTA                                |
|  22 |        20 | 29.653   | 5.44546 | 3.58149  |  6.46072 |          0.1  | TSM+LLM-COT-RF-HDELTA                                |
|  23 |        30 | 59.0645  | 7.68534 | 5.35403  | 10.1195  |          0.1  | TSM+LLM-COT-RF-HDELTA                                |
|  24 |         1 |  1.19792 | 1.0945  | 0.813286 |  1.43696 |          0.2  | TSM+LLM-COT-RF-HDELTA                                |
|  25 |         5 |  4.62058 | 2.14955 | 1.52559  |  2.62472 |          0.2  | TSM+LLM-COT-RF-HDELTA                                |
|  26 |        20 | 29.6439  | 5.44462 | 3.58727  |  6.47011 |          0.2  | TSM+LLM-COT-RF-HDELTA                                |
|  27 |        30 | 59.0119  | 7.68192 | 5.34428  | 10.1018  |          0.2  | TSM+LLM-COT-RF-HDELTA                                |
|  28 |         1 |  1.19665 | 1.09391 | 0.81183  |  1.43679 |          0.3  | TSM+LLM-COT-RF-HDELTA                                |
|  29 |         5 |  4.66769 | 2.16048 | 1.52827  |  2.63205 |          0.3  | TSM+LLM-COT-RF-HDELTA                                |
|  30 |        20 | 29.6445  | 5.44467 | 3.59328  |  6.47995 |          0.3  | TSM+LLM-COT-RF-HDELTA                                |
|  31 |        30 | 58.9664  | 7.67896 | 5.33504  | 10.085   |          0.3  | TSM+LLM-COT-RF-HDELTA                                |
|  32 |         1 |  1.21497 | 1.10226 | 0.820349 |  1.44587 |          0.05 | TSM+LLM-COT-RF-HDELTA_rulegate_bestval               |
|  33 |         5 |  4.57857 | 2.13976 | 1.5275   |  2.62468 |          0.05 | TSM+LLM-COT-RF-HDELTA_rulegate_bestval               |
|  34 |        20 | 29.7288  | 5.45241 | 3.57841  |  6.45596 |          0.05 | TSM+LLM-COT-RF-HDELTA_rulegate_bestval               |
|  35 |        30 | 59.2963  | 7.70041 | 5.37602  | 10.1598  |          0.05 | TSM+LLM-COT-RF-HDELTA_rulegate_bestval               |
|  36 |         1 |  1.207   | 1.09863 | 0.817621 |  1.44218 |          0.1  | TSM+LLM-COT-RF-HDELTA_rulegate_bestval               |
|  37 |         5 |  4.59207 | 2.14291 | 1.52641  |  2.62377 |          0.1  | TSM+LLM-COT-RF-HDELTA_rulegate_bestval               |
|  38 |        20 | 29.7211  | 5.4517  | 3.5812   |  6.46049 |          0.1  | TSM+LLM-COT-RF-HDELTA_rulegate_bestval               |
|  39 |        30 | 59.2673  | 7.69852 | 5.37106  | 10.1508  |          0.1  | TSM+LLM-COT-RF-HDELTA_rulegate_bestval               |
|  40 |         1 |  1.1971  | 1.09412 | 0.8128   |  1.43611 |          0.2  | TSM+LLM-COT-RF-HDELTA_rulegate_bestval               |
|  41 |         5 |  4.6278  | 2.15123 | 1.52798  |  2.62908 |          0.2  | TSM+LLM-COT-RF-HDELTA_rulegate_bestval               |
|  42 |        20 | 29.7126  | 5.45093 | 3.58677  |  6.46956 |          0.2  | TSM+LLM-COT-RF-HDELTA_rulegate_bestval               |
|  43 |        30 | 59.2145  | 7.6951  | 5.36154  | 10.1335  |          0.2  | TSM+LLM-COT-RF-HDELTA_rulegate_bestval               |
|  44 |         1 |  1.19526 | 1.09328 | 0.811101 |  1.43553 |          0.3  | TSM+LLM-COT-RF-HDELTA_rulegate_bestval               |
|  45 |         5 |  4.67518 | 2.16222 | 1.53068  |  2.6364  |          0.3  | TSM+LLM-COT-RF-HDELTA_rulegate_bestval               |
|  46 |        20 | 29.7135  | 5.45101 | 3.59257  |  6.47907 |          0.3  | TSM+LLM-COT-RF-HDELTA_rulegate_bestval               |
|  47 |        30 | 59.169   | 7.69214 | 5.35252  | 10.1172  |          0.3  | TSM+LLM-COT-RF-HDELTA_rulegate_bestval               |
|  48 |         1 |  1.21489 | 1.10222 | 0.820331 |  1.44583 |          0.05 | TSM+LLM-COT-RF-HDELTA_blend_ramp_bestval_path        |
|  49 |         5 |  4.59811 | 2.14432 | 1.52671  |  2.62331 |          0.05 | TSM+LLM-COT-RF-HDELTA_blend_ramp_bestval_path        |
|  50 |        20 | 29.8561  | 5.46407 | 3.58635  |  6.46801 |          0.05 | TSM+LLM-COT-RF-HDELTA_blend_ramp_bestval_path        |
|  51 |        30 | 59.4497  | 7.71036 | 5.3919   | 10.1879  |          0.05 | TSM+LLM-COT-RF-HDELTA_blend_ramp_bestval_path        |
|  52 |         1 |  1.20679 | 1.09854 | 0.817577 |  1.44208 |          0.1  | TSM+LLM-COT-RF-HDELTA_blend_ramp_bestval_path        |
|  53 |         5 |  4.61202 | 2.14756 | 1.52665  |  2.62418 |          0.1  | TSM+LLM-COT-RF-HDELTA_blend_ramp_bestval_path        |
|  54 |        20 | 29.8493  | 5.46345 | 3.58911  |  6.47248 |          0.1  | TSM+LLM-COT-RF-HDELTA_blend_ramp_bestval_path        |
|  55 |        30 | 59.4206  | 7.70847 | 5.38697  | 10.1789  |          0.1  | TSM+LLM-COT-RF-HDELTA_blend_ramp_bestval_path        |
|  56 |         1 |  1.19649 | 1.09384 | 0.812703 |  1.4359  |          0.2  | TSM+LLM-COT-RF-HDELTA_blend_ramp_bestval_path        |
|  57 |         5 |  4.6484  | 2.15601 | 1.52773  |  2.62835 |          0.2  | TSM+LLM-COT-RF-HDELTA_blend_ramp_bestval_path        |
|  58 |        20 | 29.8426  | 5.46284 | 3.59462  |  6.48143 |          0.2  | TSM+LLM-COT-RF-HDELTA_blend_ramp_bestval_path        |
|  59 |        30 | 59.3676  | 7.70504 | 5.37711  | 10.161   |          0.2  | TSM+LLM-COT-RF-HDELTA_blend_ramp_bestval_path        |
|  60 |         1 |  1.19405 | 1.09273 | 0.810592 |  1.43459 |          0.3  | TSM+LLM-COT-RF-HDELTA_blend_ramp_bestval_path        |
|  61 |         5 |  4.69618 | 2.16707 | 1.53203  |  2.63806 |          0.3  | TSM+LLM-COT-RF-HDELTA_blend_ramp_bestval_path        |
|  62 |        20 | 29.8449  | 5.46305 | 3.60096  |  6.49201 |          0.3  | TSM+LLM-COT-RF-HDELTA_blend_ramp_bestval_path        |
|  63 |        30 | 59.3218  | 7.70206 | 5.36768  | 10.1439  |          0.3  | TSM+LLM-COT-RF-HDELTA_blend_ramp_bestval_path        |
|  64 |         1 |  1.21489 | 1.10222 | 0.820331 |  1.44583 |          0.05 | TSM+LLM-COT-RF-HDELTA_blend_ramp_bestval_path_h1base |
|  65 |         5 |  4.59811 | 2.14432 | 1.52671  |  2.62331 |          0.05 | TSM+LLM-COT-RF-HDELTA_blend_ramp_bestval_path_h1base |
|  66 |        20 | 29.8561  | 5.46407 | 3.58635  |  6.46801 |          0.05 | TSM+LLM-COT-RF-HDELTA_blend_ramp_bestval_path_h1base |
|  67 |        30 | 59.4497  | 7.71036 | 5.3919   | 10.1879  |          0.05 | TSM+LLM-COT-RF-HDELTA_blend_ramp_bestval_path_h1base |
|  68 |         1 |  1.20679 | 1.09854 | 0.817577 |  1.44208 |          0.1  | TSM+LLM-COT-RF-HDELTA_blend_ramp_bestval_path_h1base |
|  69 |         5 |  4.61202 | 2.14756 | 1.52665  |  2.62418 |          0.1  | TSM+LLM-COT-RF-HDELTA_blend_ramp_bestval_path_h1base |
|  70 |        20 | 29.8493  | 5.46345 | 3.58911  |  6.47248 |          0.1  | TSM+LLM-COT-RF-HDELTA_blend_ramp_bestval_path_h1base |
|  71 |        30 | 59.4206  | 7.70847 | 5.38697  | 10.1789  |          0.1  | TSM+LLM-COT-RF-HDELTA_blend_ramp_bestval_path_h1base |
|  72 |         1 |  1.19649 | 1.09384 | 0.812703 |  1.4359  |          0.2  | TSM+LLM-COT-RF-HDELTA_blend_ramp_bestval_path_h1base |
|  73 |         5 |  4.6484  | 2.15601 | 1.52773  |  2.62835 |          0.2  | TSM+LLM-COT-RF-HDELTA_blend_ramp_bestval_path_h1base |
|  74 |        20 | 29.8426  | 5.46284 | 3.59462  |  6.48143 |          0.2  | TSM+LLM-COT-RF-HDELTA_blend_ramp_bestval_path_h1base |
|  75 |        30 | 59.3676  | 7.70504 | 5.37711  | 10.161   |          0.2  | TSM+LLM-COT-RF-HDELTA_blend_ramp_bestval_path_h1base |
|  76 |         1 |  1.19405 | 1.09273 | 0.810592 |  1.43459 |          0.3  | TSM+LLM-COT-RF-HDELTA_blend_ramp_bestval_path_h1base |
|  77 |         5 |  4.69618 | 2.16707 | 1.53203  |  2.63806 |          0.3  | TSM+LLM-COT-RF-HDELTA_blend_ramp_bestval_path_h1base |
|  78 |        20 | 29.8449  | 5.46305 | 3.60096  |  6.49201 |          0.3  | TSM+LLM-COT-RF-HDELTA_blend_ramp_bestval_path_h1base |
|  79 |        30 | 59.3218  | 7.70206 | 5.36768  | 10.1439  |          0.3  | TSM+LLM-COT-RF-HDELTA_blend_ramp_bestval_path_h1base |
|  80 |         1 |  1.21489 | 1.10222 | 0.820331 |  1.44583 |          0.05 | TSM+LLM-COT-RF-HDELTA_blend_ramp_bestval_h1          |
|  81 |         5 |  4.59811 | 2.14432 | 1.52671  |  2.62331 |          0.05 | TSM+LLM-COT-RF-HDELTA_blend_ramp_bestval_h1          |
|  82 |        20 | 29.8561  | 5.46407 | 3.58635  |  6.46801 |          0.05 | TSM+LLM-COT-RF-HDELTA_blend_ramp_bestval_h1          |
|  83 |        30 | 59.4497  | 7.71036 | 5.3919   | 10.1879  |          0.05 | TSM+LLM-COT-RF-HDELTA_blend_ramp_bestval_h1          |
|  84 |         1 |  1.20679 | 1.09854 | 0.817577 |  1.44208 |          0.1  | TSM+LLM-COT-RF-HDELTA_blend_ramp_bestval_h1          |
|  85 |         5 |  4.61202 | 2.14756 | 1.52665  |  2.62418 |          0.1  | TSM+LLM-COT-RF-HDELTA_blend_ramp_bestval_h1          |
|  86 |        20 | 29.8493  | 5.46345 | 3.58911  |  6.47248 |          0.1  | TSM+LLM-COT-RF-HDELTA_blend_ramp_bestval_h1          |
|  87 |        30 | 59.4206  | 7.70847 | 5.38697  | 10.1789  |          0.1  | TSM+LLM-COT-RF-HDELTA_blend_ramp_bestval_h1          |
|  88 |         1 |  1.19649 | 1.09384 | 0.812703 |  1.4359  |          0.2  | TSM+LLM-COT-RF-HDELTA_blend_ramp_bestval_h1          |
|  89 |         5 |  4.6484  | 2.15601 | 1.52773  |  2.62835 |          0.2  | TSM+LLM-COT-RF-HDELTA_blend_ramp_bestval_h1          |
|  90 |        20 | 29.8426  | 5.46284 | 3.59462  |  6.48143 |          0.2  | TSM+LLM-COT-RF-HDELTA_blend_ramp_bestval_h1          |
|  91 |        30 | 59.3676  | 7.70504 | 5.37711  | 10.161   |          0.2  | TSM+LLM-COT-RF-HDELTA_blend_ramp_bestval_h1          |
|  92 |         1 |  1.19405 | 1.09273 | 0.810592 |  1.43459 |          0.3  | TSM+LLM-COT-RF-HDELTA_blend_ramp_bestval_h1          |
|  93 |         5 |  4.69618 | 2.16707 | 1.53203  |  2.63806 |          0.3  | TSM+LLM-COT-RF-HDELTA_blend_ramp_bestval_h1          |
|  94 |        20 | 29.8449  | 5.46305 | 3.60096  |  6.49201 |          0.3  | TSM+LLM-COT-RF-HDELTA_blend_ramp_bestval_h1          |
|  95 |        30 | 59.3218  | 7.70206 | 5.36768  | 10.1439  |          0.3  | TSM+LLM-COT-RF-HDELTA_blend_ramp_bestval_h1          |
|  96 |         1 |  1.21489 | 1.10222 | 0.820331 |  1.44583 |          0.05 | TSM+LLM-COT-RF-HDELTA_blend_ramp_bestval_h5          |
|  97 |         5 |  4.59811 | 2.14432 | 1.52671  |  2.62331 |          0.05 | TSM+LLM-COT-RF-HDELTA_blend_ramp_bestval_h5          |
|  98 |        20 | 29.8561  | 5.46407 | 3.58635  |  6.46801 |          0.05 | TSM+LLM-COT-RF-HDELTA_blend_ramp_bestval_h5          |
|  99 |        30 | 59.4497  | 7.71036 | 5.3919   | 10.1879  |          0.05 | TSM+LLM-COT-RF-HDELTA_blend_ramp_bestval_h5          |
| 100 |         1 |  1.20679 | 1.09854 | 0.817577 |  1.44208 |          0.1  | TSM+LLM-COT-RF-HDELTA_blend_ramp_bestval_h5          |
| 101 |         5 |  4.61202 | 2.14756 | 1.52665  |  2.62418 |          0.1  | TSM+LLM-COT-RF-HDELTA_blend_ramp_bestval_h5          |
| 102 |        20 | 29.8493  | 5.46345 | 3.58911  |  6.47248 |          0.1  | TSM+LLM-COT-RF-HDELTA_blend_ramp_bestval_h5          |
| 103 |        30 | 59.4206  | 7.70847 | 5.38697  | 10.1789  |          0.1  | TSM+LLM-COT-RF-HDELTA_blend_ramp_bestval_h5          |
| 104 |         1 |  1.19649 | 1.09384 | 0.812703 |  1.4359  |          0.2  | TSM+LLM-COT-RF-HDELTA_blend_ramp_bestval_h5          |
| 105 |         5 |  4.6484  | 2.15601 | 1.52773  |  2.62835 |          0.2  | TSM+LLM-COT-RF-HDELTA_blend_ramp_bestval_h5          |
| 106 |        20 | 29.8426  | 5.46284 | 3.59462  |  6.48143 |          0.2  | TSM+LLM-COT-RF-HDELTA_blend_ramp_bestval_h5          |
| 107 |        30 | 59.3676  | 7.70504 | 5.37711  | 10.161   |          0.2  | TSM+LLM-COT-RF-HDELTA_blend_ramp_bestval_h5          |
| 108 |         1 |  1.19405 | 1.09273 | 0.810592 |  1.43459 |          0.3  | TSM+LLM-COT-RF-HDELTA_blend_ramp_bestval_h5          |
| 109 |         5 |  4.69618 | 2.16707 | 1.53203  |  2.63806 |          0.3  | TSM+LLM-COT-RF-HDELTA_blend_ramp_bestval_h5          |
| 110 |        20 | 29.8449  | 5.46305 | 3.60096  |  6.49201 |          0.3  | TSM+LLM-COT-RF-HDELTA_blend_ramp_bestval_h5          |
| 111 |        30 | 59.3218  | 7.70206 | 5.36768  | 10.1439  |          0.3  | TSM+LLM-COT-RF-HDELTA_blend_ramp_bestval_h5          |
| 112 |         1 |  1.21489 | 1.10222 | 0.820331 |  1.44583 |          0.05 | TSM+LLM-COT-RF-HDELTA_blend_ramp_bestval_h20         |
| 113 |         5 |  4.59811 | 2.14432 | 1.52671  |  2.62331 |          0.05 | TSM+LLM-COT-RF-HDELTA_blend_ramp_bestval_h20         |
| 114 |        20 | 29.8561  | 5.46407 | 3.58635  |  6.46801 |          0.05 | TSM+LLM-COT-RF-HDELTA_blend_ramp_bestval_h20         |
| 115 |        30 | 59.4497  | 7.71036 | 5.3919   | 10.1879  |          0.05 | TSM+LLM-COT-RF-HDELTA_blend_ramp_bestval_h20         |
| 116 |         1 |  1.20679 | 1.09854 | 0.817577 |  1.44208 |          0.1  | TSM+LLM-COT-RF-HDELTA_blend_ramp_bestval_h20         |
| 117 |         5 |  4.61202 | 2.14756 | 1.52665  |  2.62418 |          0.1  | TSM+LLM-COT-RF-HDELTA_blend_ramp_bestval_h20         |
| 118 |        20 | 29.8493  | 5.46345 | 3.58911  |  6.47248 |          0.1  | TSM+LLM-COT-RF-HDELTA_blend_ramp_bestval_h20         |
| 119 |        30 | 59.4206  | 7.70847 | 5.38697  | 10.1789  |          0.1  | TSM+LLM-COT-RF-HDELTA_blend_ramp_bestval_h20         |
| 120 |         1 |  1.19649 | 1.09384 | 0.812703 |  1.4359  |          0.2  | TSM+LLM-COT-RF-HDELTA_blend_ramp_bestval_h20         |
| 121 |         5 |  4.6484  | 2.15601 | 1.52773  |  2.62835 |          0.2  | TSM+LLM-COT-RF-HDELTA_blend_ramp_bestval_h20         |
| 122 |        20 | 29.8426  | 5.46284 | 3.59462  |  6.48143 |          0.2  | TSM+LLM-COT-RF-HDELTA_blend_ramp_bestval_h20         |
| 123 |        30 | 59.3676  | 7.70504 | 5.37711  | 10.161   |          0.2  | TSM+LLM-COT-RF-HDELTA_blend_ramp_bestval_h20         |
| 124 |         1 |  1.19405 | 1.09273 | 0.810592 |  1.43459 |          0.3  | TSM+LLM-COT-RF-HDELTA_blend_ramp_bestval_h20         |
| 125 |         5 |  4.69618 | 2.16707 | 1.53203  |  2.63806 |          0.3  | TSM+LLM-COT-RF-HDELTA_blend_ramp_bestval_h20         |
| 126 |        20 | 29.8449  | 5.46305 | 3.60096  |  6.49201 |          0.3  | TSM+LLM-COT-RF-HDELTA_blend_ramp_bestval_h20         |
| 127 |        30 | 59.3218  | 7.70206 | 5.36768  | 10.1439  |          0.3  | TSM+LLM-COT-RF-HDELTA_blend_ramp_bestval_h20         |
| 128 |         1 |  1.21489 | 1.10222 | 0.820331 |  1.44583 |          0.05 | TSM+LLM-COT-RF-HDELTA_blend_ramp_bestval_h30         |
| 129 |         5 |  4.59811 | 2.14432 | 1.52671  |  2.62331 |          0.05 | TSM+LLM-COT-RF-HDELTA_blend_ramp_bestval_h30         |
| 130 |        20 | 29.8561  | 5.46407 | 3.58635  |  6.46801 |          0.05 | TSM+LLM-COT-RF-HDELTA_blend_ramp_bestval_h30         |
| 131 |        30 | 59.4497  | 7.71036 | 5.3919   | 10.1879  |          0.05 | TSM+LLM-COT-RF-HDELTA_blend_ramp_bestval_h30         |
| 132 |         1 |  1.20679 | 1.09854 | 0.817577 |  1.44208 |          0.1  | TSM+LLM-COT-RF-HDELTA_blend_ramp_bestval_h30         |
| 133 |         5 |  4.61202 | 2.14756 | 1.52665  |  2.62418 |          0.1  | TSM+LLM-COT-RF-HDELTA_blend_ramp_bestval_h30         |
| 134 |        20 | 29.8493  | 5.46345 | 3.58911  |  6.47248 |          0.1  | TSM+LLM-COT-RF-HDELTA_blend_ramp_bestval_h30         |
| 135 |        30 | 59.4206  | 7.70847 | 5.38697  | 10.1789  |          0.1  | TSM+LLM-COT-RF-HDELTA_blend_ramp_bestval_h30         |
| 136 |         1 |  1.19649 | 1.09384 | 0.812703 |  1.4359  |          0.2  | TSM+LLM-COT-RF-HDELTA_blend_ramp_bestval_h30         |
| 137 |         5 |  4.6484  | 2.15601 | 1.52773  |  2.62835 |          0.2  | TSM+LLM-COT-RF-HDELTA_blend_ramp_bestval_h30         |
| 138 |        20 | 29.8426  | 5.46284 | 3.59462  |  6.48143 |          0.2  | TSM+LLM-COT-RF-HDELTA_blend_ramp_bestval_h30         |
| 139 |        30 | 59.3676  | 7.70504 | 5.37711  | 10.161   |          0.2  | TSM+LLM-COT-RF-HDELTA_blend_ramp_bestval_h30         |
| 140 |         1 |  1.19405 | 1.09273 | 0.810592 |  1.43459 |          0.3  | TSM+LLM-COT-RF-HDELTA_blend_ramp_bestval_h30         |
| 141 |         5 |  4.69618 | 2.16707 | 1.53203  |  2.63806 |          0.3  | TSM+LLM-COT-RF-HDELTA_blend_ramp_bestval_h30         |
| 142 |        20 | 29.8449  | 5.46305 | 3.60096  |  6.49201 |          0.3  | TSM+LLM-COT-RF-HDELTA_blend_ramp_bestval_h30         |
| 143 |        30 | 59.3218  | 7.70206 | 5.36768  | 10.1439  |          0.3  | TSM+LLM-COT-RF-HDELTA_blend_ramp_bestval_h30         |

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


