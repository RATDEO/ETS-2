# EU ETS Futures Forecasting with TSM + LLM Refinement

**Authors**: Auto-generated

**Generated**: 2026-02-26 16:14:07

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
- Best performing method: naive_persistence_llm_subset


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

|    |   horizon |        mse |     rmse |      mae |      mape | model                                  |
|---:|----------:|-----------:|---------:|---------:|----------:|:---------------------------------------|
|  0 |         1 |  1.39098   | 1.1794   | 0.947696 |  1.32876  | naive_persistence                      |
|  1 |         5 |  6.4397    | 2.53766  | 2.01736  |  2.83588  | naive_persistence                      |
|  2 |        20 | 26.3936    | 5.13747  | 4.04948  |  5.56409  | naive_persistence                      |
|  3 |        30 | 41.3395    | 6.42958  | 5.15615  |  7.07198  | naive_persistence                      |
|  4 |         1 |  6.53148   | 2.55568  | 2.04024  |  2.87018  | seasonal_naive                         |
|  5 |         5 |  6.4397    | 2.53766  | 2.01736  |  2.83588  | seasonal_naive                         |
|  6 |        20 | 26.3936    | 5.13747  | 4.04948  |  5.56409  | seasonal_naive                         |
|  7 |        30 | 41.3395    | 6.42958  | 5.15615  |  7.07198  | seasonal_naive                         |
|  8 |         1 |  1.40541   | 1.1855   | 0.953819 |  1.3368   | linear_ridge                           |
|  9 |         5 |  6.65042   | 2.57884  | 2.04762  |  2.88303  | linear_ridge                           |
| 10 |        20 | 26.8354    | 5.18029  | 3.98828  |  5.59742  | linear_ridge                           |
| 11 |        30 | 39.7633    | 6.30582  | 4.961    |  6.90319  | linear_ridge                           |
| 12 |         1 |  5.34846   | 2.31267  | 2.02053  |  2.77429  | linear_lasso                           |
| 13 |         5 | 10.4036    | 3.22547  | 2.69319  |  3.68562  | linear_lasso                           |
| 14 |        20 | 28.5853    | 5.34652  | 4.47365  |  6.08462  | linear_lasso                           |
| 15 |        30 | 39.992     | 6.32392  | 5.21733  |  7.12347  | linear_lasso                           |
| 16 |         1 |  0.0908557 | 0.301423 | 0.232685 |  0.324059 | tsm                                    |
| 17 |         5 |  1.77245   | 1.33133  | 1.017    |  1.4141   | tsm                                    |
| 18 |        20 | 20.0121    | 4.47348  | 3.45806  |  4.77619  | tsm                                    |
| 19 |        30 | 45.4267    | 6.73993  | 5.22796  |  7.26316  | tsm                                    |
| 20 |         1 |  0.163677  | 0.40457  | 0.357503 |  0.492305 | TSM+LLM-COT-RF                         |
| 21 |         5 |  1.7199    | 1.31145  | 0.935    |  1.31616  | TSM+LLM-COT-RF                         |
| 22 |        20 |  8.65328   | 2.94165  | 2.8975   |  4.05064  | TSM+LLM-COT-RF                         |
| 23 |        30 | 53.8918    | 7.3411   | 5.26     |  7.76674  | TSM+LLM-COT-RF                         |
| 24 |         1 |  0.0722438 | 0.268782 | 0.223582 |  0.306055 | TSM+LLM-COT-RF_blend_ramp_bestval_path |
| 25 |         5 |  2.34171   | 1.53026  | 1.17886  |  1.65844  | TSM+LLM-COT-RF_blend_ramp_bestval_path |
| 26 |        20 | 15.8223    | 3.97772  | 3.80178  |  5.39329  | TSM+LLM-COT-RF_blend_ramp_bestval_path |
| 27 |        30 | 60.4585    | 7.77551  | 6.16903  |  9.13342  | TSM+LLM-COT-RF_blend_ramp_bestval_path |
| 28 |         1 |  0.0722438 | 0.268782 | 0.223582 |  0.306055 | TSM+LLM-COT-RF_blend_ramp_bestval_h1   |
| 29 |         5 |  2.34171   | 1.53026  | 1.17886  |  1.65844  | TSM+LLM-COT-RF_blend_ramp_bestval_h1   |
| 30 |        20 | 15.8223    | 3.97772  | 3.80178  |  5.39329  | TSM+LLM-COT-RF_blend_ramp_bestval_h1   |
| 31 |        30 | 60.4585    | 7.77551  | 6.16903  |  9.13342  | TSM+LLM-COT-RF_blend_ramp_bestval_h1   |
| 32 |         1 |  0.0722438 | 0.268782 | 0.223582 |  0.306055 | TSM+LLM-COT-RF_blend_ramp_bestval_h5   |
| 33 |         5 |  2.34171   | 1.53026  | 1.17886  |  1.65844  | TSM+LLM-COT-RF_blend_ramp_bestval_h5   |
| 34 |        20 | 15.8223    | 3.97772  | 3.80178  |  5.39329  | TSM+LLM-COT-RF_blend_ramp_bestval_h5   |
| 35 |        30 | 60.4585    | 7.77551  | 6.16903  |  9.13342  | TSM+LLM-COT-RF_blend_ramp_bestval_h5   |
| 36 |         1 |  0.0722438 | 0.268782 | 0.223582 |  0.306055 | TSM+LLM-COT-RF_blend_ramp_bestval_h20  |
| 37 |         5 |  2.34171   | 1.53026  | 1.17886  |  1.65844  | TSM+LLM-COT-RF_blend_ramp_bestval_h20  |
| 38 |        20 | 15.8223    | 3.97772  | 3.80178  |  5.39329  | TSM+LLM-COT-RF_blend_ramp_bestval_h20  |
| 39 |        30 | 60.4585    | 7.77551  | 6.16903  |  9.13342  | TSM+LLM-COT-RF_blend_ramp_bestval_h20  |
| 40 |         1 |  0.0722438 | 0.268782 | 0.223582 |  0.306055 | TSM+LLM-COT-RF_blend_ramp_bestval_h30  |
| 41 |         5 |  2.34171   | 1.53026  | 1.17886  |  1.65844  | TSM+LLM-COT-RF_blend_ramp_bestval_h30  |
| 42 |        20 | 15.8223    | 3.97772  | 3.80178  |  5.39329  | TSM+LLM-COT-RF_blend_ramp_bestval_h30  |
| 43 |        30 | 60.4585    | 7.77551  | 6.16903  |  9.13342  | TSM+LLM-COT-RF_blend_ramp_bestval_h30  |
| 44 |         1 |  0.0440497 | 0.20988  | 0.199999 |  0.274481 | naive_persistence_llm_subset           |
| 45 |         5 |  3.69212   | 1.92149  | 1.6025   |  2.25938  | naive_persistence_llm_subset           |
| 46 |        20 | 16.992     | 4.12213  | 3.8325   |  5.36929  | naive_persistence_llm_subset           |
| 47 |        30 | 42.1764    | 6.49434  | 5.4      |  8.06976  | naive_persistence_llm_subset           |
| 48 |         1 |  6.48712   | 2.54698  | 2.35501  |  3.2241   | seasonal_naive_llm_subset              |
| 49 |         5 |  3.69212   | 1.92149  | 1.6025   |  2.25938  | seasonal_naive_llm_subset              |
| 50 |        20 | 16.992     | 4.12213  | 3.8325   |  5.36929  | seasonal_naive_llm_subset              |
| 51 |        30 | 42.1764    | 6.49434  | 5.4      |  8.06976  | seasonal_naive_llm_subset              |
| 52 |         1 |  0.134559  | 0.366823 | 0.347515 |  0.476464 | linear_ridge_llm_subset                |
| 53 |         5 |  5.59398   | 2.36516  | 2.00279  |  2.82082  | linear_ridge_llm_subset                |
| 54 |        20 | 26.4122    | 5.13928  | 4.95034  |  7.01897  | linear_ridge_llm_subset                |
| 55 |        30 | 63.073     | 7.94185  | 6.4233   |  9.67933  | linear_ridge_llm_subset                |
| 56 |         1 |  3.98815   | 1.99703  | 1.98846  |  2.73207  | linear_lasso_llm_subset                |
| 57 |         5 |  2.47749   | 1.57401  | 1.39379  |  1.94803  | linear_lasso_llm_subset                |
| 58 |        20 | 17.8272    | 4.22222  | 3.86762  |  5.33297  | linear_lasso_llm_subset                |
| 59 |        30 | 48.1255    | 6.93725  | 5.60024  |  8.34382  | linear_lasso_llm_subset                |
| 60 |         1 |  0.100167  | 0.316492 | 0.255537 |  0.349186 | tsm_llm_subset                         |
| 61 |         5 |  2.54152   | 1.59421  | 1.25648  |  1.76709  | tsm_llm_subset                         |
| 62 |        20 | 22.2498    | 4.71697  | 4.3544   |  6.21381  | tsm_llm_subset                         |
| 63 |        30 | 69.9827    | 8.36556  | 7.07806  | 10.5001   | tsm_llm_subset                         |

### 4.2 Trend Classification Accuracy

Table 2 shows the accuracy of predicting price direction (up/flat/down).

|    |   horizon |   accuracy |   accuracy_up |   accuracy_down |   accuracy_flat |   n_up |   n_down |   n_flat | model                                  |
|---:|----------:|-----------:|--------------:|----------------:|----------------:|-------:|---------:|---------:|:---------------------------------------|
|  0 |         1 |  0.303665  |     0         |       0         |       1         |    133 |      133 |      116 | naive_persistence                      |
|  1 |         5 |  0.149215  |     0         |       0         |       1         |    177 |      148 |       57 | naive_persistence                      |
|  2 |        20 |  0.0863874 |     0         |       0         |       1         |    226 |      123 |       33 | naive_persistence                      |
|  3 |        30 |  0.0628272 |     0         |       0         |       1         |    229 |      129 |       24 | naive_persistence                      |
|  4 |         1 |  0.342932  |     0.406015  |       0.451128  |       0.146552  |    133 |      133 |      116 | seasonal_naive                         |
|  5 |         5 |  0.149215  |     0         |       0         |       1         |    177 |      148 |       57 | seasonal_naive                         |
|  6 |        20 |  0.0863874 |     0         |       0         |       1         |    226 |      123 |       33 | seasonal_naive                         |
|  7 |        30 |  0.0628272 |     0         |       0         |       1         |    229 |      129 |       24 | seasonal_naive                         |
|  8 |         1 |  0.303665  |     0.0075188 |       0         |       0.991379  |    133 |      133 |      116 | linear_ridge                           |
|  9 |         5 |  0.256545  |     0.265537  |       0.0608108 |       0.736842  |    177 |      148 |       57 | linear_ridge                           |
| 10 |        20 |  0.531414  |     0.862832  |       0.0569106 |       0.030303  |    226 |      123 |       33 | linear_ridge                           |
| 11 |        30 |  0.549738  |     0.842795  |       0.0930233 |       0.208333  |    229 |      129 |       24 | linear_ridge                           |
| 12 |         1 |  0.348168  |     0         |       1         |       0         |    133 |      133 |      116 | linear_lasso                           |
| 13 |         5 |  0.390052  |     0         |       1         |       0.0175439 |    177 |      148 |       57 | linear_lasso                           |
| 14 |        20 |  0.295812  |     0.0929204 |       0.682927  |       0.242424  |    226 |      123 |       33 | linear_lasso                           |
| 15 |        30 |  0.384817  |     0.253275  |       0.612403  |       0.416667  |    229 |      129 |       24 | linear_lasso                           |
| 16 |         1 |  0.827225  |     0.834586  |       0.819549  |       0.827586  |    133 |      133 |      116 | tsm                                    |
| 17 |         5 |  0.73822   |     0.80791   |       0.783784  |       0.403509  |    177 |      148 |       57 | tsm                                    |
| 18 |        20 |  0.578534  |     0.690265  |       0.471545  |       0.212121  |    226 |      123 |       33 | tsm                                    |
| 19 |        30 |  0.518325  |     0.655022  |       0.348837  |       0.125     |    229 |      129 |       24 | tsm                                    |
| 20 |         1 |  1         |   nan         |     nan         |       1         |      0 |        0 |        4 | TSM+LLM-COT-RF                         |
| 21 |         5 |  0.75      |   nan         |       0.666667  |       1         |      0 |        3 |        1 | TSM+LLM-COT-RF                         |
| 22 |        20 |  0.5       |     1         |       0.333333  |     nan         |      1 |        3 |        0 | TSM+LLM-COT-RF                         |
| 23 |        30 |  0.25      |     0         |       0.5       |       0         |      1 |        2 |        1 | TSM+LLM-COT-RF                         |
| 24 |         1 |  1         |   nan         |     nan         |       1         |      0 |        0 |        4 | TSM+LLM-COT-RF_blend_ramp_bestval_path |
| 25 |         5 |  0.5       |   nan         |       0.666667  |       0         |      0 |        3 |        1 | TSM+LLM-COT-RF_blend_ramp_bestval_path |
| 26 |        20 |  0.25      |     1         |       0         |     nan         |      1 |        3 |        0 | TSM+LLM-COT-RF_blend_ramp_bestval_path |
| 27 |        30 |  0.25      |     0         |       0.5       |       0         |      1 |        2 |        1 | TSM+LLM-COT-RF_blend_ramp_bestval_path |
| 28 |         1 |  1         |   nan         |     nan         |       1         |      0 |        0 |        4 | TSM+LLM-COT-RF_blend_ramp_bestval_h1   |
| 29 |         5 |  0.5       |   nan         |       0.666667  |       0         |      0 |        3 |        1 | TSM+LLM-COT-RF_blend_ramp_bestval_h1   |
| 30 |        20 |  0.25      |     1         |       0         |     nan         |      1 |        3 |        0 | TSM+LLM-COT-RF_blend_ramp_bestval_h1   |
| 31 |        30 |  0.25      |     0         |       0.5       |       0         |      1 |        2 |        1 | TSM+LLM-COT-RF_blend_ramp_bestval_h1   |
| 32 |         1 |  1         |   nan         |     nan         |       1         |      0 |        0 |        4 | TSM+LLM-COT-RF_blend_ramp_bestval_h5   |
| 33 |         5 |  0.5       |   nan         |       0.666667  |       0         |      0 |        3 |        1 | TSM+LLM-COT-RF_blend_ramp_bestval_h5   |
| 34 |        20 |  0.25      |     1         |       0         |     nan         |      1 |        3 |        0 | TSM+LLM-COT-RF_blend_ramp_bestval_h5   |
| 35 |        30 |  0.25      |     0         |       0.5       |       0         |      1 |        2 |        1 | TSM+LLM-COT-RF_blend_ramp_bestval_h5   |
| 36 |         1 |  1         |   nan         |     nan         |       1         |      0 |        0 |        4 | TSM+LLM-COT-RF_blend_ramp_bestval_h20  |
| 37 |         5 |  0.5       |   nan         |       0.666667  |       0         |      0 |        3 |        1 | TSM+LLM-COT-RF_blend_ramp_bestval_h20  |
| 38 |        20 |  0.25      |     1         |       0         |     nan         |      1 |        3 |        0 | TSM+LLM-COT-RF_blend_ramp_bestval_h20  |
| 39 |        30 |  0.25      |     0         |       0.5       |       0         |      1 |        2 |        1 | TSM+LLM-COT-RF_blend_ramp_bestval_h20  |
| 40 |         1 |  1         |   nan         |     nan         |       1         |      0 |        0 |        4 | TSM+LLM-COT-RF_blend_ramp_bestval_h30  |
| 41 |         5 |  0.5       |   nan         |       0.666667  |       0         |      0 |        3 |        1 | TSM+LLM-COT-RF_blend_ramp_bestval_h30  |
| 42 |        20 |  0.25      |     1         |       0         |     nan         |      1 |        3 |        0 | TSM+LLM-COT-RF_blend_ramp_bestval_h30  |
| 43 |        30 |  0.25      |     0         |       0.5       |       0         |      1 |        2 |        1 | TSM+LLM-COT-RF_blend_ramp_bestval_h30  |
| 44 |         1 |  1         |   nan         |     nan         |       1         |      0 |        0 |        4 | naive_persistence_llm_subset           |
| 45 |         5 |  0.25      |   nan         |       0         |       1         |      0 |        3 |        1 | naive_persistence_llm_subset           |
| 46 |        20 |  0         |     0         |       0         |     nan         |      1 |        3 |        0 | naive_persistence_llm_subset           |
| 47 |        30 |  0.25      |     0         |       0         |       1         |      1 |        2 |        1 | naive_persistence_llm_subset           |
| 48 |         1 |  0         |   nan         |     nan         |       0         |      0 |        0 |        4 | seasonal_naive_llm_subset              |
| 49 |         5 |  0.25      |   nan         |       0         |       1         |      0 |        3 |        1 | seasonal_naive_llm_subset              |
| 50 |        20 |  0         |     0         |       0         |     nan         |      1 |        3 |        0 | seasonal_naive_llm_subset              |
| 51 |        30 |  0.25      |     0         |       0         |       1         |      1 |        2 |        1 | seasonal_naive_llm_subset              |
| 52 |         1 |  1         |   nan         |     nan         |       1         |      0 |        0 |        4 | linear_ridge_llm_subset                |
| 53 |         5 |  0.25      |   nan         |       0         |       1         |      0 |        3 |        1 | linear_ridge_llm_subset                |
| 54 |        20 |  0.25      |     1         |       0         |     nan         |      1 |        3 |        0 | linear_ridge_llm_subset                |
| 55 |        30 |  0.25      |     1         |       0         |       0         |      1 |        2 |        1 | linear_ridge_llm_subset                |
| 56 |         1 |  0         |   nan         |     nan         |       0         |      0 |        0 |        4 | linear_lasso_llm_subset                |
| 57 |         5 |  0.75      |   nan         |       1         |       0         |      0 |        3 |        1 | linear_lasso_llm_subset                |
| 58 |        20 |  0.5       |     0         |       0.666667  |     nan         |      1 |        3 |        0 | linear_lasso_llm_subset                |
| 59 |        30 |  0.5       |     0         |       0.5       |       1         |      1 |        2 |        1 | linear_lasso_llm_subset                |
| 60 |         1 |  1         |   nan         |     nan         |       1         |      0 |        0 |        4 | tsm_llm_subset                         |
| 61 |         5 |  0.5       |   nan         |       0.666667  |       0         |      0 |        3 |        1 | tsm_llm_subset                         |
| 62 |        20 |  0.25      |     1         |       0         |     nan         |      1 |        3 |        0 | tsm_llm_subset                         |
| 63 |        30 |  0.25      |     0         |       0.5       |       0         |      1 |        2 |        1 | tsm_llm_subset                         |

### 4.3 Statistical Significance

Table 3 reports the results of significance tests comparing methods.

|    |   horizon | model                                  | baseline                     |   mean_error_model |   mean_error_baseline |   improvement_pct |   t_statistic |   t_pvalue | t_significant   |   wilcoxon_statistic |   wilcoxon_pvalue | wilcoxon_significant   |      dm_statistic |   dm_pvalue | dm_significant   | model_better   |
|---:|----------:|:---------------------------------------|:-----------------------------|-------------------:|----------------------:|------------------:|--------------:|-----------:|:----------------|---------------------:|------------------:|:-----------------------|------------------:|------------:|:-----------------|:---------------|
|  0 |         1 | seasonal_naive_llm_subset              | naive_persistence_llm_subset |          6.48712   |             0.0440497 |  -14626.8         |      2.4807   | 0.0892147  | False           |                  nan |               nan | False                  |       2.4807      | 0.0131125   | True             | False          |
|  1 |         5 | seasonal_naive_llm_subset              | naive_persistence_llm_subset |          3.69212   |             3.69212   |      -6.09607e-06 |      0.390438 | 0.722286   | False           |                  nan |               nan | False                  |       0.0103697   | 0.991726    | False            | False          |
|  2 |        20 | seasonal_naive_llm_subset              | naive_persistence_llm_subset |         16.992     |            16.992     |       6.16025e-06 |     -2.2194   | 0.113106   | False           |                  nan |               nan | False                  |      -0.0424567   | 0.966135    | False            | True           |
|  3 |        30 | seasonal_naive_llm_subset              | naive_persistence_llm_subset |         42.1764    |            42.1764    |       1.97895e-06 |     -0.394085 | 0.71986    | False           |                  nan |               nan | False                  |      -0.0330991   | 0.973596    | False            | True           |
|  4 |         1 | linear_ridge_llm_subset                | naive_persistence_llm_subset |          0.134559  |             0.0440497 |    -205.471       |      2.30507  | 0.104501   | False           |                  nan |               nan | False                  |       2.30507     | 0.0211625   | True             | False          |
|  5 |         5 | linear_ridge_llm_subset                | naive_persistence_llm_subset |          5.59398   |             3.69212   |     -51.5111      |      1.265    | 0.295202   | False           |                  nan |               nan | False                  |       3.00006     | 0.00269925  | True             | False          |
|  6 |        20 | linear_ridge_llm_subset                | naive_persistence_llm_subset |         26.4122    |            16.992     |     -55.4394      |      0.816136 | 0.474199   | False           |                  nan |               nan | False                  |       1.88405e+06 | 0           | True             | False          |
|  7 |        30 | linear_ridge_llm_subset                | naive_persistence_llm_subset |         63.073     |            42.1764    |     -49.5456      |      0.887991 | 0.439955   | False           |                  nan |               nan | False                  |       2.42257     | 0.0154113   | True             | False          |
|  8 |         1 | linear_lasso_llm_subset                | naive_persistence_llm_subset |          3.98815   |             0.0440497 |   -8953.74        |      9.43188  | 0.00252566 | True            |                  nan |               nan | False                  |       9.43188     | 0           | True             | False          |
|  9 |         5 | linear_lasso_llm_subset                | naive_persistence_llm_subset |          2.47749   |             3.69212   |      32.8979      |     -0.45276  | 0.68146    | False           |                  nan |               nan | False                  |      -5.33341     | 9.63841e-08 | True             | True           |
| 10 |        20 | linear_lasso_llm_subset                | naive_persistence_llm_subset |         17.8272    |            16.992     |      -4.91519     |      0.119705 | 0.912283   | False           |                  nan |               nan | False                  |  167038           | 0           | True             | False          |
| 11 |        30 | linear_lasso_llm_subset                | naive_persistence_llm_subset |         48.1255    |            42.1764    |     -14.1052      |      0.493607 | 0.65546    | False           |                  nan |               nan | False                  |       1.96454     | 0.0494681   | True             | False          |
| 12 |         1 | tsm_llm_subset                         | naive_persistence_llm_subset |          0.100167  |             0.0440497 |    -127.396       |      0.756591 | 0.504288   | False           |                  nan |               nan | False                  |       0.756591    | 0.449295    | False            | False          |
| 13 |         5 | tsm_llm_subset                         | naive_persistence_llm_subset |          2.54152   |             3.69212   |      31.1638      |     -0.986072 | 0.396802   | False           |                  nan |               nan | False                  | -230121           | 0           | True             | True           |
| 14 |        20 | tsm_llm_subset                         | naive_persistence_llm_subset |         22.2498    |            16.992     |     -30.943       |      0.59609  | 0.593096   | False           |                  nan |               nan | False                  |       1.05157e+06 | 0           | True             | False          |
| 15 |        30 | tsm_llm_subset                         | naive_persistence_llm_subset |         69.9827    |            42.1764    |     -65.9284      |      0.960933 | 0.407474   | False           |                  nan |               nan | False                  |       1.83608     | 0.0663464   | False            | False          |
| 16 |         1 | TSM+LLM-COT-RF                         | naive_persistence_llm_subset |          0.163677  |             0.0440497 |    -271.573       |      1.54758  | 0.219469   | False           |                  nan |               nan | False                  |       1.54758     | 0.121724    | False            | False          |
| 17 |         5 | TSM+LLM-COT-RF                         | naive_persistence_llm_subset |          1.7199    |             3.69212   |      53.417       |     -1.73016  | 0.182037   | False           |                  nan |               nan | False                  |      -3.20545     | 0.00134853  | True             | True           |
| 18 |        20 | TSM+LLM-COT-RF                         | naive_persistence_llm_subset |          8.65328   |            16.992     |      49.0744      |     -1.21383  | 0.311655   | False           |                  nan |               nan | False                  |      -1.78276     | 0.0746258   | False            | True           |
| 19 |        30 | TSM+LLM-COT-RF                         | naive_persistence_llm_subset |         53.8918    |            42.1764    |     -27.777       |      0.468765 | 0.671198   | False           |                  nan |               nan | False                  |       1.63519     | 0.10201     | False            | False          |
| 20 |         1 | TSM+LLM-COT-RF_blend_ramp_bestval_path | naive_persistence_llm_subset |          0.0722438 |             0.0440497 |     -64.005       |      0.608215 | 0.586001   | False           |                  nan |               nan | False                  |       0.608215    | 0.543045    | False            | False          |
| 21 |         5 | TSM+LLM-COT-RF_blend_ramp_bestval_path | naive_persistence_llm_subset |          2.34171   |             3.69212   |      36.5755      |     -1.19155  | 0.319111   | False           |                  nan |               nan | False                  |      -6.3782      | 1.7918e-10  | True             | True           |
| 22 |        20 | TSM+LLM-COT-RF_blend_ramp_bestval_path | naive_persistence_llm_subset |         15.8223    |            16.992     |       6.88389     |     -0.200897 | 0.853628   | False           |                  nan |               nan | False                  | -233942           | 0           | True             | True           |
| 23 |        30 | TSM+LLM-COT-RF_blend_ramp_bestval_path | naive_persistence_llm_subset |         60.4585    |            42.1764    |     -43.3467      |      0.664544 | 0.553859   | False           |                  nan |               nan | False                  |       1.50551     | 0.132192    | False            | False          |
| 24 |         1 | TSM+LLM-COT-RF_blend_ramp_bestval_h1   | naive_persistence_llm_subset |          0.0722438 |             0.0440497 |     -64.005       |      0.608215 | 0.586001   | False           |                  nan |               nan | False                  |       0.608215    | 0.543045    | False            | False          |
| 25 |         5 | TSM+LLM-COT-RF_blend_ramp_bestval_h1   | naive_persistence_llm_subset |          2.34171   |             3.69212   |      36.5755      |     -1.19155  | 0.319111   | False           |                  nan |               nan | False                  |      -6.3782      | 1.7918e-10  | True             | True           |
| 26 |        20 | TSM+LLM-COT-RF_blend_ramp_bestval_h1   | naive_persistence_llm_subset |         15.8223    |            16.992     |       6.88389     |     -0.200897 | 0.853628   | False           |                  nan |               nan | False                  | -233942           | 0           | True             | True           |
| 27 |        30 | TSM+LLM-COT-RF_blend_ramp_bestval_h1   | naive_persistence_llm_subset |         60.4585    |            42.1764    |     -43.3467      |      0.664544 | 0.553859   | False           |                  nan |               nan | False                  |       1.50551     | 0.132192    | False            | False          |
| 28 |         1 | TSM+LLM-COT-RF_blend_ramp_bestval_h5   | naive_persistence_llm_subset |          0.0722438 |             0.0440497 |     -64.005       |      0.608215 | 0.586001   | False           |                  nan |               nan | False                  |       0.608215    | 0.543045    | False            | False          |
| 29 |         5 | TSM+LLM-COT-RF_blend_ramp_bestval_h5   | naive_persistence_llm_subset |          2.34171   |             3.69212   |      36.5755      |     -1.19155  | 0.319111   | False           |                  nan |               nan | False                  |      -6.3782      | 1.7918e-10  | True             | True           |
| 30 |        20 | TSM+LLM-COT-RF_blend_ramp_bestval_h5   | naive_persistence_llm_subset |         15.8223    |            16.992     |       6.88389     |     -0.200897 | 0.853628   | False           |                  nan |               nan | False                  | -233942           | 0           | True             | True           |
| 31 |        30 | TSM+LLM-COT-RF_blend_ramp_bestval_h5   | naive_persistence_llm_subset |         60.4585    |            42.1764    |     -43.3467      |      0.664544 | 0.553859   | False           |                  nan |               nan | False                  |       1.50551     | 0.132192    | False            | False          |
| 32 |         1 | TSM+LLM-COT-RF_blend_ramp_bestval_h20  | naive_persistence_llm_subset |          0.0722438 |             0.0440497 |     -64.005       |      0.608215 | 0.586001   | False           |                  nan |               nan | False                  |       0.608215    | 0.543045    | False            | False          |
| 33 |         5 | TSM+LLM-COT-RF_blend_ramp_bestval_h20  | naive_persistence_llm_subset |          2.34171   |             3.69212   |      36.5755      |     -1.19155  | 0.319111   | False           |                  nan |               nan | False                  |      -6.3782      | 1.7918e-10  | True             | True           |
| 34 |        20 | TSM+LLM-COT-RF_blend_ramp_bestval_h20  | naive_persistence_llm_subset |         15.8223    |            16.992     |       6.88389     |     -0.200897 | 0.853628   | False           |                  nan |               nan | False                  | -233942           | 0           | True             | True           |
| 35 |        30 | TSM+LLM-COT-RF_blend_ramp_bestval_h20  | naive_persistence_llm_subset |         60.4585    |            42.1764    |     -43.3467      |      0.664544 | 0.553859   | False           |                  nan |               nan | False                  |       1.50551     | 0.132192    | False            | False          |
| 36 |         1 | TSM+LLM-COT-RF_blend_ramp_bestval_h30  | naive_persistence_llm_subset |          0.0722438 |             0.0440497 |     -64.005       |      0.608215 | 0.586001   | False           |                  nan |               nan | False                  |       0.608215    | 0.543045    | False            | False          |
| 37 |         5 | TSM+LLM-COT-RF_blend_ramp_bestval_h30  | naive_persistence_llm_subset |          2.34171   |             3.69212   |      36.5755      |     -1.19155  | 0.319111   | False           |                  nan |               nan | False                  |      -6.3782      | 1.7918e-10  | True             | True           |
| 38 |        20 | TSM+LLM-COT-RF_blend_ramp_bestval_h30  | naive_persistence_llm_subset |         15.8223    |            16.992     |       6.88389     |     -0.200897 | 0.853628   | False           |                  nan |               nan | False                  | -233942           | 0           | True             | True           |
| 39 |        30 | TSM+LLM-COT-RF_blend_ramp_bestval_h30  | naive_persistence_llm_subset |         60.4585    |            42.1764    |     -43.3467      |      0.664544 | 0.553859   | False           |                  nan |               nan | False                  |       1.50551     | 0.132192    | False            | False          |


## 5. Robustness Analysis

### 5.1 Noise Injection Test

To assess robustness, we inject noise into forecast paths at levels of 5%, 10%, 20%, 
and 30% of the forecast standard deviation.

|     |   horizon |       mse |     rmse |      mae |     mape |   noise_level | model                                  |
|----:|----------:|----------:|---------:|---------:|---------:|--------------:|:---------------------------------------|
|   0 |         1 |  0.098265 | 0.313472 | 0.239818 | 0.33432  |          0.05 | tsm                                    |
|   1 |         5 |  1.76472  | 1.32843  | 1.01773  | 1.41545  |          0.05 | tsm                                    |
|   2 |        20 | 20.0028   | 4.47245  | 3.45817  | 4.77659  |          0.05 | tsm                                    |
|   3 |        30 | 45.3224   | 6.73219  | 5.22231  | 7.25555  |          0.05 | tsm                                    |
|   4 |         1 |  0.119587 | 0.345813 | 0.264408 | 0.368678 |          0.1  | tsm                                    |
|   5 |         5 |  1.77325  | 1.33163  | 1.02174  | 1.42125  |          0.1  | tsm                                    |
|   6 |        20 | 20.0058   | 4.47279  | 3.45955  | 4.77878  |          0.1  | tsm                                    |
|   7 |        30 | 45.2309   | 6.72539  | 5.21827  | 7.25019  |          0.1  | tsm                                    |
|   8 |         1 |  0.203967 | 0.451627 | 0.343971 | 0.480069 |          0.2  | tsm                                    |
|   9 |         5 |  1.83911  | 1.35614  | 1.04426  | 1.45242  |          0.2  | tsm                                    |
|  10 |        20 | 20.0484   | 4.47754  | 3.46885  | 4.79159  |          0.2  | tsm                                    |
|  11 |        30 | 45.0862   | 6.71463  | 5.21147  | 7.24117  |          0.2  | tsm                                    |
|  12 |         1 |  0.343997 | 0.586513 | 0.447324 | 0.624281 |          0.3  | tsm                                    |
|  13 |         5 |  1.97006  | 1.40359  | 1.07901  | 1.50058  |          0.3  | tsm                                    |
|  14 |        20 | 20.1397   | 4.48773  | 3.48023  | 4.80727  |          0.3  | tsm                                    |
|  15 |        30 | 44.9926   | 6.70766  | 5.20628  | 7.23423  |          0.3  | tsm                                    |
|  16 |         1 |  5.95313  | 2.4399   | 2.36525  | 3.38618  |          0.05 | TSM+LLM-COT-RF                         |
|  17 |         5 | 10.0412   | 3.16879  | 2.98283  | 4.33505  |          0.05 | TSM+LLM-COT-RF                         |
|  18 |        20 | 21.1619   | 4.60021  | 4.22388  | 6.1398   |          0.05 | TSM+LLM-COT-RF                         |
|  19 |        30 | 12.6683   | 3.55926  | 3.08009  | 4.3155   |          0.05 | TSM+LLM-COT-RF                         |
|  20 |         1 |  5.94822  | 2.4389   | 2.358    | 3.3767   |          0.1  | TSM+LLM-COT-RF                         |
|  21 |         5 | 10.2397   | 3.19995  | 3.00317  | 4.36466  |          0.1  | TSM+LLM-COT-RF                         |
|  22 |        20 | 20.2976   | 4.50529  | 4.11277  | 5.97727  |          0.1  | TSM+LLM-COT-RF                         |
|  23 |        30 | 13.0187   | 3.60815  | 3.11018  | 4.35732  |          0.1  | TSM+LLM-COT-RF                         |
|  24 |         1 |  5.94991  | 2.43924  | 2.34351  | 3.35774  |          0.2  | TSM+LLM-COT-RF                         |
|  25 |         5 | 10.6524   | 3.2638   | 3.04385  | 4.42389  |          0.2  | TSM+LLM-COT-RF                         |
|  26 |        20 | 18.6725   | 4.32116  | 3.89054  | 5.65223  |          0.2  | TSM+LLM-COT-RF                         |
|  27 |        30 | 13.7412   | 3.70691  | 3.17036  | 4.44096  |          0.2  | TSM+LLM-COT-RF                         |
|  28 |         1 |  5.96694  | 2.44273  | 2.32901  | 3.33878  |          0.3  | TSM+LLM-COT-RF                         |
|  29 |         5 | 11.0862   | 3.32959  | 3.08452  | 4.48312  |          0.3  | TSM+LLM-COT-RF                         |
|  30 |        20 | 17.1852   | 4.1455   | 3.66831  | 5.32718  |          0.3  | TSM+LLM-COT-RF                         |
|  31 |        30 | 14.4923   | 3.80687  | 3.23054  | 4.52459  |          0.3  | TSM+LLM-COT-RF                         |
|  32 |         1 |  8.0205   | 2.83205  | 2.75582  | 3.94241  |          0.05 | TSM+LLM-COT-RF_blend_ramp_bestval_path |
|  33 |         5 | 11.8535   | 3.44289  | 3.29876  | 4.78906  |          0.05 | TSM+LLM-COT-RF_blend_ramp_bestval_path |
|  34 |        20 | 27.1531   | 5.21086  | 5.13247  | 7.47764  |          0.05 | TSM+LLM-COT-RF_blend_ramp_bestval_path |
|  35 |        30 | 11.681    | 3.41774  | 2.88447  | 4.02432  |          0.05 | TSM+LLM-COT-RF_blend_ramp_bestval_path |
|  36 |         1 |  7.97165  | 2.82341  | 2.74419  | 3.92659  |          0.1  | TSM+LLM-COT-RF_blend_ramp_bestval_path |
|  37 |         5 | 12.0858   | 3.47647  | 3.32267  | 4.82384  |          0.1  | TSM+LLM-COT-RF_blend_ramp_bestval_path |
|  38 |        20 | 26.0355   | 5.1025   | 5.02491  | 7.32047  |          0.1  | TSM+LLM-COT-RF_blend_ramp_bestval_path |
|  39 |        30 | 12.0448   | 3.47056  | 2.91444  | 4.06591  |          0.1  | TSM+LLM-COT-RF_blend_ramp_bestval_path |
|  40 |         1 |  7.88432  | 2.8079   | 2.72093  | 3.89495  |          0.2  | TSM+LLM-COT-RF_blend_ramp_bestval_path |
|  41 |         5 | 12.5673   | 3.54504  | 3.37051  | 4.8934   |          0.2  | TSM+LLM-COT-RF_blend_ramp_bestval_path |
|  42 |        20 | 23.8974   | 4.88849  | 4.8098   | 7.00612  |          0.2  | TSM+LLM-COT-RF_blend_ramp_bestval_path |
|  43 |        30 | 12.7945   | 3.57694  | 2.97438  | 4.14909  |          0.2  | TSM+LLM-COT-RF_blend_ramp_bestval_path |
|  44 |         1 |  7.81081  | 2.79478  | 2.69767  | 3.86331  |          0.3  | TSM+LLM-COT-RF_blend_ramp_bestval_path |
|  45 |         5 | 13.0712   | 3.61541  | 3.41834  | 4.96296  |          0.3  | TSM+LLM-COT-RF_blend_ramp_bestval_path |
|  46 |        20 | 21.8886   | 4.67852  | 4.59468  | 6.69178  |          0.3  | TSM+LLM-COT-RF_blend_ramp_bestval_path |
|  47 |        30 | 13.5736   | 3.68424  | 3.03431  | 4.23227  |          0.3  | TSM+LLM-COT-RF_blend_ramp_bestval_path |
|  48 |         1 |  8.0205   | 2.83205  | 2.75582  | 3.94241  |          0.05 | TSM+LLM-COT-RF_blend_ramp_bestval_h1   |
|  49 |         5 | 11.8535   | 3.44289  | 3.29876  | 4.78906  |          0.05 | TSM+LLM-COT-RF_blend_ramp_bestval_h1   |
|  50 |        20 | 27.1531   | 5.21086  | 5.13247  | 7.47764  |          0.05 | TSM+LLM-COT-RF_blend_ramp_bestval_h1   |
|  51 |        30 | 11.681    | 3.41774  | 2.88447  | 4.02432  |          0.05 | TSM+LLM-COT-RF_blend_ramp_bestval_h1   |
|  52 |         1 |  7.97165  | 2.82341  | 2.74419  | 3.92659  |          0.1  | TSM+LLM-COT-RF_blend_ramp_bestval_h1   |
|  53 |         5 | 12.0858   | 3.47647  | 3.32267  | 4.82384  |          0.1  | TSM+LLM-COT-RF_blend_ramp_bestval_h1   |
|  54 |        20 | 26.0355   | 5.1025   | 5.02491  | 7.32047  |          0.1  | TSM+LLM-COT-RF_blend_ramp_bestval_h1   |
|  55 |        30 | 12.0448   | 3.47056  | 2.91444  | 4.06591  |          0.1  | TSM+LLM-COT-RF_blend_ramp_bestval_h1   |
|  56 |         1 |  7.88432  | 2.8079   | 2.72093  | 3.89495  |          0.2  | TSM+LLM-COT-RF_blend_ramp_bestval_h1   |
|  57 |         5 | 12.5673   | 3.54504  | 3.37051  | 4.8934   |          0.2  | TSM+LLM-COT-RF_blend_ramp_bestval_h1   |
|  58 |        20 | 23.8974   | 4.88849  | 4.8098   | 7.00612  |          0.2  | TSM+LLM-COT-RF_blend_ramp_bestval_h1   |
|  59 |        30 | 12.7945   | 3.57694  | 2.97438  | 4.14909  |          0.2  | TSM+LLM-COT-RF_blend_ramp_bestval_h1   |
|  60 |         1 |  7.81081  | 2.79478  | 2.69767  | 3.86331  |          0.3  | TSM+LLM-COT-RF_blend_ramp_bestval_h1   |
|  61 |         5 | 13.0712   | 3.61541  | 3.41834  | 4.96296  |          0.3  | TSM+LLM-COT-RF_blend_ramp_bestval_h1   |
|  62 |        20 | 21.8886   | 4.67852  | 4.59468  | 6.69178  |          0.3  | TSM+LLM-COT-RF_blend_ramp_bestval_h1   |
|  63 |        30 | 13.5736   | 3.68424  | 3.03431  | 4.23227  |          0.3  | TSM+LLM-COT-RF_blend_ramp_bestval_h1   |
|  64 |         1 |  8.0205   | 2.83205  | 2.75582  | 3.94241  |          0.05 | TSM+LLM-COT-RF_blend_ramp_bestval_h5   |
|  65 |         5 | 11.8535   | 3.44289  | 3.29876  | 4.78906  |          0.05 | TSM+LLM-COT-RF_blend_ramp_bestval_h5   |
|  66 |        20 | 27.1531   | 5.21086  | 5.13247  | 7.47764  |          0.05 | TSM+LLM-COT-RF_blend_ramp_bestval_h5   |
|  67 |        30 | 11.681    | 3.41774  | 2.88447  | 4.02432  |          0.05 | TSM+LLM-COT-RF_blend_ramp_bestval_h5   |
|  68 |         1 |  7.97165  | 2.82341  | 2.74419  | 3.92659  |          0.1  | TSM+LLM-COT-RF_blend_ramp_bestval_h5   |
|  69 |         5 | 12.0858   | 3.47647  | 3.32267  | 4.82384  |          0.1  | TSM+LLM-COT-RF_blend_ramp_bestval_h5   |
|  70 |        20 | 26.0355   | 5.1025   | 5.02491  | 7.32047  |          0.1  | TSM+LLM-COT-RF_blend_ramp_bestval_h5   |
|  71 |        30 | 12.0448   | 3.47056  | 2.91444  | 4.06591  |          0.1  | TSM+LLM-COT-RF_blend_ramp_bestval_h5   |
|  72 |         1 |  7.88432  | 2.8079   | 2.72093  | 3.89495  |          0.2  | TSM+LLM-COT-RF_blend_ramp_bestval_h5   |
|  73 |         5 | 12.5673   | 3.54504  | 3.37051  | 4.8934   |          0.2  | TSM+LLM-COT-RF_blend_ramp_bestval_h5   |
|  74 |        20 | 23.8974   | 4.88849  | 4.8098   | 7.00612  |          0.2  | TSM+LLM-COT-RF_blend_ramp_bestval_h5   |
|  75 |        30 | 12.7945   | 3.57694  | 2.97438  | 4.14909  |          0.2  | TSM+LLM-COT-RF_blend_ramp_bestval_h5   |
|  76 |         1 |  7.81081  | 2.79478  | 2.69767  | 3.86331  |          0.3  | TSM+LLM-COT-RF_blend_ramp_bestval_h5   |
|  77 |         5 | 13.0712   | 3.61541  | 3.41834  | 4.96296  |          0.3  | TSM+LLM-COT-RF_blend_ramp_bestval_h5   |
|  78 |        20 | 21.8886   | 4.67852  | 4.59468  | 6.69178  |          0.3  | TSM+LLM-COT-RF_blend_ramp_bestval_h5   |
|  79 |        30 | 13.5736   | 3.68424  | 3.03431  | 4.23227  |          0.3  | TSM+LLM-COT-RF_blend_ramp_bestval_h5   |
|  80 |         1 |  8.0205   | 2.83205  | 2.75582  | 3.94241  |          0.05 | TSM+LLM-COT-RF_blend_ramp_bestval_h20  |
|  81 |         5 | 11.8535   | 3.44289  | 3.29876  | 4.78906  |          0.05 | TSM+LLM-COT-RF_blend_ramp_bestval_h20  |
|  82 |        20 | 27.1531   | 5.21086  | 5.13247  | 7.47764  |          0.05 | TSM+LLM-COT-RF_blend_ramp_bestval_h20  |
|  83 |        30 | 11.681    | 3.41774  | 2.88447  | 4.02432  |          0.05 | TSM+LLM-COT-RF_blend_ramp_bestval_h20  |
|  84 |         1 |  7.97165  | 2.82341  | 2.74419  | 3.92659  |          0.1  | TSM+LLM-COT-RF_blend_ramp_bestval_h20  |
|  85 |         5 | 12.0858   | 3.47647  | 3.32267  | 4.82384  |          0.1  | TSM+LLM-COT-RF_blend_ramp_bestval_h20  |
|  86 |        20 | 26.0355   | 5.1025   | 5.02491  | 7.32047  |          0.1  | TSM+LLM-COT-RF_blend_ramp_bestval_h20  |
|  87 |        30 | 12.0448   | 3.47056  | 2.91444  | 4.06591  |          0.1  | TSM+LLM-COT-RF_blend_ramp_bestval_h20  |
|  88 |         1 |  7.88432  | 2.8079   | 2.72093  | 3.89495  |          0.2  | TSM+LLM-COT-RF_blend_ramp_bestval_h20  |
|  89 |         5 | 12.5673   | 3.54504  | 3.37051  | 4.8934   |          0.2  | TSM+LLM-COT-RF_blend_ramp_bestval_h20  |
|  90 |        20 | 23.8974   | 4.88849  | 4.8098   | 7.00612  |          0.2  | TSM+LLM-COT-RF_blend_ramp_bestval_h20  |
|  91 |        30 | 12.7945   | 3.57694  | 2.97438  | 4.14909  |          0.2  | TSM+LLM-COT-RF_blend_ramp_bestval_h20  |
|  92 |         1 |  7.81081  | 2.79478  | 2.69767  | 3.86331  |          0.3  | TSM+LLM-COT-RF_blend_ramp_bestval_h20  |
|  93 |         5 | 13.0712   | 3.61541  | 3.41834  | 4.96296  |          0.3  | TSM+LLM-COT-RF_blend_ramp_bestval_h20  |
|  94 |        20 | 21.8886   | 4.67852  | 4.59468  | 6.69178  |          0.3  | TSM+LLM-COT-RF_blend_ramp_bestval_h20  |
|  95 |        30 | 13.5736   | 3.68424  | 3.03431  | 4.23227  |          0.3  | TSM+LLM-COT-RF_blend_ramp_bestval_h20  |
|  96 |         1 |  8.0205   | 2.83205  | 2.75582  | 3.94241  |          0.05 | TSM+LLM-COT-RF_blend_ramp_bestval_h30  |
|  97 |         5 | 11.8535   | 3.44289  | 3.29876  | 4.78906  |          0.05 | TSM+LLM-COT-RF_blend_ramp_bestval_h30  |
|  98 |        20 | 27.1531   | 5.21086  | 5.13247  | 7.47764  |          0.05 | TSM+LLM-COT-RF_blend_ramp_bestval_h30  |
|  99 |        30 | 11.681    | 3.41774  | 2.88447  | 4.02432  |          0.05 | TSM+LLM-COT-RF_blend_ramp_bestval_h30  |
| 100 |         1 |  7.97165  | 2.82341  | 2.74419  | 3.92659  |          0.1  | TSM+LLM-COT-RF_blend_ramp_bestval_h30  |
| 101 |         5 | 12.0858   | 3.47647  | 3.32267  | 4.82384  |          0.1  | TSM+LLM-COT-RF_blend_ramp_bestval_h30  |
| 102 |        20 | 26.0355   | 5.1025   | 5.02491  | 7.32047  |          0.1  | TSM+LLM-COT-RF_blend_ramp_bestval_h30  |
| 103 |        30 | 12.0448   | 3.47056  | 2.91444  | 4.06591  |          0.1  | TSM+LLM-COT-RF_blend_ramp_bestval_h30  |
| 104 |         1 |  7.88432  | 2.8079   | 2.72093  | 3.89495  |          0.2  | TSM+LLM-COT-RF_blend_ramp_bestval_h30  |
| 105 |         5 | 12.5673   | 3.54504  | 3.37051  | 4.8934   |          0.2  | TSM+LLM-COT-RF_blend_ramp_bestval_h30  |
| 106 |        20 | 23.8974   | 4.88849  | 4.8098   | 7.00612  |          0.2  | TSM+LLM-COT-RF_blend_ramp_bestval_h30  |
| 107 |        30 | 12.7945   | 3.57694  | 2.97438  | 4.14909  |          0.2  | TSM+LLM-COT-RF_blend_ramp_bestval_h30  |
| 108 |         1 |  7.81081  | 2.79478  | 2.69767  | 3.86331  |          0.3  | TSM+LLM-COT-RF_blend_ramp_bestval_h30  |
| 109 |         5 | 13.0712   | 3.61541  | 3.41834  | 4.96296  |          0.3  | TSM+LLM-COT-RF_blend_ramp_bestval_h30  |
| 110 |        20 | 21.8886   | 4.67852  | 4.59468  | 6.69178  |          0.3  | TSM+LLM-COT-RF_blend_ramp_bestval_h30  |
| 111 |        30 | 13.5736   | 3.68424  | 3.03431  | 4.23227  |          0.3  | TSM+LLM-COT-RF_blend_ramp_bestval_h30  |

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

- **Price accuracy**: naive_persistence_llm_subset has the lowest average MSE (15.726).
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
Price accuracy is best for naive_persistence_llm_subset, while directional accuracy is highest for tsm.
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


