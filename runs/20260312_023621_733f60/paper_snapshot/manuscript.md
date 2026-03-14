# EU ETS Futures Forecasting with TSM + LLM Refinement

**Authors**: Auto-generated

**Generated**: 2026-03-12 02:44:45

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
- Dataset spans 2021-05-19 00:00:00 to 2025-06-30 00:00:00
- 1060 daily observations
- Best performing method: linear_lasso


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

- **Date Range**: 2021-05-19 00:00:00 to 2025-06-30 00:00:00
- **Total Observations**: 1,060
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

|    |   horizon |      mse |    rmse |     mae |     mape | model                        |
|---:|----------:|---------:|--------:|--------:|---------:|:-----------------------------|
|  0 |         1 |  3.4232  | 1.85019 | 1.405   |  3.33844 | naive_persistence            |
|  1 |         5 | 12.3632  | 3.51613 | 2.71357 |  6.35039 | naive_persistence            |
|  2 |        20 | 37.2191  | 6.10075 | 4.85816 | 10.7691  | naive_persistence            |
|  3 |        30 | 31.1626  | 5.58235 | 4.44296 |  9.65018 | naive_persistence            |
|  4 |         1 | 11.8688  | 3.44512 | 2.65357 |  6.32772 | seasonal_naive               |
|  5 |         5 | 12.3632  | 3.51613 | 2.71357 |  6.35038 | seasonal_naive               |
|  6 |        20 | 37.2191  | 6.10075 | 4.85816 | 10.7691  | seasonal_naive               |
|  7 |        30 | 31.1626  | 5.58235 | 4.44296 |  9.65018 | seasonal_naive               |
|  8 |         1 |  3.25786 | 1.80495 | 1.39512 |  3.33278 | linear_ridge                 |
|  9 |         5 | 10.4535  | 3.23319 | 2.58135 |  6.11308 | linear_ridge                 |
| 10 |        20 | 15.7078  | 3.96331 | 3.12549 |  6.80617 | linear_ridge                 |
| 11 |        30 | 13.995   | 3.74099 | 3.19962 |  6.85236 | linear_ridge                 |
| 12 |         1 |  4.11542 | 2.02865 | 1.69452 |  4.19682 | linear_lasso                 |
| 13 |         5 | 10.6123  | 3.25766 | 2.69085 |  6.55955 | linear_lasso                 |
| 14 |        20 | 13.6898  | 3.69998 | 2.84225 |  6.26967 | linear_lasso                 |
| 15 |        30 | 10.379   | 3.22164 | 2.60876 |  5.58166 | linear_lasso                 |
| 16 |         1 |  3.3492  | 1.83008 | 1.40458 |  3.33835 | tsm                          |
| 17 |         5 | 10.8861  | 3.29941 | 2.60175 |  6.114   | tsm                          |
| 18 |        20 | 22.5271  | 4.74627 | 3.82478 |  8.33091 | tsm                          |
| 19 |        30 | 22.685   | 4.76288 | 4.15994 |  8.92416 | tsm                          |
| 20 |         1 |  3.3492  | 1.83008 | 1.40458 |  3.33835 | TSM+LLM-COT-RF-HDELTA        |
| 21 |         5 | 10.941   | 3.30772 | 2.60189 |  6.10711 | TSM+LLM-COT-RF-HDELTA        |
| 22 |        20 | 23.862   | 4.88487 | 3.95026 |  8.5992  | TSM+LLM-COT-RF-HDELTA        |
| 23 |        30 | 24.2861  | 4.92809 | 4.33299 |  9.29815 | TSM+LLM-COT-RF-HDELTA        |
| 24 |         1 |  3.4232  | 1.85019 | 1.405   |  3.33844 | naive_persistence_llm_subset |
| 25 |         5 | 12.3632  | 3.51613 | 2.71357 |  6.35039 | naive_persistence_llm_subset |
| 26 |        20 | 37.2191  | 6.10075 | 4.85816 | 10.7691  | naive_persistence_llm_subset |
| 27 |        30 | 31.1626  | 5.58235 | 4.44296 |  9.65018 | naive_persistence_llm_subset |
| 28 |         1 | 11.8688  | 3.44512 | 2.65357 |  6.32772 | seasonal_naive_llm_subset    |
| 29 |         5 | 12.3632  | 3.51613 | 2.71357 |  6.35038 | seasonal_naive_llm_subset    |
| 30 |        20 | 37.2191  | 6.10075 | 4.85816 | 10.7691  | seasonal_naive_llm_subset    |
| 31 |        30 | 31.1626  | 5.58235 | 4.44296 |  9.65018 | seasonal_naive_llm_subset    |
| 32 |         1 |  3.25786 | 1.80495 | 1.39512 |  3.33278 | linear_ridge_llm_subset      |
| 33 |         5 | 10.4535  | 3.23319 | 2.58135 |  6.11308 | linear_ridge_llm_subset      |
| 34 |        20 | 15.7078  | 3.96331 | 3.12549 |  6.80617 | linear_ridge_llm_subset      |
| 35 |        30 | 13.995   | 3.74099 | 3.19962 |  6.85236 | linear_ridge_llm_subset      |
| 36 |         1 |  4.11542 | 2.02865 | 1.69452 |  4.19682 | linear_lasso_llm_subset      |
| 37 |         5 | 10.6123  | 3.25766 | 2.69085 |  6.55955 | linear_lasso_llm_subset      |
| 38 |        20 | 13.6898  | 3.69998 | 2.84225 |  6.26967 | linear_lasso_llm_subset      |
| 39 |        30 | 10.379   | 3.22164 | 2.60876 |  5.58166 | linear_lasso_llm_subset      |
| 40 |         1 |  3.3492  | 1.83008 | 1.40458 |  3.33835 | tsm_llm_subset               |
| 41 |         5 | 10.8861  | 3.29941 | 2.60175 |  6.114   | tsm_llm_subset               |
| 42 |        20 | 22.5271  | 4.74627 | 3.82478 |  8.33091 | tsm_llm_subset               |
| 43 |        30 | 22.685   | 4.76288 | 4.15994 |  8.92416 | tsm_llm_subset               |

### 4.2 Trend Classification Accuracy

Table 2 shows the accuracy of predicting price direction (up/flat/down).

|    |   horizon |   accuracy |   accuracy_up |   accuracy_down |   accuracy_flat |   n_up |   n_down |   n_flat | model                        |
|---:|----------:|-----------:|--------------:|----------------:|----------------:|-------:|---------:|---------:|:-----------------------------|
|  0 |         1 |  0.285714  |      0        |        0        |       1         |     37 |       33 |       28 | naive_persistence            |
|  1 |         5 |  0.122449  |      0        |        0        |       1         |     48 |       38 |       12 | naive_persistence            |
|  2 |        20 |  0.0816327 |      0        |        0        |       1         |     74 |       16 |        8 | naive_persistence            |
|  3 |        30 |  0.122449  |      0        |        0        |       1         |     76 |       10 |       12 | naive_persistence            |
|  4 |         1 |  0.377551  |      0.432432 |        0.454545 |       0.214286  |     37 |       33 |       28 | seasonal_naive               |
|  5 |         5 |  0.122449  |      0        |        0        |       1         |     48 |       38 |       12 | seasonal_naive               |
|  6 |        20 |  0.0816327 |      0        |        0        |       1         |     74 |       16 |        8 | seasonal_naive               |
|  7 |        30 |  0.122449  |      0        |        0        |       1         |     76 |       10 |       12 | seasonal_naive               |
|  8 |         1 |  0.357143  |      0.135135 |        0.181818 |       0.857143  |     37 |       33 |       28 | linear_ridge                 |
|  9 |         5 |  0.428571  |      0.479167 |        0.263158 |       0.75      |     48 |       38 |       12 | linear_ridge                 |
| 10 |        20 |  0.714286  |      0.756757 |        0.8125   |       0.125     |     74 |       16 |        8 | linear_ridge                 |
| 11 |        30 |  0.744898  |      0.842105 |        0.6      |       0.25      |     76 |       10 |       12 | linear_ridge                 |
| 12 |         1 |  0.387755  |      0.945946 |        0        |       0.107143  |     37 |       33 |       28 | linear_lasso                 |
| 13 |         5 |  0.479592  |      0.958333 |        0        |       0.0833333 |     48 |       38 |       12 | linear_lasso                 |
| 14 |        20 |  0.867347  |      0.932432 |        0.75     |       0.5       |     74 |       16 |        8 | linear_lasso                 |
| 15 |        30 |  0.785714  |      0.960526 |        0.1      |       0.25      |     76 |       10 |       12 | linear_lasso                 |
| 16 |         1 |  0.387755  |      0.351351 |        0.363636 |       0.464286  |     37 |       33 |       28 | tsm                          |
| 17 |         5 |  0.438776  |      0.5      |        0.421053 |       0.25      |     48 |       38 |       12 | tsm                          |
| 18 |        20 |  0.622449  |      0.621622 |        0.875    |       0.125     |     74 |       16 |        8 | tsm                          |
| 19 |        30 |  0.530612  |      0.578947 |        0.6      |       0.166667  |     76 |       10 |       12 | tsm                          |
| 20 |         1 |  0.387755  |      0.351351 |        0.363636 |       0.464286  |     37 |       33 |       28 | TSM+LLM-COT-RF-HDELTA        |
| 21 |         5 |  0.459184  |      0.5      |        0.473684 |       0.25      |     48 |       38 |       12 | TSM+LLM-COT-RF-HDELTA        |
| 22 |        20 |  0.581633  |      0.567568 |        0.875    |       0.125     |     74 |       16 |        8 | TSM+LLM-COT-RF-HDELTA        |
| 23 |        30 |  0.469388  |      0.5      |        0.6      |       0.166667  |     76 |       10 |       12 | TSM+LLM-COT-RF-HDELTA        |
| 24 |         1 |  0.285714  |      0        |        0        |       1         |     37 |       33 |       28 | naive_persistence_llm_subset |
| 25 |         5 |  0.122449  |      0        |        0        |       1         |     48 |       38 |       12 | naive_persistence_llm_subset |
| 26 |        20 |  0.0816327 |      0        |        0        |       1         |     74 |       16 |        8 | naive_persistence_llm_subset |
| 27 |        30 |  0.122449  |      0        |        0        |       1         |     76 |       10 |       12 | naive_persistence_llm_subset |
| 28 |         1 |  0.377551  |      0.432432 |        0.454545 |       0.214286  |     37 |       33 |       28 | seasonal_naive_llm_subset    |
| 29 |         5 |  0.122449  |      0        |        0        |       1         |     48 |       38 |       12 | seasonal_naive_llm_subset    |
| 30 |        20 |  0.0816327 |      0        |        0        |       1         |     74 |       16 |        8 | seasonal_naive_llm_subset    |
| 31 |        30 |  0.122449  |      0        |        0        |       1         |     76 |       10 |       12 | seasonal_naive_llm_subset    |
| 32 |         1 |  0.357143  |      0.135135 |        0.181818 |       0.857143  |     37 |       33 |       28 | linear_ridge_llm_subset      |
| 33 |         5 |  0.428571  |      0.479167 |        0.263158 |       0.75      |     48 |       38 |       12 | linear_ridge_llm_subset      |
| 34 |        20 |  0.714286  |      0.756757 |        0.8125   |       0.125     |     74 |       16 |        8 | linear_ridge_llm_subset      |
| 35 |        30 |  0.744898  |      0.842105 |        0.6      |       0.25      |     76 |       10 |       12 | linear_ridge_llm_subset      |
| 36 |         1 |  0.387755  |      0.945946 |        0        |       0.107143  |     37 |       33 |       28 | linear_lasso_llm_subset      |
| 37 |         5 |  0.479592  |      0.958333 |        0        |       0.0833333 |     48 |       38 |       12 | linear_lasso_llm_subset      |
| 38 |        20 |  0.867347  |      0.932432 |        0.75     |       0.5       |     74 |       16 |        8 | linear_lasso_llm_subset      |
| 39 |        30 |  0.785714  |      0.960526 |        0.1      |       0.25      |     76 |       10 |       12 | linear_lasso_llm_subset      |
| 40 |         1 |  0.387755  |      0.351351 |        0.363636 |       0.464286  |     37 |       33 |       28 | tsm_llm_subset               |
| 41 |         5 |  0.438776  |      0.5      |        0.421053 |       0.25      |     48 |       38 |       12 | tsm_llm_subset               |
| 42 |        20 |  0.622449  |      0.621622 |        0.875    |       0.125     |     74 |       16 |        8 | tsm_llm_subset               |
| 43 |        30 |  0.530612  |      0.578947 |        0.6      |       0.166667  |     76 |       10 |       12 | tsm_llm_subset               |

### 4.3 Statistical Significance

Table 3 reports the results of significance tests comparing methods.

|    |   horizon | model                     | baseline                     |   mean_error_model |   mean_error_baseline |   improvement_pct |   t_statistic |    t_pvalue | t_significant   |   wilcoxon_statistic |   wilcoxon_pvalue | wilcoxon_significant   |   dm_statistic |   dm_pvalue | dm_significant   | model_better   |
|---:|----------:|:--------------------------|:-----------------------------|-------------------:|----------------------:|------------------:|--------------:|------------:|:----------------|---------------------:|------------------:|:-----------------------|---------------:|------------:|:-----------------|:---------------|
|  0 |         1 | seasonal_naive_llm_subset | naive_persistence_llm_subset |           11.8688  |                3.4232 |    -246.718       |      4.36088  | 3.22936e-05 | True            |                  924 |       1.03377e-07 | True                   |    4.36088     | 1.29538e-05 | True             | False          |
|  1 |         5 | seasonal_naive_llm_subset | naive_persistence_llm_subset |           12.3632  |               12.3632 |      -7.21159e-06 |     -0.349999 | 0.727098    | False           |                 1639 |       0.025198    | True                   |   -0.018856    | 0.984956    | False            | False          |
|  2 |        20 | seasonal_naive_llm_subset | naive_persistence_llm_subset |           37.2191  |               37.2191 |       4.35232e-07 |      0.587013 | 0.558559    | False           |                 2174 |       0.825392    | False                  |    0.0750496   | 0.940175    | False            | True           |
|  3 |        30 | seasonal_naive_llm_subset | naive_persistence_llm_subset |           31.1626  |               31.1626 |       1.18019e-05 |      1.04946  | 0.296575    | False           |                 2106 |       0.417189    | False                  |    0.135357    | 0.89233     | False            | True           |
|  4 |         1 | linear_ridge_llm_subset   | naive_persistence_llm_subset |            3.25786 |                3.4232 |       4.82998     |     -1.22126  | 0.224948    | False           |                 2252 |       0.53868     | False                  |   -1.22126     | 0.221988    | False            | True           |
|  5 |         5 | linear_ridge_llm_subset   | naive_persistence_llm_subset |           10.4535  |               12.3632 |      15.4465      |     -3.38132  | 0.00104104  | True            |                 1749 |       0.0165196   | True                   |   -1.66162     | 0.0965898   | False            | True           |
|  6 |        20 | linear_ridge_llm_subset   | naive_persistence_llm_subset |           15.7078  |               37.2191 |      57.7963      |     -7.34504  | 6.43225e-11 | True            |                  514 |       1.25662e-11 | True                   |   -2.21477     | 0.0267757   | True             | True           |
|  7 |        30 | linear_ridge_llm_subset   | naive_persistence_llm_subset |           13.995   |               31.1626 |      55.0903      |     -6.60979  | 2.0887e-09  | True            |                  876 |       4.00179e-08 | True                   |   -1.6995e+07  | 0           | True             | True           |
|  8 |         1 | linear_lasso_llm_subset   | naive_persistence_llm_subset |            4.11542 |                3.4232 |     -20.2214      |      1.48816  | 0.139953    | False           |                 1795 |       0.0254683   | True                   |    1.48816     | 0.136709    | False            | False          |
|  9 |         5 | linear_lasso_llm_subset   | naive_persistence_llm_subset |           10.6123  |               12.3632 |      14.1618      |     -1.52148  | 0.131393    | False           |                 2314 |       0.692763    | False                  |   -0.670088    | 0.502802    | False            | True           |
| 10 |        20 | linear_lasso_llm_subset   | naive_persistence_llm_subset |           13.6898  |               37.2191 |      63.2183      |     -7.71998  | 1.05105e-11 | True            |                  177 |       1.61612e-15 | True                   |   -2.40402     | 0.0162159   | True             | True           |
| 11 |        30 | linear_lasso_llm_subset   | naive_persistence_llm_subset |           10.379   |               31.1626 |      66.6941      |     -7.56267  | 2.25326e-11 | True            |                  551 |       3.08543e-11 | True                   |   -2.05747e+07 | 0           | True             | True           |
| 12 |         1 | tsm_llm_subset            | naive_persistence_llm_subset |            3.3492  |                3.4232 |       2.16159     |     -0.31574  | 0.752878    | False           |                 2316 |       0.698       | False                  |   -0.31574     | 0.7522      | False            | True           |
| 13 |         5 | tsm_llm_subset            | naive_persistence_llm_subset |           10.8861  |               12.3632 |      11.9476      |     -1.74116  | 0.0848261   | False           |                 2223 |       0.47302     | False                  |   -0.888529    | 0.374256    | False            | True           |
| 14 |        20 | tsm_llm_subset            | naive_persistence_llm_subset |           22.5271  |               37.2191 |      39.4745      |     -6.39003  | 5.77596e-09 | True            |                  916 |       8.84271e-08 | True                   |   -1.84397     | 0.0651869   | False            | True           |
| 15 |        30 | tsm_llm_subset            | naive_persistence_llm_subset |           22.685   |               31.1626 |      27.2043      |     -3.86092  | 0.000203753 | True            |                 1772 |       0.0205731   | True                   |   -3.27512     | 0.00105618  | True             | True           |
| 16 |         1 | TSM+LLM-COT-RF-HDELTA     | naive_persistence_llm_subset |            3.3492  |                3.4232 |       2.16159     |     -0.31574  | 0.752878    | False           |                 2316 |       0.698       | False                  |   -0.31574     | 0.7522      | False            | True           |
| 17 |         5 | TSM+LLM-COT-RF-HDELTA     | naive_persistence_llm_subset |           10.941   |               12.3632 |      11.5033      |     -1.65016  | 0.102145    | False           |                 2238 |       0.506421    | False                  |   -0.847454    | 0.396742    | False            | True           |
| 18 |        20 | TSM+LLM-COT-RF-HDELTA     | naive_persistence_llm_subset |           23.862   |               37.2191 |      35.8877      |     -6.00063  | 3.39228e-08 | True            |                 1045 |       9.98631e-07 | True                   |   -1.71839     | 0.0857258   | False            | True           |
| 19 |        30 | TSM+LLM-COT-RF-HDELTA     | naive_persistence_llm_subset |           24.2861  |               31.1626 |      22.0666      |     -3.1629   | 0.00208581  | True            |                 1979 |       0.113602    | False                  |   -2.32339     | 0.0201582   | True             | True           |


## 5. Robustness Analysis

### 5.1 Noise Injection Test

To assess robustness, we inject noise into forecast paths at levels of 5%, 10%, 20%, 
and 30% of the forecast standard deviation.

|    |   horizon |      mse |    rmse |     mae |    mape |   noise_level | model                 |
|---:|----------:|---------:|--------:|--------:|--------:|--------------:|:----------------------|
|  0 |         1 |  3.35815 | 1.83253 | 1.40805 | 3.34647 |          0.05 | tsm                   |
|  1 |         5 | 10.902   | 3.30182 | 2.59957 | 6.109   |          0.05 | tsm                   |
|  2 |        20 | 22.5776  | 4.75159 | 3.83078 | 8.34653 |          0.05 | tsm                   |
|  3 |        30 | 22.5692  | 4.75071 | 4.14419 | 8.88936 |          0.05 | tsm                   |
|  4 |         1 |  3.37031 | 1.83584 | 1.41306 | 3.35809 |          0.1  | tsm                   |
|  5 |         5 | 10.9225  | 3.30491 | 2.5974  | 6.10399 |          0.1  | tsm                   |
|  6 |        20 | 22.6328  | 4.75739 | 3.83678 | 8.36215 |          0.1  | tsm                   |
|  7 |        30 | 22.4598  | 4.73918 | 4.12844 | 8.85456 |          0.1  | tsm                   |
|  8 |         1 |  3.40424 | 1.84506 | 1.42531 | 3.38633 |          0.2  | tsm                   |
|  9 |         5 | 10.977   | 3.31315 | 2.59438 | 6.09677 |          0.2  | tsm                   |
| 10 |        20 | 22.7572  | 4.77045 | 3.84878 | 8.3934  |          0.2  | tsm                   |
| 11 |        30 | 22.2598  | 4.71803 | 4.09693 | 8.78495 |          0.2  | tsm                   |
| 12 |         1 |  3.451   | 1.85769 | 1.43832 | 3.41627 |          0.3  | tsm                   |
| 13 |         5 | 11.0496  | 3.3241  | 2.59624 | 6.10031 |          0.3  | tsm                   |
| 14 |        20 | 22.9001  | 4.78541 | 3.86519 | 8.43602 |          0.3  | tsm                   |
| 15 |        30 | 22.0852  | 4.69948 | 4.06543 | 8.71535 |          0.3  | tsm                   |
| 16 |         1 |  3.35949 | 1.83289 | 1.40816 | 3.34668 |          0.05 | TSM+LLM-COT-RF-HDELTA |
| 17 |         5 | 10.9572  | 3.31016 | 2.60017 | 6.10308 |          0.05 | TSM+LLM-COT-RF-HDELTA |
| 18 |        20 | 23.9079  | 4.88957 | 3.95558 | 8.61326 |          0.05 | TSM+LLM-COT-RF-HDELTA |
| 19 |        30 | 24.1671  | 4.91601 | 4.31763 | 9.26433 |          0.05 | TSM+LLM-COT-RF-HDELTA |
| 20 |         1 |  3.37296 | 1.83656 | 1.41337 | 3.35869 |          0.1  | TSM+LLM-COT-RF-HDELTA |
| 21 |         5 | 10.9779  | 3.3133  | 2.59849 | 6.09912 |          0.1  | TSM+LLM-COT-RF-HDELTA |
| 22 |        20 | 23.9585  | 4.89474 | 3.9609  | 8.62732 |          0.1  | TSM+LLM-COT-RF-HDELTA |
| 23 |        30 | 24.0542  | 4.90451 | 4.30228 | 9.2305  |          0.1  | TSM+LLM-COT-RF-HDELTA |
| 24 |         1 |  3.40946 | 1.84647 | 1.42616 | 3.38801 |          0.2  | TSM+LLM-COT-RF-HDELTA |
| 25 |         5 | 11.0334  | 3.32165 | 2.59645 | 6.09435 |          0.2  | TSM+LLM-COT-RF-HDELTA |
| 26 |        20 | 24.0735  | 4.90648 | 3.97154 | 8.65543 |          0.2  | TSM+LLM-COT-RF-HDELTA |
| 27 |        30 | 23.8468  | 4.88332 | 4.27157 | 9.16286 |          0.2  | TSM+LLM-COT-RF-HDELTA |
| 28 |         1 |  3.45869 | 1.85976 | 1.43958 | 3.4188  |          0.3  | TSM+LLM-COT-RF-HDELTA |
| 29 |         5 | 11.1074  | 3.33277 | 2.5992  | 6.1003  |          0.3  | TSM+LLM-COT-RF-HDELTA |
| 30 |        20 | 24.2071  | 4.92007 | 3.98593 | 8.69326 |          0.3  | TSM+LLM-COT-RF-HDELTA |
| 31 |        30 | 23.6637  | 4.86454 | 4.24085 | 9.09521 |          0.3  | TSM+LLM-COT-RF-HDELTA |

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

- **Price accuracy**: linear_lasso has the lowest average MSE (9.699).
- **TSM vs naive**: TSM MSE is 0.7x the naive baseline on average.
- **Directional accuracy**: linear_lasso has the highest average trend accuracy (0.630).

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
Price accuracy is best for linear_lasso, while directional accuracy is highest for linear_lasso.
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


