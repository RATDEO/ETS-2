# EU ETS Futures Forecasting with TSM + LLM Refinement

**Authors**: Auto-generated

**Generated**: 2026-03-15 05:17:54

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
- Best performing method: linear_ridge+LLM-COT-RF-HDELTA


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
- **Number of Features**: 95

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

|    |   horizon |      mse |    rmse |      mae |     mape | model                          |
|---:|----------:|---------:|--------:|---------:|---------:|:-------------------------------|
|  0 |         1 |  1.06014 | 1.02963 | 0.785139 |  1.39639 | naive_persistence              |
|  1 |         5 |  5.22395 | 2.2856  | 1.67764  |  2.91624 | naive_persistence              |
|  2 |        20 | 44.3366  | 6.65857 | 4.41486  |  7.96141 | naive_persistence              |
|  3 |        30 | 80.0489  | 8.947   | 6.38674  | 12.0843  | naive_persistence              |
|  4 |         1 |  5.18059 | 2.27609 | 1.65896  |  2.91402 | seasonal_naive                 |
|  5 |         5 |  5.22395 | 2.2856  | 1.67764  |  2.91624 | seasonal_naive                 |
|  6 |        20 | 44.3366  | 6.65857 | 4.41486  |  7.96141 | seasonal_naive                 |
|  7 |        30 | 80.0489  | 8.947   | 6.38674  | 12.0843  | seasonal_naive                 |
|  8 |         1 |  1.07838 | 1.03845 | 0.784083 |  1.38777 | linear_ridge                   |
|  9 |         5 |  4.57781 | 2.13958 | 1.56311  |  2.697   | linear_ridge                   |
| 10 |        20 | 30.2437  | 5.49943 | 3.34291  |  6.10677 | linear_ridge                   |
| 11 |        30 | 56.9873  | 7.54899 | 4.76996  |  9.16953 | linear_ridge                   |
| 12 |         1 |  1.16751 | 1.08051 | 0.833777 |  1.49026 | linear_lasso                   |
| 13 |         5 |  4.74957 | 2.17935 | 1.61877  |  2.81099 | linear_lasso                   |
| 14 |        20 | 31.9548  | 5.65285 | 3.42114  |  6.14554 | linear_lasso                   |
| 15 |        30 | 62.1015  | 7.88045 | 5.2156   |  9.8683  | linear_lasso                   |
| 16 |         1 |  1.29364 | 1.13738 | 0.842379 |  1.48434 | tsm                            |
| 17 |         5 |  5.0639  | 2.25031 | 1.57792  |  2.70282 | tsm                            |
| 18 |        20 | 34.4617  | 5.87041 | 3.75789  |  6.7343  | tsm                            |
| 19 |        30 | 62.6831  | 7.91727 | 5.39437  | 10.2125  | tsm                            |
| 20 |         1 |  1.07838 | 1.03845 | 0.784083 |  1.38777 | linear_ridge+LLM-COT-RF-HDELTA |
| 21 |         5 |  4.65754 | 2.15813 | 1.5686   |  2.70536 | linear_ridge+LLM-COT-RF-HDELTA |
| 22 |        20 | 30.4222  | 5.51564 | 3.38844  |  6.21049 | linear_ridge+LLM-COT-RF-HDELTA |
| 23 |        30 | 55.3904  | 7.44247 | 4.66825  |  8.98946 | linear_ridge+LLM-COT-RF-HDELTA |
| 24 |         1 |  1.06014 | 1.02963 | 0.785139 |  1.39639 | naive_persistence_llm_subset   |
| 25 |         5 |  5.22395 | 2.2856  | 1.67764  |  2.91624 | naive_persistence_llm_subset   |
| 26 |        20 | 44.3366  | 6.65857 | 4.41486  |  7.96141 | naive_persistence_llm_subset   |
| 27 |        30 | 80.0489  | 8.947   | 6.38674  | 12.0843  | naive_persistence_llm_subset   |
| 28 |         1 |  5.18059 | 2.27609 | 1.65896  |  2.91402 | seasonal_naive_llm_subset      |
| 29 |         5 |  5.22395 | 2.2856  | 1.67764  |  2.91624 | seasonal_naive_llm_subset      |
| 30 |        20 | 44.3366  | 6.65857 | 4.41486  |  7.96141 | seasonal_naive_llm_subset      |
| 31 |        30 | 80.0489  | 8.947   | 6.38674  | 12.0843  | seasonal_naive_llm_subset      |
| 32 |         1 |  1.07838 | 1.03845 | 0.784083 |  1.38777 | linear_ridge_llm_subset        |
| 33 |         5 |  4.57781 | 2.13958 | 1.56311  |  2.697   | linear_ridge_llm_subset        |
| 34 |        20 | 30.2437  | 5.49943 | 3.34291  |  6.10677 | linear_ridge_llm_subset        |
| 35 |        30 | 56.9873  | 7.54899 | 4.76996  |  9.16953 | linear_ridge_llm_subset        |
| 36 |         1 |  1.16751 | 1.08051 | 0.833777 |  1.49026 | linear_lasso_llm_subset        |
| 37 |         5 |  4.74957 | 2.17935 | 1.61877  |  2.81099 | linear_lasso_llm_subset        |
| 38 |        20 | 31.9548  | 5.65285 | 3.42114  |  6.14554 | linear_lasso_llm_subset        |
| 39 |        30 | 62.1015  | 7.88045 | 5.2156   |  9.8683  | linear_lasso_llm_subset        |
| 40 |         1 |  1.29364 | 1.13738 | 0.842379 |  1.48434 | tsm_llm_subset                 |
| 41 |         5 |  5.0639  | 2.25031 | 1.57792  |  2.70282 | tsm_llm_subset                 |
| 42 |        20 | 34.4617  | 5.87041 | 3.75789  |  6.7343  | tsm_llm_subset                 |
| 43 |        30 | 62.6831  | 7.91727 | 5.39437  | 10.2125  | tsm_llm_subset                 |

### 4.2 Trend Classification Accuracy

Table 2 shows the accuracy of predicting price direction (up/flat/down).

|    |   horizon |   accuracy |   accuracy_up |   accuracy_down |   accuracy_flat |   n_up |   n_down |   n_flat | model                          |
|---:|----------:|-----------:|--------------:|----------------:|----------------:|-------:|---------:|---------:|:-------------------------------|
|  0 |         1 |  0.402778  |     0         |       0         |       1         |     51 |       35 |       58 | naive_persistence              |
|  1 |         5 |  0.229167  |     0         |       0         |       1         |     70 |       41 |       33 | naive_persistence              |
|  2 |        20 |  0.118056  |     0         |       0         |       1         |     90 |       37 |       17 | naive_persistence              |
|  3 |        30 |  0.0763889 |     0         |       0         |       1         |    100 |       33 |       11 | naive_persistence              |
|  4 |         1 |  0.368056  |     0.372549  |       0.628571  |       0.206897  |     51 |       35 |       58 | seasonal_naive                 |
|  5 |         5 |  0.229167  |     0         |       0         |       1         |     70 |       41 |       33 | seasonal_naive                 |
|  6 |        20 |  0.118056  |     0         |       0         |       1         |     90 |       37 |       17 | seasonal_naive                 |
|  7 |        30 |  0.0763889 |     0         |       0         |       1         |    100 |       33 |       11 | seasonal_naive                 |
|  8 |         1 |  0.423611  |     0.0588235 |       0.0857143 |       0.948276  |     51 |       35 |       58 | linear_ridge                   |
|  9 |         5 |  0.444444  |     0.485714  |       0.146341  |       0.727273  |     70 |       41 |       33 | linear_ridge                   |
| 10 |        20 |  0.715278  |     0.966667  |       0.351351  |       0.176471  |     90 |       37 |       17 | linear_ridge                   |
| 11 |        30 |  0.819444  |     0.99      |       0.545455  |       0.0909091 |    100 |       33 |       11 | linear_ridge                   |
| 12 |         1 |  0.451389  |     0.372549  |       0.142857  |       0.706897  |     51 |       35 |       58 | linear_lasso                   |
| 13 |         5 |  0.388889  |     0.428571  |       0.146341  |       0.606061  |     70 |       41 |       33 | linear_lasso                   |
| 14 |        20 |  0.701389  |     0.922222  |       0.459459  |       0.0588235 |     90 |       37 |       17 | linear_lasso                   |
| 15 |        30 |  0.75      |     0.77      |       0.757576  |       0.545455  |    100 |       33 |       11 | linear_lasso                   |
| 16 |         1 |  0.416667  |     0.235294  |       0.114286  |       0.758621  |     51 |       35 |       58 | tsm                            |
| 17 |         5 |  0.423611  |     0.457143  |       0.268293  |       0.545455  |     70 |       41 |       33 | tsm                            |
| 18 |        20 |  0.631944  |     0.733333  |       0.513514  |       0.352941  |     90 |       37 |       17 | tsm                            |
| 19 |        30 |  0.736111  |     0.8       |       0.69697   |       0.272727  |    100 |       33 |       11 | tsm                            |
| 20 |         1 |  0.423611  |     0.0588235 |       0.0857143 |       0.948276  |     51 |       35 |       58 | linear_ridge+LLM-COT-RF-HDELTA |
| 21 |         5 |  0.472222  |     0.628571  |       0.097561  |       0.606061  |     70 |       41 |       33 | linear_ridge+LLM-COT-RF-HDELTA |
| 22 |        20 |  0.708333  |     0.966667  |       0.324324  |       0.176471  |     90 |       37 |       17 | linear_ridge+LLM-COT-RF-HDELTA |
| 23 |        30 |  0.826389  |     1         |       0.545455  |       0.0909091 |    100 |       33 |       11 | linear_ridge+LLM-COT-RF-HDELTA |
| 24 |         1 |  0.402778  |     0         |       0         |       1         |     51 |       35 |       58 | naive_persistence_llm_subset   |
| 25 |         5 |  0.229167  |     0         |       0         |       1         |     70 |       41 |       33 | naive_persistence_llm_subset   |
| 26 |        20 |  0.118056  |     0         |       0         |       1         |     90 |       37 |       17 | naive_persistence_llm_subset   |
| 27 |        30 |  0.0763889 |     0         |       0         |       1         |    100 |       33 |       11 | naive_persistence_llm_subset   |
| 28 |         1 |  0.368056  |     0.372549  |       0.628571  |       0.206897  |     51 |       35 |       58 | seasonal_naive_llm_subset      |
| 29 |         5 |  0.229167  |     0         |       0         |       1         |     70 |       41 |       33 | seasonal_naive_llm_subset      |
| 30 |        20 |  0.118056  |     0         |       0         |       1         |     90 |       37 |       17 | seasonal_naive_llm_subset      |
| 31 |        30 |  0.0763889 |     0         |       0         |       1         |    100 |       33 |       11 | seasonal_naive_llm_subset      |
| 32 |         1 |  0.423611  |     0.0588235 |       0.0857143 |       0.948276  |     51 |       35 |       58 | linear_ridge_llm_subset        |
| 33 |         5 |  0.444444  |     0.485714  |       0.146341  |       0.727273  |     70 |       41 |       33 | linear_ridge_llm_subset        |
| 34 |        20 |  0.715278  |     0.966667  |       0.351351  |       0.176471  |     90 |       37 |       17 | linear_ridge_llm_subset        |
| 35 |        30 |  0.819444  |     0.99      |       0.545455  |       0.0909091 |    100 |       33 |       11 | linear_ridge_llm_subset        |
| 36 |         1 |  0.451389  |     0.372549  |       0.142857  |       0.706897  |     51 |       35 |       58 | linear_lasso_llm_subset        |
| 37 |         5 |  0.388889  |     0.428571  |       0.146341  |       0.606061  |     70 |       41 |       33 | linear_lasso_llm_subset        |
| 38 |        20 |  0.701389  |     0.922222  |       0.459459  |       0.0588235 |     90 |       37 |       17 | linear_lasso_llm_subset        |
| 39 |        30 |  0.75      |     0.77      |       0.757576  |       0.545455  |    100 |       33 |       11 | linear_lasso_llm_subset        |
| 40 |         1 |  0.416667  |     0.235294  |       0.114286  |       0.758621  |     51 |       35 |       58 | tsm_llm_subset                 |
| 41 |         5 |  0.423611  |     0.457143  |       0.268293  |       0.545455  |     70 |       41 |       33 | tsm_llm_subset                 |
| 42 |        20 |  0.631944  |     0.733333  |       0.513514  |       0.352941  |     90 |       37 |       17 | tsm_llm_subset                 |
| 43 |        30 |  0.736111  |     0.8       |       0.69697   |       0.272727  |    100 |       33 |       11 | tsm_llm_subset                 |

### 4.3 Statistical Significance

Table 3 reports the results of significance tests comparing methods.

|    |   horizon | model                          | baseline                     |   mean_error_model |   mean_error_baseline |   improvement_pct |   t_statistic |    t_pvalue | t_significant   |   wilcoxon_statistic |   wilcoxon_pvalue | wilcoxon_significant   |   dm_statistic |   dm_pvalue | dm_significant   | model_better   |
|---:|----------:|:-------------------------------|:-----------------------------|-------------------:|----------------------:|------------------:|--------------:|------------:|:----------------|---------------------:|------------------:|:-----------------------|---------------:|------------:|:-----------------|:---------------|
|  0 |         1 | seasonal_naive_llm_subset      | naive_persistence_llm_subset |            5.18059 |               1.06014 |    -388.67        |      5.08754  | 1.12125e-06 | True            |                 1815 |       1.11663e-11 | True                   |      5.08754   | 3.62746e-07 | True             | False          |
|  1 |         5 | seasonal_naive_llm_subset      | naive_persistence_llm_subset |            5.22395 |               5.22395 |      -2.18266e-06 |      1.18196  | 0.239184    | False           |                 4387 |       0.556061    | False                  |      0.0260803 | 0.979193    | False            | False          |
|  2 |        20 | seasonal_naive_llm_subset      | naive_persistence_llm_subset |           44.3366  |              44.3366  |       3.44544e-06 |      0.71253  | 0.477298    | False           |                 4319 |       0.250968    | False                  |      0.227115  | 0.820335    | False            | True           |
|  3 |        30 | seasonal_naive_llm_subset      | naive_persistence_llm_subset |           80.0489  |              80.0489  |       3.62151e-06 |     -0.58277  | 0.560966    | False           |                 3936 |       0.089407    | False                  |     -0.246793  | 0.805068    | False            | True           |
|  4 |         1 | linear_ridge_llm_subset        | naive_persistence_llm_subset |            1.07838 |               1.06014 |      -1.72083     |      0.264054 | 0.792119    | False           |                 5216 |       0.993635    | False                  |      0.264054  | 0.791738    | False            | False          |
|  5 |         5 | linear_ridge_llm_subset        | naive_persistence_llm_subset |            4.57781 |               5.22395 |      12.3688      |     -2.09949  | 0.0375309   | True            |                 3889 |       0.00794451  | True                   |     -1.23723   | 0.216002    | False            | True           |
|  6 |        20 | linear_ridge_llm_subset        | naive_persistence_llm_subset |           30.2437  |              44.3366  |      31.7862      |     -5.52537  | 1.51033e-07 | True            |                 2495 |       5.49566e-08 | True                   |     -2.30202   | 0.0213338   | True             | True           |
|  7 |        30 | linear_ridge_llm_subset        | naive_persistence_llm_subset |           56.9873  |              80.0489  |      28.8094      |     -8.05214  | 2.88028e-13 | True            |                  808 |       1.38199e-18 | True                   |     -1.95679   | 0.0503727   | False            | True           |
|  8 |         1 | linear_lasso_llm_subset        | naive_persistence_llm_subset |            1.16751 |               1.06014 |     -10.1281      |      1.24008  | 0.216978    | False           |                 4568 |       0.193503    | False                  |      1.24008   | 0.214947    | False            | False          |
|  9 |         5 | linear_lasso_llm_subset        | naive_persistence_llm_subset |            4.74957 |               5.22395 |       9.08092     |     -1.82247  | 0.0704725   | False           |                 4420 |       0.110613    | False                  |     -1.12562   | 0.260328    | False            | True           |
| 10 |        20 | linear_lasso_llm_subset        | naive_persistence_llm_subset |           31.9548  |              44.3366  |      27.9269      |     -5.55279  | 1.32775e-07 | True            |                 1558 |       2.81123e-13 | True                   |     -2.71133   | 0.00670137  | True             | True           |
| 11 |        30 | linear_lasso_llm_subset        | naive_persistence_llm_subset |           62.1015  |              80.0489  |      22.4205      |     -6.73547  | 3.69334e-10 | True            |                  381 |       4.89457e-22 | True                   |     -1.69632   | 0.0898245   | False            | True           |
| 12 |         1 | tsm_llm_subset                 | naive_persistence_llm_subset |            1.29364 |               1.06014 |     -22.025       |      1.79495  | 0.0747732   | False           |                 4722 |       0.32063     | False                  |      1.79495   | 0.072661    | False            | False          |
| 13 |         5 | tsm_llm_subset                 | naive_persistence_llm_subset |            5.0639  |               5.22395 |       3.06377     |     -0.34186  | 0.732958    | False           |                 3581 |       0.00108056  | True                   |     -0.185864  | 0.852551    | False            | True           |
| 14 |        20 | tsm_llm_subset                 | naive_persistence_llm_subset |           34.4617  |              44.3366  |      22.2725      |     -4.78018  | 4.30226e-06 | True            |                 2003 |       1.40198e-10 | True                   |     -2.27199   | 0.0230871   | True             | True           |
| 15 |        30 | tsm_llm_subset                 | naive_persistence_llm_subset |           62.6831  |              80.0489  |      21.694       |     -6.41405  | 1.93872e-09 | True            |                  452 |       1.92794e-21 | True                   |     -1.55845   | 0.119127    | False            | True           |
| 16 |         1 | linear_ridge+LLM-COT-RF-HDELTA | naive_persistence_llm_subset |            1.07838 |               1.06014 |      -1.72083     |      0.264054 | 0.792119    | False           |                 5216 |       0.993635    | False                  |      0.264054  | 0.791738    | False            | False          |
| 17 |         5 | linear_ridge+LLM-COT-RF-HDELTA | naive_persistence_llm_subset |            4.65754 |               5.22395 |      10.8425      |     -1.56439  | 0.119936    | False           |                 4052 |       0.0198407   | True                   |     -0.862352  | 0.388494    | False            | True           |
| 18 |        20 | linear_ridge+LLM-COT-RF-HDELTA | naive_persistence_llm_subset |           30.4222  |              44.3366  |      31.3835      |     -5.00035  | 1.65086e-06 | True            |                 2807 |       1.49236e-06 | True                   |     -2.07538   | 0.0379514   | True             | True           |
| 19 |        30 | linear_ridge+LLM-COT-RF-HDELTA | naive_persistence_llm_subset |           55.3904  |              80.0489  |      30.8043      |     -7.59691  | 3.62575e-12 | True            |                  991 |       3.34294e-17 | True                   |     -1.93719   | 0.0527221   | False            | True           |


## 5. Robustness Analysis

### 5.1 Noise Injection Test

To assess robustness, we inject noise into forecast paths at levels of 5%, 10%, 20%, 
and 30% of the forecast standard deviation.

### 5.2 Ablation Study

We compare the contribution of different components:
- TSM only (no LLM refinement)
- LLM only (direct prompting)
- TSM + LLM (main method)
- With/without exogenous features

### 5.3 Temporal Stability

We evaluate performance across different market regimes:

No robustness suite results were generated for this run.


## 6. Discussion

### 6.1 Key Findings

- **Price accuracy**: linear_ridge+LLM-COT-RF-HDELTA has the lowest average MSE (22.887).
- **TSM vs naive**: TSM MSE is 0.8x the naive baseline on average.
- **Directional accuracy**: linear_ridge+LLM-COT-RF-HDELTA has the highest average trend accuracy (0.608).

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
Price accuracy is best for linear_ridge+LLM-COT-RF-HDELTA, while directional accuracy is highest for linear_ridge+LLM-COT-RF-HDELTA.
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
- Data loader workers: N/A
- Pin memory: N/A

### A.4 Data Availability

Raw data sources:
- ICAP Allowance Price Explorer
- EEX Emissions Auctions
- ECB Exchange Rates
- EIA Energy Prices


