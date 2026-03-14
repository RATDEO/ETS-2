# EU ETS Futures Forecasting with TSM + LLM Refinement

**Authors**: Auto-generated

**Generated**: 2026-03-11 15:14:07

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
- Dataset spans 2009-04-01 00:00:00 to 2026-01-05 00:00:00
- 4311 daily observations
- Best performing method: tsm


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

- **Date Range**: 2009-04-01 00:00:00 to 2026-01-05 00:00:00
- **Total Observations**: 4,311
- **Number of Features**: 26

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
|  0 |         1 |  0.772863 | 0.879126 | 0.714854 | 0.947148 | naive_persistence            |
|  1 |         5 |  2.68059  | 1.63725  | 1.3134   | 1.73415  | naive_persistence            |
|  2 |        20 |  8.7274   | 2.95422  | 2.47874  | 3.16248  | naive_persistence            |
|  3 |        30 | 19.5016   | 4.41606  | 3.94971  | 4.92961  | naive_persistence            |
|  4 |         1 |  2.78176  | 1.66786  | 1.34476  | 1.78295  | seasonal_naive               |
|  5 |         5 |  2.68059  | 1.63725  | 1.3134   | 1.73415  | seasonal_naive               |
|  6 |        20 |  8.7274   | 2.95422  | 2.47874  | 3.16248  | seasonal_naive               |
|  7 |        30 | 19.5016   | 4.41606  | 3.94971  | 4.92961  | seasonal_naive               |
|  8 |         1 |  0.784744 | 0.885858 | 0.728571 | 0.965012 | linear_ridge                 |
|  9 |         5 |  2.96226  | 1.72112  | 1.38586  | 1.82807  | linear_ridge                 |
| 10 |        20 |  6.12546  | 2.47497  | 2.02295  | 2.57499  | linear_ridge                 |
| 11 |        30 | 25.0238   | 5.00238  | 4.53246  | 5.6454   | linear_ridge                 |
| 12 |         1 |  4.56123  | 2.1357   | 1.95245  | 2.58358  | linear_lasso                 |
| 13 |         5 |  7.83144  | 2.79847  | 2.37871  | 3.11203  | linear_lasso                 |
| 14 |        20 | 19.9609   | 4.46776  | 4.12048  | 5.25701  | linear_lasso                 |
| 15 |        30 | 35.2973   | 5.94115  | 5.52904  | 6.89866  | linear_lasso                 |
| 16 |         1 |  0.794017 | 0.891076 | 0.729556 | 0.967605 | tsm                          |
| 17 |         5 |  2.46441  | 1.56984  | 1.27273  | 1.68479  | tsm                          |
| 18 |        20 |  5.40115  | 2.32404  | 1.87853  | 2.38821  | tsm                          |
| 19 |        30 | 11.6841   | 3.4182   | 2.97811  | 3.70898  | tsm                          |
| 20 |         1 |  0.794017 | 0.891076 | 0.729556 | 0.967605 | TSM+LLM-COT-RF-HDELTA        |
| 21 |         5 |  2.50155  | 1.58163  | 1.28453  | 1.70051  | TSM+LLM-COT-RF-HDELTA        |
| 22 |        20 |  5.36355  | 2.31593  | 1.8597   | 2.36986  | TSM+LLM-COT-RF-HDELTA        |
| 23 |        30 | 11.7179   | 3.42314  | 2.9341   | 3.65702  | TSM+LLM-COT-RF-HDELTA        |
| 24 |         1 |  0.772863 | 0.879126 | 0.714854 | 0.947148 | naive_persistence_llm_subset |
| 25 |         5 |  2.68059  | 1.63725  | 1.3134   | 1.73415  | naive_persistence_llm_subset |
| 26 |        20 |  8.7274   | 2.95422  | 2.47874  | 3.16248  | naive_persistence_llm_subset |
| 27 |        30 | 19.5016   | 4.41606  | 3.94971  | 4.92961  | naive_persistence_llm_subset |
| 28 |         1 |  2.78176  | 1.66786  | 1.34476  | 1.78295  | seasonal_naive_llm_subset    |
| 29 |         5 |  2.68059  | 1.63725  | 1.3134   | 1.73415  | seasonal_naive_llm_subset    |
| 30 |        20 |  8.7274   | 2.95422  | 2.47874  | 3.16248  | seasonal_naive_llm_subset    |
| 31 |        30 | 19.5016   | 4.41606  | 3.94971  | 4.92961  | seasonal_naive_llm_subset    |
| 32 |         1 |  0.784744 | 0.885858 | 0.728571 | 0.965012 | linear_ridge_llm_subset      |
| 33 |         5 |  2.96226  | 1.72112  | 1.38586  | 1.82807  | linear_ridge_llm_subset      |
| 34 |        20 |  6.12546  | 2.47497  | 2.02295  | 2.57499  | linear_ridge_llm_subset      |
| 35 |        30 | 25.0238   | 5.00238  | 4.53246  | 5.6454   | linear_ridge_llm_subset      |
| 36 |         1 |  4.56123  | 2.1357   | 1.95245  | 2.58358  | linear_lasso_llm_subset      |
| 37 |         5 |  7.83144  | 2.79847  | 2.37871  | 3.11203  | linear_lasso_llm_subset      |
| 38 |        20 | 19.9609   | 4.46776  | 4.12048  | 5.25701  | linear_lasso_llm_subset      |
| 39 |        30 | 35.2973   | 5.94115  | 5.52904  | 6.89866  | linear_lasso_llm_subset      |
| 40 |         1 |  0.794017 | 0.891076 | 0.729556 | 0.967605 | tsm_llm_subset               |
| 41 |         5 |  2.46441  | 1.56984  | 1.27273  | 1.68479  | tsm_llm_subset               |
| 42 |        20 |  5.40115  | 2.32404  | 1.87853  | 2.38821  | tsm_llm_subset               |
| 43 |        30 | 11.6841   | 3.4182   | 2.97811  | 3.70898  | tsm_llm_subset               |

### 4.2 Trend Classification Accuracy

Table 2 shows the accuracy of predicting price direction (up/flat/down).

|    |   horizon |   accuracy |   accuracy_up |   accuracy_down |   accuracy_flat |   n_up |   n_down |   n_flat | model                        |
|---:|----------:|-----------:|--------------:|----------------:|----------------:|-------:|---------:|---------:|:-----------------------------|
|  0 |         1 | 0.398058   |      0        |        0        |        1        |     34 |       28 |       41 | naive_persistence            |
|  1 |         5 | 0.223301   |      0        |        0        |        1        |     48 |       32 |       23 | naive_persistence            |
|  2 |        20 | 0.0873786  |      0        |        0        |        1        |     89 |        5 |        9 | naive_persistence            |
|  3 |        30 | 0.00970874 |      0        |        0        |        1        |    100 |        2 |        1 | naive_persistence            |
|  4 |         1 | 0.378641   |      0.470588 |        0.571429 |        0.170732 |     34 |       28 |       41 | seasonal_naive               |
|  5 |         5 | 0.223301   |      0        |        0        |        1        |     48 |       32 |       23 | seasonal_naive               |
|  6 |        20 | 0.0873786  |      0        |        0        |        1        |     89 |        5 |        9 | seasonal_naive               |
|  7 |        30 | 0.00970874 |      0        |        0        |        1        |    100 |        2 |        1 | seasonal_naive               |
|  8 |         1 | 0.398058   |      0        |        0        |        1        |     34 |       28 |       41 | linear_ridge                 |
|  9 |         5 | 0.194175   |      0        |        0.0625   |        0.782609 |     48 |       32 |       23 | linear_ridge                 |
| 10 |        20 | 0.621359   |      0.662921 |        0        |        0.555556 |     89 |        5 |        9 | linear_ridge                 |
| 11 |        30 | 0.0776699  |      0.07     |        0        |        1        |    100 |        2 |        1 | linear_ridge                 |
| 12 |         1 | 0.271845   |      0        |        1        |        0        |     34 |       28 |       41 | linear_lasso                 |
| 13 |         5 | 0.31068    |      0        |        1        |        0        |     48 |       32 |       23 | linear_lasso                 |
| 14 |        20 | 0.0485437  |      0        |        1        |        0        |     89 |        5 |        9 | linear_lasso                 |
| 15 |        30 | 0.0194175  |      0        |        1        |        0        |    100 |        2 |        1 | linear_lasso                 |
| 16 |         1 | 0.398058   |      0        |        0        |        1        |     34 |       28 |       41 | tsm                          |
| 17 |         5 | 0.31068    |      0.208333 |        0        |        0.956522 |     48 |       32 |       23 | tsm                          |
| 18 |        20 | 0.660194   |      0.707865 |        0        |        0.555556 |     89 |        5 |        9 | tsm                          |
| 19 |        30 | 0.902913   |      0.93     |        0        |        0        |    100 |        2 |        1 | tsm                          |
| 20 |         1 | 0.398058   |      0        |        0        |        1        |     34 |       28 |       41 | TSM+LLM-COT-RF-HDELTA        |
| 21 |         5 | 0.31068    |      0.1875   |        0.03125  |        0.956522 |     48 |       32 |       23 | TSM+LLM-COT-RF-HDELTA        |
| 22 |        20 | 0.728155   |      0.786517 |        0        |        0.555556 |     89 |        5 |        9 | TSM+LLM-COT-RF-HDELTA        |
| 23 |        30 | 0.873786   |      0.9      |        0        |        0        |    100 |        2 |        1 | TSM+LLM-COT-RF-HDELTA        |
| 24 |         1 | 0.398058   |      0        |        0        |        1        |     34 |       28 |       41 | naive_persistence_llm_subset |
| 25 |         5 | 0.223301   |      0        |        0        |        1        |     48 |       32 |       23 | naive_persistence_llm_subset |
| 26 |        20 | 0.0873786  |      0        |        0        |        1        |     89 |        5 |        9 | naive_persistence_llm_subset |
| 27 |        30 | 0.00970874 |      0        |        0        |        1        |    100 |        2 |        1 | naive_persistence_llm_subset |
| 28 |         1 | 0.378641   |      0.470588 |        0.571429 |        0.170732 |     34 |       28 |       41 | seasonal_naive_llm_subset    |
| 29 |         5 | 0.223301   |      0        |        0        |        1        |     48 |       32 |       23 | seasonal_naive_llm_subset    |
| 30 |        20 | 0.0873786  |      0        |        0        |        1        |     89 |        5 |        9 | seasonal_naive_llm_subset    |
| 31 |        30 | 0.00970874 |      0        |        0        |        1        |    100 |        2 |        1 | seasonal_naive_llm_subset    |
| 32 |         1 | 0.398058   |      0        |        0        |        1        |     34 |       28 |       41 | linear_ridge_llm_subset      |
| 33 |         5 | 0.194175   |      0        |        0.0625   |        0.782609 |     48 |       32 |       23 | linear_ridge_llm_subset      |
| 34 |        20 | 0.621359   |      0.662921 |        0        |        0.555556 |     89 |        5 |        9 | linear_ridge_llm_subset      |
| 35 |        30 | 0.0776699  |      0.07     |        0        |        1        |    100 |        2 |        1 | linear_ridge_llm_subset      |
| 36 |         1 | 0.271845   |      0        |        1        |        0        |     34 |       28 |       41 | linear_lasso_llm_subset      |
| 37 |         5 | 0.31068    |      0        |        1        |        0        |     48 |       32 |       23 | linear_lasso_llm_subset      |
| 38 |        20 | 0.0485437  |      0        |        1        |        0        |     89 |        5 |        9 | linear_lasso_llm_subset      |
| 39 |        30 | 0.0194175  |      0        |        1        |        0        |    100 |        2 |        1 | linear_lasso_llm_subset      |
| 40 |         1 | 0.398058   |      0        |        0        |        1        |     34 |       28 |       41 | tsm_llm_subset               |
| 41 |         5 | 0.31068    |      0.208333 |        0        |        0.956522 |     48 |       32 |       23 | tsm_llm_subset               |
| 42 |        20 | 0.660194   |      0.707865 |        0        |        0.555556 |     89 |        5 |        9 | tsm_llm_subset               |
| 43 |        30 | 0.902913   |      0.93     |        0        |        0        |    100 |        2 |        1 | tsm_llm_subset               |

### 4.3 Statistical Significance

Table 3 reports the results of significance tests comparing methods.

|    |   horizon | model                     | baseline                     |   mean_error_model |   mean_error_baseline |   improvement_pct |   t_statistic |    t_pvalue | t_significant   |   wilcoxon_statistic |   wilcoxon_pvalue | wilcoxon_significant   |   dm_statistic |   dm_pvalue | dm_significant   | model_better   |
|---:|----------:|:--------------------------|:-----------------------------|-------------------:|----------------------:|------------------:|--------------:|------------:|:----------------|---------------------:|------------------:|:-----------------------|---------------:|------------:|:-----------------|:---------------|
|  0 |         1 | seasonal_naive_llm_subset | naive_persistence_llm_subset |           2.78176  |              0.772863 |    -259.929       |      5.95634  | 3.6995e-08  | True            |                 1000 |       3.38068e-08 | True                   |      5.95634   | 2.57948e-09 | True             | False          |
|  1 |         5 | seasonal_naive_llm_subset | naive_persistence_llm_subset |           2.68059  |              2.68059  |       2.17956e-06 |     -2.9605   | 0.00382008  | True            |                 1752 |       0.0352709   | True                   |     -0.0250733 | 0.979996    | False            | True           |
|  2 |        20 | seasonal_naive_llm_subset | naive_persistence_llm_subset |           8.7274   |              8.7274   |      -1.25151e-06 |     -1.60825  | 0.110871    | False           |                 1661 |       0.002968    | True                   |     -0.0550538 | 0.956096    | False            | False          |
|  3 |        30 | seasonal_naive_llm_subset | naive_persistence_llm_subset |          19.5016   |             19.5016   |       1.70368e-06 |      0.844412 | 0.400415    | False           |                 2467 |       0.841911    | False                  |      0.0480418 | 0.961683    | False            | True           |
|  4 |         1 | linear_ridge_llm_subset   | naive_persistence_llm_subset |           0.784744 |              0.772863 |      -1.53727     |      0.753306 | 0.453001    | False           |                 2504 |       0.56702     | False                  |      0.753306  | 0.451266    | False            | False          |
|  5 |         5 | linear_ridge_llm_subset   | naive_persistence_llm_subset |           2.96226  |              2.68059  |     -10.5076      |      3.34043  | 0.00116948  | True            |                 1795 |       0.00367255  | True                   |      2.5999    | 0.00932516  | True             | False          |
|  6 |        20 | linear_ridge_llm_subset   | naive_persistence_llm_subset |           6.12546  |              8.7274   |      29.8134      |     -7.72229  | 8.12262e-12 | True            |                  662 |       3.30102e-11 | True                   |     -5.42534   | 5.78441e-08 | True             | True           |
|  7 |        30 | linear_ridge_llm_subset   | naive_persistence_llm_subset |          25.0238   |             19.5016   |     -28.3168      |     13.7839   | 4.92212e-25 | True            |                  160 |       1.1909e-16  | True                   |      4.1267    | 3.68007e-05 | True             | False          |
|  8 |         1 | linear_lasso_llm_subset   | naive_persistence_llm_subset |           4.56123  |              0.772863 |    -490.173       |     11.977    | 3.63801e-21 | True            |                  192 |       2.86832e-16 | True                   |     11.977     | 0           | True             | False          |
|  9 |         5 | linear_lasso_llm_subset   | naive_persistence_llm_subset |           7.83144  |              2.68059  |    -192.153       |      9.24089  | 3.89442e-15 | True            |                  526 |       1.44253e-12 | True                   |      5.41442   | 6.14867e-08 | True             | False          |
| 10 |        20 | linear_lasso_llm_subset   | naive_persistence_llm_subset |          19.9609   |              8.7274   |    -128.715       |     18.1715   | 9.38168e-34 | True            |                    6 |       1.4868e-18  | True                   |      5.03097   | 4.87998e-07 | True             | False          |
| 11 |        30 | linear_lasso_llm_subset   | naive_persistence_llm_subset |          35.2973   |             19.5016   |     -80.9967      |     18.9295   | 3.73314e-35 | True            |                    9 |       1.62326e-18 | True                   |      5.5497    | 2.86157e-08 | True             | False          |
| 12 |         1 | tsm_llm_subset            | naive_persistence_llm_subset |           0.794017 |              0.772863 |      -2.73702     |      1.11102  | 0.269174    | False           |                 2238 |       0.147739    | False                  |      1.11102   | 0.266562    | False            | False          |
| 13 |         5 | tsm_llm_subset            | naive_persistence_llm_subset |           2.46441  |              2.68059  |       8.06469     |     -1.94885  | 0.0540609   | False           |                 2212 |       0.125251    | False                  |     -1.28465   | 0.198916    | False            | True           |
| 14 |        20 | tsm_llm_subset            | naive_persistence_llm_subset |           5.40115  |              8.7274   |      38.1127      |     -8.847    | 2.87772e-14 | True            |                  345 |       1.64929e-14 | True                   |     -4.23673   | 2.26795e-05 | True             | True           |
| 15 |        30 | tsm_llm_subset            | naive_persistence_llm_subset |          11.6841   |             19.5016   |      40.0867      |    -13.1606   | 1.02574e-23 | True            |                  118 |       3.69505e-17 | True                   |     -8.21186   | 2.22045e-16 | True             | True           |
| 16 |         1 | TSM+LLM-COT-RF-HDELTA     | naive_persistence_llm_subset |           0.794017 |              0.772863 |      -2.73702     |      1.11102  | 0.269174    | False           |                 2238 |       0.147739    | False                  |      1.11102   | 0.266562    | False            | False          |
| 17 |         5 | TSM+LLM-COT-RF-HDELTA     | naive_persistence_llm_subset |           2.50155  |              2.68059  |       6.67927     |     -1.54462  | 0.125535    | False           |                 2315 |       0.232384    | False                  |     -1.02361   | 0.306018    | False            | True           |
| 18 |        20 | TSM+LLM-COT-RF-HDELTA     | naive_persistence_llm_subset |           5.36355  |              8.7274   |      38.5435      |     -8.54981  | 1.29338e-13 | True            |                  344 |       1.60748e-14 | True                   |     -4.92367   | 8.49364e-07 | True             | True           |
| 19 |        30 | TSM+LLM-COT-RF-HDELTA     | naive_persistence_llm_subset |          11.7179   |             19.5016   |      39.9133      |    -12.5922   | 1.69333e-22 | True            |                  102 |       2.35422e-17 | True                   |     -7.26315   | 3.78142e-13 | True             | True           |


## 5. Robustness Analysis

### 5.1 Noise Injection Test

To assess robustness, we inject noise into forecast paths at levels of 5%, 10%, 20%, 
and 30% of the forecast standard deviation.

|    |   horizon |       mse |     rmse |      mae |     mape |   noise_level | model                 |
|---:|----------:|----------:|---------:|---------:|---------:|--------------:|:----------------------|
|  0 |         1 |  0.797723 | 0.893154 | 0.732574 | 0.971505 |          0.05 | tsm                   |
|  1 |         5 |  2.46746  | 1.57082  | 1.27531  | 1.68825  |          0.05 | tsm                   |
|  2 |        20 |  5.39197  | 2.32206  | 1.87737  | 2.3868   |          0.05 | tsm                   |
|  3 |        30 | 11.666    | 3.41556  | 2.97681  | 3.70776  |          0.05 | tsm                   |
|  4 |         1 |  0.802145 | 0.895626 | 0.735593 | 0.975404 |          0.1  | tsm                   |
|  5 |         5 |  2.47123  | 1.57201  | 1.2779   | 1.6917   |          0.1  | tsm                   |
|  6 |        20 |  5.38359  | 2.32026  | 1.87622  | 2.38539  |          0.1  | tsm                   |
|  7 |        30 | 11.6491   | 3.41308  | 2.97552  | 3.70653  |          0.1  | tsm                   |
|  8 |         1 |  0.813136 | 0.90174  | 0.741629 | 0.983204 |          0.2  | tsm                   |
|  9 |         5 |  2.48091  | 1.57509  | 1.28307  | 1.69862  |          0.2  | tsm                   |
| 10 |        20 |  5.36924  | 2.31716  | 1.87493  | 2.38402  |          0.2  | tsm                   |
| 11 |        30 | 11.6186   | 3.4086   | 2.973    | 3.70415  |          0.2  | tsm                   |
| 12 |         1 |  0.826987 | 0.909388 | 0.747816 | 0.991211 |          0.3  | tsm                   |
| 13 |         5 |  2.49345  | 1.57906  | 1.2883   | 1.7056   |          0.3  | tsm                   |
| 14 |        20 |  5.35809  | 2.31476  | 1.8748   | 2.38428  |          0.3  | tsm                   |
| 15 |        30 | 11.5924   | 3.40477  | 2.9708   | 3.70219  |          0.3  | tsm                   |
| 16 |         1 |  0.798151 | 0.893393 | 0.732843 | 0.9718   |          0.05 | TSM+LLM-COT-RF-HDELTA |
| 17 |         5 |  2.5056   | 1.58291  | 1.28681  | 1.70352  |          0.05 | TSM+LLM-COT-RF-HDELTA |
| 18 |        20 |  5.35658  | 2.31443  | 1.85963  | 2.37     |          0.05 | TSM+LLM-COT-RF-HDELTA |
| 19 |        30 | 11.6944   | 3.41971  | 2.93122  | 3.65369  |          0.05 | TSM+LLM-COT-RF-HDELTA |
| 20 |         1 |  0.803077 | 0.896146 | 0.73613  | 0.975995 |          0.1  | TSM+LLM-COT-RF-HDELTA |
| 21 |         5 |  2.5104   | 1.58442  | 1.28943  | 1.707    |          0.1  | TSM+LLM-COT-RF-HDELTA |
| 22 |        20 |  5.35041  | 2.31309  | 1.85966  | 2.37027  |          0.1  | TSM+LLM-COT-RF-HDELTA |
| 23 |        30 | 11.672    | 3.41644  | 2.92834  | 3.65036  |          0.1  | TSM+LLM-COT-RF-HDELTA |
| 24 |         1 |  0.815302 | 0.90294  | 0.742705 | 0.984384 |          0.2  | TSM+LLM-COT-RF-HDELTA |
| 25 |         5 |  2.52225  | 1.58816  | 1.29483  | 1.71419  |          0.2  | TSM+LLM-COT-RF-HDELTA |
| 26 |        20 |  5.34044  | 2.31094  | 1.86113  | 2.37276  |          0.2  | TSM+LLM-COT-RF-HDELTA |
| 27 |        30 | 11.6305   | 3.41035  | 2.9226   | 3.64373  |          0.2  | TSM+LLM-COT-RF-HDELTA |
| 28 |         1 |  0.83069  | 0.911422 | 0.749466 | 0.993032 |          0.3  | TSM+LLM-COT-RF-HDELTA |
| 29 |         5 |  2.53712  | 1.59283  | 1.30023  | 1.72137  |          0.3  | TSM+LLM-COT-RF-HDELTA |
| 30 |        20 |  5.33365  | 2.30947  | 1.86278  | 2.37548  |          0.3  | TSM+LLM-COT-RF-HDELTA |
| 31 |        30 | 11.5932   | 3.40487  | 2.91768  | 3.63817  |          0.3  | TSM+LLM-COT-RF-HDELTA |

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

- **Price accuracy**: tsm has the lowest average MSE (5.086).
- **TSM vs naive**: TSM MSE is 0.6x the naive baseline on average.
- **Directional accuracy**: TSM+LLM-COT-RF-HDELTA has the highest average trend accuracy (0.578).

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
Price accuracy is best for tsm, while directional accuracy is highest for TSM+LLM-COT-RF-HDELTA.
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


