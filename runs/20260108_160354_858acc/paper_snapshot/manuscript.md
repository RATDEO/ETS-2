# EU ETS Futures Forecasting with TSM + LLM Refinement

**Authors**: Auto-generated

**Generated**: 2026-01-08 16:18:20

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
- **Number of Features**: 39

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

where $L = 60$ (lookback window) and $H = 30$ (forecast horizon).

### 3.2 Baseline Models

We implement several baseline models for comparison:

1. **Naive Persistence**: $\hat{y}_{t+h} = y_t$ for all horizons
2. **Seasonal Naive**: $\hat{y}_{t+h} = y_{t+h-5}$ (weekly seasonality)
3. **Linear Regression**: Ridge regression on lagged features
4. **ARIMA**: Autoregressive integrated moving average model

### 3.3 Time Series Model (TSM)

The primary TSM uses an Autoformer-style architecture with:
- **Model dimension**: 64
- **Attention heads**: 2
- **Encoder layers**: 1
- **Feed-forward dimension**: 128
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

|    |   horizon |        mse |     rmse |      mae |     mape | model                        |
|---:|----------:|-----------:|---------:|---------:|---------:|:-----------------------------|
|  0 |         1 |  1.41603   | 1.18997  | 0.963454 | 1.36058  | naive_persistence            |
|  1 |         5 |  6.54358   | 2.55804  | 2.04373  | 2.89746  | naive_persistence            |
|  2 |        20 | 25.939     | 5.09303  | 3.97253  | 5.52109  | naive_persistence            |
|  3 |        30 | 37.7942    | 6.1477   | 4.98591  | 6.84486  | naive_persistence            |
|  4 |         1 |  6.64202   | 2.57721  | 2.0671   | 2.93212  | seasonal_naive               |
|  5 |         5 |  6.54358   | 2.55804  | 2.04373  | 2.89746  | seasonal_naive               |
|  6 |        20 | 25.939     | 5.09303  | 3.97253  | 5.52109  | seasonal_naive               |
|  7 |        30 | 37.7942    | 6.1477   | 4.98591  | 6.84486  | seasonal_naive               |
|  8 |         1 |  1.43079   | 1.19615  | 0.969301 | 1.36841  | linear_ridge                 |
|  9 |         5 |  6.77679   | 2.60323  | 2.07369  | 2.94511  | linear_ridge                 |
| 10 |        20 | 26.9499    | 5.19132  | 3.96345  | 5.61785  | linear_ridge                 |
| 11 |        30 | 35.8673    | 5.98893  | 4.75943  | 6.63854  | linear_ridge                 |
| 12 |         1 |  5.24942   | 2.29116  | 1.99612  | 2.77277  | linear_lasso                 |
| 13 |         5 | 10.0643    | 3.17243  | 2.63507  | 3.65352  | linear_lasso                 |
| 14 |        20 | 28.2284    | 5.31304  | 4.42377  | 6.08084  | linear_lasso                 |
| 15 |        30 | 37.7403    | 6.14331  | 5.10021  | 6.96977  | linear_lasso                 |
| 16 |         1 |  0.0974587 | 0.312184 | 0.243567 | 0.341986 | tsm                          |
| 17 |         5 |  1.89011   | 1.37481  | 1.04876  | 1.47156  | tsm                          |
| 18 |        20 | 19.2321    | 4.38545  | 3.38502  | 4.72689  | tsm                          |
| 19 |        30 | 40.765     | 6.38475  | 5.02288  | 6.99315  | tsm                          |
| 20 |         1 |  3.12008   | 1.76637  | 1.0512   | 1.48022  | TSM+LLM-COT-RF               |
| 21 |         5 |  3.72084   | 1.92895  | 1.4153   | 2.0172   | TSM+LLM-COT-RF               |
| 22 |        20 | 15.9856    | 3.9982   | 3.0416   | 4.31012  | TSM+LLM-COT-RF               |
| 23 |        30 | 61.15      | 7.81985  | 6.1998   | 8.9277   | TSM+LLM-COT-RF               |
| 24 |         1 |  1.60737   | 1.26782  | 1.0618   | 1.52048  | naive_persistence_llm_subset |
| 25 |         5 |  8.20252   | 2.864    | 2.3981   | 3.41566  | naive_persistence_llm_subset |
| 26 |        20 | 29.5019    | 5.43157  | 4.5637   | 6.46288  | naive_persistence_llm_subset |
| 27 |        30 | 36.6221    | 6.05162  | 5.0147   | 7.19351  | naive_persistence_llm_subset |
| 28 |         1 |  9.68672   | 3.11235  | 2.6153   | 3.74072  | seasonal_naive_llm_subset    |
| 29 |         5 |  8.20252   | 2.864    | 2.3981   | 3.41566  | seasonal_naive_llm_subset    |
| 30 |        20 | 29.5019    | 5.43157  | 4.5637   | 6.46288  | seasonal_naive_llm_subset    |
| 31 |        30 | 36.6221    | 6.05162  | 5.0147   | 7.19351  | seasonal_naive_llm_subset    |
| 32 |         1 |  1.65977   | 1.28832  | 1.06328  | 1.52157  | linear_ridge_llm_subset      |
| 33 |         5 |  9.0197    | 3.00328  | 2.48627  | 3.54271  | linear_ridge_llm_subset      |
| 34 |        20 | 40.1465    | 6.33612  | 5.38789  | 7.67675  | linear_ridge_llm_subset      |
| 35 |        30 | 50.7189    | 7.12172  | 5.88859  | 8.56612  | linear_ridge_llm_subset      |
| 36 |         1 |  5.15281   | 2.26998  | 1.94272  | 2.76732  | linear_lasso_llm_subset      |
| 37 |         5 | 12.0827    | 3.47602  | 2.89839  | 4.079    | linear_lasso_llm_subset      |
| 38 |        20 | 37.2617    | 6.10423  | 5.1811   | 7.25324  | linear_lasso_llm_subset      |
| 39 |        30 | 45.3235    | 6.73227  | 5.73056  | 8.21553  | linear_lasso_llm_subset      |
| 40 |         1 |  0.166419  | 0.407945 | 0.327969 | 0.462165 | tsm_llm_subset               |
| 41 |         5 |  3.32726   | 1.82408  | 1.49252  | 2.09253  | tsm_llm_subset               |
| 42 |        20 | 26.1638    | 5.11505  | 4.26148  | 6.0184   | tsm_llm_subset               |
| 43 |        30 | 50.1414    | 7.08106  | 5.76622  | 8.33668  | tsm_llm_subset               |

### 4.2 Trend Classification Accuracy

Table 2 shows the accuracy of predicting price direction (up/flat/down).

|    |   horizon |   accuracy |   accuracy_up |   accuracy_down |   accuracy_flat |   n_up |   n_down |   n_flat | model             |
|---:|----------:|-----------:|--------------:|----------------:|----------------:|-------:|---------:|---------:|:------------------|
|  0 |         1 |  0.295265  |    0          |       0         |       1         |    124 |      129 |      106 | naive_persistence |
|  1 |         5 |  0.142061  |    0          |       0         |       1         |    162 |      146 |       51 | naive_persistence |
|  2 |        20 |  0.091922  |    0          |       0         |       1         |    212 |      114 |       33 | naive_persistence |
|  3 |        30 |  0.0584958 |    0          |       0         |       1         |    224 |      114 |       21 | naive_persistence |
|  4 |         1 |  0.345404  |    0.427419   |       0.44186   |       0.132075  |    124 |      129 |      106 | seasonal_naive    |
|  5 |         5 |  0.142061  |    0          |       0         |       1         |    162 |      146 |       51 | seasonal_naive    |
|  6 |        20 |  0.091922  |    0          |       0         |       1         |    212 |      114 |       33 | seasonal_naive    |
|  7 |        30 |  0.0584958 |    0          |       0         |       1         |    224 |      114 |       21 | seasonal_naive    |
|  8 |         1 |  0.295265  |    0.00806452 |       0         |       0.990566  |    124 |      129 |      106 | linear_ridge      |
|  9 |         5 |  0.256267  |    0.290123   |       0.0616438 |       0.705882  |    162 |      146 |       51 | linear_ridge      |
| 10 |        20 |  0.526462  |    0.853774   |       0.0614035 |       0.030303  |    212 |      114 |       33 | linear_ridge      |
| 11 |        30 |  0.571031  |    0.839286   |       0.105263  |       0.238095  |    224 |      114 |       21 | linear_ridge      |
| 12 |         1 |  0.359331  |    0          |       1         |       0         |    124 |      129 |      106 | linear_lasso      |
| 13 |         5 |  0.409471  |    0          |       1         |       0.0196078 |    162 |      146 |       51 | linear_lasso      |
| 14 |        20 |  0.289694  |    0.0990566  |       0.657895  |       0.242424  |    212 |      114 |       33 | linear_lasso      |
| 15 |        30 |  0.367688  |    0.258929   |       0.570175  |       0.428571  |    224 |      114 |       21 | linear_lasso      |
| 16 |         1 |  0.832869  |    0.862903   |       0.829457  |       0.801887  |    124 |      129 |      106 | tsm               |
| 17 |         5 |  0.752089  |    0.802469   |       0.773973  |       0.529412  |    162 |      146 |       51 | tsm               |
| 18 |        20 |  0.579387  |    0.679245   |       0.491228  |       0.242424  |    212 |      114 |       33 | tsm               |
| 19 |        30 |  0.51532   |    0.633929   |       0.350877  |       0.142857  |    224 |      114 |       21 | tsm               |

### 4.3 Statistical Significance

Table 3 reports the results of significance tests comparing methods.

|    |   horizon | model                     | baseline                     |   mean_error_model |   mean_error_baseline |   improvement_pct |   t_statistic |    t_pvalue | t_significant   |   wilcoxon_statistic |   wilcoxon_pvalue | wilcoxon_significant   |   dm_statistic |   dm_pvalue | dm_significant   | model_better   |
|---:|----------:|:--------------------------|:-----------------------------|-------------------:|----------------------:|------------------:|--------------:|------------:|:----------------|---------------------:|------------------:|:-----------------------|---------------:|------------:|:-----------------|:---------------|
|  0 |         1 | seasonal_naive_llm_subset | naive_persistence_llm_subset |           9.68672  |               1.60737 |    -502.643       |      7.53708  | 2.32122e-11 | True            |                  460 |       1.2463e-12  | True                   |     7.53708    | 4.79616e-14 | True             | False          |
|  1 |         5 | seasonal_naive_llm_subset | naive_persistence_llm_subset |           8.20252  |               8.20252 |      -8.45393e-07 |      0.631766 | 0.528996    | False           |                 2191 |       0.504428    | False                  |     0.0168908  | 0.986524    | False            | False          |
|  2 |        20 | seasonal_naive_llm_subset | naive_persistence_llm_subset |          29.5019   |              29.5019  |      -4.85025e-06 |     -1.44194  | 0.152475    | False           |                 1833 |       0.0357451   | True                   |    -0.194836   | 0.845521    | False            | False          |
|  3 |        30 | seasonal_naive_llm_subset | naive_persistence_llm_subset |          36.6221   |              36.6221  |       8.13018e-06 |     -0.522872 | 0.602232    | False           |                 2070 |       0.345694    | False                  |    -0.065127   | 0.948073    | False            | True           |
|  4 |         1 | linear_ridge_llm_subset   | naive_persistence_llm_subset |           1.65977  |               1.60737 |      -3.25998     |      0.936392 | 0.35135     | False           |                 2471 |       0.852704    | False                  |     0.936392   | 0.349072    | False            | False          |
|  5 |         5 | linear_ridge_llm_subset   | naive_persistence_llm_subset |           9.0197   |               8.20252 |      -9.96252     |      1.96413  | 0.0523193   | False           |                 2046 |       0.0995668   | False                  |     0.844557   | 0.398358    | False            | False          |
|  6 |        20 | linear_ridge_llm_subset   | naive_persistence_llm_subset |          40.1465   |              29.5019  |     -36.0809      |      4.45817  | 2.18288e-05 | True            |                 1426 |       0.000157637 | True                   |     1.01884    | 0.308278    | False            | False          |
|  7 |        30 | linear_ridge_llm_subset   | naive_persistence_llm_subset |          50.7189   |              36.6221  |     -38.4927      |      3.97846  | 0.00013219  | True            |                 1832 |       0.0171834   | True                   |     1.1749     | 0.240036    | False            | False          |
|  8 |         1 | linear_lasso_llm_subset   | naive_persistence_llm_subset |           5.15281  |               1.60737 |    -220.573       |      7.5413   | 2.27401e-11 | True            |                  749 |       1.01862e-09 | True                   |     7.5413     | 4.66294e-14 | True             | False          |
|  9 |         5 | linear_lasso_llm_subset   | naive_persistence_llm_subset |          12.0827   |               8.20252 |     -47.3052      |      4.0292   | 0.000109917 | True            |                 1490 |       0.000372743 | True                   |     1.97617    | 0.0481358   | True             | False          |
| 10 |        20 | linear_lasso_llm_subset   | naive_persistence_llm_subset |          37.2617   |              29.5019  |     -26.3026      |      3.64255  | 0.000431815 | True            |                 1494 |       0.000392743 | True                   |     1.42391    | 0.154473    | False            | False          |
| 11 |        30 | linear_lasso_llm_subset   | naive_persistence_llm_subset |          45.3235   |              36.6221  |     -23.7601      |      4.39969  | 2.73675e-05 | True            |                 1327 |       3.80333e-05 | True                   |     1.22246    | 0.221533    | False            | False          |
| 12 |         1 | tsm_llm_subset            | naive_persistence_llm_subset |           0.166419 |               1.60737 |      89.6465      |     -8.05284  | 1.84946e-12 | True            |                  223 |       2.47206e-15 | True                   |    -8.05284    | 8.88178e-16 | True             | True           |
| 13 |         5 | tsm_llm_subset            | naive_persistence_llm_subset |           3.32726  |               8.20252 |      59.4362      |     -5.62822  | 1.69947e-07 | True            |                 1060 |       4.72525e-07 | True                   |    -4.60733    | 4.07875e-06 | True             | True           |
| 14 |        20 | tsm_llm_subset            | naive_persistence_llm_subset |          26.1638   |              29.5019  |      11.3149      |     -1.51605  | 0.132692    | False           |                 2228 |       0.307168    | False                  |    -2.0269     | 0.0426724   | True             | True           |
| 15 |        30 | tsm_llm_subset            | naive_persistence_llm_subset |          50.1414   |              36.6221  |     -36.9158      |      3.1451   | 0.00219248  | True            |                 1795 |       0.0120739   | True                   |     2.59523    | 0.00945279  | True             | False          |
| 16 |         1 | TSM+LLM-COT-RF            | naive_persistence_llm_subset |           3.12008  |               1.60737 |     -94.1103      |      1.49844  | 0.1372      | False           |                 2184 |       0.241009    | False                  |     1.49844    | 0.134018    | False            | False          |
| 17 |         5 | TSM+LLM-COT-RF            | naive_persistence_llm_subset |           3.72084  |               8.20252 |      54.6378      |     -4.48917  | 1.93496e-05 | True            |                 1320 |       3.42521e-05 | True                   |    -4.02887    | 5.60445e-05 | True             | True           |
| 18 |        20 | TSM+LLM-COT-RF            | naive_persistence_llm_subset |          15.9856   |              29.5019  |      45.815       |     -3.61397  | 0.000476086 | True            |                 1412 |       0.000129791 | True                   |    -4.1508     | 3.31309e-05 | True             | True           |
| 19 |        30 | TSM+LLM-COT-RF            | naive_persistence_llm_subset |          61.15     |              36.6221  |     -66.9759      |      2.25393  | 0.0264049   | True            |                 2213 |       0.28338     | False                  |     2.4528e+07 | 0           | True             | False          |


## 5. Robustness Analysis

### 5.1 Noise Injection Test

To assess robustness, we inject noise into forecast paths at levels of 5%, 10%, 20%, 
and 30% of the forecast standard deviation.

|    |   horizon |        mse |      rmse |      mae |      mape |   noise_level | model          |
|---:|----------:|-----------:|----------:|---------:|----------:|--------------:|:---------------|
|  0 |         1 |   0.106027 |  0.325618 | 0.252294 |  0.354599 |          0.05 | tsm            |
|  1 |         5 |   1.88481  |  1.37288  | 1.05032  |  1.47396  |          0.05 | tsm            |
|  2 |        20 |  19.2307   |  4.38528  | 3.38717  |  4.7299   |          0.05 | tsm            |
|  3 |        30 |  40.718    |  6.38107  | 5.01641  |  6.98411  |          0.05 | tsm            |
|  4 |         1 |   0.128185 |  0.358029 | 0.27776  |  0.390995 |          0.1  | tsm            |
|  5 |         5 |   1.89516  |  1.37665  | 1.05649  |  1.48297  |          0.1  | tsm            |
|  6 |        20 |  19.2416   |  4.38653  | 3.39037  |  4.73418  |          0.1  | tsm            |
|  7 |        30 |  40.683    |  6.37832  | 5.01188  |  6.97805  |          0.1  | tsm            |
|  8 |         1 |   0.213268 |  0.461809 | 0.354609 |  0.500396 |          0.2  | tsm            |
|  9 |         5 |   1.96284  |  1.40101  | 1.08268  |  1.52055  |          0.2  | tsm            |
| 10 |        20 |  19.3007   |  4.39325  | 3.39904  |  4.74613  |          0.2  | tsm            |
| 11 |        30 |  40.6488   |  6.37564  | 5.0059   |  6.97032  |          0.2  | tsm            |
| 12 |         1 |   0.352708 |  0.593892 | 0.45259  |  0.639237 |          0.3  | tsm            |
| 13 |         5 |   2.09313  |  1.44676  | 1.12078  |  1.57466  |          0.3  | tsm            |
| 14 |        20 |  19.4092   |  4.40558  | 3.41193  |  4.764    |          0.3  | tsm            |
| 15 |        30 |  40.6624   |  6.37671  | 5.00313  |  6.96678  |          0.3  | tsm            |
| 16 |         1 |  44.2807   |  6.65438  | 5.23264  |  7.96964  |          0.05 | TSM+LLM-COT-RF |
| 17 |         5 |  47.3008   |  6.87756  | 5.29899  |  8.08095  |          0.05 | TSM+LLM-COT-RF |
| 18 |        20 |  75.9712   |  8.71615  | 6.68792  | 10.1254   |          0.05 | TSM+LLM-COT-RF |
| 19 |        30 | 117.344    | 10.8325   | 9.05861  | 13.6546   |          0.05 | TSM+LLM-COT-RF |
| 20 |         1 |  44.45     |  6.66708  | 5.25538  |  8.00397  |          0.1  | TSM+LLM-COT-RF |
| 21 |         5 |  47.3153   |  6.87861  | 5.28768  |  8.06385  |          0.1  | TSM+LLM-COT-RF |
| 22 |        20 |  76.3097   |  8.73554  | 6.70384  | 10.15     |          0.1  | TSM+LLM-COT-RF |
| 23 |        30 | 117.506    | 10.84     | 9.05013  | 13.6438   |          0.1  | TSM+LLM-COT-RF |
| 24 |         1 |  44.9159   |  6.70193  | 5.3059   |  8.08008  |          0.2  | TSM+LLM-COT-RF |
| 25 |         5 |  47.5082   |  6.89262  | 5.27501  |  8.0446   |          0.2  | TSM+LLM-COT-RF |
| 26 |        20 |  77.1017   |  8.78076  | 6.73569  | 10.1992   |          0.2  | TSM+LLM-COT-RF |
| 27 |        30 | 117.993    | 10.8625   | 9.03805  | 13.6289   |          0.2  | TSM+LLM-COT-RF |
| 28 |         1 |  45.5515   |  6.74919  | 5.35809  |  8.15868  |          0.3  | TSM+LLM-COT-RF |
| 29 |         5 |  47.9199   |  6.92242  | 5.28427  |  8.05881  |          0.3  | TSM+LLM-COT-RF |
| 30 |        20 |  78.047    |  8.83442  | 6.77462  | 10.2583   |          0.3  | TSM+LLM-COT-RF |
| 31 |        30 | 118.699    | 10.8949   | 9.02885  | 13.6181   |          0.3  | TSM+LLM-COT-RF |

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

- **Price accuracy**: tsm has the lowest average MSE (15.496).
- **TSM vs naive**: TSM MSE is 0.9x the naive baseline on average.
- **Directional accuracy**: tsm has the highest average trend accuracy (0.670).

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
Price accuracy is best for tsm, while directional accuracy is highest for tsm.
The framework remains suitable for future runs with full LLM and robustness evaluations enabled.


## Appendix A: Reproducibility

### A.1 Configuration

```yaml
# Key configuration parameters
Random Seed: 42
Prediction Length: 30
Sequence Length: 60
TSM Type: autoformer
LLM Model: gpt-5.2
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


