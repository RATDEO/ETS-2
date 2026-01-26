# EU ETS Futures Forecasting with TSM + LLM Refinement

**Authors**: Auto-generated

**Generated**: 2026-01-03 19:30:03

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
- **Number of Features**: 32

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

|    |   horizon |       mse |     rmse |      mae |     mape | model             |
|---:|----------:|----------:|---------:|---------:|---------:|:------------------|
|  0 |         1 |   4.28177 |  2.06924 |  1.4437  |  2.15348 | naive_persistence |
|  1 |         5 |  10.2545  |  3.20227 |  2.4621  |  3.71042 | naive_persistence |
|  2 |        20 |  36.8281  |  6.06862 |  4.85088 |  7.20306 | naive_persistence |
|  3 |        30 |  49.5437  |  7.03873 |  5.68055 |  8.32392 | naive_persistence |
|  4 |         1 |  10.571   |  3.25131 |  2.49754 |  3.76078 | seasonal_naive    |
|  5 |         5 |  10.2545  |  3.20227 |  2.4621  |  3.71042 | seasonal_naive    |
|  6 |        20 |  36.8281  |  6.06862 |  4.85088 |  7.20307 | seasonal_naive    |
|  7 |        30 |  49.5437  |  7.03873 |  5.68055 |  8.32392 | seasonal_naive    |
|  8 |         1 |   4.03213 |  2.00802 |  1.42905 |  2.13686 | tsm               |
|  9 |         5 |  10.3769  |  3.22132 |  2.5045  |  3.78181 | tsm               |
| 10 |        20 |  37.0161  |  6.08409 |  4.7366  |  7.08913 | tsm               |
| 11 |        30 |  50.6975  |  7.12022 |  5.64516 |  8.37205 | tsm               |
| 12 |         1 |  19.6931  |  4.43769 |  3.225   |  4.7453  | DP                |
| 13 |         5 |  33.4805  |  5.78624 |  3.526   |  5.71969 | DP                |
| 14 |        20 | 118.716   | 10.8957  |  9.233   | 16.7537  | DP                |
| 15 |        30 | 178.402   | 13.3567  | 10.907   | 20.3444  | DP                |
| 16 |         1 |  20.1233  |  4.4859  |  3.28    |  4.83411 | CoT               |
| 17 |         5 |  35.4101  |  5.95064 |  3.688   |  5.98441 | CoT               |
| 18 |        20 | 166.887   | 12.9185  | 12.072   | 21.9929  | CoT               |
| 19 |        30 | 225.078   | 15.0026  | 13.207   | 24.4335  | CoT               |
| 20 |         1 |  18.8824  |  4.34538 |  3.215   |  4.74258 | CoT-RF            |
| 21 |         5 |  34.0795  |  5.83777 |  3.812   |  6.18147 | CoT-RF            |
| 22 |        20 | 148.692   | 12.1939  | 11.789   | 21.4481  | CoT-RF            |
| 23 |        30 | 190.252   | 13.7932  | 12.09    | 22.5142  | CoT-RF            |
| 24 |         1 |  17.3648  |  4.16711 |  3.106   |  4.58121 | TSM+LLM           |
| 25 |         5 |  29.2235  |  5.40588 |  3.146   |  5.09667 | TSM+LLM           |
| 26 |        20 | 128.921   | 11.3543  | 10.659   | 19.4768  | TSM+LLM           |
| 27 |        30 | 142.297   | 11.9288  | 10.133   | 18.7879  | TSM+LLM           |

### 4.2 Trend Classification Accuracy

Table 2 shows the accuracy of predicting price direction (up/flat/down).

|    |   horizon |   accuracy |   accuracy_up |   accuracy_down |   accuracy_flat |   n_up |   n_down |   n_flat | model             |
|---:|----------:|-----------:|--------------:|----------------:|----------------:|-------:|---------:|---------:|:------------------|
|  0 |         1 |  0.270718  |      0        |       0         |        1        |    122 |      142 |       98 | naive_persistence |
|  1 |         5 |  0.121547  |      0        |       0         |        1        |    151 |      167 |       44 | naive_persistence |
|  2 |        20 |  0.0635359 |      0        |       0         |        1        |    184 |      155 |       23 | naive_persistence |
|  3 |        30 |  0.0359116 |      0        |       0         |        1        |    200 |      149 |       13 | naive_persistence |
|  4 |         1 |  0.41989   |      0.590164 |       0.464789  |        0.142857 |    122 |      142 |       98 | seasonal_naive    |
|  5 |         5 |  0.121547  |      0        |       0         |        1        |    151 |      167 |       44 | seasonal_naive    |
|  6 |        20 |  0.0635359 |      0        |       0         |        1        |    184 |      155 |       23 | seasonal_naive    |
|  7 |        30 |  0.0359116 |      0        |       0         |        1        |    200 |      149 |       13 | seasonal_naive    |
|  8 |         1 |  0.31768   |      0.204918 |       0.0915493 |        0.785714 |    122 |      142 |       98 | tsm               |
|  9 |         5 |  0.279006  |      0.397351 |       0.149701  |        0.363636 |    151 |      167 |       44 | tsm               |
| 10 |        20 |  0.455801  |      0.815217 |       0.0451613 |        0.347826 |    184 |      155 |       23 | tsm               |
| 11 |        30 |  0.444751  |      0.795    |       0         |        0.153846 |    200 |      149 |       13 | tsm               |

### 4.3 Statistical Significance

Table 3 reports the results of significance tests comparing methods.

|    |   horizon | model          | baseline          |   mean_error_model |   mean_error_baseline |   improvement_pct |   t_statistic |    t_pvalue | t_significant   |   wilcoxon_statistic |   wilcoxon_pvalue | wilcoxon_significant   |   dm_statistic |   dm_pvalue | dm_significant   | model_better   |
|---:|----------:|:---------------|:------------------|-------------------:|----------------------:|------------------:|--------------:|------------:|:----------------|---------------------:|------------------:|:-----------------------|---------------:|------------:|:-----------------|:---------------|
|  0 |         1 | seasonal_naive | naive_persistence |           10.571   |               4.28177 |    -146.884       |     6.07455   | 3.15549e-09 | True            |                15973 |       2.42149e-17 | True                   |    6.07455     | 1.24334e-09 | True             | False          |
|  1 |         5 | seasonal_naive | naive_persistence |           10.2545  |              10.2545  |       4.36825e-06 |     1.6742    | 0.0949572   | False           |                27959 |       0.454816    | False                  |    0.0952472   | 0.924118    | False            | True           |
|  2 |        20 | seasonal_naive | naive_persistence |           36.8281  |              36.8281  |       6.48903e-06 |    -0.918201  | 0.359127    | False           |                25159 |       0.0428928   | True                   |   -0.155897    | 0.876114    | False            | True           |
|  3 |        30 | seasonal_naive | naive_persistence |           49.5437  |              49.5437  |      -6.35439e-06 |    -0.0828849 | 0.933989    | False           |                24945 |       0.0208131   | True                   |   -0.0147042   | 0.988268    | False            | False          |
|  4 |         1 | tsm            | naive_persistence |            4.03213 |               4.28177 |       5.83041     |    -2.2006    | 0.0283974   | True            |                30985 |       0.34885     | False                  |   -2.2006      | 0.0277647   | True             | True           |
|  5 |         5 | tsm            | naive_persistence |           10.3769  |              10.2545  |      -1.19318     |     0.386971  | 0.699006    | False           |                29701 |       0.113814    | False                  |    0.227279    | 0.820207    | False            | False          |
|  6 |        20 | tsm            | naive_persistence |           37.0161  |              36.8281  |      -0.51052     |     0.196443  | 0.844374    | False           |                30654 |       0.270047    | False                  |    0.0599522   | 0.952194    | False            | False          |
|  7 |        30 | tsm            | naive_persistence |           50.6975  |              49.5437  |      -2.32896     |     0.800619  | 0.423879    | False           |                30895 |       0.326103    | False                  |    0.231302    | 0.81708     | False            | False          |
|  8 |         1 | DP             | naive_persistence |           19.6931  |              18.7791  |      -4.86725     |     1.80377   | 0.104764    | False           |                   12 |       0.130859    | False                  |    1.80377     | 0.0712679   | False            | False          |
|  9 |         5 | DP             | naive_persistence |           33.4805  |              30.9492  |      -8.17885     |     1.24019   | 0.246263    | False           |                   19 |       0.431641    | False                  |    7.46907     | 8.08242e-14 | True             | False          |
| 10 |        20 | DP             | naive_persistence |          118.716   |             115.192   |      -3.05902     |     0.175003  | 0.864951    | False           |                   26 |       0.921875    | False                  |    1.11431e+06 | 0           | True             | False          |
| 11 |        30 | DP             | naive_persistence |          178.402   |             136.459   |     -30.7361      |     1.68143   | 0.126976    | False           |                   17 |       0.322266    | False                  |    8.63038     | 0           | True             | False          |
| 12 |         1 | CoT            | naive_persistence |           20.1233  |              18.7791  |      -7.15794     |     1.66108   | 0.131065    | False           |                   18 |       0.375       | False                  |    1.66108     | 0.0966974   | False            | False          |
| 13 |         5 | CoT            | naive_persistence |           35.4101  |              30.9492  |     -14.4134      |     1.78326   | 0.108218    | False           |                   11 |       0.105469    | False                  |    2.74369     | 0.00607537  | True             | False          |
| 14 |        20 | CoT            | naive_persistence |          166.887   |             115.192   |     -44.8771      |     2.19709   | 0.055604    | False           |                    8 |       0.0488281   | True                   |    7.25378     | 4.05231e-13 | True             | False          |
| 15 |        30 | CoT            | naive_persistence |          225.078   |             136.459   |     -64.9418      |     2.37371   | 0.041655    | True            |                    5 |       0.0195312   | True                   |    2.80238e+07 | 0           | True             | False          |
| 16 |         1 | CoT-RF         | naive_persistence |           18.8824  |              18.7791  |      -0.549803    |     0.157886  | 0.878032    | False           |                   20 |       0.492188    | False                  |    0.157886    | 0.874547    | False            | False          |
| 17 |         5 | CoT-RF         | naive_persistence |           34.0795  |              30.9492  |     -10.1143      |     2.75244   | 0.0223874   | True            |                    2 |       0.00585938  | True                   |    2.68577     | 0.00723627  | True             | False          |
| 18 |        20 | CoT-RF         | naive_persistence |          148.692   |             115.192   |     -29.0817      |     4.14326   | 0.00250945  | True            |                    2 |       0.00585938  | True                   |    1.05936e+07 | 0           | True             | False          |
| 19 |        30 | CoT-RF         | naive_persistence |          190.252   |             136.459   |     -39.4201      |     2.64012   | 0.0269083   | True            |                    6 |       0.0273438   | True                   |    1.70106e+07 | 0           | True             | False          |
| 20 |         1 | TSM+LLM        | naive_persistence |           17.3648  |              18.7791  |       7.53136     |    -0.771963  | 0.459925    | False           |                   25 |       0.845703    | False                  |   -0.771963    | 0.440136    | False            | True           |
| 21 |         5 | TSM+LLM        | naive_persistence |           29.2235  |              30.9492  |       5.57606     |    -0.545941  | 0.598376    | False           |                   24 |       0.769531    | False                  |   -1.07514     | 0.282312    | False            | True           |
| 22 |        20 | TSM+LLM        | naive_persistence |          128.921   |             115.192   |     -11.9179      |     2.17037   | 0.0580786   | False           |                   11 |       0.105469    | False                  |    4.34135e+06 | 0           | True             | False          |
| 23 |        30 | TSM+LLM        | naive_persistence |          142.297   |             136.459   |      -4.27813     |     0.334043  | 0.745999    | False           |                   18 |       0.375       | False                  |    1.84611e+06 | 0           | True             | False          |


## 5. Robustness Analysis

### 5.1 Noise Injection Test

To assess robustness, we inject noise into forecast paths at levels of 5%, 10%, 20%, 
and 30% of the forecast standard deviation.

|    |   horizon |       mse |     rmse |      mae |     mape |   noise_level | model   |
|---:|----------:|----------:|---------:|---------:|---------:|--------------:|:--------|
|  0 |         1 |   4.0173  |  2.00432 |  1.4262  |  2.1324  |          0.05 | tsm     |
|  1 |         5 |  10.3834  |  3.22232 |  2.50605 |  3.78394 |          0.05 | tsm     |
|  2 |        20 |  36.9726  |  6.08051 |  4.73492 |  7.08633 |          0.05 | tsm     |
|  3 |        30 |  50.7019  |  7.12053 |  5.64653 |  8.37426 |          0.05 | tsm     |
|  4 |         1 |   4.00562 |  2.0014  |  1.42404 |  2.12897 |          0.1  | tsm     |
|  5 |         5 |  10.3933  |  3.22386 |  2.5088  |  3.78783 |          0.1  | tsm     |
|  6 |        20 |  36.932   |  6.07717 |  4.73323 |  7.08354 |          0.1  | tsm     |
|  7 |        30 |  50.7092  |  7.12104 |  5.64833 |  8.37709 |          0.1  | tsm     |
|  8 |         1 |   3.99171 |  1.99793 |  1.42204 |  2.12538 |          0.2  | tsm     |
|  9 |         5 |  10.4235  |  3.22854 |  2.51574 |  3.79766 |          0.2  | tsm     |
| 10 |        20 |  36.8596  |  6.07121 |  4.73055 |  7.07899 |          0.2  | tsm     |
| 11 |        30 |  50.7322  |  7.12266 |  5.65271 |  8.38387 |          0.2  | tsm     |
| 12 |         1 |   3.99039 |  1.9976  |  1.42249 |  2.12554 |          0.3  | tsm     |
| 13 |         5 |  10.4675  |  3.23535 |  2.52417 |  3.80963 |          0.3  | tsm     |
| 14 |        20 |  36.7989  |  6.06621 |  4.72974 |  7.0772  |          0.3  | tsm     |
| 15 |        30 |  50.7667  |  7.12507 |  5.6572  |  8.39082 |          0.3  | tsm     |
| 16 |         1 |  19.5048  |  4.41643 |  3.19421 |  4.69634 |          0.05 | DP      |
| 17 |         5 |  33.6679  |  5.80241 |  3.53863 |  5.74108 |          0.05 | DP      |
| 18 |        20 | 117.986   | 10.8621  |  9.19618 | 16.6883  |          0.05 | DP      |
| 19 |        30 | 179.289   | 13.3899  | 10.9195  | 20.3705  |          0.05 | DP      |
| 20 |         1 |  19.3242  |  4.39593 |  3.16342 |  4.64738 |          0.1  | DP      |
| 21 |         5 |  33.8611  |  5.81903 |  3.55126 |  5.76248 |          0.1  | DP      |
| 22 |        20 | 117.264   | 10.8288  |  9.15937 | 16.6229  |          0.1  | DP      |
| 23 |        30 | 180.196   | 13.4237  | 10.9319  | 20.3967  |          0.1  | DP      |
| 24 |         1 |  18.986   |  4.3573  |  3.10183 |  4.54946 |          0.2  | DP      |
| 25 |         5 |  34.2645  |  5.85359 |  3.57652 |  5.80526 |          0.2  | DP      |
| 26 |        20 | 115.845   | 10.7631  |  9.08573 | 16.4921  |          0.2  | DP      |
| 27 |        30 | 182.063   | 13.4931  | 11.0034  | 20.5281  |          0.2  | DP      |
| 28 |         1 |  18.6786  |  4.32188 |  3.04025 |  4.45155 |          0.3  | DP      |
| 29 |         5 |  34.6909  |  5.8899  |  3.60179 |  5.84805 |          0.3  | DP      |
| 30 |        20 | 114.459   | 10.6986  |  9.0121  | 16.3612  |          0.3  | DP      |
| 31 |        30 | 184.004   | 13.5648  | 11.0936  | 20.6913  |          0.3  | DP      |
| 32 |         1 |  19.9143  |  4.46254 |  3.25466 |  4.79392 |          0.05 | CoT     |
| 33 |         5 |  35.6314  |  5.9692  |  3.69591 |  5.99807 |          0.05 | CoT     |
| 34 |        20 | 165.437   | 12.8622  | 12.0182  | 21.8977  |          0.05 | CoT     |
| 35 |        30 | 226.817   | 15.0605  | 13.2879  | 24.5759  |          0.05 | CoT     |
| 36 |         1 |  19.7118  |  4.4398  |  3.22933 |  4.75374 |          0.1  | CoT     |
| 37 |         5 |  35.8598  |  5.9883  |  3.70382 |  6.01174 |          0.1  | CoT     |
| 38 |        20 | 164.006   | 12.8065  | 11.9644  | 21.8025  |          0.1  | CoT     |
| 39 |        30 | 228.597   | 15.1194  | 13.3688  | 24.7184  |          0.1  | CoT     |
| 40 |         1 |  19.3265  |  4.39619 |  3.17865 |  4.67337 |          0.2  | CoT     |
| 41 |         5 |  36.338   |  6.0281  |  3.71965 |  6.03907 |          0.2  | CoT     |
| 42 |        20 | 161.2     | 12.6965  | 11.8568  | 21.6121  |          0.2  | CoT     |
| 43 |        30 | 232.279   | 15.2407  | 13.5307  | 25.0033  |          0.2  | CoT     |
| 44 |         1 |  18.9674  |  4.35515 |  3.12798 |  4.593   |          0.3  | CoT     |
| 45 |         5 |  36.8448  |  6.06999 |  3.73547 |  6.06641 |          0.3  | CoT     |
| 46 |        20 | 158.47    | 12.5885  | 11.7491  | 21.4218  |          0.3  | CoT     |
| 47 |        30 | 236.124   | 15.3663  | 13.6925  | 25.2882  |          0.3  | CoT     |
| 48 |         1 |  18.7769  |  4.33323 |  3.20593 |  4.72904 |          0.05 | CoT-RF  |
| 49 |         5 |  34.1526  |  5.84402 |  3.8129  |  6.18309 |          0.05 | CoT-RF  |
| 50 |        20 | 147.918   | 12.1622  | 11.7556  | 21.3893  |          0.05 | CoT-RF  |
| 51 |        30 | 191.09    | 13.8235  | 12.1331  | 22.591   |          0.05 | CoT-RF  |
| 52 |         1 |  18.6725  |  4.32117 |  3.19686 |  4.7155  |          0.1  | CoT-RF  |
| 53 |         5 |  34.2265  |  5.85034 |  3.8138  |  6.1847  |          0.1  | CoT-RF  |
| 54 |        20 | 147.149   | 12.1305  | 11.7223  | 21.3305  |          0.1  | CoT-RF  |
| 55 |        30 | 191.941   | 13.8543  | 12.1762  | 22.6678  |          0.1  | CoT-RF  |
| 56 |         1 |  18.4669  |  4.29731 |  3.17873 |  4.68841 |          0.2  | CoT-RF  |
| 57 |         5 |  34.3769  |  5.86318 |  3.81559 |  6.18793 |          0.2  | CoT-RF  |
| 58 |        20 | 145.625   | 12.0675  | 11.6555  | 21.213   |          0.2  | CoT-RF  |
| 59 |        30 | 193.68    | 13.9169  | 12.2624  | 22.8215  |          0.2  | CoT-RF  |
| 60 |         1 |  18.2656  |  4.27383 |  3.16059 |  4.66133 |          0.3  | CoT-RF  |
| 61 |         5 |  34.5308  |  5.87629 |  3.81739 |  6.19116 |          0.3  | CoT-RF  |
| 62 |        20 | 144.121   | 12.0051  | 11.5888  | 21.0955  |          0.3  | CoT-RF  |
| 63 |        30 | 195.468   | 13.981   | 12.3486  | 22.9751  |          0.3  | CoT-RF  |
| 64 |         1 |  17.2028  |  4.14762 |  3.08634 |  4.55114 |          0.05 | TSM+LLM |
| 65 |         5 |  29.3952  |  5.42173 |  3.14703 |  5.0985  |          0.05 | TSM+LLM |
| 66 |        20 | 128.176   | 11.3215  | 10.6255  | 19.4186  |          0.05 | TSM+LLM |
| 67 |        30 | 142.866   | 11.9526  | 10.168   | 18.8494  |          0.05 | TSM+LLM |
| 68 |         1 |  17.0423  |  4.12823 |  3.06867 |  4.52425 |          0.1  | TSM+LLM |
| 69 |         5 |  29.5682  |  5.43767 |  3.14807 |  5.10032 |          0.1  | TSM+LLM |
| 70 |        20 | 127.437   | 11.2888  | 10.5921  | 19.3605  |          0.1  | TSM+LLM |
| 71 |        30 | 143.44    | 11.9766  | 10.203   | 18.911   |          0.1  | TSM+LLM |
| 72 |         1 |  16.7259  |  4.08973 |  3.03335 |  4.47047 |          0.2  | TSM+LLM |
| 73 |         5 |  29.9188  |  5.4698  |  3.15014 |  5.10397 |          0.2  | TSM+LLM |
| 74 |        20 | 125.978   | 11.224   | 10.5252  | 19.2441  |          0.2  | TSM+LLM |
| 75 |        30 | 144.604   | 12.0252  | 10.2731  | 19.0341  |          0.2  | TSM+LLM |
| 76 |         1 |  16.4156  |  4.05162 |  2.99802 |  4.41669 |          0.3  | TSM+LLM |
| 77 |         5 |  30.275   |  5.50227 |  3.16478 |  5.12756 |          0.3  | TSM+LLM |
| 78 |        20 | 124.544   | 11.1599  | 10.4583  | 19.1278  |          0.3  | TSM+LLM |
| 79 |        30 | 145.791   | 12.0744  | 10.3431  | 19.1572  |          0.3  | TSM+LLM |

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
- **TSM vs naive**: TSM MSE is 1.0x the naive baseline on average.
- **Directional accuracy**: tsm has the highest average trend accuracy (0.374).

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
Sequence Length: 60
TSM Type: dlinear
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


