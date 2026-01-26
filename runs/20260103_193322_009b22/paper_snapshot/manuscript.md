# EU ETS Futures Forecasting with TSM + LLM Refinement

**Authors**: Auto-generated

**Generated**: 2026-01-03 19:37:35

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
| 12 |         1 |  17.2841  |  4.15742 |  3.04    |  4.47298 | DP                |
| 13 |         5 |  28.4284  |  5.33183 |  4.018   |  6.51531 | DP                |
| 14 |        20 | 186.193   | 13.6453  | 13.467   | 24.6261  | DP                |
| 15 |        30 | 245.545   | 15.6699  | 14.852   | 27.3585  | DP                |
| 16 |         1 |  18.6655  |  4.32036 |  3.39    |  5.03596 | CoT               |
| 17 |         5 |  31.659   |  5.62664 |  4.291   |  6.95525 | CoT               |
| 18 |        20 | 202.064   | 14.2149  | 13.997   | 25.6009  | CoT               |
| 19 |        30 | 227.271   | 15.0755  | 14.302   | 26.3301  | CoT               |
| 20 |         1 |  18.6421  |  4.31765 |  3.34    |  4.95072 | CoT-RF            |
| 21 |         5 |  29.886   |  5.46681 |  3.981   |  6.45023 | CoT-RF            |
| 22 |        20 | 164.363   | 12.8204  | 12.502   | 22.8475  | CoT-RF            |
| 23 |        30 | 184.886   | 13.5973  | 12.532   | 23.1851  | CoT-RF            |
| 24 |         1 |  17.7014  |  4.2073  |  3.117   |  4.59376 | TSM+LLM           |
| 25 |         5 |  30.0996  |  5.48631 |  3.321   |  5.38159 | TSM+LLM           |
| 26 |        20 | 146.98    | 12.1235  | 11.687   | 21.2923  | TSM+LLM           |
| 27 |        30 | 186.86    | 13.6697  | 12.182   | 22.5921  | TSM+LLM           |

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

|    |   horizon | model          | baseline          |   mean_error_model |   mean_error_baseline |   improvement_pct |   t_statistic |    t_pvalue | t_significant   |   wilcoxon_statistic |   wilcoxon_pvalue | wilcoxon_significant   |     dm_statistic |   dm_pvalue | dm_significant   | model_better   |
|---:|----------:|:---------------|:------------------|-------------------:|----------------------:|------------------:|--------------:|------------:|:----------------|---------------------:|------------------:|:-----------------------|-----------------:|------------:|:-----------------|:---------------|
|  0 |         1 | seasonal_naive | naive_persistence |           10.571   |               4.28177 |    -146.884       |     6.07455   | 3.15549e-09 | True            |                15973 |       2.42149e-17 | True                   |      6.07455     | 1.24334e-09 | True             | False          |
|  1 |         5 | seasonal_naive | naive_persistence |           10.2545  |              10.2545  |       4.36825e-06 |     1.6742    | 0.0949572   | False           |                27959 |       0.454816    | False                  |      0.0952472   | 0.924118    | False            | True           |
|  2 |        20 | seasonal_naive | naive_persistence |           36.8281  |              36.8281  |       6.48903e-06 |    -0.918201  | 0.359127    | False           |                25159 |       0.0428928   | True                   |     -0.155897    | 0.876114    | False            | True           |
|  3 |        30 | seasonal_naive | naive_persistence |           49.5437  |              49.5437  |      -6.35439e-06 |    -0.0828849 | 0.933989    | False           |                24945 |       0.0208131   | True                   |     -0.0147042   | 0.988268    | False            | False          |
|  4 |         1 | tsm            | naive_persistence |            4.03213 |               4.28177 |       5.83041     |    -2.2006    | 0.0283974   | True            |                30985 |       0.34885     | False                  |     -2.2006      | 0.0277647   | True             | True           |
|  5 |         5 | tsm            | naive_persistence |           10.3769  |              10.2545  |      -1.19318     |     0.386971  | 0.699006    | False           |                29701 |       0.113814    | False                  |      0.227279    | 0.820207    | False            | False          |
|  6 |        20 | tsm            | naive_persistence |           37.0161  |              36.8281  |      -0.51052     |     0.196443  | 0.844374    | False           |                30654 |       0.270047    | False                  |      0.0599522   | 0.952194    | False            | False          |
|  7 |        30 | tsm            | naive_persistence |           50.6975  |              49.5437  |      -2.32896     |     0.800619  | 0.423879    | False           |                30895 |       0.326103    | False                  |      0.231302    | 0.81708     | False            | False          |
|  8 |         1 | DP             | naive_persistence |           17.2841  |              18.7791  |       7.96098     |    -1.75366   | 0.113389    | False           |                   12 |       0.130859    | False                  |     -1.75366     | 0.0794886   | False            | True           |
|  9 |         5 | DP             | naive_persistence |           28.4284  |              30.9492  |       8.14509     |    -0.395047  | 0.702007    | False           |                   20 |       0.492188    | False                  |     -1.02212     | 0.306725    | False            | True           |
| 10 |        20 | DP             | naive_persistence |          186.193   |             115.192   |     -61.637       |     2.02417   | 0.073625    | False           |                   11 |       0.105469    | False                  |      2.24525e+07 | 0           | True             | False          |
| 11 |        30 | DP             | naive_persistence |          245.545   |             136.459   |     -79.9402      |     2.19007   | 0.0562441   | False           |                    8 |       0.0488281   | True                   |      3.4496e+07  | 0           | True             | False          |
| 12 |         1 | CoT            | naive_persistence |           18.6655  |              18.7791  |       0.604936    |    -0.0972365 | 0.92467     | False           |                   21 |       0.556641    | False                  |     -0.0972365   | 0.922539    | False            | True           |
| 13 |         5 | CoT            | naive_persistence |           31.659   |              30.9492  |      -2.29345     |     0.0808785 | 0.937309    | False           |                   12 |       0.130859    | False                  | 224459           | 0           | True             | False          |
| 14 |        20 | CoT            | naive_persistence |          202.064   |             115.192   |     -75.4149      |     4.41589   | 0.00168118  | True            |                    2 |       0.00585938  | True                   |      2.74714e+07 | 0           | True             | False          |
| 15 |        30 | CoT            | naive_persistence |          227.271   |             136.459   |     -66.5487      |     4.26536   | 0.00209455  | True            |                    0 |       0.00195312  | True                   |      2.87172e+07 | 0           | True             | False          |
| 16 |         1 | CoT-RF         | naive_persistence |           18.6421  |              18.7791  |       0.729543    |    -0.153984  | 0.88102     | False           |                   21 |       0.556641    | False                  |     -0.153984    | 0.877622    | False            | True           |
| 17 |         5 | CoT-RF         | naive_persistence |           29.886   |              30.9492  |       3.43529     |    -0.161994  | 0.874889    | False           |                   12 |       0.130859    | False                  |     -0.984595    | 0.324823    | False            | True           |
| 18 |        20 | CoT-RF         | naive_persistence |          164.363   |             115.192   |     -42.6854      |     3.96339   | 0.00328736  | True            |                    4 |       0.0136719   | True                   |      1.5549e+07  | 0           | True             | False          |
| 19 |        30 | CoT-RF         | naive_persistence |          184.886   |             136.459   |     -35.4878      |     4.39094   | 0.00174317  | True            |                    1 |       0.00390625  | True                   |      1.53138e+07 | 0           | True             | False          |
| 20 |         1 | TSM+LLM        | naive_persistence |           17.7014  |              18.7791  |       5.7391      |    -0.790212  | 0.449723    | False           |                   21 |       0.556641    | False                  |     -0.790212    | 0.429404    | False            | True           |
| 21 |         5 | TSM+LLM        | naive_persistence |           30.0996  |              30.9492  |       2.74513     |    -0.207676  | 0.840105    | False           |                   23 |       0.695312    | False                  |     -0.526569    | 0.598493    | False            | True           |
| 22 |        20 | TSM+LLM        | naive_persistence |          146.98    |             115.192   |     -27.5955      |     7.62787   | 3.23187e-05 | True            |                    0 |       0.00195312  | True                   |     17.6681      | 0           | True             | False          |
| 23 |        30 | TSM+LLM        | naive_persistence |          186.86    |             136.459   |     -36.9347      |     4.87272   | 0.000880201 | True            |                    0 |       0.00195312  | True                   |      1.59381e+07 | 0           | True             | False          |


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
| 16 |         1 |  17.1169  |  4.13725 |  3.01695 |  4.43763 |          0.05 | DP      |
| 17 |         5 |  28.5643  |  5.34455 |  4.01358 |  6.50784 |          0.05 | DP      |
| 18 |        20 | 184.306   | 13.5759  | 13.3996  | 24.5052  |          0.05 | DP      |
| 19 |        30 | 249.199   | 15.786   | 14.9729  | 27.573   |          0.05 | DP      |
| 20 |         1 |  16.964   |  4.11874 |  2.99989 |  4.41196 |          0.1  | DP      |
| 21 |         5 |  28.7078  |  5.35796 |  4.00917 |  6.50037 |          0.1  | DP      |
| 22 |        20 | 182.437   | 13.5069  | 13.3322  | 24.3842  |          0.1  | DP      |
| 23 |        30 | 252.928   | 15.9037  | 15.0938  | 27.7875  |          0.1  | DP      |
| 24 |         1 |  16.7015  |  4.08675 |  2.96579 |  4.36064 |          0.2  | DP      |
| 25 |         5 |  29.0176  |  5.3868  |  4.00034 |  6.48544 |          0.2  | DP      |
| 26 |        20 | 178.758   | 13.37    | 13.1973  | 24.1422  |          0.2  | DP      |
| 27 |        30 | 260.616   | 16.1436  | 15.3355  | 28.2165  |          0.2  | DP      |
| 28 |         1 |  16.4965  |  4.06159 |  2.97365 |  4.37701 |          0.3  | DP      |
| 29 |         5 |  29.358   |  5.4183  |  3.99151 |  6.4705  |          0.3  | DP      |
| 30 |        20 | 175.155   | 13.2346  | 13.0625  | 23.9003  |          0.3  | DP      |
| 31 |        30 | 268.61    | 16.3893  | 15.5773  | 28.6455  |          0.3  | DP      |
| 32 |         1 |  18.4942  |  4.30049 |  3.35903 |  4.98709 |          0.05 | CoT     |
| 33 |         5 |  31.7593  |  5.63553 |  4.28421 |  6.94432 |          0.05 | CoT     |
| 34 |        20 | 200.699   | 14.1668  | 13.9459  | 25.5098  |          0.05 | CoT     |
| 35 |        30 | 230.086   | 15.1686  | 14.4136  | 26.5265  |          0.05 | CoT     |
| 36 |         1 |  18.3313  |  4.28151 |  3.32806 |  4.93821 |          0.1  | CoT     |
| 37 |         5 |  31.8645  |  5.64487 |  4.27742 |  6.93339 |          0.1  | CoT     |
| 38 |        20 | 199.347   | 14.119   | 13.8948  | 25.4187  |          0.1  | CoT     |
| 39 |        30 | 232.98    | 15.2637  | 14.5253  | 26.7229  |          0.1  | CoT     |
| 40 |         1 |  18.0307  |  4.24626 |  3.26613 |  4.84047 |          0.2  | CoT     |
| 41 |         5 |  32.0903  |  5.66483 |  4.26385 |  6.91152 |          0.2  | CoT     |
| 42 |        20 | 196.686   | 14.0245  | 13.7925  | 25.2365  |          0.2  | CoT     |
| 43 |        30 | 239.004   | 15.4598  | 14.7485  | 27.1157  |          0.2  | CoT     |
| 44 |         1 |  17.7636  |  4.21469 |  3.20419 |  4.74273 |          0.3  | CoT     |
| 45 |         5 |  32.3363  |  5.68651 |  4.25873 |  6.9033  |          0.3  | CoT     |
| 46 |        20 | 194.083   | 13.9314  | 13.6903  | 25.0544  |          0.3  | CoT     |
| 47 |        30 | 245.344   | 15.6635  | 14.9718  | 27.5085  |          0.3  | CoT     |
| 48 |         1 |  18.534   |  4.30511 |  3.32009 |  4.91931 |          0.05 | CoT-RF  |
| 49 |         5 |  29.9537  |  5.473   |  3.97891 |  6.44681 |          0.05 | CoT-RF  |
| 50 |        20 | 163.634   | 12.792   | 12.4722  | 22.7943  |          0.05 | CoT-RF  |
| 51 |        30 | 186.15    | 13.6437  | 12.5953  | 23.2971  |          0.05 | CoT-RF  |
| 52 |         1 |  18.4293  |  4.29293 |  3.30019 |  4.8879  |          0.1  | CoT-RF  |
| 53 |         5 |  30.0236  |  5.47937 |  3.97682 |  6.44339 |          0.1  | CoT-RF  |
| 54 |        20 | 162.911   | 12.7636  | 12.4424  | 22.7411  |          0.1  | CoT-RF  |
| 55 |        30 | 187.436   | 13.6907  | 12.6585  | 23.409   |          0.1  | CoT-RF  |
| 56 |         1 |  18.23    |  4.26966 |  3.26038 |  4.82507 |          0.2  | CoT-RF  |
| 57 |         5 |  30.1696  |  5.49269 |  3.97265 |  6.43654 |          0.2  | CoT-RF  |
| 58 |        20 | 161.477   | 12.7073  | 12.3827  | 22.6348  |          0.2  | CoT-RF  |
| 59 |        30 | 190.073   | 13.7867  | 12.7851  | 23.6328  |          0.2  | CoT-RF  |
| 60 |         1 |  18.0443  |  4.24786 |  3.22057 |  4.76224 |          0.3  | CoT-RF  |
| 61 |         5 |  30.3242  |  5.50674 |  3.96847 |  6.4297  |          0.3  | CoT-RF  |
| 62 |        20 | 160.061   | 12.6515  | 12.3231  | 22.5284  |          0.3  | CoT-RF  |
| 63 |        30 | 192.797   | 13.8851  | 12.9116  | 23.8566  |          0.3  | CoT-RF  |
| 64 |         1 |  17.5669  |  4.19129 |  3.08768 |  4.54781 |          0.05 | TSM+LLM |
| 65 |         5 |  30.2573  |  5.50067 |  3.32382 |  5.38642 |          0.05 | TSM+LLM |
| 66 |        20 | 145.981   | 12.0823  | 11.6461  | 21.2196  |          0.05 | TSM+LLM |
| 67 |        30 | 188.071   | 13.7139  | 12.2464  | 22.7047  |          0.05 | TSM+LLM |
| 68 |         1 |  17.4355  |  4.17559 |  3.06541 |  4.51307 |          0.1  | TSM+LLM |
| 69 |         5 |  30.4171  |  5.51517 |  3.32663 |  5.39126 |          0.1  | TSM+LLM |
| 70 |        20 | 144.99    | 12.0412  | 11.6053  | 21.147   |          0.1  | TSM+LLM |
| 71 |        30 | 189.309   | 13.759   | 12.3107  | 22.8172  |          0.1  | TSM+LLM |
| 72 |         1 |  17.1817  |  4.14509 |  3.03581 |  4.46732 |          0.2  | TSM+LLM |
| 73 |         5 |  30.743   |  5.54464 |  3.33227 |  5.40093 |          0.2  | TSM+LLM |
| 74 |        20 | 143.032   | 11.9596  | 11.5235  | 21.0017  |          0.2  | TSM+LLM |
| 75 |        30 | 191.873   | 13.8518  | 12.4395  | 23.0422  |          0.2  | TSM+LLM |
| 76 |         1 |  16.94    |  4.11582 |  3.00622 |  4.42158 |          0.3  | TSM+LLM |
| 77 |         5 |  31.0772  |  5.57469 |  3.34139 |  5.41614 |          0.3  | TSM+LLM |
| 78 |        20 | 141.108   | 11.8789  | 11.4418  | 20.8564  |          0.3  | TSM+LLM |
| 79 |        30 | 194.55    | 13.9481  | 12.5682  | 23.2672  |          0.3  | TSM+LLM |

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


