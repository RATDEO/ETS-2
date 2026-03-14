# EU ETS Futures Forecasting with TSM + LLM Refinement

**Authors**: Auto-generated

**Generated**: 2026-03-06 18:44:20

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
- **Number of Features**: 46

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

|    |   horizon |      mse |    rmse |      mae |     mape | model             |
|---:|----------:|---------:|--------:|---------:|---------:|:------------------|
|  0 |         1 |  1.06139 | 1.03024 | 0.784861 |  1.39543 | naive_persistence |
|  1 |         5 |  5.2119  | 2.28296 | 1.67     |  2.90235 | naive_persistence |
|  2 |        20 | 44.2759  | 6.65401 | 4.40799  |  7.94956 | naive_persistence |
|  3 |        30 | 79.9881  | 8.94361 | 6.37986  | 12.0727  | naive_persistence |
|  4 |         1 |  5.16853 | 2.27344 | 1.65132  |  2.90012 | seasonal_naive    |
|  5 |         5 |  5.2119  | 2.28296 | 1.67     |  2.90235 | seasonal_naive    |
|  6 |        20 | 44.2759  | 6.65401 | 4.40799  |  7.94956 | seasonal_naive    |
|  7 |        30 | 79.9881  | 8.94361 | 6.37986  | 12.0727  | seasonal_naive    |
|  8 |         1 |  1.06154 | 1.03031 | 0.781283 |  1.38304 | linear_ridge      |
|  9 |         5 |  4.59363 | 2.14328 | 1.56632  |  2.70398 | linear_ridge      |
| 10 |        20 | 30.1544  | 5.4913  | 3.32793  |  6.07247 | linear_ridge      |
| 11 |        30 | 56.9094  | 7.54383 | 4.78768  |  9.18917 | linear_ridge      |
| 12 |         1 |  1.1541  | 1.07429 | 0.830618 |  1.48473 | linear_lasso      |
| 13 |         5 |  4.73942 | 2.17702 | 1.61188  |  2.79853 | linear_lasso      |
| 14 |        20 | 32.1131  | 5.66684 | 3.43547  |  6.16613 | linear_lasso      |
| 15 |        30 | 62.0974  | 7.88019 | 5.23504  |  9.89649 | linear_lasso      |
| 16 |         1 |  1.20574 | 1.09806 | 0.855881 |  1.51781 | tsm               |
| 17 |         5 |  5.64911 | 2.37679 | 1.80329  |  3.12164 | tsm               |
| 18 |        20 | 40.2462  | 6.34399 | 4.28359  |  7.6831  | tsm               |
| 19 |        30 | 72.369   | 8.507   | 6.23287  | 11.696   | tsm               |

### 4.2 Trend Classification Accuracy

Table 2 shows the accuracy of predicting price direction (up/flat/down).

|    |   horizon |   accuracy |   accuracy_up |   accuracy_down |   accuracy_flat |   n_up |   n_down |   n_flat | model             |
|---:|----------:|-----------:|--------------:|----------------:|----------------:|-------:|---------:|---------:|:------------------|
|  0 |         1 |  0.395833  |     0         |       0         |       1         |     52 |       35 |       57 | naive_persistence |
|  1 |         5 |  0.236111  |     0         |       0         |       1         |     70 |       40 |       34 | naive_persistence |
|  2 |        20 |  0.118056  |     0         |       0         |       1         |     90 |       37 |       17 | naive_persistence |
|  3 |        30 |  0.0763889 |     0         |       0         |       1         |    100 |       33 |       11 | naive_persistence |
|  4 |         1 |  0.375     |     0.365385  |       0.628571  |       0.22807   |     52 |       35 |       57 | seasonal_naive    |
|  5 |         5 |  0.236111  |     0         |       0         |       1         |     70 |       40 |       34 | seasonal_naive    |
|  6 |        20 |  0.118056  |     0         |       0         |       1         |     90 |       37 |       17 | seasonal_naive    |
|  7 |        30 |  0.0763889 |     0         |       0         |       1         |    100 |       33 |       11 | seasonal_naive    |
|  8 |         1 |  0.430556  |     0.0576923 |       0.0857143 |       0.982456  |     52 |       35 |       57 | linear_ridge      |
|  9 |         5 |  0.451389  |     0.514286  |       0.125     |       0.705882  |     70 |       40 |       34 | linear_ridge      |
| 10 |        20 |  0.708333  |     0.955556  |       0.351351  |       0.176471  |     90 |       37 |       17 | linear_ridge      |
| 11 |        30 |  0.833333  |     0.99      |       0.606061  |       0.0909091 |    100 |       33 |       11 | linear_ridge      |
| 12 |         1 |  0.423611  |     0.288462  |       0.114286  |       0.736842  |     52 |       35 |       57 | linear_lasso      |
| 13 |         5 |  0.388889  |     0.414286  |       0.15      |       0.617647  |     70 |       40 |       34 | linear_lasso      |
| 14 |        20 |  0.708333  |     0.922222  |       0.459459  |       0.117647  |     90 |       37 |       17 | linear_lasso      |
| 15 |        30 |  0.736111  |     0.75      |       0.757576  |       0.545455  |    100 |       33 |       11 | linear_lasso      |
| 16 |         1 |  0.381944  |     0.0192308 |       0.114286  |       0.877193  |     52 |       35 |       57 | tsm               |
| 17 |         5 |  0.256944  |     0.0714286 |       0.425     |       0.441176  |     70 |       40 |       34 | tsm               |
| 18 |        20 |  0.388889  |     0.288889  |       0.702703  |       0.235294  |     90 |       37 |       17 | tsm               |
| 19 |        30 |  0.361111  |     0.19      |       0.909091  |       0.272727  |    100 |       33 |       11 | tsm               |

### 4.3 Statistical Significance

Table 3 reports the results of significance tests comparing methods.

|    |   horizon | model          | baseline          |   mean_error_model |   mean_error_baseline |   improvement_pct |   t_statistic |    t_pvalue | t_significant   |   wilcoxon_statistic |   wilcoxon_pvalue | wilcoxon_significant   |   dm_statistic |   dm_pvalue | dm_significant   | model_better   |
|---:|----------:|:---------------|:------------------|-------------------:|----------------------:|------------------:|--------------:|------------:|:----------------|---------------------:|------------------:|:-----------------------|---------------:|------------:|:-----------------|:---------------|
|  0 |         1 | seasonal_naive | naive_persistence |            5.16853 |               1.06139 |    -386.957       |    5.07044    | 1.21001e-06 | True            |                 1834 |       1.45104e-11 | True                   |     5.07044    | 3.96893e-07 | True             | False          |
|  1 |         5 | seasonal_naive | naive_persistence |            5.2119  |               5.2119  |       5.49269e-07 |    1.19107    | 0.235598    | False           |                 4258 |       0.556904    | False                  |     0.0263232  | 0.979       | False            | True           |
|  2 |        20 | seasonal_naive | naive_persistence |           44.2759  |              44.2759  |       2.22491e-06 |    0.67977    | 0.497749    | False           |                 4233 |       0.183904    | False                  |     0.216494   | 0.828603    | False            | True           |
|  3 |        30 | seasonal_naive | naive_persistence |           79.9881  |              79.9881  |       3.104e-06   |   -0.578827   | 0.563616    | False           |                 3978 |       0.082287    | False                  |    -0.245123   | 0.806361    | False            | True           |
|  4 |         1 | linear_ridge   | naive_persistence |            1.06154 |               1.06139 |      -0.0135551   |    0.00242328 | 0.99807     | False           |                 5155 |       0.896859    | False                  |     0.00242328 | 0.998067    | False            | False          |
|  5 |         5 | linear_ridge   | naive_persistence |            4.59363 |               5.2119  |      11.8626      |   -2.00294    | 0.0470741   | True            |                 3955 |       0.0116426   | True                   |    -1.15055    | 0.249918    | False            | True           |
|  6 |        20 | linear_ridge   | naive_persistence |           30.1544  |              44.2759  |      31.8943      |   -5.67076    | 7.5947e-08  | True            |                 2425 |       2.4884e-08  | True                   |    -2.3465     | 0.0189507   | True             | True           |
|  7 |        30 | linear_ridge   | naive_persistence |           56.9094  |              79.9881  |      28.8527      |   -7.86447    | 8.23678e-13 | True            |                  726 |       3.17675e-19 | True                   |    -1.91128    | 0.0559692   | False            | True           |
|  8 |         1 | linear_lasso   | naive_persistence |            1.1541  |               1.06139 |      -8.73423     |    1.16037    | 0.24783     | False           |                 4553 |       0.183452    | False                  |     1.16037    | 0.245896    | False            | False          |
|  9 |         5 | linear_lasso   | naive_persistence |            4.73942 |               5.2119  |       9.06537     |   -1.85969    | 0.0649833   | False           |                 4418 |       0.109725    | False                  |    -1.14841    | 0.250799    | False            | True           |
| 10 |        20 | linear_lasso   | naive_persistence |           32.1131  |              44.2759  |      27.4704      |   -5.48329    | 1.83923e-07 | True            |                 1499 |       1.16409e-13 | True                   |    -2.64988    | 0.00805199  | True             | True           |
| 11 |        30 | linear_lasso   | naive_persistence |           62.0974  |              79.9881  |      22.3667      |   -6.53209    | 1.0595e-09  | True            |                  443 |       1.62218e-21 | True                   |    -1.64973    | 0.0989982   | False            | True           |
| 12 |         1 | tsm            | naive_persistence |            1.20574 |               1.06139 |     -13.5994      |    2.1918     | 0.0300118   | True            |                 3956 |       0.0117088   | True                   |     2.1918     | 0.0283939   | True             | False          |
| 13 |         5 | tsm            | naive_persistence |            5.64911 |               5.2119  |      -8.38883     |    1.54257    | 0.125144    | False           |                 4440 |       0.119813    | False                  |     0.774825   | 0.438443    | False            | False          |
| 14 |        20 | tsm            | naive_persistence |           40.2462  |              44.2759  |       9.10133     |   -2.80664    | 0.0057048   | True            |                 4600 |       0.216284    | False                  |    -1.24077    | 0.214692    | False            | True           |
| 15 |        30 | tsm            | naive_persistence |           72.369   |              79.9881  |       9.52529     |   -3.08456    | 0.00244758  | True            |                 4970 |       0.618078    | False                  |    -0.821835   | 0.411171    | False            | True           |


## 5. Robustness Analysis

### 5.1 Noise Injection Test

To assess robustness, we inject noise into forecast paths at levels of 5%, 10%, 20%, 
and 30% of the forecast standard deviation.

|    |   horizon |      mse |    rmse |      mae |     mape |   noise_level | model   |
|---:|----------:|---------:|--------:|---------:|---------:|--------------:|:--------|
|  0 |         1 |  1.20222 | 1.09646 | 0.854421 |  1.51606 |          0.05 | tsm     |
|  1 |         5 |  5.65917 | 2.3789  | 1.8026   |  3.12079 |          0.05 | tsm     |
|  2 |        20 | 40.2456  | 6.34394 | 4.28232  |  7.6806  |          0.05 | tsm     |
|  3 |        30 | 72.3504  | 8.5059  | 6.22953  | 11.69    |          0.05 | tsm     |
|  4 |         1 |  1.19999 | 1.09544 | 0.853554 |  1.51538 |          0.1  | tsm     |
|  5 |         5 |  5.67089 | 2.38136 | 1.80194  |  3.11996 |          0.1  | tsm     |
|  6 |        20 | 40.2464  | 6.34401 | 4.28108  |  7.67814 |          0.1  | tsm     |
|  7 |        30 | 72.3328  | 8.50487 | 6.2262   | 11.684   |          0.1  | tsm     |
|  8 |         1 |  1.19937 | 1.09516 | 0.85291  |  1.51572 |          0.2  | tsm     |
|  9 |         5 |  5.69935 | 2.38733 | 1.80217  |  3.12112 |          0.2  | tsm     |
| 10 |        20 | 40.2522  | 6.34446 | 4.27922  |  7.67436 |          0.2  | tsm     |
| 11 |        30 | 72.301   | 8.503   | 6.21953  | 11.6721  |          0.2  | tsm     |
| 12 |         1 |  1.20386 | 1.0972  | 0.854502 |  1.51948 |          0.3  | tsm     |
| 13 |         5 |  5.73449 | 2.39468 | 1.80254  |  3.12249 |          0.3  | tsm     |
| 14 |        20 | 40.2634  | 6.34534 | 4.27937  |  7.67435 |          0.3  | tsm     |
| 15 |        30 | 72.2734  | 8.50138 | 6.21286  | 11.6602  |          0.3  | tsm     |

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
- **LLM refinements**: No LLM results are available for this run, so LLM comparisons remain pending.

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
LLM refinement results are not included in this run, so the hybrid TSM+LLM comparison remains pending.
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


