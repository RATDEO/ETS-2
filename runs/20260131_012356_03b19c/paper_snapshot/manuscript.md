# EU ETS Futures Forecasting with TSM + LLM Refinement

**Authors**: Auto-generated

**Generated**: 2026-01-31 01:24:12

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
| 20 |         1 |  1.67731   | 1.29511  | 1.0775   | 1.5526   | naive_persistence_llm_subset |
| 21 |         5 |  7.68218   | 2.77167  | 2.27465  | 3.24384  | naive_persistence_llm_subset |
| 22 |        20 | 27.603     | 5.25386  | 4.33965  | 6.10924  | naive_persistence_llm_subset |
| 23 |        30 | 34.7508    | 5.89498  | 4.80306  | 6.80999  | naive_persistence_llm_subset |
| 24 |         1 |  9.74664   | 3.12196  | 2.54875  | 3.68218  | seasonal_naive_llm_subset    |
| 25 |         5 |  7.68218   | 2.77167  | 2.27465  | 3.24384  | seasonal_naive_llm_subset    |
| 26 |        20 | 27.603     | 5.25386  | 4.33965  | 6.10924  | seasonal_naive_llm_subset    |
| 27 |        30 | 34.7508    | 5.89498  | 4.80306  | 6.80999  | seasonal_naive_llm_subset    |
| 28 |         1 |  1.706     | 1.30614  | 1.07214  | 1.54506  | linear_ridge_llm_subset      |
| 29 |         5 |  7.99287   | 2.82717  | 2.31863  | 3.30694  | linear_ridge_llm_subset      |
| 30 |        20 | 31.8061    | 5.63969  | 4.71317  | 6.67496  | linear_ridge_llm_subset      |
| 31 |        30 | 39.8822    | 6.31524  | 5.08051  | 7.30859  | linear_ridge_llm_subset      |
| 32 |         1 |  5.31903   | 2.3063   | 1.96791  | 2.81276  | linear_lasso_llm_subset      |
| 33 |         5 | 12.2571    | 3.50101  | 2.93188  | 4.13583  | linear_lasso_llm_subset      |
| 34 |        20 | 33.6776    | 5.80324  | 4.97083  | 6.92883  | linear_lasso_llm_subset      |
| 35 |        30 | 39.2735    | 6.26686  | 5.21034  | 7.3889   | linear_lasso_llm_subset      |
| 36 |         1 |  0.144622  | 0.380292 | 0.302281 | 0.430786 | tsm_llm_subset               |
| 37 |         5 |  2.81045   | 1.67644  | 1.34055  | 1.89531  | tsm_llm_subset               |
| 38 |        20 | 22.0102    | 4.6915   | 3.88331  | 5.45972  | tsm_llm_subset               |
| 39 |        30 | 42.9259    | 6.55179  | 5.3459   | 7.64394  | tsm_llm_subset               |

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
|  0 |         1 | seasonal_naive_llm_subset | naive_persistence_llm_subset |           9.74664  |               1.67731 |    -481.088       |      8.08798  | 2.35423e-13 | True            |                 1259 |       2.8016e-15  | True                   |       8.08798  | 6.66134e-16 | True             | False          |
|  1 |         5 | seasonal_naive_llm_subset | naive_persistence_llm_subset |           7.68218  |               7.68218 |       3.94151e-06 |     -0.134806 | 0.892955    | False           |                 4328 |       0.320401    | False                  |      -0.003824 | 0.996949    | False            | True           |
|  2 |        20 | seasonal_naive_llm_subset | naive_persistence_llm_subset |          27.603    |              27.603   |       6.83956e-06 |     -2.07309  | 0.0399586   | True            |                 3710 |       0.00538487  | True                   |      -0.251557 | 0.801383    | False            | True           |
|  3 |        30 | seasonal_naive_llm_subset | naive_persistence_llm_subset |          34.7508   |              34.7508  |      -5.13734e-06 |     -0.490223 | 0.624728    | False           |                 4258 |       0.253253    | False                  |      -0.056952 | 0.954583    | False            | False          |
|  4 |         1 | linear_ridge_llm_subset   | naive_persistence_llm_subset |           1.706    |               1.67731 |      -1.71039     |      0.625369 | 0.532725    | False           |                 5114 |       0.832578    | False                  |       0.625369 | 0.531729    | False            | False          |
|  5 |         5 | linear_ridge_llm_subset   | naive_persistence_llm_subset |           7.99287  |               7.68218 |      -4.04423     |      0.923451 | 0.357328    | False           |                 4783 |       0.383475    | False                  |       0.427485 | 0.669026    | False            | False          |
|  6 |        20 | linear_ridge_llm_subset   | naive_persistence_llm_subset |          31.8061   |              27.603   |     -15.2269      |      2.03699  | 0.0434958   | True            |                 4257 |       0.0547929   | False                  |       0.553491 | 0.579928    | False            | False          |
|  7 |        30 | linear_ridge_llm_subset   | naive_persistence_llm_subset |          39.8822   |              34.7508  |     -14.7665      |      1.76399  | 0.0798687   | False           |                 5090 |       0.795435    | False                  |       0.666845 | 0.504871    | False            | False          |
|  8 |         1 | linear_lasso_llm_subset   | naive_persistence_llm_subset |           5.31903  |               1.67731 |    -217.117       |      9.25671  | 2.83537e-16 | True            |                 1493 |       1.06345e-13 | True                   |       9.25671  | 0           | True             | False          |
|  9 |         5 | linear_lasso_llm_subset   | naive_persistence_llm_subset |          12.2571   |               7.68218 |     -59.552       |      6.10148  | 9.34077e-09 | True            |                 2475 |       4.39082e-08 | True                   |       3.00038  | 0.00269639  | True             | False          |
| 10 |        20 | linear_lasso_llm_subset   | naive_persistence_llm_subset |          33.6776   |              27.603   |     -22.007       |      3.63942  | 0.000380934 | True            |                 3186 |       4.98316e-05 | True                   |       1.32817  | 0.184121    | False            | False          |
| 11 |        30 | linear_lasso_llm_subset   | naive_persistence_llm_subset |          39.2735   |              34.7508  |     -13.0148      |      2.68666  | 0.00807209  | True            |                 3974 |       0.0129587   | True                   |       0.970509 | 0.331793    | False            | False          |
| 12 |         1 | tsm_llm_subset            | naive_persistence_llm_subset |           0.144622 |               1.67731 |      91.3777      |     -9.39326  | 1.27324e-16 | True            |                  432 |       1.31296e-21 | True                   |      -9.39326  | 0           | True             | True           |
| 13 |         5 | tsm_llm_subset            | naive_persistence_llm_subset |           2.81045  |               7.68218 |      63.416       |     -6.66079  | 5.44856e-10 | True            |                 2259 |       3.52357e-09 | True                   |      -4.89961  | 9.6025e-07  | True             | True           |
| 14 |        20 | tsm_llm_subset            | naive_persistence_llm_subset |          22.0102   |              27.603   |      20.2618      |     -2.89808  | 0.0043464   | True            |                 4185 |       0.0390078   | True                   |      -1.70145  | 0.0888587   | False            | True           |
| 15 |        30 | tsm_llm_subset            | naive_persistence_llm_subset |          42.9259   |              34.7508  |     -23.5251      |      2.41345  | 0.0170681   | True            |                 4072 |       0.0220526   | True                   |       1.59987  | 0.109627    | False            | False          |


## 5. Robustness Analysis

### 5.1 Noise Injection Test

To assess robustness, we inject noise into forecast paths at levels of 5%, 10%, 20%, 
and 30% of the forecast standard deviation.

|    |   horizon |       mse |     rmse |      mae |     mape |   noise_level | model   |
|---:|----------:|----------:|---------:|---------:|---------:|--------------:|:--------|
|  0 |         1 |  0.106027 | 0.325618 | 0.252294 | 0.354599 |          0.05 | tsm     |
|  1 |         5 |  1.88481  | 1.37288  | 1.05032  | 1.47396  |          0.05 | tsm     |
|  2 |        20 | 19.2307   | 4.38528  | 3.38717  | 4.7299   |          0.05 | tsm     |
|  3 |        30 | 40.718    | 6.38107  | 5.01641  | 6.98411  |          0.05 | tsm     |
|  4 |         1 |  0.128185 | 0.358029 | 0.27776  | 0.390995 |          0.1  | tsm     |
|  5 |         5 |  1.89516  | 1.37665  | 1.05649  | 1.48297  |          0.1  | tsm     |
|  6 |        20 | 19.2416   | 4.38653  | 3.39037  | 4.73418  |          0.1  | tsm     |
|  7 |        30 | 40.683    | 6.37832  | 5.01188  | 6.97805  |          0.1  | tsm     |
|  8 |         1 |  0.213268 | 0.461809 | 0.354609 | 0.500396 |          0.2  | tsm     |
|  9 |         5 |  1.96284  | 1.40101  | 1.08268  | 1.52055  |          0.2  | tsm     |
| 10 |        20 | 19.3007   | 4.39325  | 3.39904  | 4.74613  |          0.2  | tsm     |
| 11 |        30 | 40.6488   | 6.37564  | 5.0059   | 6.97032  |          0.2  | tsm     |
| 12 |         1 |  0.352708 | 0.593892 | 0.45259  | 0.639237 |          0.3  | tsm     |
| 13 |         5 |  2.09313  | 1.44676  | 1.12078  | 1.57466  |          0.3  | tsm     |
| 14 |        20 | 19.4092   | 4.40558  | 3.41193  | 4.764    |          0.3  | tsm     |
| 15 |        30 | 40.6624   | 6.37671  | 5.00313  | 6.96678  |          0.3  | tsm     |

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


