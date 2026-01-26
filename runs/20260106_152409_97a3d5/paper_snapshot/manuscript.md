# EU ETS Futures Forecasting with TSM + LLM Refinement

**Authors**: Auto-generated

**Generated**: 2026-01-06 15:24:48

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

|    |   horizon |        mse |     rmse |      mae |     mape | model             |
|---:|----------:|-----------:|---------:|---------:|---------:|:------------------|
|  0 |         1 |  1.41603   | 1.18997  | 0.963454 | 1.36058  | naive_persistence |
|  1 |         5 |  6.54358   | 2.55804  | 2.04373  | 2.89746  | naive_persistence |
|  2 |        20 | 25.939     | 5.09303  | 3.97253  | 5.52109  | naive_persistence |
|  3 |        30 | 37.7942    | 6.1477   | 4.98591  | 6.84486  | naive_persistence |
|  4 |         1 |  6.64202   | 2.57721  | 2.0671   | 2.93212  | seasonal_naive    |
|  5 |         5 |  6.54358   | 2.55804  | 2.04373  | 2.89746  | seasonal_naive    |
|  6 |        20 | 25.939     | 5.09303  | 3.97253  | 5.52109  | seasonal_naive    |
|  7 |        30 | 37.7942    | 6.1477   | 4.98591  | 6.84486  | seasonal_naive    |
|  8 |         1 |  1.43079   | 1.19615  | 0.969301 | 1.36841  | linear_ridge      |
|  9 |         5 |  6.77679   | 2.60323  | 2.07369  | 2.94511  | linear_ridge      |
| 10 |        20 | 26.9499    | 5.19132  | 3.96345  | 5.61785  | linear_ridge      |
| 11 |        30 | 35.8673    | 5.98893  | 4.75943  | 6.63854  | linear_ridge      |
| 12 |         1 |  5.24942   | 2.29116  | 1.99612  | 2.77277  | linear_lasso      |
| 13 |         5 | 10.0643    | 3.17243  | 2.63507  | 3.65352  | linear_lasso      |
| 14 |        20 | 28.2284    | 5.31304  | 4.42377  | 6.08084  | linear_lasso      |
| 15 |        30 | 37.7403    | 6.14331  | 5.10021  | 6.96977  | linear_lasso      |
| 16 |         1 |  0.0974587 | 0.312184 | 0.243567 | 0.341986 | tsm               |
| 17 |         5 |  1.89011   | 1.37481  | 1.04876  | 1.47156  | tsm               |
| 18 |        20 | 19.2321    | 4.38545  | 3.38502  | 4.72689  | tsm               |
| 19 |        30 | 40.765     | 6.38475  | 5.02288  | 6.99315  | tsm               |

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

|    |   horizon | model          | baseline          |   mean_error_model |   mean_error_baseline |   improvement_pct |   t_statistic |    t_pvalue | t_significant   |   wilcoxon_statistic |   wilcoxon_pvalue | wilcoxon_significant   |   dm_statistic |   dm_pvalue | dm_significant   | model_better   |
|---:|----------:|:---------------|:------------------|-------------------:|----------------------:|------------------:|--------------:|------------:|:----------------|---------------------:|------------------:|:-----------------------|---------------:|------------:|:-----------------|:---------------|
|  0 |         1 | seasonal_naive | naive_persistence |          6.64202   |               1.41603 |    -369.059       |     11.0044   | 1.85805e-24 | True            |                 9862 |       3.80151e-30 | True                   |     11.0044    | 0           | True             | False          |
|  1 |         5 | seasonal_naive | naive_persistence |          6.54358   |               6.54358 |       2.91701e-06 |     -0.438472 | 0.661309    | False           |                23703 |       0.00276108  | True                   |     -0.0115703 | 0.990768    | False            | True           |
|  2 |        20 | seasonal_naive | naive_persistence |         25.939     |              25.939   |       6.53918e-06 |     -0.838913 | 0.402078    | False           |                24683 |       0.00190834  | True                   |     -0.0932438 | 0.92571     | False            | True           |
|  3 |        30 | seasonal_naive | naive_persistence |         37.7942    |              37.7942  |      -7.39789e-06 |     -0.183111 | 0.854815    | False           |                27510 |       0.17839     | False                  |     -0.0293692 | 0.97657     | False            | False          |
|  4 |         1 | linear_ridge   | naive_persistence |          1.43079   |               1.41603 |      -1.04207     |      0.677097 | 0.498782    | False           |                30687 |       0.409472    | False                  |      0.677097  | 0.498344    | False            | False          |
|  5 |         5 | linear_ridge   | naive_persistence |          6.77679   |               6.54358 |      -3.56385     |      1.37546  | 0.16985     | False           |                30354 |       0.320195    | False                  |      0.64678   | 0.517775    | False            | False          |
|  6 |        20 | linear_ridge   | naive_persistence |         26.9499    |              25.939   |      -3.89711     |      0.832612 | 0.405619    | False           |                31715 |       0.762359    | False                  |      0.221934  | 0.824365    | False            | False          |
|  7 |        30 | linear_ridge   | naive_persistence |         35.8673    |              37.7942  |       5.09825     |     -1.35643  | 0.175816    | False           |                22995 |       2.2017e-06  | True                   |     -0.373587  | 0.708711    | False            | True           |
|  8 |         1 | linear_lasso   | naive_persistence |          5.24942   |               1.41603 |    -270.714       |     15.963    | 1.06245e-43 | True            |                 7605 |       3.71706e-36 | True                   |     15.963     | 0           | True             | False          |
|  9 |         5 | linear_lasso   | naive_persistence |         10.0643    |               6.54358 |     -53.8039      |      7.46693  | 6.31679e-13 | True            |                17986 |       3.34832e-13 | True                   |      3.23881   | 0.0012003   | True             | False          |
| 10 |        20 | linear_lasso   | naive_persistence |         28.2284    |              25.939   |      -8.82599     |      2.37212  | 0.0182145   | True            |                22201 |       2.78434e-07 | True                   |      0.941802  | 0.346294    | False            | False          |
| 11 |        30 | linear_lasso   | naive_persistence |         37.7403    |              37.7942  |       0.142518    |     -0.052122 | 0.958461    | False           |                28208 |       0.0370988   | True                   |     -0.0335436 | 0.973241    | False            | True           |
| 12 |         1 | tsm            | naive_persistence |          0.0974587 |               1.41603 |      93.1175      |    -13.342    | 3.05489e-33 | True            |                 2347 |       2.32382e-52 | True                   |    -13.342     | 0           | True             | True           |
| 13 |         5 | tsm            | naive_persistence |          1.89011   |               6.54358 |      71.1151      |    -10.4029   | 2.54106e-22 | True            |                11380 |       2.00862e-26 | True                   |     -5.93302   | 2.97419e-09 | True             | True           |
| 14 |        20 | tsm            | naive_persistence |         19.2321    |              25.939   |      25.8562      |     -5.43811  | 9.99217e-08 | True            |                21206 |       1.6696e-08  | True                   |     -3.68315   | 0.000230373 | True             | True           |
| 15 |        30 | tsm            | naive_persistence |         40.765     |              37.7942  |      -7.86049     |      1.41536  | 0.157832    | False           |                31043 |       0.519639    | False                  |      0.760804  | 0.446774    | False            | False          |


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
Price accuracy is best for tsm, while directional accuracy is highest for tsm.
LLM refinement results are not included in this run, so the hybrid TSM+LLM comparison remains pending.
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


