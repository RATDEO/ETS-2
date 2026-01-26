# EU ETS Futures Forecasting with TSM + LLM Refinement

**Authors**: Auto-generated

**Generated**: 2026-01-06 13:16:22

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

|    |   horizon |      mse |    rmse |      mae |    mape | model             |
|---:|----------:|---------:|--------:|---------:|--------:|:------------------|
|  0 |         1 |  1.41603 | 1.18997 | 0.963454 | 1.36058 | naive_persistence |
|  1 |         5 |  6.54358 | 2.55804 | 2.04373  | 2.89746 | naive_persistence |
|  2 |        20 | 25.939   | 5.09303 | 3.97253  | 5.52109 | naive_persistence |
|  3 |        30 | 37.7942  | 6.1477  | 4.98591  | 6.84486 | naive_persistence |
|  4 |         1 |  6.64202 | 2.57721 | 2.0671   | 2.93212 | seasonal_naive    |
|  5 |         5 |  6.54358 | 2.55804 | 2.04373  | 2.89746 | seasonal_naive    |
|  6 |        20 | 25.939   | 5.09303 | 3.97253  | 5.52109 | seasonal_naive    |
|  7 |        30 | 37.7942  | 6.1477  | 4.98591  | 6.84486 | seasonal_naive    |
|  8 |         1 |  1.45037 | 1.20431 | 0.967349 | 1.36607 | tsm               |
|  9 |         5 |  6.69273 | 2.58703 | 2.08965  | 2.96723 | tsm               |
| 10 |        20 | 25.6175  | 5.06137 | 3.91153  | 5.48196 | tsm               |
| 11 |        30 | 36.2558  | 6.02128 | 4.76264  | 6.62402 | tsm               |

### 4.2 Trend Classification Accuracy

Table 2 shows the accuracy of predicting price direction (up/flat/down).

|    |   horizon |   accuracy |   accuracy_up |   accuracy_down |   accuracy_flat |   n_up |   n_down |   n_flat | model             |
|---:|----------:|-----------:|--------------:|----------------:|----------------:|-------:|---------:|---------:|:------------------|
|  0 |         1 |  0.295265  |     0         |       0         |       1         |    124 |      129 |      106 | naive_persistence |
|  1 |         5 |  0.142061  |     0         |       0         |       1         |    162 |      146 |       51 | naive_persistence |
|  2 |        20 |  0.091922  |     0         |       0         |       1         |    212 |      114 |       33 | naive_persistence |
|  3 |        30 |  0.0584958 |     0         |       0         |       1         |    224 |      114 |       21 | naive_persistence |
|  4 |         1 |  0.345404  |     0.427419  |       0.44186   |       0.132075  |    124 |      129 |      106 | seasonal_naive    |
|  5 |         5 |  0.142061  |     0         |       0         |       1         |    162 |      146 |       51 | seasonal_naive    |
|  6 |        20 |  0.091922  |     0         |       0         |       1         |    212 |      114 |       33 | seasonal_naive    |
|  7 |        30 |  0.0584958 |     0         |       0         |       1         |    224 |      114 |       21 | seasonal_naive    |
|  8 |         1 |  0.300836  |     0.0241935 |       0.0155039 |       0.971698  |    124 |      129 |      106 | tsm               |
|  9 |         5 |  0.220056  |     0.259259  |       0.0616438 |       0.54902   |    162 |      146 |       51 | tsm               |
| 10 |        20 |  0.426184  |     0.650943  |       0.0350877 |       0.333333  |    212 |      114 |       33 | tsm               |
| 11 |        30 |  0.584958  |     0.928571  |       0         |       0.0952381 |    224 |      114 |       21 | tsm               |

### 4.3 Statistical Significance

Table 3 reports the results of significance tests comparing methods.

|    |   horizon | model          | baseline          |   mean_error_model |   mean_error_baseline |   improvement_pct |   t_statistic |    t_pvalue | t_significant   |   wilcoxon_statistic |   wilcoxon_pvalue | wilcoxon_significant   |   dm_statistic |   dm_pvalue | dm_significant   | model_better   |
|---:|----------:|:---------------|:------------------|-------------------:|----------------------:|------------------:|--------------:|------------:|:----------------|---------------------:|------------------:|:-----------------------|---------------:|------------:|:-----------------|:---------------|
|  0 |         1 | seasonal_naive | naive_persistence |            6.64202 |               1.41603 |    -369.059       |     11.0044   | 1.85805e-24 | True            |                 9862 |       3.80151e-30 | True                   |     11.0044    |    0        | True             | False          |
|  1 |         5 | seasonal_naive | naive_persistence |            6.54358 |               6.54358 |       2.91701e-06 |     -0.438472 | 0.661309    | False           |                23703 |       0.00276108  | True                   |     -0.0115703 |    0.990768 | False            | True           |
|  2 |        20 | seasonal_naive | naive_persistence |           25.939   |              25.939   |       6.53918e-06 |     -0.838913 | 0.402078    | False           |                24683 |       0.00190834  | True                   |     -0.0932438 |    0.92571  | False            | True           |
|  3 |        30 | seasonal_naive | naive_persistence |           37.7942  |              37.7942  |      -7.39789e-06 |     -0.183111 | 0.854815    | False           |                27510 |       0.17839     | False                  |     -0.0293692 |    0.97657  | False            | False          |
|  4 |         1 | tsm            | naive_persistence |            1.45037 |               1.41603 |      -2.42499     |      1.24885  | 0.212536    | False           |                30716 |       0.417891    | False                  |      1.24885   |    0.21172  | False            | False          |
|  5 |         5 | tsm            | naive_persistence |            6.69273 |               6.54358 |      -2.27932     |      1.15495  | 0.248882    | False           |                30498 |       0.357115    | False                  |      0.738296  |    0.460334 | False            | False          |
|  6 |        20 | tsm            | naive_persistence |           25.6175  |              25.939   |       1.23938     |     -0.545662 | 0.585638    | False           |                29034 |       0.0959339   | False                  |     -0.151595  |    0.879507 | False            | True           |
|  7 |        30 | tsm            | naive_persistence |           36.2558  |              37.7942  |       4.0703      |     -1.75517  | 0.0800863   | False           |                25737 |       0.000836396 | True                   |     -0.37393   |    0.708457 | False            | True           |


## 5. Robustness Analysis

### 5.1 Noise Injection Test

To assess robustness, we inject noise into forecast paths at levels of 5%, 10%, 20%, 
and 30% of the forecast standard deviation.

|    |   horizon |      mse |    rmse |      mae |    mape |   noise_level | model   |
|---:|----------:|---------:|--------:|---------:|--------:|--------------:|:--------|
|  0 |         1 |  1.45335 | 1.20555 | 0.968759 | 1.36806 |          0.05 | tsm     |
|  1 |         5 |  6.68102 | 2.58477 | 2.08935  | 2.96677 |          0.05 | tsm     |
|  2 |        20 | 25.6158  | 5.06121 | 3.9107   | 5.48078 |          0.05 | tsm     |
|  3 |        30 | 36.2557  | 6.02127 | 4.76202  | 6.62341 |          0.05 | tsm     |
|  4 |         1 |  1.45761 | 1.20732 | 0.970759 | 1.37088 |          0.1  | tsm     |
|  5 |         5 |  6.67061 | 2.58275 | 2.08928  | 2.96661 |          0.1  | tsm     |
|  6 |        20 | 25.6152  | 5.06115 | 3.90992  | 5.47968 |          0.1  | tsm     |
|  7 |        30 | 36.2568  | 6.02136 | 4.76149  | 6.62292 |          0.1  | tsm     |
|  8 |         1 |  1.46996 | 1.21242 | 0.975854 | 1.37815 |          0.2  | tsm     |
|  9 |         5 |  6.65368 | 2.57947 | 2.0896   | 2.96691 |          0.2  | tsm     |
| 10 |        20 | 25.6175  | 5.06137 | 3.90836  | 5.47748 |          0.2  | tsm     |
| 11 |        30 | 36.2622  | 6.02181 | 4.76055  | 6.62215 |          0.2  | tsm     |
| 12 |         1 |  1.48741 | 1.21959 | 0.98158  | 1.38629 |          0.3  | tsm     |
| 13 |         5 |  6.64194 | 2.5772  | 2.09028  | 2.9677  |          0.3  | tsm     |
| 14 |        20 | 25.6242  | 5.06204 | 3.9068   | 5.47527 |          0.3  | tsm     |
| 15 |        30 | 36.2721  | 6.02263 | 4.75991  | 6.6218  |          0.3  | tsm     |

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

- **Price accuracy**: tsm has the lowest average MSE (17.504).
- **TSM vs naive**: TSM MSE is 1.0x the naive baseline on average.
- **Directional accuracy**: tsm has the highest average trend accuracy (0.383).
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


