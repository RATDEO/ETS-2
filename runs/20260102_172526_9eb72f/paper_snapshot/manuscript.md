# EU ETS Futures Forecasting with TSM + LLM Refinement

**Authors**: Auto-generated

**Generated**: 2026-01-02 17:25:37

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

|    |   horizon |        mse |     rmse |      mae |     mape | model             |
|---:|----------:|-----------:|---------:|---------:|---------:|:------------------|
|  0 |         1 |    4.28177 |  2.06924 |  1.4437  |  2.15348 | naive_persistence |
|  1 |         5 |   10.2545  |  3.20227 |  2.4621  |  3.71042 | naive_persistence |
|  2 |        20 |   36.8281  |  6.06862 |  4.85088 |  7.20306 | naive_persistence |
|  3 |        30 |   49.5437  |  7.03873 |  5.68055 |  8.32392 | naive_persistence |
|  4 |         1 |   10.571   |  3.25131 |  2.49754 |  3.76078 | seasonal_naive    |
|  5 |         5 |   10.2545  |  3.20227 |  2.4621  |  3.71042 | seasonal_naive    |
|  6 |        20 |   36.8281  |  6.06862 |  4.85088 |  7.20307 | seasonal_naive    |
|  7 |        30 |   49.5437  |  7.03873 |  5.68055 |  8.32392 | seasonal_naive    |
|  8 |         1 | 2023.74    | 44.986   | 44.6809  | 66.1163  | tsm               |
|  9 |         5 | 2028.81    | 45.0423  | 44.7326  | 66.1596  | tsm               |
| 10 |        20 | 2074.58    | 45.5476  | 45.2535  | 66.4559  | tsm               |
| 11 |        30 | 2123.3     | 46.0793  | 45.7711  | 66.5974  | tsm               |

### 4.2 Trend Classification Accuracy

Table 2 shows the accuracy of predicting price direction (up/flat/down).

|    |   horizon |   accuracy |   accuracy_up |   accuracy_down |   accuracy_flat |   n_up |   n_down |   n_flat | model             |
|---:|----------:|-----------:|--------------:|----------------:|----------------:|-------:|---------:|---------:|:------------------|
|  0 |         1 |  0.270718  |      0        |        0        |        1        |    122 |      142 |       98 | naive_persistence |
|  1 |         5 |  0.121547  |      0        |        0        |        1        |    151 |      167 |       44 | naive_persistence |
|  2 |        20 |  0.0635359 |      0        |        0        |        1        |    184 |      155 |       23 | naive_persistence |
|  3 |        30 |  0.0359116 |      0        |        0        |        1        |    200 |      149 |       13 | naive_persistence |
|  4 |         1 |  0.41989   |      0.590164 |        0.464789 |        0.142857 |    122 |      142 |       98 | seasonal_naive    |
|  5 |         5 |  0.121547  |      0        |        0        |        1        |    151 |      167 |       44 | seasonal_naive    |
|  6 |        20 |  0.0635359 |      0        |        0        |        1        |    184 |      155 |       23 | seasonal_naive    |
|  7 |        30 |  0.0359116 |      0        |        0        |        1        |    200 |      149 |       13 | seasonal_naive    |
|  8 |         1 |  0.392265  |      0        |        1        |        0        |    122 |      142 |       98 | tsm               |
|  9 |         5 |  0.461326  |      0        |        1        |        0        |    151 |      167 |       44 | tsm               |
| 10 |        20 |  0.428177  |      0        |        1        |        0        |    184 |      155 |       23 | tsm               |
| 11 |        30 |  0.411602  |      0        |        1        |        0        |    200 |      149 |       13 | tsm               |

### 4.3 Statistical Significance

Table 3 reports the results of significance tests comparing methods.

|    |   horizon | model          | baseline          |   mean_error_model |   mean_error_baseline |   improvement_pct |   t_statistic |     t_pvalue | t_significant   |   wilcoxon_statistic |   wilcoxon_pvalue | wilcoxon_significant   |   dm_statistic |   dm_pvalue | dm_significant   | model_better   |
|---:|----------:|:---------------|:------------------|-------------------:|----------------------:|------------------:|--------------:|-------------:|:----------------|---------------------:|------------------:|:-----------------------|---------------:|------------:|:-----------------|:---------------|
|  0 |         1 | seasonal_naive | naive_persistence |            10.571  |               4.28177 |    -146.884       |      6.07455  | 3.15547e-09  | True            |                15973 |       2.42149e-17 | True                   |      6.07455   | 1.24333e-09 | True             | False          |
|  1 |         5 | seasonal_naive | naive_persistence |            10.2545 |              10.2545  |      -1.06104e-05 |      1.41614  | 0.157596     | False           |                26437 |       0.187844    | False                  |      0.0636682 | 0.949234    | False            | False          |
|  2 |        20 | seasonal_naive | naive_persistence |            36.8281 |              36.8281  |       2.26968e-06 |     -0.698159 | 0.485527     | False           |                24292 |       0.00966441  | True                   |     -0.117386  | 0.906554    | False            | True           |
|  3 |        30 | seasonal_naive | naive_persistence |            49.5437 |              49.5437  |       3.0175e-06  |     -0.155382 | 0.876607     | False           |                27420 |       0.258151    | False                  |     -0.0294068 | 0.97654     | False            | True           |
|  4 |         1 | tsm            | naive_persistence |          2023.74   |               4.28177 |  -47164           |     83.6293   | 2.22375e-238 | True            |                    0 |       4.42975e-61 | True                   |     83.6293    | 0           | True             | False          |
|  5 |         5 | tsm            | naive_persistence |          2028.81   |              10.2545  |  -19684.5         |     82.6776   | 1.12913e-236 | True            |                    0 |       4.42975e-61 | True                   |     27.8445    | 0           | True             | False          |
|  6 |        20 | tsm            | naive_persistence |          2074.58   |              36.8281  |   -5533.15        |     85.2337   | 3.24388e-241 | True            |                    0 |       4.42975e-61 | True                   |     15.7868    | 0           | True             | False          |
|  7 |        30 | tsm            | naive_persistence |          2123.3    |              49.5437  |   -4185.71        |     82.911    | 4.29274e-237 | True            |                    0 |       4.42975e-61 | True                   |     15.4916    | 0           | True             | False          |


## 5. Robustness Analysis

### 5.1 Noise Injection Test

To assess robustness, we inject noise into forecast paths at levels of 5%, 10%, 20%, 
and 30% of the forecast standard deviation.

|    |   horizon |     mse |    rmse |     mae |    mape |   noise_level | model   |
|---:|----------:|--------:|--------:|--------:|--------:|--------------:|:--------|
|  0 |         1 | 2023.71 | 44.9857 | 44.6806 | 66.1159 |          0.05 | tsm     |
|  1 |         5 | 2029    | 45.0444 | 44.7351 | 66.1636 |          0.05 | tsm     |
|  2 |        20 | 2074.68 | 45.5487 | 45.2546 | 66.4572 |          0.05 | tsm     |
|  3 |        30 | 2122.27 | 46.0682 | 45.7595 | 66.5804 |          0.05 | tsm     |
|  4 |         1 | 2023.7  | 44.9855 | 44.6802 | 66.1155 |          0.1  | tsm     |
|  5 |         5 | 2029.19 | 45.0466 | 44.7376 | 66.1677 |          0.1  | tsm     |
|  6 |        20 | 2074.79 | 45.5498 | 45.2556 | 66.4585 |          0.1  | tsm     |
|  7 |        30 | 2121.26 | 46.0571 | 45.748  | 66.5634 |          0.1  | tsm     |
|  8 |         1 | 2023.7  | 44.9855 | 44.6796 | 66.1148 |          0.2  | tsm     |
|  9 |         5 | 2029.61 | 45.0512 | 44.7426 | 66.1757 |          0.2  | tsm     |
| 10 |        20 | 2075.03 | 45.5524 | 45.2577 | 66.4611 |          0.2  | tsm     |
| 11 |        30 | 2119.26 | 46.0355 | 45.725  | 66.5294 |          0.2  | tsm     |
| 12 |         1 | 2023.74 | 44.986  | 44.679  | 66.114  |          0.3  | tsm     |
| 13 |         5 | 2030.07 | 45.0563 | 44.7476 | 66.1838 |          0.3  | tsm     |
| 14 |        20 | 2075.3  | 45.5555 | 45.2597 | 66.4636 |          0.3  | tsm     |
| 15 |        30 | 2117.31 | 46.0142 | 45.702  | 66.4955 |          0.3  | tsm     |

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
- **TSM vs naive**: TSM MSE is 81.8x the naive baseline on average.
- **Directional accuracy**: tsm has the highest average trend accuracy (0.423).
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
Price accuracy is best for naive_persistence, while directional accuracy is highest for tsm.
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


