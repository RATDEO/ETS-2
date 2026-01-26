# EU ETS Futures Forecasting with TSM + LLM Refinement

**Authors**: Auto-generated

**Generated**: 2026-01-03 20:58:03

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

|    |   horizon |      mse |    rmse |     mae |    mape | model             |
|---:|----------:|---------:|--------:|--------:|--------:|:------------------|
|  0 |         1 |  4.28177 | 2.06924 | 1.4437  | 2.15348 | naive_persistence |
|  1 |         5 | 10.2545  | 3.20227 | 2.4621  | 3.71042 | naive_persistence |
|  2 |        20 | 36.8281  | 6.06862 | 4.85088 | 7.20306 | naive_persistence |
|  3 |        30 | 49.5437  | 7.03873 | 5.68055 | 8.32392 | naive_persistence |
|  4 |         1 | 10.571   | 3.25131 | 2.49754 | 3.76078 | seasonal_naive    |
|  5 |         5 | 10.2545  | 3.20227 | 2.4621  | 3.71042 | seasonal_naive    |
|  6 |        20 | 36.8281  | 6.06862 | 4.85088 | 7.20307 | seasonal_naive    |
|  7 |        30 | 49.5437  | 7.03873 | 5.68055 | 8.32392 | seasonal_naive    |
|  8 |         1 |  4.05796 | 2.01444 | 1.41937 | 2.11981 | tsm               |
|  9 |         5 | 10.5025  | 3.24076 | 2.49905 | 3.75555 | tsm               |
| 10 |        20 | 39.4493  | 6.28087 | 4.8644  | 7.16743 | tsm               |
| 11 |        30 | 57.4344  | 7.57855 | 5.8492  | 8.49644 | tsm               |

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
|  8 |         1 |  0.350829  |      0.147541 |        0.232394 |        0.77551  |    122 |      142 |       98 | tsm               |
|  9 |         5 |  0.356354  |      0.225166 |        0.449102 |        0.454545 |    151 |      167 |       44 | tsm               |
| 10 |        20 |  0.466851  |      0.233696 |        0.787097 |        0.173913 |    184 |      155 |       23 | tsm               |
| 11 |        30 |  0.444751  |      0.27     |        0.704698 |        0.153846 |    200 |      149 |       13 | tsm               |

### 4.3 Statistical Significance

Table 3 reports the results of significance tests comparing methods.

|    |   horizon | model          | baseline          |   mean_error_model |   mean_error_baseline |   improvement_pct |   t_statistic |    t_pvalue | t_significant   |   wilcoxon_statistic |   wilcoxon_pvalue | wilcoxon_significant   |   dm_statistic |   dm_pvalue | dm_significant   | model_better   |
|---:|----------:|:---------------|:------------------|-------------------:|----------------------:|------------------:|--------------:|------------:|:----------------|---------------------:|------------------:|:-----------------------|---------------:|------------:|:-----------------|:---------------|
|  0 |         1 | seasonal_naive | naive_persistence |           10.571   |               4.28177 |    -146.884       |     6.07455   | 3.15549e-09 | True            |                15973 |       2.42149e-17 | True                   |      6.07455   | 1.24334e-09 | True             | False          |
|  1 |         5 | seasonal_naive | naive_persistence |           10.2545  |              10.2545  |       4.36825e-06 |     1.6742    | 0.0949572   | False           |                27959 |       0.454816    | False                  |      0.0952472 | 0.924118    | False            | True           |
|  2 |        20 | seasonal_naive | naive_persistence |           36.8281  |              36.8281  |       6.48903e-06 |    -0.918201  | 0.359127    | False           |                25159 |       0.0428928   | True                   |     -0.155897  | 0.876114    | False            | True           |
|  3 |        30 | seasonal_naive | naive_persistence |           49.5437  |              49.5437  |      -6.35439e-06 |    -0.0828849 | 0.933989    | False           |                24945 |       0.0208131   | True                   |     -0.0147042 | 0.988268    | False            | False          |
|  4 |         1 | tsm            | naive_persistence |            4.05796 |               4.28177 |       5.22707     |    -1.99786   | 0.0464818   | True            |                29895 |       0.137833    | False                  |     -1.99786   | 0.0457319   | True             | True           |
|  5 |         5 | tsm            | naive_persistence |           10.5025  |              10.2545  |      -2.41855     |     0.798997  | 0.424818    | False           |                32128 |       0.716504    | False                  |      0.526033  | 0.598865    | False            | False          |
|  6 |        20 | tsm            | naive_persistence |           39.4493  |              36.8281  |      -7.1174      |     1.94897   | 0.0520733   | False           |                32098 |       0.705288    | False                  |      0.495887  | 0.619974    | False            | False          |
|  7 |        30 | tsm            | naive_persistence |           57.4344  |              49.5437  |     -15.9268      |     2.98129   | 0.00306509  | True            |                30176 |       0.179314    | False                  |      0.671007  | 0.502216    | False            | False          |


## 5. Robustness Analysis

### 5.1 Noise Injection Test

To assess robustness, we inject noise into forecast paths at levels of 5%, 10%, 20%, 
and 30% of the forecast standard deviation.

|    |   horizon |      mse |    rmse |     mae |    mape |   noise_level | model   |
|---:|----------:|---------:|--------:|--------:|--------:|--------------:|:--------|
|  0 |         1 |  4.03426 | 2.00855 | 1.41424 | 2.1122  |          0.05 | tsm     |
|  1 |         5 | 10.5389  | 3.24636 | 2.50339 | 3.7617  |          0.05 | tsm     |
|  2 |        20 | 39.4099  | 6.27773 | 4.86236 | 7.16383 |          0.05 | tsm     |
|  3 |        30 | 57.3492  | 7.57293 | 5.84946 | 8.49692 |          0.05 | tsm     |
|  4 |         1 |  4.01557 | 2.00389 | 1.41046 | 2.10648 |          0.1  | tsm     |
|  5 |         5 | 10.5803  | 3.25274 | 2.50824 | 3.7686  |          0.1  | tsm     |
|  6 |        20 | 39.3752  | 6.27497 | 4.86144 | 7.16182 |          0.1  | tsm     |
|  7 |        30 | 57.2685  | 7.5676  | 5.84974 | 8.49745 |          0.1  | tsm     |
|  8 |         1 |  3.99321 | 1.9983  | 1.4061  | 2.09967 |          0.2  | tsm     |
|  9 |         5 | 10.6786  | 3.26781 | 2.51879 | 3.78361 |          0.2  | tsm     |
| 10 |        20 | 39.3199  | 6.27056 | 4.86009 | 7.1585  |          0.2  | tsm     |
| 11 |        30 | 57.1206  | 7.55782 | 5.85069 | 8.49909 |          0.2  | tsm     |
| 12 |         1 |  3.99087 | 1.99772 | 1.40994 | 2.10477 |          0.3  | tsm     |
| 13 |         5 | 10.7974  | 3.28594 | 2.53039 | 3.80026 |          0.3  | tsm     |
| 14 |        20 | 39.2833  | 6.26764 | 4.85938 | 7.15611 |          0.3  | tsm     |
| 15 |        30 | 56.9907  | 7.54922 | 5.8521  | 8.5014  |          0.3  | tsm     |

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
- **TSM vs naive**: TSM MSE is 1.1x the naive baseline on average.
- **Directional accuracy**: tsm has the highest average trend accuracy (0.405).
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


