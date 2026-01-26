# EU ETS Futures Forecasting with TSM + LLM Refinement

**Authors**: Auto-generated

**Generated**: 2026-01-03 21:22:30

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

|    |   horizon |      mse |    rmse |     mae |     mape | model                        |
|---:|----------:|---------:|--------:|--------:|---------:|:-----------------------------|
|  0 |         1 |  4.28177 | 2.06924 | 1.4437  |  2.15348 | naive_persistence            |
|  1 |         5 | 10.2545  | 3.20227 | 2.4621  |  3.71042 | naive_persistence            |
|  2 |        20 | 36.8281  | 6.06862 | 4.85088 |  7.20306 | naive_persistence            |
|  3 |        30 | 49.5437  | 7.03873 | 5.68055 |  8.32392 | naive_persistence            |
|  4 |         1 | 10.571   | 3.25131 | 2.49754 |  3.76078 | seasonal_naive               |
|  5 |         5 | 10.2545  | 3.20227 | 2.4621  |  3.71042 | seasonal_naive               |
|  6 |        20 | 36.8281  | 6.06862 | 4.85088 |  7.20307 | seasonal_naive               |
|  7 |        30 | 49.5437  | 7.03873 | 5.68055 |  8.32392 | seasonal_naive               |
|  8 |         1 |  4.03212 | 2.00802 | 1.42905 |  2.13686 | tsm                          |
|  9 |         5 | 10.3769  | 3.22132 | 2.5045  |  3.78182 | tsm                          |
| 10 |        20 | 37.0161  | 6.08408 | 4.73659 |  7.08912 | tsm                          |
| 11 |        30 | 50.697   | 7.12018 | 5.64514 |  8.37203 | tsm                          |
| 12 |         1 |  6.41154 | 2.5321  | 1.83288 |  2.82729 | TSM+LLM-DELTA                |
| 13 |         5 | 16.0169  | 4.00211 | 3.15488 |  4.97229 | TSM+LLM-DELTA                |
| 14 |        20 | 51.5379  | 7.17899 | 6.07936 |  9.59483 | TSM+LLM-DELTA                |
| 15 |        30 | 73.0325  | 8.54591 | 7.25041 | 11.2028  | TSM+LLM-DELTA                |
| 16 |         1 |  6.93499 | 2.63344 | 1.8919  |  2.91034 | naive_persistence_llm_subset |
| 17 |         5 | 17.0235  | 4.12595 | 3.2639  |  5.09711 | naive_persistence_llm_subset |
| 18 |        20 | 51.9427  | 7.20713 | 6.0737  |  9.49516 | naive_persistence_llm_subset |
| 19 |        30 | 68.1377  | 8.25456 | 6.6707  | 10.2125  | naive_persistence_llm_subset |
| 20 |         1 | 17.6057  | 4.19591 | 3.3152  |  5.14732 | seasonal_naive_llm_subset    |
| 21 |         5 | 17.0235  | 4.12595 | 3.2639  |  5.09711 | seasonal_naive_llm_subset    |
| 22 |        20 | 51.9427  | 7.20713 | 6.0737  |  9.49516 | seasonal_naive_llm_subset    |
| 23 |        30 | 68.1377  | 8.25456 | 6.6707  | 10.2125  | seasonal_naive_llm_subset    |
| 24 |         1 |  6.40641 | 2.53109 | 1.83306 |  2.82804 | tsm_llm_subset               |
| 25 |         5 | 16.0045  | 4.00057 | 3.15594 |  4.97476 | tsm_llm_subset               |
| 26 |        20 | 51.1212  | 7.14991 | 6.05225 |  9.55164 | tsm_llm_subset               |
| 27 |        30 | 71.7357  | 8.46969 | 7.17542 | 11.0864  | tsm_llm_subset               |

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

|    |   horizon | model                     | baseline                     |   mean_error_model |   mean_error_baseline |   improvement_pct |   t_statistic |    t_pvalue | t_significant   |   wilcoxon_statistic |   wilcoxon_pvalue | wilcoxon_significant   |   dm_statistic |   dm_pvalue | dm_significant   | model_better   |
|---:|----------:|:--------------------------|:-----------------------------|-------------------:|----------------------:|------------------:|--------------:|------------:|:----------------|---------------------:|------------------:|:-----------------------|---------------:|------------:|:-----------------|:---------------|
|  0 |         1 | seasonal_naive_llm_subset | naive_persistence_llm_subset |           17.6057  |               6.93499 |    -153.867       |      3.53121  | 0.000629775 | True            |                 1285 |       2.01229e-05 | True                   |      3.53121   | 0.000413667 | True             | False          |
|  1 |         5 | seasonal_naive_llm_subset | naive_persistence_llm_subset |           17.0235  |              17.0235  |       6.00632e-06 |      1.28448  | 0.201972    | False           |                 2304 |       0.930106    | False                  |      0.118121  | 0.905972    | False            | True           |
|  2 |        20 | seasonal_naive_llm_subset | naive_persistence_llm_subset |           51.9427  |              51.9427  |      -3.02218e-06 |     -0.238721 | 0.811816    | False           |                 2097 |       0.870069    | False                  |     -0.0511463 | 0.959209    | False            | False          |
|  3 |        30 | seasonal_naive_llm_subset | naive_persistence_llm_subset |           68.1377  |              68.1377  |       6.34305e-06 |     -0.311102 | 0.756377    | False           |                 1924 |       0.139831    | False                  |     -0.0784615 | 0.937461    | False            | True           |
|  4 |         1 | tsm_llm_subset            | naive_persistence_llm_subset |            6.40641 |               6.93499 |       7.62181     |     -1.64398  | 0.103352    | False           |                 2170 |       0.222235    | False                  |     -1.64398   | 0.10018     | False            | True           |
|  5 |         5 | tsm_llm_subset            | naive_persistence_llm_subset |           16.0045  |              17.0235  |       5.98557     |     -1.09165  | 0.277635    | False           |                 2247 |       0.339146    | False                  |     -0.725838  | 0.467938    | False            | True           |
|  6 |        20 | tsm_llm_subset            | naive_persistence_llm_subset |           51.1212  |              51.9427  |       1.58143     |     -0.412265 | 0.681036    | False           |                 2462 |       0.828509    | False                  |     -0.121835  | 0.903029    | False            | True           |
|  7 |        30 | tsm_llm_subset            | naive_persistence_llm_subset |           71.7357  |              68.1377  |      -5.28045     |      1.04134  | 0.300255    | False           |                 1835 |       0.0176708   | True                   |      0.469293  | 0.63886     | False            | False          |
|  8 |         1 | TSM+LLM-DELTA             | naive_persistence_llm_subset |            6.41154 |               6.93499 |       7.54792     |     -1.67912  | 0.0962814   | False           |                 2173 |       0.226168    | False                  |     -1.67912   | 0.0931278   | False            | True           |
|  9 |         5 | TSM+LLM-DELTA             | naive_persistence_llm_subset |           16.0169  |              17.0235  |       5.91317     |     -1.11702  | 0.26669     | False           |                 2232 |       0.313729    | False                  |     -0.736832  | 0.461225    | False            | True           |
| 10 |        20 | TSM+LLM-DELTA             | naive_persistence_llm_subset |           51.5379  |              51.9427  |       0.779248    |     -0.204388 | 0.83847     | False           |                 2473 |       0.8581      | False                  |     -0.0596706 | 0.952418    | False            | True           |
| 11 |        30 | TSM+LLM-DELTA             | naive_persistence_llm_subset |           73.0325  |              68.1377  |      -7.18362     |      1.47246  | 0.14407     | False           |                 1721 |       0.00570253  | True                   |      0.60942   | 0.542246    | False            | False          |


## 5. Robustness Analysis

### 5.1 Noise Injection Test

To assess robustness, we inject noise into forecast paths at levels of 5%, 10%, 20%, 
and 30% of the forecast standard deviation.

|    |   horizon |      mse |    rmse |     mae |     mape |   noise_level | model         |
|---:|----------:|---------:|--------:|--------:|---------:|--------------:|:--------------|
|  0 |         1 |  4.0173  | 2.00432 | 1.4262  |  2.1324  |          0.05 | tsm           |
|  1 |         5 | 10.3834  | 3.22232 | 2.50605 |  3.78394 |          0.05 | tsm           |
|  2 |        20 | 36.9726  | 6.08051 | 4.73491 |  7.08632 |          0.05 | tsm           |
|  3 |        30 | 50.7014  | 7.12049 | 5.64651 |  8.37423 |          0.05 | tsm           |
|  4 |         1 |  4.00562 | 2.0014  | 1.42404 |  2.12897 |          0.1  | tsm           |
|  5 |         5 | 10.3933  | 3.22386 | 2.5088  |  3.78783 |          0.1  | tsm           |
|  6 |        20 | 36.932   | 6.07717 | 4.73323 |  7.08353 |          0.1  | tsm           |
|  7 |        30 | 50.7087  | 7.121   | 5.6483  |  8.37705 |          0.1  | tsm           |
|  8 |         1 |  3.99171 | 1.99793 | 1.42204 |  2.12538 |          0.2  | tsm           |
|  9 |         5 | 10.4235  | 3.22854 | 2.51574 |  3.79766 |          0.2  | tsm           |
| 10 |        20 | 36.8596  | 6.07121 | 4.73054 |  7.07898 |          0.2  | tsm           |
| 11 |        30 | 50.7317  | 7.12262 | 5.65268 |  8.38384 |          0.2  | tsm           |
| 12 |         1 |  3.99039 | 1.9976  | 1.42249 |  2.12554 |          0.3  | tsm           |
| 13 |         5 | 10.4675  | 3.23535 | 2.52417 |  3.80963 |          0.3  | tsm           |
| 14 |        20 | 36.7989  | 6.06621 | 4.72974 |  7.0772  |          0.3  | tsm           |
| 15 |        30 | 50.7662  | 7.12504 | 5.65717 |  8.39078 |          0.3  | tsm           |
| 16 |         1 | 17.1227  | 4.13796 | 3.03957 |  4.87731 |          0.05 | TSM+LLM-DELTA |
| 17 |         5 | 17.2196  | 4.14965 | 3.28612 |  5.38057 |          0.05 | TSM+LLM-DELTA |
| 18 |        20 | 36.8385  | 6.06947 | 4.96714 |  8.08914 |          0.05 | TSM+LLM-DELTA |
| 19 |        30 | 61.5937  | 7.84816 | 6.51576 | 10.2246  |          0.05 | TSM+LLM-DELTA |
| 20 |         1 | 17.0746  | 4.13214 | 3.03362 |  4.86839 |          0.1  | TSM+LLM-DELTA |
| 21 |         5 | 17.2063  | 4.14805 | 3.28421 |  5.37816 |          0.1  | TSM+LLM-DELTA |
| 22 |        20 | 36.7121  | 6.05905 | 4.95911 |  8.07572 |          0.1  | TSM+LLM-DELTA |
| 23 |        30 | 61.5775  | 7.84713 | 6.51524 | 10.225   |          0.1  | TSM+LLM-DELTA |
| 24 |         1 | 16.9846  | 4.12123 | 3.02206 |  4.85115 |          0.2  | TSM+LLM-DELTA |
| 25 |         5 | 17.1888  | 4.14593 | 3.28138 |  5.37479 |          0.2  | TSM+LLM-DELTA |
| 26 |        20 | 36.4675  | 6.03883 | 4.94462 |  8.05114 |          0.2  | TSM+LLM-DELTA |
| 27 |        30 | 61.5561  | 7.84577 | 6.51526 | 10.2273  |          0.2  | TSM+LLM-DELTA |
| 28 |         1 | 16.9029  | 4.11131 | 3.012   |  4.83659 |          0.3  | TSM+LLM-DELTA |
| 29 |         5 | 17.1831  | 4.14525 | 3.2786  |  5.37148 |          0.3  | TSM+LLM-DELTA |
| 30 |        20 | 36.2339  | 6.01946 | 4.93014 |  8.02656 |          0.3  | TSM+LLM-DELTA |
| 31 |        30 | 61.5491  | 7.84532 | 6.51535 | 10.2298  |          0.3  | TSM+LLM-DELTA |

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


