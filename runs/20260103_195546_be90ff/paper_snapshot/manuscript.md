# EU ETS Futures Forecasting with TSM + LLM Refinement

**Authors**: Auto-generated

**Generated**: 2026-01-03 20:00:49

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

|    |   horizon |      mse |    rmse |     mae |     mape | model             |
|---:|----------:|---------:|--------:|--------:|---------:|:------------------|
|  0 |         1 |  4.28177 | 2.06924 | 1.4437  |  2.15348 | naive_persistence |
|  1 |         5 | 10.2545  | 3.20227 | 2.4621  |  3.71042 | naive_persistence |
|  2 |        20 | 36.8281  | 6.06862 | 4.85088 |  7.20306 | naive_persistence |
|  3 |        30 | 49.5437  | 7.03873 | 5.68055 |  8.32392 | naive_persistence |
|  4 |         1 | 10.571   | 3.25131 | 2.49754 |  3.76078 | seasonal_naive    |
|  5 |         5 | 10.2545  | 3.20227 | 2.4621  |  3.71042 | seasonal_naive    |
|  6 |        20 | 36.8281  | 6.06862 | 4.85088 |  7.20307 | seasonal_naive    |
|  7 |        30 | 49.5437  | 7.03873 | 5.68055 |  8.32392 | seasonal_naive    |
|  8 |         1 |  4.03213 | 2.00802 | 1.42905 |  2.13686 | tsm               |
|  9 |         5 | 10.3769  | 3.22132 | 2.5045  |  3.78181 | tsm               |
| 10 |        20 | 37.0161  | 6.08409 | 4.7366  |  7.08913 | tsm               |
| 11 |        30 | 50.6975  | 7.12022 | 5.64516 |  8.37205 | tsm               |
| 12 |         1 |  7.12905 | 2.67003 | 1.9069  |  2.99459 | TSM+LLM           |
| 13 |         5 | 17.0584  | 4.13018 | 3.3134  |  5.31984 | TSM+LLM           |
| 14 |        20 | 53.7791  | 7.33342 | 6.1481  |  9.84906 | TSM+LLM           |
| 15 |        30 | 74.2979  | 8.61962 | 7.2633  | 11.3259  | TSM+LLM           |

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
|  0 |         1 | seasonal_naive | naive_persistence |           10.571   |               4.28177 |    -146.884       |     6.07455   | 3.15549e-09 | True            |                15973 |       2.42149e-17 | True                   |      6.07455   | 1.24334e-09 | True             | False          |
|  1 |         5 | seasonal_naive | naive_persistence |           10.2545  |              10.2545  |       4.36825e-06 |     1.6742    | 0.0949572   | False           |                27959 |       0.454816    | False                  |      0.0952472 | 0.924118    | False            | True           |
|  2 |        20 | seasonal_naive | naive_persistence |           36.8281  |              36.8281  |       6.48903e-06 |    -0.918201  | 0.359127    | False           |                25159 |       0.0428928   | True                   |     -0.155897  | 0.876114    | False            | True           |
|  3 |        30 | seasonal_naive | naive_persistence |           49.5437  |              49.5437  |      -6.35439e-06 |    -0.0828849 | 0.933989    | False           |                24945 |       0.0208131   | True                   |     -0.0147042 | 0.988268    | False            | False          |
|  4 |         1 | tsm            | naive_persistence |            4.03213 |               4.28177 |       5.83041     |    -2.2006    | 0.0283974   | True            |                30985 |       0.34885     | False                  |     -2.2006    | 0.0277647   | True             | True           |
|  5 |         5 | tsm            | naive_persistence |           10.3769  |              10.2545  |      -1.19318     |     0.386971  | 0.699006    | False           |                29701 |       0.113814    | False                  |      0.227279  | 0.820207    | False            | False          |
|  6 |        20 | tsm            | naive_persistence |           37.0161  |              36.8281  |      -0.51052     |     0.196443  | 0.844374    | False           |                30654 |       0.270047    | False                  |      0.0599522 | 0.952194    | False            | False          |
|  7 |        30 | tsm            | naive_persistence |           50.6975  |              49.5437  |      -2.32896     |     0.800619  | 0.423879    | False           |                30895 |       0.326103    | False                  |      0.231302  | 0.81708     | False            | False          |
|  8 |         1 | TSM+LLM        | naive_persistence |            7.12905 |               7.33969 |       2.86982     |    -0.884209  | 0.378725    | False           |                 2346 |       0.538251    | False                  |     -0.884209  | 0.376583    | False            | True           |
|  9 |         5 | TSM+LLM        | naive_persistence |           17.0584  |              17.1104  |       0.303883    |    -0.0718707 | 0.94285     | False           |                 2376 |       0.608434    | False                  |     -0.0605645 | 0.951706    | False            | True           |
| 10 |        20 | TSM+LLM        | naive_persistence |           53.7791  |              53.0067  |      -1.45726     |     0.348976  | 0.727848    | False           |                 2456 |       0.812467    | False                  |      0.0983881 | 0.921624    | False            | False          |
| 11 |        30 | TSM+LLM        | naive_persistence |           74.2979  |              73.5514  |      -1.01491     |     0.171258  | 0.864371    | False           |                 2300 |       0.439154    | False                  |      0.060528  | 0.951735    | False            | False          |


## 5. Robustness Analysis

### 5.1 Noise Injection Test

To assess robustness, we inject noise into forecast paths at levels of 5%, 10%, 20%, 
and 30% of the forecast standard deviation.

|    |   horizon |      mse |    rmse |     mae |     mape |   noise_level | model   |
|---:|----------:|---------:|--------:|--------:|---------:|--------------:|:--------|
|  0 |         1 |  4.0173  | 2.00432 | 1.4262  |  2.1324  |          0.05 | tsm     |
|  1 |         5 | 10.3834  | 3.22232 | 2.50605 |  3.78394 |          0.05 | tsm     |
|  2 |        20 | 36.9726  | 6.08051 | 4.73492 |  7.08633 |          0.05 | tsm     |
|  3 |        30 | 50.7019  | 7.12053 | 5.64653 |  8.37426 |          0.05 | tsm     |
|  4 |         1 |  4.00562 | 2.0014  | 1.42404 |  2.12897 |          0.1  | tsm     |
|  5 |         5 | 10.3933  | 3.22386 | 2.5088  |  3.78783 |          0.1  | tsm     |
|  6 |        20 | 36.932   | 6.07717 | 4.73323 |  7.08354 |          0.1  | tsm     |
|  7 |        30 | 50.7092  | 7.12104 | 5.64833 |  8.37709 |          0.1  | tsm     |
|  8 |         1 |  3.99171 | 1.99793 | 1.42204 |  2.12538 |          0.2  | tsm     |
|  9 |         5 | 10.4235  | 3.22854 | 2.51574 |  3.79766 |          0.2  | tsm     |
| 10 |        20 | 36.8596  | 6.07121 | 4.73055 |  7.07899 |          0.2  | tsm     |
| 11 |        30 | 50.7322  | 7.12266 | 5.65271 |  8.38387 |          0.2  | tsm     |
| 12 |         1 |  3.99039 | 1.9976  | 1.42249 |  2.12554 |          0.3  | tsm     |
| 13 |         5 | 10.4675  | 3.23535 | 2.52417 |  3.80963 |          0.3  | tsm     |
| 14 |        20 | 36.7989  | 6.06621 | 4.72974 |  7.0772  |          0.3  | tsm     |
| 15 |        30 | 50.7667  | 7.12507 | 5.6572  |  8.39082 |          0.3  | tsm     |
| 16 |         1 |  7.07589 | 2.66006 | 1.89534 |  2.9767  |          0.05 | TSM+LLM |
| 17 |         5 | 17.0925  | 4.1343  | 3.31628 |  5.32378 |          0.05 | TSM+LLM |
| 18 |        20 | 53.6522  | 7.32476 | 6.1393  |  9.83467 |          0.05 | TSM+LLM |
| 19 |        30 | 74.3945  | 8.62523 | 7.26431 | 11.3299  |          0.05 | TSM+LLM |
| 20 |         1 |  7.02616 | 2.65069 | 1.88558 |  2.96174 |          0.1  | TSM+LLM |
| 21 |         5 | 17.1306  | 4.13891 | 3.32074 |  5.33004 |          0.1  | TSM+LLM |
| 22 |        20 | 53.5301  | 7.31642 | 6.1305  |  9.82027 |          0.1  | TSM+LLM |
| 23 |        30 | 74.5003  | 8.63136 | 7.26532 | 11.3339  |          0.1  | TSM+LLM |
| 24 |         1 |  6.93699 | 2.63382 | 1.86824 |  2.93521 |          0.2  | TSM+LLM |
| 25 |         5 | 17.2189  | 4.14957 | 3.33308 |  5.34763 |          0.2  | TSM+LLM |
| 26 |        20 | 53.3004  | 7.30071 | 6.11291 |  9.79148 |          0.2  | TSM+LLM |
| 27 |        30 | 74.7396  | 8.64521 | 7.2689  | 11.3446  |          0.2  | TSM+LLM |
| 28 |         1 |  6.86153 | 2.61945 | 1.85644 |  2.91684 |          0.3  | TSM+LLM |
| 29 |         5 | 17.3233  | 4.16213 | 3.34542 |  5.36521 |          0.3  | TSM+LLM |
| 30 |        20 | 53.0902  | 7.2863  | 6.09531 |  9.76269 |          0.3  | TSM+LLM |
| 31 |        30 | 75.0157  | 8.66116 | 7.27495 | 11.3595  |          0.3  | TSM+LLM |

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


