# EU ETS Futures Forecasting with TSM + LLM Refinement

**Authors**: Auto-generated

**Generated**: 2026-03-06 13:04:20

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
- **Number of Features**: 59

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

where $L = 120$ (lookback window) and $H = 30$ (forecast horizon).

### 3.2 Baseline Models

We implement several baseline models for comparison:

1. **Naive Persistence**: $\hat{y}_{t+h} = y_t$ for all horizons
2. **Seasonal Naive**: $\hat{y}_{t+h} = y_{t+h-5}$ (weekly seasonality)
3. **Linear Regression**: Ridge regression on lagged features
4. **ARIMA**: Autoregressive integrated moving average model

### 3.3 Time Series Model (TSM)

The primary TSM uses an Autoformer-style architecture with:
- **Model dimension**: 512
- **Attention heads**: 8
- **Encoder layers**: 2
- **Feed-forward dimension**: 2048
- **Dropout**: 0.05

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
|  8 |         1 |  1.12585 | 1.06106 | 0.796709 |  1.40645 | linear_ridge      |
|  9 |         5 |  5.21664 | 2.284   | 1.67399  |  2.89164 | linear_ridge      |
| 10 |        20 | 30.9876  | 5.56665 | 3.38132  |  6.12745 | linear_ridge      |
| 11 |        30 | 57.2455  | 7.56608 | 5.07786  |  9.64473 | linear_ridge      |
| 12 |         1 |  1.16458 | 1.07916 | 0.834786 |  1.49408 | linear_lasso      |
| 13 |         5 |  4.76582 | 2.18308 | 1.61612  |  2.80347 | linear_lasso      |
| 14 |        20 | 34.9088  | 5.90837 | 3.74737  |  6.69006 | linear_lasso      |
| 15 |        30 | 64.8328  | 8.05188 | 5.7663   | 10.7914  | linear_lasso      |
| 16 |         1 |  1.28868 | 1.1352  | 0.828851 |  1.46802 | tsm               |
| 17 |         5 |  9.41471 | 3.06834 | 2.41686  |  4.18725 | tsm               |
| 18 |        20 | 54.814   | 7.40365 | 5.46772  |  9.81378 | tsm               |
| 19 |        30 | 89.8564  | 9.47926 | 7.67568  | 14.1569  | tsm               |

### 4.2 Trend Classification Accuracy

Table 2 shows the accuracy of predicting price direction (up/flat/down).

|    |   horizon |   accuracy |   accuracy_up |   accuracy_down |   accuracy_flat |   n_up |   n_down |   n_flat | model             |
|---:|----------:|-----------:|--------------:|----------------:|----------------:|-------:|---------:|---------:|:------------------|
|  0 |         1 |  0.395833  |     0         |        0        |       1         |     52 |       35 |       57 | naive_persistence |
|  1 |         5 |  0.236111  |     0         |        0        |       1         |     70 |       40 |       34 | naive_persistence |
|  2 |        20 |  0.118056  |     0         |        0        |       1         |     90 |       37 |       17 | naive_persistence |
|  3 |        30 |  0.0763889 |     0         |        0        |       1         |    100 |       33 |       11 | naive_persistence |
|  4 |         1 |  0.375     |     0.365385  |        0.628571 |       0.22807   |     52 |       35 |       57 | seasonal_naive    |
|  5 |         5 |  0.236111  |     0         |        0        |       1         |     70 |       40 |       34 | seasonal_naive    |
|  6 |        20 |  0.118056  |     0         |        0        |       1         |     90 |       37 |       17 | seasonal_naive    |
|  7 |        30 |  0.0763889 |     0         |        0        |       1         |    100 |       33 |       11 | seasonal_naive    |
|  8 |         1 |  0.444444  |     0.0576923 |        0.142857 |       0.982456  |     52 |       35 |       57 | linear_ridge      |
|  9 |         5 |  0.319444  |     0.0714286 |        0.325    |       0.823529  |     70 |       40 |       34 | linear_ridge      |
| 10 |        20 |  0.729167  |     0.933333  |        0.513514 |       0.117647  |     90 |       37 |       17 | linear_ridge      |
| 11 |        30 |  0.854167  |     0.93      |        0.878788 |       0.0909091 |    100 |       33 |       11 | linear_ridge      |
| 12 |         1 |  0.451389  |     0.384615  |        0.114286 |       0.719298  |     52 |       35 |       57 | linear_lasso      |
| 13 |         5 |  0.375     |     0.357143  |        0.15     |       0.676471  |     70 |       40 |       34 | linear_lasso      |
| 14 |        20 |  0.625     |     0.644444  |        0.621622 |       0.529412  |     90 |       37 |       17 | linear_lasso      |
| 15 |        30 |  0.465278  |     0.27      |        0.969697 |       0.727273  |    100 |       33 |       11 | linear_lasso      |
| 16 |         1 |  0.423611  |     0.0961538 |        0.142857 |       0.894737  |     52 |       35 |       57 | tsm               |
| 17 |         5 |  0.270833  |     0.128571  |        0.45     |       0.352941  |     70 |       40 |       34 | tsm               |
| 18 |        20 |  0.270833  |     0.0666667 |        0.864865 |       0.0588235 |     90 |       37 |       17 | tsm               |
| 19 |        30 |  0.222222  |     0.03      |        0.848485 |       0.0909091 |    100 |       33 |       11 | tsm               |

### 4.3 Statistical Significance

Table 3 reports the results of significance tests comparing methods.

|    |   horizon | model          | baseline          |   mean_error_model |   mean_error_baseline |   improvement_pct |   t_statistic |    t_pvalue | t_significant   |   wilcoxon_statistic |   wilcoxon_pvalue | wilcoxon_significant   |   dm_statistic |   dm_pvalue | dm_significant   | model_better   |
|---:|----------:|:---------------|:------------------|-------------------:|----------------------:|------------------:|--------------:|------------:|:----------------|---------------------:|------------------:|:-----------------------|---------------:|------------:|:-----------------|:---------------|
|  0 |         1 | seasonal_naive | naive_persistence |            5.16853 |               1.06139 |    -386.957       |     5.07044   | 1.21001e-06 | True            |                 1834 |       1.45104e-11 | True                   |      5.07044   | 3.96893e-07 | True             | False          |
|  1 |         5 | seasonal_naive | naive_persistence |            5.2119  |               5.2119  |       5.49269e-07 |     1.19107   | 0.235598    | False           |                 4258 |       0.556904    | False                  |      0.0263232 | 0.979       | False            | True           |
|  2 |        20 | seasonal_naive | naive_persistence |           44.2759  |              44.2759  |       2.22491e-06 |     0.67977   | 0.497749    | False           |                 4233 |       0.183904    | False                  |      0.216494  | 0.828603    | False            | True           |
|  3 |        30 | seasonal_naive | naive_persistence |           79.9881  |              79.9881  |       3.104e-06   |    -0.578827  | 0.563616    | False           |                 3978 |       0.082287    | False                  |     -0.245123  | 0.806361    | False            | True           |
|  4 |         1 | linear_ridge   | naive_persistence |            1.12585 |               1.06139 |      -6.07269     |     0.874405  | 0.383364    | False           |                 4999 |       0.6594      | False                  |      0.874405  | 0.381898    | False            | False          |
|  5 |         5 | linear_ridge   | naive_persistence |            5.21664 |               5.2119  |      -0.0909839   |     0.0181722 | 0.985527    | False           |                 4879 |       0.496468    | False                  |      0.0110311 | 0.991199    | False            | False          |
|  6 |        20 | linear_ridge   | naive_persistence |           30.9876  |              44.2759  |      30.0125      |    -5.87176   | 2.88698e-08 | True            |                 1922 |       4.79338e-11 | True                   |     -2.55081   | 0.0107474   | True             | True           |
|  7 |        30 | linear_ridge   | naive_persistence |           57.2455  |              79.9881  |      28.4324      |    -6.6463    | 5.87392e-10 | True            |                  379 |       4.7078e-22  | True                   |     -1.56448   | 0.117704    | False            | True           |
|  8 |         1 | linear_lasso   | naive_persistence |            1.16458 |               1.06139 |      -9.72149     |     1.21133   | 0.227767    | False           |                 4508 |       0.155624    | False                  |      1.21133   | 0.225769    | False            | False          |
|  9 |         5 | linear_lasso   | naive_persistence |            4.76582 |               5.2119  |       8.55879     |    -1.81065   | 0.072294    | False           |                 4281 |       0.0611165   | False                  |     -1.11004   | 0.266983    | False            | True           |
| 10 |        20 | linear_lasso   | naive_persistence |           34.9088  |              44.2759  |      21.156       |    -4.35363   | 2.53581e-05 | True            |                 1590 |       4.50923e-13 | True                   |     -1.99486   | 0.0460582   | True             | True           |
| 11 |        30 | linear_lasso   | naive_persistence |           64.8328  |              79.9881  |      18.9469      |    -4.5339    | 1.21449e-05 | True            |                 2811 |       1.5531e-06  | True                   |     -1.15744   | 0.247094    | False            | True           |
| 12 |         1 | tsm            | naive_persistence |            1.28868 |               1.06139 |     -21.4144      |     2.47099   | 0.0146486   | True            |                 4521 |       0.163312    | False                  |      2.47099   | 0.0134739   | True             | False          |
| 13 |         5 | tsm            | naive_persistence |            9.41471 |               5.2119  |     -80.6388      |     5.15527   | 8.27767e-07 | True            |                 1981 |       1.05013e-10 | True                   |      2.98141   | 0.0028692   | True             | False          |
| 14 |        20 | tsm            | naive_persistence |           54.814   |              44.2759  |     -23.8011      |     4.46354   | 1.62262e-05 | True            |                 1931 |       5.40728e-11 | True                   |      3.40942   | 0.000651014 | True             | False          |
| 15 |        30 | tsm            | naive_persistence |           89.8564  |              79.9881  |     -12.3372      |     2.15022   | 0.0332192   | True            |                 2332 |       8.43379e-09 | True                   |      0.724869  | 0.468532    | False            | False          |


## 5. Robustness Analysis

### 5.1 Noise Injection Test

To assess robustness, we inject noise into forecast paths at levels of 5%, 10%, 20%, 
and 30% of the forecast standard deviation.

|    |   horizon |      mse |    rmse |      mae |     mape |   noise_level | model   |
|---:|----------:|---------:|--------:|---------:|---------:|--------------:|:--------|
|  0 |         1 |  1.2769  | 1.13    | 0.827081 |  1.46522 |          0.05 | tsm     |
|  1 |         5 |  9.42187 | 3.06951 | 2.41641  |  4.18682 |          0.05 | tsm     |
|  2 |        20 | 54.8037  | 7.40295 | 5.46965  |  9.81691 |          0.05 | tsm     |
|  3 |        30 | 89.7521  | 9.47376 | 7.66684  | 14.141   |          0.05 | tsm     |
|  4 |         1 |  1.26838 | 1.12622 | 0.82609  |  1.46374 |          0.1  | tsm     |
|  5 |         5 |  9.43257 | 3.07125 | 2.41601  |  4.18649 |          0.1  | tsm     |
|  6 |        20 | 54.7956  | 7.40241 | 5.47159  |  9.82005 |          0.1  | tsm     |
|  7 |        30 | 89.6509  | 9.46842 | 7.658    | 14.125   |          0.1  | tsm     |
|  8 |         1 |  1.26118 | 1.12302 | 0.82795  |  1.46735 |          0.2  | tsm     |
|  9 |         5 |  9.46456 | 3.07645 | 2.4157   |  4.18674 |          0.2  | tsm     |
| 10 |        20 | 54.7865  | 7.40179 | 5.47547  |  9.82632 |          0.2  | tsm     |
| 11 |        30 | 89.4577  | 9.45821 | 7.64031  | 14.0931  |          0.2  | tsm     |
| 12 |         1 |  1.26707 | 1.12564 | 0.833099 |  1.47647 |          0.3  | tsm     |
| 13 |         5 |  9.51069 | 3.08394 | 2.41718  |  4.19046 |          0.3  | tsm     |
| 14 |        20 | 54.7867  | 7.4018  | 5.48014  |  9.83401 |          0.3  | tsm     |
| 15 |        30 | 89.2768  | 9.44864 | 7.62263  | 14.0611  |          0.3  | tsm     |

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

- **Price accuracy**: linear_ridge has the lowest average MSE (23.644).
- **TSM vs naive**: TSM MSE is 1.2x the naive baseline on average.
- **Directional accuracy**: linear_ridge has the highest average trend accuracy (0.587).
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
Sequence Length: 120
TSM Type: autoformer
LLM Model: gpt-5.2
```

### A.2 Environment

- Python: 3.10+
- Key packages: torch, pytorch-forecasting, openai, pandas, numpy

### A.3 Compute Resources

- Device preference: auto
- Data loader workers: 0
- Pin memory: True

### A.4 Data Availability

Raw data sources:
- ICAP Allowance Price Explorer
- EEX Emissions Auctions
- ECB Exchange Rates
- EIA Energy Prices


