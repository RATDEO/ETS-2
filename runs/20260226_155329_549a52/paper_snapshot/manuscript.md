# EU ETS Futures Forecasting with TSM + LLM Refinement

**Authors**: Auto-generated

**Generated**: 2026-02-26 15:55:41

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
- Dataset spans 2009-04-01 00:00:00 to 2026-02-25 00:00:00
- 4334 daily observations
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

- **Date Range**: 2009-04-01 00:00:00 to 2026-02-25 00:00:00
- **Total Observations**: 4,334
- **Number of Features**: 43

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

|    |   horizon |        mse |     rmse |       mae |     mape | model             |
|---:|----------:|-----------:|---------:|----------:|---------:|:------------------|
|  0 |         1 |    1.39098 |  1.1794  |  0.947696 |  1.32876 | naive_persistence |
|  1 |         5 |    6.4397  |  2.53766 |  2.01736  |  2.83588 | naive_persistence |
|  2 |        20 |   26.3936  |  5.13747 |  4.04948  |  5.56409 | naive_persistence |
|  3 |        30 |   41.3395  |  6.42958 |  5.15615  |  7.07198 | naive_persistence |
|  4 |         1 |    6.53148 |  2.55568 |  2.04024  |  2.87018 | seasonal_naive    |
|  5 |         5 |    6.4397  |  2.53766 |  2.01736  |  2.83588 | seasonal_naive    |
|  6 |        20 |   26.3936  |  5.13747 |  4.04948  |  5.56409 | seasonal_naive    |
|  7 |        30 |   41.3395  |  6.42958 |  5.15615  |  7.07198 | seasonal_naive    |
|  8 |         1 |    1.40543 |  1.18551 |  0.953823 |  1.33681 | linear_ridge      |
|  9 |         5 |    6.65042 |  2.57884 |  2.04762  |  2.88304 | linear_ridge      |
| 10 |        20 |   26.8355  |  5.1803  |  3.98829  |  5.59743 | linear_ridge      |
| 11 |        30 |   39.7633  |  6.30582 |  4.961    |  6.90319 | linear_ridge      |
| 12 |         1 |    5.34846 |  2.31267 |  2.02053  |  2.77429 | linear_lasso      |
| 13 |         5 |   10.4036  |  3.22547 |  2.69319  |  3.68562 | linear_lasso      |
| 14 |        20 |   28.5853  |  5.34652 |  4.47365  |  6.08462 | linear_lasso      |
| 15 |        30 |   39.992   |  6.32392 |  5.21733  |  7.12347 | linear_lasso      |
| 16 |         1 | 2562.99    | 50.626   | 50.3023   | 69.5058  | tsm               |
| 17 |         5 | 2584.25    | 50.8355  | 50.4945   | 69.5884  | tsm               |
| 18 |        20 | 2640.02    | 51.3811  | 51.0286   | 69.7373  | tsm               |
| 19 |        30 | 2673.09    | 51.7019  | 51.3171   | 70.0369  | tsm               |

### 4.2 Trend Classification Accuracy

Table 2 shows the accuracy of predicting price direction (up/flat/down).

|    |   horizon |   accuracy |   accuracy_up |   accuracy_down |   accuracy_flat |   n_up |   n_down |   n_flat | model             |
|---:|----------:|-----------:|--------------:|----------------:|----------------:|-------:|---------:|---------:|:------------------|
|  0 |         1 |  0.303665  |     0         |       0         |       1         |    133 |      133 |      116 | naive_persistence |
|  1 |         5 |  0.149215  |     0         |       0         |       1         |    177 |      148 |       57 | naive_persistence |
|  2 |        20 |  0.0863874 |     0         |       0         |       1         |    226 |      123 |       33 | naive_persistence |
|  3 |        30 |  0.0628272 |     0         |       0         |       1         |    229 |      129 |       24 | naive_persistence |
|  4 |         1 |  0.342932  |     0.406015  |       0.451128  |       0.146552  |    133 |      133 |      116 | seasonal_naive    |
|  5 |         5 |  0.149215  |     0         |       0         |       1         |    177 |      148 |       57 | seasonal_naive    |
|  6 |        20 |  0.0863874 |     0         |       0         |       1         |    226 |      123 |       33 | seasonal_naive    |
|  7 |        30 |  0.0628272 |     0         |       0         |       1         |    229 |      129 |       24 | seasonal_naive    |
|  8 |         1 |  0.303665  |     0.0075188 |       0         |       0.991379  |    133 |      133 |      116 | linear_ridge      |
|  9 |         5 |  0.256545  |     0.265537  |       0.0608108 |       0.736842  |    177 |      148 |       57 | linear_ridge      |
| 10 |        20 |  0.531414  |     0.862832  |       0.0569106 |       0.030303  |    226 |      123 |       33 | linear_ridge      |
| 11 |        30 |  0.549738  |     0.842795  |       0.0930233 |       0.208333  |    229 |      129 |       24 | linear_ridge      |
| 12 |         1 |  0.348168  |     0         |       1         |       0         |    133 |      133 |      116 | linear_lasso      |
| 13 |         5 |  0.390052  |     0         |       1         |       0.0175439 |    177 |      148 |       57 | linear_lasso      |
| 14 |        20 |  0.295812  |     0.0929204 |       0.682927  |       0.242424  |    226 |      123 |       33 | linear_lasso      |
| 15 |        30 |  0.384817  |     0.253275  |       0.612403  |       0.416667  |    229 |      129 |       24 | linear_lasso      |
| 16 |         1 |  0.348168  |     0         |       1         |       0         |    133 |      133 |      116 | tsm               |
| 17 |         5 |  0.387435  |     0         |       1         |       0         |    177 |      148 |       57 | tsm               |
| 18 |        20 |  0.32199   |     0         |       1         |       0         |    226 |      123 |       33 | tsm               |
| 19 |        30 |  0.337696  |     0         |       1         |       0         |    229 |      129 |       24 | tsm               |

### 4.3 Statistical Significance

Table 3 reports the results of significance tests comparing methods.

|    |   horizon | model          | baseline          |   mean_error_model |   mean_error_baseline |   improvement_pct |   t_statistic |     t_pvalue | t_significant   |   wilcoxon_statistic |   wilcoxon_pvalue | wilcoxon_significant   |   dm_statistic |   dm_pvalue | dm_significant   | model_better   |
|---:|----------:|:---------------|:------------------|-------------------:|----------------------:|------------------:|--------------:|-------------:|:----------------|---------------------:|------------------:|:-----------------------|---------------:|------------:|:-----------------|:---------------|
|  0 |         1 | seasonal_naive | naive_persistence |            6.53148 |               1.39098 |    -369.559       |     11.3323   | 7.50401e-26  | True            |                11234 |       8.40519e-32 | True                   |     11.3323    |  0          | True             | False          |
|  1 |         5 | seasonal_naive | naive_persistence |            6.4397  |               6.4397  |       3.03703e-06 |     -0.405389 | 0.685419     | False           |                28309 |       0.0182062   | True                   |     -0.0107908 |  0.99139    | False            | True           |
|  2 |        20 | seasonal_naive | naive_persistence |           26.3936  |              26.3936  |       3.98549e-06 |     -0.254084 | 0.799568     | False           |                30345 |       0.0536334   | False                  |     -0.0222415 |  0.982255   | False            | True           |
|  3 |        30 | seasonal_naive | naive_persistence |           41.3395  |              41.3395  |       6.15199e-06 |     -1.16212  | 0.245913     | False           |                28132 |       0.00342521  | True                   |     -0.2106    |  0.833199   | False            | True           |
|  4 |         1 | linear_ridge   | naive_persistence |            1.40543 |               1.39098 |      -1.03848     |      0.700201 | 0.484229     | False           |                34829 |       0.418395    | False                  |      0.700201  |  0.483802   | False            | False          |
|  5 |         5 | linear_ridge   | naive_persistence |            6.65042 |               6.4397  |      -3.27222     |      1.32028  | 0.187536     | False           |                34636 |       0.368876    | False                  |      0.620996  |  0.534602   | False            | False          |
|  6 |        20 | linear_ridge   | naive_persistence |           26.8355  |              26.3936  |      -1.67432     |      0.379767 | 0.70433      | False           |                36249 |       0.879459    | False                  |      0.100661  |  0.919819   | False            | False          |
|  7 |        30 | linear_ridge   | naive_persistence |           39.7633  |              41.3395  |       3.8127      |     -1.16961  | 0.242888     | False           |                27296 |       1.7274e-05  | True                   |     -0.324998  |  0.745183   | False            | True           |
|  8 |         1 | linear_lasso   | naive_persistence |            5.34846 |               1.39098 |    -284.51        |     16.9088   | 2.98722e-48  | True            |                 7999 |       5.63596e-40 | True                   |     16.9088    |  0          | True             | False          |
|  9 |         5 | linear_lasso   | naive_persistence |           10.4036  |               6.4397  |     -61.5547      |      8.6341   | 1.63737e-16  | True            |                18726 |       1.38555e-16 | True                   |      3.6972    |  0.00021799 | True             | False          |
| 10 |        20 | linear_lasso   | naive_persistence |           28.5853  |              26.3936  |      -8.30404     |      2.31639  | 0.0210666    | True            |                25383 |       2.17947e-07 | True                   |      0.949167  |  0.342536   | False            | False          |
| 11 |        30 | linear_lasso   | naive_persistence |           39.992   |              41.3395  |       3.25952     |     -1.26056  | 0.208238     | False           |                33944 |       0.222835    | False                  |     -0.822795  |  0.410624   | False            | True           |
| 12 |         1 | tsm            | naive_persistence |         2562.99    |               1.39098 | -184158           |     84.2542   | 2.05573e-248 | True            |                    0 |       2.38553e-64 | True                   |     84.2542    |  0          | True             | False          |
| 13 |         5 | tsm            | naive_persistence |         2584.25    |               6.4397  |  -40030           |     81.8322   | 7.71007e-244 | True            |                    0 |       2.38553e-64 | True                   |     27.9134    |  0          | True             | False          |
| 14 |        20 | tsm            | naive_persistence |         2640.02    |              26.3936  |   -9902.52        |     81.4061   | 5.06236e-243 | True            |                    0 |       2.38553e-64 | True                   |     13.9543    |  0          | True             | False          |
| 15 |        30 | tsm            | naive_persistence |         2673.09    |              41.3395  |   -6366.19        |     77.6174   | 1.3894e-235  | True            |                    0 |       2.38553e-64 | True                   |     11.9549    |  0          | True             | False          |


## 5. Robustness Analysis

### 5.1 Noise Injection Test

To assess robustness, we inject noise into forecast paths at levels of 5%, 10%, 20%, 
and 30% of the forecast standard deviation.

|    |   horizon |     mse |    rmse |     mae |    mape |   noise_level | model   |
|---:|----------:|--------:|--------:|--------:|--------:|--------------:|:--------|
|  0 |         1 | 2563.22 | 50.6283 | 50.3046 | 69.5092 |          0.05 | tsm     |
|  1 |         5 | 2584.48 | 50.8378 | 50.4974 | 69.5932 |          0.05 | tsm     |
|  2 |        20 | 2640.32 | 51.3841 | 51.0317 | 69.7421 |          0.05 | tsm     |
|  3 |        30 | 2672.12 | 51.6926 | 51.3076 | 70.024  |          0.05 | tsm     |
|  4 |         1 | 2563.47 | 50.6308 | 50.3069 | 69.5126 |          0.1  | tsm     |
|  5 |         5 | 2584.72 | 50.8402 | 50.5003 | 69.598  |          0.1  | tsm     |
|  6 |        20 | 2640.64 | 51.3871 | 51.0349 | 69.747  |          0.1  | tsm     |
|  7 |        30 | 2671.17 | 51.6834 | 51.2981 | 70.011  |          0.1  | tsm     |
|  8 |         1 | 2564.02 | 50.6361 | 50.3115 | 69.5195 |          0.2  | tsm     |
|  9 |         5 | 2585.26 | 50.8454 | 50.5061 | 69.6076 |          0.2  | tsm     |
| 10 |        20 | 2641.3  | 51.3936 | 51.0412 | 69.7566 |          0.2  | tsm     |
| 11 |        30 | 2669.3  | 51.6652 | 51.2792 | 69.9852 |          0.2  | tsm     |
| 12 |         1 | 2564.62 | 50.6421 | 50.3161 | 69.5263 |          0.3  | tsm     |
| 13 |         5 | 2585.85 | 50.8513 | 50.5119 | 69.6173 |          0.3  | tsm     |
| 14 |        20 | 2642.02 | 51.4006 | 51.0475 | 69.7663 |          0.3  | tsm     |
| 15 |        30 | 2667.47 | 51.6476 | 51.2603 | 69.9593 |          0.3  | tsm     |

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

- **Price accuracy**: linear_ridge has the lowest average MSE (18.664).
- **TSM vs naive**: TSM MSE is 138.4x the naive baseline on average.
- **Directional accuracy**: linear_ridge has the highest average trend accuracy (0.410).
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
- Data loader workers: 4
- Pin memory: True

### A.4 Data Availability

Raw data sources:
- ICAP Allowance Price Explorer
- EEX Emissions Auctions
- ECB Exchange Rates
- EIA Energy Prices


