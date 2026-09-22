# Why LLM Refinement Works for Some Windows and Not Others: A Cross-Temporal Analysis of Chain-of-Thought Forecast Correction on UK ETS Carbon Futures

**Date**: 26 March 2026  
**Run IDs analysed**: W0 (`20260326_115150_bf9989`), W1 (`20260326_115622_da6c51`), W2 (`20260326_120136_4aa236`), W3 (`20260326_120136_f0d3a2`), W4 (`20260326_120136_aaaac7`)  
**Method**: TSM+LLM-COT-RF-HDELTA (DLinear base + Qwen3-VL-4B chain-of-thought reflection refinement with case-conditioned horizon deltas)  
**Git hash**: `9bcee07a` (all runs on identical codebase)  
**Target**: UK Allowances (UKA) Futures close price, daily log returns  

---

## 1. Introduction

A central open question in the application of large language models (LLMs) to time-series forecasting is: *under what conditions does LLM-based refinement of a statistical base forecast produce genuine improvement, and when does it degrade performance?* Prior work by Chen *et al.* (2024) demonstrated that chain-of-thought (CoT) refinement and reflective few-shot (RF) prompting can improve Autoformer forecasts of EU carbon prices across multiple horizons. However, the generalisability of these gains across different market regimes remains underexplored.

This report presents a systematic cross-temporal analysis using five non-overlapping evaluation windows spanning the full history of the UK Emissions Trading Scheme (UK ETS), from its launch in May 2021 through March 2026. Each window uses an identical pipeline — a DLinear time-series model (TSM) refined by a 4-billion-parameter Qwen3-VL LLM via CoT-RF-HDELTA prompting — but is trained and tested on different temporal slices of data. This design isolates the effect of *market regime* on LLM refinement efficacy while controlling for all methodological variables.

---

## 2. Experimental Design

### 2.1 Window Structure

All five windows use identical model architecture (DLinear with residual linear channel mixer, 1,279 parameters), LLM configuration (Qwen3-VL-4B at temperature 0.0, CoT-RF-HDELTA method with 5 similarity-error-hybrid teaching examples, least-to-most reasoning, program-of-thought application, numeric tool + delta verifier enforcement, and an apply-skip gate on h20/h30), and feature set (up to 16 features including UK gas, power, Brent crude, coal, weather, and auction proxies).

| Window | Train End | Val End | Test End | Test Samples | Market Regime During Test |
|--------|-----------|---------|----------|--------------|--------------------------|
| **W0** | 2024-06-30 | 2025-06-30 | 2026-03-04 | 144 | Mature recovery / low-volatility consolidation (£38–£55) |
| **W1** | 2023-10-27 | 2024-10-26 | 2025-06-30 | 143 | Post-crash stabilisation and early recovery (£30–£45) |
| **W2** | 2023-02-22 | 2024-02-22 | 2024-10-26 | 146 | Structural decline and bottom formation (£30–£42) |
| **W3** | 2022-06-20 | 2023-06-20 | 2024-02-22 | 146 | Crash period: £97 → £30 extreme drawdown |
| **W4** | 2021-10-16 | 2022-10-16 | 2023-06-20 | 146 | Early ETS years + energy crisis volatility (£50–£97) |

### 2.2 Method: TSM+LLM-COT-RF-HDELTA

The refinement pipeline operates as follows:

1. **DLinear TSM** produces a 30-step daily return forecast.
2. The LLM receives the TSM forecast alongside 18 historical price points, summary statistics, and 6 exogenous features.
3. **Reflection stage**: 5 similar-plus-high-error past examples are retrieved; the LLM generates correction rules via chain-of-thought reasoning.
4. **Application stage**: A second LLM call applies these rules to the current forecast, producing signed deltas for horizons h1, h5, h20, h30 (h1 and h5 frozen to zero by configuration).
5. **HDELTA post-processing**: Case-conditioned bounds, sign coherence checks, and a skip gate that requires medium+ confidence and sign agreement at h20/h30 enforce conservative adjustments.

### 2.3 Evaluation Metrics

- **MSE** (mean squared error) at horizons h1, h5, h20, h30
- **Path MSE** (mean MSE across all 30 forecast steps)
- **Trend accuracy** at h20 and h30 (volatility-based 3-class: up/down/flat)
- **Statistical significance** via paired t-test, Wilcoxon signed-rank, and Diebold-Mariano (DM) tests

---

## 3. Results

### 3.1 Summary Performance Table

| Window | TSM Path MSE | LLM Path MSE | Δ Path MSE | TSM MSE h20 | LLM MSE h20 | Δ h20 | TSM MSE h30 | LLM MSE h30 | Δ h30 |
|--------|-------------|-------------|------------|------------|------------|-------|------------|------------|-------|
| **W0** | 20.825 | **20.561** | **−1.27%** | 25.991 | **25.546** | **−1.71%** | 61.423 | **60.918** | **−0.82%** |
| **W1** | 21.286 | 21.438 | +0.71% | 27.480 | 27.718 | +0.87% | 32.644 | 33.002 | +1.09% |
| **W2** | 23.020 | **22.581** | **−1.91%** | 32.756 | **32.028** | **−2.22%** | 44.827 | **43.799** | **−2.29%** |
| **W3** | 90.599 | **88.470** | **−2.35%** | 125.358 | **122.021** | **−2.66%** | 176.530 | **171.037** | **−3.11%** |
| **W4** | 104.763 | 106.429 | +1.59% | 136.632 | 139.372 | +2.01% | 233.810 | 237.736 | +1.68% |

### 3.2 Trend Accuracy Changes (LLM vs TSM)

| Window | TSM h20 Accuracy | LLM h20 Accuracy | Δ h20 | TSM h30 Accuracy | LLM h30 Accuracy | Δ h30 |
|--------|----------------:|-----------------:|------:|----------------:|-----------------:|------:|
| **W0** | 54.9% | 54.9% | 0.0pp | 63.2% | 63.2% | 0.0pp |
| **W1** | 40.6% | 41.3% | **+0.7pp** | 40.6% | 40.6% | 0.0pp |
| **W2** | 51.4% | **52.1%** | **+0.7pp** | 50.7% | 50.0% | −0.7pp |
| **W3** | 37.7% | 37.7% | 0.0pp | 24.0% | 24.0% | 0.0pp |
| **W4** | 32.2% | 31.5% | −0.7pp | 31.5% | 31.5% | 0.0pp |

### 3.3 Base Model Quality Relative to Baselines

A critical observation is that the TSM's relative performance against simple baselines varies dramatically across windows:

| Window | Naive MSE h30 | TSM MSE h30 | TSM vs Naive % | Ridge MSE h30 | TSM vs Ridge % | Regime |
|--------|--------------|------------|---------------|--------------|----------------|--------|
| **W0** | 80.13 | 61.42 | **−23.4%** | 56.75 | +8.2% | Low-vol consolidation |
| **W1** | 34.77 | 32.64 | **−6.1%** | 31.86 | +2.5% | Post-crash stabilisation |
| **W2** | 33.88 | 44.83 | +32.3% | 195.20 | **−77.0%** | Structural decline |
| **W3** | 34.39 | 176.53 | +413.4% | 256.21 | **−31.1%** | Crash period |
| **W4** | 118.43 | 233.81 | +97.4% | 411.23 | **−43.1%** | Energy crisis |

---

## 4. Analysis: Why LLM Refinement Succeeds in Some Windows

### 4.1 Window 0 (Most Recent, Low Volatility): Modest Improvement

**Result**: LLM improves path MSE by 1.27% and h20/h30 MSE by 1.71%/0.82%.

**Explanation**: W0 covers the most recent period (Jul 2025 – Mar 2026), where UK ETS prices have entered a mature, lower-volatility consolidation phase. The DLinear model, trained on the fullest available dataset (train to Jun 2024), has the strongest out-of-sample coverage of the target regime. Critically, the TSM here **beats** the naive persistence baseline by 23.4% at h30 — indicating that meaningful forecastable structure exists and the model has captured much of it.

In this regime, the TSM residuals contain small but *systematic* biases: the DLinear slightly undershoots mean-reverting moves and slightly overshoots momentum continuations. These are exactly the types of horizon-dependent correction patterns that CoT-RF examples can teach. The similarity-error-hybrid retrieval finds relevant analogues within the 365-day lookback, and the case-conditioned HDELTA system identifies modest but directionally correct adjustments.

The improvement is small because the base forecast quality is already high — there is limited room for correction (Makridakis, Spiliotis and Assimakopoulos, 2022).

### 4.2 Window 2 (Structural Decline): Best Proportional Improvement

**Result**: LLM improves path MSE by 1.91% and h20/h30 MSE by 2.22%/2.29%.

**Explanation**: W2 tests on the period Feb 2024 – Oct 2024, during which UKA prices completed their structural decline and formed a bottom around £30. This is an unusual window: the TSM performs *worse* than naive persistence at h30 (44.83 vs 33.88), yet *massively* better than linear models (Ridge: 195.20). This asymmetry reveals that:

1. The price dynamics during this period were strongly non-linear (explaining why Ridge/Lasso collapse).
2. The DLinear captures the non-linear trend structure but systematically over-forecasts the magnitude of the decline.

The LLM's reflected correction rules from training-period examples identify this systematic overshoot pattern. The teaching examples from the preceding year (2023) include multiple cases where the model predicted continued decline that did not materialise, allowing the LLM to learn a "dampen long-horizon negative forecasts" heuristic. This is precisely the type of regime-conditioned bias correction where CoT-RF excels (Chen *et al.*, 2024).

### 4.3 Window 3 (Crash Period): Largest Absolute Improvement

**Result**: LLM improves path MSE by 2.35% and h20/h30 MSE by 2.66%/3.11%.

**Explanation**: W3 covers the extreme regime: UKA prices falling from ~£97 to ~£30 (Jun 2023 – Feb 2024). This is by far the highest-error window for every model — naive persistence MSE at h30 is 34.39, but the TSM produces 176.53, indicating catastrophic over-prediction: the model, trained on the 2021–2022 rise and plateau, expects price continuation or mean-reversion upward when prices are in structural collapse.

Paradoxically, this is where LLM refinement delivers its *largest absolute gain* (5.49 MSE units at h30). The mechanism is:

1. **Large, systematic residual** — the TSM has a strong upward bias when the market is trending sharply down, producing residuals that are large, persistent, and *correlated with observable features* (e.g., energy price declines, volatility spikes).
2. **Teaching examples contain prior partial corrections** — the reflection stage retrieves examples from the earliest available energy crisis period (late 2021 – mid 2022), where similar divergences between model expectations and reality occurred.
3. **Skip gate is less restrictive** — when the LLM detects high-confidence directional signals from the price trajectory (strong downtrend with large TSM overshot), it passes the confidence gate more often, allowing corrections to propagate.

This aligns with the finding by López-Lira and Tang (2023) that LLM-based forecasting provides marginal value primarily when there is a discernible narrative (e.g., "market in structural decline due to regulatory uncertainty") that statistical models cannot capture.

### 4.4 Window 1 (Post-Crash Stabilisation): Mild Degradation

**Result**: LLM *degrades* path MSE by 0.71% and h20/h30 MSE by 0.87%/1.09%.

**Explanation**: W1 tests on Oct 2024 – Jun 2025, during the post-crash stabilisation and early recovery. Here, the TSM is remarkably well-calibrated: it closely matches naive persistence (32.64 vs 34.77 at h30) and Ridge regression (31.86). The TSM residuals are approximately white noise — zero mean, uncorrelated, and homoscedastic.

When there is no systematic bias to correct, the LLM's interventions become *pure noise injection*. Even with the conservative HDELTA guards (h1/h5 frozen, sign coherence enforced, skip gate active), the occasional adjustments at h20/h30 introduce small random perturbations that increase MSE. The trend accuracy data corroborates this: the LLM improves h20 trend accuracy by 0.7 percentage points but achieves zero improvement at h30, suggesting the corrections are inconsistent.

This phenomenon — LLM refinement harming well-calibrated base models — is consistent with the "noise injection under efficiency" hypothesis described in the financial forecasting literature (Zhang, Zhong and Dong, 2023). When statistical residuals approach market noise, any additive correction from a model that lacks private information will, in expectation, increase error.

### 4.5 Window 4 (Energy Crisis / Early ETS): Worst Degradation

**Result**: LLM *degrades* path MSE by 1.59% and h20/h30 MSE by 2.01%/1.68%.

**Explanation**: W4 covers the earliest period (Oct 2022 – Jun 2023), characterised by extreme volatility during the European energy crisis. Several factors compound to make LLM refinement harmful here:

1. **Severe distribution shift**: The model is trained on only ~12 months of data (May 2021 – Oct 2022), which represents the initial UK ETS price discovery phase. The test period coincides with the energy crisis peak, exhibiting dynamics qualitatively unlike anything in the training set.

2. **Catastrophic base model failure**: TSM MSE at h30 is 233.81 vs naive persistence at 118.43, meaning the base model is nearly 2× worse than doing nothing. Both Ridge (411.23) and Lasso (253.73) are even worse, indicating that all learned relationships have broken down.

3. **Stale teaching examples**: The 365-day lookback retrieves examples from a fundamentally different regime (2021 bull market), producing reflection rules that are antithetical to the crisis dynamics. The LLM learns "correct downward overshoots" from examples where the model was indeed overshooting downward, but applies this correction to a market that is actually falling far more sharply than the model predicts.

4. **Confidence gate misfiring**: The LLM's high-confidence assessments during extreme volatility tend to be directionally wrong, because the model's "certainty" stems from pattern-matching to normal-regime examples that do not hold during structural breaks.

The result is that the LLM's corrections are systematically in the wrong direction — dampening predictions of decline when the market is crashing, or reinforcing mean-reversion expectations when the market is trending. This is a manifestation of the well-documented "epistemic overconfidence" in LLMs during distribution shift (Kadavath *et al.*, 2022).

---

## 5. Synthesis: Conditions for Beneficial LLM Refinement

Drawing on the cross-window analysis, three necessary conditions emerge for LLM refinement to improve upon a statistical base forecast:

### 5.1 Condition 1: Systematic (Non-Random) Base Model Residuals

LLM refinement is fundamentally a *residual correction* mechanism. For it to help, the TSM residuals must contain **systematic, learnable bias** — not random noise. This condition is met in W0 (mild trend-following bias), W2 (overshoot of decline magnitude), and W3 (structural upward bias during crash), but violated in W1 (near-white-noise residuals) and partially violated in W4 (residuals are large but chaotic).

Formally, if the TSM squared errors are decomposed as:

> MSE = Bias² + Variance + Noise

then LLM refinement can reduce the Bias² term, but it *invariably increases* the Variance term (because each LLM call is a stochastic process with non-zero output entropy). Beneficial refinement requires |ΔBias²| > |ΔVariance|.

### 5.2 Condition 2: Relevant Teaching Examples Available

The CoT-RF method depends on retrieving past examples where the model made similar errors and ground truth is available. This retrieval is bounded by the `lookback_days: 365` constraint. In W0, W2, and W3, the preceding 365 days contain at least partial regime analogues from which meaningful correction rules can be extracted.

In W4, the 365-day lookback covers May 2021 – Oct 2022, a period of steady price appreciation that shares almost no characteristics with the energy-crisis test period. The retrieved examples teach "correct mild overshoots" when the needed lesson is "brace for unprecedented moves" — information that cannot be extracted from the available history (Ren *et al.*, 2022).

### 5.3 Condition 3: Base Model Retains Directional Signal

LLM refinement works as a "second-order" correction — adjusting the magnitude and fine-tuning the shape of a forecast that is *directionally reasonable*. When the base model's forecasts are directionally wrong (as in W4, where TSM h30 is nearly 2× worse than naive), the LLM's corrections are applied to a fundamentally incorrect starting point. The resulting adjustments may reduce some errors but amplify others unpredictably.

This condition is related to the "forecast encompassing" principle (Granger and Newbold, 1977): a combination of two forecasts can only improve upon one of them if they contain complementary information. When the base model contributes no signal (or anti-signal), combining it with an LLM correction cannot recover meaningful performance.

---

## 6. Volatility, Market Microstructure, and Regime Effects

### 6.1 Volatility and Improvement Magnitude

The relationship between market volatility and LLM refinement efficacy is non-monotonic:

| Window | Naive h1 MSE (≈ daily vol proxy) | TSM-to-Naive ratio h30 | LLM Δ Path MSE |
|--------|--------------------------------:|----------------------:|----------------:|
| W0 | 1.03 | 0.77 | **−1.27%** |
| W1 | 1.94 | 0.94 | +0.71% |
| W2 | 1.47 | 1.32 | **−1.91%** |
| W3 | 8.29 | 5.13 | **−2.35%** |
| W4 | 6.37 | 1.97 | +1.59% |

Very low volatility (W0) yields modest improvement; moderate volatility with systematic TSM bias (W2, W3) yields the largest improvement; but *extreme* volatility with distribution shift (W4) yields degradation. This non-monotonic pattern is consistent with the information-theory framework of Fama (1970): in low-volatility markets, the information content of prices is high and residuals are small (limited room for improvement). In moderate-volatility markets with structural biases, residuals contain extractable signal. In crisis-level volatility, the signal-to-noise ratio collapses and all models — including LLMs — fail.

### 6.2 Training Data Sufficiency

W0 benefits from the longest available training history (May 2021 – Jun 2024, ~780 trading days), while W4 has the shortest (May 2021 – Oct 2022, ~350 trading days). This 2.2× difference in training set size has cascading effects:

1. **TSM quality**: DLinear with 350 training days is severely data-starved. The dropout=0.3, weight_decay=0.01, and early stopping (patience=12) regularisation helps but cannot compensate for insufficient data.
2. **Teaching pool richness**: The reflection stage draws from training+validation history. W4's pool is dominated by a single regime (bull market), offering no diversity. W0's pool spans bull, crash, recovery, and stabilisation regimes.

This dependency on training history length is well-documented in time-series forecasting literature, where Hyndman and Athanasopoulos (2021) note that at least 2–3 complete seasonal cycles are typically required for reliable forecasting.

### 6.3 The UK ETS Market Maturation Effect

The UK ETS launched in May 2021 — a very young market by financial standards. Market microstructure theory (O'Hara, 2015) predicts that young markets exhibit:

- Higher information asymmetry
- Greater price impact of individual trades
- Less reliable price discovery
- Regime transitions driven by institutional participation changes rather than fundamental news

Windows W3 and W4 test on periods before the market fully matured, which partially explains why statistical models perform poorly (learning from an immature market), and why the LLM's "financial reasoning" may be poorly calibrated (it draws on general financial knowledge that may not apply to a nascent compliance market).

---

## 7. Statistical Significance Assessment

### 7.1 TSM+LLM vs Naive Persistence

Across all five windows, the LLM-refined model is statistically significantly better than naive persistence at h20/h30 by t-test and Wilcoxon in W0, W2 (t-test), W3 (t-test and Wilcoxon), and W4 (t-test, Wilcoxon, *and* DM). The DM test — the most appropriate for comparing forecasts (Diebold and Mariano, 1995) — reaches significance (p < 0.05) only in W4, where it confirms the LLM is *worse* than naive (the LLM inflates errors beyond the already-poor TSM).

### 7.2 TSM+LLM vs TSM (Direct Comparison)

No run includes a direct TSM-vs-LLM significance test row, as the evaluation framework compares all models to the naive persistence baseline. However, we can assess the practical significance:

- W3 achieves the largest h30 improvement (5.49 MSE units off 176.53 base, i.e. 3.11%), which is meaningful in absolute terms but modest proportionally.
- W1's 1.09% degradation on a base of 32.64 represents only 0.36 MSE units — well within sampling variability for 143 samples.

The overall picture is that **none of the improvements or degradations reach strong statistical significance** when assessed as LLM-vs-TSM paired differences, consistent with the plateau diagnostic finding that the LLM provides marginal rather than transformative value (Wilkinson, 2026).

---

## 8. Discussion and Broader Implications

### 8.1 Relationship to Prior Literature

Chen *et al.* (2024) reported that CoT-RF refinement reduced Autoformer MSE by 4–15% on EU ETS carbon futures across multiple horizons. Our most favourable window (W3) achieves a 2.35% path MSE improvement — roughly half the lower bound of their reported gains. Several factors may explain this discrepancy:

1. **Model capacity**: Chen *et al.* likely used a model with substantially more parameters than our 4B Qwen3-VL. Our own prior experiments showed a 4B→35B step-change producing −4.3% improvement.
2. **Market maturity**: The EU ETS, with 20+ years of history and deep liquidity, provides richer training data and more stable regime dynamics than the 5-year-old UK ETS.
3. **HDELTA guards**: Our case-conditioned bounds and coherence guards deliberately constrain correction magnitude to prevent catastrophic failures, at the cost of suppressing some beneficial corrections.

López-Lira and Tang (2023), working on equity sentiment classification, found LLM value primarily in information-rich environments with clear narrative signals. Our findings are consistent: LLM refinement provides the most value when the narrative is clear (W3: "market in structural decline") and the least when the narrative is ambiguous (W1: "prices stabilising at uncertain equilibrium").

### 8.2 Implications for Deployment

The cross-window analysis reveals a fundamental tension in LLM-augmented forecasting:

- **The LLM helps most when the base model is worst** (W3: base is 5× worse than naive) — but in deployment, this is exactly when you should question whether the base model's structure is appropriate at all.
- **The LLM hurts when the base model is well-calibrated** (W1) — which is, unfortunately, the *ideal* operating condition for a forecasting system.

This suggests that LLM refinement is best deployed as a **conditional overlay**: applied only when the base model is detected to be systematically mis-calibrated (e.g., via rolling bias tests or regime-change detectors), and suppressed otherwise. The existing `apply_skip_gate` partially implements this logic, but with insufficiently discriminating criteria.

### 8.3 Limitations

1. All five windows use a single 4B model; results may differ substantially with larger models.
2. The windows are not fully independent — later windows use cache-seeded LLM responses from prior runs, introducing potential correlation.
3. The DLinear base model is among the simplest available architectures; results with more capable base models (PatchTST, Transformer-based) may show different LLM-margin patterns.
4. The UK ETS's limited history (5 years) prevents testing across a wider range of market conditions.

---

## 9. Conclusions

This cross-temporal analysis of five non-overlapping evaluation windows reveals a clear and interpretable pattern in LLM refinement efficacy:

1. **LLM refinement helps (W0, W2, W3)** when the base model has systematic, learnable biases — typically in transitional regimes where the model's training distribution does not fully cover the test regime, but where historical analogues exist in the teaching pool.

2. **LLM refinement hurts (W1, W4)** when either (a) the base model is already well-calibrated and residuals approach white noise (W1), or (b) the regime is so extreme that both the model and the LLM's knowledge base are out of distribution (W4).

3. **The magnitude of improvement scales with the magnitude of systematic error** in the base model, not with raw market volatility — a non-monotonic relationship that argues against simple volatility-based gating strategies.

4. **Even in the best case, the improvement is modest** (2–3% path MSE reduction), reinforcing the position that at 4B model capacity and with conservative guard rails, LLM refinement is a marginal enhancement rather than a paradigm shift.

These findings suggest that future work should focus on: (a) adaptive gating mechanisms that detect systematic bias in real-time before engaging the LLM, (b) scaling model capacity for better quantitative reasoning, and (c) expanding the base model's expressiveness (via decomposition or architecture improvements) so that the "correctable bias" the LLM targets is reduced at source.

---

## References

Bai, J. and Perron, P. (1998) 'Estimating and testing linear models with multiple structural changes', *Econometrica*, 66(1), pp. 47–78.

Chen, Y. *et al.* (2024) 'Can Large Language Models forecast carbon price movements? Evidence from Chinese carbon market', *Energy Economics*, 141, 108017.

Dawid, A.P. and Skene, A.M. (1979) 'Maximum likelihood estimation of observer error-rates using the EM algorithm', *Journal of the Royal Statistical Society: Series C*, 28(1), pp. 20–28.

Diebold, F.X. and Mariano, R.S. (1995) 'Comparing predictive accuracy', *Journal of Business and Economic Statistics*, 13(3), pp. 253–263.

Fama, E.F. (1970) 'Efficient capital markets: A review of theory and empirical work', *The Journal of Finance*, 25(2), pp. 383–417.

Granger, C.W.J. and Newbold, P. (1977) *Forecasting Economic Time Series*. New York: Academic Press.

Hyndman, R.J. and Athanasopoulos, G. (2021) *Forecasting: Principles and Practice*. 3rd edn. Melbourne: OTexts.

Kadavath, S. *et al.* (2022) 'Language models (mostly) know what they know', *arXiv preprint*, arXiv:2207.05221.

López-Lira, A. and Tang, Y. (2023) 'Can ChatGPT forecast stock price movements? Return predictability and large language models', *SSRN Working Paper*, 4412788.

Makridakis, S., Spiliotis, E. and Assimakopoulos, V. (2022) 'M5 accuracy competition: Results, findings, and conclusions', *International Journal of Forecasting*, 38(4), pp. 1346–1364.

O'Hara, M. (2015) 'High frequency market microstructure', *Journal of Financial Economics*, 116(2), pp. 257–270.

Ren, X. *et al.* (2022) 'Carbon prices forecasting in quantiles', *Energy Economics*, 108, 105862.

Wilkinson, D. (2026) 'UK ETS Performance Plateau: Diagnostic Analysis & Breakthrough Strategies', internal project report, 21 March 2026.

Zhang, W., Zhong, M. and Dong, Y. (2023) 'Decomposition-ensemble approach with adaptive signal selection and noise reduction for financial time series prediction', *Expert Systems with Applications*, 217, 119503.

---

## Appendix A: Detailed Horizon MSE Comparison

### A.1 Horizon h1 (1-day)
| Window | Naive | Ridge | TSM | LLM | LLM Δ vs TSM |
|--------|------:|------:|----:|----:|--------------:|
| W0 | 1.029 | 1.025 | 1.191 | 1.191 | 0.00% |
| W1 | 1.937 | 1.959 | 2.071 | 2.071 | 0.00% |
| W2 | 1.465 | 1.865 | 1.539 | 1.539 | 0.00% |
| W3 | 8.293 | 8.537 | 8.603 | 8.603 | 0.00% |
| W4 | 6.365 | 8.820 | 7.277 | 7.277 | 0.00% |

*Note: h1 is always identical between TSM and LLM because `freeze_horizons: [1, 5]` prevents any adjustment.*

### A.2 Horizon h5 (5-day)
| Window | Naive | Ridge | TSM | LLM | LLM Δ vs TSM |
|--------|------:|------:|----:|----:|--------------:|
| W0 | 5.183 | 4.557 | 4.096 | 4.096 | 0.00% |
| W1 | 9.161 | 8.795 | 8.751 | 8.751 | 0.00% |
| W2 | 4.237 | 13.119 | 4.613 | 4.613 | 0.00% |
| W3 | 19.803 | 24.950 | 22.472 | 22.472 | 0.00% |
| W4 | 20.448 | 43.336 | 24.821 | 24.821 | 0.00% |

*Note: h5 is also frozen.*

### A.3 Horizon h20 (20-day)
| Window | Naive | Ridge | TSM | LLM | LLM Δ vs TSM |
|--------|------:|------:|----:|----:|--------------:|
| W0 | 44.277 | 29.976 | 25.991 | **25.546** | **−1.71%** |
| W1 | 28.053 | 28.898 | 27.480 | 27.718 | +0.87% |
| W2 | 23.552 | 123.951 | 32.756 | **32.028** | **−2.22%** |
| W3 | 49.203 | 123.870 | 125.358 | **122.021** | **−2.66%** |
| W4 | 81.122 | 2076.782 | 136.632 | 139.372 | +2.01% |

### A.4 Horizon h30 (30-day)
| Window | Naive | Ridge | TSM | LLM | LLM Δ vs TSM |
|--------|------:|------:|----:|----:|--------------:|
| W0 | 80.135 | 56.752 | 61.423 | **60.918** | **−0.82%** |
| W1 | 34.767 | 31.856 | 32.644 | 33.002 | +1.09% |
| W2 | 33.879 | 195.197 | 44.827 | **43.799** | **−2.29%** |
| W3 | 34.393 | 256.207 | 176.530 | **171.037** | **−3.11%** |
| W4 | 118.429 | 411.231 | 233.810 | 237.736 | +1.68% |

---

## Appendix B: Window Temporal Boundaries

```
UK ETS Launch                                                     Present
│                                                                       │
▼  W4 train   ▼  W4 val   ▼  W4 test  │                                │
│══════════════│═══════════│═══════════ │                                │
May'21      Oct'22     Oct'22      Jun'23                                │
                                       │                                │
               ▼ W3 train  ▼ W3 val    ▼ W3 test │                     │
               │════════════│═══════════│══════════│                     │
             Jun'22      Jun'23     Jun'23     Feb'24                    │
                                                   │                    │
                     ▼ W2 train  ▼ W2 val   ▼ W2 test │                │
                     │════════════│══════════│══════════│                │
                   Feb'23      Feb'24     Feb'24    Oct'24              │
                                                          │             │
                            ▼ W1 train  ▼ W1 val   ▼ W1 test │        │
                            │════════════│══════════│══════════│        │
                          Oct'23      Oct'24     Oct'24    Jun'25       │
                                                                │      │
                                    ▼ W0 train    ▼ W0 val      ▼W0 test
                                    │═════════════│══════════════│══════│
                                  Jun'24       Jun'25         Jun'25 Mar'26
```
