# Continuous Uplift Model Analysis

- Best variant by mean W1-W4 path MSE: `uplift_logistic_helpful_strict`
- Mean filtered path MSE: `52.7042`
- Mean uplift vs improved base: `0.5372%`
- Mean uplift vs always-on wide8: `0.0545%`

Interpretation should focus on whether the best continuous uplift model can beat the improved base out of sample. If it cannot, the result still narrows the failure mode: the current residual signal is too weak or too poorly represented for case-level scoring on this sample size.
