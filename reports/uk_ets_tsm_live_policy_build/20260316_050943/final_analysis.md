# Final Analysis

## Experimental Design
- Base forecaster: `TSM` only, because prior live-online sweeps showed the LLM edge is concentrated there.
- Shared inference stack: numeric tool, delta verifier, online realized memory, regime-specific retrieval, full 103-window live-online holdout.
- Policy levers tested: heuristic gate, split horizon banks, logistic learned gate, boosted-tree learned gate, stricter admission, and selected combinations.

## Expectations Vs Actuals
- `learned_gbdt_regime`: expected `Expected path MSE around 3.95-4.10. If the gain is mostly from the gate, this should stay close to or beat the split-bank boosted-tree run.` actual path MSE `3.729913` with gain `11.244%`.
- `learned_gbdt_split_horizon_regime`: expected `Expected path MSE around 4.08-4.16. This is the more flexible learned-gate alternative if logistic is too linear.` actual path MSE `3.923331` with gain `6.642%`.
- `learned_logistic_regime`: expected `Expected path MSE around 4.12-4.18. A learned gate should reduce false positives and preserve only useful LLM calls.` actual path MSE `4.012189` with gain `4.527%`.
- `learned_logistic_regime_high_utility`: expected `Expected path MSE around 4.00-4.12. If stricter promotion helps on its own, it should improve on the plain learned logistic gate.` actual path MSE `4.050407` with gain `3.618%`.
- `learned_logistic_split_horizon_regime`: expected `Expected path MSE around 4.08-4.16. This is the main candidate to beat the current TSM live benchmark.` actual path MSE `4.117427` with gain `2.023%`.
- `baseline_regime_specific`: expected `Expected path MSE around 4.16-4.20. This is the live benchmark to beat.` actual path MSE `4.174480` with gain `0.665%`.
- `learned_logistic_split_horizon_regime_high_utility`: expected `Expected path MSE around 4.08-4.17. This may improve memory quality if noisy positive cases are currently diluting the bank.` actual path MSE `4.204232` with gain `-0.043%`.
- `split_horizon_banks_regime`: expected `Expected path MSE around 4.13-4.18. Separate horizon banks should reduce contradictory long-horizon memories.` actual path MSE `4.214644` with gain `-0.290%`.

## Interpretation
- The main bottleneck was the live apply policy, not prompt structure. Once the gate became learned rather than heuristic, the live edge expanded sharply.
- The best-performing memory policy remained simple regime-specific retrieval. Attempts to over-structure the memory bank with split horizons or stricter promotion rules mostly reduced useful `h20` recall.
- The boosted-tree gate likely works because it captures nonlinear interactions between base forecast shape, regime similarity, and memory helpfulness that the heuristic and logistic gates only approximate.
- The `select_size=0` field in the saved gate-selection JSON comes from the final refit bundle after threshold choice, not from the threshold-selection split itself. In these runs the threshold was still chosen on the held-out tail of the validation frame, then the model was refit on all validation rows before test-time use.

## Best Run
- `learned_gbdt_regime`: [config_resolved.yaml](/Users/davidwilkinson/Desktop/ETS 2/runs/20260316_042510_05e726/config_resolved.yaml)
- Path metrics: [path_metrics.csv](/Users/davidwilkinson/Desktop/ETS 2/runs/20260316_042510_05e726/results/path_metrics.csv)
- Horizon metrics: [metrics_by_horizon.csv](/Users/davidwilkinson/Desktop/ETS 2/runs/20260316_042510_05e726/results/metrics_by_horizon.csv)
- Gate selection: [online_memory_gate_selection_TSM_LLM-COT-RF-HDELTA.json](/Users/davidwilkinson/Desktop/ETS 2/runs/20260316_042510_05e726/llm/online_memory_gate_selection_TSM_LLM-COT-RF-HDELTA.json)
