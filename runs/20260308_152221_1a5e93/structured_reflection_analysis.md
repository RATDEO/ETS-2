# Structured Reflection Run

## What changed

This run enabled structured per-horizon HDELTA reflection on top of the UK case-conditioned `TSM+LLM-COT-RF-HDELTA` flow.

- Reflection returned per-horizon JSON guidance instead of free-text rules.
- Apply prompts consumed explicit `mode`, `preferred_sign`, `confidence`, `magnitude`, and `reason` for each key horizon.
- Code changes live in:
  - [refine.py](/Users/davidwilkinson/Desktop/ETS%202/src/llm/refine.py)
  - [prompts.py](/Users/davidwilkinson/Desktop/ETS%202/src/llm/prompts.py)
  - [test_llm_guarded_rf_hdelta.py](/Users/davidwilkinson/Desktop/ETS%202/tests/test_llm_guarded_rf_hdelta.py)

## Result

- Base TSM path MSE: `22.726353`
- Previous best raw UK TSM+LLM path MSE: `22.618999`
- This structured-reflection run: `22.591195`

So the new path improved:

- `-0.135158` vs base TSM
- `-0.027804` vs the prior best raw UK TSM+LLM run

Headline metrics are in [path_metrics.csv](/Users/davidwilkinson/Desktop/ETS%202/runs/20260308_152221_1a5e93/results/path_metrics.csv). Horizon detail is in [metrics_by_horizon.csv](/Users/davidwilkinson/Desktop/ETS%202/runs/20260308_152221_1a5e93/results/metrics_by_horizon.csv).

## Interpretation

The gain came mainly from removing the long-horizon drag:

- `h5` got slightly worse than the previous best raw LLM path.
- `h20` stayed effectively flat.
- `h30` improved materially, which more than offset the small `h5` giveback.

The blend grid did not improve on the raw result. Validation selected `w=0.00`, and the raw structured LLM path remained the best answer to keep from this run. See [blend_grid_summary.csv](/Users/davidwilkinson/Desktop/ETS%202/runs/20260308_152221_1a5e93/results/blend_grid_ramp/TSM_LLM-COT-RF-HDELTA/test/blend_grid_summary.csv).

## Next move

The next highest-value refinement is to keep this structured reflection path and add a stronger long-horizon gate so `h30` only moves when the matched-example evidence is unusually strong. That should preserve the current improvement and may recover the small `h5` loss without reintroducing the old `h30` damage.
