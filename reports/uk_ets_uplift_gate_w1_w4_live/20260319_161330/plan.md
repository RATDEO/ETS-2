# W1-W4 Full Live Uplift Gate Benchmark Plan

Generated: 2026-03-19T16:13:30.261819

## Objective

Run a true live rerun of the `W1-W4` benchmark where the learned gate is fit on prior completed windows plus the current window's validation rows, then used directly inside the full causal LLM pipeline.

Frozen stack:
- UK data root: `uk_ets/Data_auto_uk`
- Regularized shared DLinear base: `lr=0.001`, `epochs=40`, `patience=8`, `dropout=0.5`, `weight_decay=0.02`
- Same `top20` gate feature schema
- Same 4B LLM refinement path

## Candidates

- `uplift_logistic_live`
- `uplift_gbdt_live`

## Training protocol

- Process windows in chronological order `W4 -> W3 -> W2 -> W1`
- For each candidate, bootstrap learned-gate training rows from that candidate's prior completed windows
- Fit the gate on prior bootstrap rows plus the current window's validation rows
- Select threshold from the current window's validation rows only

## Expected outcome

- More realistic than the conservative offline suppression test because the gate can both add and remove applications in the live run
- Best chance of success: `uplift_gbdt_live`
- Success condition: beat the current `gate_top20` mean `W1-W4` path MSE `54.613995`
- Realistic expected lift if it works: about `+0.2%` to `+1.5%`
