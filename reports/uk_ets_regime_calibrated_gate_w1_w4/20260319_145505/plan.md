# W1-W4 Regime-Calibrated Gate Benchmark Plan

Generated: 2026-03-19T14:55:05.903392

## Objective

Test whether the current `top20` learned gate improves on hard historical windows when the apply threshold is calibrated by simple regime bucket rather than kept global.

Frozen stack:
- UK-specific data root: `uk_ets/Data_auto_uk`
- Regularized shared DLinear base: `lr=0.001`, `epochs=40`, `patience=8`, `dropout=0.5`, `weight_decay=0.02`
- Same 4B LLM refinement stack
- Same `top20` learned-gate features

## Benchmark windows

- `W1`: 2024-10-27 to 2025-06-30
- `W2`: 2024-02-23 to 2024-10-26
- `W3`: 2023-06-21 to 2024-02-22
- `W4`: 2022-10-17 to 2023-06-20

## Candidates

- `gate_top20_baseline`: current global-threshold learned gate
- `regime_cal_move_default`: moderate/large move buckets at `h20=2.5%`, `h30=4.0%`
- `regime_cal_move_loose`: same idea with looser `3.0%` / `5.0%` caps
- `regime_cal_move_tight`: tighter `2.0%` / `3.0%` caps
- `regime_cal_move_vol_default`: move buckets plus `profile_vol_pct > 3.8` split
- `regime_cal_move_vol_loose`: move buckets plus a looser volatility split

## Expected outcome

Hard safe-regime blocking already failed, so the expected gain here is modest rather than dramatic.

Expected ranking:
- Best candidate likely `regime_cal_move_default` or `regime_cal_move_vol_default`
- Expected mean `W1-W4` path-MSE lift versus `gate_top20_baseline`: roughly `+0.1%` to `+0.8%`
- Success condition: improve mean `llm_path_mse` below the current baseline `54.613995` without collapsing the only clearly positive historical window (`W3`)

## Interpretation rule

If none of the regime-calibrated variants beat the current baseline, then the next bottleneck is not threshold calibration and we should move on to a stronger regime/uplift model rather than more threshold engineering.
