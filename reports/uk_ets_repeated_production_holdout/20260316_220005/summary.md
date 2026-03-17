# Repeated Production Holdout Benchmark

Generated: 2026-03-16T23:17:55.597334

Protocol:
- Same split geometry as the current promoted production run.
- Window geometry is shifted backward by one full test-span at a time.
- Methodology is frozen: `DLinear + learned_gbdt(top20) + selective LLM`.
- Only causal training data changes per window.

## Window Results

| window                   | train_end   | val_end    | test_end   | run_dir                                                         |   tsm_path_mse |   llm_path_mse |   path_improvement_pct |   tsm_h1 |   llm_h1 |   tsm_h5 |   llm_h5 |   tsm_h20 |   llm_h20 |   h20_improvement_pct |   tsm_h30 |   llm_h30 |   h30_improvement_pct | llm_cutoff_status   | llm_cutoff_date   |   llm_n_pre_cutoff |   llm_n_post_cutoff |
|:-------------------------|:------------|:-----------|:-----------|:----------------------------------------------------------------|---------------:|---------------:|-----------------------:|---------:|---------:|---------:|---------:|----------:|----------:|----------------------:|----------:|----------:|----------------------:|:--------------------|:------------------|-------------------:|--------------------:|
| W0_2025-07-01_2026-03-04 | 2024-06-30  | 2025-06-30 | 2026-03-04 | /Users/davidwilkinson/Desktop/ETS 2/runs/20260316_170659_fb765d |        4.20244 |        3.72991 |             0.11244    | 0.782575 | 0.782575 |  2.39944 |  2.45368 |   4.11202 |   3.44901 |            0.161237   |   9.13173 |   7.89196 |             0.135765  | post_cutoff_only    | 2025-01-01        |                  0 |                 103 |
| W1_2024-10-27_2025-06-30 | 2023-10-27  | 2024-10-26 | 2025-06-30 | /Users/davidwilkinson/Desktop/ETS 2/runs/20260316_220006_2bb82e |       36.1355  |       36.4843  |            -0.00965268 | 1.99434  | 1.99434  | 10.4053  | 10.2942  |  49.8403  |  50.3273  |           -0.00977259 |  63.8589  |  65.0505  |            -0.01866   | crosses_cutoff      | 2025-01-01        |                 45 |                  97 |
| W2_2024-02-23_2024-10-26 | 2023-02-22  | 2024-02-22 | 2024-10-26 | /Users/davidwilkinson/Desktop/ETS 2/runs/20260316_221954_fe316a |       30.2879  |       31.9178  |            -0.0538135  | 2.76543  | 2.76543  | 12.5841  | 13.072   |  40.6861  |  43.5869  |           -0.0712984  |  55.4939  |  58.0529  |            -0.0461139 | pre_cutoff_only     | 2025-01-01        |                145 |                   0 |
| W3_2023-06-21_2024-02-22 | 2022-06-20  | 2023-06-20 | 2024-02-22 | /Users/davidwilkinson/Desktop/ETS 2/runs/20260316_224342_3472d0 |       85.6555  |       84.5692  |             0.0126824  | 2.71935  | 2.71935  | 22.0147  | 21.9207  | 108.954   | 107.178   |            0.0162965  | 188.063   | 185.42    |             0.0140545 | pre_cutoff_only     | 2025-01-01        |                145 |                   0 |
| W4_2022-10-17_2023-06-20 | 2021-10-16  | 2022-10-16 | 2023-06-20 | /Users/davidwilkinson/Desktop/ETS 2/runs/20260316_225915_798bce |       58.979   |       59.8434  |            -0.0146555  | 4.71426  | 4.71426  | 24.539   | 24.6446  |  75.9278  |  77.1827  |           -0.0165274  | 101.185   | 103.067   |            -0.0185925 | pre_cutoff_only     | 2025-01-01        |                145 |                   0 |

## Summary
- LLM beat raw `TSM` on path MSE in `2/5` repeated holdouts.
- Mean path-MSE improvement: `0.94%`.
- Median path-MSE improvement: `-0.97%`.
- Mean `h20` improvement: `1.60%`.
- Mean `h30` improvement: `1.33%`.

## Post-2025 Windows
- Windows: `1`.
- Mean path-MSE improvement: `11.24%`.
- Mean `h20` improvement: `16.12%`.
- Mean `h30` improvement: `13.58%`.
