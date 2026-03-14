# UK ETS LLM Literature Fast Sweep

Generated: 2026-03-09T03:19:31.603484

This sweep screened literature-backed UK `TSM+LLM-COT-RF-HDELTA` variants on the exploratory UK test subset to iterate quickly.

## Candidate Results

| Candidate | Literature | Aim | Expected | LLM Path MSE | Delta vs TSM | h5 Delta | h20 Delta | h30 Delta | Apply Samples | Unique Apply |
|---|---|---|---|---:|---:|---:|---:|---:|---|---:|
| least_to_most_pot | Least-to-Most + Program of Thoughts | Combine sequential horizon reasoning with a stricter numeric apply-stage contract. | Could preserve h5/h20 gains while preventing noisy h30 moves; target roughly 22.50-22.57. | 22.418530 | -0.307822 | +0.007464 | -0.652601 | -0.393346 | 1 | 62 |
| least_to_most | Least-to-Most / Zhou et al. 2022 | Force the reflection step to decide horizons sequentially so h30 cannot drift without earlier-horizon support. | Should help if residual error still comes from horizon coupling; target roughly 22.54-22.60. | 22.440286 | -0.286067 | +0.013706 | -0.598846 | -0.388525 | 1 | 57 |
| active_prompt_recent_high_error | Active-Prompt / Diao et al. 2023 | Bias teaching examples toward recent high-error UK cases where the base TSM is least reliable. | Could help if recent hard regimes matter more than broad average cases; target roughly 22.56-22.62. | 22.489051 | -0.237302 | +0.026225 | -0.353010 | -0.601114 | 1 | 66 |
| self_consistency_k3 | Self-Consistency / Wang et al. 2022 | Sample multiple apply-stage adjustment paths and aggregate them by median to reduce unstable numeric outputs. | Highest upside among single-change variants if apply noise is still material; target roughly 22.50-22.58. | 22.503846 | -0.222507 | -0.035690 | -0.311208 | -0.616085 | 3 | 84 |
| program_of_thought_apply | Program of Thoughts / Chen et al. 2022 | Make the apply stage behave more like bounded numeric computation than free-form adjustment writing. | Could reduce apply-stage numeric noise and improve h20/h30; target roughly 22.53-22.59. | 22.509311 | -0.217042 | -0.055022 | -0.302467 | -0.539454 | 1 | 65 |
| hybrid_memory_sc | Reflexion + Harmonized CoT + Self-Consistency | Combine persistent failure memory, dominant-pattern reflection, and median aggregation of multiple apply samples. | Best upside but also highest risk of over-constraining the model; target roughly 22.47-22.56. | 22.510896 | -0.215456 | -0.099794 | -0.427597 | -0.000003 | 3 | 71 |
| harmonized_reasoning | Self-Harmonized CoT / Fu et al. 2024 | Force the model to harmonize matched examples into one dominant pattern instead of chasing example-specific noise. | Could help if the reflection stage is too sensitive to outliers; target roughly 22.54-22.60. | 22.517808 | -0.208545 | -0.010668 | -0.288401 | -0.532394 | 1 | 62 |
| plan_and_solve | Plan-and-Solve / Wang et al. 2023 | Make reflection plan first and only then emit structured horizon decisions. | Could stabilize structured guidance if the model benefits from an explicit plan phase; target roughly 22.55-22.61. | 22.518543 | -0.207809 | -0.029065 | -0.352344 | -0.438298 | 1 | 41 |
| auto_cot_error_stratified | Auto-CoT / Zhang et al. 2022 | Broaden the teaching set to cover diverse historical error modes instead of mostly similar cases. | Could improve h20/h30 if the current prompt is overfitting to a narrow case family; target roughly 22.55-22.61. | 22.528369 | -0.197984 | -0.003921 | -0.298013 | -0.454317 | 1 | 49 |
| reflexion_memory | Reflexion / Shinn et al. 2023 | Inject persistent UK-specific failure memory so the model avoids repeating known bad long-horizon behaviors. | Could trim repeated h30 mistakes if the model responds to explicit memory; target roughly 22.54-22.60. | 22.569588 | -0.156765 | -0.057480 | -0.285606 | -0.000003 | 1 | 32 |

## Best Screened Variant

- Candidate: `least_to_most_pot`
- Literature: `Least-to-Most + Program of Thoughts`
- Aim: Combine sequential horizon reasoning with a stricter numeric apply-stage contract.
- Expected: Could preserve h5/h20 gains while preventing noisy h30 moves; target roughly 22.50-22.57.
- Run dir: `/Users/davidwilkinson/Desktop/ETS 2/runs/20260309_025246_3f76b2`
- Raw path MSE: `22.418530`
- Delta vs base TSM: `-0.307822`
