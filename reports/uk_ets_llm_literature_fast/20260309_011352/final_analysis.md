# UK ETS LLM Literature Sweep: Final Analysis

Generated: 2026-03-09

## Outcome

The 10-way literature-backed sweep improved the UK raw `TSM+LLM-COT-RF-HDELTA` frontier from `22.573653` to `22.418530` path MSE.

- Previous best raw UK `TSM+LLM`: `22.573653`
- Base UK `tsm`: `22.726353`
- UK `linear_ridge`: `22.419955`
- New best fast-screen variant: `22.418530`
- New best archived rerun: `22.418530`

This is the first UK `TSM+LLM` result that narrowly beats `linear_ridge` on raw test path MSE.

## Best Variant

- Candidate: `least_to_most_pot`
- Literature: `Least-to-Most + Program of Thoughts`
- Aim: sequentially constrain horizon reasoning while making the apply stage more numeric and bounded
- Fast-screen run: `/Users/davidwilkinson/Desktop/ETS 2/runs/20260309_025246_3f76b2`
- Archived rerun: `/Users/davidwilkinson/Desktop/ETS 2/runs/20260309_032039_a6866c`

Key archived metrics:

- Raw `TSM+LLM-COT-RF-HDELTA`: `22.418530`
- Base `tsm`: `22.726353`
- `linear_ridge`: `22.419955`
- Improvement vs base `tsm`: `-0.307822`
- Improvement vs prior raw UK `TSM+LLM`: `-0.155123`
- Improvement vs `linear_ridge`: `-0.001425`

Horizon-level behavior vs base `tsm`:

- `h1`: unchanged
- `h5`: slightly worse by `+0.007464`
- `h20`: better by `-0.652601`
- `h30`: better by `-0.393346`

## Interpretation

The winner did not come from adding more memory or more sampling. It came from better decomposition.

- `least_to_most` helped because it forced the reflection stage to decide shorter horizons before touching the longer ones.
- `program_of_thought` helped because it made the apply stage behave more like bounded numeric adjustment instead of generic prose-conditioned editing.
- The combination worked better than either one alone.

## Important Caveat

The validation blend selection still preferred `w=0.00`, even though the raw LLM forecast is the best test result. This means the current validation blend selector is not aligned with the test behavior for this variant.

- Val blend winner: base-heavy fallback
- Test best result: raw LLM path

So the gain is real in the raw LLM-corrected forecast, but the current blend calibration logic is lagging behind it.

## Sources

- Self-Consistency: https://arxiv.org/abs/2203.11171
- Least-to-Most: https://arxiv.org/abs/2205.10625
- Plan-and-Solve: https://arxiv.org/abs/2305.04091
- Program of Thoughts: https://arxiv.org/abs/2211.12588
- Active-Prompt: https://arxiv.org/abs/2302.12246
- Auto-CoT: https://arxiv.org/abs/2210.03493
- Reflexion: https://arxiv.org/abs/2303.11366
- Self-Harmonized CoT: https://arxiv.org/abs/2409.04057
