# UK ETS 35B Recent-Window Sweep

Generated: 2026-03-09T20:58:18.745322

## Baseline

- Baseline run: `/Users/davidwilkinson/Desktop/ETS 2/runs/20260309_151421_2aca41`
- Baseline raw `TSM+LLM-COT-RF-HDELTA` path MSE: `21.737755`
- Reference `linear_ridge` path MSE: `22.419955`

## Candidate Results

| Candidate | Change | Aim | Expected | Actual | Vs baseline | Outcome vs expected | h5 vs baseline | h20 vs baseline | h30 vs baseline | Unique apply |
|---|---|---|---|---:|---:|---|---:|---:|---:|---:|
| recent_high_error_120 | Shorten the recent high-error teaching pool to 120 days. | Test whether an even more local regime window improves the recent-high-error signal or becomes too sparse. | 21.78-21.88 | 21.751460 | +0.013705 | beat | +0.036510 | +0.009908 | -0.078292 | 45 |
| recent_high_error_270 | Extend the recent high-error teaching pool to 270 days. | Add more high-error UK cases while staying recent enough to preserve the regime-local benefit. | 21.68-21.76 | 21.937856 | +0.200101 | missed | -0.041072 | +0.163544 | +0.972672 | 35 |
| recent_high_error_270_case_top2 | Use a 270-day recent high-error pool but sharpen case controls to two matched examples. | Keep the broader recent hard-case pool while reducing matched-example dilution inside case-conditioned bounds. | 21.66-21.75 | 22.036700 | +0.298944 | missed | +0.023531 | +0.215241 | +1.062121 | 28 |
| recent_high_error_365 | Extend the recent high-error teaching pool to a full 365 days. | Check whether the 35B model benefits from a broader hard-case memory once the pool is restricted to high-error examples. | 21.70-21.80 | 22.359700 | +0.621945 | missed | -0.112316 | +0.802553 | +1.937426 | 36 |

## Best Candidate

- Candidate: `recent_high_error_120`
- Change: Shorten the recent high-error teaching pool to 120 days.
- Aim: Test whether an even more local regime window improves the recent-high-error signal or becomes too sparse.
- Expected range: `21.78-21.88`
- Actual raw path MSE: `21.751460`
- Delta vs baseline raw LLM: `+0.013705`
- Delta vs base TSM: `-0.974892`
- Delta vs ridge: `-0.668495`
- Outcome vs expectation: `beat`
- Run dir: `/Users/davidwilkinson/Desktop/ETS 2/runs/20260309_193126_299c18`
