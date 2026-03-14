# UK ETS 35B Horizon Follow-Up Sweep

Generated: 2026-03-09T19:29:38.561969

## Baseline

- Baseline run: `/Users/davidwilkinson/Desktop/ETS 2/runs/20260309_151421_2aca41`
- Baseline raw `TSM+LLM-COT-RF-HDELTA` path MSE: `21.737755`
- Reference `linear_ridge` path MSE: `22.419955`

## Candidate Results

| Candidate | Change | Aim | Expected | Actual | Vs baseline | Outcome vs expected | h5 vs baseline | h20 vs baseline | h30 vs baseline | Apply samples | Unique apply |
|---|---|---|---|---:|---:|---|---:|---:|---:|---|---:|
| h5_shrink_selective | Shrink h5 case bounds and demand stronger sign agreement only for h5. | Keep h5 actionable when evidence is clean, but cut its adjustment size and frequency when the signal is noisy. | 21.67-21.76 | 21.828217 | +0.090462 | missed | +0.036343 | +0.149973 | +0.050092 | 1 | 42 |
| structured_h5_guard | Require higher structured confidence before keeping h5 adjustments, while keeping h20/h30 medium-confidence capable. | Reduce short-horizon overcorrection but preserve the larger medium and long-horizon gains. | 21.66-21.76 | 21.855855 | +0.118100 | missed | +0.159329 | +0.070447 | -0.029425 | 1 | 44 |
| horizon_specific_matching | Enable horizon-specific matched-example selection within the recent high-error teaching pool. | Give h5 more local analogs without sacrificing the recent hard-case evidence that is helping h20 and h30. | 21.65-21.74 | 22.045273 | +0.307518 | missed | -0.164492 | +0.529731 | +1.099881 | 1 | 34 |
| hmatch_plus_h5_shrink | Combine horizon-specific matching with a tighter, more selective h5 bound. | Highest-upside variant: use better per-horizon matches while keeping h5 from reacting too aggressively. | 21.60-21.72 | 22.045273 | +0.307518 | missed | -0.164492 | +0.529731 | +1.099881 | 1 | 34 |

## Best Candidate

- Candidate: `h5_shrink_selective`
- Change: Shrink h5 case bounds and demand stronger sign agreement only for h5.
- Aim: Keep h5 actionable when evidence is clean, but cut its adjustment size and frequency when the signal is noisy.
- Expected range: `21.67-21.76`
- Actual raw path MSE: `21.828217`
- Delta vs baseline raw LLM: `+0.090462`
- Delta vs base TSM: `-0.898135`
- Delta vs ridge: `-0.591738`
- Outcome vs expectation: `missed`
- Run dir: `/Users/davidwilkinson/Desktop/ETS 2/runs/20260309_184621_22d8c4`
