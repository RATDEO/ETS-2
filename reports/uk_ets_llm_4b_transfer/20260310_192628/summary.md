# UK ETS 4B Transfer Sweep

Generated: 2026-03-10T20:52:14.919451

## Checkpoint

- Base checkpoint run: `/Users/davidwilkinson/Desktop/ETS 2/runs/20260309_032039_a6866c`
- Checkpoint raw `TSM+LLM-COT-RF-HDELTA` path MSE: `22.418530`
- Reference `linear_ridge` path MSE: `22.419955`

## Candidate Results

| Candidate | Change | Aim | Expected | Actual | Vs checkpoint | Outcome vs expected | h5 vs checkpoint | h20 vs checkpoint | h30 vs checkpoint | Nonzero deltas |
|---|---|---|---|---:|---:|---|---:|---:|---:|---:|
| recent_high_error_180_k4 | Use the same recent high-error 180-day pool, but reduce the teaching set from 6 to 4 examples. | Reduce context load for the smaller 4B model while keeping the evidence focused on recent failure modes. | 22.15-22.30 | 22.198106 | -0.220424 | within | +0.018949 | -0.279300 | -0.798447 | 3767 |
| recent_high_error_180_k4_longloose | Use recent high-error 180-day teaching examples, 4 examples total, and loosen `h20/h30` case bounds. | Test whether 4B is under-adjusting the long horizons even when it gets the direction right. | 22.12-22.28 | 22.198106 | -0.220424 | within | +0.018949 | -0.279300 | -0.798447 | 3767 |
| recent_high_error_180_k6 | Switch teaching examples to recent high-error UK cases from the last 180 days. | Transfer the best 35B example-pool idea to 4B without changing the reasoning contract. | 22.20-22.35 | 22.282915 | -0.135616 | within | +0.050171 | -0.074470 | -0.582779 | 3722 |
| recent_high_error_180_k4_h5freeze | Use recent high-error 180-day teaching examples, 4 examples total, and force `h5` to freeze. | Protect the weak short-horizon behavior of 4B while concentrating its capacity on `h20` and `h30`. | 22.10-22.25 | 22.299505 | -0.119025 | missed | -0.007464 | +0.012255 | -0.945564 | 2589 |
| recent_high_error_180_k4_h5freeze_longloose | Combine recent high-error 180-day teaching examples, 4 examples, `h5` freeze, and looser `h20/h30` bounds. | Try the highest-upside 4B transfer: remove the short-horizon drag while giving the long horizons room to move. | 22.05-22.22 | 22.299505 | -0.119025 | missed | -0.007464 | +0.012255 | -0.945564 | 2589 |
| control_hybrid365_k6 | Reproduce the prior 4B checkpoint with blend selection disabled for a clean raw-path control. | Confirm the current code path still reproduces the stored 4B checkpoint before testing transfers. | 22.38-22.45 | 22.371401 | -0.047130 | beat | -0.004207 | -0.039871 | -0.184221 | 3675 |
| hybrid180_k6 | Keep similarity-error hybrid retrieval, but restrict the teaching pool to the most recent 180 days. | Test whether locality alone helps 4B without abandoning similarity matching. | 22.22-22.36 | 22.414139 | -0.004391 | missed | -0.021896 | +0.082883 | -0.131600 | 3763 |

## Best Candidate

- Candidate: `recent_high_error_180_k4`
- Change: Use the same recent high-error 180-day pool, but reduce the teaching set from 6 to 4 examples.
- Aim: Reduce context load for the smaller 4B model while keeping the evidence focused on recent failure modes.
- Expected range: `22.15-22.30`
- Actual raw path MSE: `22.198106`
- Delta vs checkpoint raw LLM: `-0.220424`
- Delta vs base TSM: `-0.528247`
- Delta vs ridge: `-0.221849`
- Outcome vs expectation: `within`
- Run dir: `/Users/davidwilkinson/Desktop/ETS 2/runs/20260310_195100_69e04e`
