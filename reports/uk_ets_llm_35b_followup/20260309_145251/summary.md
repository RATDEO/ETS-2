# UK ETS 35B Targeted Follow-Up Sweep

Generated: 2026-03-09T16:58:12.111513

## Checkpoint

- Base checkpoint run: `/Users/davidwilkinson/Desktop/ETS 2/runs/20260309_122028_0e017f`
- Checkpoint raw `TSM+LLM-COT-RF-HDELTA` path MSE: `22.143373`
- Reference `linear_ridge` path MSE: `22.419955`

## Candidate Results

| Candidate | Change | Aim | Expected | Actual | Vs checkpoint | Outcome vs expected | h5 vs checkpoint | h20 vs checkpoint | h30 vs checkpoint | Apply samples | Unique apply |
|---|---|---|---|---:|---:|---|---:|---:|---:|---|---:|
| recent_high_error_180 | Switch teaching examples to recent high-error cases from the last 180 days. | Bias the model toward the recent UK failure modes where the base TSM is least reliable. | 22.06-22.16 | 21.737755 | -0.405618 | beat | +0.140934 | -0.478738 | -1.512019 | 1 | 44 |
| recent_high_error_180_sc3 | Combine recent high-error teaching examples with 3-sample median apply self-consistency. | Combine the strongest example-pool shift with apply-stage variance reduction for the highest upside run. | 21.98-22.10 | 21.761403 | -0.381970 | beat | +0.138491 | -0.437985 | -1.473730 | 3 | 84 |
| recent_high_error_180_k8 | Use recent high-error teaching examples and increase the evidence set from 6 to 8 examples. | Exploit the 35B context window with more recent hard cases while keeping the evidence regime-local. | 22.02-22.12 | 21.781881 | -0.361492 | beat | +0.166841 | -0.436425 | -1.257809 | 1 | 44 |
| self_consistency_k3 | Keep the checkpoint prompt stack, but aggregate 3 apply-stage samples with a median. | Reduce numeric apply jitter while preserving the current reasoning and example policy. | 22.08-22.16 | 22.155557 | +0.012184 | within | +0.011616 | +0.011943 | +0.028745 | 3 | 83 |
| hybrid_recent180 | Keep similarity-error hybrid selection, but restrict the teaching pool to the most recent 180 days. | Test whether shorter-horizon exemplar locality helps the 35B model without giving up structured similarity matching. | 22.10-22.18 | 22.190640 | +0.047267 | missed | +0.058964 | +0.211072 | -0.384476 | 1 | 50 |

## Best Candidate

- Candidate: `recent_high_error_180`
- Change: Switch teaching examples to recent high-error cases from the last 180 days.
- Aim: Bias the model toward the recent UK failure modes where the base TSM is least reliable.
- Expected range: `22.06-22.16`
- Actual raw path MSE: `21.737755`
- Delta vs checkpoint raw LLM: `-0.405618`
- Delta vs base TSM: `-0.988597`
- Delta vs ridge: `-0.682200`
- Outcome vs expectation: `beat`
- Run dir: `/Users/davidwilkinson/Desktop/ETS 2/runs/20260309_151421_2aca41`
