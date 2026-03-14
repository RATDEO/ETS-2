# UK ETS 35B Tool Follow-Up Sweep

Generated: 2026-03-14T08:08:33.970332

## References

- Baseline run: `/Users/davidwilkinson/Desktop/ETS 2/runs/20260313_141036_907f02`
- Baseline raw `TSM+LLM-COT-RF-HDELTA` path MSE: `21.509699`
- `linear_ridge` path MSE: `22.419955`
- Old best 35B no-tool step-back path MSE: `21.484623`

## Results

| Candidate | Actual | Vs baseline | Vs ridge | Vs old best | Outcome | h5 vs baseline | h20 vs baseline | h30 vs baseline | Numeric | Verifier | Retrieval | Market | Counterexample | Unique reflect | Unique apply |
|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| optional_tools_default | 21.509699 | +0.000000 | -0.910256 | +0.025076 | within | +0.000000 | +0.000000 | +0.000000 | 144 | 144 | 0 | 0 | 0 | 144 | 22 |
| numeric_only_default | 21.502944 | -0.006755 | -0.917011 | +0.018321 | within | -0.002653 | -0.009915 | -0.015608 | 144 | 0 | 0 | 0 | 0 | 144 | 23 |
| verifier_only_default | 21.541713 | +0.032014 | -0.878242 | +0.057091 | within | +0.001793 | +0.055911 | +0.083427 | 0 | 144 | 0 | 0 | 0 | 144 | 21 |
| react_evidence_optional_tools | 22.072692 | +0.562993 | -0.347264 | +0.588069 | missed | +0.236046 | +0.756422 | +1.073786 | 144 | 144 | 0 | 0 | 0 | 140 | 12 |
| plan_and_solve_optional_tools | 21.700990 | +0.191291 | -0.718965 | +0.216367 | missed | +0.143502 | +0.166470 | +0.400668 | 144 | 144 | 0 | 0 | 0 | 144 | 17 |
| skill_tag_recent_high_error_tools | 22.041922 | +0.532223 | -0.378033 | +0.557300 | missed | +0.199174 | +0.672206 | +1.648144 | 144 | 144 | 0 | 0 | 0 | 144 | 17 |
| balanced_long_horizon_tools | 22.369328 | +0.859629 | -0.050628 | +0.884705 | missed | -0.066264 | +1.116453 | +3.419948 | 144 | 144 | 0 | 0 | 0 | 143 | 21 |
| counterexample_recent_high_error_tools | 21.635979 | +0.126280 | -0.783976 | +0.151356 | missed | +0.111443 | +0.075352 | +0.642189 | 144 | 144 | 0 | 0 | 0 | 144 | 20 |
| step_back_optional_tools_k6 | 21.789554 | +0.279855 | -0.630401 | +0.304931 | missed | +0.111648 | +0.390314 | +1.065021 | 144 | 144 | 0 | 0 | 0 | 142 | 14 |
| step_back_numeric_optional_k6 | 21.793566 | +0.283867 | -0.626389 | +0.308943 | missed | +0.111648 | +0.397309 | +1.079260 | 144 | 0 | 0 | 0 | 0 | 142 | 16 |

## Best Candidate

- Candidate: `numeric_only_default`
- Literature: [Program of Thoughts](https://arxiv.org/abs/2211.12588)
- Change: Remove the verifier and keep only the numeric tool under the current least-to-most stack.
- Aim: Test whether exact arithmetic helps more than post-hoc clipping on 35B.
- Expected range: `21.38-21.54`
- Actual raw path MSE: `21.502944`
- Delta vs baseline: `-0.006755`
- Delta vs ridge: `-0.917011`
- Delta vs old best: `+0.018321`
- Outcome vs expectation: `within`
- Run dir: `/Users/davidwilkinson/Desktop/ETS 2/runs/20260314_033401_10feb9`
