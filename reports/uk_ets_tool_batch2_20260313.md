# UK ETS Tool Batch 2 - March 13, 2026

Objective: starting from the current 4B verifier-only default, test the next tool batch in isolated full 144-window runs and one combined run.

Baseline/default:
- Config: `uk_ets/config/uk_ets_llm_4b_current_default.yaml`
- Run: `runs/20260313_020215_90a3c5`
- Tools: numeric analysis + delta verifier
- Path MSE: `21.516885`

Comparison set:

| Variant | Config | Run ID | Tools | Path MSE |
|---|---|---|---|---:|
| Numeric baseline | `uk_ets_llm_4b_numeric_delta_verifier.yaml` | `20260313_020215_90a3c5` | numeric + verifier | `21.516885` |
| Market microstructure | `uk_ets_llm_4b_verifier_market_microstructure.yaml` | `20260313_125334_94b530` | market + numeric + verifier | `21.547346` |
| Freeze counterexample | `uk_ets_llm_4b_verifier_freeze_counterexample.yaml` | `20260313_131402_441a7f` | counterexample + numeric + verifier | `21.537228` |
| Combined | `uk_ets_llm_4b_verifier_market_and_counterexample.yaml` | `20260313_133407_245a87` | market + counterexample + numeric + verifier | `21.556038` |

Reference models:
- Base `tsm`: `22.726353`
- `linear_ridge`: `22.419955`

Horizon MSE vs verifier-only default:

| Variant | h1 | h5 | h20 | h30 |
|---|---:|---:|---:|---:|
| Verifier-only default | `1.224958` | `4.628752` | `27.970310` | `56.340463` |
| Market microstructure | `1.224958` | `4.630733` | `28.000940` | `56.382949` |
| Freeze counterexample | `1.224958` | `4.612618` | `28.008500` | `56.322824` |
| Combined | `1.224958` | `4.629105` | `28.043817` | `56.387725` |

Observed tool invocation counts:
- Market run: `get_market_microstructure_state=144`, `get_numeric_analysis=144`, `verify_hdelta_adjustments=144`
- Counterexample run: `get_freeze_counterexample=144`, `get_numeric_analysis=144`, `verify_hdelta_adjustments=144`
- Combined run: `get_market_microstructure_state=144`, `get_freeze_counterexample=144`, `get_numeric_analysis=144`, `verify_hdelta_adjustments=144`

Findings:
- None of the new variants beat the verifier-only default.
- Freeze counterexample was the closest challenger, but still regressed by `+0.020343` path MSE.
- Market microstructure added useful context but did not translate into better full-run accuracy.
- Combining both extra tools diluted the gain further rather than compounding it.

Conclusion:
- Keep verifier-only as the current default 4B benchmark path.
- Do not promote market microstructure or freeze counterexample tools into the default stack.
- If revisiting these tools later, treat them as regime-conditional or optional gates rather than always-on tools.
