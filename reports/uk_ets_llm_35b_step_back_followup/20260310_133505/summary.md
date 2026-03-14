# UK ETS 35B Step-Back Follow-Up Sweep

Generated: 2026-03-10T15:48:33.767064

## Baseline

- Baseline run: `/Users/davidwilkinson/Desktop/ETS 2/runs/20260309_234219_249d62`
- Baseline raw `TSM+LLM-COT-RF-HDELTA` path MSE: `21.484623`
- Reference `linear_ridge` path MSE: `22.419955`

## Planned Variants

| Candidate | Literature | Aim | Expected range |
|---|---|---|---|
| thought_propagation_step_back | [Thought Propagation](https://arxiv.org/abs/2310.03965) | Improve h20 and h30 by reusing successful analogous correction patterns rather than reasoning from scratch each time. | 21.34-21.50 |
| uncertainty_routed_step_back | [UnCert-CoT / Uncertainty of Thoughts](https://arxiv.org/abs/2503.15341) | Preserve the current winner on easy windows while spending extra deliberation only where evidence is ambiguous. | 21.32-21.48 |
| analogical_step_back | [Large Language Models as Analogical Reasoners](https://arxiv.org/abs/2310.01714) | Exploit the matched-example memory more directly without giving up the strong regime-first reasoning structure. | 21.36-21.50 |
| self_verification_step_back | [Self-Verification Improves Few-Shot CoT](https://arxiv.org/abs/2212.09561) | Keep the regime abstraction but reject horizon moves that do not survive a backward consistency check. | 21.37-21.52 |
| self_discover_step_back | [Self-Discover](https://arxiv.org/abs/2402.03620) | See whether dynamic reasoning-structure selection can improve on a fixed step-back regime contract. | 21.40-21.56 |

## Results

| Candidate | Actual | Vs baseline | Vs ridge | Outcome | h5 vs baseline | h20 vs baseline | h30 vs baseline | Reflect samples | Routed reflect calls | Unique reflect | Unique apply |
|---|---:|---:|---:|---|---:|---:|---:|---|---:|---:|---:|
| thought_propagation_step_back | 21.547992 | +0.063369 | -0.871963 | missed | +0.000023 | +0.167175 | +0.094604 | 1 | 0 | 144 | 30 |
| uncertainty_routed_step_back | 21.571120 | +0.086497 | -0.848835 | missed | +0.089501 | +0.216065 | -0.087537 | 3 | 144 | 144 | 28 |
| analogical_step_back | 21.603815 | +0.119193 | -0.816140 | missed | -0.060465 | +0.151231 | +0.435481 | 1 | 0 | 144 | 34 |
| self_verification_step_back | 22.216036 | +0.731414 | -0.203919 | missed | -0.107533 | +1.267717 | +1.815881 | 1 | 0 | 144 | 14 |
| self_discover_step_back | 22.450465 | +0.965842 | +0.030510 | missed | -0.050964 | +1.509246 | +2.334695 | 1 | 0 | 144 | 11 |

## Best Candidate

- Candidate: `thought_propagation_step_back`
- Literature: [Thought Propagation](https://arxiv.org/abs/2310.03965)
- Change: Ask the model to propagate reusable reasoning patterns from matched examples after the step-back regime abstraction.
- Aim: Improve h20 and h30 by reusing successful analogous correction patterns rather than reasoning from scratch each time.
- Expected range: `21.34-21.50`
- Actual raw path MSE: `21.547992`
- Delta vs baseline raw LLM: `+0.063369`
- Delta vs ridge: `-0.871963`
- Run dir: `/Users/davidwilkinson/Desktop/ETS 2/runs/20260310_135720_7d8551`
