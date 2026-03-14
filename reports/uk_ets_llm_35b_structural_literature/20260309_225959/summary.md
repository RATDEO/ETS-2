# UK ETS 35B Structural Literature Sweep

Generated: 2026-03-10T03:27:16.169547

## Baseline

- Baseline run: `/Users/davidwilkinson/Desktop/ETS 2/runs/20260309_151421_2aca41`
- Baseline raw `TSM+LLM-COT-RF-HDELTA` path MSE: `21.737755`
- Reference `linear_ridge` path MSE: `22.419955`

## Planned Variants

| Candidate | Literature | Aim | Expected range |
|---|---|---|---|
| step_back_regime | [Step-Back Prompting](https://arxiv.org/abs/2310.06117) | Improve medium and long-horizon coherence by reasoning at the regime level before local deltas. | 21.63-21.78 |
| tree_of_thought_consensus | [Tree of Thoughts](https://arxiv.org/abs/2305.10601) | Search over multiple horizon plans and keep only consensus structure that survives aggregation. | 21.50-21.70 |
| react_evidence | [ReAct](https://arxiv.org/abs/2210.03629) | Tighten the link between retrieved evidence and bounded horizon actions. | 21.60-21.75 |
| skeleton_then_fill | [Skeleton-of-Thought](https://arxiv.org/abs/2307.15337) | Reduce over-adjustment by deciding freeze-versus-adjust structure before sizing any move. | 21.66-21.80 |
| chain_of_verification | [Chain-of-Verification](https://arxiv.org/abs/2309.11495) | Suppress hallucinated sign changes and keep only evidence-backed horizon moves. | 21.58-21.72 |
| self_refine_verifier | [Self-Refine](https://arxiv.org/abs/2303.17651) | Reduce unsupported horizon moves by forcing the model to criticize its own first-pass plan before output. | 21.60-21.74 |
| self_rag_critique | [Self-RAG](https://arxiv.org/abs/2310.11511) | Improve robustness by filtering adjustments through an evidence-sufficiency check rather than direct prompting alone. | 21.55-21.71 |
| self_ask_horizons | [Self-Ask](https://arxiv.org/abs/2210.03350) | Make the h5 and h20 decisions more deliberate and less coupled to the dominant long-horizon narrative. | 21.64-21.79 |
| debate_consensus | [Multi-Agent Debate](https://arxiv.org/abs/2305.19118) | Retain only the horizon decisions that survive a skeptic-versus-adjuster conflict. | 21.52-21.72 |
| rarr_attribution | [RARR](https://arxiv.org/abs/2210.08726) | Cut unsupported corrections by making every non-zero move justify itself from the matched examples. | 21.56-21.72 |

## Results

| Candidate | Actual | Vs baseline | Vs ridge | Outcome vs expected | h5 vs baseline | h20 vs baseline | h30 vs baseline | Reflect samples | Mean valid reflect | Unique reflect | Unique apply |
|---|---:|---:|---:|---|---:|---:|---:|---|---:|---:|---:|
| step_back_regime | 21.484623 | -0.253133 | -0.935333 | beat | -0.026986 | -0.380149 | -0.663947 | 1 | 1.00 | 142 | 25 |
| tree_of_thought_consensus | 21.773477 | +0.035722 | -0.646478 | missed | -0.006139 | +0.129600 | -0.345057 | 3 | 3.00 | 144 | 66 |
| react_evidence | 21.808505 | +0.070749 | -0.611451 | missed | +0.019544 | +0.254614 | -0.264166 | 1 | 1.00 | 141 | 22 |
| skeleton_then_fill | 21.931373 | +0.193617 | -0.488582 | missed | -0.015348 | +0.319436 | +0.131168 | 1 | 1.00 | 144 | 44 |
| chain_of_verification | 22.091358 | +0.353603 | -0.328597 | missed | +0.003347 | +0.557973 | +0.716948 | 1 | 1.00 | 142 | 8 |
| self_refine_verifier | 22.373823 | +0.636068 | -0.046132 | missed | -0.092597 | +1.079405 | +1.470615 | 1 | 1.00 | 144 | 17 |
| self_rag_critique | 22.459442 | +0.721687 | +0.039487 | missed | -0.008705 | +1.249047 | +1.523804 | 1 | 1.00 | 143 | 11 |
| self_ask_horizons | 22.487078 | +0.749323 | +0.067123 | missed | +0.008413 | +1.224889 | +1.619490 | 1 | 1.00 | 142 | 10 |
| debate_consensus | 22.697666 | +0.959911 | +0.277711 | missed | +0.004084 | +1.552121 | +2.055496 | 3 | 3.00 | 144 | 2 |
| rarr_attribution | 22.701313 | +0.963558 | +0.281358 | missed | -0.024339 | +1.559075 | +2.015527 | 1 | 1.00 | 140 | 9 |

## Best Candidate

- Candidate: `step_back_regime`
- Literature: [Step-Back Prompting](https://arxiv.org/abs/2310.06117)
- Change: Force the reflection stage to infer a higher-level regime first, then map it to horizon decisions.
- Aim: Improve medium and long-horizon coherence by reasoning at the regime level before local deltas.
- Expected range: `21.63-21.78`
- Actual raw path MSE: `21.484623`
- Delta vs baseline raw LLM: `-0.253133`
- Delta vs base TSM: `-1.241730`
- Delta vs ridge: `-0.935333`
- Outcome vs expectation: `beat`
- Run dir: `/Users/davidwilkinson/Desktop/ETS 2/runs/20260309_234219_249d62`
