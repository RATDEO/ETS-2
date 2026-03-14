# UK ETS 35B Retrieval Follow-Up Sweep

Generated: 2026-03-10T18:32:31.507015

## Baseline

- Baseline run: `/Users/davidwilkinson/Desktop/ETS 2/runs/20260309_234219_249d62`
- Baseline raw `TSM+LLM-COT-RF-HDELTA` path MSE: `21.484623`
- Reference `linear_ridge` path MSE: `22.419955`

## Planned Variants

| Candidate | Literature | Aim | Expected range |
|---|---|---|---|
| utility_score_retrieval | [Learning To Retrieve Prompts for In-Context Learning](https://arxiv.org/abs/2112.08633) | Choose examples that are not just hard, but also close to the current regime and recent enough to transfer cleanly. | 21.38-21.50 |
| utility_mmr_retrieval | [Active Example Selection for In-Context Learning](https://arxiv.org/abs/2211.04486) | Increase evidence coverage without sacrificing regime fit, especially at h20 and h30. | 21.34-21.48 |
| regime_filtered_mmr | [Large Language Models as Analogical Reasoners](https://arxiv.org/abs/2310.01714) | Make the model reason over truly analogous UK cases instead of mixing incompatible regimes inside the prompt. | 21.32-21.47 |
| agreement_routed_regime_mmr | [UnCert-CoT / Uncertainty of Thoughts](https://arxiv.org/abs/2503.15341) | Keep the current winner on easy cases while giving ambiguous h20/h30 cases more deliberate consensus search. | 21.28-21.45 |
| long_context_regime_mmr | [In-Context Learning with Long-Context Models](https://arxiv.org/abs/2405.00200) | Test whether the 35B model can exploit a wider UK example bank without collapsing into repetition. | 21.30-21.48 |

## Results

| Candidate | Actual | Vs baseline | Vs ridge | Outcome | h5 vs baseline | h20 vs baseline | h30 vs baseline | Reflect samples | Routed reflect calls | Unique reflect | Unique apply |
|---|---:|---:|---:|---|---:|---:|---:|---|---:|---:|---:|
| utility_score_retrieval | 21.764066 | +0.279444 | -0.655889 | missed | -0.103345 | +0.562944 | +0.905499 | 1 | 0 | 144 | 39 |
| utility_mmr_retrieval | 22.315308 | +0.830686 | -0.104647 | missed | -0.103228 | +1.316211 | +2.721577 | 1 | 0 | 144 | 28 |
| regime_filtered_mmr | 22.691136 | +1.206513 | +0.271180 | missed | -0.019970 | +1.869502 | +3.087341 | 1 | 0 | 143 | 38 |
| agreement_routed_regime_mmr | 22.817317 | +1.332695 | +0.397362 | missed | +0.011833 | +2.165402 | +3.335060 | 3 | 144 | 144 | 45 |
| long_context_regime_mmr | 22.938428 | +1.453806 | +0.518473 | missed | -0.013113 | +2.266882 | +3.705627 | 1 | 0 | 136 | 46 |

## Best Candidate

- Candidate: `utility_score_retrieval`
- Literature: [Learning To Retrieve Prompts for In-Context Learning](https://arxiv.org/abs/2112.08633)
- Change: Replace recent-high-error retrieval with a utility score combining similarity, correction value, and recency.
- Aim: Choose examples that are not just hard, but also close to the current regime and recent enough to transfer cleanly.
- Expected range: `21.38-21.50`
- Actual raw path MSE: `21.764066`
- Delta vs baseline raw LLM: `+0.279444`
- Delta vs ridge: `-0.655889`
- Run dir: `/Users/davidwilkinson/Desktop/ETS 2/runs/20260310_161502_568bf0`
