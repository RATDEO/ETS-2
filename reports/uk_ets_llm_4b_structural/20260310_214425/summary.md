# UK ETS 4B Structural Sweep

Generated: 2026-03-10T22:58:09.244702

## Checkpoint

- Base checkpoint run: `/Users/davidwilkinson/Desktop/ETS 2/runs/20260310_195100_69e04e`
- Checkpoint raw `TSM+LLM-COT-RF-HDELTA` path MSE: `22.198106`
- Reference `linear_ridge` path MSE: `22.419955`

## Candidate Results

| Candidate | Change | Aim | Expected | Actual | Vs checkpoint | Outcome vs expected | h5 vs checkpoint | h20 vs checkpoint | h30 vs checkpoint | Nonzero deltas |
|---|---|---|---|---:|---:|---|---:|---:|---:|---:|
| control_recent_high_error_k4 | Re-run the current 4B winner unchanged as a control on the new pipeline code path. | Confirm the structural runner changes preserve the existing recent-high-error baseline before adding retrieval/context layers. | 22.14-22.26 | 22.198106 | +0.000000 | within | +0.000000 | +0.000000 | +0.000000 | 3767 |
| recent_high_error_k4_example_exo | Keep recent high-error selection, but attach UK exogenous summaries to each teaching example. | Let the 4B model compare hard historical fixes with the matching UK auction and ICAP state instead of only prices and forecasts. | 22.08-22.23 | 22.198106 | +0.000000 | within | +0.000000 | +0.000000 | +0.000000 | 3767 |
| utility_mmr_augmented_uk_context_k4 | Use utility-MMR retrieval with UK-augmented case profiles and example exogenous summaries. | Test whether a more diverse but still utility-ranked UK retrieval set beats plain recent-high-error on the smaller 4B model. | 22.06-22.22 | 22.210074 | +0.011968 | within | -0.025721 | +0.078258 | +0.018432 | 3720 |
| event_similarity_uk_context_k4 | Switch retrieval to UK event similarity using auction, volume, and ICAP print-state features. | Select demonstrations by market-state similarity rather than only forecast error, while keeping the 4B prompt compact. | 22.10-22.26 | 22.279624 | +0.081518 | missed | +0.070055 | +0.138536 | +0.161769 | 3605 |
| event_similarity_uk_context_k4_example_exo | Combine UK event-similarity retrieval with exogenous summaries on the retrieved teaching examples. | Align both retrieval and prompt evidence around the same UK market-state cues instead of mixing price-only retrieval with context-free examples. | 22.04-22.20 | 22.279624 | +0.081518 | missed | +0.070055 | +0.138536 | +0.161769 | 3605 |
| similarity_error_hybrid_augmented_uk_context_k4 | Use similarity-error hybrid retrieval, but augment the case profile with UK exogenous state and include exogenous summaries on examples. | Keep the hard-case bias that has been working, while making similarity matching aware of auctions and ICAP event structure. | 22.02-22.18 | 22.316951 | +0.118846 | missed | +0.026532 | +0.116360 | +0.473647 | 3717 |

## Best Candidate

- Candidate: `control_recent_high_error_k4`
- Change: Re-run the current 4B winner unchanged as a control on the new pipeline code path.
- Aim: Confirm the structural runner changes preserve the existing recent-high-error baseline before adding retrieval/context layers.
- Expected range: `22.14-22.26`
- Actual raw path MSE: `22.198106`
- Delta vs checkpoint raw LLM: `+0.000000`
- Delta vs base TSM: `-0.528247`
- Delta vs ridge: `-0.221849`
- Outcome vs expectation: `within`
- Run dir: `/Users/davidwilkinson/Desktop/ETS 2/runs/20260310_214425_026870`
