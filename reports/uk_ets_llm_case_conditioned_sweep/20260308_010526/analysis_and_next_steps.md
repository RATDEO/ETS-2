# UK ETS TSM+LLM Sweep Analysis

## What was tested

This sweep tested 10 UK `TSM+LLM-COT-RF-HDELTA` candidates across four hypothesis families:

1. Shorter teaching-example horizons (`365`, `270`, `180`, `120`, `90` days).
2. Smaller evidence sets (`k_examples=4`, `case_match_top_k=2`).
3. Tighter adjustment controls (`case_bound_scale=0.65`, `max_adjustment_pct=0.8`).
4. More conservative matching / freezing (`feature_window=12`, stricter sign agreement).

The sweep script is [sweep_uk_tsm_llm_case_conditioned.py](/Users/davidwilkinson/Desktop/ETS%202/uk_ets/scripts/sweep_uk_tsm_llm_case_conditioned.py). The candidate table is in [candidate_results.csv](/Users/davidwilkinson/Desktop/ETS%202/reports/uk_ets_llm_case_conditioned_sweep/20260308_010526/candidate_results.csv).

## Main findings

- Best raw candidate stayed the baseline `baseline_365_k6_tk3` at `22.618999` path MSE.
- That baseline still improves over base TSM `22.726353` by `-0.107354`, but it remains behind ridge `22.419955`.
- Shorter teaching horizons did not help. Ordering was:
  - `365` best
  - `180`
  - `120`
  - `270`
  - `90` worst
- Smaller evidence sets did not help on their own.
- Tighter global bounds did not help.
- Stricter sign consensus mostly turned the method into a near-noop.
- Shorter local regime matching windows also failed to beat the baseline.

The best full rerun is [20260308_022050_c9a217](/Users/davidwilkinson/Desktop/ETS%202/runs/20260308_022050_c9a217/config_resolved.yaml). Headline metrics are in [path_metrics.csv](/Users/davidwilkinson/Desktop/ETS%202/runs/20260308_022050_c9a217/results/path_metrics.csv).

## Interpretation

The earlier hypothesis that stale teaching examples were the main source of the remaining UK LLM error was not supported. The current best UK LLM path appears to benefit from a broader evidence pool, and the remaining issue is more likely horizon-specific adjustment design than example recency.

Evidence for that:

- The best raw candidate improved `h5` by `-0.148820` MSE and `h20` by `-0.152681`, but still worsened `h30` by `+0.058892`.
- The stricter consensus candidate froze nearly all horizons and collapsed to base TSM, which means the current stronger freeze rule is too blunt.
- The local-match and smaller-evidence variants reduced action diversity without producing better long-horizon behavior.

## Next hypotheses

These are the next highest-value improvements to test.

### 1. Horizon-specific structured reflection

Change:
- Replace free-text rules with strict per-horizon reflection output such as `sign`, `confidence`, `magnitude_class`, and `reason` for `h5`, `h20`, and `h30`.
- Keep apply-stage numeric only.

Why:
- Current sweep shows `144` unique rule texts but only `40-54` unique apply responses in most active candidates, which suggests the apply stage is compressing diverse reasoning into a small response template set.

Expected outcome:
- Preserve current `h5` and `h20` gains while reducing accidental `h30` coupling.
- Expected raw path MSE target: roughly `22.55` to `22.60`.

### 2. Long-horizon gated application

Change:
- Allow normal LLM action at `h5` and `h20`, but freeze `h30` unless matched-example evidence clears a stronger horizon-specific threshold than the short horizons.

Why:
- The best current candidate already shows the pattern: short/intermediate horizons improve while `h30` still drifts the wrong way.

Expected outcome:
- Keep most of the current gain while removing the long-tail drag.
- Expected raw path MSE target: roughly `22.57` to `22.62`.

### 3. Horizon-specific example matching

Change:
- Match teaching examples separately for `h5/h20/h30` instead of one shared case set, using forecast-anchor drift plus historical error signature for that horizon.

Why:
- Shorter global lookbacks failed, which suggests the problem is not generic recency but horizon relevance. The right examples for `h5` may not be the right examples for `h30`.

Expected outcome:
- Better sign stability at `h20/h30` without sacrificing the active short-horizon corrections.
- Expected raw path MSE target: roughly `22.54` to `22.60`.

## Operational conclusion

The current UK experimental default should remain the baseline case-conditioned `365/k6/top3` path. It is still the best tested `TSM+LLM` configuration in this sweep, and the full rerun reproduced the prior small raw gain over TSM.
