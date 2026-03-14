# Weekly Log Summary — 3 Mar to 10 Mar 2026

## Scope

This note summarises the main findings from the project logs, run reports, and experiment summaries produced over the last week.

## Executive summary

The week was dominated by a successful pivot from the earlier EU ETS ceiling to a more promising UK ETS workflow.

Main takeaways:

- The earlier EU ETS postmortem confirmed that honest, leakage-free LLM gains were real but small, which justified shifting effort toward UK ETS.
- The first UK ETS full runs showed that a simple `linear_ridge` baseline was initially much stronger than both DLinear and early LLM overlays.
- The first meaningful UK ETS LLM progress came from case-conditioned HDELTA prompting, then improved further through literature-backed prompt design.
- The major breakthrough this week was the 35B UK ETS stack with recent high-error teaching examples, which pushed raw path MSE down to `21.737755`.
- That result clearly beat both base `tsm` (`22.726353`) and `linear_ridge` (`22.419955`).
- Follow-up sweeps showed that **recency and prompt decomposition help**, while broader lookbacks, horizon-specific matching, and over-complicated horizon guards usually hurt.
- The validation blend selector still lags behind the raw LLM frontier: the raw 35B forecast is often best, even when blend tuning prefers conservative weights.

## Chronology of experiments

### 1. 3 Mar: EU ETS postmortem set the direction

Source reviewed:
- `docs/PROJECT_EVALUATION_AND_FUTURE_DIRECTIONS.md`

What changed:
- The project formally accepted that EU ETS was close to the limit of the current method.
- After leakage fixes, the best honest EU ETS LLM gain was modest rather than transformative.

Key finding:
- The best clean EU ETS result was only a small improvement over the base TSM, while a simple linear baseline remained highly competitive.

Practical consequence:
- The team shifted toward UK ETS, where there was a better chance that prompt-driven correction could add value.

### 2. 6 Mar: UK ETS pivot and first full benchmark picture

Source reviewed:
- `docs/UK_ETS_PIVOT_ANALYSIS_AND_NEXT_STEPS.md`

What was built/tested:
- A separate UK ETS pipeline and automated data flow.
- Full-density UK ETS runs with `Autoformer`, `DLinear`, `linear_ridge`, and early CoT-RF overlays.

Key findings:
- `linear_ridge` established the initial UK ETS benchmark at `22.419955` path MSE.
- Tuned `DLinear` was much worse at `29.78`.
- Early LLM overlays did not help enough; ridge-backed or DLinear-backed blends still lost to ridge.

Interpretation:
- UK ETS was not immediately easier than EU ETS.
- At this stage, the story was still “simple linear model wins.”

### 3. 8 Mar: Case-conditioned sweep showed baseline prompt design was still fragile

Sources reviewed:
- `reports/uk_ets_llm_case_conditioned_sweep/20260308_010526/summary.md`
- `reports/uk_ets_llm_case_conditioned_sweep/20260308_010526/analysis_and_next_steps.md`

What was tested:
- Shorter teaching lookbacks (`270`, `180`, `120`, `90` days).
- Smaller evidence sets.
- Tighter bounds.
- Stricter sign-consensus and local matching variants.

Key findings:
- Best candidate stayed the original broad case-conditioned baseline: `22.618999`.
- Shorter lookbacks did **not** help in this early sweep.
- Stricter sign consensus mostly collapsed into a near-no-op.
- The residual weakness was mainly at the long horizon (`h30`).

Interpretation:
- The problem was not simply “examples are too stale.”
- Horizon-specific control and application logic looked more important than generic recency tightening.

### 4. 8 Mar: Targeted tranche 2 improved long-horizon control, but only modestly

Source reviewed:
- `reports/uk_ets_llm_targeted_tranche2/20260308_184246/summary.md`

What was tested:
- `coherence_guard`
- `baseline_control`
- `horizon_specific_matching`
- `horizon_specific_plus_coherence`

Key findings:
- Best result: `coherence_guard` at `22.573653`.
- This was a real step forward from `22.618999`, but still behind `linear_ridge`.
- Horizon-specific matching did not deliver the hoped-for upside.

Interpretation:
- Better coherence control helped more than more elaborate per-horizon matching.
- The strongest gains were still coming from reducing bad medium/long-horizon corrections rather than increasing activity.

### 5. 9 Mar: Literature-backed prompt sweep produced the first ridge-beating UK LLM result

Sources reviewed:
- `reports/uk_ets_llm_literature_fast/20260309_011352/summary.md`
- `reports/uk_ets_llm_literature_fast/20260309_011352/final_analysis.md`

What was tested:
- `least_to_most`
- `least_to_most_pot`
- `active_prompt_recent_high_error`
- `self_consistency_k3`
- `program_of_thought_apply`
- several other literature-inspired prompt variants

Key findings:
- Winner: `least_to_most_pot` at `22.418530`.
- This narrowly beat `linear_ridge` (`22.419955`).
- The gain came mainly from better `h20` and `h30` behavior, not from short-horizon gains.

Interpretation:
- The first real UK ETS breakthrough came from **better decomposition**, not more complexity.
- Sequential reasoning plus a more numeric apply stage worked better than extra memory or sampling alone.

### 6. 9 Mar: 35B follow-up produced the strongest result of the week

Sources reviewed:
- `reports/uk_ets_35b_followup_20260309.md`
- `reports/uk_ets_llm_35b_followup/20260309_145251/summary.md`

What changed:
- The best UK prompt stack was moved onto the 35B endpoint.
- Follow-up variants focused on recent high-error teaching examples and small selection changes.

Key findings:
- Initial 35B raw result: `22.143373`.
- Best targeted follow-up: `recent_high_error_180` at `21.737755`.
- That improved by:
  - `-0.988597` vs base `tsm`
  - `-0.682200` vs `linear_ridge`
  - `-0.405618` vs the earlier 35B checkpoint
- `recent_high_error_180_sc3` and `recent_high_error_180_k8` were also strong, but both trailed the plain `recent_high_error_180` winner.

Interpretation:
- This was the clearest signal all week that the UK ETS task does respond to the right LLM recipe.
- The best lever was **recent high-error example selection**, not added aggregation or larger evidence sets.

### 7. 9 Mar: Horizon-specific and recent-window follow-ups clarified what not to do

Sources reviewed:
- `reports/uk_ets_llm_35b_horizon_followup/20260309_180319/summary.md`
- `reports/uk_ets_llm_35b_recent_window/20260309_193126/summary.md`

What was tested:
- h5-specific shrink/guard rules
- horizon-specific matching
- shorter and longer recent windows (`120`, `270`, `365` days)

Key findings:
- Horizon-specific follow-up mostly missed expectations; the best of that group was `21.828217`, still worse than the `21.737755` baseline.
- `recent_high_error_120` was close at `21.751460`, but still slightly worse than `180` days.
- Longer windows (`270`, `365`) were clearly worse.

Interpretation:
- The sweet spot was not “as recent as possible” or “as broad as possible.”
- Around `180` days looked like the best trade-off between locality and evidence coverage.
- Horizon-specific matching was more harmful than helpful in this regime.

### 8. 9 Mar: Blend-selection fix improved calibration logic, but raw forecasts still led

Source reviewed:
- `reports/uk_ets_35b_followup_20260309.md`

What changed:
- Blend tuning was modified to allow recent-tail validation selection instead of full-validation tuning only.

Key findings:
- Previous tuning always collapsed to `w=0.00`.
- Recent-tail tuning switched the selected 35B blend to `w=1.00`.
- Best recent-tail-selected ramp blend improved to `22.335848`.
- Raw 35B forecast still remained better at `22.143373` in that report, and later targeted runs improved raw performance further to `21.737755`.

Interpretation:
- Calibration is now directionally more sensible.
- But the best UK ETS results are still coming from the raw LLM-corrected forecast, not the tuned blend.

### 9. 10 Mar: Prompt-style micro-sweep around the new 35B setup

Sources reviewed:
- `runs/20260310_000350_ae6ed8/run_20260310_000350.log`
- `runs/20260310_002543_0788b3/run_20260310_002543.log`
- `runs/20260310_004631_fa08f9/run_20260310_004631.log`
- `runs/20260310_010806_f8d995/run_20260310_010806.log`
- `runs/20260310_013116_610e74/run_20260310_013116.log`
- `runs/20260310_015255_fe3d9e/run_20260310_015255.log`

Common setup:
- 35B endpoint
- `recent_high_error` example selection
- `lookback_days=180`
- structured horizon reflection + case-conditioned HDELTA guards

Finished runs:

| Run | Reasoning style | Apply style | Path MSE | Readout |
|---|---|---|---:|---|
| `20260310_004631_fa08f9` | `react_evidence` | `verifier_program` | `21.808505` | Best finished run on 10 Mar |
| `20260310_000350_ae6ed8` | `skeleton` | `minimal` | `21.931373` | Strong, but behind the Mar 9 best |
| `20260310_013116_610e74` | `self_rag` | `citation_bounded` | `22.459442` | Beat TSM, but lost to ridge |
| `20260310_002543_0788b3` | `self_ask` | `program_of_thought` | `22.487078` | Clear regression vs best frontier |
| `20260310_010806_f8d995` | `rarr_attribution` | `citation_bounded` | `22.701313` | Nearly no gain over TSM |

Incomplete run:
- `20260310_015255_fe3d9e`: `tree_of_thought` + `verifier_program` with 3 reflection samples. The log stops at `130/144` processed samples, so there is no final metric yet.

Interpretation:
- Prompt style still matters a lot even after the move to the stronger 35B + recent-high-error setup.
- `react_evidence` + `verifier_program` was the best 10 Mar finished combination.
- None of the 10 Mar completed prompt variants beat the 9 Mar `recent_high_error_180` frontier of `21.737755`.

## Overall findings from the week

### What clearly worked

1. **Pivoting to UK ETS was justified**
   - UK ETS produced a larger upside for prompt-based correction than the mature EU ETS setup.

2. **Recent high-error retrieval worked**
   - The strongest gains came from teaching the model with recent hard UK cases rather than generic similarity alone.

3. **Better decomposition worked**
   - `least_to_most`, `program_of_thought`, and later strong structured prompt/application pairings improved medium and long horizons.

4. **35B scale helped**
   - Moving from the 4B frontier to the 35B endpoint produced a real step-change in raw path MSE.

5. **Structured coherence guards helped more than exotic matching**
   - Long-horizon mistakes were better managed by guardrails and coherent application logic than by increasingly complex matching schemes.

### What did not work reliably

1. **Longer or broader example windows**
   - `270` and `365` day recent-high-error windows were worse than `180`.

2. **Horizon-specific matching**
   - It repeatedly underperformed and often damaged `h20/h30`.

3. **Heavy extra aggregation**
   - Self-consistency and wider evidence sets sometimes stayed competitive, but they did not beat the simpler best candidate.

4. **Validation-selected blends as final answer**
   - Blend selection improved, but raw LLM outputs still define the frontier.

## Current frontier at the end of the week

Best result identified in the reviewed logs/reports:

- **Run**: `20260309_151421_2aca41`
- **Configuration**: 35B UK ETS, `recent_high_error_180`
- **Method**: `TSM+LLM-COT-RF-HDELTA`
- **Raw path MSE**: **`21.737755`**

Reference benchmarks:

- Base `tsm`: `22.726353`
- `linear_ridge`: `22.419955`
- First ridge-beating 4B literature-style result: `22.418530`
- Earlier 35B checkpoint: `22.143373`

## Recommended next step

Based on the last week of logs, the most sensible immediate default is:

- keep the `recent_high_error_180` 35B raw configuration as the working frontier,
- treat `react_evidence` + `verifier_program` as the strongest 10 Mar prompt-style follow-up,
- avoid spending more time on horizon-specific matching or long-window retrieval,
- and only revisit blending after the raw frontier stops moving.

## Files reviewed

- `docs/PROJECT_EVALUATION_AND_FUTURE_DIRECTIONS.md`
- `docs/UK_ETS_PIVOT_ANALYSIS_AND_NEXT_STEPS.md`
- `reports/uk_ets_35b_followup_20260309.md`
- `reports/uk_ets_llm_case_conditioned_sweep/20260308_010526/summary.md`
- `reports/uk_ets_llm_case_conditioned_sweep/20260308_010526/analysis_and_next_steps.md`
- `reports/uk_ets_llm_targeted_tranche2/20260308_184246/summary.md`
- `reports/uk_ets_llm_literature_fast/20260309_011352/summary.md`
- `reports/uk_ets_llm_literature_fast/20260309_011352/final_analysis.md`
- `reports/uk_ets_llm_35b_followup/20260309_145251/summary.md`
- `reports/uk_ets_llm_35b_horizon_followup/20260309_180319/summary.md`
- `reports/uk_ets_llm_35b_recent_window/20260309_193126/summary.md`
- `runs/20260310_000350_ae6ed8/run_20260310_000350.log`
- `runs/20260310_002543_0788b3/run_20260310_002543.log`
- `runs/20260310_004631_fa08f9/run_20260310_004631.log`
- `runs/20260310_010806_f8d995/run_20260310_010806.log`
- `runs/20260310_013116_610e74/run_20260310_013116.log`
- `runs/20260310_015255_fe3d9e/run_20260310_015255.log`
