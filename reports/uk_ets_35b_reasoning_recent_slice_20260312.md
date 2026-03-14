# UK ETS 35B Reasoning Recent-Slice Validation

Date: 2026-03-12

## Objective

Validate the repaired `qwen3.5-35b-a3b-ud-q4-k-xl` reasoning-mode integration on more than a 1-sample smoke test, using the most recent UK ETS test windows.

## Setup

Base config:

- `uk_ets/config/uk_ets_llm_35b_reasoning_wrangle.yaml`

Run:

- `runs/20260312_134647_185e5e`

Overrides:

- `llm.max_samples = 20`
- `llm.subset.scope = recent_tail`
- `llm.subset.recent_tail_min_samples = 20`
- `llm.subset.strategy = first`
- `llm.cot_rf.compact_reasoning = true`
- `llm.cot_rf.k_examples = 3`
- `llm.cot_rf.reflect_max_tokens = 6144`
- `llm.cot_rf.apply_max_tokens = 2048`
- `llm.blend_grid.enabled = false`

## Result

The endpoint remained operational across the completed portion of the recent slice, but throughput was too slow to justify waiting for all 20 windows.

Completed windows before manual stop:

- `6`

Completed calls logged:

- `6` reflection calls
- `6` apply calls
- `12` total successful calls

## Stability Stats

Reflection stage:

- Successes: `6/6`
- `finish_reason = stop`: `3`
- `finish_reason = length`: `3`
- `reasoning_content` present: `6/6`
- Recovered from reasoning fallback: `3/6`
- Needed retry beyond first attempt: `2/6`

Apply stage:

- Successes: `6/6`
- `finish_reason = length`: `6/6`
- `reasoning_content` present: `6/6`
- Recovered from reasoning fallback: `6/6`
- Needed retry beyond first attempt: `2/6`

## Interpretation

This is enough to treat the 35B reasoning endpoint as operational for the project pipeline:

- Reflection is mixed-mode.
  - Sometimes it returns clean structured JSON in assistant `content`.
  - Sometimes it returns only usable reasoning traces, which the new parser now recovers.
- Apply is reasoning-trace native on this endpoint.
  - In the completed recent windows, every apply call ended with `finish_reason=length`.
  - The repaired parser still extracted valid horizon adjustments every time.

The key practical conclusion is that the integration no longer depends on the endpoint behaving like a normal strict-JSON model. It can now tolerate the actual live behavior of the reasoning server.

## Caveat

This run was manually stopped because the endpoint is slow in reasoning mode and the objective here was compatibility validation, not a full performance benchmark.

For current use, the right framing is:

- The 35B reasoning endpoint is now wrangled successfully.
- It is stable enough for controlled experiments.
- It is still too slow and too reasoning-heavy to treat as a clean structured-output backend.
