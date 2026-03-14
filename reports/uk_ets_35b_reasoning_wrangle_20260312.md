# UK ETS 35B Reasoning Endpoint Wrangle

Date: 2026-03-12

## Objective

Make the live `qwen3.5-35b-a3b-ud-q4-k-xl` reasoning endpoint at `http://192.168.1.140:9881/v1` usable inside the UK ETS `TSM+LLM-COT-RF-HDELTA` pipeline, even when it returns long reasoning traces instead of clean final JSON.

## What Was Wrong

The live endpoint behavior had drifted from the earlier successful 35B runs:

- It now exposes long `reasoning_content` and often leaves assistant `content` empty.
- `response_format=json_schema` is rejected by the server, so strict-schema requests are not reliable on this endpoint.
- Reflection and apply calls can end with `finish_reason=length` before a terminal JSON object appears.
- The old parser path treated these responses as unusable.
- The apply-stage parser could also pick up the prompt example JSON embedded inside the reasoning trace instead of the model's chosen adjustments.

## Fixes Applied

Code changes:

- `src/llm/refine.py`
  - Added reasoning-aware extraction for `reasoning_content`.
  - Added response-format disable policy support.
  - Added per-stage token budgets: `cot_rf.reflect_max_tokens` and `cot_rf.apply_max_tokens`.
  - Fixed `_call_llm` cache-key handling so suffix-specific calls do not overwrite each other.
  - Added reasoning-trace recovery for structured horizon guidance.
  - Added reasoning-trace recovery for horizon adjustments, preferring late-stage chosen values over quoted example JSON.
  - Added richer call metadata for finish reason, response model, and reasoning fallback usage.
- `src/llm/prompts.py`
  - Added a compact reasoning mode for structured reflection prompts.
  - Strengthened the no-loop / one-pass instruction for reasoning models.
- `tests/test_llm_guarded_rf_hdelta.py`
  - Added coverage for reasoning-trace guidance parsing.
  - Added coverage for apply-stage tail-value parsing.
  - Added coverage for response-format disable policy.

Config added:

- `uk_ets/config/uk_ets_llm_35b_reasoning_wrangle.yaml`

## Validation Runs

### Failed pre-fix style smoke

Run:

- `runs/20260312_131214_e8d3d4`

Behavior:

- Reflection returned `finish_reason=length`.
- `reasoning_content` was present, but the old path still failed reflection.

### First successful end-to-end reasoning smoke

Run:

- `runs/20260312_132102_28a061`

Settings:

- `compact_reasoning=true`
- `k_examples=3`
- `reflect_max_tokens=6144`
- `apply_max_tokens=2048`
- `response_format` disabled for the endpoint

Outcome:

- Reflection succeeded from reasoning-trace fallback.
- Apply also succeeded from reasoning-trace fallback.
- This proved the live reasoning endpoint can complete the project pipeline.

### Final validation after apply-parser fix

Run:

- `runs/20260312_133521_23cb2b`

Outcome:

- Reflection: success
- Apply: success
- Both calls still ended with `finish_reason=length`, but both were successfully recovered from `reasoning_content`.
- Final extracted adjustments were:
  - `h1 = 0.0`
  - `h5 = 0.5`
  - `h20 = 0.5`
  - `h30 = 0.5`

This run is the cleanest proof that the repaired parser no longer grabs the prompt example JSON and can instead recover the model's late-stage chosen values.

## Current Practical Conclusion

The 35B reasoning endpoint is now operational for the UK ETS refinement flow, but with an important caveat:

- It is not producing neat terminal JSON reliably.
- The working path currently depends on reasoning-trace extraction and conservative fallback parsing.

So the endpoint is now usable for experimentation and controlled project runs, but it should still be treated as a compatibility path rather than a clean native structured-output path.

## Next Improvements

If this path is kept active, the next best steps are:

- run a small multi-sample UK slice on the repaired reasoning path, not just a 1-sample smoke;
- harden the reflection fallback so freeze-horizon fields default to `sign=zero` and `magnitude=zero` more cleanly;
- add a dedicated reasoning-endpoint mode that logs a short extracted decision summary alongside the raw reasoning trace.
