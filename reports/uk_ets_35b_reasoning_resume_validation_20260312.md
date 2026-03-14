# UK ETS 35B Reasoning Resume Validation

Date: 2026-03-12

## Objective

Validate the new per-window LLM checkpoint/resume flow end to end on the live
`qwen3.5-35b-a3b-ud-q4-k-xl` reasoning endpoint.

## Setup

Config:

- `uk_ets/config/uk_ets_llm_35b_reasoning_wrangle.yaml`

Shared overrides:

- `llm.max_samples = 2`
- `llm.subset.scope = recent_tail`
- `llm.subset.recent_tail_min_samples = 2`
- `llm.subset.strategy = first`
- `llm.cot_rf.compact_reasoning = true`
- `llm.cot_rf.k_examples = 3`
- `llm.cot_rf.reflect_max_tokens = 6144`
- `llm.cot_rf.apply_max_tokens = 2048`
- `llm.blend_grid.enabled = false`

## Phase 1: Interrupted run

Run:

- `runs/20260312_143815_bb948d`

Behavior:

- First window completed successfully.
- Checkpoint written:
  - `runs/20260312_143815_bb948d/llm/checkpoints/TSM_LLM-COT-RF-HDELTA/141.json`
- Logged LLM calls before interruption:
  - `2` total (`1` reflect + `1` apply)
- Run was then manually interrupted during the next window.

## Phase 2: Resume run

Run:

- `runs/20260312_144514_e38099`

Override:

- `llm.resume_from_run_dir = runs/20260312_143815_bb948d`

Observed behavior:

- Log line:
  - `Resumed 1/2 LLM samples from checkpoint`
- New LLM calls written:
  - `2` total (`1` reflect + `1` apply)
- New checkpoint directory contains:
  - `141.json`
  - `142.json`

Checkpoint metadata:

- `141.json`
  - `resumed_from_checkpoint = true`
- `142.json`
  - fresh completion in the resume run

## Conclusion

The live resume path works as intended:

- completed windows from an interrupted reasoning-mode run are reused;
- the resumed run only calls the endpoint for unfinished windows;
- checkpoint state is copied forward into the new run directory so the resumed run is self-contained.

This is enough to treat long 35B reasoning runs as resumable instead of all-or-nothing.
