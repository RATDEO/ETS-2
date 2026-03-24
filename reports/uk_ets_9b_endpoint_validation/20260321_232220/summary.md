# 9B Endpoint Validation Summary

## Objective
- Validate that the swapped 9B endpoint can serve the current UK `TSM+LLM-COT-RF-HDELTA` schema before launching full W1-W4 reruns.

## Endpoint Discovery
- Base URL: `http://192.168.1.140:9877/v1`
- `/models` response advertises one served model:
  - `qwen3.5-9b-ud-q4-k-xl`

## Direct Chat Probe
- Initial simple JSON probe succeeded earlier against the endpoint and resolved to model `qwen3.5-9b-ud-q4-k-xl`.
- Repeated direct chat-completion probes then failed consistently with:
  - `APIConnectionError('Connection error.')`
- Failure persisted for:
  - plain text completion
  - `response_format={"type":"json_object"}`

## Smoke Run
- Smoke run directory:
  - [/Users/davidwilkinson/Desktop/ETS 2/runs/20260321_231624_c4ef3c](/Users/davidwilkinson/Desktop/ETS%202/runs/20260321_231624_c4ef3c)
- Controls:
  - improved Huber + energy base
  - current `wide8` retrieval schema
  - model override `qwen3.5-9b-ud-q4-k-xl`
  - cache disabled
  - `llm.max_samples=8`

## Smoke Run Outcome
- The run completed structurally.
- However, the LLM path did not execute successfully:
  - all `8` LLM log rows were reflection rows only
  - all `8` had empty response content
  - the run log shows repeated `LLM reflection call failed ... Connection error`
- Metrics from the smoke subset:
  - `tsm_llm_subset` path MSE: `38.997112`
  - `TSM+LLM-COT-RF-HDELTA` path MSE: `38.997114`
- That means the 9B smoke run effectively collapsed to failed-reflection fallback behavior and is not a valid model-comparison result.

## Conclusion
- The current UK schema is not the blocker.
- The swapped 9B serving endpoint is not yet stable enough for valid full reruns.
- Full W1-W4 reruns should not be treated as scientific evidence until direct chat completions succeed reliably.

## Ready-To-Run Scripts
- Current-schema 9B rerun:
  - [/Users/davidwilkinson/Desktop/ETS 2/uk_ets/scripts/benchmark_effective_retrieval_w1_w4_huber_energy_9b.py](/Users/davidwilkinson/Desktop/ETS%202/uk_ets/scripts/benchmark_effective_retrieval_w1_w4_huber_energy_9b.py)
- Selective-uplift follow-up from a completed report:
  - [/Users/davidwilkinson/Desktop/ETS 2/uk_ets/scripts/run_selective_uplift_from_completed_report.py](/Users/davidwilkinson/Desktop/ETS%202/uk_ets/scripts/run_selective_uplift_from_completed_report.py)
