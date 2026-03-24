# 9B Wide8 Benchmark Plan

## Objective
- Re-run the current best 4B UK live schema on the swapped 9B endpoint.
- Hold everything else fixed: improved Huber + energy base, wide effective retrieval, same live heuristic policy.

## Endpoint Controls
- Model label forced to `qwen3.5-9b-ud-q4-k-xl`.
- Cache disabled to prevent stale 4B reuse on the same base URL.

## Comparison Baseline
- Last completed 4B current-schema benchmark: `effective_retrieval_wide8`.
