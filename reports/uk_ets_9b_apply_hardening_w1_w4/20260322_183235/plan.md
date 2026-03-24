# 9B Apply Hardening Sweep Plan

Generated: 2026-03-22T18:32:35.425535

## Objective
- Keep the improved Huber + energy base and wide8 retrieval fixed.
- Harden only the 9B apply stage, because the completed 9B benchmark showed apply-loop failures but no endpoint/schema transport failures.

## Baseline
- Completed 9B wide8 report: `reports/uk_ets_effective_retrieval_w1_w4_huber_energy_9b/20260322_002454/results.csv`.

## Candidates

| Candidate | Aim | Expected uplift vs 9B baseline | Change |
|---|---|---:|---|
| optional_verifier | Reduce apply-loop dead-ends caused by repeated verifier chaining while preserving numeric grounding. | 0.10% to 0.60% | Keep numeric tool forced but make the verifier optional; retain wide8 retrieval and strict JSON. |
| numeric_only_compact | Test whether the 9B model performs better when the apply stage is reduced to one deterministic tool and a shorter loop. | 0.20% to 1.00% | Disable the verifier and shrink the apply tool loop to numeric-only with max_tool_rounds=3. |
| optional_both_compact | Test whether the 9B model improves when it can skip tools entirely on easy cases without getting trapped in required tool loops. | -0.20% to 0.60% | Keep both tools enabled but make both optional and reduce max_tool_rounds to 3. |
