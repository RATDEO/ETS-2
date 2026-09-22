# Qwen 27B refinement completion report

## Decision

Use `linear_ridge+LLM-COT-RF-HDELTA_base_path_slope_gate` as the selected
refinement output. The previously unstable LLM stage now has a leakage-free,
first-class policy that improved path MSE on both development batches and on a
disjoint holdout selected before its outcomes were inspected.

The result is promising rather than conclusive: the untouched holdout contains
only 16 origins, their 30-day target paths overlap, and the paired tests do not
reach a conventional 5% threshold. Raw Qwen should remain a diagnostic
comparator, not the default decision path.

## Frozen holdout result

- Run: `runs/20260809_233427_cdabf1`
- Base: linear ridge
- Served model: `qwen3.6-27b-q4-k-m`
- Origins: 16, seed `20261954`
- Origin dates: 2025-07-14 through 2026-01-06
- No origin overlaps the earlier 8-origin DLinear smoke or either 16-origin
  ridge development batch.
- The gate was frozen before this seed's outcomes were evaluated.

Lower MSE is better.

| Same-origin model | Path MSE | Change vs ridge | Origin outcomes |
|---|---:|---:|---:|
| Linear ridge | 11.132485 | baseline | — |
| Raw Qwen refinement | **10.691408** | **3.962% better** | 9 better / 6 worse / 1 equal |
| Frozen slope-gated Qwen | **10.764087** | **3.309% better** | 8 better / 5 worse / 3 equal |

The gate applied Qwen on 14/16 origins. For the other two, the saved gated
prediction is byte-for-byte equal to ridge; every accepted row is byte-for-byte
equal to the raw Qwen row.

### Horizon breakdown

| Horizon | Ridge MSE | Raw Qwen MSE | Gated Qwen MSE | Gated change |
|---|---:|---:|---:|---:|
| H1 | 0.542734 | 0.542734 | 0.542734 | unchanged |
| H5 | 2.479077 | 2.425944 | 2.425944 | **2.143% better** |
| H20 | 8.336861 | 7.851537 | 8.010405 | **3.916% better** |
| H30 | 40.820056 | 39.668406 | 39.632570 | **2.909% better** |

The configured H1 freeze was preserved exactly. All predictions were finite;
the largest absolute Qwen adjustment was 0.75%, below the 1.0% cap.

## Cross-batch comparison

The gate admits Qwen only when the observable ridge path has a non-negative
H1-to-H30 percentage slope. It uses neither realized outcomes nor the LLM's
self-reported confidence at forecast time.

| Batch | Status | Origins | Ridge MSE | Raw Qwen | Raw change | Gated Qwen | Gated change |
|---|---|---:|---:|---:|---:|---:|---:|
| `20260809_222348_481284` | development A | 16 | 25.827881 | 25.680803 | +0.569% | 25.715598 | **+0.435%** |
| `20260809_231234_f46753` | development B | 16 | 41.115495 | 42.083585 | -2.355% | 40.740279 | **+0.913%** |
| `20260809_233427_cdabf1` | frozen holdout | 16 | 11.132485 | 10.691408 | +3.962% | 10.764087 | **+3.309%** |

Across the 48 disjoint ridge origins, the descriptive aggregate is:

- Raw Qwen: 26.151932 versus ridge 26.025287, **0.487% worse**.
- Gated Qwen: 25.739988 versus ridge 26.025287, **1.096% better**.

The 48-origin aggregate is not an untouched significance test because the first
32 origins informed gate selection. It does show why unconditional Qwen is not
the selected policy and that the gate improved every observed ridge batch.

## Uncertainty

On the 16-origin holdout, a paired t-test gives `p=0.069` for raw Qwen and
`p=0.103` for gated Qwen; Wilcoxon gives `p=0.115` and `p=0.179`, respectively.
An origin-resampling percentile bootstrap gives a 95% relative-gain interval of
`[+0.38%, +7.67%]` for raw and `[-0.16%, +7.68%]` for gated. Those bootstrap
intervals are not dependence-safe because several 30-day forecast paths
overlap. The evidence supports continued frozen-policy evaluation, not a broad
production accuracy claim yet.

## Reliability and cost on the frozen holdout

| Measure | Result |
|---|---:|
| Successful reflect/apply outputs | 16/16 |
| JSON parse success | 100% |
| Fallbacks | 0 |
| Retries | 0 |
| Underlying model requests | 64 (4/origin) |
| Prompt tokens | 185,672 |
| Completion tokens | 6,148 |
| Total tokens | 191,820 (11,989/origin) |
| Aggregate model latency | 760.52 s |
| Mean model latency | 47.53 s/origin |

The current first-class gate is applied after generation so its rejected rows
still incur Qwen cost. Moving the same observable check ahead of the LLM call is
a safe future cost optimization; it is not needed for forecast equivalence.

## What changed

- Ridge, the strongest corrected full-test baseline (path MSE `22.399658`), is
  now the Qwen base rather than DLinear (`26.375227`).
- A bounded base-path slope reject mask and first-class gated result were added
  to `src/run_experiment.py`, including persisted masks/predictions and audit
  metadata.
- Joint piecewise-linear validation calibration was implemented for optional
  experiments, with scale bounds, ridge stabilization, and fixed anchors.
- The runnable Qwen config is locked to the frozen gate by default; exploratory
  calibration and blend selection are retained but disabled.
- LLM logs now expose complete request, token, cache, endpoint, latency, tool,
  and retry accounting.
- Regression coverage was added for gate semantics and interpolated calibration.

The earlier review-completion fixes also remain in place: correct DLinear edge
padding, normalized weighted loss, consistent scheduler stepping, best-state
restoration, safe checkpoint metadata/loading, and factual attention-model
naming.

## Verification

- Full suite: `197 passed in 28.48s` under Python 3.11.
- `git diff --check`: clean.
- Final prediction artifact:
  `runs/20260809_233427_cdabf1/predictions/linear_ridge+LLM-COT-RF-HDELTA_pred_test_subset.npz`
- Final metrics:
  `runs/20260809_233427_cdabf1/results/path_metrics.csv`
- Gate audit:
  `runs/20260809_233427_cdabf1/llm/base_path_slope_gate_linear_ridge_LLM-COT-RF-HDELTA.json`
- Complete call log:
  `runs/20260809_233427_cdabf1/llm/logs/llm_calls.jsonl`

## Reproduction

The endpoint and credential remain runtime-only; neither is stored in the
config.

```bash
export OPENAI_BASE_URL="http://192.168.68.140:9881/v1"
export OPENAI_API_KEY="<local-server-key>"

/opt/homebrew/bin/python3.11 -m src.run_experiment \
  --config uk_ets/config/uk_ets_qwen_27b_refinement_calibrated.yaml \
  --data-dir uk_ets/Data_auto_uk
```

If macOS denies LAN sockets to Python 3.11 on this host, use an approved
loopback forwarder and point `OPENAI_BASE_URL` at that local port.
