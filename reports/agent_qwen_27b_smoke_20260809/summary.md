# Qwen 27B bounded-refinement smoke comparison

**Decision:** do not promote the current Qwen policy. The systems smoke passed,
but the eight-origin quality result is negative and is not statistically
powered. Continue only with validation-selected gating or revised long-horizon
policy, then evaluate on newly frozen outer windows.

## Run identity

- Run: `runs/20260809_212216_1f78df`
- Code base: `3ff051630` plus the uncommitted review-completion fixes described below
- Python: 3.11.12
- Served model: `qwen3.6-27b-q4-k-m`
- Model size/format reported by the server: 27.32B parameters, GGUF Q4_K_M
- Smoke origins: 8, selected with seed 42
- Historical test boundary: 2026-03-04
- LLM origin range: 2025-07-17 through 2026-01-21
- Temperature: 0.0
- Cache hits: 0

The run reached the LAN server through a local loopback TCP forwarder because
macOS denied LAN sockets to the project Python 3.11 binary while allowing the
current Homebrew Python and `curl`. The upstream endpoint remained
`192.168.68.140:9881`; the model ID and responses came from that server.

## Same-origin comparison

The LLM rows below and all subset baselines use exactly the same eight origins.
Lower MSE and MAE are better.

| Model | Path MSE | Path MAE | Change in MSE vs DLinear |
|---|---:|---:|---:|
| Linear lasso | 62.831528 | 5.067264 | -5.44% |
| Linear ridge | 66.026629 | 5.225037 | -0.63% |
| DLinear base | 66.444298 | 5.088158 | baseline |
| Qwen-refined DLinear | 67.723674 | 5.092636 | **+1.93% worse** |

Per-origin path MSE improved on 4/8 origins and worsened on 4/8. The sample is
too small for a dependence-aware performance claim.

## Horizon comparison

| Horizon | DLinear MSE | Qwen MSE | MSE change |
|---|---:|---:|---:|
| H1 | 4.258080 | 4.258080 | unchanged |
| H5 | 6.803757 | 6.589782 | **3.14% better** |
| H20 | 119.272545 | 122.366814 | 2.59% worse |
| H30 | 126.903137 | 129.270456 | 1.87% worse |

The frozen H1 anchor was preserved exactly. The current prompt/policy improved
H5 but its positive long-horizon corrections increased overall path error.

## Full test-set baseline floor

These rows use all 144 test origins, so they must not be numerically compared
with the eight-origin Qwen subset as though sample sizes were equal. They show
which base model currently clears the strongest-simple-baseline requirement.

| Model | Full test path MSE |
|---|---:|
| Linear ridge | **22.399658** |
| Linear lasso | 24.172185 |
| Corrected DLinear | 26.375227 |
| Naive persistence | 32.003593 |

Ridge beats DLinear by 15.07% on the full test set. Qwen should not be promoted
on top of DLinear while the simple linear floor remains stronger.

## Reliability and cost

| Measure | Result |
|---|---:|
| Selected origins | 8 |
| Successful reflections | 8/8 |
| Successful apply stages | 8/8 |
| Parse success | 100% |
| Fallbacks | 0 |
| Retries | 0 |
| Underlying model requests | 32 (4/origin) |
| Numeric-tool invocations | 8 |
| Delta-verifier invocations | 8 |
| Prompt tokens | 93,768 |
| Completion tokens | 3,176 |
| Total tokens | 96,944 (12,118/origin) |
| Total model latency | 381.73 s |
| Mean latency per origin | 47.72 s |
| Median latency per origin | 48.18 s |
| P95 latency per origin | 48.88 s |

Every output was finite, the maximum absolute applied adjustment was 0.75%,
and H1 had exactly zero adjustment. The configured global cap was 1.0%.

## Comparison with the pulled review evidence

- The provisional 0.5B CPU smoke also slightly worsened its base
  (`25.280014` to `25.296493`, a 0.065% degradation). Its origins/base are not
  directly interchangeable with this run.
- The corrected historical selective policy showed only 0.09-0.10% macro
  uplift and its 30-origin block-bootstrap interval crossed zero.
- The newly served 27B model parses and follows the tool protocol much better
  than a systems failure, but capacity alone did not solve the quality problem.

## Review-completion fixes made in this work

- DLinear now uses replicate/endpoint padding rather than zero padding.
- Weighted horizon losses are normalized by effective weight sum.
- Cosine warm restarts use one fractional-epoch stepping convention.
- Training automatically restores the best validation state.
- New checkpoints store scheduler/environment metadata and load through the
  restricted PyTorch loader; explicitly trusted legacy files use an opt-in.
  The final NumPy-scalar serialization fix landed after this run completed and
  is verified by the regression suite, so future checkpoints get that guarantee.
- The historical simplified attention model is factually named
  `SimpleAttentionForecaster`, with `SimpleAutoformer` retained only as a
  compatibility alias.
- LLM logs now retain request count, token use, resolved endpoint, cache status,
  and wall latency, including all tool rounds.

Validation: `194 passed` under Python 3.11.

## Data and interpretation limits

- This is a systems smoke on eight historical origins, not promotion evidence.
- The data manifest records an ICAP proxy for ICE auctions because the actual
  ICE source is reCAPTCHA-gated.
- The bootstrap refresh recorded a Brent download failure, although an existing
  local Brent series through 2026-03-02 was loaded by this run.
- The selected 30-day paths can overlap; eight observations cannot support a
  meaningful dependence-aware confidence interval.
- The next quality test must freeze new rolling-origin windows and select any
  prompt, long-horizon bounds, or gate solely on older validation periods.

## Reproduction

```bash
export OPENAI_BASE_URL="http://192.168.68.140:9881/v1"
export OPENAI_API_KEY="<local-server-key>"

/opt/homebrew/bin/python3.11 -m src.run_experiment \
  --config uk_ets/config/uk_ets_qwen_27b_local_smoke.yaml \
  --data-dir uk_ets/Data_auto_uk
```

If macOS denies LAN access to Python 3.11 on this host, use an approved local
forwarder and point `OPENAI_BASE_URL` at its loopback port.
