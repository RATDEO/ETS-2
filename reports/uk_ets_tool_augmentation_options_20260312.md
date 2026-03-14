# UK ETS 4B Tool Augmentation Options

Date: 2026-03-12

## Scope

This note evaluates which external tools are worth wiring into the current UK ETS non-reasoning 4B LLM refinement path.

Current live baseline:
- Base forecaster: UK `DLinear` / `TSM`
- LLM endpoint: `qwen3-vl-4b-gpu` on port `9877`
- Best current 4B config: [uk_ets_llm_4b_feedback_recent_high_error_k4.yaml](/Users/davidwilkinson/Desktop/ETS%202/uk_ets/config/uk_ets_llm_4b_feedback_recent_high_error_k4.yaml)
- Main failure mode: the LLM helps mostly at `h20/h30`, but the gain is regime-dependent and not robust across older folds

## Live Endpoint Check

The local `9877` endpoint supports native OpenAI-style function calling in non-reasoning mode.

Observed locally on 2026-03-12:
- `finish_reason='tool_calls'` on the first assistant turn
- valid JSON tool arguments
- successful second-turn final answer after sending the tool result back

That means we do not need the slow reasoning-mode workaround to add tools. We can use a narrow function-calling loop around the current 4B model.

Relevant implementation reference:
- Qwen function-calling docs: https://qwen.readthedocs.io/en/latest/framework/function_call.html

## Literature Signals

The most relevant literature points in the same direction:

- Tool use helps when the model is bad at arithmetic, retrieval, or factual access.
  Source: Toolformer, https://arxiv.org/abs/2302.04761
- Interleaving reasoning with external actions is useful when answers depend on outside information.
  Source: ReAct, https://arxiv.org/abs/2210.03629
- Offloading the solving step to a program/runtime improves numerical reliability.
  Source: PAL, https://arxiv.org/abs/2211.10435
- Example selection and retrieval quality materially change ICL performance.
  Source: Active Example Selection for In-Context Learning, https://arxiv.org/abs/2211.04486
- Forecasting papers that do well with LLMs usually rely on stronger time-series representations, not just generic CoT.
  Sources: Time-LLM, https://arxiv.org/abs/2310.01728 ; Chronos, https://arxiv.org/abs/2403.07815
- Forecasting agents benefit from structured event databases and domain APIs, not only free text.
  Source: MIRAI, https://arxiv.org/abs/2407.01231
- Selective retrieval is often better than always retrieving.
  Source: Self-Routing RAG, https://arxiv.org/abs/2504.01018
- Raw LLM forecasting remains brittle without domain-specific structure and calibration.
  Source: Large Language Models Are Zero-Shot Time Series Forecasters, https://arxiv.org/abs/2310.07820

## Recommended Tool Candidates

### 1. Numeric Analytics Tool

What it does:
- Computes deterministic quantities from the current window and base forecast:
- spot-to-horizon drift
- 5d/20d momentum
- volatility
- z-scores
- spread changes
- allowed adjustment caps

Why it is useful:
- The 4B model is strongest when it only has to choose direction and regime, not do arithmetic in text.
- PAL and Toolformer both support offloading calculations to a runtime.

Expected impact:
- Highest near-term value
- Likely benefit: `0.05-0.15` path MSE on the current 4B branch
- Most likely gain location: better `h20/h30` sizing, fewer malformed or oversized deltas

Implementation cost:
- Low

Verdict:
- Build first

### 2. Structured Historical Case Retrieval Tool

What it does:
- Given the current history, base forecast, and exogenous state, returns top historical analogue windows with:
- base forecast
- truth
- realized long-horizon error
- hindsight lesson
- regime tag

Why it is useful:
- The current pipeline already lives or dies on retrieval quality.
- A callable retrieval tool is better than forcing one static example set into every prompt.

Expected impact:
- Likely benefit: `0.05-0.20` path MSE
- Most likely gain location: regime-local `h20/h30`

Implementation cost:
- Medium

Verdict:
- Build second

### 3. 35B Teacher Memory Tool

What it does:
- Queries archived successful 35B correction cases and returns:
- similar teacher examples
- teacher delta profile
- concise hindsight summary

Why it is useful:
- We already have better 35B trajectories and rule artifacts.
- This is a cheap way to distill 35B behavior into the 4B path without needing live 35B access.

Expected impact:
- Likely benefit: `0.05-0.20` path MSE if teacher examples are filtered tightly
- Main gain location: long horizons

Implementation cost:
- Medium

Verdict:
- Very promising

### 4. Auction and Supply Calendar Tool

What it does:
- Returns future known auction dates, expected supply, and known schedule information inside the forecast horizon.

Why it is useful:
- This is exactly the kind of deterministic future information that the LLM should not have to infer from prose.
- For UK ETS, known auction timing is plausibly relevant at `h5/h20/h30`.

Expected impact:
- Likely benefit: `0.03-0.15` path MSE
- Most likely gain location: `h20/h30`, possibly some `h5`

Implementation cost:
- Low to medium

Verdict:
- Worth building early

### 5. Official Event Retrieval Tool

What it does:
- Pulls structured official event summaries from local DG CLIMA / EEX / UK auction event records.
- Returns only events known by the forecast origin date.

Why it is useful:
- MIRAI supports the value of structured event and news access for forecasting.
- The repo already has local official-event machinery; it is just not exposed as a callable inference tool yet.

Expected impact:
- Likely benefit: `0.03-0.12` path MSE
- Most likely gain location: long-horizon directional corrections

Implementation cost:
- Low

Verdict:
- Build soon, but keep it curated and official-source only

### 6. Regime Audit / Gating Tool

What it does:
- Returns a deterministic regime label and historical LLM-vs-TSM win statistics for similar regimes.
- Example outputs:
- `trend_regime_20d`
- `volatility_regime`
- `official_event_alignment_5d`
- historical LLM gain rate in that bucket

Why it is useful:
- Rolling-origin results show the LLM is not universally better than the base TSM.
- A regime-aware gate is more plausible than always applying the LLM.

Expected impact:
- Likely benefit: modest headline MSE, larger robustness gain
- Expected range: `0.03-0.10` average path MSE improvement across folds, mostly by avoiding bad regimes

Implementation cost:
- Medium

Verdict:
- High-value robustness tool

### 7. Delta Verification Tool

What it does:
- Checks a proposed adjustment vector against deterministic constraints:
- horizon sign coherence
- max move vs historical volatility
- consistency with retrieved analogue outcomes
- consistency with upcoming auction schedule

Why it is useful:
- The current 4B model is directionally decent at `h20/h30` but still fragile on magnitude.
- A verifier is cheaper and safer than asking the LLM to self-police numerically.

Expected impact:
- Likely benefit: `0.03-0.10` path MSE
- Main gain location: preventing large bad long-horizon corrections

Implementation cost:
- Low to medium

Verdict:
- Good complement to Tools 1 and 2

### 8. Cross-Market Query Tool

What it does:
- Computes current UK-specific spread and macro context features on demand:
- UKA-EUA spread
- UK auction primary-secondary spread
- UK gas / power proxies if available locally
- rolling percentile / anomaly state

Why it is useful:
- The current exogenous summary is static and compact.
- A query tool lets the LLM ask for exactly the feature it needs, rather than always seeing the same fixed block.

Expected impact:
- Likely benefit: `0.03-0.10` path MSE
- Main gain location: directional context for `h20/h30`

Implementation cost:
- Medium

Verdict:
- Useful after Tools 1 to 5

### 9. Uncertainty and Retrieval Router Tool

What it does:
- Estimates whether the model should:
- use no external tool
- use historical retrieval only
- use events only
- use both
- or abstain and keep base TSM unchanged

Why it is useful:
- Self-Routing RAG suggests selective retrieval can improve both quality and latency.
- Our prior experiments already showed that more context is not always better.

Expected impact:
- Likely benefit: `0.02-0.08` path MSE
- Bigger benefit in robustness and latency than in peak single-split MSE

Implementation cost:
- Medium

Verdict:
- Worth doing after the core tools exist

### 10. Trading Utility / Scenario Tool

What it does:
- Scores candidate adjusted paths for:
- directional stability
- sign flips
- expected long-short utility under simple cost assumptions

Why it is useful:
- Could help if the target objective shifts from pure path MSE to deployable trading utility.
- Less directly useful for pure forecasting accuracy.

Expected impact:
- Small for MSE
- More relevant for Sharpe / trade hit-rate than forecast error

Implementation cost:
- Medium

Verdict:
- Not a first-wave tool

## Tools To Avoid For Now

- Generic web search at inference time
- Open-ended code execution by the model
- Multi-agent debate style orchestration
- Large general-purpose RAG over noisy news text

Why:
- Too slow for the 4B path
- Too much variance
- Too many failure modes unrelated to the actual forecasting bottleneck

## Best First Implementation Order

1. Numeric analytics tool
2. Structured historical case retrieval tool
3. 35B teacher memory tool
4. Auction and supply calendar tool
5. Official event retrieval tool
6. Delta verification tool
7. Regime audit / gating tool

## Practical Conclusion

The highest-value tool strategy is not to make the 4B model a general agent. It is to turn it into a narrow decision layer on top of deterministic market tools.

The most promising architecture is:
- base TSM proposes the path
- tool layer computes exact market state, similar cases, known future auction schedule, and optional teacher cases
- 4B model chooses bounded horizon adjustments using those tool outputs
- verifier tool clips or rejects unsafe deltas

That is the shortest path to improving accuracy without moving back into expensive reasoning-mode behavior.
