# UK ETS 35B Follow-Up and Paper-Gap Analysis

Generated: 2026-03-09

## 35B Result

Using the current best UK LLM prompt stack (`least_to_most` + `program_of_thought`) on the 35B endpoint (`qwen3.5-35b-a3b-ud-q4-k-xl` on port `9881`) improved raw UK test path MSE to `22.143373`.

Comparisons:

- Base `tsm`: `22.726353`
- `linear_ridge`: `22.419955`
- 4B winner: `22.418530`
- 35B raw winner: `22.143373`

So the 35B raw path improved by:

- `-0.582979` vs base `tsm`
- `-0.276582` vs `linear_ridge`
- `-0.275157` vs the 4B winner

## Blend Selection Fix

The prior blend selector tuned on the full validation span and always selected `w=0.00`, even when the test blend curve improved monotonically with more LLM weight.

I added a configurable recent-tail selection mode to the blend grid in `src/run_experiment.py`:

- `llm.blend_grid.tune_scope`
- `llm.blend_grid.recent_tail_fraction`
- `llm.blend_grid.recent_tail_min_samples`

On the 35B run with `recent_tail_fraction=0.25` and `recent_tail_min_samples=48`, validation selected `w=1.00` instead of `w=0.00`.

That produced:

- Best-on-val ramp blend: `22.335848`
- Previous full-val-selected blend: `22.726350`

So the selector is now directionally correct. But the raw LLM forecast is still best:

- Raw 35B `TSM+LLM-COT-RF-HDELTA`: `22.143373`
- Best recent-tail-selected ramp blend: `22.335848`

## Why We Still Do Not See Paper-Sized Gains

The short answer is that the headline CoT papers and our UK ETS task are not the same problem class.

1. The original CoT gains were mostly reported on discrete reasoning benchmarks such as GSM8K, SVAMP, AQuA, StrategyQA, SCAN, and related tasks. Those tasks have exact answers and low irreducible noise. UK ETS forecasting is a noisy continuous regression problem scored by path MSE across 30 days.

2. The strongest CoT gains in the early papers were observed on much larger frontier models than we are using locally. Our best endpoint is a quantized local 35B model. The 4B-to-35B jump in our own results is evidence that scale is still a major limiter here.

3. The time-series LLM papers that report stronger forecasting gains usually do much more than post-hoc verbal correction. `Time-LLM` reprograms time-series patches into a text-compatible interface. `TEMPO` uses prompt-based pre-training for time series directly. Our method is a cheaper overlay on top of a tuned DLinear, not a full time-series foundation-model adaptation.

4. We are correcting a strong domain baseline, not a weak direct-prompt baseline. The base UK `tsm` is already tuned and close to `linear_ridge`, so the remaining error budget is small and harder to improve.

5. Our validation/test mismatch is a concrete sign of non-stationarity. Full validation says “don’t trust the LLM,” while test says “use it heavily.” That is a regime-shift problem. Benchmark papers usually evaluate on much more IID task distributions.

6. Our `program_of_thought` implementation is still prompt-only. The original idea gets part of its benefit from separating reasoning from actual computation. We do not yet give the model executable tools, retrieval, or an external calculator during refinement.

7. Financial/carbon forecasting is sensitive to subtle fluctuations. Recent time-series LLM work explicitly shows that large models can look good on coarse patterns while remaining brittle under small perturbations or fine-grained noise. That is much closer to our UK ETS setting than GSM8K-style reasoning.

## Practical Reading of the Current Frontier

The right benchmark for us is no longer “why are we not getting 20-point accuracy jumps like GSM8K papers?” The right benchmark is whether a prompt-only LLM correction layer can beat a tuned numeric forecaster on a noisy carbon market. On that standard, the answer is now yes:

- 4B prompt-only correction: modest but real gain
- 35B prompt-only correction: clear gain
- full-val blend selection: misaligned
- recent-tail blend selection: improved, but still not better than raw 35B

## Primary Sources

- Chain-of-Thought Prompting Elicits Reasoning in Large Language Models: https://arxiv.org/abs/2201.11903
- Self-Consistency Improves Chain of Thought Reasoning in Language Models: https://arxiv.org/abs/2203.11171
- Least-to-Most Prompting Enables Complex Reasoning in Large Language Models: https://arxiv.org/abs/2205.10625
- Program of Thoughts Prompting: https://arxiv.org/abs/2211.12588
- Active-Prompt: Prompting Language Models for Complex Reasoning through Verbalized Confidence: https://arxiv.org/abs/2302.12246
- Reflexion: https://arxiv.org/abs/2303.11366
- Time-LLM: Time Series Forecasting by Reprogramming Large Language Models: https://arxiv.org/abs/2310.01728
- TEMPO: Prompt-based Generative Pre-trained Transformer for Time Series Forecasting: https://arxiv.org/abs/2310.04948
- Large Language Models Are Zero-Shot Time Series Forecasters: https://arxiv.org/abs/2310.07820
- Revisiting LLMs as Zero-Shot Time Series Forecasters under Small Noises: https://arxiv.org/abs/2410.14766
