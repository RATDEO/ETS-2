# 4B vs 27B W0-W4 Comparison

## Setup

This benchmark compares the current 4B reference LLM refiner against the 27B refiner on the same UK ETS batch forecasting task.

- Base forecaster in both cases: `tsm` with the same `dlinear` branch.
- Refinement family in both cases: `TSM+LLM-COT-RF-HDELTA`.
- Policy family in both cases: same reference retrieval, gating, and tool-use structure.
- The main intentional model-side difference is the LLM size.
- The 27B endpoint required explicit non-thinking controls so it would behave like a plain instruction model inside the structured refinement pipeline.

Source artifacts:

- 4B W0: `/Users/davidwilkinson/Desktop/ETS 2/runs/20260331_105619_b6bb37`
- 27B W0: `/Users/davidwilkinson/Desktop/ETS 2/runs/20260331_181339_b40355`
- 4B W1-W4 source table: `/Users/davidwilkinson/Desktop/ETS 2/reports/uk_ets_best_model_w0_w4/20260326_115150/window_results.csv`
- 27B W1-W4 source table: `/Users/davidwilkinson/Desktop/ETS 2/reports/uk_ets_27b_reference_w1_w4/20260331_192217/window_results.csv`

The full side-by-side table is in `comparison.csv`.

## Headline Results

Across `W0-W4`, the 27B model beats the 4B model on average forecast error:

- Mean path improvement vs base: `4B +1.07%`, `27B +1.88%`
- Mean `20d` improvement vs base: `4B +1.26%`, `27B +2.42%`
- Mean `30d` improvement vs base: `4B +1.16%`, `27B +1.88%`

Window by window:

- `W0`: 27B beats 4B on path and `20d`, but loses on `30d`
- `W1`: both models hurt performance, and 27B hurts more
- `W2`: 27B beats 4B clearly
- `W3`: 27B beats 4B clearly
- `W4`: 27B beats 4B clearly

So the 27B wins `4/5` windows on path MSE uplift, and loses only on the one clearly adverse window.

## Why This Looks Like A Real LLM Effect

This pattern is more consistent with real model capability differences than with random perturbation noise.

Why:

1. The positive effects scale up with model size.
   In windows where the LLM helps (`W0`, `W2`, `W3`, `W4`), the 27B usually helps more than the 4B. The largest example is `W2`, where path uplift rises from `+1.71%` to `+4.75%`, `20d` uplift from `+2.06%` to `+5.74%`, and `30d` uplift from `+1.95%` to `+5.42%`.

2. The negative effects also scale up with model size.
   In the bad window (`W1`), both models make the forecast worse, but the 27B makes it worse by more than the 4B. Path degradation moves from `-0.71%` to `-1.67%`. `20d` degradation moves from `-0.87%` to `-1.85%`. `30d` degradation moves from `-1.09%` to `-2.16%`.

3. That is what you would expect if the model is genuinely doing more than adding noise.
   If refinement were mostly random or mostly a prompt-format artifact, increasing model capability would not be expected to amplify both upside and downside in a regime-sensitive way. A stronger model should make stronger interventions. That is what the results show.

4. The mechanism-level behavior also scales.
   On W0, the 27B intervened much more aggressively than the 4B: more active applies, fewer skip-gated cases, and larger average long-horizon deltas. That lines up with the metric pattern. The larger model is not passively copying the same behavior. It is making stronger judgments, which helps more in good regimes and hurts more in bad ones.

The simplest interpretation is:

- the LLM refinement capability is real
- model size increases the strength of that capability
- stronger capability improves the good regimes more, but also overshoots harder in the bad regime

That is a much more coherent story than “the LLM is random.”

## Important Caveats

This is strong evidence, but not strict proof.

- The sample is still only five windows.
- The 27B had endpoint-specific non-thinking controls enabled.
- The W1-W4 4B numbers come from an earlier completed benchmark run rather than a same-minute rerun.
- The comparison is still fair at the forecasting-task level, but it is not a formal controlled scientific study.

So the right claim is:

> These results strongly suggest that the LLM contribution is real and scales with model capability, because both the gains and the mistakes become larger as model size increases.

That is a defensible conclusion. A stronger causal claim would need more matched windows and repeated reruns.

## Research Implication

The next bottleneck is no longer “does the LLM do anything real?” The answer looks like yes.

The bottleneck is calibration:

- 27B appears better at extracting useful signal on average
- but it is also more forceful, so it overshoots in the adverse regime

That means the next improvement work should focus on making the 27B more selective or better calibrated in bad regimes, not reverting to the 4B by default.
