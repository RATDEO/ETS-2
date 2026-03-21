# Why UK SENT Helps W4 But Not W0-W3

Generated: 2026-03-21

## Question

Why does `TSM+LLM-COT-SENT-RF` help `W4` but hurt `W0-W3` in the UK transfer benchmark?

Runs analysed:

- `W0`: [20260320_222119_e08dd7](/Users/davidwilkinson/Desktop/ETS%202/runs/20260320_222119_e08dd7)
- `W1`: [20260320_223153_4c4227](/Users/davidwilkinson/Desktop/ETS%202/runs/20260320_223153_4c4227)
- `W2`: [20260320_225159_526b64](/Users/davidwilkinson/Desktop/ETS%202/runs/20260320_225159_526b64)
- `W3`: [20260320_230730_25c2ea](/Users/davidwilkinson/Desktop/ETS%202/runs/20260320_230730_25c2ea)
- `W4`: [20260320_231642_c8397d](/Users/davidwilkinson/Desktop/ETS%202/runs/20260320_231642_c8397d)

## Main Finding

`W4` works because the base forecaster has a strong positive long-horizon bias there, and the SENT refiner applies strong downward corrections that reduce that bias. In the other windows, the same refinement layer either:

- pushes an already negative bias further negative,
- over-corrects a milder positive bias,
- or even picks the wrong direction entirely.

So the `W4` gain is mostly a **bias-alignment win**, not evidence of a broadly robust sentiment signal.

## 1. The UK Sentiment Series Is Extremely Sparse

After the official-source rebuild, the UK sentiment daily series has only `6` non-zero dates:

| Date | Score |
|---|---:|
| `2021-05-19` | `+0.733` |
| `2021-06-02` | `+0.700` |
| `2021-06-03` | `-0.500` |
| `2022-10-28` | `-0.300` |
| `2022-11-03` | `-0.300` |
| `2024-11-28` | `+0.700` |

That means:

- `W0`: no non-zero sentiment in test or in the prior 18-day prompt history
- `W2`: no non-zero sentiment in test or in the prior 18-day prompt history
- `W3`: no non-zero sentiment in test or in the prior 18-day prompt history
- `W1`: one positive event date, and only `18/247` test days see it inside the prior 18-day history
- `W4`: two negative event dates, and only `24/247` test days see them inside the prior 18-day history

So most of the time, the prompt sees a flat zero sentiment vector.

## 2. W4 Has The Strongest Base Overprediction Problem At Long Horizons

Mean signed bias, base vs refined:

| Window | h1 base | h1 ref | h20 base | h20 ref | h30 base | h30 ref |
|---|---:|---:|---:|---:|---:|---:|
| `W0` | `-0.119` | `-0.584` | `-0.163` | `-1.300` | `+0.258` | `-1.140` |
| `W1` | `-0.050` | `-0.386` | `-2.035` | `-2.313` | `-2.914` | `-3.258` |
| `W2` | `-0.023` | `-1.175` | `+0.811` | `-0.612` | `+0.952` | `-0.679` |
| `W3` | `+0.292` | `+1.117` | `+6.023` | `+6.906` | `+8.143` | `+8.818` |
| `W4` | `+0.114` | `-1.123` | `+5.495` | `+3.655` | `+7.109` | `+4.796` |

Interpretation:

- `W4` base is materially too high at `h20/h30`, and the refiner moves it down in the right direction.
- `W0` base is only mildly high at `h30` and slightly low before that; the same downward correction overshoots.
- `W1` base is already too low across the curve; downward correction is the wrong move.
- `W2` has only a mild positive long-horizon bias; the refiner pushes too far down.
- `W3` is the worst directional failure: the refiner pushes forecasts **up**, even though the base is already too high.

## 3. The Learned Rules Change By Window, And W4 Gets The Right Rule Family

The reflection-stage rule files show different window-level stories:

- `W4` rules are consistently like:
  - “Model consistently overestimates early in the 30-day horizon”
  - “apply a 3–5% downward correction”
  - “dampen momentum-based forecasts”
- `W1/W2` rules are also mostly downward/cool-down rules
- `W3` rules flip into the opposite narrative:
  - “Model consistently underestimates prices”
  - “adjust upward by 2–5%”
  - “add buffer to avoid underestimating rebound”

This explains the sign of the adjustments:

- `W4`: correct diagnosis, downward corrections help
- `W3`: wrong diagnosis, upward corrections hurt badly

## 4. Actual Adjustment Direction And Size

Mean forecast delta, refined minus base:

| Window | Mean delta | Mean abs delta | h20 delta | h30 delta | Path uplift |
|---|---:|---:|---:|---:|---:|
| `W0` | `-0.96` | `0.96` | `-1.14` | `-1.40` | `-3.17%` |
| `W1` | `-0.29` | `0.74` | `-0.28` | `-0.34` | `-7.72%` |
| `W2` | `-1.32` | `1.36` | `-1.42` | `-1.63` | `-19.89%` |
| `W3` | `+0.96` | `1.13` | `+0.88` | `+0.68` | `-7.04%` |
| `W4` | `-1.68` | `1.97` | `-1.84` | `-2.31` | `+16.22%` |

This is the cleanest summary:

- `W4` is the only window where large downward long-horizon adjustments match the true base error.
- `W2` gets the same kind of downward correction, but the base bias is not large enough to justify it.
- `W3` is a sign error.

## 5. W4 Improvement Is Carried By A Few Large Wins

The `W4` gain is real, but it is concentrated:

Best per-date path gains:

- `2023-02-14`: `+524.46`
- `2022-12-07`: `+174.48`
- `2022-12-02`: `+127.61`
- `2023-04-14`: `+124.25`
- `2022-11-02`: `+123.78`

Worst per-date path losses:

- `2023-02-15`: `-171.63`
- `2023-01-25`: `-99.95`
- `2023-01-30`: `-99.52`

So `W4` is not uniformly better. It has a few very large positive rescue cases that outweigh some very large misses.

## 6. The Sparse Sentiment Signal Helps W4 Only Marginally

For the importance-weighted UK sentiment series:

- `W4` has only `2` non-zero sentiment dates in the entire test window
- only `24/247` test days see a non-zero score in the prior 18-day history

Path gain by last observed sentiment in prior 18 days:

| W4 last sentiment | Count | Mean path gain |
|---|---:|---:|
| `0.0` | `130` | `+16.41` |
| `-0.3` | `16` | `+17.28` |

That means:

- the negative sentiment dates do line up with slightly stronger gains,
- but the improvement is already present on the many all-zero-history days.

So the actual driver is not “the prompt sees negative sentiment and becomes smart.” The real driver is:

- the reflection/apply pipeline learns a strong **downward correction regime** for `W4`,
- and that regime happens to be right for the base model there.

## 7. Why W1 Does Not Convert Even Though It Sees Non-Zero Sentiment

`W1` is the only other window with non-zero sentiment history in test:

- one positive event date: `2024-11-28`
- `18/247` test days have that signal inside the prior 18-day history

Path gain by last sentiment in prior 18 days:

| W1 last sentiment | Count | Mean path gain |
|---|---:|---:|
| `0.0` | `131` | `-1.97` |
| `+0.7` | `12` | `+0.07` |

So the positive sentiment days do look slightly less bad than the zero-sentiment days, but the window still fails overall because:

- the learned correction policy remains mostly downward,
- and the base is already underpredicting there.

## Bottom Line

`W4` works because:

1. the base has a large long-horizon positive bias there,
2. the SENT reflection stage generates strong downward correction rules,
3. those corrections are directionally right for that window,
4. and a few large rescue cases dominate the average.

It does **not** work because the UK sentiment series is broadly rich or because the LLM is using a dense stream of news information. The UK series is far too sparse for that. The pipeline is mostly acting like a regime-specific rule generator, and only in `W4` does that rule family align with the true error structure.

## Practical Implication

If UK `SENT` is kept at all, it should not be broad always-on refinement. The only plausible next use is:

- a selective `W4`-like regime filter,
- likely based on large base overprediction at `h20/h30`,
- with sentiment as a weak secondary cue, not the primary source of alpha.
