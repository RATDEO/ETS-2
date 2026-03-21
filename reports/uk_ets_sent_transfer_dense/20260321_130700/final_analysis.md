# UK SENT Dense Rebuild Final Analysis

## Dataset Validity
- Dense daily sentiment rows: `1760 / 1760`
- Date span: `2021-05-19` to `2026-03-13`
- Exact same-day headline coverage: `1669` days (`94.83%`)
- Carried days: `91`

This removes the earlier sample-size failure mode. The dense rebuild is a valid daily-input benchmark for UK SENT.

## Base Reference
- `base_energy16` mean path MSE over `W0-W4`: `48.237`

## Dense Feature-Side Result
- `sent_feature_raw` mean path MSE: `48.829`
- mean uplift vs base: `-1.36%`
- window wins: `0 / 5`

- `sent_feature_importance` mean path MSE: `48.859`
- mean uplift vs base: `-1.39%`
- window wins: `0 / 5`

Dense sentiment as a direct model feature is unambiguously worse than the improved UK base on every window.

## Dense Prompt-Side Result
- `cot_sent_raw` mean path MSE: `47.727`
- mean uplift vs base: `-0.80%`
- window wins: `2 / 5`

- `cot_sent_importance` mean path MSE: `47.561`
- mean uplift vs base: `-3.50%`
- window wins: `2 / 5`

Per-window prompt-side behavior:
- `W0`: worse for both prompt variants
- `W1`: worse for both prompt variants
- `W2`: `cot_sent_raw` helps, `cot_sent_importance` hurts badly
- `W3`: near flat for `cot_sent_importance`, worse for `cot_sent_raw`
- `W4`: both prompt variants help materially, with `cot_sent_importance` best

## Interpretation
- The earlier sparse-corpus `W0` feature win does not survive the dense rebuild.
- The `W4` prompt-side benefit persists, but it is smaller than before and still not enough to make UK SENT robust overall.
- So the original positive UK SENT story was mostly an artifact of sparse sampling plus a regime-local `W4` effect.

## Project-Level Conclusion
With a defensible dense daily UK sentiment series, UK SENT is not a generally useful always-on transfer from the EU setup for UK ETS:
- feature-side UK SENT: reject
- prompt-side UK SENT: regime-specific only, not robust enough for default use
