# Selective Residual Refinement Methodology

## Objective
- Replace broad always-on LLM refinement with a scientifically testable selective residual-correction program.
- Learn from realized helpful vs harmful LLM cases before changing the live refiner again.

## Phase I Scope
- Use like-for-like improved-base UK runs only.
- Current source runs: W1, W2, W3, W4.
- Do not mix in older weaker-base runs when defining helpful regimes.

## Case-Level Dataset
- One row per forecast origin date.
- Extract realized base and LLM errors from saved prediction arrays.
- Join contemporaneous panel features from the saved run panel.
- Join LLM apply metadata from the saved JSONL call logs.

## Primary Labels
- `helpful_loose`: path MSE improved.
- `helpful_long_only`: both h20 and h30 improved.
- `helpful_strict`: path uplift >= 1%, h20 and h30 both improved, h5 not damaged by more than 10%.
- `harmful_strict`: path uplift <= -1% and at least one of h20/h30 worsened.

## Regime Definitions
- Volatility regime: tertiles of `y_vol_20d`.
- Base long-horizon move regime: tertiles of `abs(base_move_h20_pct)`.
- Retrieval-support regime: tertiles of `matched_teaching_count`.

## Scientific Use
- This phase is descriptive and label-building only.
- No new live policy is fit on the same rows being evaluated.
- The purpose is to define a preregistered helpful-case regime for the next benchmark tranche.
