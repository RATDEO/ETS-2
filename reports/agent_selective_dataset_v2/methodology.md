# Selective residual dataset v2 methodology

## Corrections

- Market features are attached from the most recent panel row strictly before the prediction start date (`feature_date < date`).
- Every lag-safe market column is namespaced with `asof_`.
- `base_move_h20_pct` and `base_move_h30_pct` use `asof_y`, not the first realised forecast-day price.
- Future-error fields (`base_bias_h20`, `base_bias_h30`, `llm_bias_h20`, `llm_bias_h30`) are not saved as predictors.
- Absolute local paths are replaced with a portable `run_id`.

## Evaluation warning

Only historical rows where the previous system actually produced an LLM adjustment reveal that adjustment's counterfactual. A pre-call gate evaluated on this dataset is therefore conditional on the old opportunity set.

## Source runs

- W1: `20260320_182015_da9e9d`
- W2: `20260320_183041_c7484b`
- W3: `20260320_183637_a4d80c`
- W4: `20260320_184725_2cded6`
