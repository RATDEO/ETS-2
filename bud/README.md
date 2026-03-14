# ETS Automated Bud (Production)

This bud is the production-focused automated pipeline branch of the project.
It keeps automated data in `Data_auto/` and runs the experiment stack against that data root.

## What To Copy To A New Project

Copy these paths together:

1. `src/`
2. `scripts/`
3. `requirements.txt`
4. `Data_auto/` (recommended, for self-seeding updates)
5. `bud/`

If you do not copy `Data_auto/`, you must provide a seed directory with historical baseline data (at minimum EUA futures seed CSVs and auction seed files).

## Runtime Dependencies

Install with:

```bash
pip install -r requirements.txt
```

Critical packages include:

1. `requests`
2. `pandas`
3. `yfinance`
4. `openpyxl`
5. `xlrd`
6. `openai`
7. `torch` / model dependencies from `requirements.txt`

## Environment Variables

Use the same LLM access model as main project:

```bash
export OPENAI_API_KEY=deo
export OPENAI_BASE_URL=http://192.168.1.140:9877/v1
```

Notes:

1. CLI args override env vars.
2. If no API key is passed and `OPENAI_API_KEY` is missing, runner falls back to local default key `deo`.

## Production Run Command

Recommended command:

```bash
python scripts/run_automated_pipeline_once.py \
  --config src/config/default.yaml \
  --data-dir Data_auto \
  --seed-data-dir Data_auto \
  --cot-end-to-end \
  --cot-method cot_rf \
  --llm-max-samples 100000 \
  --no-paper-snapshot
```

Why `--seed-data-dir Data_auto`:

1. Makes the bud self-seeding if copied without the original `Data/`.
2. EUA updater can merge incremental pull with existing `Data_auto` snapshot.

## Wrapper Script

You can also run:

```bash
bash bud/run_production_once.sh
```

This wrapper uses env vars and production defaults.

## Expected Outputs

Per run, check:

1. `runs/<run_id>/results/path_metrics.csv`
2. `runs/<run_id>/results/metrics_by_horizon.csv`
3. `runs/<run_id>/results/four_option_comparison_with_cot_ramp.csv`
4. `runs/<run_id>/llm/blend_grid_selection_TSM_LLM-COT-RF.json`
5. `Data_auto/automation_manifest.json`

## Production Health Checks

After each run:

1. Confirm `automation_manifest.json` has `ok=true` for required sources.
2. Confirm `path_metrics.csv` includes:
   - `TSM+LLM-COT-RF_blend_ramp_bestval_path_h1base`
3. Confirm horizon MSE for that model includes:
   - `h1` equal to base `tsm` `h1` for same run.

## Known Source Constraints

1. EUA source currently provides reliable incremental updates, not guaranteed full-history rebuild from scratch.
2. Keep a persistent `Data_auto` snapshot (or baseline seed data) for disaster recovery.

