# ETS Bud Standalone

This folder is designed to be copied by itself into another project and run independently.

## Quick Start

1. Copy `bud_standalone/` into your target project.
2. Open a terminal in that copied folder.
3. Initialize environment:

```bash
./init.sh
```

4. Set LLM endpoint/access (same as main project):

```bash
export OPENAI_API_KEY=deo
export OPENAI_BASE_URL=http://192.168.1.140:9877/v1
```

5. Run once:

```bash
./run_once.sh
```

Or do both in one command:

```bash
./init_and_run.sh
```

## What Is Included

1. `src/` model + data + eval pipeline code.
2. `scripts/run_automated_pipeline_once.py`
3. `scripts/bootstrap_data_sources.py`
4. `requirements.txt`
5. `Data_auto/` seed snapshot for reliable incremental updates.

## Required Dependencies

Installed via `./init.sh` from `requirements.txt`, including:

1. `torch`
2. `pandas`
3. `requests`
4. `yfinance`
5. `openai`
6. `openpyxl`
7. `xlrd`

## Runtime Controls

You can override defaults with env vars:

1. `OPENAI_API_KEY` or `LLM_API_KEY`
2. `OPENAI_BASE_URL` or `LLM_BASE_URL`
3. `LLM_MODEL` (default `qwen3-vl-4b-gpu`)
4. `LLM_MAX_SAMPLES` (default `100000`)
5. `TARGET_MODE` (default `returns`)
6. `COT_METHOD` (default `cot_rf`)

## Outputs

Run outputs are written under:

1. `runs/<run_id>/results/path_metrics.csv`
2. `runs/<run_id>/results/metrics_by_horizon.csv`
3. `runs/<run_id>/results/four_option_comparison_with_cot_ramp.csv`

Data refresh status:

1. `Data_auto/automation_manifest.json`

## Notes

1. This is operationally standalone, but still relies on internet data sources and your reachable LLM endpoint.
2. If sources are temporarily unavailable, bootstrap step can fail.
