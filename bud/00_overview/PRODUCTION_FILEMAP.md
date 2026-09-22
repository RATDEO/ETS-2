# Production File Map

This is the active UK ETS live-production path after the readability rewrite.

Read order:

1. `uk_ets/scripts/run_uk_live_forecast_backend.py`
2. `src/uk_ets_live_deployment.py`
3. `src/live_backend/forecast.py`
4. `src/live_backend/archive.py`
5. `src/live_backend/online_memory.py`
6. `src/live_backend/common.py`

Snapshot of the old production code:

- `snapshots/production_20260405_live_backend/`

## Top-level flow

```text
uk_ets/scripts/run_uk_live_forecast_backend.py
  -> optional data bootstrap
  -> optional fresh canonical experiment run
  -> load existing archive CSV
  -> seed archive from canonical backtest predictions
  -> rebuild missing recent live-history rows
  -> build current live forecast rows
  -> backfill realized actuals
  -> write website export JSON
  -> optional rsync deploy
```

## Files That Matter

```text
uk_ets/
├── scripts/
│   ├── run_uk_live_forecast_backend.py      # main production entrypoint
│   └── run_uk_automated_pipeline_once.py    # experiment-only runner, still used when you want a backtest run without the live export path
└── config/
    ├── uk_ets_llm_4b_canonical_default.yaml # default live-backend config
    └── uk_ets_default.yaml                  # default experiment-only UK config

src/
├── uk_ets_live_deployment.py                # compatibility surface; keep imports stable for scripts/tests
├── live_backend/
│   ├── common.py                            # constants, archive schema, small shared helpers
│   ├── archive.py                           # archive load/save, seed rows, export JSON, deploy helpers
│   ├── forecast.py                          # live panel loading, live row generation, recent-history rebuild
│   └── online_memory.py                     # online-memory selection, simulation, learned gate helpers
├── frontend_live_bundle.py                  # lower-level train/predict helpers used by the live backend
├── run_experiment.py                        # legacy experiment engine still used to produce fresh canonical runs and some gate helper functions
└── config/
    └── config.py                            # config loading and run-dir creation
```

## Responsibility Split

- `run_uk_live_forecast_backend.py`: orchestration only.
- `uk_ets_live_deployment.py`: stable import surface for the rest of the repo.
- `live_backend/forecast.py`: “given config + data, build forecast rows.”
- `live_backend/archive.py`: “given rows, maintain/export/archive them.”
- `live_backend/online_memory.py`: “given validation history, decide how memory examples affect live LLM use.”
- `frontend_live_bundle.py`: model-training and base-forecast mechanics.
- `run_experiment.py`: large legacy engine. Production still calls it for a fresh canonical run, but it is no longer where the live-backend flow is defined.

## What You Can Ignore For Live Production

- `reports/`
- `paper/`
- most files under `scripts/`
- most files under `uk_ets/scripts/` other than the two runners above
- historical `runs/` contents

