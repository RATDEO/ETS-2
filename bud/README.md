Readable bundle reading order:

1. `00_overview/PRODUCTION_FILEMAP.md`
2. `01_runner/run_uk_live_forecast_backend.py`
3. `01_runner/uk_ets_live_deployment.py`
4. `02_backend_core/common.py`
5. `02_backend_core/archive.py`
6. `02_backend_core/forecast.py`
7. `02_backend_core/online_memory.py`
8. `03_live_bundle_json/metrics.json`
9. `03_live_bundle_json/latest.json`
10. `03_live_bundle_json/horizon_20d.json`
11. `03_live_bundle_json/horizon_30d.json`
12. `04_full_archive/forecast_history.csv`

Folder purpose:

- `00_overview`: file map and orientation
- `01_runner`: top-level entrypoint and compatibility layer
- `02_backend_core`: the readable production backend modules
- `03_live_bundle_json`: the current exported site bundle
- `04_full_archive`: the full realized/live history CSV behind the bundle
