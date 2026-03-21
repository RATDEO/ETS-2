# UK SENT Dense Rebuild Interim

## Dense Corpus
- Build summary: [/Users/davidwilkinson/Desktop/ETS 2/uk_ets/Data_auto_uk/news/dense_daily1_build_summary.json](/Users/davidwilkinson/Desktop/ETS%202/uk_ets/Data_auto_uk/news/dense_daily1_build_summary.json)
- Dense raw daily sentiment: [/Users/davidwilkinson/Desktop/ETS 2/uk_ets/Data_auto_uk/news/dense_daily1_daily_sentiment_uk_qwen_votes3.csv](/Users/davidwilkinson/Desktop/ETS%202/uk_ets/Data_auto_uk/news/dense_daily1_daily_sentiment_uk_qwen_votes3.csv)
- Dense importance daily sentiment: [/Users/davidwilkinson/Desktop/ETS 2/uk_ets/Data_auto_uk/news/dense_daily1_daily_sentiment_uk_qwen_votes3_importance.csv](/Users/davidwilkinson/Desktop/ETS%202/uk_ets/Data_auto_uk/news/dense_daily1_daily_sentiment_uk_qwen_votes3_importance.csv)

Coverage achieved:
- `1760 / 1760` daily rows from `2021-05-19` to `2026-03-13`
- `1669` exact same-day selected headlines
- `91` carry days
- exact coverage: `94.83%`

## Completed Dense Benchmark Rows
Current results file: [/Users/davidwilkinson/Desktop/ETS 2/reports/uk_ets_sent_transfer_dense/20260321_130700/results.csv](/Users/davidwilkinson/Desktop/ETS%202/reports/uk_ets_sent_transfer_dense/20260321_130700/results.csv)

Completed candidates:
- `base_energy16`
- `sent_feature_raw`
- `sent_feature_importance`

Completed feature-side results:

| Candidate | W0 | W1 | W2 | W3 | W4 | Mean |
|---|---:|---:|---:|---:|---:|---:|
| `base_energy16` | `29.075` | `23.258` | `18.598` | `68.515` | `101.735` | `48.237` |
| `sent_feature_raw` | `29.784` | `23.689` | `18.622` | `69.791` | `102.258` | `48.829` |
| `sent_feature_importance` | `29.714` | `23.718` | `18.622` | `69.938` | `102.301` | `48.859` |

Feature-side implication:
- dense UK sentiment features are worse than the improved UK base on all `W0-W4` windows completed so far
- the earlier sparse-corpus `W0` feature gain does not survive the dense rebuild

## Still Running
- `cot_sent_raw`
- `cot_sent_importance`

The prompt-side rerun is much slower because it requires full LLM refinement over each test window.
