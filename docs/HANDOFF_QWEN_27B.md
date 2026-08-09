# ETS2 handoff for the local coding agent and Qwen 27B

This is the operational handoff for continuing the UK ETS forecasting project from the audited branch.

## 1. Starting point

Use this branch, not `main`:

```bash
git fetch origin
git checkout agent/ets2-workflow-benchmark
git pull --ff-only origin agent/ets2-workflow-benchmark
```

The branch contains:

- `docs/WORKFLOW_AUDIT_2026-07-28.md`
- `tools/benchmark_frozen_compact.py`
- `uk_ets/scripts/build_selective_residual_helpful_case_dataset_v2.py`
- `tools/benchmark_selective_policy_v2.py`
- `uk_ets/config/uk_ets_qwen_27b_local_smoke.yaml`
- focused tests under `tests/`
- the GitHub Actions benchmark suite

Do not merge the branch or make performance claims until the clean benchmark suite has passed and the generated evidence has been reviewed.

## 2. Non-negotiable scientific rules

1. **Never tune on a test window.** Model family, prompt, adjustment limits, blend weights and gate thresholds must be fixed using older training/validation windows.
2. **Never use same-day or future features.** A forecast starting on date `t` may use only information with an as-of timestamp strictly before `t`.
3. **Never use `base_bias_h20`, `base_bias_h30`, `llm_bias_h20` or `llm_bias_h30` as predictors.** They require realised future outcomes.
4. **Do not call the current `SimpleAutoformer` a faithful Autoformer.** Its encoder output is not consumed by the decoder.
5. **The strongest simple baseline is the minimum bar.** In the saved W1 evidence, ordinary ridge beats DLinear.
6. **Treat adjacent 30-day paths as dependent.** Use a 30-origin block bootstrap or another dependence-aware paired test.
7. **Separate pre-call and post-call policies.** A pre-call gate can reduce inference cost. A post-call veto has already paid the inference cost and can only improve acceptance quality.
8. **Report failures and cost.** Include request count, parse success, retries, fallback rate, tokens and latency.

## 3. Create a clean Python environment

Python 3.11 is the reference version used by CI.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install "pytest>=8,<9" "tabulate>=0.9,<1"
```

For benchmark-only work, the smaller dependency set is sufficient:

```bash
python -m pip install \
  "numpy>=1.24,<3" \
  "pandas>=2,<3" \
  "pyarrow>=14,<22" \
  "scikit-learn>=1.3,<2" \
  "pyyaml>=6,<7" \
  "tabulate>=0.9,<1" \
  "pytest>=8,<9"
```

Record the exact resolved package versions before a publishable run:

```bash
python --version
python -m pip freeze > environment.freeze.txt
git rev-parse HEAD
```

Do not commit `environment.freeze.txt` until it has been checked for machine-specific package paths.

## 4. Run the focused tests first

```bash
python -m pytest -q \
  tests/test_benchmark_frozen_compact.py \
  tests/test_benchmark_frozen_object_dates.py \
  tests/test_selective_policy_v2.py \
  tests/test_selective_dataset_v2.py
```

These tests cover:

- return-to-price reconstruction;
- legacy NPZ object-date compatibility;
- validation-only compact-model selection;
- chronological outer holdouts;
- exclusion of realised-outcome features;
- strict previous-row as-of alignment;
- deterministic block bootstrap behaviour.

Stop immediately if any test fails.

## 5. Reproduce the frozen non-LLM benchmark

```bash
python tools/benchmark_frozen_compact.py
```

Expected outputs:

```text
reports/agent_compact_benchmark/
  benchmark_results.csv
  candidate_validation.csv
  errors.csv
  manifest.json
  summary.md
```

Acceptance checks:

- `errors.csv` is empty;
- `alignment_mae` is effectively zero for every run;
- the selected model and shrinkage were chosen on validation only;
- the test result is compared against `linear_ridge`, `naive_persistence` and `tsm` where available;
- no test metric was used to choose a candidate.

The compact benchmark is intentionally small. Its purpose is to establish a trustworthy floor, not to exhaustively optimise a model family.

## 6. Rebuild and evaluate the corrected selective-policy dataset

For the historical default W1–W4 runs:

```bash
python uk_ets/scripts/build_selective_residual_helpful_case_dataset_v2.py
python tools/benchmark_selective_policy_v2.py
```

Expected outputs:

```text
reports/agent_selective_dataset_v2/
  helpful_case_dataset_v2.csv
  manifest.json
  methodology.md

reports/agent_selective_policy_v2/
  aggregate.csv
  bootstrap.json
  candidate_selection.csv
  case_decisions.csv
  manifest.json
  results_by_window.csv
  summary.md
```

Mandatory checks:

```python
import pandas as pd

frame = pd.read_csv(
    "reports/agent_selective_dataset_v2/helpful_case_dataset_v2.csv",
    parse_dates=["date", "feature_date"],
)
assert (frame["feature_date"] < frame["date"]).all()
assert not {
    "base_bias_h20", "base_bias_h30", "llm_bias_h20", "llm_bias_h30"
}.intersection(frame.columns)
```

Interpretation:

- `precall` uses only strictly lagged market/base features and may be used to avoid an LLM request;
- `postcall` uses response metadata and proposed adjustments and may veto a returned adjustment;
- W2 and W1 are chronological outer holdouts under the historical four-window archive;
- a confidence interval crossing zero is inconclusive;
- these historical windows do not establish July 2026 performance.

## 7. Prepare current UK ETS data

The project’s bootstrap entry point is:

```bash
python uk_ets/scripts/bootstrap_uk_ets_data_sources.py \
  --output-dir uk_ets/Data_auto_uk \
  --start-date 2021-05-19 \
  --end-date YYYY-MM-DD
```

Set `YYYY-MM-DD` to the intended frozen data cutoff. Do not use “today” implicitly in a publishable experiment; record the exact cutoff.

Inspect:

```text
uk_ets/Data_auto_uk/automation_manifest.json
```

Every source must have:

- source URL or source identifier;
- minimum and maximum date;
- row count;
- success/failure status;
- local content hash added by the next implementation step.

The ICE auction source can be gated by reCAPTCHA. The current bootstrap may fall back to an ICAP primary-market proxy unless `--require-actual-auctions` is set. Treat actual and proxy auction data as different features and disclose which one was used.

A fresh clone may not contain every raw local target snapshot used in old runs. When that occurs, either point `--data-dir` at the owner’s existing verified data root or regenerate a new frozen data root. Do not silently substitute a different target series under the same run label.

## 8. Connect the local Qwen 27B server

Use an OpenAI-compatible endpoint. The project already reads endpoint and key from the environment.

```bash
export OPENAI_BASE_URL="http://127.0.0.1:8000/v1"
export OPENAI_API_KEY="local-inference"
```

Verify the server and copy the exact served model ID:

```bash
curl -sS \
  -H "Authorization: Bearer ${OPENAI_API_KEY}" \
  "${OPENAI_BASE_URL}/models"
```

Open:

```text
uk_ets/config/uk_ets_qwen_27b_local_smoke.yaml
```

Replace:

```yaml
model: "qwen-27b-local"
```

with the exact ID returned by `/v1/models`. Do not assume the display name, quantisation suffix or chat-template alias.

The smoke config deliberately uses:

- temperature `0.0`;
- eight LLM samples;
- bounded horizon adjustments;
- a frozen H1 anchor;
- strict JSON prompting and response format;
- no committed endpoint or secret.

## 9. Run the eight-sample Qwen smoke test

```bash
python -m src.run_experiment \
  --config uk_ets/config/uk_ets_qwen_27b_local_smoke.yaml \
  --data-dir uk_ets/Data_auto_uk
```

Locate the generated run:

```bash
RUN_DIR="$(ls -td runs/* | head -n 1)"
echo "${RUN_DIR}"
```

Inspect at least:

```text
${RUN_DIR}/config_resolved.yaml
${RUN_DIR}/reproducibility.md
${RUN_DIR}/results/path_metrics.csv
${RUN_DIR}/results/metrics_by_horizon.csv
${RUN_DIR}/llm/logs/llm_calls.jsonl
${RUN_DIR}/predictions/TSM+LLM-COT-RF-HDELTA_pred_test_subset.npz
```

Smoke acceptance gates:

- the exact Qwen model ID appears in resolved configuration/log metadata;
- all eight requests complete or fall back explicitly;
- JSON parse success is at least 99% for a larger validation batch and 100% is preferred for this eight-case smoke;
- no response produces NaN, infinity or a price outside configured adjustment bounds;
- the saved base forecast is byte-for-byte unchanged by the LLM path;
- failure/fallback rows are distinguishable from genuine zero-adjustment rows;
- the run is reproducible with temperature zero and the same seed, subject to the inference engine’s deterministic guarantees.

A successful smoke test is a systems check, not a model-quality result.

## 10. Create a proper Qwen rolling-window experiment

Copy the smoke config rather than editing it in place:

```bash
cp \
  uk_ets/config/uk_ets_qwen_27b_local_smoke.yaml \
  uk_ets/config/uk_ets_qwen_27b_local_full.yaml
```

Create at least four newly frozen rolling-origin windows. For each window, record:

```text
window
train_start
train_end
validation_start
validation_end
test_start
test_end
data_cutoff
Git SHA
data-manifest hash
base-model checkpoint hash
Qwen served model ID
prompt-template hash
```

Use the same frozen base predictions for every LLM comparison. Do not retrain the base separately for “LLM on” and “LLM off”.

Recommended comparison set:

1. strongest naive baseline;
2. strongest direct ridge/tree baseline;
3. corrected DLinear;
4. validation-selected non-LLM ensemble;
5. Qwen always accept;
6. Qwen plus pre-call gate;
7. Qwen plus post-call veto;
8. Qwen plus pre-call gate and post-call veto.

The headline endpoint should be declared before revealing test outcomes. Recommended:

- primary: macro-average 30-day path MSE across outer windows;
- secondary: H1, H5, H20 and H30 MSE; path MAE; direction accuracy;
- operational: parse rate, fallback rate, median/p95 latency and token use.

## 11. Build a corrected gate dataset from new Qwen runs

Create a portable mapping file after the Qwen runs complete:

```csv
window,train_end,val_end,test_end,run_dir
W4,YYYY-MM-DD,YYYY-MM-DD,YYYY-MM-DD,/absolute/path/to/run_w4
W3,YYYY-MM-DD,YYYY-MM-DD,YYYY-MM-DD,/absolute/path/to/run_w3
W2,YYYY-MM-DD,YYYY-MM-DD,YYYY-MM-DD,/absolute/path/to/run_w2
W1,YYYY-MM-DD,YYYY-MM-DD,YYYY-MM-DD,/absolute/path/to/run_w1
```

Then run:

```bash
python uk_ets/scripts/build_selective_residual_helpful_case_dataset_v2.py \
  --runs-csv qwen27b_runs.csv \
  --out-dir reports/qwen27b_selective_dataset_v2

python tools/benchmark_selective_policy_v2.py \
  --input reports/qwen27b_selective_dataset_v2/helpful_case_dataset_v2.csv \
  --output-dir reports/qwen27b_selective_policy_v2 \
  --mode both \
  --bootstrap 5000
```

Do not train a gate on rows from the held-out test window. Do not add a feature merely because it correlates with realised uplift; first prove it exists at the exact live decision point.

## 12. Repair the base-model implementation before a full rerun

Implement these changes in a separate commit, with tests:

### TSM training loop

- choose a single scheduler stepping convention;
- remove `scheduler.step(val_loss)` from cosine warm restarts;
- reload the best validation checkpoint before final prediction;
- normalise weighted losses by weight sum;
- save scheduler state and environment metadata;
- add deterministic CPU smoke training.

### DLinear

- replace zero-padded moving average with replicate/endpoint padding;
- test constant input, linear trend input and shape preservation;
- verify target-only and residual channel mixers independently;
- compare corrected DLinear against the frozen current DLinear before replacement.

### “Autoformer”

Choose one:

1. rename it to `SimpleAttentionForecaster` and document that it is not Autoformer; or
2. replace it with a tested encoder–decoder implementation whose decoder consumes encoder state.

Do not invest in transformer tuning until the corrected simple baselines have been rerun.

## 13. Recommended acceptance standard

A proposed improvement is ready for promotion only when all of the following hold:

- no leakage or same-day feature violations;
- no model/threshold/prompt choice made from the outer test outcomes;
- improvement over the strongest simple baseline, not only over DLinear;
- positive macro improvement on the predeclared primary endpoint;
- dependence-aware 95% interval does not cross zero;
- result is not driven by one window;
- no material degradation at H1/H5 unless explicitly accepted in advance;
- parse/fallback rate is operationally acceptable;
- cost and latency are reported;
- exact data, code, model and prompt hashes are archived.

Recommended promotion rule for a new four-window benchmark:

- beat the strongest baseline on at least three of four windows;
- positive macro improvement;
- positive block-bootstrap lower bound versus the strongest baseline;
- LLM parse/fallback failures below 2%;
- no hidden manual intervention.

These thresholds are project governance recommendations, not evidence that the current model already meets them.

## 14. Expected deliverables from the local coding agent

Return one PR containing:

1. repaired TSM/DLinear implementation and tests;
2. a locked or reproducible environment definition;
3. newly frozen rolling-window manifests;
4. baseline and Qwen benchmark result tables;
5. block-bootstrap uncertainty;
6. LLM reliability/cost table;
7. exact commands needed to reproduce every result;
8. a short decision memo: promote, continue experimenting, or reject.

The decision memo must explicitly separate:

- historical saved results;
- results reproduced from old frozen artifacts;
- new Qwen 27B results;
- speculative next steps.

## 15. Immediate command sequence

```bash
# 1. Branch and environment
git checkout agent/ets2-workflow-benchmark
source .venv/bin/activate

# 2. Tests
python -m pytest -q \
  tests/test_benchmark_frozen_compact.py \
  tests/test_benchmark_frozen_object_dates.py \
  tests/test_selective_policy_v2.py \
  tests/test_selective_dataset_v2.py

# 3. Frozen non-LLM evidence
python tools/benchmark_frozen_compact.py

# 4. Corrected historical policy evidence
python uk_ets/scripts/build_selective_residual_helpful_case_dataset_v2.py
python tools/benchmark_selective_policy_v2.py

# 5. Local Qwen endpoint
export OPENAI_BASE_URL="http://127.0.0.1:8000/v1"
export OPENAI_API_KEY="local-inference"
curl -sS -H "Authorization: Bearer ${OPENAI_API_KEY}" "${OPENAI_BASE_URL}/models"

# 6. After inserting the exact model ID into the smoke YAML
python -m src.run_experiment \
  --config uk_ets/config/uk_ets_qwen_27b_local_smoke.yaml \
  --data-dir uk_ets/Data_auto_uk
```

## Final instruction to the coding agent

Do not optimise the headline number first. First reproduce the branch benchmark exactly, prove every feature is available at decision time, repair the base-model training defects, and freeze a genuinely unseen outer test. Only then run Qwen 27B and decide whether its bounded residual adjustments add value beyond the strongest simple model.
