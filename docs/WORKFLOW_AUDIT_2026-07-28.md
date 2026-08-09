# ETS2 workflow and modelling audit

**Audit date:** 2026-07-28  
**Reviewed baseline:** `main` at `9bcee07a327685cf4ec3cbed8f3f58c15a6e9cb1`  
**Improvement branch:** `agent/ets2-workflow-benchmark`  
**Pull request:** `#1 — Harden ETS2 benchmarking and local-model handoff`

## Executive verdict

The repository contains substantial useful work: multiple UK ETS datasets, frozen prediction artifacts, rolling-window experiments, a working OpenAI-compatible LLM client, and a contamination-safe core window constructor. It is not yet a trustworthy production forecasting system or a sound basis for a performance claim without correction.

The most important problem is not model capacity. It is experimental validity. The previous selective LLM gate used future test outcomes as input features, joined market features from the first forecast day rather than the last observed day, and selected thresholds using fitted training predictions. Several apparently positive selective-refinement results are therefore invalid as out-of-sample evidence.

The second major problem is that the class presented as an Autoformer is neither a faithful Autoformer implementation nor a functioning encoder–decoder forecaster: its “autocorrelation” block is ordinary scaled dot-product attention, and its decoder never consumes the encoder output. The current DLinear model is the more credible neural baseline, but the saved W1 evidence shows an ordinary ridge baseline beating it.

The branch created by this audit prioritises validity before novelty. It adds a frozen compact-model benchmark, reconstructs a causally aligned selective-refinement dataset, evaluates pre-call and post-call policies using chronological outer holdouts, adds focused regression tests and CI, isolates local LLM configuration, and prevents future generated artifacts from bloating source control.

## What is already credible

### Window construction

`src/data/windows.py` creates encoder windows that end before the prediction start, fills decoder future slots with placeholders rather than realised future rows, and can drop boundary-crossing windows. This is a good foundation and should be preserved.

### Existing baseline evidence

The saved W1 run `20260322_194840_381bee` reports the following 30-day path MSE values:

| Model | Path MSE |
|---|---:|
| Linear ridge | 21.583701 |
| Naive persistence | 22.605219 |
| DLinear / TSM | 23.257938 |

This is an important result: the minimum acceptance bar for a new neural or LLM-enhanced system must be the strongest simple baseline, not merely the current DLinear model.

The saved 9B optional-verifier experiment improved the DLinear base on W2, W3 and W4 but worsened W1. Its improvements were modest and its apply-stage failure rate ranged from roughly 4.8% to 14.0%. This is useful diagnostic evidence, but not sufficient for a production claim.

## Confirmed critical defects

### 1. Future-outcome leakage in the old selective LLM gate

The legacy case builder calculates:

```python
base_bias_h20 = base_pred[idx, 19] - y_true[idx, 19]
base_bias_h30 = base_pred[idx, 29] - y_true[idx, 29]
```

These values require the realised future outcome and were then included as predictors by `evaluate_selective_uplift_models_w1_w4.py`. A live gate cannot know them at decision time.

The same builder merges panel features on `date`, but the windowing code defines `date` as the prediction start date. That attaches the first forecast day’s market values to the decision row. Its `base_move_h20_pct` and `base_move_h30_pct` also use the target price from that first forecast day rather than the last observed target price.

**Consequence:** the previous selective-policy results must not be described as causal out-of-sample performance.

**Branch fix:** `build_selective_residual_helpful_case_dataset_v2.py` attaches the latest panel row strictly before the forecast date, prefixes those fields with `asof_`, recomputes base moves from `asof_y`, removes all future-error bias fields, replaces absolute local paths with `run_id`, and asserts `feature_date < date` for every row.

### 2. In-sample policy threshold selection

The previous uplift scripts fitted a model on all training rows, generated predictions on those same rows, and selected the keep threshold by maximising uplift on those fitted predictions. This is optimistic even without the feature leakage.

**Branch fix:** `benchmark_selective_policy_v2.py` uses forward out-of-fold predictions on older windows to select both model family and threshold. It then refits on all older windows and evaluates once on the next unseen window. With the historical W4→W1 ordering, W2 and W1 are genuine chronological outer tests.

### 3. Mislabelled and disconnected “Autoformer”

In `src/models/tsm.py`:

- `AutoCorrelation` computes softmax scaled dot-product attention rather than Autoformer’s time-delay aggregation mechanism.
- `SimpleAutoformer.forward()` computes `enc_out`, but the decoder is only another self-attention stack over `x_dec`; it never consumes `enc_out`.
- The model therefore has a dead encoder path with respect to the output.

**Required action:** either remove/rename this implementation to make its behaviour explicit, or replace it with a tested implementation whose decoder genuinely consumes encoder state. Do not compare it to published Autoformer results under the current name.

### 4. Training-loop defects

The TSM training loop uses `CosineAnnealingWarmRestarts`, calls `scheduler.step()` after every batch, and then calls `scheduler.step(val_loss)` after every epoch. The second call interprets validation loss as a scheduler epoch value rather than as a plateau metric. This creates an unintended learning-rate schedule.

The best validation checkpoint is saved, but the training method does not automatically reload it before final prediction. A later, worse epoch can therefore be evaluated unless the caller explicitly reloads the checkpoint.

Horizon weights are multiplied into the loss and then averaged over all cells without normalising by the sum of weights. Changing the weights also changes the overall loss scale and therefore the effective optimisation dynamics.

**Required action:** use one scheduler with one stepping convention; reload the best checkpoint automatically; and normalise weighted loss by the effective weight sum.

### 5. DLinear decomposition edge distortion

The DLinear moving average uses an `AvgPool1d` layer with zero padding. Near each sequence edge, zeros enter the trend estimate. The separate `MovingAvg` class elsewhere in the same file uses repeated endpoint padding, which is more appropriate for this use.

**Required action:** make DLinear use endpoint/replicate padding and add a numerical test for constant and linear input sequences.

### 6. Unsafe or overly trusted serialization

The repository loads scaler state through pickle and explicitly falls back to `torch.load(..., weights_only=False)` for legacy checkpoints. This is acceptable only for artifacts generated and controlled by the project owner.

**Required action:** never load checkpoints or pickles from an untrusted run directory. Prefer JSON/NPZ for scaler metadata and a weights-only checkpoint plus a separately validated configuration.

## Confirmed experimental and workflow weaknesses

### 1. A previous ridge benchmark compared incompatible scales

The saved hybrid-base report shows ridge path MSE values around 1,595–4,844 while DLinear is around 19–102, and validation selects 100% DLinear in every window. This is inconsistent with the ordinary ridge results saved by the main experiment and indicates an incorrect target/price reconstruction in that benchmark.

**Branch fix:** the compact benchmark infers the return convention from frozen targets, reconstructs the actual price path, records an alignment error, and refuses to treat a candidate as valid unless target-to-price reconstruction is correct.

### 2. Repeated adaptation to the same four windows

The repository contains many sequential W1–W4 sweeps. Even where each individual script avoids direct test tuning, repeatedly inspecting the same test windows and designing the next method around them creates adaptive test-set overfitting.

**Required action:** freeze the historical W1–W4 set as a development archive. Establish new rolling-origin outer windows whose outcomes are not inspected until the model, prompt, gate and thresholds are frozen.

### 3. Overlapping 30-day paths are not independent

Adjacent forecast origins share most of their realised 30-day target path. Ordinary row-level standard errors and tests substantially overstate effective sample size.

**Branch fix:** the policy benchmark reports a 30-origin moving/circular block bootstrap. New statistical comparisons should use a block/HAC-aware method and report both effect size and uncertainty.

### 4. Macro and pooled metrics answer different questions

Pooling all origins lets a high-error or longer window dominate. Averaging window-level MSE treats each market period equally. Both should be reported, but model selection should use a predeclared primary endpoint.

**Recommended primary endpoint:** macro-average 30-day path MSE across outer windows, with horizon-specific H1/H5/H20/H30 MSE as secondary endpoints.

### 5. Stale evaluation horizon

The historical windows end in 2025. They are valuable for reproducibility but cannot establish current performance in July 2026.

**Required action:** add newly frozen windows ending in 2026, generated only from data that would have been available at each forecast origin.

### 6. LLM failure handling is too permissive for a claimed improvement

Some saved experiments have double-digit apply failure rates. A method that silently falls back to the base can appear statistically safe while still being operationally unreliable and expensive.

**Required reporting:** request count, successful parse rate, fallback rate, retry count, tokens, latency, adjustment acceptance rate, and MSE conditional on accepted versus rejected responses.

### 7. Repository bloat and mixed source/artifact ownership

The repository is approximately 868 MB according to GitHub metadata and tracks full run directories, checkpoints, prediction binaries, timestamped reports, LLM request/response files, local paths and macOS metadata. This makes checkout slow, obscures the authoritative source, and risks accidental secret or private-data commits.

**Branch fix:** the expanded `.gitignore` blocks generated runs, checkpoints, timestamped reports, virtual environments, machine files and `.env` secrets. Existing Git history is not rewritten by this branch.

**Longer-term action:** move reproducibility bundles to GitHub Releases, object storage or DVC and keep a small manifest/hash in Git. A separate, deliberate history-cleanup migration can be considered later.

### 8. Monolithic code and duplicated project variants

`src/run_experiment.py` is roughly 7,000 lines and `src/llm/refine.py` is over 4,000 lines. The repository also contains related `bud`, `bud_standalone`, EU and UK variants. This makes invariant enforcement and regression testing difficult.

**Recommended target modules:**

```text
src/
  data/          # source adapters, as-of joins, feature contracts
  datasets/      # window creation, split policies, frozen manifests
  models/        # one model per module, common protocol
  llm/           # client, prompt builder, parser, bounded policy
  policies/      # pre-call gate, post-call veto, calibration
  evaluation/    # metrics, block bootstrap, reports
  pipelines/     # train, backtest, live forecast orchestration
```

### 9. Unpinned dependency surface

`requirements.txt` uses broad lower bounds for a large scientific and deep-learning stack. A fresh environment can resolve to materially different versions and behaviour.

**Required action:** maintain a minimal benchmark dependency file and a fully locked runtime environment. Record Python version, package lock hash, Git SHA, data-manifest hashes and served LLM model ID for every publishable run.

### 10. Machine-specific LLM configuration

Older YAML files contain a LAN endpoint and placeholder key. The code already supports `OPENAI_BASE_URL` and `OPENAI_API_KEY`; runtime endpoints should not live in committed experiment configurations.

**Branch fix:** `.env.example` documents the variables, and `uk_ets_qwen_27b_local_smoke.yaml` contains no host-specific address or real key.

## Improvements implemented on the branch

| Area | Change |
|---|---|
| Compact benchmark | Validation-selected Ridge and ExtraTrees candidates on frozen artifacts, with explicit price-path reconstruction and alignment evidence |
| Selective dataset | Strict previous-row as-of alignment; no future biases; portable run IDs |
| Selective policy | Chronological outer holdouts; forward OOF model/threshold selection; pre-call and post-call policy separation |
| Statistical evidence | 30-origin block bootstrap for overlapping paths |
| Testing | Synthetic end-to-end tests for reconstruction, causal split ordering, leakage exclusion, strict as-of joins and deterministic bootstrap |
| CI | Clean Python 3.11 GitHub-hosted Ubuntu benchmark suite with uploaded evidence |
| Local LLM | Safe eight-sample Qwen 27B smoke config using environment variables |
| Repository hygiene | Generated artifacts, checkpoints, secrets and machine files ignored for future commits |

## Recommended modelling programme

### Stage A — establish the non-LLM floor

Benchmark, without test tuning:

1. Naive persistence.
2. Direct per-horizon ridge on lagged returns and strictly lagged exogenous summaries.
3. ExtraTrees or histogram gradient boosting on the same safe features.
4. Corrected DLinear with replicate-padded decomposition and a repaired training loop.
5. A validation-selected convex ensemble of the strongest diverse models.

No LLM experiment should proceed as the headline model until it beats the strongest of these baselines on the same frozen outer windows.

### Stage B — constrain the LLM’s role

The LLM should not regenerate an unrestricted 30-point price path. A safer design is:

- provide the frozen base forecast and a compact, strictly as-of context;
- request bounded anchor adjustments at H5/H20/H30 plus calibrated confidence;
- interpolate anchors deterministically;
- apply a hard maximum adjustment;
- use a pre-call gate only to decide whether inference is worth paying for;
- use a post-call veto to reject incoherent or low-value returned adjustments;
- tune every bound and threshold using older validation windows only.

### Stage C — current rolling-origin confirmation

After all code, model, prompt and policy choices are frozen:

- run at least four new rolling-origin outer windows;
- report base, always-accept LLM, pre-call policy and post-call policy;
- report macro and pooled MSE, H1/H5/H20/H30, MAE and directional accuracy;
- use a block bootstrap or HAC-aware paired comparison;
- disclose parse/fallback rate, token use and latency;
- treat a confidence interval crossing zero as inconclusive, not as a win.

## Merge acceptance criteria

The branch should be considered ready to merge only when:

- all focused tests pass in a clean environment;
- the frozen benchmark artifacts are generated and archived;
- target-to-price reconstruction alignment error is effectively zero;
- every selective dataset row satisfies `feature_date < forecast date`;
- no policy feature contains realised outcome, future error, same-day market data or post-outcome metadata;
- model and threshold selection occur exclusively on older windows;
- the strongest proposed model is compared against the strongest simple baseline;
- result uncertainty is reported with dependence-aware intervals;
- the PR description states clearly which results are historical, newly reproduced, or still pending.

## Bottom line

The old project’s strongest asset is its accumulated frozen evidence. Its biggest weakness is that the workflow made it too easy to produce a plausible-looking improvement without a genuinely unseen test. The branch therefore improves the scientific control plane first. Once the corrected benchmark identifies a stable base and a valid selective policy, Qwen 27B can be evaluated as a bounded residual reasoner rather than trusted as an unconstrained forecaster.
