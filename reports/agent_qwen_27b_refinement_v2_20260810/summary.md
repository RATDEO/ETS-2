# Qwen 27B refinement redesign — 2026-08-10

## Outcome

The redesigned Qwen refinement policy improved ridge path MSE on both development batches and on the frozen holdout. The gains are small and come from one admitted correction per 16-origin batch; the important result is that the refinement no longer collapses into broadly positive, template-shaped corrections or damages the second development batch.

| Batch | Role | Ridge path MSE | Final Qwen path MSE | Improvement | Changed origins |
|---|---|---:|---:|---:|---:|
| seed 20260810 | development A | 25.827881 | 25.813991 | +0.054% | 1/16 |
| seed 20260812 | development B | 41.115495 | 41.095705 | +0.048% | 1/16 |
| seed 20261954 | frozen holdout | 11.132485 | 11.101937 | +0.274% | 1/16 |
| pooled, descriptive | 48 origins | 26.025287 | 26.003878 | +0.082% | 3/48 |

The pooled paired path-error t-test is not conventionally significant (`t=1.680`, `p=0.100`, 48 origins). With only three nonzero decisions, this is evidence of stable selective behavior, not evidence of a large or proven general improvement.

## Comparison with the reviewed policy

| Batch | Old raw Qwen | Old post-hoc slope gate | Final redesign |
|---|---:|---:|---:|
| development A | +0.569% | +0.435% | +0.054% |
| development B | -2.355% | +0.913% | +0.048% |
| holdout | +3.962% | +3.309% | +0.274% |

The old raw policy had larger upside but failed badly on development B. The old slope gate produced larger observed gains, but still admitted 11–14 of 16 Qwen paths and inherited the old prompt/retrieval behavior. The redesign is deliberately more conservative: it admitted exactly one correction in each batch and improved all three.

## What changed

### Retrieval and evidence

- Replaced high-error-only retrieval with a regime-, time-, and sign-balanced selector.
- Seeds positive, negative, and near-zero residual analogues, then uses MMR for diversity.
- Adds a hard teaching-example reuse cap (`2`); the final runs observed a maximum reuse of exactly `2` and no violations.
- Uses horizon-specific matched residual cases, weighted sign agreement, weighted median correction centers, MAD dispersion, and continuous reliability shrinkage.
- Includes only origin-known future context: forecast dates, weekday/month composition, and explicitly allowlisted binary schedule fields (`is_auction_day`). Continuous future features are rejected to prevent leakage.

### Prompt and Qwen behavior

- Removed the positive `+0.3%` default example and replaced it with a zero-action example.
- Marked sign-balanced examples as contrastive analogues, not votes or instructions.
- Made abstention explicit and required a causal link between scheduled evidence and a proposed move.
- Switched apply output from stock quarter-point actions to continuous decimal targets.
- Thinking is explicitly disabled using both `chat_template_kwargs.enable_thinking: false` and `/no_think`.
- All 48 final-run checkpoint responses recorded `llm_reasoning_present=false`.

### Deterministic safety policy

- Downward ridge paths are rejected before Qwen, saving all requests and tokens for those origins.
- Reflection must give H20 and H30 at least medium confidence with the same nonzero sign.
- At least one long-horizon residual target must be nonzero, and any nonzero target must align with Qwen's direction.
- High-confidence aligned actions may proceed on a trending ridge path.
- Medium-confidence actions may proceed only when the ridge H1→H30 path is near-flat (absolute slope at most `1%`); they cannot override a strong quantitative trend.
- Final adjustments are deterministically wrong-sign-zeroed and capped by the continuous evidence target.

## Final-run audit

| Batch | Pre-Qwen skips | Reflection/apply skips | Applies | Qwen requests | Tokens | Qwen latency | Max example reuse |
|---|---:|---:|---:|---:|---:|---:|---:|
| development A | 2 | 13 | 1 | 17 | 59,665 | 432.7 s | 2 |
| development B | 5 | 10 | 1 | 14 | 47,633 | 345.3 s | 2 |
| holdout | 2 | 13 | 1 | 17 | 59,698 | 438.8 s | 2 |

The admitted origins were:

- Development A: index `1`, origin `2025-07-02`, path-MSE reduction `0.2222`.
- Development B: index `6`, origin `2025-07-09`, path-MSE reduction `0.3167`.
- Holdout: index `99`, origin `2025-11-18`, path-MSE reduction `0.4888`.

Each was a high-confidence positive H20/H30 correction aligned with positive residual targets. The final policy made no holdout-driven changes.

## Verification

- Focused refinement/retrieval tests: `96 passed`.
- Full repository suite: `208 passed in 28.74s`.
- `git diff --check`: clean.
- Endpoint/model: `192.168.68.140:9881`, `qwen3.6-27b-q4-k-m`.
- Final run artifacts:
  - Development A: `runs/20260810_004048_4ffc53`
  - Development B: `runs/20260810_004912_e61cf8`
  - Holdout: `runs/20260810_005615_dc7e66`

## Interpretation

The LLM stage now works as a selective residual-refinement system: Qwen interprets contrasting historical analogues and can move the forecast in either direction, while deterministic evidence and confidence gates decide whether the move is admissible and how large it may be. It is not a broad forecast replacement and it does not create future information. The observed improvement is modest and statistically uncertain, but the failure mode that motivated the redesign—one-sided, coarse, overactive corrections—is removed in the tested batches.
