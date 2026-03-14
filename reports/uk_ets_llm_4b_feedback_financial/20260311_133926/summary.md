# UK ETS 4B Feedback Financial Sweep

Generated: 2026-03-11T14:52:57.951328

## Checkpoint

- Base checkpoint run: `/Users/davidwilkinson/Desktop/ETS 2/runs/20260311_045606_c3f654`
- Checkpoint raw `TSM+LLM-COT-RF-HDELTA` path MSE: `21.999591`
- Reference `linear_ridge` path MSE: `22.419955`
- Checkpoint h20 non-overlap total return: `nan%`
- Checkpoint h30 non-overlap total return: `nan%`

## Candidate Results

| Candidate | Expected | Actual MSE | Outcome | h20 vs chkpt | h30 vs chkpt | h20 total | h30 total | h20 Sharpe | h30 Sharpe |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|
| control_feedback_recent_high_error_k4 | 21.96-22.04 | 21.999591 | within | +0.000000 | +0.000000 | 114.57% | 0.92% | 2.191 | 0.177 |
| feedback_skill_tag_recent_high_error_k4 | 21.90-22.00 | 21.852871 | beat | -0.272153 | -0.272377 | 120.46% | 0.92% | 2.283 | 0.177 |
| feedback_plus_freeze_counterexample_k4 | 21.92-22.02 | 21.964541 | within | -0.089625 | +0.257689 | 120.46% | 0.92% | 2.283 | 0.177 |
| feedback_long_horizon_only_k4 | 21.92-22.04 | 21.999591 | within | +0.000000 | +0.000000 | 114.57% | 0.92% | 2.191 | 0.177 |
| feedback_compact_tags_k4 | 21.90-22.02 | 22.097110 | missed | +0.216519 | +0.178683 | 114.57% | 22.72% | 2.191 | 2.625 |
| feedback_skill_tag_plus_counterexample_k4 | 21.88-22.00 | 21.911751 | within | -0.209207 | +0.063597 | 114.57% | 22.72% | 2.191 | 2.625 |

## Best Candidate

- Candidate: `feedback_skill_tag_recent_high_error_k4`
- Change: Combine hindsight feedback with regime-skill-tag filtering.
- Literature basis: Chain of Hindsight plus Skill-KNN style skill matching.
- Aim: Test whether feedback becomes more reusable when the examples also match the same coarse market regime.
- Expected range: `21.90-22.00`
- Actual raw path MSE: `21.852871`
- Delta vs checkpoint raw LLM: `-0.146720`
- Delta vs base TSM: `-0.873482`
- Delta vs ridge: `-0.567084`
- h20 non-overlap total return: `120.46%`
- h20 non-overlap annualized return: `247.33%`
- h20 non-overlap Sharpe: `2.283`
- h20 non-overlap max drawdown: `2.69%`
- h30 non-overlap total return: `0.92%`
- h30 non-overlap annualized return: `1.56%`
- h30 non-overlap Sharpe: `0.177`
- h30 non-overlap max drawdown: `12.29%`
- Outcome vs expectation: `beat`
- Run dir: `/Users/davidwilkinson/Desktop/ETS 2/runs/20260311_135144_bb26cf`
