# Qwen refinement experiment continuation — 2026-08-10

## Decision

The supported result is the leakage-safe nested scalar-ramp stacker. Across six non-overlapping outer periods and 360 forecast origins, it reduced ridge path MSE from **18.796736** to **17.265561**, an **8.146% improvement**. The paired t-test p-value is **0.000081** and the Wilcoxon p-value is **0.000204**.

The LLM refinement stage should therefore use Qwen-derived event features inside the supervised scalar-ramp stacker. It should not use Qwen to generate numerical corrections directly.

## Experiment ladder

| Experiment | Path MSE | Improvement vs ridge | Status |
|---|---:|---:|---|
| Ridge base | 18.796736 | 0.000% | Reference |
| Nested adaptive scalar-ramp stacker | 17.265561 | +8.146% | Supported |
| Matched best-Qwen-per-fold comparator | 17.672748 | +5.980% | Supports incremental Qwen value |
| Matched numeric-only comparator | 18.480723 | +1.681% | Ablation |
| Two-validation-block stability selector | 17.581591 | +6.465% | Rejected; weaker than v1 |
| Broad direct-Qwen conflict gate | 18.250968 | +2.904% | Rejected; over-abstains |
| Confirmed countertrend-break Qwen gate | 16.580452 | +11.791% | Exploratory; freeze for prospective test |

## Direct 27B regime calls

Twelve monthly regime decisions were obtained from `qwen3.6-27b-q4-k-m` at `192.168.68.140:9881`. Every request used both `/no_think` and `chat_template_kwargs.enable_thinking=false`. None returned reasoning content.

The broad gate was insufficiently calibrated and removed profitable corrections. A narrower rule activated only when all of the following held:

1. Qwen reported medium or high confidence.
2. Qwen flagged a policy/supply break.
3. Qwen flagged an energy/macro break.
4. Qwen's direction opposed the observed 20-period price trend.
5. The supervised correction conflicted with Qwen's direction.

That rule rejected 52 of 360 refinement rows and improved retrospective MSE by 11.791%. It was formulated after inspecting the nested-v1 failures, so this number is not confirmatory. The rule is now frozen in `uk_ets_qwen_regime_break_gate_v2.yaml` for evaluation on newly acquired data.

## Reproducibility

- Nested supported report: `reports/qwen_scalar_ramp_nested_backtest_v1/20260810_primary/`
- Two-block stability report: `reports/qwen_scalar_ramp_nested_backtest_v2/20260810_primary/`
- Broad Qwen gate report: `reports/qwen_regime_break_gate_v1/20260810_primary/`
- Exploratory narrow gate report: `reports/qwen_regime_break_gate_v2/20260810_exploratory/`
- Scalar-ramp implementation: `src/eval/residual_event_model.py`
- Nested runner: `scripts/run_qwen_scalar_ramp_nested_backtest_v1.py`
- Direct regime-gate runner: `scripts/run_qwen_regime_break_gate_v1.py`

All Qwen prompts and raw JSON decisions are cached with prompt SHA-256 hashes and their evidence cutoffs. The complete repository test suite passes: **216 tests**.
