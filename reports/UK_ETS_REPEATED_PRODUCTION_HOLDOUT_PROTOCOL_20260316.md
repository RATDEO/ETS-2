# UK ETS Repeated Production Holdout Protocol

## Objective

Evaluate whether the current live production forecast methodology generalizes across multiple historical windows when the benchmark geometry is held constant.

This protocol is intended to replace ad hoc comparisons across different fold shapes with one repeated-holdout design that mirrors the latest proper production-style run.

## Frozen Method

The method is frozen at the current promoted live forecast stack:

- Base forecaster: `DLinear`
- LLM apply gate: learned `hist_gbdt` with the promoted `top_20_mi` feature schema
- LLM refinement path: current selective online-memory workflow
- Endpoint: live `4B` non-reasoning endpoint

No prompt, gate, memory, feature, or tool redesign is allowed between windows.

Only the causal training data moves as the window rolls backward.

## Window Geometry

Anchor split is taken from the latest proper live run:

- `train_end = 2024-06-30`
- `val_end = 2025-06-30`
- `test_end = 2026-03-04`

This defines:

- validation span: `365` calendar days
- test span: `247` calendar days

Repeated holdouts are then created by shifting **all three split boundaries backward by exactly one test-span (`247` days) at a time**.

This produces a sequence of same-shape historical windows with non-overlapping test blocks.

## Benchmark Scope

First benchmark tranche uses the most recent five same-shape windows, including the anchor window.

Rationale:

- covers the recent production-relevant era
- preserves identical split geometry
- gives a distribution of outcomes instead of one holdout
- remains computationally tractable with the live endpoint

## Causality Rules

For each repeated holdout window:

- the base `DLinear` model is trained only on data available before that window's `train_end`
- validation uses only the window's validation block
- test uses only the window's test block
- online memory remains causal relative to each prediction date

Therefore the benchmark asks:

> If we had deployed the exact current production methodology at multiple earlier dates, how often would it have improved `TSM`?

## Primary Metrics

- Path MSE improvement vs raw `TSM`
- `h1`, `h5`, `h20`, `h30` MSE deltas
- Win rate across repeated holdouts

## Secondary Metrics

- Cutoff annotation relative to `2025-01-01`
- Optional downstream financial conversion analyses can be run on the same repeated windows later, but they are not part of the core repeated-holdout forecast benchmark

## Interpretation Rule

This repeated-holdout benchmark is the main retrospective production-style forecast benchmark going forward.

Existing broader fold studies remain useful as secondary robustness checks, but this protocol is the one that best matches the deployed product claim:

> the same live methodology should repeat across prior windows of the same practical shape.
