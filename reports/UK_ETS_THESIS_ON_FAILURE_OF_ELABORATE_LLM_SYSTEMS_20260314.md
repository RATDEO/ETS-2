# UK ETS Thesis: Why the More Elaborate LLM Systems Fail, and What That Implies for the Project

Generated: 2026-03-14

## Thesis

The elaborate LLM systems usually fail in this project because the task is not an open-ended reasoning problem. It is a constrained forecast-correction problem with a strong base model, a low-dimensional action space, and a harsh loss function that punishes even small medium-horizon mistakes. In that setting, extra agentic structure, extra retrieval logic, and extra verification layers usually add variance faster than they add useful information.

The strongest systems in this project do not behave like autonomous analysts. They behave like narrow correction modules. They work when they compress a small amount of relevant historical signal into a bounded numerical adjustment. They fail when they try to reason too broadly, retrieve too much, justify too much, or coordinate too many moving parts.

## Core Evidence

### 1. The best 35B result came from a simple regime abstraction, not from the most elaborate pipeline

The strongest 35B result remains the old `step_back_regime` run in [summary.md](/Users/davidwilkinson/Desktop/ETS%202/reports/uk_ets_llm_35b_structural_literature/20260309_225959/summary.md) and [path_metrics.csv](/Users/davidwilkinson/Desktop/ETS%202/runs/20260309_234219_249d62/results/path_metrics.csv):

- `TSM+LLM-COT-RF-HDELTA`: `21.484623`
- `linear_ridge`: `22.419955`
- base `tsm`: `22.726353`

That winner improved `h20` and `h30` while remaining structurally simple: infer the regime first, then map it into bounded horizon actions.

By contrast, the more elaborate 35B literature variants in [summary.md](/Users/davidwilkinson/Desktop/ETS%202/reports/uk_ets_llm_35b_structural_literature/20260309_225959/summary.md) mostly failed:

- `tree_of_thought_consensus`: `21.773477`
- `react_evidence`: `21.808505`
- `skeleton_then_fill`: `21.931373`
- `chain_of_verification`: `22.091358`
- `self_refine_verifier`: `22.373823`
- `self_rag_critique`: `22.459442`
- `debate_consensus`: `22.697666`
- `rarr_attribution`: `22.701313`

The pattern is clear: more branches, more internal checks, and more evidence policing did not improve forecast quality. They degraded it.

### 2. In the later 35B tool-backed sweep, only the smallest relaxation helped

In the 35B tool follow-up sweep at [summary.md](/Users/davidwilkinson/Desktop/ETS%202/reports/uk_ets_llm_35b_tool_followup/20260314_030154/summary.md), the only improvement over the current tool-backed baseline was the smallest one:

- baseline `optional_tools_default`: `21.509699`
- best new candidate `numeric_only_default`: `21.502944`

Everything more elaborate missed:

- `react_evidence_optional_tools`: `22.072692`
- `plan_and_solve_optional_tools`: `21.700990`
- `skill_tag_recent_high_error_tools`: `22.041922`
- `balanced_long_horizon_tools`: `22.369328`
- `counterexample_recent_high_error_tools`: `21.635979`
- `step_back_optional_tools_k6`: `21.789554`
- `step_back_numeric_optional_k6`: `21.793566`

This matters. The winning modification was not "more intelligence." It was removing one overly strong control layer. Exact arithmetic helped a little. Extra structure hurt.

### 3. 4B also favored compressed, usable guidance rather than broader intelligence scaffolding

On 4B, the strongest structural improvement in [summary.md](/Users/davidwilkinson/Desktop/ETS%202/reports/uk_ets_llm_4b_literature_structural/20260311_034231/summary.md) was `feedback_recent_high_error_k4`:

- checkpoint raw LLM path MSE: `22.198106`
- best 4B structural variant: `21.999591`

What helped was adding short hindsight lessons to recent high-error examples. That is a compression move. It makes demonstrations easier for a smaller model to reuse.

What failed on 4B was exactly the opposite:

- long-horizon retrieval variants
- prototype balancing
- ridge-teacher context
- ridge-gain selection
- ridge-gain plus counterexample

Those failed because they increased conceptual load, not because they lacked sophistication.

### 4. Deterministic tools help only when they solve a precise failure mode

The 4B tool results in [uk_ets_tool_batch2_20260313.md](/Users/davidwilkinson/Desktop/ETS%202/reports/uk_ets_tool_batch2_20260313.md) show the same shape.

The useful additions were:

- numeric analysis
- delta verifier

Those tools help because they target concrete error modes: arithmetic sizing and bad delta shape.

The tools that did not help were:

- market microstructure state
- freeze counterexample tool
- combined larger tool stacks

Again, more context was not better. The project benefited when a tool eliminated a specific known failure, not when it broadened the model's informational field.

### 5. Rolling-origin evaluation shows the LLM edge is conditional, not universal

The rolling-origin summary in [summary.md](/Users/davidwilkinson/Desktop/ETS%202/reports/uk_ets_llm_4b_rolling_origin_forecast/20260312_021028/summary.md) is important because it explains why elaborate systems are so fragile here.

- `h20`: LLM beat TSM on path MSE in `3/5` folds, but beat it on net return in only `2/5`
- `h30`: LLM beat TSM on path MSE in `3/5` folds, but beat it on net return in only `1/5`

This means the LLM layer is not a universal performance multiplier. It is a regime-conditional overlay. When the environment shifts, a more elaborate system has more ways to be wrong.

## Why the Elaborate Systems Fail

### 1. The task rewards calibration, not expansiveness

The LLM is not being asked to create a full analysis of the carbon market. It is being asked to slightly improve a 30-step path that already comes from a tuned base forecaster. The incremental correction is small relative to the full signal.

That means the optimal behavior is usually:

- decide whether to intervene at all
- make a small directional or amplitude correction
- avoid damaging neighboring horizons

Most elaborate prompting methods were invented for discrete reasoning benchmarks where the problem is under-specified until the model reasons through it. This task is almost the opposite. The problem is already mostly specified by the base forecast and the local historical context. The LLM only needs to avoid making it worse.

When the task is narrow and the action space is small, elaborate reasoning often produces over-explanation rather than better decisions.

### 2. Every added module creates another interference surface

The failed systems almost all add another control surface:

- extra reflection stages
- extra example-routing logic
- extra consensus logic
- extra evidence gating
- extra teacher signals
- extra verification passes

In principle each module should reduce one failure mode. In practice, each module also changes the distribution of outputs seen by the next stage.

That creates interference:

- good medium-horizon deltas get softened
- useful long-horizon adjustments get clipped
- example diversity collapses into a template
- the model spends capacity satisfying prompt structure instead of sizing the correction well

This is especially clear in the worse 35B variants, where unique apply outputs fell sharply in several failed systems. The system became more controlled, but also more generic.

### 3. Retrieval quality matters more than retrieval cleverness

The project repeatedly shows that retrieval construction helps only when it increases direct reuse value.

What helped:

- recent hard examples
- hindsight feedback attached to those examples
- small `k`
- direct compressible lessons

What hurt:

- balanced or diversified example pools
- skill tags on top of forced tool stacks
- teacher-guided retrieval
- long-horizon prototype selection

This suggests that the retrieval bottleneck is not "finding more theoretically relevant examples." It is "finding demonstrations the model can actually imitate under the current prompt contract."

In other words, the limiting factor is cognitive usability, not retrieval sophistication.

### 4. Tooling helps when it is surgical, not supervisory

The numeric tool is useful because it is surgical. It gives exact local quantities the model can use immediately.

The verifier is mixed because it is supervisory. It does not supply new information. It constrains behavior after the fact. That is helpful on 4B, where the model is noisier, but less helpful on 35B, where it can suppress useful corrections.

This suggests a general rule for the project:

- information tools are good when they directly reduce uncertainty in a known weak dimension
- policing tools are only good when the model is clearly unstable in that dimension

The elaborate systems often failed because they turned the pipeline into a bureaucracy. Too much of the system existed to constrain outputs that were not the true bottleneck.

### 5. The models are not failing to think; they are failing to preserve the right signal

The failed systems are often easy to misread as "the model is not smart enough." That is not the central problem.

The stronger explanation is:

- the useful correction signal is small
- it exists mainly at `h20` and `h30`
- elaborate systems distort that signal while trying to make it more explicit, safer, or better justified

That is why many variants worsen `h20` and `h30` despite sounding more rigorous.

They are not under-reasoning. They are over-processing.

## What This Suggests About the Current Project

### 1. This is a bounded correction project, not an autonomous market-intelligence project

The strongest systems in this repo are not the ones that behave most like analysts. They are the ones that behave like calibrated residual correctors.

That suggests the project's real identity is:

- strong base forecaster
- small LLM overlay
- limited horizon-specific interventions
- deterministic tools for arithmetic and shape control
- tightly curated demonstration memory

The project should be optimized around that identity rather than around generic agentic LLM ambition.

### 2. Model-specific pipelines are justified

The evidence says 4B and 35B do not want the same pipeline.

For 4B:

- compact hindsight feedback helps
- deterministic verifier helps
- extra teacher or retrieval complexity hurts

For 35B:

- high-level regime abstraction can help
- numeric arithmetic helps slightly
- verifier hurts slightly
- step-back plus modern tool stack is a bad mix

So the project should stop treating "one unified best LLM pipeline" as the goal. The right goal is a shared infrastructure with model-specific policies.

### 3. The correct next unit of improvement is calibration, not more cognition

If the elaborate systems fail because they are over-processing weak signals, then the next improvements should focus on:

- better optional tool use
- cleaner accept/reject decisions
- horizon-specific magnitude control
- regime gating
- smaller, more reusable demonstration units

They should not focus on:

- more branches
- more debate
- more search
- more open-ended retrieval
- broader internal prompting ceremonies

### 4. Claims about trading uplift should remain conservative

The rolling-origin evidence shows the LLM effect is modest and conditional. That means the project can legitimately claim:

- the LLM layer can improve forecast error in the right setup
- the gain is concentrated at medium and long horizons
- some structural tool support helps

But it should not yet claim:

- a robust universal trading edge
- that more elaborate reasoning reliably improves market performance
- that generic CoT-style sophistication transfers directly into this domain

### 5. The project is actually closer to "time-series control" than to "LLM reasoning"

The repeated winners all share one property: they improve control over the correction process.

They improve:

- what examples enter the prompt
- how compactly they are represented
- whether arithmetic is exact
- whether deltas stay bounded
- whether the model only changes the part of the path that matters

This is much closer to control engineering than to free-form reasoning research.

That is a useful realization. It suggests the most productive future work is not to imitate bigger and bigger prompting papers. It is to make the correction layer more stable, sparse, and selectively active.

## Final Conclusion

The elaborate systems fail because they are solving the wrong problem. They assume the challenge is extracting more intelligence from the LLM. The project evidence says the challenge is preserving a small, valuable correction signal without damaging the strong base forecast beneath it.

That changes the interpretation of almost every result:

- simpler systems do not win because they are crude
- they win because they are better aligned to the task

The lesson for the project is therefore not "we need a smarter agent." It is:

we need a narrower, more disciplined correction mechanism, with model-specific policies and only the minimum structure necessary to improve the base path.

Under that view, the project is already succeeding. It has discovered the shape of the useful LLM contribution. The remaining work is not to make the system more elaborate. It is to make the useful part of the system more selective, more calibrated, and harder to destabilize.
