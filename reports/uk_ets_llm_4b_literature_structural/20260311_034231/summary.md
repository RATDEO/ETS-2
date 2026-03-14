# UK ETS 4B Literature Structural Sweep

Generated: 2026-03-11T06:00:52.549768

## Checkpoint

- Base checkpoint run: `/Users/davidwilkinson/Desktop/ETS 2/runs/20260310_195100_69e04e`
- Checkpoint raw `TSM+LLM-COT-RF-HDELTA` path MSE: `22.198106`
- Reference `linear_ridge` path MSE: `22.419955`

## Candidate Results

| Candidate | Change | Literature basis | Aim | Expected | Actual | Vs checkpoint | Outcome | h5 vs checkpoint | h20 vs checkpoint | h30 vs checkpoint |
|---|---|---|---|---|---:|---:|---|---:|---:|---:|
| control_recent_high_error_k4 | Re-run the current 4B winner unchanged as a control. | Control | Verify the new structural code path reproduces the current checkpoint before changing evidence policy. | 22.14-22.24 | 22.165979 | -0.032127 | within | +0.001280 | -0.016757 | -0.120225 |
| recent_long_error_k4 | Replace path-MSE retrieval with recent long-horizon error retrieval centered on h20/h30. | Time-series LLM papers emphasize horizon/task alignment more than generic ICL breadth. | Bias the demo bank toward the horizons where the 4B LLM actually adds value. | 22.02-22.14 | 22.280864 | +0.082758 | missed | +0.013117 | +0.100848 | +0.325868 |
| recent_consistent_long_error_k4 | Keep only recent long-horizon cases whose h20 and h30 error signs agree. | Comparable demonstration work argues that conflicting examples degrade small-model in-context learning. | Remove mixed-tail demonstrations that teach contradictory long-horizon corrections. | 21.98-22.12 | 22.277726 | +0.079621 | missed | +0.011189 | +0.093691 | +0.320421 |
| balanced_long_horizon_k4 | Split the evidence set between h20-heavy and h30-heavy mistakes. | Demonstration-selection literature repeatedly finds coverage and diversity help more than raw score ranking alone. | Avoid letting one long horizon dominate the entire teaching set. | 22.00-22.15 | 22.278441 | +0.080336 | missed | -0.017420 | +0.185924 | +0.335243 |
| prototype_recent_long_error_k4 | Take prototypes from the top recent long-error pool instead of the top few ranked examples directly. | Active-example and retrieval papers show redundancy reduction is important in small-context ICL. | Reduce redundant demonstrations and improve coverage of distinct correction patterns. | 22.05-22.18 | 22.358486 | +0.160380 | missed | +0.028844 | +0.276390 | +0.490927 |
| skill_tag_recent_high_error_k4 | Filter recent hard examples through coarse regime-skill tags before ranking them. | Skill-KNN suggests abstract skill matching beats surface-level similarity for few-shot selection. | Retrieve demonstrations that match the same market regime, not just the same error magnitude. | 22.04-22.18 | 22.132513 | -0.065592 | within | -0.000245 | -0.069317 | -0.131469 |
| feedback_recent_high_error_k4 | Add explicit hindsight feedback sentences to each recent high-error teaching example. | Chain of Hindsight shows compact feedback can outperform raw trajectories alone. | Convert raw historical mistakes into short natural-language lessons the 4B model can reuse more easily. | 21.98-22.12 | 21.999591 | -0.198515 | within | -0.017599 | -0.178836 | -0.741277 |
| support_plus_freeze_counterexample_k4 | Augment recent high-error support examples with one similar low-error freeze counterexample. | Supportive/comparable demonstration work argues explicit counterexamples sharpen decision boundaries. | Teach the 4B model both when to correct and when not to create tail drift. | 21.96-22.10 | 22.009833 | -0.188272 | within | -0.047357 | -0.277948 | -0.405673 |
| ridge_context_recent_high_error_k4 | Keep recent high-error retrieval but show the ridge forecast deltas as a teacher signal in the examples and current case. | Teacher/distillation literature for small models suggests auxiliary teacher signals help more than added prompt complexity. | Let the LLM see what a stronger non-LLM forecaster would do without changing the base TSM path directly. | 21.92-22.08 | 22.615683 | +0.417577 | missed | -0.017176 | +0.791440 | +0.920839 |
| ridge_gain_selection_k4 | Retrieve examples where the ridge teacher most improved on the long horizon, and show teacher summaries. | Active example selection favors demonstrations with the highest expected utility, not the largest raw loss alone. | Prioritize demonstrations whose correction pattern is both difficult and actually recoverable by a stronger model. | 21.88-22.06 | 22.677760 | +0.479655 | missed | -0.028895 | +0.917741 | +1.079055 |
| ridge_gain_plus_counterexample_k4 | Combine ridge-gain retrieval with one similar low-gain counterexample. | Teacher-guided selection plus comparable demos is the highest-upside structural combination from the reviewed literature. | Keep the teacher-guided upside while explicitly teaching the model when not to trust a correction pattern. | 21.86-22.05 | 22.558470 | +0.360364 | missed | -0.029029 | +0.699656 | +0.885417 |

## Best Candidate

- Candidate: `feedback_recent_high_error_k4`
- Change: Add explicit hindsight feedback sentences to each recent high-error teaching example.
- Literature basis: Chain of Hindsight shows compact feedback can outperform raw trajectories alone.
- Aim: Convert raw historical mistakes into short natural-language lessons the 4B model can reuse more easily.
- Expected range: `21.98-22.12`
- Actual raw path MSE: `21.999591`
- Delta vs checkpoint raw LLM: `-0.198515`
- Delta vs base TSM: `-0.726762`
- Delta vs ridge: `-0.420365`
- Outcome vs expectation: `within`
- Run dir: `/Users/davidwilkinson/Desktop/ETS 2/runs/20260311_045606_c3f654`
