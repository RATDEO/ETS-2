# TSM Live Gate Feature Ablation Plan

## Objective
- Test whether the current 29-feature learned gbdt live gate is actually optimal.
- Compare semantic feature-family subsets against MI-ranked top-k subsets.
- Keep the live TSM online-memory architecture fixed and only change the learned gate feature columns.

## Ranking Source
- Validation feature ranking derived from [val_learned_gate_features.csv](/Users/davidwilkinson/Desktop/ETS 2/runs/20260316_103639_143185/results/online_memory_gate/TSM_LLM-COT-RF-HDELTA/val_learned_gate_features.csv) and [val_learned_gate_labels.csv](/Users/davidwilkinson/Desktop/ETS 2/runs/20260316_103639_143185/results/online_memory_gate/TSM_LLM-COT-RF-HDELTA/val_learned_gate_labels.csv).

## Candidates
- `full_29_control` (29 features): Rerun the current winning learned gbdt gate with the full 29-feature schema. Expected path MSE around 3.70-3.78. This is the control and current bar to beat.
- `top_25_mi` (25 features): Trim the weakest four validation-ranked features while keeping most of the full schema. Expected path MSE around 3.70-3.80 if the current full schema has mild redundancy.
- `top_20_mi` (20 features): Use only the top 20 validation-ranked learned-gate features. Expected path MSE around 3.70-3.82 if the winner mostly depends on the strongest memory and forecast features.
- `top_15_mi` (15 features): Force the gate down to a medium-sized 15-feature schema. Expected path MSE around 3.74-3.88. This should reveal whether the full gate is overparameterized.
- `top_10_mi` (10 features): Force the gate to a compact 10-feature schema. Expected path MSE around 3.80-3.98. If this still works, the gate can likely be simplified materially.
- `top_5_mi` (5 features): Stress-test whether only the highest-ranked features carry most of the live signal. Expected path MSE around 3.95-4.20. Likely too sparse, but a useful lower bound.
- `memory_full_20` (20 features): Use only memory-bank evidence and similarity/helpfulness features, with no direct forecast-shape regime inputs. Expected path MSE around 3.82-4.05. If this holds up, the gate is mostly a memory-quality model.
- `counts_signals_14` (14 features): Use only count/signal/support features, dropping similarity/helpfulness and forecast-shape inputs. Expected path MSE around 3.90-4.12. This tests whether raw memory balance is enough on its own.
- `forecast_regime_9` (9 features): Use only direct forecast-shape and regime-profile features, with no explicit memory-quality inputs. Expected path MSE around 3.95-4.18. This tests whether the gate is really just a contextual base-forecast filter.
- `similarity_helpfulness_6` (6 features): Use only similarity/helpfulness quality features from the memory banks. Expected path MSE around 4.00-4.20. This should be informative, but likely too weak on its own.
