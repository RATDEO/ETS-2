# UK ETS LLM Targeted Tranche 2

Generated: 2026-03-08T19:30:33.599359

| Candidate | Expected | LLM Path MSE | Delta vs TSM | h5 Delta | h20 Delta | h30 Delta | Run Dir |
|---|---|---:|---:|---:|---:|---:|---|
| coherence_guard | About 22.57-22.59 path MSE if conflicting h20/h30 rule families are the residual drag. | 22.573653 | -0.152699 | -0.025528 | -0.193455 | -0.356214 | /Users/davidwilkinson/Desktop/ETS 2/runs/20260308_185436_003dbe |
| baseline_control | About 22.59-22.60 path MSE. | 22.595492 | -0.130861 | -0.025528 | -0.130082 | -0.346211 | /Users/davidwilkinson/Desktop/ETS 2/runs/20260308_184246_f626a7 |
| horizon_specific_matching | About 22.55-22.58 path MSE if shared-example contamination is still hurting h20/h30 decisions. | 22.642395 | -0.083957 | +0.001797 | -0.127866 | -0.171327 | /Users/davidwilkinson/Desktop/ETS 2/runs/20260308_190625_f71666 |
| horizon_specific_plus_coherence | Best upside in this tranche, about 22.53-22.57, with some risk of collapsing back toward the baseline. | 22.659473 | -0.066880 | +0.001797 | -0.092948 | -0.131336 | /Users/davidwilkinson/Desktop/ETS 2/runs/20260308_191829_ce411a |
