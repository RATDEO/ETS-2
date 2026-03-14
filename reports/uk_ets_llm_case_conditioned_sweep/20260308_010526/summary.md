# UK ETS TSM+LLM Case-Conditioned Sweep

Generated: 2026-03-08T02:41:09.685565

## Candidate Results

| Candidate | Stage | Hypothesis | Expected | LLM Path MSE | Delta vs TSM | h5 Delta | h20 Delta | h30 Delta | Unique Apply | Unique Rules |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| baseline_365_k6_tk3 | lookback | Control: current case-conditioned settings. | Should reproduce the small positive TSM gain already observed. | 22.618999 | -0.107354 | -0.148820 | -0.152681 | +0.058892 | 49 | 144 |
| recent_180_k6_tk3 | lookback | A 180-day horizon should better match the late-2025 UK regime for TSM correction. | Best chance to cut h30 drift while keeping h5/h20 improvement. | 22.699555 | -0.026798 | -0.105901 | -0.063544 | +0.050694 | 54 | 144 |
| recent_120_k6_tk3 | lookback | A 120-day teaching horizon may isolate the most recent UK regime without starving the matcher. | Could outperform 180 days if the market regime really shifted late in the sample. | 22.722307 | -0.004046 | -0.050172 | -0.041456 | +0.033875 | 49 | 143 |
| recent_180_k4_tk2_localmatch | matching | Shorter regime feature windows should sharpen case matching around the latest UK pattern. | More diverse but more locally relevant rules, with improved h20/h30 sign stability. | 22.725880 | -0.000472 | -0.046052 | -0.067418 | +0.107593 | 52 | 144 |
| recent_180_k4_tk2_strictsign | consensus | Stricter sign agreement and more required evidence should freeze noisy long-horizon corrections. | Smaller active coverage, but cleaner raw test path if contradictory examples are the main problem. | 22.726350 | -0.000003 | -0.000000 | +0.000001 | -0.000003 | 1 | 144 |
| recent_180_k4_tk2 | evidence | Smaller evidence set should force tighter local matching and reduce conflicting horizon guidance. | Lower h30 error and fewer contradictory long-horizon adjustments. | 22.740394 | +0.014041 | -0.039694 | +0.044847 | +0.061647 | 48 | 144 |
| recent_270_k6_tk3 | lookback | Shorter teaching horizon should reduce stale pre-regime examples without losing too much evidence. | Slightly better h30 than baseline, similar h5/h20 gains. | 22.755521 | +0.029168 | -0.131190 | +0.024012 | +0.521033 | 40 | 143 |
| recent_180_k4_tk2_tightbound | bounds | Tighter per-horizon bounds should preserve h5/h20 gains while reducing overcorrection at h30. | Path MSE improves if h30 drag is mostly caused by excessive adjustment magnitude. | 22.769883 | +0.043530 | -0.042109 | +0.040879 | +0.209461 | 44 | 144 |
| recent_90_k6_tk3 | lookback | Very short history may track the local regime best but risks sparse or noisy teaching evidence. | Could improve h30, but may lose stability and diversity. | 22.789328 | +0.062975 | -0.072374 | +0.088358 | +0.181680 | 52 | 144 |
| recent_120_k4_tk2_tightbound | combined | Best recent-regime horizon plus smaller evidence set and tighter bounds may be the strongest anti-h30 configuration. | Best chance to beat the current raw TSM+LLM result if recency and magnitude control both matter. | 22.792157 | +0.065804 | +0.029745 | +0.100032 | +0.069292 | 46 | 144 |

## Best Raw Candidate

- Candidate: `baseline_365_k6_tk3`
- Stage: `lookback`
- Run dir: `/Users/davidwilkinson/Desktop/ETS 2/runs/20260308_010526_501c5b`
- Raw TSM+LLM path MSE: `22.618999`
- Delta vs base TSM: `-0.107354`

## Best Full Run

- Candidate: `baseline_365_k6_tk3`
- Run dir: `/Users/davidwilkinson/Desktop/ETS 2/runs/20260308_022050_c9a217`
- Raw TSM+LLM path MSE: `22.618999`
- Delta vs base TSM: `-0.107354`
