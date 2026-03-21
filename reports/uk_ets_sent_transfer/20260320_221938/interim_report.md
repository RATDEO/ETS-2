# UK SENT Transfer Interim Report

Generated: 2026-03-20

## Corpus

- Source mode: `govuk_plus_elexon`
- Raw official-source headlines: `65`
- Filtered UK ETS headlines: `31`
- Labeled headlines: `31`
- Daily sentiment rows: `28`
- Date range: `2021-05-23` to `2026-03-12`
- Source mix after filtering: `31/31 gov.uk`

## Completed Results

The completed benchmark tranche is:

- `base_energy16`
- `sent_feature_raw`
- `sent_feature_importance`

These runs use the current improved UK base:

- shared `DLinear`
- `seq_len=20`
- regularized training
- `Huber(beta=0.5)`
- curated `energy_interactions16`

## Path-MSE Results By Window

| Window | Base | SENT raw | Raw uplift | SENT importance | Importance uplift |
|---|---:|---:|---:|---:|---:|
| `W0` | `29.0755` | `28.5155` | `+1.93%` | `27.9860` | `+3.75%` |
| `W1` | `23.2579` | `23.3910` | `-0.57%` | `23.4654` | `-0.89%` |
| `W2` | `18.5983` | `18.7430` | `-0.78%` | `18.7054` | `-0.58%` |
| `W3` | `68.5154` | `70.0929` | `-2.30%` | `70.0682` | `-2.27%` |
| `W4` | `101.7353` | `102.2001` | `-0.46%` | `102.2001` | `-0.46%` |

## Aggregate Read

- Mean path uplift, `sent_feature_raw`: `-0.44%`
- Mean path uplift, `sent_feature_importance`: `-0.09%`

The UK official-source sentiment transfer is therefore:

- genuinely positive on `W0`
- strongest with the importance-weighted daily series
- not robust on `W1-W4`

The prompt-side `CoT-SENT-RF` runs are still in progress and were not part of this interim table.
