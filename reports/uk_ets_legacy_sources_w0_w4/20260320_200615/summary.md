# W0-W4 Legacy Source Benchmark

Generated: 2026-03-20

## Scope

- Start from the corrected UK-specific `W0-W4` base anchor:
  - shared `DLinear`
  - `seq_len=20`
  - regularized
  - `Huber(beta=0.5)`
  - `energy_interactions16`
- Reintroduce legacy source groups from `/Users/davidwilkinson/Desktop/ETS 2/Data`
- Test whether any of those groups improve the current `W0-W4` benchmark

## Method

- The benchmark built merged data roots under:
  - [/Users/davidwilkinson/Desktop/ETS 2/tmp/legacy_source_merge](/Users/davidwilkinson/Desktop/ETS%202/tmp/legacy_source_merge)
- `uk_ets/Data_auto_uk` stayed the base
- legacy directories were overlaid only where needed:
  - `carbon-market-indices`
  - `emission-spot-primary-market-auction`
  - `volatility-proxy`
  - `usd.xml`
- The broad base-only sweep completed across all `W0-W4`
- I intentionally stopped the follow-up live shortlist after the base pass because no legacy bundle beat the current base on mean `W0-W4` path MSE, so the expected value of a full 15-run live rerun was low

## Base Results

| candidate | mean path MSE | mean h20 MSE | mean h30 MSE |
|---|---:|---:|---:|
| `baseline_energy16` | `48.2365` | `61.5289` | `93.9697` |
| `legacy_auction_volume` | `48.5082` | `62.1140` | `94.5322` |
| `legacy_vstoxx` | `49.0127` | `62.6834` | `95.7157` |
| `legacy_market_compact20` | `49.0953` | `63.4073` | `95.5878` |
| `legacy_icap_secondary` | `49.3208` | `63.3712` | `96.4935` |
| `legacy_indices5` | `49.8594` | `64.9636` | `97.3139` |
| `legacy_market_stack_all` | `51.3571` | `67.8275` | `100.5584` |

Raw rows:
- [base_results.csv](/Users/davidwilkinson/Desktop/ETS%202/reports/uk_ets_legacy_sources_w0_w4/20260320_200615/base_results.csv)

## Per-Window Detail

### `baseline_energy16`

| window | path MSE | h20 | h30 |
|---|---:|---:|---:|
| `W0` | `29.0755` | `39.5853` | `72.0932` |
| `W1` | `23.2579` | `28.1896` | `36.6271` |
| `W2` | `18.5983` | `25.4010` | `35.6188` |
| `W3` | `68.5154` | `89.8141` | `109.4663` |
| `W4` | `101.7353` | `124.6544` | `216.0431` |

### `legacy_auction_volume`

| window | path MSE | h20 | h30 |
|---|---:|---:|---:|
| `W0` | `29.0837` | `39.6814` | `72.1072` |
| `W1` | `22.9386` | `27.7859` | `36.0820` |
| `W2` | `18.7613` | `25.6457` | `35.9664` |
| `W3` | `69.5430` | `91.7838` | `111.5157` |
| `W4` | `102.2144` | `125.6733` | `216.9899` |

### `legacy_vstoxx`

| window | path MSE | h20 | h30 |
|---|---:|---:|---:|
| `W0` | `28.9876` | `39.4611` | `71.5771` |
| `W1` | `23.4745` | `28.4928` | `37.3979` |
| `W2` | `18.8731` | `25.8548` | `36.2116` |
| `W3` | `71.1033` | `93.0582` | `115.4907` |
| `W4` | `102.6252` | `126.5499` | `217.9011` |

### `legacy_market_compact20`

| window | path MSE | h20 | h30 |
|---|---:|---:|---:|
| `W0` | `28.4017` | `38.7408` | `69.7898` |
| `W1` | `23.4307` | `28.4572` | `37.2415` |
| `W2` | `18.7582` | `25.6839` | `35.9189` |
| `W3` | `69.4011` | `91.4975` | `110.4574` |
| `W4` | `105.4848` | `132.6569` | `224.5314` |

### `legacy_icap_secondary`

| window | path MSE | h20 | h30 |
|---|---:|---:|---:|
| `W0` | `28.7369` | `39.0971` | `71.1739` |
| `W1` | `23.1624` | `27.9890` | `36.8023` |
| `W2` | `19.1296` | `26.3043` | `36.7379` |
| `W3` | `71.3368` | `93.4046` | `116.1405` |
| `W4` | `104.2381` | `130.0607` | `221.6131` |

### `legacy_indices5`

| window | path MSE | h20 | h30 |
|---|---:|---:|---:|
| `W0` | `28.3541` | `38.7728` | `69.1601` |
| `W1` | `23.3746` | `28.4402` | `37.0182` |
| `W2` | `18.6628` | `25.4702` | `35.8063` |
| `W3` | `70.3033` | `93.0147` | `113.2583` |
| `W4` | `108.6022` | `139.1199` | `231.3265` |

### `legacy_market_stack_all`

| window | path MSE | h20 | h30 |
|---|---:|---:|---:|
| `W0` | `28.8345` | `39.6507` | `70.7129` |
| `W1` | `22.8071` | `27.7742` | `35.6748` |
| `W2` | `19.5544` | `26.9780` | `37.7835` |
| `W3` | `72.1182` | `95.3711` | `117.2838` |
| `W4` | `113.4713` | `149.3637` | `241.3368` |

## Conclusion

- None of the legacy add-back bundles beat the current improved base on mean `W0-W4` path MSE.
- The closest miss was `legacy_auction_volume`, but it was still worse on average than the current base.
- The legacy source groups tend to help `W0/W1` slightly while making `W3/W4` worse.
- The strongest negative result is the “throw everything in” candidate, which is clearly worse.

So the current evidence says:

- the old `/Data` sources are **not** the missing ingredient
- the hard windows are **not** failing because we dropped those legacy source groups
- we should keep the current corrected UK-specific base as the anchor and look elsewhere for the next gains

