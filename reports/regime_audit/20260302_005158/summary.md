# Benchmark Regime Audit

## Overall

```text
                  model  n_samples  path_mse   h1_mse   h5_mse   h20_mse   h30_mse
Baseline paper CoT-SENT        382 20.604430 1.433835 6.455291 25.792649 41.087699
          4B raw HDELTA        382 20.618083 1.395824 6.393483 25.921562 40.998180
                Raw tsm        382 20.725666 1.384549 6.391054 26.071205 41.277091
 35B raw guarded HDELTA        382 20.872258 1.384549 6.620939 26.321728 40.864872
```

## Thresholds

- Volatility median: `0.016589`
- Sentiment abs median: `0.050000`
- Return abs median: `0.023617`

## Best Model By Regime Bucket

### sentiment_alignment_5d

```text
regime_value                   model  n_samples  path_mse
     aligned  35B raw guarded HDELTA         24 16.393058
   divergent           4B raw HDELTA         28 36.000963
     neutral Baseline paper CoT-SENT        330 19.372391
```

```text
regime_value                   model  winner_count  winner_share  avg_margin_to_second
     aligned  35B raw guarded HDELTA            14      0.583333              0.939309
   divergent  35B raw guarded HDELTA            13      0.464286              1.307841
     neutral Baseline paper CoT-SENT           159      0.481818              1.879165
```

### trend_regime_20d

```text
regime_value                   model  n_samples  path_mse
        down Baseline paper CoT-SENT        136 19.689594
        flat  35B raw guarded HDELTA         18 29.639063
          up           4B raw HDELTA        228 20.266885
```

```text
regime_value                   model  winner_count  winner_share  avg_margin_to_second
        down Baseline paper CoT-SENT            67      0.492647              1.615591
        flat  35B raw guarded HDELTA             9      0.500000              1.401558
          up Baseline paper CoT-SENT            99      0.434211              1.975217
```

### trend_vol_regime

```text
  regime_value                   model  n_samples  path_mse
down__high_vol Baseline paper CoT-SENT         77 28.872135
 down__low_vol Baseline paper CoT-SENT         59  7.705599
flat__high_vol           4B raw HDELTA         12 34.440593
 flat__low_vol  35B raw guarded HDELTA          6 18.962565
  up__high_vol Baseline paper CoT-SENT        102 22.731910
   up__low_vol  35B raw guarded HDELTA        126 17.944181
```

```text
  regime_value                   model  winner_count  winner_share  avg_margin_to_second
down__high_vol Baseline paper CoT-SENT            38      0.493506              1.923931
 down__low_vol Baseline paper CoT-SENT            29      0.491525              1.211559
flat__high_vol Baseline paper CoT-SENT             5      0.416667              1.789711
 flat__low_vol  35B raw guarded HDELTA             4      0.666667              1.762308
  up__high_vol Baseline paper CoT-SENT            44      0.431373              2.157403
   up__low_vol Baseline paper CoT-SENT            55      0.436508              1.829467
```

### volatility_regime

```text
regime_value                   model  n_samples  path_mse
    high_vol Baseline paper CoT-SENT        191 26.054460
     low_vol  35B raw guarded HDELTA        191 14.896868
```

```text
regime_value                   model  winner_count  winner_share  avg_margin_to_second
    high_vol Baseline paper CoT-SENT            87      0.455497              2.034295
     low_vol Baseline paper CoT-SENT            85      0.445026              1.611648
```

## Gating Candidate Summary

```json
{
  "families": [
    {
      "regime_family": "sentiment_alignment_5d",
      "values": [
        {
          "regime_value": "aligned",
          "best_model": "35B raw guarded HDELTA",
          "best_path_mse": 16.393057975323284,
          "n_samples": 24,
          "runner_up_model": "Raw tsm",
          "margin_to_runner_up": 0.2766033170774307
        },
        {
          "regime_value": "divergent",
          "best_model": "4B raw HDELTA",
          "best_path_mse": 36.00096331225688,
          "n_samples": 28,
          "runner_up_model": "Raw tsm",
          "margin_to_runner_up": 0.45704771532197697
        },
        {
          "regime_value": "neutral",
          "best_model": "Baseline paper CoT-SENT",
          "best_path_mse": 19.37239067123385,
          "n_samples": 330,
          "runner_up_model": "4B raw HDELTA",
          "margin_to_runner_up": 0.22005675700823346
        }
      ],
      "candidate": true,
      "distinct_winners": [
        "35B raw guarded HDELTA",
        "4B raw HDELTA",
        "Baseline paper CoT-SENT"
      ]
    },
    {
      "regime_family": "trend_regime_20d",
      "values": [
        {
          "regime_value": "down",
          "best_model": "Baseline paper CoT-SENT",
          "best_path_mse": 19.689593730490575,
          "n_samples": 136,
          "runner_up_model": "Raw tsm",
          "margin_to_runner_up": 0.2986672650888025
        },
        {
          "regime_value": "flat",
          "best_model": "35B raw guarded HDELTA",
          "best_path_mse": 29.639062833750714,
          "n_samples": 18,
          "runner_up_model": "4B raw HDELTA",
          "margin_to_runner_up": 0.016557821230314573
        },
        {
          "regime_value": "up",
          "best_model": "4B raw HDELTA",
          "best_path_mse": 20.266884678317318,
          "n_samples": 228,
          "runner_up_model": "Baseline paper CoT-SENT",
          "margin_to_runner_up": 0.015270578990072892
        }
      ],
      "candidate": false,
      "distinct_winners": [
        "35B raw guarded HDELTA",
        "4B raw HDELTA",
        "Baseline paper CoT-SENT"
      ]
    },
    {
      "regime_family": "trend_vol_regime",
      "values": [
        {
          "regime_value": "down__high_vol",
          "best_model": "Baseline paper CoT-SENT",
          "best_path_mse": 28.872135493610845,
          "n_samples": 77,
          "runner_up_model": "4B raw HDELTA",
          "margin_to_runner_up": 0.20899690992434117
        },
        {
          "regime_value": "down__low_vol",
          "best_model": "Baseline paper CoT-SENT",
          "best_path_mse": 7.705598548113279,
          "n_samples": 59,
          "runner_up_model": "35B raw guarded HDELTA",
          "margin_to_runner_up": 0.2699878618414999
        },
        {
          "regime_value": "flat__high_vol",
          "best_model": "4B raw HDELTA",
          "best_path_mse": 34.4405929880453,
          "n_samples": 12,
          "runner_up_model": "Raw tsm",
          "margin_to_runner_up": 0.07463812392640534
        },
        {
          "regime_value": "flat__low_vol",
          "best_model": "35B raw guarded HDELTA",
          "best_path_mse": 18.96256479751347,
          "n_samples": 6,
          "runner_up_model": "4B raw HDELTA",
          "margin_to_runner_up": 1.123111191338996
        },
        {
          "regime_value": "up__high_vol",
          "best_model": "Baseline paper CoT-SENT",
          "best_path_mse": 22.731910432242238,
          "n_samples": 102,
          "runner_up_model": "4B raw HDELTA",
          "margin_to_runner_up": 0.1692138909798686
        },
        {
          "regime_value": "up__low_vol",
          "best_model": "35B raw guarded HDELTA",
          "best_path_mse": 17.944180898128216,
          "n_samples": 126,
          "runner_up_model": "4B raw HDELTA",
          "margin_to_runner_up": 0.1902240676471294
        }
      ],
      "candidate": true,
      "distinct_winners": [
        "35B raw guarded HDELTA",
        "4B raw HDELTA",
        "Baseline paper CoT-SENT"
      ]
    },
    {
      "regime_family": "volatility_regime",
      "values": [
        {
          "regime_value": "high_vol",
          "best_model": "Baseline paper CoT-SENT",
          "best_path_mse": 26.054459738112943,
          "n_samples": 191,
          "runner_up_model": "4B raw HDELTA",
          "margin_to_runner_up": 0.06307425087872431
        },
        {
          "regime_value": "low_vol",
          "best_model": "35B raw guarded HDELTA",
          "best_path_mse": 14.896867958830194,
          "n_samples": 191,
          "runner_up_model": "4B raw HDELTA",
          "margin_to_runner_up": 0.2217649754189459
        }
      ],
      "candidate": false,
      "distinct_winners": [
        "35B raw guarded HDELTA",
        "Baseline paper CoT-SENT"
      ]
    }
  ],
  "recommended_families": [
    "sentiment_alignment_5d",
    "trend_vol_regime"
  ]
}
```
