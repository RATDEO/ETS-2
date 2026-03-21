# Helpful-Case Mining Summary

Generated: 2026-03-21T18:09:09.871604

## Source Runs
- `W1`: `/Users/davidwilkinson/Desktop/ETS 2/runs/20260320_182015_da9e9d`
- `W2`: `/Users/davidwilkinson/Desktop/ETS 2/runs/20260320_183041_c7484b`
- `W3`: `/Users/davidwilkinson/Desktop/ETS 2/runs/20260320_183637_a4d80c`
- `W4`: `/Users/davidwilkinson/Desktop/ETS 2/runs/20260320_184725_2cded6`

## Overall
- Cases: `581`
- Mean path uplift: `0.55%`
- Mean h20 uplift: `-99.71%`
- Mean h30 uplift: `-1322.31%`
- Helpful loose rate: `20.31%`
- Helpful strict rate: `9.81%`
- Harmful strict rate: `10.15%`

## By Window

| window   |   path_uplift_pct |   h20_uplift_pct |   h30_uplift_pct |   helpful_loose |   helpful_long_only |   helpful_strict |   harmful_strict |
|:---------|------------------:|-----------------:|-----------------:|----------------:|--------------------:|-----------------:|-----------------:|
| W1       |       0.0113506   |       -3.47206   |        0.0147361 |       0.272727  |           0.160839  |       0.146853   |        0.111888  |
| W2       |      -0.00418921  |       -0.0772143 |       -0.361701  |       0.0958904 |           0.0479452 |       0.00684932 |        0.0958904 |
| W3       |       0.0150181   |       -0.0773236 |        0.0142863 |       0.30137   |           0.219178  |       0.205479   |        0.109589  |
| W4       |      -0.000234576 |       -0.412594  |      -52.2875    |       0.143836  |           0.0684932 |       0.0342466  |        0.0890411 |

## Volatility x Base-Move Regimes

| vol_regime   | move_regime   |   n_cases |   mean_path_uplift_pct |   mean_h20_uplift_pct |   mean_h30_uplift_pct |   helpful_strict_rate |   harmful_strict_rate |
|:-------------|:--------------|----------:|-----------------------:|----------------------:|----------------------:|----------------------:|----------------------:|
| low_vol      | small_move    |        82 |             0.0165433  |           -1.32382    |         -93.0912      |             0.158537  |             0.146341  |
| mid_vol      | large_move    |        48 |             0.0146705  |            0.0245468  |          -0.154229    |             0.145833  |             0.0416667 |
| low_vol      | large_move    |        40 |             0.0126844  |           -0.550686   |           0.0304204   |             0.1       |             0.05      |
| high_vol     | small_move    |        38 |             0.0103488  |           -0.00904183 |           4.49254e-05 |             0.0789474 |             0.0263158 |
| high_vol     | mid_move      |        50 |             0.00731366 |           -0.0350159  |          -0.0106638   |             0.14      |             0.08      |
| high_vol     | large_move    |       106 |             0.00704077 |           -0.0968484  |           0.0150905   |             0.141509  |             0.122642  |
| low_vol      | mid_move      |        72 |             0.00144643 |           -0.0905904  |           0.00630599  |             0.0694444 |             0.111111  |
| mid_vol      | mid_move      |        71 |            -0.00107826 |           -0.0100645  |          -0.630916    |             0.0422535 |             0.112676  |
| mid_vol      | small_move    |        74 |            -0.0125817  |           -5.81496    |           0.00463234  |             0         |             0.121622  |

## Candidate Helpful Regimes

| vol_regime   | move_regime   |   n_cases |   mean_path_uplift_pct |   mean_h20_uplift_pct |   mean_h30_uplift_pct |   helpful_strict_rate |   harmful_strict_rate |
|:-------------|:--------------|----------:|-----------------------:|----------------------:|----------------------:|----------------------:|----------------------:|
| low_vol      | small_move    |        82 |             0.0165433  |           -1.32382    |         -93.0912      |             0.158537  |             0.146341  |
| mid_vol      | large_move    |        48 |             0.0146705  |            0.0245468  |          -0.154229    |             0.145833  |             0.0416667 |
| low_vol      | large_move    |        40 |             0.0126844  |           -0.550686   |           0.0304204   |             0.1       |             0.05      |
| high_vol     | small_move    |        38 |             0.0103488  |           -0.00904183 |           4.49254e-05 |             0.0789474 |             0.0263158 |
| high_vol     | mid_move      |        50 |             0.00731366 |           -0.0350159  |          -0.0106638   |             0.14      |             0.08      |
| high_vol     | large_move    |       106 |             0.00704077 |           -0.0968484  |           0.0150905   |             0.141509  |             0.122642  |
| low_vol      | mid_move      |        72 |             0.00144643 |           -0.0905904  |           0.00630599  |             0.0694444 |             0.111111  |
