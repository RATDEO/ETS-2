|    |   horizon |   accuracy |   accuracy_up |   accuracy_down |   accuracy_flat |   n_up |   n_down |   n_flat | model             |
|---:|----------:|-----------:|--------------:|----------------:|----------------:|-------:|---------:|---------:|:------------------|
|  0 |         1 |       0.25 |          0    |        0        |               1 |      2 |        1 |        1 | naive_persistence |
|  1 |         5 |       0    |          0    |      nan        |             nan |      4 |        0 |        0 | naive_persistence |
|  2 |        20 |       0.25 |        nan    |        0        |               1 |      0 |        3 |        1 | naive_persistence |
|  3 |        30 |       0    |        nan    |        0        |             nan |      0 |        4 |        0 | naive_persistence |
|  4 |         1 |       0.25 |          0.5  |        0        |               0 |      2 |        1 |        1 | seasonal_naive    |
|  5 |         5 |       0    |          0    |      nan        |             nan |      4 |        0 |        0 | seasonal_naive    |
|  6 |        20 |       0.25 |        nan    |        0        |               1 |      0 |        3 |        1 | seasonal_naive    |
|  7 |        30 |       0    |        nan    |        0        |             nan |      0 |        4 |        0 | seasonal_naive    |
|  8 |         1 |       0.5  |          0.5  |        0        |               1 |      2 |        1 |        1 | linear_ridge      |
|  9 |         5 |       0.75 |          0.75 |      nan        |             nan |      4 |        0 |        0 | linear_ridge      |
| 10 |        20 |       0.5  |        nan    |        0.666667 |               0 |      0 |        3 |        1 | linear_ridge      |
| 11 |        30 |       1    |        nan    |        1        |             nan |      0 |        4 |        0 | linear_ridge      |
| 12 |         1 |       0.5  |          0.5  |        0        |               1 |      2 |        1 |        1 | linear_lasso      |
| 13 |         5 |       0.25 |          0.25 |      nan        |             nan |      4 |        0 |        0 | linear_lasso      |
| 14 |        20 |       0.75 |        nan    |        1        |               0 |      0 |        3 |        1 | linear_lasso      |
| 15 |        30 |       1    |        nan    |        1        |             nan |      0 |        4 |        0 | linear_lasso      |
| 16 |         1 |       0.5  |          0    |        1        |               1 |      2 |        1 |        1 | tsm               |
| 17 |         5 |       0    |          0    |      nan        |             nan |      4 |        0 |        0 | tsm               |
| 18 |        20 |       0.75 |        nan    |        1        |               0 |      0 |        3 |        1 | tsm               |
| 19 |        30 |       1    |        nan    |        1        |             nan |      0 |        4 |        0 | tsm               |