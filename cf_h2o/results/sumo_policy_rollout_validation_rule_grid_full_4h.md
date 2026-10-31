# Libsumo Policy Rollout Validation

Verdict: **PASS**

This is a closed-loop diagnostic over generated SUMO bus-stop events. First-arrival warm-up events are excluded from metrics. It is not yet a calibrated field counterfactual.

| Policy | Cities | Events | Warm-up | Reward | Headway error | Bunching | Hold sec |
|---|---:|---:|---:|---:|---:|---:|---:|
| daganzo_a03_policy | 4 | 9583 | 7967 | -531.128 | 529.800 | 0.340 | 26.566 |
| daganzo_a04_policy | 4 | 9566 | 7972 | -531.228 | 529.880 | 0.340 | 26.961 |
| daganzo_a05_policy | 4 | 9558 | 7972 | -531.109 | 529.748 | 0.342 | 27.203 |
| daganzo_a07_policy | 4 | 9551 | 7977 | -531.511 | 530.138 | 0.343 | 27.461 |
| daganzo_a08_policy | 4 | 9550 | 7981 | -531.013 | 529.636 | 0.343 | 27.546 |
| daganzo_policy | 4 | 9555 | 7976 | -531.040 | 529.673 | 0.343 | 27.337 |
| no_hold | 4 | 10000 | 7102 | -683.843 | 683.843 | 0.492 | 0.000 |
| threshold_aggressive_policy | 4 | 9543 | 7988 | -538.194 | 536.792 | 0.341 | 28.049 |
| threshold_equalization_policy | 4 | 9574 | 7947 | -532.651 | 531.349 | 0.336 | 26.030 |
| threshold_fine_policy | 4 | 9552 | 7970 | -535.917 | 534.550 | 0.340 | 27.344 |
| threshold_soft_policy | 4 | 9699 | 7805 | -560.790 | 559.769 | 0.364 | 20.421 |
