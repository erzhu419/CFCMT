# Libsumo Policy Rollout Validation

Verdict: **PASS**

This is a closed-loop diagnostic over generated SUMO bus-stop events. First-arrival warm-up events are excluded from metrics. It is not yet a calibrated field counterfactual.

| Policy | Cities | Events | Warm-up | Reward | Headway error | Bunching | Hold sec |
|---|---:|---:|---:|---:|---:|---:|---:|
| cfcmt_similarity_weighted_mpc_daganzo_guard_policy | 4 | 9522 | 7448 | -587.920 | 586.423 | 0.377 | 29.935 |
| cfcmt_similarity_weighted_mpc_policy | 4 | 9671 | 7269 | -654.464 | 653.803 | 0.422 | 13.219 |
| cfcmt_similarity_weighted_mpc_threshold_guard_policy | 4 | 9543 | 7431 | -584.435 | 582.984 | 0.369 | 29.024 |
| daganzo_policy | 4 | 9555 | 7976 | -531.040 | 529.673 | 0.343 | 27.337 |
| no_hold | 4 | 10000 | 7102 | -683.843 | 683.843 | 0.492 | 0.000 |
| threshold_equalization_policy | 4 | 9574 | 7947 | -532.651 | 531.349 | 0.336 | 26.030 |
