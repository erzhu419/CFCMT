# Libsumo Policy Rollout Validation

Verdict: **PASS**

This is a closed-loop diagnostic over generated SUMO bus-stop events. First-arrival warm-up events are excluded from metrics. It is not yet a calibrated field counterfactual.

| Policy | Cities | Events | Warm-up | Reward | Headway error | Bunching | Hold sec |
|---|---:|---:|---:|---:|---:|---:|---:|
| cfcmt_similarity_weighted_mpc_daganzo_guard_policy | 4 | 9474 | 7870 | -542.394 | 540.911 | 0.355 | 29.652 |
| cfcmt_similarity_weighted_mpc_policy | 4 | 9633 | 7221 | -645.939 | 645.352 | 0.443 | 11.744 |
| cfcmt_similarity_weighted_mpc_threshold_guard_policy | 4 | 9482 | 7843 | -542.095 | 540.660 | 0.350 | 28.703 |
| daganzo_policy | 4 | 9555 | 7976 | -531.040 | 529.673 | 0.343 | 27.337 |
| no_hold | 4 | 10000 | 7102 | -683.843 | 683.843 | 0.492 | 0.000 |
| threshold_equalization_policy | 4 | 9574 | 7947 | -532.651 | 531.349 | 0.336 | 26.030 |
