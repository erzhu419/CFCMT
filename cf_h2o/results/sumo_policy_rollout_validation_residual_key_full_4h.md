# Libsumo Policy Rollout Validation

Verdict: **PASS**

This is a closed-loop diagnostic over generated SUMO bus-stop events. First-arrival warm-up events are excluded from metrics. It is not yet a calibrated field counterfactual.

| Policy | Cities | Events | Warm-up | Reward | Headway error | Bunching | Hold sec |
|---|---:|---:|---:|---:|---:|---:|---:|
| cfcmt_similarity_weighted_mpc_daganzo_residual_policy | 4 | 9523 | 7718 | -559.598 | 558.366 | 0.362 | 24.655 |
| cfcmt_similarity_weighted_mpc_policy | 4 | 9633 | 7221 | -645.939 | 645.352 | 0.443 | 11.744 |
| cfcmt_similarity_weighted_mpc_rule_selector_policy | 4 | 9558 | 7951 | -533.166 | 531.850 | 0.338 | 26.329 |
| cfcmt_similarity_weighted_mpc_threshold_guard_policy | 4 | 9482 | 7843 | -542.095 | 540.660 | 0.350 | 28.703 |
| cfcmt_similarity_weighted_mpc_threshold_residual_policy | 4 | 9532 | 7709 | -559.489 | 558.278 | 0.360 | 24.236 |
| daganzo_policy | 4 | 9555 | 7976 | -531.040 | 529.673 | 0.343 | 27.337 |
| no_hold | 4 | 10000 | 7102 | -683.843 | 683.843 | 0.492 | 0.000 |
| threshold_equalization_policy | 4 | 9574 | 7947 | -532.651 | 531.349 | 0.336 | 26.030 |
