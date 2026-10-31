# Libsumo Policy Rollout Validation

Verdict: **PASS**

This is a closed-loop diagnostic over generated SUMO bus-stop events. First-arrival warm-up events are excluded from metrics. It is not yet a calibrated field counterfactual.

| Policy | Cities | Events | Warm-up | Reward | Headway error | Bunching | Hold sec |
|---|---:|---:|---:|---:|---:|---:|---:|
| cfcmt_mechanism_mpc_policy | 4 | 9970 | 6973 | -689.063 | 689.006 | 0.487 | 1.132 |
| cfcmt_mechanism_policy | 4 | 9964 | 6976 | -693.211 | 693.166 | 0.498 | 0.908 |
| cfcmt_similarity_weighted_mpc_policy | 4 | 9633 | 7221 | -645.939 | 645.352 | 0.443 | 11.744 |
| cfcmt_similarity_weighted_policy | 4 | 10000 | 7076 | -691.120 | 691.069 | 0.487 | 1.006 |
| daganzo_policy | 4 | 9555 | 7976 | -531.040 | 529.673 | 0.343 | 27.337 |
| fixed_30 | 4 | 9719 | 6762 | -671.070 | 669.570 | 0.487 | 30.000 |
| h2oplus_dense_mpc_policy | 4 | 9830 | 7008 | -685.660 | 685.554 | 0.488 | 2.113 |
| h2oplus_dense_policy | 4 | 9869 | 6988 | -693.189 | 693.161 | 0.495 | 0.560 |
| no_hold | 4 | 10000 | 7102 | -683.843 | 683.843 | 0.492 | 0.000 |
| threshold_equalization_policy | 4 | 9574 | 7947 | -532.651 | 531.349 | 0.336 | 26.030 |
