# Libsumo Policy Rollout Validation

Verdict: **PASS**

This is a closed-loop diagnostic over generated SUMO bus-stop events. First-arrival warm-up events are excluded from metrics. It is not yet a calibrated field counterfactual.

| Policy | Cities | Events | Warm-up | Reward | Headway error | Bunching | Hold sec |
|---|---:|---:|---:|---:|---:|---:|---:|
| cfcmt_similarity_weighted_mpc_policy | 4 | 9633 | 7221 | -645.939 | 645.352 | 0.443 | 11.744 |
| cfcmt_source_target_fewshot_mpc_policy | 4 | 9735 | 7260 | -655.747 | 655.316 | 0.424 | 8.623 |
| cfcmt_source_target_fewshot_mpc_rule_selector_policy | 4 | 9561 | 7957 | -532.944 | 531.627 | 0.339 | 26.342 |
| cfcmt_source_target_fewshot_mpc_threshold_guard_policy | 4 | 9523 | 7963 | -541.208 | 539.797 | 0.343 | 28.209 |
| cfcmt_target_fewshot_mpc_policy | 4 | 9640 | 7964 | -610.867 | 609.686 | 0.352 | 23.617 |
| daganzo_policy | 4 | 9555 | 7976 | -531.040 | 529.673 | 0.343 | 27.337 |
| no_hold | 4 | 10000 | 7102 | -683.843 | 683.843 | 0.492 | 0.000 |
| threshold_equalization_policy | 4 | 9574 | 7947 | -532.651 | 531.349 | 0.336 | 26.030 |
