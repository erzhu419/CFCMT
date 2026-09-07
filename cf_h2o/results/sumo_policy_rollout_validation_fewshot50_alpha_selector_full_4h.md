# Libsumo Policy Rollout Validation

Verdict: **PASS**

This is a closed-loop diagnostic over generated SUMO bus-stop events. First-arrival warm-up events are excluded from metrics. It is not yet a calibrated field counterfactual.

| Policy | Cities | Events | Warm-up | Reward | Headway error | Bunching | Hold sec |
|---|---:|---:|---:|---:|---:|---:|---:|
| cfcmt_source_target_fewshot_mpc_alpha_selector_policy | 4 | 9571 | 7950 | -532.982 | 531.681 | 0.337 | 26.016 |
| cfcmt_target_fewshot_mpc_alpha_selector_policy | 4 | 9551 | 7974 | -531.310 | 529.935 | 0.344 | 27.492 |
| daganzo_a03_policy | 4 | 9583 | 7967 | -531.128 | 529.800 | 0.340 | 26.566 |
| daganzo_a08_policy | 4 | 9550 | 7981 | -531.013 | 529.636 | 0.343 | 27.546 |
| threshold_equalization_policy | 4 | 9574 | 7947 | -532.651 | 531.349 | 0.336 | 26.030 |
