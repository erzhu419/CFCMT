# Libsumo Policy Rollout Validation

Verdict: **PASS**

This is a closed-loop diagnostic over generated SUMO bus-stop events. First-arrival warm-up events are excluded from metrics. It is not yet a calibrated field counterfactual.

| Policy | Cities | Events | Warm-up | Reward | Headway error | Bunching | Hold sec |
|---|---:|---:|---:|---:|---:|---:|---:|
| cfcmt_target_fewshot_mpc_alpha_selector_policy | 4 | 9549 | 7978 | -531.931 | 530.556 | 0.344 | 27.497 |
| daganzo_a08_policy | 4 | 9550 | 7981 | -531.013 | 529.636 | 0.343 | 27.546 |
