# Libsumo Policy Rollout Validation

Verdict: **PASS**

This is a closed-loop diagnostic over generated SUMO bus-stop events. First-arrival warm-up events are excluded from metrics. It is not yet a calibrated field counterfactual.

| Policy | Cities | Events | Warm-up | Reward | Headway error | Bunching | Hold sec |
|---|---:|---:|---:|---:|---:|---:|---:|
| cfcmt_similarity_weighted_mpc_policy | 4 | 94 | 61 | -182.719 | 181.271 | 0.281 | 28.945 |
| cfcmt_source_target_fewshot_mpc_policy | 4 | 94 | 62 | -177.582 | 176.146 | 0.266 | 28.711 |
| cfcmt_source_target_fewshot_mpc_rule_selector_policy | 4 | 94 | 62 | -180.061 | 178.638 | 0.266 | 28.453 |
| cfcmt_source_target_fewshot_mpc_threshold_guard_policy | 4 | 94 | 62 | -177.582 | 176.146 | 0.266 | 28.711 |
| cfcmt_target_fewshot_mpc_policy | 4 | 96 | 62 | -259.367 | 259.298 | 0.375 | 1.367 |
| daganzo_policy | 4 | 94 | 62 | -179.824 | 178.396 | 0.266 | 28.567 |
| no_hold | 4 | 96 | 62 | -259.025 | 259.025 | 0.375 | 0.000 |
| threshold_equalization_policy | 4 | 94 | 62 | -195.252 | 193.951 | 0.266 | 26.016 |
