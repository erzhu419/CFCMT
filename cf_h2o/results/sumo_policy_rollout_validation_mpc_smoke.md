# Libsumo Policy Rollout Validation

Verdict: **PASS**

This is a closed-loop smoke over generated SUMO bus-stop events. It is not yet a calibrated field counterfactual.

| Policy | Cities | Events | Reward | Headway error | Bunching | Hold sec |
|---|---:|---:|---:|---:|---:|---:|
| cfcmt_similarity_weighted_mpc_policy | 4 | 222 | -85.337 | 84.570 | 0.117 | 15.350 |
| cfcmt_similarity_weighted_policy | 4 | 222 | -85.337 | 84.570 | 0.117 | 15.350 |
| daganzo_policy | 4 | 222 | -86.862 | 86.105 | 0.117 | 15.137 |
| h2oplus_dense_mpc_policy | 4 | 220 | -95.759 | 94.912 | 0.136 | 16.931 |
| h2oplus_dense_policy | 4 | 220 | -97.260 | 96.379 | 0.136 | 17.618 |
| no_hold | 4 | 237 | -172.585 | 172.585 | 0.250 | 0.000 |
| threshold_equalization_policy | 4 | 223 | -97.292 | 96.631 | 0.112 | 13.203 |
