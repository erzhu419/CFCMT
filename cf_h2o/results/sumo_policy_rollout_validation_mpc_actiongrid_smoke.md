# Libsumo Policy Rollout Validation

Verdict: **PASS**

This is a closed-loop smoke over generated SUMO bus-stop events. It is not yet a calibrated field counterfactual.

| Policy | Cities | Events | Reward | Headway error | Bunching | Hold sec |
|---|---:|---:|---:|---:|---:|---:|
| cfcmt_similarity_weighted_mpc_policy | 4 | 222 | -85.221 | 84.446 | 0.117 | 15.487 |
| cfcmt_similarity_weighted_policy | 4 | 222 | -85.221 | 84.446 | 0.117 | 15.487 |
| daganzo_policy | 4 | 222 | -86.862 | 86.105 | 0.117 | 15.137 |
| h2oplus_dense_mpc_policy | 4 | 237 | -172.876 | 172.804 | 0.250 | 1.448 |
| h2oplus_dense_policy | 4 | 220 | -126.565 | 126.125 | 0.178 | 8.783 |
| no_hold | 4 | 237 | -172.585 | 172.585 | 0.250 | 0.000 |
| threshold_equalization_policy | 4 | 223 | -97.292 | 96.631 | 0.112 | 13.203 |
