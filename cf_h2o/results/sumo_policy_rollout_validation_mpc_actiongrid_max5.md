# Libsumo Policy Rollout Validation

Verdict: **PASS**

This is a closed-loop smoke over generated SUMO bus-stop events. It is not yet a calibrated field counterfactual.

| Policy | Cities | Events | Reward | Headway error | Bunching | Hold sec |
|---|---:|---:|---:|---:|---:|---:|
| cfcmt_mechanism_mpc_policy | 4 | 2120 | -1023.684 | 1023.007 | 0.282 | 13.530 |
| cfcmt_mechanism_policy | 4 | 2120 | -1023.129 | 1022.454 | 0.281 | 13.493 |
| cfcmt_similarity_weighted_mpc_policy | 4 | 2120 | -1013.322 | 1012.649 | 0.269 | 13.462 |
| cfcmt_similarity_weighted_policy | 4 | 2120 | -1013.139 | 1012.466 | 0.269 | 13.462 |
| daganzo_policy | 4 | 2090 | -942.982 | 941.825 | 0.261 | 23.141 |
| fixed_30 | 4 | 2102 | -1078.765 | 1077.265 | 0.354 | 30.000 |
| h2oplus_dense_mpc_policy | 4 | 2120 | -1067.658 | 1067.497 | 0.336 | 3.231 |
| h2oplus_dense_policy | 4 | 2120 | -1053.674 | 1053.383 | 0.318 | 5.831 |
| no_hold | 4 | 2120 | -1088.840 | 1088.840 | 0.360 | 0.000 |
| threshold_equalization_policy | 4 | 2099 | -932.290 | 931.260 | 0.244 | 20.606 |
