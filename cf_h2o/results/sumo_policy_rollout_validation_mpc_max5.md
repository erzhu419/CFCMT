# Libsumo Policy Rollout Validation

Verdict: **PASS**

This is a closed-loop smoke over generated SUMO bus-stop events. It is not yet a calibrated field counterfactual.

| Policy | Cities | Events | Reward | Headway error | Bunching | Hold sec |
|---|---:|---:|---:|---:|---:|---:|
| cfcmt_mechanism_mpc_policy | 4 | 2120 | -1010.224 | 1009.480 | 0.268 | 14.875 |
| cfcmt_mechanism_policy | 4 | 2120 | -1010.034 | 1009.288 | 0.268 | 14.912 |
| cfcmt_similarity_weighted_mpc_policy | 4 | 2120 | -1007.598 | 1006.885 | 0.265 | 14.250 |
| cfcmt_similarity_weighted_policy | 4 | 2120 | -1007.144 | 1006.433 | 0.265 | 14.212 |
| daganzo_policy | 4 | 2090 | -942.982 | 941.825 | 0.261 | 23.141 |
| fixed_30 | 4 | 2102 | -1078.765 | 1077.265 | 0.354 | 30.000 |
| h2oplus_dense_mpc_policy | 4 | 2120 | -1006.965 | 1006.304 | 0.263 | 13.219 |
| h2oplus_dense_policy | 4 | 2120 | -1007.509 | 1006.831 | 0.265 | 13.556 |
| no_hold | 4 | 2120 | -1088.840 | 1088.840 | 0.360 | 0.000 |
| threshold_equalization_policy | 4 | 2099 | -932.290 | 931.260 | 0.244 | 20.606 |
