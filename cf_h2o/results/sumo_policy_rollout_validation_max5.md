# Libsumo Policy Rollout Validation

Verdict: **PASS**

This is a closed-loop smoke over generated SUMO bus-stop events. It is not yet a calibrated field counterfactual.

| Policy | Cities | Events | Reward | Headway error | Bunching | Hold sec |
|---|---:|---:|---:|---:|---:|---:|
| cfcmt_mechanism_policy | 4 | 2120 | -1010.034 | 1009.288 | 0.268 | 14.912 |
| cfcmt_similarity_weighted_policy | 4 | 2120 | -1007.144 | 1006.433 | 0.265 | 14.212 |
| fixed_30 | 4 | 2102 | -1078.765 | 1077.265 | 0.354 | 30.000 |
| h2oplus_dense_policy | 4 | 2120 | -1007.509 | 1006.831 | 0.265 | 13.556 |
| no_hold | 4 | 2120 | -1088.840 | 1088.840 | 0.360 | 0.000 |
| threshold_equalization_policy | 4 | 2099 | -932.290 | 931.260 | 0.244 | 20.606 |
