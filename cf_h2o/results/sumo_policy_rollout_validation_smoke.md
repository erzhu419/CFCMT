# Libsumo Policy Rollout Validation

Verdict: **PASS**

This is a closed-loop smoke over generated SUMO bus-stop events. It is not yet a calibrated field counterfactual.

| Policy | Cities | Events | Reward | Headway error | Bunching | Hold sec |
|---|---:|---:|---:|---:|---:|---:|
| cfcmt_mechanism_policy | 4 | 204 | -90.212 | 89.401 | 0.129 | 16.234 |
| cfcmt_similarity_weighted_policy | 4 | 204 | -86.420 | 85.650 | 0.121 | 15.406 |
| fixed_30 | 4 | 204 | -168.646 | 167.146 | 0.250 | 30.000 |
| h2oplus_dense_policy | 4 | 204 | -97.964 | 97.077 | 0.138 | 17.750 |
| no_hold | 4 | 204 | -172.537 | 172.537 | 0.250 | 0.000 |
| threshold_equalization_policy | 4 | 204 | -98.782 | 98.115 | 0.117 | 13.344 |
