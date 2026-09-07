# Libsumo Policy Rollout Validation

Verdict: **PASS**

This is a closed-loop smoke over generated SUMO bus-stop events. It is not yet a calibrated field counterfactual.

| Policy | Cities | Events | Reward | Headway error | Bunching | Hold sec |
|---|---:|---:|---:|---:|---:|---:|
| cfcmt_mechanism_mpc_policy | 4 | 9239 | -249.633 | 249.339 | 0.148 | 5.884 |
| cfcmt_mechanism_policy | 4 | 9239 | -248.241 | 247.949 | 0.151 | 5.835 |
| cfcmt_similarity_weighted_mpc_policy | 4 | 9239 | -242.502 | 242.350 | 0.138 | 3.041 |
| cfcmt_similarity_weighted_policy | 4 | 9239 | -246.612 | 246.489 | 0.150 | 2.457 |
| daganzo_policy | 4 | 9239 | -178.999 | 178.611 | 0.098 | 7.768 |
| fixed_30 | 4 | 9239 | -254.402 | 252.902 | 0.154 | 30.000 |
| h2oplus_dense_mpc_policy | 4 | 9239 | -241.669 | 241.342 | 0.149 | 6.543 |
| h2oplus_dense_policy | 4 | 9239 | -242.362 | 242.008 | 0.149 | 7.086 |
| no_hold | 4 | 9239 | -241.104 | 241.104 | 0.149 | 0.000 |
| threshold_equalization_policy | 4 | 9239 | -177.162 | 176.797 | 0.097 | 7.302 |
