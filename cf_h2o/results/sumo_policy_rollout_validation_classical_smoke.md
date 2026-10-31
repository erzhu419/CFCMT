# Libsumo Policy Rollout Validation

Verdict: **PASS**

This is a closed-loop smoke over generated SUMO bus-stop events. It is not yet a calibrated field counterfactual.

| Policy | Cities | Events | Reward | Headway error | Bunching | Hold sec |
|---|---:|---:|---:|---:|---:|---:|
| fixed_30 | 4 | 134 | -181.400 | 179.900 | 0.267 | 30.000 |
| no_hold | 4 | 134 | -186.677 | 186.677 | 0.267 | 0.000 |
| threshold_equalization_policy | 4 | 132 | -124.764 | 123.948 | 0.175 | 16.312 |
