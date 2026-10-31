# V43 Deployment-Mechanism Diagnostic

Status: **post-hoc explanatory analysis (not preregistered confirmation)**  
Validity gate: **PASS**  
New diagnostic rollouts: 48  

The frozen confirmatory controller is a pressure-anchored sparse residual controller. `direct` executes every eligible learned override; `spatial_only` retains only conflict-graph coordination.

## Macro Results

| Policy | Wait (s) | Completed wait (s) | System veh-h | Completion | Incidents |
|---|---:|---:|---:|---:|---:|
| selected_source_prior | 142.116 | 125.851 | 579.361 | 0.9025 | 0.250 |
| cfcmt_selected_mpc | 149.167 | 133.065 | 594.142 | 0.8887 | 0.000 |
| cfcmt_selected_spatial_only | 150.604 | 132.763 | 585.374 | 0.8953 | 0.500 |
| cfcmt_selected_direct | 152.820 | 133.581 | 583.106 | 0.9025 | 0.250 |
| rigid_anchor_mpc | 152.914 | 138.396 | 606.299 | 0.8821 | 0.500 |
| rigid_anchor_direct | 190.584 | 178.896 | 662.219 | 0.8770 | 1.000 |
| simulator_only_mpc | 156.712 | 138.677 | 619.658 | 0.8773 | 0.000 |
| simulator_only_direct | 1926.408 | 0.029 | 1749.562 | 0.1717 | 0.000 |
| h2oplus_style_dense_residual_mpc | 159.149 | 147.474 | 618.155 | 0.8701 | 0.250 |
| h2oplus_style_dense_residual_direct | 1782.519 | 45.099 | 1802.427 | 0.1449 | 0.000 |
| phase_pressure | 143.620 | 125.318 | 583.052 | 0.8976 | 0.250 |
| max_pressure | 141.737 | 135.422 | 580.286 | 0.8957 | 0.250 |
| fixed_time | 338.583 | 308.717 | 885.184 | 0.6608 | 2.000 |

## Paired Descriptive Comparisons

Positive waiting-time gains favor the named method. Intervals are descriptive because there are only two independent target cities.

| Comparison | Gain (s) | 95% interval | P(gain > 0) |
|---|---:|---:|---:|
| cfcmt_selected_mpc_vs_selected_source_prior | -7.066 | [-10.483, -3.951] | 0.000 |
| cfcmt_selected_spatial_only_vs_selected_source_prior | -8.535 | [-15.392, -0.059] | 0.016 |
| cfcmt_selected_direct_vs_selected_source_prior | -10.703 | [-22.205, 0.205] | 0.079 |
| cfcmt_selected_mpc_vs_h2oplus_style_dense_residual_mpc | 9.913 | [2.014, 17.191] | 0.984 |
| cfcmt_selected_direct_vs_h2oplus_style_dense_residual_direct | 1630.817 | [1142.289, 2159.863] | 1.000 |
| cfcmt_selected_mpc_vs_rigid_anchor_mpc | 3.704 | [-8.749, 15.617] | 0.796 |
| cfcmt_selected_direct_vs_rigid_anchor_direct | 37.977 | [11.158, 67.747] | 1.000 |
| cfcmt_selected_mpc_vs_phase_pressure | -5.541 | [-9.541, -2.396] | 0.000 |
| cfcmt_selected_mpc_vs_max_pressure | -7.428 | [-15.167, -1.543] | 0.000 |

## Deployment Audit

| Policy | Mode | Proposals | Accepted | Acceptance | Cooldown rejects | Spatial rejects |
|---|---|---:|---:|---:|---:|---:|
| cfcmt_selected_mpc | sparse | 4596 | 1389 | 0.3022 | 3054 | 153 |
| rigid_anchor_mpc | sparse | 6913 | 1678 | 0.2427 | 5147 | 88 |
| simulator_only_mpc | sparse | 7879 | 1532 | 0.1944 | 6140 | 207 |
| h2oplus_style_dense_residual_mpc | sparse | 7062 | 1624 | 0.2300 | 5220 | 218 |
| cfcmt_selected_spatial_only | spatial_only | 4787 | 3669 | 0.7665 | 0 | 1118 |
| cfcmt_selected_direct | direct | 4944 | 4944 | 1.0000 | 0 | 0 |
| rigid_anchor_direct | direct | 7125 | 7125 | 1.0000 | 0 | 0 |
| simulator_only_direct | direct | 28172 | 28172 | 1.0000 | 0 | 0 |
| h2oplus_style_dense_residual_direct | direct | 19056 | 19056 | 1.0000 | 0 | 0 |
