# V43 External Closed-Loop Confirmation

Validity gate: **PASS**  
Rollouts: 56/56  
SUMO: `1.22.0`  

The primary waiting-time metric covers all departed vehicles, including write-unfinished records accrued to the fixed 3,600-second horizon. Completed-only waiting, system vehicle-hours and completion ratio are reported beside it to expose horizon and throughput trade-offs.

## Macro Results

| Policy | All-departed wait (s) | Completed wait (s) | Duration (s) | System veh-h | Unfinished | Completion | Incidents |
|---|---:|---:|---:|---:|---:|---:|---:|
| cfcmt_selected_mpc | 149.167 | 133.065 | 330.541 | 594.142 | 0.0667 | 0.8887 | 0.000 |
| fixed_time | 338.583 | 308.717 | 523.127 | 885.184 | 0.1325 | 0.6608 | 2.000 |
| phase_pressure | 143.620 | 125.318 | 324.460 | 583.052 | 0.0676 | 0.8976 | 0.250 |
| max_pressure | 141.737 | 135.422 | 323.896 | 580.286 | 0.0655 | 0.8957 | 0.250 |
| rigid_anchor_mpc | 152.914 | 138.396 | 334.913 | 606.299 | 0.0720 | 0.8821 | 0.500 |
| simulator_only_mpc | 156.712 | 138.677 | 337.872 | 619.658 | 0.0691 | 0.8773 | 0.000 |
| h2oplus_style_dense_residual_mpc | 159.149 | 147.474 | 341.629 | 618.155 | 0.0661 | 0.8701 | 0.250 |

## Method Comparisons

Positive values favor CFCMT. Confidence intervals are descriptive paired hierarchical bootstrap intervals, not population-level inference, because there are only two target cities.

| Baseline | Waiting gain (s) | Relative gain | 95% interval | City gains (s) | Collision noninferior |
|---|---:|---:|---:|---|---|
| fixed_time | 189.416 | 55.94% | [67.106, 308.321] | los_angeles=302.905, jinan=75.927 | macro=True, all-paired=True |
| phase_pressure | -5.547 | -3.86% | [-9.541, -2.452] | los_angeles=-6.906, jinan=-4.188 | macro=True, all-paired=True |
| max_pressure | -7.430 | -5.24% | [-15.167, -1.556] | los_angeles=-12.525, jinan=-2.334 | macro=True, all-paired=True |
| rigid_anchor_mpc | 3.747 | 2.45% | [-8.749, 15.703] | los_angeles=2.359, jinan=5.136 | macro=True, all-paired=True |
| simulator_only_mpc | 7.545 | 4.81% | [-3.618, 25.270] | los_angeles=14.747, jinan=0.343 | macro=True, all-paired=True |
| h2oplus_style_dense_residual_mpc | 9.982 | 6.27% | [2.014, 17.191] | los_angeles=8.463, jinan=11.501 | macro=True, all-paired=True |

## Protocol Note

All policy comparisons are paired on the same scenario and seed. Scenarios are averaged equally within each city, then cities are averaged equally. The validity gate checks provenance, matrix completeness, source identity, SUMO version, tripinfo hashes, population accounting and zero teleports. Collisions are retained as audited safety outcomes under paired noninferiority; the validity gate does not impose a performance or collision threshold.
