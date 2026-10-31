# TSC CFCMT Budget Matrix Audit

Audit: **FAIL**  
Protocol SHA-256: `5e54e346b3dd833a163b0bd11883d88192a46a86682bfc2e896ae2f1fce4cc31`  
Primary metric: `mean_system_vehicles_per_controlled_lane`  
Declared primary policy: `cfcmt_fused_contrast_guard`

| Budget | Regime | Primary cost | Best paired policy | Best absolute policy | Strongest paired rule | Delta vs rule | Paired rank |
|---:|---|---:|---|---|---|---:|---:|
| 0 | zero-shot | 5.104153 | cfcmt_fused_contrast_mpc | cfcmt_fused_contrast_mpc | max_pressure | 0.54% | 11 |
| 8 | target-simulator-few-shot | 5.104153 | dense_balanced_advantage_contrast_guard | dense_balanced_advantage_contrast_guard | max_pressure | 0.54% | 9 |
| 16 | target-simulator-few-shot | 5.104153 | cfcmt_fused_contrast_mpc | cfcmt_fused_contrast_mpc | max_pressure | 0.54% | 9 |
| 32 | target-simulator-few-shot | 5.092956 | cfcmt_fused_contrast_mpc | cfcmt_fused_contrast_mpc | max_pressure | 0.35% | 10 |
| 60 | target-simulator-few-shot | 4.755137 | cfcmt_fused_contrast_mpc | cfcmt_fused_contrast_mpc | max_pressure | -1.10% | 6 |
| 120 | target-simulator-few-shot | 4.664209 | cfcmt_fused_contrast_mpc | cfcmt_fused_contrast_mpc | max_pressure | -1.54% | 5 |

## Cross-city breadth

| Budget | Mean vs prior | Median vs prior | Winning cities | Worst city | Top-gain city share | Broad |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.00% | 0.00% | 0/6 | 0.00% | 0.0% | False |
| 8 | 0.00% | 0.00% | 0/6 | 0.00% | 0.0% | False |
| 16 | 0.00% | 0.00% | 0/6 | 0.00% | 0.0% | False |
| 32 | -0.18% | 0.00% | 1/6 | 0.00% | 100.0% | False |
| 60 | -1.63% | -0.06% | 3/6 | 0.56% | 97.3% | True |
| 120 | -2.07% | 0.00% | 2/6 | 0.52% | 97.8% | False |

## Monotonicity

| Policy | Non-increasing | Inversions |
|---|---:|---:|
| selected_source_prior | True | 0 |
| simulator_contrast_guard | False | 3 |
| dense_contrast_guard | False | 3 |
| dense_balanced_advantage_contrast_guard | False | 2 |
| causal_rigid_advantage_contrast_guard | False | 2 |
| cfcmt_mechanism_contrast_guard | False | 1 |
| causal_target_only_contrast_guard | False | 1 |
| cfcmt_fused_contrast_regularized | False | 1 |
| cfcmt_fused_contrast_guard | True | 0 |

## Errors

- budget 60: primary policy fails paired collision-incident noninferiority against selected_source_prior: 45 > 40

## Warnings

- budget 0: raw benchmark junction collisions were observed and retained for paired incident analysis: {'collision_events': 1227.0, 'collision_incidents': 552.0}
- budget 0: explicitly unguarded MPC ablations produced collision/teleport events: {'collision_events': 123.0, 'collision_incidents': 44.0}
- budget 0: SUMO emergency-stop events are nonzero: 58
- budget 8: raw benchmark junction collisions were observed and retained for paired incident analysis: {'collision_events': 1217.0, 'collision_incidents': 565.0}
- budget 8: explicitly unguarded MPC ablations produced collision/teleport events: {'collision_events': 123.0, 'collision_incidents': 44.0}
- budget 8: SUMO emergency-stop events are nonzero: 58
- budget 16: raw benchmark junction collisions were observed and retained for paired incident analysis: {'collision_events': 1222.0, 'collision_incidents': 567.0}
- budget 16: explicitly unguarded MPC ablations produced collision/teleport events: {'collision_events': 93.0, 'collision_incidents': 36.0}
- budget 16: SUMO emergency-stop events are nonzero: 58
- budget 32: raw benchmark junction collisions were observed and retained for paired incident analysis: {'collision_events': 1241.0, 'collision_incidents': 574.0}
- budget 32: explicitly unguarded MPC ablations produced collision/teleport events: {'collision_events': 76.0, 'collision_incidents': 39.0}
- budget 32: SUMO emergency-stop events are nonzero: 58
- budget 60: raw benchmark junction collisions were observed and retained for paired incident analysis: {'collision_events': 1261.0, 'collision_incidents': 577.0}
- budget 60: explicitly unguarded MPC ablations produced collision/teleport events: {'collision_events': 82.0, 'collision_incidents': 43.0}
- budget 60: SUMO emergency-stop events are nonzero: 58
- budget 120: raw benchmark junction collisions were observed and retained for paired incident analysis: {'collision_events': 1247.0, 'collision_incidents': 575.0}
- budget 120: explicitly unguarded MPC ablations produced collision/teleport events: {'collision_events': 83.0, 'collision_incidents': 45.0}
- budget 120: SUMO emergency-stop events are nonzero: 59
- budget 32: primary mean gain is not broad across city groups
- budget 32: 100.0% of primary improvement is concentrated in cologne
- budget 60: 97.3% of primary improvement is concentrated in atlanta
- budget 120: primary mean gain is not broad across city groups
- budget 120: 97.8% of primary improvement is concentrated in atlanta
