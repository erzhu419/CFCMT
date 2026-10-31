# TSC CFCMT Budget Matrix Audit

Audit: **FAIL**  
Protocol SHA-256: `dedf342b432e1455cc0ddefd6f353b43eb008764c9aa227f1ff529663238a114`  
Primary metric: `mean_system_vehicles_per_controlled_lane`  
Declared primary policy: `cfcmt_fused_contrast_guard`

| Budget | Regime | Primary cost | Best paired policy | Best absolute policy | Strongest paired rule | Delta vs rule | Paired rank |
|---:|---|---:|---|---|---|---:|---:|
| 0 | zero-shot | 5.104153 | cfcmt_fused_rigid_contrast_mpc | cfcmt_fused_rigid_contrast_mpc | max_pressure | 0.54% | 14 |
| 8 | target-simulator-few-shot | 5.104153 | dense_balanced_advantage_contrast_guard | dense_balanced_advantage_contrast_guard | max_pressure | 0.54% | 12 |
| 16 | target-simulator-few-shot | 5.104153 | cfcmt_fused_contrast_mpc | cfcmt_fused_contrast_mpc | max_pressure | 0.54% | 12 |
| 32 | target-simulator-few-shot | 4.754439 | causal_target_only_contrast_mpc | causal_target_only_contrast_mpc | max_pressure | -1.14% | 7 |
| 60 | target-simulator-few-shot | 4.795749 | causal_target_only_contrast_mpc | causal_target_only_contrast_mpc | max_pressure | -0.92% | 8 |
| 120 | target-simulator-few-shot | 4.654512 | causal_target_only_contrast_mpc | causal_target_only_contrast_mpc | max_pressure | -1.72% | 3 |

## Cross-city breadth

| Budget | Mean vs prior | Median vs prior | Winning cities | Worst city | Top-gain city share | Broad |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.00% | 0.00% | 0/6 | 0.00% | 0.0% | False |
| 8 | 0.00% | 0.00% | 0/6 | 0.00% | 0.0% | False |
| 16 | 0.00% | 0.00% | 0/6 | 0.00% | 0.0% | False |
| 32 | -1.68% | 0.00% | 1/6 | 0.00% | 100.0% | False |
| 60 | -1.45% | -0.08% | 3/6 | 0.56% | 96.0% | True |
| 120 | -2.25% | 0.00% | 2/6 | 0.52% | 90.3% | False |

## Preregistered method acceptance

Overall method gate: **FAIL**  
Efficacy gate: **FAIL**  
Mechanism contribution gate: **FAIL**

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
| cfcmt_fused_rigid_contrast_mpc | False | 2 |
| cfcmt_fused_rigid_contrast_guard | False | 1 |
| cfcmt_fused_contrast_mpc | False | 1 |
| cfcmt_fused_contrast_regularized | True | 0 |
| cfcmt_fused_contrast_guard | False | 1 |

## Errors

- budget 60: primary policy fails paired collision-incident noninferiority against selected_source_prior: 41 > 40
- budget 120: primary policy fails paired collision-incident noninferiority against selected_source_prior: 43 > 40

## Warnings

- budget 0: raw benchmark junction collisions were observed and retained for paired incident analysis: {'collision_events': 1307.0, 'collision_incidents': 592.0}
- budget 0: explicitly unguarded MPC ablations produced collision/teleport events: {'collision_events': 350.0, 'collision_incidents': 143.0}
- budget 0: SUMO emergency-stop events are nonzero: 58
- budget 8: raw benchmark junction collisions were observed and retained for paired incident analysis: {'collision_events': 1297.0, 'collision_incidents': 605.0}
- budget 8: explicitly unguarded MPC ablations produced collision/teleport events: {'collision_events': 274.0, 'collision_incidents': 142.0}
- budget 8: SUMO emergency-stop events are nonzero: 58
- budget 16: raw benchmark junction collisions were observed and retained for paired incident analysis: {'collision_events': 1302.0, 'collision_incidents': 607.0}
- budget 16: explicitly unguarded MPC ablations produced collision/teleport events: {'collision_events': 286.0, 'collision_incidents': 126.0}
- budget 16: SUMO emergency-stop events are nonzero: 58
- budget 32: raw benchmark junction collisions were observed and retained for paired incident analysis: {'collision_events': 1328.0, 'collision_incidents': 619.0}
- budget 32: explicitly unguarded MPC ablations produced collision/teleport events: {'collision_events': 260.0, 'collision_incidents': 134.0}
- budget 32: SUMO emergency-stop events are nonzero: 58
- budget 60: raw benchmark junction collisions were observed and retained for paired incident analysis: {'collision_events': 1348.0, 'collision_incidents': 609.0}
- budget 60: explicitly unguarded MPC ablations produced collision/teleport events: {'collision_events': 250.0, 'collision_incidents': 128.0}
- budget 60: SUMO emergency-stop events are nonzero: 58
- budget 120: raw benchmark junction collisions were observed and retained for paired incident analysis: {'collision_events': 1333.0, 'collision_incidents': 618.0}
- budget 120: explicitly unguarded MPC ablations produced collision/teleport events: {'collision_events': 267.0, 'collision_incidents': 143.0}
- budget 120: SUMO emergency-stop events are nonzero: 59
- declared primary policy is non-monotonic across target budgets
- budget 32: primary mean gain is not broad across city groups
- budget 32: 100.0% of primary improvement is concentrated in atlanta
- budget 60: 96.0% of primary improvement is concentrated in atlanta
- budget 120: primary mean gain is not broad across city groups
- budget 120: 90.3% of primary improvement is concentrated in atlanta
- fused CFCMT does not pass all preregistered efficacy and mechanism-contribution gates
