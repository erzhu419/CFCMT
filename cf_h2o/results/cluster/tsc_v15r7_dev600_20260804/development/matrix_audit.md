# TSC CFCMT Budget Matrix Audit

Audit: **PASS**  
Protocol SHA-256: `496d47f5a2998e91a43405eafd1e8bac9d7c003f07964c2405dc72c3212a385d`  
Primary metric: `mean_system_vehicles_per_controlled_lane`  
Declared primary policy: `cfcmt_mechanism_contrast_regularized`

| Budget | Regime | Primary cost | Best paired policy | Best absolute policy | Strongest paired rule | Delta vs rule | Paired rank |
|---:|---|---:|---|---|---|---:|---:|
| 0 | zero-shot | 5.081614 | causal_balanced_advantage_contrast_mpc | causal_balanced_advantage_contrast_mpc | max_pressure | 0.06% | 12 |
| 8 | target-simulator-few-shot | 5.081614 | causal_core_advantage_contrast_mpc | causal_core_advantage_contrast_mpc | max_pressure | 0.06% | 14 |
| 16 | target-simulator-few-shot | 4.679215 | dense_balanced_advantage_contrast_regularized | causal_core_advantage_contrast_mpc | max_pressure | -1.91% | 8 |
| 32 | target-simulator-few-shot | 5.072758 | causal_core_advantage_contrast_guard | causal_core_advantage_contrast_mpc | max_pressure | -0.19% | 11 |
| 60 | target-simulator-few-shot | 4.575449 | dense_balanced_advantage_contrast_mpc | dense_balanced_advantage_contrast_mpc | max_pressure | -2.44% | 5 |
| 120 | target-simulator-few-shot | 4.582578 | causal_rigid_advantage_contrast_guard | causal_core_advantage_contrast_mpc | max_pressure | -2.34% | 9 |

## Cross-city breadth

| Budget | Mean vs prior | Median vs prior | Winning cities | Worst city | Top-gain city share | Broad |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.00% | 0.00% | 0/6 | 0.00% | 0.0% | False |
| 8 | 0.00% | 0.00% | 0/6 | 0.00% | 0.0% | False |
| 16 | -1.97% | 0.00% | 2/6 | 0.00% | 97.5% | False |
| 32 | -0.25% | 0.00% | 2/6 | 0.00% | 56.1% | False |
| 60 | -2.50% | 0.00% | 2/6 | 0.00% | 96.1% | False |
| 120 | -2.39% | 0.00% | 1/6 | 0.00% | 100.0% | False |

## Monotonicity

| Policy | Non-increasing | Inversions |
|---|---:|---:|
| selected_source_prior | True | 0 |
| simulator_contrast_guard | False | 1 |
| dense_contrast_guard | False | 2 |
| dense_balanced_advantage_contrast_guard | False | 1 |
| causal_core_advantage_contrast_guard | False | 2 |
| causal_balanced_advantage_contrast_guard | True | 0 |
| causal_rigid_advantage_contrast_guard | False | 1 |
| cfcmt_mechanism_contrast_regularized | False | 2 |
| cfcmt_mechanism_contrast_guard | False | 1 |
| causal_target_adapter_contrast_guard | False | 2 |
| causal_target_only_contrast_guard | False | 2 |

## Warnings

- budget 0: explicitly unguarded MPC ablations produced collision/teleport events: {'collision_events': 2.0, 'starting_teleports': 2.0, 'ending_teleports': 1.0}
- budget 0: SUMO emergency-stop events are nonzero: 59
- budget 8: explicitly unguarded MPC ablations produced collision/teleport events: {'collision_events': 1.0, 'starting_teleports': 1.0}
- budget 8: SUMO emergency-stop events are nonzero: 58
- budget 16: explicitly unguarded MPC ablations produced collision/teleport events: {'collision_events': 1.0, 'starting_teleports': 1.0}
- budget 16: SUMO emergency-stop events are nonzero: 59
- budget 32: explicitly unguarded MPC ablations produced collision/teleport events: {'collision_events': 1.0, 'starting_teleports': 1.0, 'ending_teleports': 1.0}
- budget 32: SUMO emergency-stop events are nonzero: 58
- budget 60: explicitly unguarded MPC ablations produced collision/teleport events: {'collision_events': 1.0, 'starting_teleports': 1.0}
- budget 60: SUMO emergency-stop events are nonzero: 58
- budget 120: SUMO emergency-stop events are nonzero: 60
- declared primary policy is non-monotonic across target budgets
- budget 16: primary mean gain is not broad across city groups
- budget 16: 97.5% of primary improvement is concentrated in atlanta
- budget 32: primary mean gain is not broad across city groups
- budget 60: primary mean gain is not broad across city groups
- budget 60: 96.1% of primary improvement is concentrated in atlanta
- budget 120: primary mean gain is not broad across city groups
- budget 120: 100.0% of primary improvement is concentrated in atlanta
