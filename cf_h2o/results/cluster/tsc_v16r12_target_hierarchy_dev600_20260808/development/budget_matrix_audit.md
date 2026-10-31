# TSC CFCMT Budget Matrix Audit

Audit: **PASS**  
Protocol SHA-256: `5cff2f92d2b14e305e63d3d343094796a05db51b5ec46905960bc07009ea224b`  
Primary metric: `mean_system_vehicles_per_controlled_lane`  
Declared primary policy: `cfcmt_mechanism_contrast_regularized`

| Budget | Regime | Primary cost | Best paired policy | Best absolute policy | Strongest paired rule | Delta vs rule | Paired rank |
|---:|---|---:|---|---|---|---:|---:|
| 0 | zero-shot | 5.081614 | simulator_contrast_guard | causal_rigid_advantage_contrast_mpc | max_pressure | 0.06% | 13 |
| 8 | target-simulator-few-shot | 5.082005 | causal_core_advantage_contrast_guard | causal_core_advantage_contrast_guard | max_pressure | 0.09% | 20 |
| 16 | target-simulator-few-shot | 4.615194 | cfcmt_mechanism_contrast_regularized | dense_balanced_advantage_contrast_mpc | max_pressure | -2.16% | 1 |
| 32 | target-simulator-few-shot | 4.757455 | causal_target_only_contrast_mpc | causal_target_only_contrast_mpc | max_pressure | -1.48% | 9 |
| 60 | target-simulator-few-shot | 4.643890 | causal_target_only_contrast_guard | dense_balanced_advantage_contrast_mpc | max_pressure | -2.03% | 8 |
| 120 | target-simulator-few-shot | 4.655467 | causal_target_only_contrast_guard | causal_target_only_contrast_mpc | max_pressure | -1.99% | 8 |

## Cross-city breadth

| Budget | Mean vs prior | Median vs prior | Winning cities | Worst city | Top-gain city share | Broad |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.00% | 0.00% | 0/6 | 0.00% | 0.0% | False |
| 8 | 0.04% | 0.00% | 0/6 | 0.21% | 0.0% | False |
| 16 | -2.21% | 0.00% | 2/6 | 0.28% | 99.1% | False |
| 32 | -1.54% | 0.00% | 1/6 | 0.11% | 100.0% | False |
| 60 | -2.09% | 0.00% | 1/6 | 0.08% | 100.0% | False |
| 120 | -2.04% | 0.00% | 1/6 | 0.00% | 100.0% | False |

## Monotonicity

| Policy | Non-increasing | Inversions |
|---|---:|---:|
| selected_source_prior | True | 0 |
| simulator_contrast_guard | False | 3 |
| dense_contrast_guard | False | 3 |
| dense_balanced_advantage_contrast_guard | False | 2 |
| causal_core_advantage_contrast_guard | False | 2 |
| causal_balanced_advantage_contrast_guard | True | 0 |
| causal_rigid_advantage_contrast_guard | False | 2 |
| cfcmt_mechanism_contrast_regularized | False | 3 |
| cfcmt_mechanism_contrast_guard | False | 1 |
| causal_target_adapter_contrast_regularized | False | 1 |
| causal_target_adapter_contrast_guard | False | 2 |
| causal_target_only_contrast_regularized | False | 4 |
| causal_target_only_contrast_guard | False | 1 |

## Warnings

- budget 0: SUMO emergency-stop events are nonzero: 58
- budget 8: explicitly unguarded MPC ablations produced collision/teleport events: {'collision_events': 3.0, 'starting_teleports': 3.0}
- budget 8: SUMO emergency-stop events are nonzero: 60
- budget 16: SUMO emergency-stop events are nonzero: 58
- budget 32: explicitly unguarded MPC ablations produced collision/teleport events: {'collision_events': 2.0, 'starting_teleports': 2.0}
- budget 32: SUMO emergency-stop events are nonzero: 59
- budget 60: explicitly unguarded MPC ablations produced collision/teleport events: {'collision_events': 3.0, 'starting_teleports': 3.0, 'ending_teleports': 2.0}
- budget 60: SUMO emergency-stop events are nonzero: 58
- budget 120: SUMO emergency-stop events are nonzero: 60
- declared primary policy is non-monotonic across target budgets
- budget 16: primary mean gain is not broad across city groups
- budget 16: 99.1% of primary improvement is concentrated in atlanta
- budget 32: primary mean gain is not broad across city groups
- budget 32: 100.0% of primary improvement is concentrated in atlanta
- budget 60: primary mean gain is not broad across city groups
- budget 60: 100.0% of primary improvement is concentrated in atlanta
- budget 120: primary mean gain is not broad across city groups
- budget 120: 100.0% of primary improvement is concentrated in atlanta
