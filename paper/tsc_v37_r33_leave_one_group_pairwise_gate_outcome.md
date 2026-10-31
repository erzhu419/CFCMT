# TSC v37/r33 Leave-One-Group-Out Pairwise Gate Outcome

Generated: 2026-08-08T22:08:17.722795+00:00

## Integrity

**PASS**: 6 shards, 360 independent 59-group fits, and 6 final results passed snapshot, SHA-256, LOGO identity, explicit-fit, causal-parent, complete-city holdout, decision reproduction, and external evaluation audits.

- snapshot SHA-256: `35eae1ce7a884fe8e3a99fe2b76d1f8df536787bf30595b0a78ca204e1998196`
- v34 audit SHA-256: `1120f7952634dcc07849865b2a01fc0d53fa67a4cf0ecd6580bdd9a0c9db48b6`
- v35 negative audit SHA-256: `f63154910367da7ffa61bfc7a9a61e8bb5eead1922905f75d791e5159e5dbb86`
- v36 negative audit SHA-256: `0d6f5f87da65cc54e0d46184a0d1170901672376603f496e980d8d5666d4dc43`
- cache aggregate SHA-256: `5b8bac698f2bda045476614b37672edf31d73e50e266e8137baadc9e6349e72c`

## Decisions

| Target | LOGO gain | Non-worse | Worst seed | Selected | Fallback regret | Pairwise regret |
|---|---:|---:|---:|---|---:|---:|
| grid4x4 | 0.058000 | 0.900 | -0.010963 | causal_antisymmetric_pairwise_advantage | 0.317706 | 0.247764 |
| cologne1 | 0.046782 | 0.867 | -0.063111 | causal_group_normalized_rigid_advantage | 0.108695 | 0.040156 |
| ingolstadt1 | 0.009197 | 0.833 | -0.045980 | causal_group_normalized_rigid_advantage | 0.155821 | 0.147838 |
| atlanta_1x5 | -0.016667 | 0.983 | -0.052632 | causal_group_normalized_rigid_advantage | 0.075137 | 0.034608 |
| hangzhou_4x4 | -0.025972 | 0.750 | -0.086111 | causal_group_normalized_rigid_advantage | 0.377957 | 0.366856 |
| manhattan_28x7 | 0.029702 | 0.850 | -0.062265 | causal_group_normalized_rigid_advantage | 0.324763 | 0.299943 |

## Frozen Decision

Stage: **FAIL**
Decision: `reject_logo_pairwise_deployment`
Pairwise selected: 1/6
Selected-policy macro regret: 0.215023
Fallback macro regret: 0.226680
Relative improvement: +5.14%
Maximum absolute regression: 0.000000

- `macro_improvement_at_least_5_pct`: PASS
- `at_least_four_selected_cities_improved`: FAIL
- `max_regression_at_most_0_02`: PASS

Salt Lake City data were not read and cannot alter this decision.
