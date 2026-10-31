# TSC v36/r32 Cross-Fitted Pairwise Gate Outcome

Generated: 2026-08-08T21:55:31.349427+00:00

## Integrity

**PASS**: 6 shards and 6 results passed snapshot, SHA-256, 60-group provenance, five-fold OOF, explicit-fit, complete-city holdout, decision reproduction, and external evaluation audits.

- snapshot SHA-256: `1b7b54348d6bcee3375daa67e823bea9919b35ccabf96e53e7e526446456e48e`
- v34 audit SHA-256: `1120f7952634dcc07849865b2a01fc0d53fa67a4cf0ecd6580bdd9a0c9db48b6`
- v35 negative audit SHA-256: `f63154910367da7ffa61bfc7a9a61e8bb5eead1922905f75d791e5159e5dbb86`
- cache aggregate SHA-256: `5b8bac698f2bda045476614b37672edf31d73e50e266e8137baadc9e6349e72c`

## Decisions

| Target | OOF gain | Non-worse | Worst seed | Worst fold | Selected | Fallback regret | Pairwise regret |
|---|---:|---:|---:|---:|---|---:|---:|
| grid4x4 | 0.095103 | 0.833 | -0.074204 | -0.020969 | causal_group_normalized_rigid_advantage | 0.317706 | 0.247764 |
| cologne1 | 0.012271 | 0.800 | -0.062521 | -0.096782 | causal_group_normalized_rigid_advantage | 0.108695 | 0.040156 |
| ingolstadt1 | 0.049603 | 0.783 | 0.020879 | -0.001907 | causal_antisymmetric_pairwise_advantage | 0.155821 | 0.147838 |
| atlanta_1x5 | -0.050000 | 0.933 | -0.117647 | -0.166667 | causal_group_normalized_rigid_advantage | 0.075137 | 0.034608 |
| hangzhou_4x4 | -0.090417 | 0.683 | -0.185069 | -0.208333 | causal_group_normalized_rigid_advantage | 0.377957 | 0.366856 |
| manhattan_28x7 | -0.026730 | 0.850 | -0.166699 | -0.083333 | causal_group_normalized_rigid_advantage | 0.324763 | 0.299943 |

## Frozen Decision

Stage: **FAIL**
Decision: `reject_cross_fitted_pairwise_deployment`
Pairwise selected: 1/6
Selected-policy macro regret: 0.225349
Fallback macro regret: 0.226680
Relative improvement: +0.59%
Maximum absolute regression: -0.000000

- `macro_improvement_at_least_5_pct`: FAIL
- `at_least_four_selected_cities_improved`: FAIL
- `max_regression_at_most_0_02`: PASS

Salt Lake City data were not read and cannot alter this decision.
