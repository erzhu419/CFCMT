# TSC v35/r31 Calibration-Only Deployment Gate Outcome

Generated: 2026-08-08T21:40:53.767829+00:00

## Integrity

**PASS**: 6 shards and 6 results passed snapshot, SHA-256, deterministic split, disjoint calibration, decision reproduction, same-city holdout, and exact v34 metric checks.

- snapshot SHA-256: `00c944b0242d40f94cfa585ba4f5f0a9637c7ee2d812208260dcf3a346274e99`
- v34 audit SHA-256: `1120f7952634dcc07849865b2a01fc0d53fa67a4cf0ecd6580bdd9a0c9db48b6`
- cache aggregate SHA-256: `5b8bac698f2bda045476614b37672edf31d73e50e266e8137baadc9e6349e72c`

## Decisions

| Target | Mean gain | 95% LCB | Non-worse | Worst block | Selected | Fallback regret | Pairwise regret |
|---|---:|---:|---:|---:|---|---:|---:|
| grid4x4 | 0.253849 | 0.058485 | 0.833 | -0.123494 | causal_group_normalized_rigid_advantage | 0.307592 | 0.277159 |
| cologne1 | 0.063728 | -0.033518 | 0.792 | -0.040575 | causal_group_normalized_rigid_advantage | 0.108695 | 0.054628 |
| ingolstadt1 | 0.084707 | -0.024436 | 0.833 | -0.085405 | causal_group_normalized_rigid_advantage | 0.160969 | 0.153198 |
| atlanta_1x5 | 0.000000 | 0.000000 | 1.000 | 0.000000 | causal_group_normalized_rigid_advantage | 0.095824 | 0.040073 |
| hangzhou_4x4 | 0.102778 | -0.091057 | 0.792 | -0.166667 | causal_group_normalized_rigid_advantage | 0.375630 | 0.374778 |
| manhattan_28x7 | 0.015873 | -0.022336 | 0.958 | 0.000000 | causal_group_normalized_rigid_advantage | 0.336539 | 0.295280 |

## Frozen Decision

Stage: **FAIL**
Decision: `reject_calibration_only_pairwise_deployment`
Pairwise selected: 0/6
Selected-policy macro regret: 0.230875
Fallback macro regret: 0.230875
Relative improvement: +0.00%
Maximum absolute regression: -0.000000

- `macro_improvement_at_least_5_pct`: FAIL
- `at_least_four_selected_cities_improved`: FAIL
- `max_regression_at_most_0_02`: PASS

Salt Lake City data were not read and cannot alter this decision.
