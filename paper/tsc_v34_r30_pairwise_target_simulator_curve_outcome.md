# TSC v34/r30 Pairwise Target-Simulator Adaptation Outcome

Generated: 2026-08-08T21:30:10.018638+00:00

## Integrity

**PASS**: 30 city-budget shards and 120 independent family results passed snapshot, SHA-256, frozen-cache, complete-city holdout, deterministic target-split, same-subset, causal-parent, and action-group audits.

- snapshot SHA-256: `23de05528236c9d929c9a989d92e4ddb6b2fb0a616f6382f85da6baf434fbce4`
- cache aggregate SHA-256: `5b8bac698f2bda045476614b37672edf31d73e50e266e8137baadc9e6349e72c`
- v26 reference SHA-256: `05fc6e94757b74221b781878fa6bdae3b4142ee4d866aceea13332b5e4932154`

## Curve

| B | Adapt | Calib | Rigid | Group-rigid | Target-only | Pairwise | Pair vs rigid | Cities | Max regression |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0 | 0 | 0.291137 | 0.281557 | 0.360810 | 0.379082 | -30.21% | 4/6 | 0.391201 |
| 8 | 4 | 4 | 0.266294 | 0.257888 | 0.272691 | 0.241211 | +9.42% | 4/6 | 0.020024 |
| 16 | 9 | 7 | 0.287499 | 0.246891 | 0.262484 | 0.242368 | +15.70% | 5/6 | 0.012928 |
| 32 | 19 | 13 | 0.277852 | 0.241229 | 0.234000 | 0.206953 | +25.52% | 6/6 | 0.000000 |
| 60 | 36 | 24 | 0.284935 | 0.230875 | 0.232123 | 0.199186 | +30.09% | 6/6 | 0.000000 |

## Frozen Decision

Stage: **PASS**
Decision: `promote_pairwise_target_adaptation`
Pairwise budget-regret Spearman: -0.900000

- `budget_60_macro_improvement_at_least_10_pct`: PASS
- `budget_60_at_least_four_cities_improved`: PASS
- `budget_60_max_regression_at_most_0_05`: PASS
- `pairwise_budget_60_better_than_zero_shot`: PASS
- `pairwise_budget_regret_spearman_at_most_minus_0_8`: PASS

Salt Lake City data were not read and cannot alter this decision.
