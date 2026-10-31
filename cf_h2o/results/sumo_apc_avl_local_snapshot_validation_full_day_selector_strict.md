# SUMO APC/AVL Local Snapshot Validation

Task: one-step occupancy prediction from full-day synchronized SUMO APC/AVL snapshots.

## Dataset

| City | Transitions | Raw transitions | Snapshots | Lines | Vehicles | Cache |
|---|---:|---:|---:|---:|---:|---|
| Austin / CapMetro | 1494124 | 1494124 | 1216 | 234 | 28329 | True |
| Halifax Transit | 287943 | 287943 | 1179 | 176 | 8710 | True |
| MBTA | 2324457 | 2324457 | 1374 | 940 | 67065 | True |
| Singapore | 2591781 | 2591781 | 1440 | 789 | 56539 | True |

## Cross-City Results

| Target | No Local MSE | Static Local MSE | Static/No | Selector | Selector/No |
|---|---:|---:|---:|---|---:|
| austin_capmetro_all | 0.000155 | 0.000150 | 0.9666 | no_local | 1.0000 |
| halifax_transit_all | 0.000081 | 0.000060 | 0.7413 | no_local | 1.0000 |
| mbta_all | 0.000266 | 0.000323 | 1.2107 | no_local | 1.0000 |
| singapore_lta_all | 0.006597 | 0.006409 | 0.9714 | static_local | 0.9714 |

## Summary

- Static local mean vs no-local: `0.9725`.
- Selector-gated mean vs no-local: `0.9929`.
- Static local wins vs no-local: `3/4`.
- Selector-gated wins vs no-local: `1/4`.

Interpretation: selector-gated static-local should be used for CFCMT local graph claims only when source-city leave-one validation selects it; otherwise no-local remains the conservative variant.
