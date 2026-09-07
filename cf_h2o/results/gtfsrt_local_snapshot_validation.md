# GTFS-RT Local Snapshot Validation

Task: one-step occupancy ordinal prediction from synchronized AVL/occupancy snapshots.

This is not a full APC/control validation because GTFS-RT occupancy is crowding status/percentage and has no holding action.

## Dataset

| Feed | Transitions | Snapshots | Routes |
|---|---:|---:|---:|
| NYC MTA Bus | 52920 | 95 | 241 |
| SEPTA Bus | 17040 | 94 | 96 |
| MBTA live | 1666 | 11 | 63 |

## Cross-City Results

| Target | No Local MSE | Static Local MSE | Static/No | Selector | Selector/No |
|---|---:|---:|---:|---|---:|
| mbta_live | 0.0471 | 0.0484 | 1.0279 | no_local | 1.0000 |
| nyc_mta_bus | 0.1569 | 0.1564 | 0.9970 | no_local | 1.0000 |
| septa_bus | 0.6399 | 0.6457 | 1.0091 | no_local | 1.0000 |

## Summary

- Static local mean vs no-local: `1.0114`.
- Selector-gated mean vs no-local: `1.0000`.
- Static local wins vs no-local: `1/3`.
- Selector-gated wins vs no-local: `0/3`.

Interpretation: if static/selector ratios are not clearly below 1.0, use these feeds as evidence that public GTFS-RT occupancy snapshots are useful for data inspection but insufficient for a strong local-graph performance claim; proceed to controlled SUMO APC/AVL snapshot generation.
