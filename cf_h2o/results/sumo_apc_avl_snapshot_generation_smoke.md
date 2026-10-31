# SUMO APC/AVL Stage 3 Snapshot Generation

Verdict: **PASS**

This stage runs SUMO through libsumo and produces vehicle-level AVL snapshots plus OD-overlay APC events.

| City | AVL rows | APC rows | Vehicles | Lines | Boardings | Max occ | Result |
|---|---:|---:|---:|---:|---:|---:|---:|
| Austin / CapMetro | 145 | 42 | 5 | 2 | 48.30 | 7.57 | PASS |
| Halifax Transit | 57 | 45 | 3 | 2 | 48.50 | 11.09 | PASS |
| MBTA | 95 | 80 | 6 | 2 | 6.89 | 2.83 | PASS |
| Singapore | 54 | 55 | 3 | 1 | 99.36 | 22.68 | PASS |

## Artifacts

- Austin / CapMetro: `/home/erzhu419/mine_code/CFCMT/H2Oplus/downloads/sumo_apc_avl_benchmark/sumo/austin_capmetro_all/max2/snapshots`
- Halifax Transit: `/home/erzhu419/mine_code/CFCMT/H2Oplus/downloads/sumo_apc_avl_benchmark/sumo/halifax_transit_all/max2/snapshots`
- MBTA: `/home/erzhu419/mine_code/CFCMT/H2Oplus/downloads/sumo_apc_avl_benchmark/sumo/mbta_all/max2/snapshots`
- Singapore: `/home/erzhu419/mine_code/CFCMT/H2Oplus/downloads/sumo_apc_avl_benchmark/sumo/singapore_lta_all/max2/snapshots`
