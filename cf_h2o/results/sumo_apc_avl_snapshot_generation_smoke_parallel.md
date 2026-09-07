# SUMO APC/AVL Stage 3 Snapshot Generation

Verdict: **PASS**

This stage runs SUMO through libsumo and produces vehicle-level AVL snapshots plus OD-overlay APC events.

| City | AVL rows | APC rows | Vehicles | Lines | Boardings | Max occ | Result |
|---|---:|---:|---:|---:|---:|---:|---:|
| Halifax Transit | 56 | 46 | 3 | 2 | 50.62 | 11.80 | PASS |
| MBTA | 94 | 86 | 6 | 2 | 9.59 | 3.37 | PASS |
| Austin / CapMetro | 144 | 42 | 5 | 2 | 48.30 | 7.57 | PASS |
| Singapore | 53 | 61 | 3 | 1 | 101.27 | 22.68 | PASS |

## Artifacts

- Halifax Transit: `/home/erzhu419/mine_code/CFCMT/H2Oplus/downloads/sumo_apc_avl_benchmark/sumo/halifax_transit_all/max2/snapshots`
- MBTA: `/home/erzhu419/mine_code/CFCMT/H2Oplus/downloads/sumo_apc_avl_benchmark/sumo/mbta_all/max2/snapshots`
- Austin / CapMetro: `/home/erzhu419/mine_code/CFCMT/H2Oplus/downloads/sumo_apc_avl_benchmark/sumo/austin_capmetro_all/max2/snapshots`
- Singapore: `/home/erzhu419/mine_code/CFCMT/H2Oplus/downloads/sumo_apc_avl_benchmark/sumo/singapore_lta_all/max2/snapshots`
