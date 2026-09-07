# SUMO APC/AVL Stage 3 Snapshot Generation

Verdict: **PASS**

This stage runs SUMO through libsumo and produces vehicle-level AVL snapshots plus OD-overlay APC events.

| City | AVL rows | APC rows | Vehicles | Lines | Boardings | Denied | Max occ | Result |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Halifax Transit | 296653 | 315464 | 8710 | 176 | 169096.52 | 0.00 | 42.43 | PASS |
| Austin / CapMetro | 1522453 | 1040341 | 28329 | 234 | 903131.64 | 0.00 | 32.00 | PASS |
| MBTA | 2391548 | 1726494 | 67091 | 940 | 909025.11 | 0.00 | 60.99 | PASS |
| Singapore | 2648326 | 2184429 | 56545 | 789 | 7959598.01 | 361625.69 | 90.00 | PASS |

## Artifacts

- Halifax Transit: `/home/erzhu419/mine_code/CFCMT/H2Oplus/downloads/sumo_apc_avl_benchmark/sumo_full_day/halifax_transit_all/full/snapshots`
- Austin / CapMetro: `/home/erzhu419/mine_code/CFCMT/H2Oplus/downloads/sumo_apc_avl_benchmark/sumo_full_day/austin_capmetro_all/full/snapshots`
- MBTA: `/home/erzhu419/mine_code/CFCMT/H2Oplus/downloads/sumo_apc_avl_benchmark/sumo_full_day/mbta_all/full/snapshots`
- Singapore: `/home/erzhu419/mine_code/CFCMT/H2Oplus/downloads/sumo_apc_avl_benchmark/sumo_full_day/singapore_lta_all/full/snapshots`
