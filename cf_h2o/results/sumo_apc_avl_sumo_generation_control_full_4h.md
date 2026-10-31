# SUMO APC/AVL Stage 2 SUMO Generation

Verdict: **PASS**

This stage generates SUMO nodes, edges, routes, explicit bus-stop dwell points, and config files from the validated Stage 1 manifest.

| City | Lines | Edges | Bus stops | Vehicles | Smoke trips | Window | Result |
|---|---:|---:|---:|---:|---:|---:|---:|
| Austin / CapMetro | 234 | 6464 | 6464 | 1696 | 1236 | 13380-27780 | PASS |
| Halifax Transit | 176 | 5551 | 5551 | 1240 | 953 | 15600-30000 | PASS |
| MBTA | 940 | 20195 | 20195 | 4284 | 380 | 0-14400 | PASS |
| Singapore | 789 | 25906 | 25906 | 5384 | 116 | 0-14400 | PASS |

## Artifacts

- Austin / CapMetro: `/home/erzhu419/mine_code/CFCMT/H2Oplus/downloads/sumo_apc_avl_benchmark/sumo_control_full_4h/austin_capmetro_all/full`
- Halifax Transit: `/home/erzhu419/mine_code/CFCMT/H2Oplus/downloads/sumo_apc_avl_benchmark/sumo_control_full_4h/halifax_transit_all/full`
- MBTA: `/home/erzhu419/mine_code/CFCMT/H2Oplus/downloads/sumo_apc_avl_benchmark/sumo_control_full_4h/mbta_all/full`
- Singapore: `/home/erzhu419/mine_code/CFCMT/H2Oplus/downloads/sumo_apc_avl_benchmark/sumo_control_full_4h/singapore_lta_all/full`
