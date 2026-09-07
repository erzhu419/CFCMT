# SUMO APC/AVL Stage 1 Input Validation

Verdict: **PASS**

This stage validates existing H2O city-env tables as inputs for a later SUMO APC/AVL generator. It does not generate SUMO files or run SUMO.

## Quality Gates

| Gate | Result |
|---|---:|
| `all_configured_lines_checked` | PASS |
| `all_selected_lines_ready` | PASS |
| `no_line_errors` | PASS |
| `all_cities_have_departures` | PASS |
| `all_cities_have_positive_demand` | PASS |
| `all_cities_have_positive_distance` | PASS |

## City Summary

| City | Lines | Ready | Failed | Warnings | Stops | Segments | Departures | OD demand | Distance km | Mean speed m/s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Singapore | 789 | 789 | 0 | 227 | 26695 | 25906 | 56545 | 9612553.9 | 13487.3 | 8.85 |
| Austin / CapMetro | 234 | 234 | 0 | 221 | 6698 | 6464 | 28329 | 906529.1 | 3916.3 | 5.86 |
| Halifax Transit | 176 | 176 | 0 | 78 | 5727 | 5551 | 8710 | 178284.4 | 2522.5 | 7.36 |
| MBTA | 940 | 940 | 0 | 517 | 21135 | 20195 | 67091 | 924839.9 | 9237.9 | 5.36 |

## Artifacts

- Line-level manifest: `/home/erzhu419/mine_code/CFCMT/H2Oplus/downloads/sumo_apc_avl_benchmark/inputs/sumo_apc_avl_line_inputs_full.jsonl`
- JSON report: `/home/erzhu419/mine_code/CFCMT/cf_h2o/results/sumo_apc_avl_input_validation.json`

## Next Step

If this stage passes, the next stage may generate SUMO nodes/edges/routes from the validated line manifest and run a small SUMO smoke before full-city generation.
