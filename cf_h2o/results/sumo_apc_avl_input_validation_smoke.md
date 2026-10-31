# SUMO APC/AVL Stage 1 Input Validation

Verdict: **FAIL**

This stage validates existing H2O city-env tables as inputs for a later SUMO APC/AVL generator. It does not generate SUMO files or run SUMO.

## Quality Gates

| Gate | Result |
|---|---:|
| `all_configured_lines_checked` | FAIL |
| `all_selected_lines_ready` | PASS |
| `no_line_errors` | PASS |
| `all_cities_have_departures` | PASS |
| `all_cities_have_positive_demand` | PASS |
| `all_cities_have_positive_distance` | PASS |

## City Summary

| City | Lines | Ready | Failed | Warnings | Stops | Segments | Departures | OD demand | Distance km | Mean speed m/s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Singapore | 2 | 2 | 0 | 0 | 70 | 68 | 147 | 31840.6 | 29.5 | 9.25 |
| Austin / CapMetro | 2 | 2 | 0 | 2 | 55 | 53 | 5 | 160.0 | 40.5 | 5.50 |
| Halifax Transit | 2 | 2 | 0 | 0 | 48 | 46 | 45 | 2105.0 | 15.0 | 4.32 |
| MBTA | 2 | 2 | 0 | 2 | 34 | 32 | 336 | 715.5 | 10.1 | 5.14 |

## Artifacts

- Line-level manifest: `/home/erzhu419/mine_code/CFCMT/H2Oplus/downloads/sumo_apc_avl_benchmark/inputs/sumo_apc_avl_line_inputs_max2.jsonl`
- JSON report: `/home/erzhu419/mine_code/CFCMT/cf_h2o/results/sumo_apc_avl_input_validation_smoke.json`

## Next Step

If this stage passes, the next stage may generate SUMO nodes/edges/routes from the validated line manifest and run a small SUMO smoke before full-city generation.

## First Errors
