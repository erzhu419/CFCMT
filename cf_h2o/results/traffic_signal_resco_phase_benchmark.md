# RESCO Phase-Action Benchmark

Actions select among existing green phases from each RESCO SUMO `tlLogic`; no synthetic NS/EW phase assumption is used.

## Aggregate Metrics

| policy | mean_queue | p90_queue | throughput_ratio |
| --- | --- | --- | --- |
| phase_pressure | 7.6471 | 14.2388 | 0.9006 |
| phase_spillback_pressure | 10.0507 | 16.9058 | 0.8979 |
| phase_cycle | 20.7036 | 32.5369 | 0.8642 |
| fixed_program | 28.7691 | 47.4707 | 0.8539 |

## Scenario Metrics

| scenario | tls_count | controlled_lanes | best_policy | best_mean_queue | fixed_program | phase_cycle | phase_pressure | phase_spillback_pressure |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cologne1 | 1 | 8 | phase_pressure | 10.0937 | 36.2215 | 41.3595 | 10.0937 | 10.1141 |
| cologne3 | 3 | 19 | phase_pressure | 6.6191 | 24.4132 | 19.4134 | 6.6191 | 6.9003 |
| cologne8 | 8 | 33 | phase_pressure | 6.8539 | 22.8793 | 16.1896 | 6.8539 | 19.9107 |
| ingolstadt1 | 1 | 7 | phase_cycle | 4.6098 | 19.1296 | 4.6098 | 4.6762 | 4.7411 |
| ingolstadt7 | 7 | 59 | phase_spillback_pressure | 8.5870 | 41.2017 | 21.9458 | 9.9928 | 8.5870 |

## Notes

- `fixed_program` leaves the original RESCO static signal programs untouched.
- `phase_pressure` and `phase_spillback_pressure` overwrite signal states every control interval using one of the existing non-yellow green phases.
- This validates arbitrary RESCO phase actions; CFCMT/H2O residual policies can now be attached to this phase abstraction.
