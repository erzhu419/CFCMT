# RESCO Phase-Action Benchmark

Actions select among existing green phases from each RESCO SUMO `tlLogic`; no synthetic NS/EW phase assumption is used.

## Aggregate Metrics

| policy | mean_queue | p90_queue | throughput_ratio |
| --- | --- | --- | --- |
| phase_pressure | 4.8250 | 7.5678 | 0.5649 |
| phase_spillback_pressure | 4.9708 | 8.0934 | 0.5609 |
| fixed_program | 9.8585 | 18.1021 | 0.4220 |

## Scenario Metrics

| scenario | tls_count | controlled_lanes | best_policy | best_mean_queue | fixed_program | phase_pressure | phase_spillback_pressure |
| --- | --- | --- | --- | --- | --- | --- | --- |
| cologne1 | 1 | 8 | phase_pressure | 5.4393 | 8.7614 | 5.4393 | 5.6106 |
| cologne3 | 3 | 19 | phase_pressure | 4.2106 | 10.9556 | 4.2106 | 4.3310 |

## Notes

- `fixed_program` leaves the original RESCO static signal programs untouched.
- `phase_pressure` and `phase_spillback_pressure` overwrite signal states every control interval using one of the existing non-yellow green phases.
- This validates arbitrary RESCO phase actions; CFCMT/H2O residual policies can now be attached to this phase abstraction.
